"""
Step-by-step visualization of the fuzzy specialist ensemble.

Fits one of the fuzzy-based models in `MODEL_REGISTRY` (the main model or any
clustering / k-selection variant) once on a train/test split of a synthetic
experiment (any YAML under `tests/`), then reads the state the model keeps after
`fit` and draws every stage of the pipeline, plus how the pieces perform.

Figures go to `outputs/<experiment name>/viz/<model name>/`:

  Part A - the pipeline
    00_k_selection.png               clustering metrics vs k, chosen k marked
    01_raw_data.png                  training data before anything runs
    02_clustering.png                hard cluster + membership strength, centroids
    03_classes_vs_centroids.png      classes vs centroids, class mix per cluster
    04_subsamples.png                1-2 bootstrap subsamples per cluster
    05_membership_density.png        membership density per cluster, threshold marked
    06_final_predictions.png         held-out set, correct vs incorrect

  Part B - performance (all on the held-out test set)
    07_q_within_cluster.png          Q-statistic between the trees of each cluster
    08_q_between_clusters.png        Q-statistic between cluster ensembles (+ full model)
    09_cross_cluster_matrix.png      accuracy of cluster Y on points nearest X

plus `tree_metrics.csv`, `q_within_clusters.csv`, `q_between_clusters.csv` and
`cross_cluster_matrix.csv` (the numbers behind 07-09).

Q-statistic = Yule's Q between two classifiers' correct/incorrect outcomes on the
test set: Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10), where N11 = both right,
N00 = both wrong, N10/N01 = only one right. +1 = they err on the same points,
0 = independent, -1 = complementary. Lower Q means more diverse, which is what
an ensemble wants. Undefined (blank) if either classifier is always right/wrong.

All 2D plots use one PCA fitted on the training data, applied to points and
centroids alike. The model does not scale features, so the PCA is fitted on the
raw features too: the plot shows the space the model actually clusters in.

Per-variant wording: the score a point gets per cluster is called "membership"
(fuzzy c-means, fuzzy c-medoids), "responsibility" (GMM) or "typicality" (PCM).

Usage (from the project root):
    uv run python experiments/visualize_model_steps.py --config tests/control.yaml
    uv run python experiments/visualize_model_steps.py --config tests/control.yaml --model GMMEspecialistEnsembleClassifier
    uv run python experiments/visualize_model_steps.py --config tests/control.yaml --model all
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_synthetic_experiments import MODEL_REGISTRY, build_dataset, load_experiment_config  # noqa: E402
from src.Base_Model import FuzzyEspecialistEnsembleClassifier  # noqa: E402
from src.model_variants.Changes_Clustering_Algorithm import (  # noqa: E402
    GMMEspecialistEnsembleClassifier,
    PossibilisticCMeansEspecialistEnsembleClassifier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "p", "<", ">"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": True, "legend.framealpha": 0.85, "legend.facecolor": SURFACE, "legend.edgecolor": "none", "legend.fontsize": 9, "font.size": 10,
})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class VariantInfo(NamedTuple):
    algo: str               # clustering algorithm, shown in the figure 02 title
    word: str               # what the per-cluster score is called in titles / axes
    normalised: bool        # do a point's scores sum to 1 across clusters?
    uses_m: bool            # does fuzzy_m mean anything for this model?
    show_agreement: bool    # footnote: strongest-score cluster vs nearest centroid
    k_note: "str | None"    # footnote for figure 00


def variant_info(model):
    if isinstance(model, PossibilisticCMeansEspecialistEnsembleClassifier):
        return VariantInfo(
            "possibilistic c-means", "typicality", False, True, False,
            "FPC, MPC, FPE, Xie-Beni and Fukuyama-Sugeno assume memberships that sum to 1; PCM typicalities "
            "do not, so read those panels with care.")
    if isinstance(model, GMMEspecialistEnsembleClassifier):
        return VariantInfo(
            "Gaussian mixture", "responsibility", True, False, True,
            "FPC, MPC, FPE, Xie-Beni and Fukuyama-Sugeno are defined for fuzzy c-means memberships (they use the "
            "fuzzifier m, which a GMM does not have); read those panels with care.")
    if type(model).__name__.startswith("FuzzyKMedoids"):
        return VariantInfo("fuzzy c-medoids", "membership", True, True, False, None)
    return VariantInfo("fuzzy c-means", "membership", True, True, False, None)


def cluster_color(i):
    return CAT[i % len(CAT)]


def text_on(rgba):
    r, g, b = rgba[:3]
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "#ffffff"


def full_proba(tree, X, classes):
    """A tree's predict_proba mapped onto the model's global class order."""
    p = tree.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(tree.classes_):
        out[:, np.searchsorted(classes, c)] = p[:, j]
    return out


def score(y_true, proba, classes):
    """(accuracy, auc) from a probability matrix; AUC is NaN if undefined."""
    if len(y_true) == 0:
        return np.nan, np.nan
    acc = accuracy_score(y_true, classes[np.argmax(proba, axis=1)])
    try:
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
    except ValueError:
        auc = np.nan
    return acc, auc


class Projector:
    """PCA fitted on the training data; also owns the shared axis limits."""

    def __init__(self, X_train, X_test, centers):
        self.pca = PCA(n_components=2, random_state=0).fit(X_train)
        self.train = self.pca.transform(X_train)
        self.test = self.pca.transform(X_test)
        self.centers = self.pca.transform(centers)
        allp = np.vstack([self.train, self.test])
        # Robust limits so outlier / heavy-tail scenarios don't squash the cloud.
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        pad = 0.12 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        lo = np.minimum(lo, self.centers.min(axis=0) - pad)
        hi = np.maximum(hi, self.centers.max(axis=0) + pad)
        self.lo, self.hi = lo, hi
        inside = lambda P: np.all((P >= lo) & (P <= hi), axis=1)
        self.n_hidden = int((~inside(allp)).sum())
        self.var = self.pca.explained_variance_ratio_

    def style(self, ax):
        ax.set_xlim(self.lo[0], self.hi[0])
        ax.set_ylim(self.lo[1], self.hi[1])
        ax.set_xlabel(f"PC1 ({self.var[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({self.var[1]:.0%})")
        ax.ticklabel_format(style="sci", scilimits=(-3, 4))

    def centroids(self, ax, label=True):
        ax.scatter(self.centers[:, 0], self.centers[:, 1], marker="*", s=330,
                   c=SURFACE, edgecolors=INK, linewidths=1.6, zorder=6)
        if label:
            for i, (x, y) in enumerate(self.centers):
                ax.annotate(str(i), (x, y), xytext=(9, 8), textcoords="offset points",
                            fontsize=10, fontweight="bold", color=INK, zorder=7,
                            path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])


def centroid_legend_handle():
    return Line2D([], [], marker="*", ls="", mfc=SURFACE, mec=INK, mew=1.4, ms=13, label="Centroid (number = cluster)")


def class_legend_handles(classes, colors):
    return [Line2D([], [], marker="o", ls="", mfc=colors[c], mec=SURFACE, ms=8, label=f"Class {c}") for c in classes]


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def note_hidden(fig, proj, extra=None):
    lines = []
    if proj.n_hidden:
        lines.append(f"{proj.n_hidden} extreme point(s) fall outside the plotted window (1st-99th percentile + margin).")
    if extra:
        lines.append(extra)
    if lines:
        fig.text(0.01, -0.01, "\n".join(lines), fontsize=8, color=INK2, ha="left", va="top")


# --------------------------------------------------------------------------
# Everything we need from the fitted model
# --------------------------------------------------------------------------
def collect(model, X_test, y_test, min_cell):
    classes = model.classes_
    k = model.k
    centers = model.cluster_centers
    region = np.argmin(cdist(X_test, centers), axis=1)  # nearest centroid = how predict() weights clusters

    cluster_proba, trees, tree_correct = {}, [], {}
    for i in range(k):
        tree_list = model.models.get(f"cluster_{i}", [])
        probas = [full_proba(t, X_test, classes) for t in tree_list]
        mask = region == i
        for n, p in enumerate(probas):
            acc, auc = score(y_test, p, classes)
            racc, _ = score(y_test[mask], p[mask], classes)
            trees.append(dict(cluster=i, subsample=n, accuracy=acc, auc=auc,
                              region_accuracy=racc, region_n=int(mask.sum())))
        cluster_proba[i] = np.mean(probas, axis=0) if probas else None
        # right/wrong per test point for every tree of this cluster -> (n_trees, n_test)
        tree_correct[i] = np.array([classes[np.argmax(p, axis=1)] == y_test for p in probas], dtype=bool)

    full = model.predict_proba(X_test)
    full_acc, full_auc = score(y_test, full, classes)

    # matrix[X, Y]: accuracy of cluster Y's ensemble on test points nearest cluster X.
    # The extra last column is the full weighted model on the same points.
    acc = np.full((k, k + 1), np.nan)
    cnt = np.zeros(k, dtype=int)
    for x in range(k):
        mask = region == x
        cnt[x] = int(mask.sum())
        if cnt[x] == 0:
            continue
        for y in range(k):
            if cluster_proba[y] is not None:
                acc[x, y] = score(y_test[mask], cluster_proba[y][mask], classes)[0]
        acc[x, k] = score(y_test[mask], full[mask], classes)[0]

    # right/wrong per test point for each cluster's ensemble, last row = full model
    n_test = len(y_test)
    correct_by_cluster = np.zeros((k + 1, n_test), dtype=bool)
    for i in range(k):
        if cluster_proba[i] is not None:
            correct_by_cluster[i] = classes[np.argmax(cluster_proba[i], axis=1)] == y_test
    correct_by_cluster[k] = classes[np.argmax(full, axis=1)] == y_test

    cluster_scores = {i: (score(y_test, p, classes) if p is not None else (np.nan, np.nan))
                      for i, p in cluster_proba.items()}
    return dict(classes=classes, k=k, region=region, trees=pd.DataFrame(trees), full=full,
                full_acc=full_acc, full_auc=full_auc, matrix=acc, region_n=cnt,
                cluster_scores=cluster_scores, min_cell=min_cell,
                tree_correct=tree_correct, correct_by_cluster=correct_by_cluster)


def qstat_matrix(correct):
    """
    Pairwise Yule's Q between classifiers. `correct` is a boolean array
    (n_classifiers, n_test_points): True where that classifier got the point right.
    Returns an (n, n) matrix, NaN where the denominator is 0 (a classifier that is
    always right or always wrong has no defined Q).
    """
    C = np.asarray(correct, dtype=float)
    W = 1.0 - C
    n11, n00 = C @ C.T, W @ W.T
    n10 = C @ W.T
    n01 = n10.T
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


# Diverging blue <-> red with a neutral grey midpoint (Q = 0).
Q_CMAP = LinearSegmentedColormap.from_list("q_div", [CAT[7], "#f0efec", CAT[0]])
Q_CMAP.set_bad(GRID)
Q_LABEL = "Yule's Q  (+1 same errors, 0 independent, -1 complementary)"


# --------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------
def fig_k_selection(model, out_dir, info):
    df = model.k_optimization_results
    if df is None or len(df) == 0 or "k" not in df:
        print("  (no k-selection results stored; skipping 00)")
        return
    metrics = [c for c in df.columns if c != "k" and pd.api.types.is_numeric_dtype(df[c])]
    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.4 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        ax.plot(df["k"], df[m], color=CAT[0], lw=2, marker="o", ms=4)
        ax.axvline(model.k, color=INK2, ls="--", lw=1)
        ax.set_title(m.replace("_", " "), fontsize=9.5)
        ax.set_xticks(df["k"])
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Choosing k - clustering metrics per k (dashed = chosen k = {model.k})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if info.k_note:
        fig.text(0.01, -0.01, info.k_note, fontsize=8, color=INK2, ha="left", va="top", wrap=True)
    save(fig, out_dir, "00_k_selection.png")


def fig_raw(proj, y_train, classes, colors, out_dir):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[3.2, 1]))
    for c in classes:
        m = y_train == c
        ax.scatter(proj.train[m, 0], proj.train[m, 1], s=16, c=colors[c], edgecolors=SURFACE,
                   linewidths=0.4, alpha=0.85, label=f"Class {c}")
    proj.style(ax)
    ax.set_title("Training data before any step (PCA projection)")
    ax.legend(handles=class_legend_handles(classes, colors), loc="best")
    counts = [(y_train == c).sum() for c in classes]
    bx.bar([str(c) for c in classes], counts, color=[colors[c] for c in classes], width=0.6)
    for i, v in enumerate(counts):
        bx.text(i, v, f"{v}\n({v / len(y_train):.0%})", ha="center", va="bottom", fontsize=9, color=INK2)
    bx.set_title("Class balance")
    bx.set_xlabel("Class")
    bx.set_ylabel("Training instances")
    bx.set_ylim(0, max(counts) * 1.25)
    bx.grid(axis="x", visible=False)
    note_hidden(fig, proj)
    fig.tight_layout()
    save(fig, out_dir, "01_raw_data.png")


def fig_clustering(proj, membership, k, out_dir, info, agree_note=None):
    hard = membership.argmax(axis=0)
    strength = membership.max(axis=0)
    if info.normalised:  # scores sum to 1, so 1/k is the "fully ambiguous" level
        norm = np.clip((strength - 1 / k) / (1 - 1 / k + 1e-12), 0, 1)
    else:                # PCM typicalities: use the raw value, there is no 1/k baseline
        norm = np.clip(strength, 0, 1)
    rgba = np.array([to_rgba(cluster_color(c), 0.22 + 0.78 * a) for c, a in zip(hard, norm)])
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(proj.train[:, 0], proj.train[:, 1], s=18, c=rgba, edgecolors="none")
    proj.centroids(ax)
    proj.style(ax)
    ax.set_title(f"After clustering ({info.algo}): colour = strongest {info.word}, opacity = strength")
    handles = [Line2D([], [], marker="o", ls="", mfc=cluster_color(i), mec="none", ms=8, label=f"Cluster {i}")
               for i in range(k)] + [centroid_legend_handle()]
    ax.legend(handles=handles, loc="best", ncol=2 if k > 5 else 1)
    note_hidden(fig, proj, agree_note)
    fig.tight_layout()
    save(fig, out_dir, "02_clustering.png")


def fig_classes_vs_centroids(proj, membership, y_train, classes, colors, k, out_dir, agree_note=None):
    hard = membership.argmax(axis=0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[2.4, 1.6]))
    for i in range(k):
        m = hard == i
        ax.scatter(proj.train[m, 0], proj.train[m, 1], s=22, marker=MARKERS[i % len(MARKERS)],
                   c=[colors[c] for c in y_train[m]], edgecolors=SURFACE, linewidths=0.4, alpha=0.85)
    proj.centroids(ax)
    proj.style(ax)
    ax.set_title("Classes vs cluster centroids (colour = class, shape = cluster)")
    h1 = ax.legend(handles=class_legend_handles(classes, colors), loc="upper left")
    ax.add_artist(h1)
    shapes = [Line2D([], [], marker=MARKERS[i % len(MARKERS)], ls="", mfc=MUTED, mec=INK2, ms=7, label=f"Cluster {i}")
              for i in range(k)] + [centroid_legend_handle()]
    ax.legend(handles=shapes, loc="upper right", ncol=2 if k > 5 else 1)

    # Class mix the sampler targets in each cluster (normalised mean membership per class).
    x = np.arange(k)
    bottoms = np.zeros(k)
    mix = np.zeros((k, len(classes)))
    for i in range(k):
        means = np.array([membership[i][y_train == c].mean() for c in classes])
        mix[i] = means / means.sum() if means.sum() > 0 else 1 / len(classes)
    for j, c in enumerate(classes):
        bx.bar(x, mix[:, j], bottom=bottoms, color=colors[c], width=0.7, edgecolor=SURFACE, linewidth=2)
        for i in range(k):
            if mix[i, j] > 0.1:
                bx.text(i, bottoms[i] + mix[i, j] / 2, f"{mix[i, j]:.0%}", ha="center", va="center",
                        fontsize=8.5, color=text_on(to_rgba(colors[c])))
        bottoms += mix[:, j]
    bx.set_xticks(x)
    bx.set_xlabel("Cluster")
    bx.set_ylabel("Share of class mix")
    bx.set_title("Class mix each cluster samples")
    bx.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    bx.grid(axis="x", visible=False)
    note_hidden(fig, proj, agree_note)
    fig.tight_layout()
    save(fig, out_dir, "03_classes_vs_centroids.png")


def fig_subsamples(proj, model, y_train, classes, colors, k, sub_idx, out_dir):
    avail = len(model.models.get("cluster_0", []))
    sub_idx = [n for n in sub_idx if n < avail]
    if not sub_idx:
        print("  (no subsamples available; skipping 04)")
        return
    fig, axes = plt.subplots(k, len(sub_idx), figsize=(4.8 * len(sub_idx), 3.7 * k), squeeze=False)
    for i in range(k):
        for col, n in enumerate(sub_idx):
            ax = axes[i, col]
            ax.scatter(proj.train[:, 0], proj.train[:, 1], s=6, c=MUTED, alpha=0.5, edgecolors="none")
            data = model.raw_cluster_datasets.get(f"cluster_{i}_{n}_data")
            if data is not None:
                counts = data.index.value_counts()
                idx = counts.index.to_numpy()
                ax.scatter(proj.train[idx, 0], proj.train[idx, 1], s=14 + 14 * (counts.to_numpy() - 1),
                           c=[colors[c] for c in y_train[idx]], edgecolors=SURFACE, linewidths=0.6, alpha=0.9, zorder=3)
                ax.set_title(f"Cluster {i} - subsample {n}: {len(data)} draws, {len(idx)} unique", fontsize=9.5)
            proj.centroids(ax, label=False)
            proj.style(ax)
            if col:
                ax.set_ylabel("")
            if i < k - 1:
                ax.set_xlabel("")
    fig.legend(handles=class_legend_handles(classes, colors) + [
        Line2D([], [], marker="o", ls="", mfc=MUTED, mec="none", ms=6, label="Not drawn (all training data)"),
        centroid_legend_handle()], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0 + 0.02 / max(k, 1) * 3))
    fig.text(0.5, -0.005, "Marker size grows with the number of times an instance was drawn (sampling is with replacement).",
             ha="center", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, "04_subsamples.png")


def fig_membership_density(model, membership, k, out_dir, info):
    """Density of every training point's membership in each cluster, with the
    threshold the sampler uses to decide which points count as 'in' the cluster."""
    thr = model.membership_threshold
    ncols = min(k, 4)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.3 * nrows), squeeze=False, sharex=True)
    xs = np.linspace(0, 1, 400)
    for i in range(k):
        ax = axes[i // ncols, i % ncols]
        u = membership[i]
        try:
            dens = gaussian_kde(u)(xs)
            ax.fill_between(xs, dens, color=cluster_color(i), alpha=0.35, linewidth=0)
            ax.plot(xs, dens, color=cluster_color(i), lw=2)
        except (np.linalg.LinAlgError, ValueError):  # all memberships identical
            ax.axvline(float(u[0]), color=cluster_color(i), lw=2)
        ax.axvline(thr, color=INK, ls=":", lw=1.6)
        n_above = int((u > thr).sum())
        ax.set_title(f"Cluster {i}", fontsize=10)
        ax.text(0.97, 0.95, f"{n_above / len(u):.0%} of points above {thr:g}\n({n_above} of {len(u)})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        if i // ncols == nrows - 1 or i + ncols >= k:
            ax.set_xlabel(f"{info.word.capitalize()} in cluster")
        if i % ncols == 0:
            ax.set_ylabel("Density")
    for j in range(k, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"{info.word.capitalize()} density per cluster (dotted line = threshold {thr:g}: points above it count as in the cluster)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_dir, "05_membership_density.png")


def fig_final(proj, y_test, pred, classes, colors, out_dir):
    wrong = pred != y_test
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(proj.train[:, 0], proj.train[:, 1], s=6, c=MUTED, alpha=0.4, edgecolors="none")
    for c in classes:
        m = (y_test == c) & ~wrong
        ax.scatter(proj.test[m, 0], proj.test[m, 1], s=34, c=colors[c], edgecolors=SURFACE, linewidths=0.5,
                   alpha=0.9, zorder=3)
    for c in classes:
        m = (y_test == c) & wrong
        ax.scatter(proj.test[m, 0], proj.test[m, 1], s=95, marker="X", c=colors[c], edgecolors=INK,
                   linewidths=1.3, zorder=5)
    proj.centroids(ax)
    proj.style(ax)
    acc = 1 - wrong.mean()
    ax.set_title(f"Held-out test set: accuracy {acc:.3f} ({wrong.sum()} wrong of {len(y_test)})")
    handles = class_legend_handles(classes, colors) + [
        Line2D([], [], marker="X", ls="", mfc=MUTED, mec=INK, mew=1.3, ms=9, label="Misclassified (colour = true class)"),
        Line2D([], [], marker="o", ls="", mfc=MUTED, mec="none", ms=5, alpha=0.5, label="Training data (context)"),
        centroid_legend_handle()]
    ax.legend(handles=handles, loc="best", fontsize=8.5)
    note_hidden(fig, proj)
    fig.tight_layout()
    save(fig, out_dir, "06_final_predictions.png")


# --------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------
def fig_q_within(res, k, out_dir):
    """One Q matrix per cluster: how much the trees trained on that cluster's
    subsamples make the same mistakes as each other."""
    ncols = min(k, 4)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols + 1.2, 3.8 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i in range(k):
        ax = axes[i // ncols, i % ncols]
        C = res["tree_correct"][i]
        T = len(C)
        if T == 0:
            ax.axis("off")
            continue
        Q = qstat_matrix(C)
        np.fill_diagonal(Q, np.nan)  # a tree vs itself is trivially 1
        im = ax.imshow(np.ma.masked_invalid(Q), cmap=Q_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        if T <= 10:
            for a_ in range(T):
                for b_ in range(T):
                    if a_ != b_ and not np.isnan(Q[a_, b_]):
                        ax.text(b_, a_, f"{Q[a_, b_]:.2f}", ha="center", va="center", fontsize=7,
                                color=text_on(Q_CMAP((Q[a_, b_] + 1) / 2)))
        off = Q[~np.eye(T, dtype=bool)]
        mean_q = np.nanmean(off) if np.isfinite(off).any() else np.nan
        ax.set_title(f"Cluster {i} - mean Q {mean_q:.2f}", fontsize=10)
        ticks = np.unique(np.linspace(0, T - 1, min(T, 5)).astype(int))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Subsample tree")
        if i % ncols == 0:
            ax.set_ylabel("Subsample tree")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j in range(k, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cb.set_label(Q_LABEL)
        cb.outline.set_visible(False)
    fig.suptitle("Q-statistic between the trees of each cluster (held-out test set)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, out_dir, "07_q_within_cluster.png")


def fig_q_between(res, k, out_dir):
    """Q between cluster ensembles: each cluster's trees are averaged into one
    prediction first. The last row/column is the full weighted model."""
    Q = qstat_matrix(res["correct_by_cluster"])
    n = k + 1
    fig, ax = plt.subplots(figsize=(0.95 * n + 3.4, 0.8 * n + 2.4))
    for x in range(n):
        for y in range(n):
            v = Q[x, y]
            diag = x == y
            ok = not (diag or np.isnan(v))
            face = Q_CMAP((v + 1) / 2) if ok else to_rgba(GRID)
            ax.add_patch(Rectangle((y, x), 1, 1, facecolor=face, edgecolor=SURFACE, linewidth=2))
            if ok:
                ax.text(y + 0.5, x + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=9, color=text_on(face))
            elif not diag:
                ax.text(y + 0.5, x + 0.5, "n/a", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.axhline(k, color=SURFACE, lw=5)  # set the full model apart
    ax.axvline(k, color=SURFACE, lw=5)
    labels = [f"Cluster {i}" for i in range(k)] + ["Full model"]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=Q_CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(Q_LABEL)
    cb.outline.set_visible(False)
    ax.set_title("Q-statistic between cluster ensembles (held-out test set)")
    fig.tight_layout()
    save(fig, out_dir, "08_q_between_clusters.png")


def fig_matrix(res, k, n_classes, out_dir):
    M, cnt, min_cell = res["matrix"], res["region_n"], res["min_cell"]
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQ)
    vmin = 1 / n_classes
    fig, ax = plt.subplots(figsize=(1.0 * (k + 1) + 3.2, 0.8 * k + 2.6))
    for x in range(k):
        for y in range(k + 1):
            v = M[x, y]
            small = cnt[x] < min_cell
            ok = not (np.isnan(v) or small)
            face = cmap(np.clip((v - vmin) / (1 - vmin), 0, 1)) if ok else to_rgba(GRID)
            ax.add_patch(Rectangle((y, x), 1, 1, facecolor=face, edgecolor=SURFACE, linewidth=2))
            txt = f"{v:.2f}\nn={cnt[x]}" if ok else f"n={cnt[x]}"
            ax.text(y + 0.5, x + 0.5, txt, ha="center", va="center", fontsize=8.5,
                    color=text_on(face) if ok else INK2)
    for x in range(k):  # own-cluster diagonal
        ax.add_patch(Rectangle((x, x), 1, 1, fill=False, edgecolor=INK, linewidth=2.2, zorder=5))
    ax.set_xlim(0, k + 1)
    ax.set_ylim(k, 0)
    ax.set_xticks(np.arange(k + 1) + 0.5)
    ax.set_xticklabels([f"Cluster {y}" for y in range(k)] + ["Full model"], rotation=30, ha="right")
    ax.set_yticks(np.arange(k) + 0.5)
    ax.set_yticklabels([f"Nearest to {x}" for x in range(k)])
    ax.set_xlabel("Classifier: cluster Y's subsample trees (outlined = own cluster)")
    ax.set_ylabel("Test points, grouped by nearest centroid X")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"Accuracy (chance = {vmin:.2f})")
    cb.outline.set_visible(False)
    ax.set_title(f"Accuracy by region and classifying cluster (grey: fewer than {min_cell} test points)")
    fig.tight_layout()
    save(fig, out_dir, "09_cross_cluster_matrix.png")


# --------------------------------------------------------------------------
def fuzzy_model_names(seed):
    """Registry entries that are fuzzy-based (have centroids, memberships, subsamples)."""
    return [n for n, make in MODEL_REGISTRY.items() if isinstance(make(seed), FuzzyEspecialistEnsembleClassifier)]


def run_model(name, X_train, X_test, y_train, y_test, seed, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {name}  ->  {out_dir}")
    model = MODEL_REGISTRY[name](seed)
    info = variant_info(model)
    if args.fuzzy_m is not None:
        if info.uses_m:
            model.set_params(fuzzy_m=args.fuzzy_m)
        else:
            print(f"  note: --fuzzy-m ignored, {name} has no fuzzifier m")
    model.fit(X_train, y_train)
    classes, k = model.classes_, model.k
    colors = {c: CAT[i % len(CAT)] for i, c in enumerate(classes)}
    membership = np.asarray(model.last_training_membership)  # (k, n_train)
    degenerate = float(membership.max(axis=0).mean()) < 1 / k + 0.01
    if degenerate:
        print(f"  WARNING: degenerate clustering - mean largest membership {membership.max(axis=0).mean():.4f} "
              f"is ~1/k = {1 / k:.4f}. Centroids collapse toward the data mean (fuzzy_m={model.fuzzy_m}); "
              f"clusters are near-identical and every cluster's trees see ~the same data. Try --fuzzy-m 1.2.")
    print(f"  train={len(X_train)} test={len(X_test)} k={k} trees={sum(len(v) for v in model.models.values())}")

    agree_note = None
    if info.show_agreement:
        near = np.argmin(cdist(X_train, model.cluster_centers), axis=1)
        agree = float(np.mean(membership.argmax(axis=0) == near))
        agree_note = (f"{agree:.0%} of points get the same cluster from their strongest {info.word} as from the nearest "
                      f"centroid (prediction weights clusters by centroid distance).")

    proj = Projector(X_train, X_test, model.cluster_centers)
    res = collect(model, X_test, y_test, args.min_cell)
    pred = classes[np.argmax(res["full"], axis=1)]

    fig_k_selection(model, out_dir, info)
    fig_raw(proj, y_train, classes, colors, out_dir)
    fig_clustering(proj, membership, k, out_dir, info, agree_note)
    fig_classes_vs_centroids(proj, membership, y_train, classes, colors, k, out_dir, agree_note)
    fig_subsamples(proj, model, y_train, classes, colors, k, args.subsamples, out_dir)
    fig_membership_density(model, membership, k, out_dir, info)
    fig_final(proj, y_test, pred, classes, colors, out_dir)
    fig_q_within(res, k, out_dir)
    fig_q_between(res, k, out_dir)
    fig_matrix(res, k, len(classes), out_dir)

    res["trees"].to_csv(out_dir / "tree_metrics.csv", index=False)
    cols = [f"classified_by_cluster_{y}" for y in range(k)] + ["classified_by_full_model"]
    mat = pd.DataFrame(res["matrix"], columns=cols)
    mat.insert(0, "n_test_points", res["region_n"])
    mat.index = [f"nearest_to_cluster_{x}" for x in range(k)]
    mat.to_csv(out_dir / "cross_cluster_matrix.csv")

    names = [f"cluster_{i}" for i in range(k)] + ["full_model"]
    pd.DataFrame(qstat_matrix(res["correct_by_cluster"]), index=names, columns=names).to_csv(
        out_dir / "q_between_clusters.csv")
    rows = []
    for i in range(k):
        Qi = qstat_matrix(res["tree_correct"][i]) if len(res["tree_correct"][i]) else np.empty((0, 0))
        rows += [dict(cluster=i, tree_a=a, tree_b=b, q=Qi[a, b])
                 for a in range(len(Qi)) for b in range(a + 1, len(Qi))]
    pd.DataFrame(rows).to_csv(out_dir / "q_within_clusters.csv", index=False)
    print(f"  full model: accuracy={res['full_acc']:.4f} auc={res['full_auc']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="experiment YAML, e.g. tests/control.yaml")
    ap.add_argument("--model", default="FuzzyEspecialistEnsembleClassifier",
                    help="a fuzzy-based name from MODEL_REGISTRY, or 'all' for every one of them "
                         "(default: FuzzyEspecialistEnsembleClassifier)")
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--subsamples", type=int, nargs="+", default=[0, 1],
                    help="which subsample indices to show in figure 04 (default: 0 1)")
    ap.add_argument("--fuzzy-m", type=float, default=None,
                    help="override the model's fuzzifier m (default: the model's own default) to compare settings; "
                         "skipped for models that have no fuzzifier (GMM)")
    ap.add_argument("--min-cell", type=int, default=5,
                    help="regions with fewer test points than this are greyed out (default: 5)")
    args = ap.parse_args()

    config = load_experiment_config(args.config)
    X, y = build_dataset(config)
    seed = config.get("cv", {}).get("random_state", config["dataset"].get("random_state", 42))
    valid = fuzzy_model_names(seed)
    if args.model != "all" and args.model not in valid:
        ap.error(f"--model must be 'all' or one of the fuzzy-based registry names: {valid}")
    names = valid if args.model == "all" else [args.model]

    base_dir = args.outputs_dir / config["name"] / "viz"
    print(f"Experiment: {config['name']}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=seed)
    for name in names:
        try:
            run_model(name, X_train, X_test, y_train, y_test, seed, args, base_dir / name)
        except Exception as e:
            if len(names) == 1:
                raise
            print(f"  {name} FAILED, skipping: {e}")


if __name__ == "__main__":
    main()
