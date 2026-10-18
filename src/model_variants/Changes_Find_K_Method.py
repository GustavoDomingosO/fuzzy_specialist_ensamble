"""
Variants of FuzzyEspecialistEnsembleClassifier for the comparative analysis.

Base_Model.py is used as a base class so we can alter just one method per variant.

Section 2 (k-selection speed): changes only how `k` is searched for, by
overriding `get_best_k` the clustering algorithm, sampling, training, and
fusion are all untouched.

Section 2b (k-selection quality): same idea, but instead of voting among
cluster-validity metrics, `k` is chosen by a different criterion (downstream
CV performance, supervised class information, partition stability, or the
gap statistic). Each variant still only overrides `get_best_k`.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold

from ..Base_Model import (
    FuzzyEspecialistEnsembleClassifier,
    FuzzyClusteringMetrics,
    DEFAULT_FUZZY_M,
    DEFAULT_FUZZY_ERROR,
    DEFAULT_FUZZY_MAX_ITER,
    DEFAULT_K_RANGE,
    DEFAULT_FALLBACK_K,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_NUM_SUBSAMPLES,
    DEFAULT_MIN_SAMPLES,
)

# ==========================================
# Default hyperparameters of the k-selection variants
# ==========================================

DEFAULT_CV_FOLDS = 3                   # Inner CV folds (variant 5)
DEFAULT_CV_NUM_SUBSAMPLES = 30         # Trees per inner model, kept small for speed (variant 5)
DEFAULT_CV_METRIC = 'auc'              # 'auc' or 'accuracy' (variant 5)
DEFAULT_ONE_SE_RULE = True             # Pick the smallest k within 1 SE of the best (variant 5)
DEFAULT_MDL_PENALTY_WEIGHT = 1.0       # Weight of the BIC/MDL penalty (variant 6)
DEFAULT_STABILITY_REPEATS = 10         # Number of subsample pairs (variant 7)
DEFAULT_STABILITY_SUBSAMPLE = 0.8      # Fraction of the data in each subsample (variant 7)
DEFAULT_STABILITY_EVAL_POINTS = 500    # Points used to compare two partitions (variant 7)
DEFAULT_STABILITY_TOLERANCE = 0.01     # Stability drop tolerated when picking the largest k (variant 7); sensitive, tune it
DEFAULT_GAP_N_REFS = 10                # Reference datasets (variant 8)


# 4. Faster k-selection: fewer metrics
class FastKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Only 3 of the original 10 cluster-validity metrics are
    computed per candidate k, instead of all 10:

        - FPC (Fuzzy Partition Coefficient): kept because it's the fuzzy-
          native index in the original set that's also cheap (O(n*k), no
          pairwise distance matrix), it measures how "crisp" vs "fuzzy"
          the partition is.
        - Xie-Beni: kept because it reuses the same cached (n_samples, k)
          distances FPC's cluster computes, so it's essentially free once
          FPC is available, and it's one of the few validity indices that
          explicitly accounts for the fuzziness exponent `m`.
        - Calinski-Harabasz: kept as the cheapest *hard-label* index (a
          ratio of between/within-cluster dispersion), used as a head-label
          representation of the metrics.

    `_find_best_k_heuristic` is unchanged, as it already skips any
    metric absent from its input columns.
    """

    def get_best_k(self, X_train_processed):
        np.random.seed(self.random_state)  # reproducible k sweep 
        X_train_values = np.asarray(X_train_processed)
        X_train_T = X_train_values.T

        results = []
        for k in self.k_range:
            try:
                cntr, u = self._fit_fuzzy_clustering(X_train_T, k)
                metrics_obj = FuzzyClusteringMetrics(X_train_values, u, cntr, self.fuzzy_m)
                results.append({
                    'k': k,
                    'FPC': metrics_obj.get_fpc(),
                    'Xie_Beni': metrics_obj.get_xie_beni(),
                    'Calinski_Harabasz': metrics_obj.get_calinski_harabasz(),
                })
            except Exception as e:
                print(f"An error occurred for k={k}: {e}")

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        return self._find_best_k_heuristic(df_results)

# ==========================================
# Shared helpers
# ==========================================

def _membership_from_centers(X, centers, m):
    """
    Fuzzy c-means membership of the points in X given fixed centers
    (the standard FCM update): u_ij = 1 / sum_l (d_ij / d_lj)^(2 / (m - 1)).

    X: (n_samples, n_features), centers: (k, n_features), returns (k, n_samples).
    """
    d = np.fmax(cdist(X, centers), 1e-12)
    inv = d ** (-2.0 / (m - 1.0))
    return (inv / inv.sum(axis=1, keepdims=True)).T


def _entropy(p):
    """Shannon entropy (nats) of each row of p (or of a 1-D vector)."""
    p = np.asarray(p, dtype=float)
    safe = np.where(p > 0, p, 1.0)
    return -np.sum(p * np.log(safe), axis=-1)

# ------------------------------------------------------------------
# Internal helper model used by variant 5 (not meant to be used directly)
# ------------------------------------------------------------------
class _FixedKEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    The unchanged base pipeline, but with `k` imposed instead of searched.
    Only used internally by the CV-based variant to score one candidate k.
    """

    def __init__(self, fixed_k=DEFAULT_FALLBACK_K, **base_params):
        super().__init__(**base_params)
        self.fixed_k = fixed_k

    def get_best_k(self, X_train_processed):
        return self.fixed_k


# 5. k-selection by downstream performance (inner cross-validation)
class CVKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Treats k as a hyperparameter of the whole ensemble: for every candidate k,
    a (lighter) copy of the full pipeline is trained and evaluated with an
    inner stratified K-fold on the training data, and k is chosen by the
    resulting out-of-fold score instead of by cluster-validity metrics.

        - The inner models use `cv_num_subsamples` trees instead of
          `num_subsamples` to keep the cost manageable; everything else
          (FCM, sampling, trees, fusion) is the base pipeline.
        - `cv_metric` selects the score: 'auc' (default, falls back to
          skipping folds where AUC is undefined) or 'accuracy'.
        - `one_se_rule`: instead of the k with the best mean score, take the
          smallest k whose mean score is within one standard error of the
          best (favors simpler / fewer specialists when scores are tied).

    Cost is roughly len(k_range) * cv_folds full (small) fits.
    The labels are read from `self.last_y_train`, which `fit` stores before
    calling `get_best_k`.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        cv_folds: int = DEFAULT_CV_FOLDS,
        cv_num_subsamples: int = DEFAULT_CV_NUM_SUBSAMPLES,
        cv_metric: str = DEFAULT_CV_METRIC,
        one_se_rule: bool = DEFAULT_ONE_SE_RULE,
    ):
        super().__init__(
            debug_folder=debug_folder, detailed_logs=detailed_logs,
            random_state=random_state, fuzzy_m=fuzzy_m, fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter, k_range=k_range, fallback_k=fallback_k,
            membership_threshold=membership_threshold, num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.cv_folds = cv_folds
        self.cv_num_subsamples = cv_num_subsamples
        self.cv_metric = cv_metric
        self.one_se_rule = one_se_rule

    def get_best_k(self, X_train_processed):
        np.random.seed(self.random_state)  # reproducible k sweep (see Base_Model.get_best_k)
        X = X_train_processed
        y = np.asarray(self.last_y_train)

        _, counts = np.unique(y, return_counts=True)
        n_splits = min(self.cv_folds, int(counts.min()))
        if n_splits < 2:
            print("Not enough samples in the smallest class for an inner CV; falling back.")
            self.k_optimization_results = pd.DataFrame()
            return None

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(np.zeros(len(y)), y))

        base_params = dict(
            debug_folder=self.debug_folder, detailed_logs=False,
            random_state=self.random_state, fuzzy_m=self.fuzzy_m,
            fuzzy_error=self.fuzzy_error, fuzzy_max_iter=self.fuzzy_max_iter,
            fallback_k=self.fallback_k, membership_threshold=self.membership_threshold,
            min_samples=self.min_samples,
        )

        def take(data, idx):
            return data.iloc[idx] if hasattr(data, 'iloc') else np.asarray(data)[idx]

        results = []
        for k in self.k_range:
            fold_scores = []
            for tr_idx, te_idx in splits:
                try:
                    inner = _FixedKEnsembleClassifier(
                        fixed_k=k,
                        num_subsamples=max(self.cv_num_subsamples, k),  # need >= 1 model per cluster
                        **base_params,
                    )
                    inner.fit(take(X, tr_idx), y[tr_idx])
                    acc, auc = inner.evaluate(take(X, te_idx), y[te_idx])
                    score = auc if self.cv_metric == 'auc' else acc
                    if not np.isnan(score):
                        fold_scores.append(score)
                except Exception as e:
                    print(f"An error occurred for k={k} (inner CV): {e}")

            if fold_scores:
                std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
                results.append({
                    'k': k,
                    'CV_Score': float(np.mean(fold_scores)),
                    'CV_Std': std,
                    'CV_SE': std / np.sqrt(len(fold_scores)),
                    'CV_Folds_Used': len(fold_scores),
                })

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        if df_results.empty:
            return None

        best = df_results.loc[df_results['CV_Score'].idxmax()]
        if self.one_se_rule:
            threshold = best['CV_Score'] - best['CV_SE']
            chosen = int(df_results.loc[df_results['CV_Score'] >= threshold, 'k'].min())
        else:
            chosen = int(best['k'])

        if self.detailed_logs:
            print(f"\nCV-selected k: {chosen} (best raw score at k={int(best['k'])})")
        return chosen


# 6. k-selection by supervised class information (BIC/MDL-penalized)
class SupervisedEntropyKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Uses the labels to ask "do the fuzzy clusters make the classification
    problem easier?" instead of only measuring cluster geometry.

    For each k, with membership matrix u (k, n) and one-hot labels Y (n, C):

        class distribution of cluster i:  p_i = (u_i . Y) / sum_j u_ij
        conditional entropy:              H(Y|C) = sum_i (mass_i / n) * H(p_i)
        information gain:                 IG = H(Y) - H(Y|C)

    IG can only grow as k grows (finer partitions are always at least as
    pure), so it is penalized with the BIC / MDL term of the multinomial
    likelihood-ratio test of "class independent of cluster":

        score(k) = n * IG - w * 0.5 * (k - 1) * (C - 1) * log(n)

    and the k with the highest score is picked (w = `mdl_penalty_weight`).
    The clustering itself is still unsupervised (FCM on X only); only the
    choice of k uses y, read from `self.last_y_train` (set by `fit`).
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        mdl_penalty_weight: float = DEFAULT_MDL_PENALTY_WEIGHT,
    ):
        super().__init__(
            debug_folder=debug_folder, detailed_logs=detailed_logs,
            random_state=random_state, fuzzy_m=fuzzy_m, fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter, k_range=k_range, fallback_k=fallback_k,
            membership_threshold=membership_threshold, num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.mdl_penalty_weight = mdl_penalty_weight

    def get_best_k(self, X_train_processed):
        np.random.seed(self.random_state)  # reproducible k sweep (see Base_Model.get_best_k)
        X_train_values = np.asarray(X_train_processed)
        X_train_T = X_train_values.T

        y = np.asarray(self.last_y_train)
        classes, y_idx = np.unique(y, return_inverse=True)
        n, n_classes = len(y), len(classes)
        if n_classes < 2:
            self.k_optimization_results = pd.DataFrame()
            return None

        Y = np.eye(n_classes)[y_idx]              # (n, C) one-hot
        h_global = float(_entropy(Y.mean(axis=0)))

        results = []
        for k in self.k_range:
            try:
                cntr, u = self._fit_fuzzy_clustering(X_train_T, k)
                mass = np.fmax(u.sum(axis=1), 1e-12)          # (k,) soft cluster sizes
                p = (u @ Y) / mass[:, None]                   # (k, C) class mix per cluster
                h_cond = float(((mass / n) * _entropy(p)).sum())
                info_gain = h_global - h_cond
                penalty = 0.5 * (k - 1) * (n_classes - 1) * np.log(n)
                results.append({
                    'k': k,
                    'Cond_Entropy': h_cond,
                    'Info_Gain': info_gain,
                    'Norm_Info_Gain': info_gain / h_global if h_global > 0 else 0.0,
                    'MDL_Score': n * info_gain - self.mdl_penalty_weight * penalty,
                })
            except Exception as e:
                print(f"An error occurred for k={k}: {e}")

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        if df_results.empty:
            return None

        chosen = int(df_results.loc[df_results['MDL_Score'].idxmax(), 'k'])
        if self.detailed_logs:
            print(f"\nSupervised-entropy selected k: {chosen}")
        return chosen


# 7. k-selection by partition stability (subsample reproducibility)
class StabilityKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    A k is good if the structure it finds is reproducible. For each k and each
    of `stability_repeats` repeats:

        1. draw two random subsamples (fraction `stability_subsample`) and
           run FCM on each, getting two sets of centers;
        2. compute the fuzzy memberships of a fixed set of evaluation points
           (`stability_eval_points`) under each set of centers;
        3. compare the two partitions through their co-membership matrices
           S = U^T U (S_jl = how much points j and l share a cluster) using
           the Pearson correlation of their upper triangles.

    Comparing co-membership matrices makes the score invariant to the
    arbitrary order of the clusters, so no cluster matching is needed. The
    same subsample pairs are reused for every k, so k values are compared
    fairly.

    Stability is naturally high for very small k, so the rule is the usual
    one: pick the *largest* k whose mean stability is within
    `stability_tolerance` of the best one.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        stability_repeats: int = DEFAULT_STABILITY_REPEATS,
        stability_subsample: float = DEFAULT_STABILITY_SUBSAMPLE,
        stability_eval_points: int = DEFAULT_STABILITY_EVAL_POINTS,
        stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
    ):
        super().__init__(
            debug_folder=debug_folder, detailed_logs=detailed_logs,
            random_state=random_state, fuzzy_m=fuzzy_m, fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter, k_range=k_range, fallback_k=fallback_k,
            membership_threshold=membership_threshold, num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.stability_repeats = stability_repeats
        self.stability_subsample = stability_subsample
        self.stability_eval_points = stability_eval_points
        self.stability_tolerance = stability_tolerance

    def get_best_k(self, X_train_processed):
        X = np.asarray(X_train_processed, dtype=float)
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Evaluation points and subsample pairs are drawn once and shared by all k
        n_eval = min(n, self.stability_eval_points)
        X_eval = X[rng.choice(n, size=n_eval, replace=False)]
        iu = np.triu_indices(n_eval, k=1)

        size = min(n, max(int(round(self.stability_subsample * n)), max(self.k_range) + 1))
        pairs = [
            (rng.choice(n, size=size, replace=False), rng.choice(n, size=size, replace=False))
            for _ in range(self.stability_repeats)
        ]

        results = []
        for k in self.k_range:
            scores = []
            for r, (idx_a, idx_b) in enumerate(pairs):
                try:
                    np.random.seed(self.random_state + 2 * r)
                    cntr_a, _ = self._fit_fuzzy_clustering(X[idx_a].T, k)
                    np.random.seed(self.random_state + 2 * r + 1)
                    cntr_b, _ = self._fit_fuzzy_clustering(X[idx_b].T, k)

                    u_a = _membership_from_centers(X_eval, cntr_a, self.fuzzy_m)
                    u_b = _membership_from_centers(X_eval, cntr_b, self.fuzzy_m)
                    v_a = (u_a.T @ u_a)[iu]
                    v_b = (u_b.T @ u_b)[iu]

                    if v_a.std() > 1e-12 and v_b.std() > 1e-12:
                        scores.append(float(np.corrcoef(v_a, v_b)[0, 1]))
                except Exception as e:
                    print(f"An error occurred for k={k} (repeat {r}): {e}")

            if scores:
                results.append({
                    'k': k,
                    'Stability': float(np.mean(scores)),
                    'Stability_Std': float(np.std(scores)),
                })

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        if df_results.empty:
            return None

        best = df_results['Stability'].max()
        candidates = df_results[df_results['Stability'] >= best - self.stability_tolerance]
        chosen = int(candidates['k'].max())
        if self.detailed_logs:
            print(f"\nStability-selected k: {chosen} (best stability {best:.3f})")
        return chosen


# 8. k-selection by the gap statistic (fuzzy dispersion vs. uniform reference)
class GapStatisticKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Tibshirani's gap statistic, using the fuzzy distortion J_m (the same
    quantity as the 'Distortion' metric) as the within-cluster dispersion W_k:

        Gap(k) = mean_b[ log W_k(reference_b) ] - log W_k(data)

    The `gap_n_refs` reference datasets are uniform over the bounding box of
    the data after rotating it to its principal axes (so the reference
    follows the data's shape). Because W_k always decreases with k, the gap
    corrects that built-in trend by comparing it with what "no structure"
    would give.

    Selection rule (Tibshirani et al.): the smallest k such that
    Gap(k) >= Gap(k+1) - s_{k+1}, where s_k = sd_k * sqrt(1 + 1/B). If no k
    satisfies it (e.g. the gap keeps growing), the k with the largest gap is used.
    The clustering is rotation invariant, so FCM is run on the rotated data.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        gap_n_refs: int = DEFAULT_GAP_N_REFS,
    ):
        super().__init__(
            debug_folder=debug_folder, detailed_logs=detailed_logs,
            random_state=random_state, fuzzy_m=fuzzy_m, fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter, k_range=k_range, fallback_k=fallback_k,
            membership_threshold=membership_threshold, num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.gap_n_refs = gap_n_refs

    def _log_dispersion(self, X, k):
        """log of the fuzzy distortion J_m of an FCM fit with k clusters on X (n, d)."""
        cntr, u = self._fit_fuzzy_clustering(X.T, k)
        w = FuzzyClusteringMetrics(X, u, cntr, self.fuzzy_m).get_distortion()
        return float(np.log(max(w, 1e-12)))

    def get_best_k(self, X_train_processed):
        np.random.seed(self.random_state)  # reproducible k sweep (see Base_Model.get_best_k)
        X = np.asarray(X_train_processed, dtype=float)

        # Rotate to principal axes; reference data is uniform in this box
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        X_rot = Xc @ vt.T
        lo, hi = X_rot.min(axis=0), X_rot.max(axis=0)

        def reference(b):
            # Regenerated from a per-reference seed, so no need to keep B copies in memory
            return np.random.RandomState(self.random_state + 1 + b).uniform(lo, hi, size=X_rot.shape)

        results = []
        for k in self.k_range:
            try:
                log_w = self._log_dispersion(X_rot, k)

                log_w_ref = []
                for b in range(self.gap_n_refs):
                    try:
                        log_w_ref.append(self._log_dispersion(reference(b), k))
                    except Exception as e:
                        print(f"An error occurred for k={k} (reference {b}): {e}")
                if not log_w_ref:
                    continue

                sd = float(np.std(log_w_ref))
                results.append({
                    'k': k,
                    'Log_W': log_w,
                    'Log_W_Ref': float(np.mean(log_w_ref)),
                    'Gap': float(np.mean(log_w_ref)) - log_w,
                    'Gap_SE': sd * np.sqrt(1.0 + 1.0 / len(log_w_ref)),
                })
            except Exception as e:
                print(f"An error occurred for k={k}: {e}")

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        if df_results.empty:
            return None

        ks = df_results['k'].values
        gap = df_results['Gap'].values
        se = df_results['Gap_SE'].values

        chosen = int(ks[int(np.argmax(gap))])  # default if the 1-SE rule never triggers
        for i in range(len(ks) - 1):
            if gap[i] >= gap[i + 1] - se[i + 1]:
                chosen = int(ks[i])
                break

        if self.detailed_logs:
            print(f"\nGap-statistic selected k: {chosen}")
        return chosen