"""
Synthetic-data experiment runner.

Each experiment is a YAML file under `tests/` describing a `sklearn.datasets.make_classification` dataset.
Each experiment is repeated on several datasets (Monte Carlo: same YAML parameters,
different seeds). Results are written to `outputs/<experiment_name>/<run_timestamp>/`.

Metrics for now only computes what was asked for -- accuracy
(mean/std over the folds), and time to train/inference (mean/std over
the folds).

Each experiment run is also logged to MLflow, under
the experiment name ("fuzzy_ensemble_experiments"). This is additive to the CSV output,
set `mlflow: false` in a config to skip it for that experiment.

Usage:
    uv run python experiments/run_synthetic_experiments.py --config tests/[test_name].yaml
        # runs just that one experiment file (the default mode)

    uv run python experiments/run_synthetic_experiments.py --sweep
        # runs every *.yaml / *.yml experiment file found under tests/ instead
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.Base_Model import FuzzyEspecialistEnsembleClassifier
from src.model_variants.Changes_Clustering_Algorithm import (
    FuzzyKMedoidsEspecialistEnsembleClassifier,
    GMMEspecialistEnsembleClassifier,
    PossibilisticCMeansEspecialistEnsembleClassifier,
)
from src.model_variants.Changes_Find_K_Method import (FastKSelectionEnsembleClassifier, 
                                                      CVKSelectionEnsembleClassifier, 
                                                      SupervisedEntropyKSelectionEnsembleClassifier,
                                                      StabilityKSelectionEnsembleClassifier,
                                                      GapStatisticKSelectionEnsembleClassifier)
# Perceptron variant is disabled for now.
# from src.model_variants.Changes_Base_Classifier import PerceptronEspecialistEnsembleClassifier
from src.baselines.baselines_decision_tree import (
    BaggingBaseline,
    BoostingBaseline,
    DecisionTreeBaseline,
    RandomForestBaseline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Monte Carlo repetitions: how many datasets are generated per experiment
# (same YAML parameters, different random_state). Override with `n_datasets:`
# in the YAML or --n-datasets on the command line.
DEFAULT_N_DATASETS = 10

# Every model available to an experiment config's `models:` list, keyed by the
# name used in the YAML.
MODEL_REGISTRY = {
    # Main model.
    "FuzzyEspecialistEnsembleClassifier": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state),
    "Suggested_c": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state),
    "c2": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=2),
    "c3": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=3),
    "c4": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=4),
    "c5": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=5),
    "c6": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=6),
    "c7": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=7),
    "c8": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=8),
    "c9": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=9),
    "c10": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, c_override=10),
    # "Fuzzy_m1.2": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, fuzzy_m=1.2),
    # "Fuzzy_m1.5": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, fuzzy_m=1.5),
    # "Fuzzy_m2.5": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, fuzzy_m=2.5),
    # "Fuzzy_m3": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state, fuzzy_m=3),
    # # Clustering-algorithm / k-selection / base-classifier variants.
    # "GMMEspecialistEnsembleClassifier": lambda random_state: GMMEspecialistEnsembleClassifier(random_state=random_state),
    # "FuzzyKMedoidsEspecialistEnsembleClassifier": lambda random_state: FuzzyKMedoidsEspecialistEnsembleClassifier(random_state=random_state),
    # "PossibilisticCMeansEspecialistEnsembleClassifier": lambda random_state: PossibilisticCMeansEspecialistEnsembleClassifier(random_state=random_state),
    # "FastKSelectionEnsembleClassifier": lambda random_state: FastKSelectionEnsembleClassifier(random_state=random_state),
    # "CVKSelectionEnsembleClassifier": lambda random_state: CVKSelectionEnsembleClassifier(random_state=random_state),
    # "SupervisedEntropyKSelectionEnsembleClassifier": lambda random_state: SupervisedEntropyKSelectionEnsembleClassifier(random_state=random_state),
    # "StabilityKSelectionEnsembleClassifier": lambda random_state: StabilityKSelectionEnsembleClassifier(random_state=random_state),
    # "GapStatisticKSelectionEnsembleClassifier": lambda random_state: GapStatisticKSelectionEnsembleClassifier(random_state=random_state),
    #"PerceptronEspecialistEnsembleClassifier": lambda random_state: PerceptronEspecialistEnsembleClassifier(random_state=random_state),
    # Baselines.
    # "DecisionTree": lambda random_state: DecisionTreeBaseline(random_state=random_state),
    # "Bagging": lambda random_state: BaggingBaseline(random_state=random_state),
    # "Boosting": lambda random_state: BoostingBaseline(random_state=random_state),
    # "RandomForestBaseline": lambda random_state: RandomForestBaseline(random_state=random_state),
}


def compute_auc(model, X_test, y_test):
    """ROC AUC from `predict_proba` (binary, or multiclass one-vs-rest).
    Returns NaN if it can't be computed (e.g. a model without predict_proba,
    or a single-class test fold)."""
    try:
        probs = model.predict_proba(X_test)
    except Exception as e:
        print(f"    AUC unavailable: {e}")
        return np.nan
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(y_test, probs[:, 1]))
        labels = getattr(model, "classes_", np.unique(y_test))
        return float(roc_auc_score(y_test, probs, multi_class="ovr", labels=labels))
    except ValueError as e:
        print(f"    AUC Warning in this fold: {e}")
        return np.nan


def load_experiment_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "name" not in config:
        raise ValueError(f"{config_path}: experiment config must have a top-level 'name'")
    if "dataset" not in config:
        raise ValueError(f"{config_path}: experiment config must have a top-level 'dataset' section")

    return config


def resolve_model_names(config: dict) -> list:
    models = config.get("models", "all")
    if models == "all" or models is None:
        return list(MODEL_REGISTRY.keys())
    return list(models)


def apply_augmentation(X: np.ndarray, augmentation: dict, random_state: int) -> np.ndarray:
    """
    Applies the skew/outlier/scale transforms.
    """
    X = X.copy()
    rng = np.random.RandomState(random_state)

    skew_factor = augmentation.get("skew_factor", 0.0)
    outlier_ratio = augmentation.get("outlier_ratio", 0.0)
    scale_factor = augmentation.get("scale_factor", 1.0)
    partial_scale = augmentation.get("partial_scale", False)

    # --- Skew (exponential transform) ---
    if skew_factor > 0:
        X = np.exp(X * skew_factor)

    # --- Outliers (random rows scaled by a random 5x-10x factor) ---
    if outlier_ratio > 0:
        n_outliers = int(len(X) * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = rng.choice(len(X), n_outliers, replace=False)
            X[outlier_indices] *= rng.uniform(5, 10, size=(n_outliers, X.shape[1]))

    # --- Scale (uniform magnitude multiplier, optionally only 2 features) ---
    if scale_factor > 1.0:
        if partial_scale and X.shape[1] >= 2:
            X[:, :2] *= scale_factor
        else:
            X *= scale_factor

    return X


def build_dataset(config: dict, random_state: int = None):
    """`random_state` overrides the YAML's dataset seed (used to draw the
    Monte Carlo repetitions)."""
    dataset_params = dict(config["dataset"])  # copy -- make_classification mutates nothing, but be safe
    if random_state is not None:
        dataset_params["random_state"] = random_state
    random_state = dataset_params.get("random_state", 42)
    X, y = make_classification(**dataset_params)

    augmentation = config.get("augmentation")
    if augmentation:
        X = apply_augmentation(X, augmentation, random_state)

    return X, y

def _summarize_timings(prefix: str, fold_timings: list) -> dict:
    """
    fold_timings: one dict per successful fold, e.g. [{'find_best_k': 12.4, ...}, ...]
    Returns {f'{prefix}_{stage}_mean_s': ..., f'{prefix}_{stage}_std_s': ...}.
    Stages missing in a fold (or models without timings, like the baselines)
    are ignored / simply produce no columns.
    """
    summary = {}
    stages = sorted({stage for t in fold_timings for stage in t})
    for stage in stages:
        values = [t[stage] for t in fold_timings if stage in t]
        summary[f"{prefix}_{stage}_mean_s"] = float(np.mean(values))
        summary[f"{prefix}_{stage}_std_s"] = float(np.std(values))
    return summary

def run_one_model(model_name: str, X, y, skf: StratifiedKFold, random_state: int) -> dict:
    """Runs k-fold CV for a single model, returning summary stats. Any
    exception during a fold is caught and recorded."""
    fold_acc, fold_auc, fold_train_time, fold_infer_time = [], [], [], []
    fold_fit_stages, fold_predict_stages = [], [] 

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MODEL_REGISTRY[model_name](random_state)

        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t0

            fit_stages = dict(getattr(model, "timings_", {}) or {})
            predict_stages = dict(getattr(model, "predict_timings_", {}) or {})

            acc = accuracy_score(y_test, y_pred)
            auc = compute_auc(model, X_test, y_test)

            print(f"  [{model_name}] fold {fold_idx}: acc={acc:.4f} auc={auc:.4f} "
                  f"train={train_time:.3f}s infer={infer_time:.3f}s")

            fold_acc.append(acc)
            fold_auc.append(auc)
            fold_train_time.append(train_time)
            fold_infer_time.append(infer_time)
            fold_fit_stages.append(fit_stages)
            fold_predict_stages.append(predict_stages)


        except Exception as e:
            print(f"  [{model_name}] fold {fold_idx} FAILED: {e}")

    if not fold_acc:
        return {
            "model": model_name,
            "n_folds_ok": 0,
            "accuracy_mean": np.nan,
            "accuracy_std": np.nan,
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "train_time_mean_s": np.nan,
            "train_time_std_s": np.nan,
            "inference_time_mean_s": np.nan,
            "inference_time_std_s": np.nan,
        }

    return {
        "model": model_name,
        "n_folds_ok": len(fold_acc),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "auc_mean": float(np.nanmean(fold_auc)) if not np.all(np.isnan(fold_auc)) else np.nan,
        "auc_std": float(np.nanstd(fold_auc)) if not np.all(np.isnan(fold_auc)) else np.nan,
        "train_time_mean_s": float(np.mean(fold_train_time)),
        "train_time_std_s": float(np.std(fold_train_time)),
        "inference_time_mean_s": float(np.mean(fold_infer_time)),
        "inference_time_std_s": float(np.std(fold_infer_time)),
        **_summarize_timings("fit", fold_fit_stages),
        **_summarize_timings("predict", fold_predict_stages), 
    }


def run_experiment(config_path: Path, outputs_dir: Path, n_datasets: int = None) -> pd.DataFrame:
    """Runs the experiment on `n_datasets` Monte Carlo datasets: same YAML
    parameters, dataset seed = base seed + i. Writes `results.csv` (one row per
    model per dataset) and `results_summary.csv` (mean/std across datasets)."""
    config = load_experiment_config(config_path)
    experiment_name = config["name"]

    if n_datasets is None:
        n_datasets = config.get("n_datasets", DEFAULT_N_DATASETS)

    print(f"\nExperiment: {experiment_name} ({config_path.name}) -- {n_datasets} dataset(s) ===")

    cv_config = config.get("cv", {})
    n_splits = cv_config.get("n_splits", 5)
    base_dataset_seed = config["dataset"].get("random_state", 42)
    base_cv_seed = cv_config.get("random_state", base_dataset_seed)

    model_names = resolve_model_names(config)
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        print(f"  Skipping unknown model name(s) not in MODEL_REGISTRY: {unknown}")
        model_names = [m for m in model_names if m in MODEL_REGISTRY]

    use_mlflow = config.get("mlflow", True)
    if use_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("fuzzy_ensemble_experiments")

    results = []

    if use_mlflow:
        parent_run_cm = mlflow.start_run(run_name=f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}")
    else:
        import contextlib
        parent_run_cm = contextlib.nullcontext()

    with parent_run_cm:
        if use_mlflow:
            mlflow.log_params({
                "experiment_name": experiment_name,
                "n_splits": n_splits,
                "n_datasets": n_datasets,
                "cv_random_state": base_cv_seed,
                **{f"dataset.{k}": v for k, v in config["dataset"].items()},
                **{f"augmentation.{k}": v for k, v in config.get("augmentation", {}).items()},
            })

        for dataset_idx in range(n_datasets):
            dataset_seed = base_dataset_seed + dataset_idx
            cv_seed = base_cv_seed + dataset_idx
            print(f"\n Dataset {dataset_idx + 1}/{n_datasets} (seed={dataset_seed})")

            X, y = build_dataset(config, random_state=dataset_seed)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

            dataset_cm = (
                mlflow.start_run(run_name=f"dataset_{dataset_idx}", nested=True)
                if use_mlflow else contextlib.nullcontext()
            )
            with dataset_cm:
                if use_mlflow:
                    mlflow.log_params({"dataset_idx": dataset_idx, "dataset_seed": dataset_seed})

                for model_name in model_names:
                    print(f" Running {model_name} ({n_splits}-fold CV)...")

                    if use_mlflow:
                        with mlflow.start_run(run_name=model_name, nested=True):
                            param_probe = MODEL_REGISTRY[model_name](cv_seed)
                            if hasattr(param_probe, "get_params"):
                                mlflow.log_params(param_probe.get_params())
                            summary = run_one_model(model_name, X, y, skf, cv_seed)
                            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
                    else:
                        summary = run_one_model(model_name, X, y, skf, cv_seed)

                    results.append({"dataset_idx": dataset_idx, "dataset_seed": dataset_seed, **summary})

    results_df = pd.DataFrame(results)

    # Mean/std across the datasets, per model. Each dataset contributes its
    # own CV mean, so the std reflects dataset-to-dataset variation.
    metric_cols = [c for c in results_df.columns
                   if c.endswith("_mean") or c.endswith("_mean_s")]
    summary_df = (
        results_df.groupby("model", sort=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{col}_across_datasets_{stat}" for col, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir / experiment_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(run_dir / "results_summary.csv", index=False)

    # Copy the resolved config alongside the results for reproducibility.
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump({**config, "n_datasets": n_datasets}, f, sort_keys=False)

    print(f"\n Results written to: {results_path}")
    print(summary_df[["model", "accuracy_mean_across_datasets_mean",
                      "accuracy_mean_across_datasets_std"]].to_string(index=False))

    return results_df


def find_experiment_configs(tests_dir: Path) -> list:
    return sorted(list(tests_dir.glob("*.yaml")) + list(tests_dir.glob("*.yml")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a single experiment YAML file to run. This is the "
             "default mode -- required unless --sweep is passed.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run every *.yaml/*.yml experiment file found under --tests-dir "
             "instead of a single --config. An explicit opt-in.",
    )
    parser.add_argument(
        "--n-datasets", type=int, default=None,
        help="Monte Carlo repetitions: number of datasets generated per "
             "experiment (seed = YAML seed + i). Defaults to `n_datasets` in "
             f"the YAML, else {DEFAULT_N_DATASETS}.",
    )
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    args = parser.parse_args()

    if args.sweep:
        if args.config is not None:
            print("--sweep was passed -- ignoring --config and running every "
                  f"experiment file under {args.tests_dir} instead.")
        config_paths = find_experiment_configs(args.tests_dir)
        if not config_paths:
            print(f"No experiment configs found under {args.tests_dir}")
            return
    else:
        if args.config is None:
            parser.error(
                "Specify --config <path> to run a single experiment "
                "(the default mode), or pass --sweep to run every "
                f"*.yaml/*.yml file under {args.tests_dir} instead."
            )
        config_paths = [args.config]

    for config_path in config_paths:
        run_experiment(config_path, args.outputs_dir, args.n_datasets)


if __name__ == "__main__":
    main()
