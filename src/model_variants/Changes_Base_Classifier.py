"""
Variants of FuzzyEspecialistEnsembleClassifier for the comparative analysis.

Base_Model.py is used as a base class so we can alter just one method per variant.

Section 3 (base classifier): changes only the base classifier used during training.
"""

import numpy as np
from sklearn.linear_model import Perceptron

from ..Base_Model import (
    FuzzyEspecialistEnsembleClassifier,
    DEFAULT_FUZZY_M,
    DEFAULT_FUZZY_ERROR,
    DEFAULT_FUZZY_MAX_ITER,
    DEFAULT_K_RANGE,
    DEFAULT_FALLBACK_K,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_NUM_SUBSAMPLES,
    DEFAULT_MIN_SAMPLES,
)
# 5. Base classifier: Perceptron instead of Decision Tree
class _HardVoteProbaWrapper:
    """
    Wraps a classifier with no `predict_proba` (Perceptron has none at all --
    it's not a probabilistic model) so it still satisfies the
    (predict_proba, classes_) contract the base class's inherited
    `predict_proba` relies on, by turning a hard prediction into a one-hot
    vector.

    Chosen deliberately over probability calibration (e.g.
    `CalibratedClassifierCV`): calibration needs its own internal
    cross-validation, which is fragile exactly where this ensemble would
    need it most -- the per-cluster bootstrap subsamples `fit()` trains on
    can be small (`min_samples` defaults to 10), and a stratified internal
    CV split can fail outright if some class has fewer rows than folds.
    One-hot hard voting has no such failure mode, and it's also the
    standard way ensembles of unstable/weak linear classifiers like
    Perceptron are combined in the literature -- not a compromise made only
    for robustness.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        preds = self.base_estimator.predict(X)
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        probs = np.zeros((len(preds), len(self.classes_)))
        for row, pred in enumerate(preds):
            probs[row, class_to_idx[pred]] = 1.0
        return probs


class PerceptronEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces the per-cluster DecisionTreeClassifier with a Perceptron,
    wrapped in `_HardVoteProbaWrapper` so it still plugs into the
    inherited, unchanged `predict_proba`/fusion logic. Only
    `_train_classifiers` is overridden.

    `perceptron_max_iter` and `perceptron_alpha` (L2 regularization
    strength) are exposed as constructor parameters; add more of
    Perceptron's own kwargs (e.g. `penalty`) the same way if needed.
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
        perceptron_max_iter: int = 1000,
        perceptron_alpha: float = 0.0001,
    ):
        super().__init__(
            debug_folder=debug_folder,
            detailed_logs=detailed_logs,
            random_state=random_state,
            fuzzy_m=fuzzy_m,
            fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter,
            k_range=k_range,
            fallback_k=fallback_k,
            membership_threshold=membership_threshold,
            num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.perceptron_max_iter = perceptron_max_iter
        self.perceptron_alpha = perceptron_alpha

    def _train_classifiers(self, cluster_datasets):
        for cluster_name, data in cluster_datasets.items():
            cluster_idx = cluster_name.split('_')[1]
            base_cluster_key = f'cluster_{cluster_idx}'

            if base_cluster_key not in self.models:
                self.models[base_cluster_key] = []

            features = [c for c in data.columns if 'grau_pertencimento' not in str(c) and c != 'target']
            X = data[features]
            y = data['target']

            base_model = Perceptron(
                max_iter=self.perceptron_max_iter,
                alpha=self.perceptron_alpha,
                random_state=self.random_state,
            )
            model = _HardVoteProbaWrapper(base_model)
            model.fit(X, y)

            self.models[base_cluster_key].append(model)
