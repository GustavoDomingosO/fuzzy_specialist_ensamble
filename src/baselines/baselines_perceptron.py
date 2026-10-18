import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


class EvaluateMixin:
    """
    Adds an `evaluate(X, y) -> (accuracy, auc)` method that mirrors
    `FuzzyEspecialistEnsembleClassifier.evaluate` exactly (same binary vs.
    multiclass AUC handling, same ValueError -> np.nan fallback with a
    printed warning), so every baseline reports metrics comparably to the
    fuzzy ensemble without needing to import Base_Model.py.
    """

    def evaluate(self, X_test, y_test):
        # Ensure y_test is a 1D array
        if hasattr(y_test, "values"):
            y_true = y_test.values.ravel()
        else:
            y_true = np.asarray(y_test).ravel()

        # Get hard labels for Accuracy
        preds = self.predict(X_test)
        acc = accuracy_score(y_true, preds)

        # Get probabilities for AUC
        probs = self.predict_proba(X_test)

        try:
            if probs.shape[1] == 2:
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                # Explicitly provide the labels to prevent errors if a class is
                # missing from this specific test fold
                labels = self.classes_ if hasattr(self, 'classes_') else np.unique(y_true)
                auc = roc_auc_score(y_true, probs, multi_class='ovr', labels=labels)

        except ValueError as e:
            print(f" AUC Warning in this fold: {e}")

            # If it's a 1-class fold AUC remains mathematically impossible.
            auc = np.nan

        return acc, auc


class MLPBaseline(EvaluateMixin, MLPClassifier):
    """
    Pure (single) MLP. one base learner trained on the whole training set.
    """

    def __init__(self, hidden_layer_sizes=(64,), activation="relu",
                 solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                 max_iter=500, early_stopping=False, random_state=None):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )


class BaggingBaseline(EvaluateMixin, BaggingClassifier):
    """
    Bagging of MLPs: each MLP is trained on a bootstrap sample of the
    training set and the predicted probabilities are averaged.

    If `estimator` is None, a MLP with `base_hidden_layer_sizes` is used.
    """

    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, n_jobs=None,
                 random_state=None, base_hidden_layer_sizes=(64,),
                 base_max_iter=500):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.base_hidden_layer_sizes = base_hidden_layer_sizes
        self.base_max_iter = base_max_iter

    def fit(self, X, y, *args, **kwargs):
        if self.estimator is None:
            self.estimator = MLPClassifier(
                hidden_layer_sizes=self.base_hidden_layer_sizes,
                max_iter=self.base_max_iter,
            )
            try:
                return super().fit(X, y, *args, **kwargs)
            finally:
                # Keep the constructor params untouched (sklearn clone/get_params contract).
                self.estimator = None
        return super().fit(X, y, *args, **kwargs)


class BoostingBaseline(EvaluateMixin, AdaBoostClassifier):
    """
    Boosting of MLPs (AdaBoost / SAMME).

    Boosting needs *weak* learners: a large MLP fits the training set almost
    perfectly on round 1 and boosting stops immediately, so the default base
    learner is a small MLP (`base_hidden_layer_sizes`, default (8,)) trained
    briefly (`base_max_iter`). Pass `estimator` to override entirely.

    Requires an MLP that supports `sample_weight` in `fit`
    (scikit-learn >= 1.7).
    """

    def __init__(self, estimator=None, n_estimators=20, learning_rate=1.0,
                 base_hidden_layer_sizes=(8,), base_max_iter=300,
                 random_state=None):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.base_hidden_layer_sizes = base_hidden_layer_sizes
        self.base_max_iter = base_max_iter

    def fit(self, X, y, sample_weight=None):
        if self.estimator is None:
            self.estimator = MLPClassifier(
                hidden_layer_sizes=self.base_hidden_layer_sizes,
                max_iter=self.base_max_iter,
            )
            try:
                return super().fit(X, y, sample_weight=sample_weight)
            finally:
                # Keep the constructor params untouched (sklearn clone/get_params contract).
                self.estimator = None
        return super().fit(X, y, sample_weight=sample_weight)