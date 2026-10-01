import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# Mirrors Base_Model.py's DEFAULT_FUZZY_* constants (same numeric values).
_DEFAULT_FUZZY_M = 2
_DEFAULT_FUZZY_ERROR = 0.005
_DEFAULT_FUZZY_MAX_ITER = 1000


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


# Baselines

class RandomForestBaseline(EvaluateMixin, RandomForestClassifier):
    """
    Plain Random Forest baseline for comparison against the fuzzy ensemble.
    """
    pass


class DecisionTreeBaseline(EvaluateMixin, DecisionTreeClassifier):
    """
    Pure Decision Tree. Same base learner configuration
    the fuzzy ensemble uses for its specialists (max_depth=None).
    """

    def __init__(self, max_depth=None, random_state=None):
        super().__init__(max_depth=max_depth, random_state=random_state)


class BaggingBaseline(EvaluateMixin, BaggingClassifier):
    """
    Bagging of Decision Trees: each tree is trained on a bootstrap sample of
    the training set.

    `estimator=None` makes sklearn use a fully-grown `DecisionTreeClassifier`.
    """

    def __init__(self, estimator=None, n_estimators=50, max_samples=1.0,
                 max_features=1.0, bootstrap=True, n_jobs=None,
                 random_state=None):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class BoostingBaseline(EvaluateMixin, AdaBoostClassifier):
    """
    Boosting of Decision Trees (AdaBoost / SAMME).

    Boosting needs *weak* learners: a fully-grown tree fits the training set
    perfectly on round 1 and boosting stops immediately, so the default base
    learner is a shallow tree (`base_max_depth`, default 1 = decision stump). 
    Pass `estimator` to override entirely.
    """

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 base_max_depth=1, random_state=None):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.base_max_depth = base_max_depth

    def fit(self, X, y, sample_weight=None):
        if self.estimator is None:
            self.estimator = DecisionTreeClassifier(
                max_depth=self.base_max_depth, random_state=self.random_state
            )
            try:
                return super().fit(X, y, sample_weight=sample_weight)
            finally:
                # Keep the constructor params untouched (sklearn clone/get_params contract).
                self.estimator = None
        return super().fit(X, y, sample_weight=sample_weight)
