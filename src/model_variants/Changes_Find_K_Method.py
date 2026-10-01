"""
Variants of FuzzyEspecialistEnsembleClassifier for the comparative analysis.

Base_Model.py is used as a base class so we can alter just one method per variant.

Section 2 (k-selection speed): changes only how `k` is searched for, by
overriding `get_best_k` the clustering algorithm, sampling, training, and
fusion are all untouched.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

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
        np.random.seed(self.random_state)  # reproducible k sweep (see Base_Model.get_best_k)
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