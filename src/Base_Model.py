from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted
from .utils.FuzzyClusteringMetrics import FuzzyClusteringMetrics
from scipy.spatial.distance import cdist
import statistics
import skfuzzy as fuzz
import pandas as pd
import numpy as np

# For time tracking
import time
from contextlib import contextmanager

# ==========================================
# Default hyperparameters
# ==========================================

DEFAULT_FUZZY_M = 2                    # Fuzzy clustering fuzziness exponent
DEFAULT_FUZZY_ERROR = 0.005            # Fuzzy clustering convergence threshold
DEFAULT_FUZZY_MAX_ITER = 1000          # Fuzzy clustering max iterations
DEFAULT_C_RANGE = range(2, 11)         # Candidate c values tried when choosing c
DEFAULT_FALLBACK_C = 3                 # c used if no heuristic reaches a consensus
DEFAULT_MEMBERSHIP_THRESHOLD = 0.3     # Min membership degree to count a point "in" a cluster
DEFAULT_NUM_SUBSAMPLES = 100           # Total bootstrap subsamples (~= trees) across all clusters
DEFAULT_MIN_SAMPLES = 10               # Minimum samples per bootstrap subsample


# ==========================================
# Main Fuzzy Classifier Pipeline
# ==========================================
class FuzzyEspecialistEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Modelo Base.

    1. Tomamos os dados.
    2. Nós usamos 10 métricas de clusterização por *voto da maioria* para
       definir o c ideal.
    3. Usamos o C para realizar a clusterização fuzzy usando *fuzzy c means*.
    4. Usamos a quantidade de pontos com grau de pertinência maior que um
       threshold (default = 0.3) como tamanho dos samples, em seguida
       calculamos a média dos graus de pertinência por classe para encontrar
       a proporção ideal por cluster então puxamos aleatoriamente usando os
       graus de pertinências como pesos, mas respeitando a proporção
       calculada. Fazemos isso 100/c.
    5. Treinamos *100/c* modelos para cada subsample resultando em 100
       modelos.
    6. Para a fusão de modelos, fazemos um *soft voting com pesos, em que o
       peso é equivalente ao inverso do quadrado da distância do novo ponto
       a cada cluster.* Detalhe que nós checamos se algum modelo não foi
       treinado (cluster vazio) e pulamos ele na predição.

    Stateful: maintains trained models, scalers, and imputers between fit
    and predict.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        c_range: range = DEFAULT_C_RANGE,
        fallback_c: int = DEFAULT_FALLBACK_C,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        c_override: int = None,
    ):
        # NOTE: per sklearn's BaseEstimator convention, __init__ does nothing
        # but store constructor arguments verbatim (no derived state, no
        # validation) that's what makes get_params/set_params/clone work
        # correctly with GridSearchCV, cross_validate, etc. All fit-time
        # state is created in fit() via _reset_state() instead.
        self.detailed_logs = detailed_logs
        self.debug_folder = debug_folder
        self.random_state = random_state

        self.fuzzy_m = fuzzy_m
        self.fuzzy_error = fuzzy_error
        self.fuzzy_max_iter = fuzzy_max_iter
        self.c_range = c_range
        self.fallback_c = fallback_c
        self.membership_threshold = membership_threshold
        self.num_subsamples = num_subsamples
        self.min_samples = min_samples
        self.c_override = c_override

    def _reset_state(self):
        """
        (Re)initializes all fit-time state. Called at the start of fit()
        only (refer to the note there), so a single instance
        can be reused across cross-validation folds without leaking state
        from a previous fold, and so an unfitted instance has no fit-time
        attributes at all (the convention sklearn's check_is_fitted relies on).
        """
        self.models = {}
        self.cluster_centers = None
        self.c = None
        self.trained_features = None  # To track column names
        self.classes_ = None          # Track global classes
        self.balanced_datasets = {}   # Store for external access if needed

        # Extra attributes kept for validation/inspection
        self.raw_cluster_datasets = None
        self.last_y_train = None
        self.c_optimization_results = None
        self.last_training_membership = None
        self.last_training_data_proc = None

        self.timings_ = {}            # fit-time stage timings (seconds)
        self.predict_timings_ = {}    # last predict_proba call timings (seconds)

    @contextmanager
    def _timer(self, store, name):
        """Records elapsed wall-clock seconds for a block into store[name]."""
        start = time.perf_counter()
        try:
            yield
        finally:
            store[name] = store.get(name, 0.0) + (time.perf_counter() - start)

    # -------------------------------------------------------------------------
    # Optimization Logic (Finding C)
    # -------------------------------------------------------------------------
    @staticmethod
    def _metric_usable(df_results, metric):
        """
        True if `metric` is a column with at least one real value. A metric can
        be NaN for every c; idxmax/idxmin raise on an all-NaN column, so such
        metrics simply don't vote.
        """
        return metric in df_results.columns and df_results[metric].notna().any()

    def _find_best_c_heuristic(self, df_results):
        """Analyzes metrics DataFrame to suggest optimal c."""
        if df_results.empty:
            print("No metrics calculated.")
            return None

        recommendations = {}

        # MAXIMIZE metrics
        max_metrics = ['Silhouette', 'Calinski_Harabasz', 'Dunn', 'FPC', 'MPC']
        for metric in max_metrics:
            if self._metric_usable(df_results, metric):
                best_c = df_results.loc[df_results[metric].idxmax()]['c']
                recommendations[metric] = int(best_c)

        # MINIMIZE metrics
        min_metrics = ['Davies_Bouldin', 'FPE', 'Xie_Beni', 'Fukuyama_Sugeno']
        for metric in min_metrics:
            if self._metric_usable(df_results, metric):
                best_c = df_results.loc[df_results[metric].idxmin()]['c']
                recommendations[metric] = int(best_c)

        # ELBOW Method
        if self._metric_usable(df_results, 'Distortion'):
            x = df_results['c'].values
            y = df_results['Distortion'].values
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            vec_line = p2 - p1

            max_dist = 0
            best_elbow_c = x[0]

            for i in range(len(x)):
                p3 = np.array([x[i], y[i]])
                vec_point = p1 - p3
                cross_product = vec_line[0] * vec_point[1] - vec_line[1] * vec_point[0]
                dist = np.abs(cross_product) / np.linalg.norm(vec_line)
                if dist > max_dist:
                    max_dist = dist
                    best_elbow_c = x[i]

            recommendations['Distortion_Elbow'] = int(best_elbow_c)

        if self.detailed_logs:
            for metric, c_val in recommendations.items():
                print(f"{metric}: {c_val}")

        votes = list(recommendations.values())
        if not votes:
            return None

        try:
            suggested_c = statistics.mode(votes)
            if self.detailed_logs: print(f"\nConsensus Suggested c: {suggested_c}")
            return suggested_c
        except statistics.StatisticsError:
            fallback = recommendations.get('Silhouette', votes[0])
            if self.detailed_logs: print(f"\nNo clear consensus. Falling back to Silhouette Score: {fallback}")
            return fallback

    def get_best_c(self, X_train_processed):
        # skfuzzy's cmeans (init=None) and the metrics draw from numpy's global RNG.
        # Seed it here so the c sweep is reproducible (run_fuzzy_clustering only
        # seeds later, after c is already chosen).
        np.random.seed(self.random_state)
        X_train_values = np.asarray(X_train_processed)
        X_train_T = X_train_values.T

        results = []
        for c in self.c_range:
            try:
                cntr, u = self._fit_fuzzy_clustering(X_train_T, c)
                metrics_obj = FuzzyClusteringMetrics(X_train_values, u, cntr, self.fuzzy_m)
                metrics = metrics_obj.calculate_all()
                results.append(metrics)
            except Exception as e:
                print(f"An error occurred for c={c}: {e}")

        df_results = pd.DataFrame(results)
        self.c_optimization_results = df_results
        return self._find_best_c_heuristic(df_results)

    # -------------------------------------------------------------------------
    # Core Algorithms (Clustering & Sampling)
    # -------------------------------------------------------------------------
    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        """
        Runs the fuzzy clustering algorithm itself (currently fuzzy c-means).

        Return contract:

            X_T: ndarray, shape (n_features, n_samples) — features x samples,
                 matching skfuzzy's convention.
            returns (centers, membership):
                centers:    ndarray, shape (c, n_features)
                membership: ndarray, shape (c, n_samples)

        `m`, `error`, `max_iter` are named for fuzzy c-means specifically; a
        different algorithm's override should reinterpret or ignore them.
        """
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
            data=X_T, c=c, m=m, error=error, maxiter=max_iter, init=None
        )
        return cntr, u

    def run_fuzzy_clustering(self, X_train_processed, y_train_fold, c, m=None, error=None, max_iter=None):
        SEED_VALUE = self.random_state
        np.random.seed(SEED_VALUE)
        X_train_T = np.asarray(X_train_processed).T

        try:
            cntr, u = self._fit_fuzzy_clustering(X_train_T, c, m=m, error=error, max_iter=max_iter)

            # Store centers for later if needed
            self.cluster_centers = cntr
            self.c = c

            X_clustering = pd.DataFrame(X_train_processed).copy()
            X_clustering['target'] = np.asarray(y_train_fold)

            membership_cols = [f'grau_pertencimento_cluster_{i}' for i in range(c)]
            X_clustering[membership_cols] = u.T

            self.last_training_membership = u

            return X_clustering, membership_cols

        except Exception as e:
            print(f"\nAn error occurred during FCM execution: {e}")
            return None, None

    def _create_cluster_datasets(self, X_clustering, membership_cols, c, threshold=None, num_subsamples=None, min_samples=None):
        threshold = self.membership_threshold if threshold is None else threshold
        num_subsamples = self.num_subsamples if num_subsamples is None else num_subsamples
        min_samples = self.min_samples if min_samples is None else min_samples

        cluster_datasets = {}
        target_list = X_clustering['target']

        for n in range(num_subsamples // c):

            # Generate a unique seed for this specific iteration sequence
            # If self.random_state is 42, iteration 0 is 42, iteration 1 is 43, etc.
            current_seed = self.random_state + n if self.random_state is not None else None

            for i in range(c):
                cluster_name = f'cluster_{i}_{n}_data'
                membership_prob_col = membership_cols[i]
                probabilities = X_clustering[membership_prob_col]

                total = int((probabilities > threshold).sum())
                total = max(total, min_samples)

                data_helper = pd.DataFrame({
                    'index': X_clustering.index,
                    'target': target_list,
                    'membership': probabilities
                })

                grouped_means = data_helper.groupby('target')['membership'].mean()
                if grouped_means.sum() == 0:
                    proportional_probs = pd.Series(1.0 / len(grouped_means), index=grouped_means.index)
                else:
                    proportional_probs = grouped_means / grouped_means.sum()

                sampled_subsets = []

                # Stratified Sampling Loop
                for target_val, prop in proportional_probs.items():
                    n_class = int(round(total * prop))

                    if n_class == 0 and prop > 0:
                        n_class = 1

                    if n_class > 0:
                        class_data = data_helper[data_helper['target'] == target_val]

                        if not class_data.empty:
                            weights = None if class_data['membership'].sum() == 0 else 'membership'

                            class_sample_indices = class_data.sample(
                                n=n_class,
                                weights=weights,
                                replace=True,
                                random_state=current_seed
                            )['index']

                            sampled_subsets.append(X_clustering.loc[class_sample_indices])

                # Combine all class samples to form the cluster dataset
                if sampled_subsets:
                    cluster_datasets[cluster_name] = pd.concat(sampled_subsets).sample(frac=1, random_state=current_seed)
                else:
                    cluster_datasets[cluster_name] = X_clustering.sample(n=min_samples, replace=True, random_state=current_seed)

        return cluster_datasets

    def _train_classifiers(self, cluster_datasets):
        for cluster_name, data in cluster_datasets.items():

            # Extract just the base cluster index to group models properly
            cluster_idx = cluster_name.split('_')[1]
            base_cluster_key = f'cluster_{cluster_idx}'

            if base_cluster_key not in self.models:
                self.models[base_cluster_key] = []

            features = [c for c in data.columns if 'grau_pertencimento' not in str(c) and c != 'target']
            X = data[features]
            y = data['target']

            model = DecisionTreeClassifier(max_depth=None, random_state=self.random_state)
            model.fit(X, y)

            # Append to the grouped cluster list, not the individual dataset name
            self.models[base_cluster_key].append(model)

    # -------------------------------------------------------------------------
    # Main Public Methods (Fit / Predict)
    # -------------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Main orchestration method:
        1. Find C
        2. Cluster
        3. Create Sub-Datasets (Weighted + Balanced)
        4. Train Ensemble
        """
        self._reset_state()
        t = self.timings_

        with self._timer(t, 'total_fit'):
            if hasattr(X_train, 'columns'):
                self.trained_features = list(X_train.columns)
            else:
                self.trained_features = [f"f_{i}" for i in range(X_train.shape[1])]

            self.classes_ = np.sort(pd.Series(y_train).unique())
            self.last_y_train = np.asarray(y_train)
            self.last_training_data_proc = X_train

            # Find Best C (skipped when c_override fixes it manually)
            if self.c_override is not None:
                suggested_c = self.c_override
            else:
                with self._timer(t, 'find_best_c'):
                    suggested_c = self.get_best_c(X_train)
                if suggested_c is None:
                    suggested_c = self.fallback_c

            # Fuzzy Clustering
            with self._timer(t, 'fuzzy_clustering'):
                X_clustering, membership_cols = self.run_fuzzy_clustering(
                    X_train, y_train, c=suggested_c
                )

            if X_clustering is None:
                raise RuntimeError("Clustering failed.")

            # Dataset Creation (Weighted Bootstrapping)
            with self._timer(t, 'create_cluster_datasets'):
                cluster_datasets = self._create_cluster_datasets(
                    X_clustering, membership_cols, suggested_c
                )

            self.raw_cluster_datasets = cluster_datasets
            self.balanced_datasets = cluster_datasets  # Store for external access if needed

            # Train
            with self._timer(t, 'train_classifiers'):
                self._train_classifiers(cluster_datasets)

        return self

    def predict_proba(self, X_test):
        """Inference using Weighted Soft Voting based on cluster proximity."""

        check_is_fitted(self, attributes=["classes_"])
        if not self.models:
            raise RuntimeError("Models not trained.")

        self.predict_timings_ = {}
        t = self.predict_timings_

        with self._timer(t, 'total_predict_proba'):
            n_samples = X_test.shape[0]
            n_classes = len(self.classes_)

            # Calculate distances from each sample to each cluster center
            # Shape: (n_samples, n_clusters)

            with self._timer(t, 'cluster_weights'):
                if hasattr(X_test, 'columns'):
                    X_test_proc = X_test[self.trained_features].values
                else:
                    X_test_proc = np.asarray(X_test)

                distances = cdist(X_test_proc, self.cluster_centers, metric='euclidean')

                # Convert distances to weights (Squared Inverse Distance Weighting)
                # Adding a tiny epsilon to avoid division by zero if a point sits on a center
                weights = 1.0 / (distances + 1e-9) ** 2

                # Normalize weights so they sum to 1 for each sample
                # Shape: (n_samples, n_clusters)
                weights /= weights.sum(axis=1, keepdims=True)

            # This will hold our final weighted probabilities
            final_probs = np.zeros((n_samples, n_classes))

            # Iterate through clusters and apply weights
            # Note: This assumes self.models is a dict where keys match cluster indices
            for cluster_name, model_list in self.models.items():

                if not model_list:
                    continue
                # Extract the integer from the string 'cluster_0_data'
                # This splits by underscore and takes the second element: ['cluster', '0', 'data']
                try:
                    cluster_idx = int(cluster_name.split('_')[1])
                except (ValueError, IndexError):
                    raise ValueError(f"Could not extract cluster index from key: {cluster_name}")

                # Get the weights for this specific cluster index
                cluster_weight = weights[:, cluster_idx].reshape(-1, 1)

                # Average of models within this specific cluster
                cluster_model_probs = []
                for model in model_list:
                    with self._timer(t, 'tree_predict_proba'):
                        local_p = model.predict_proba(X_test)
                        if local_p.ndim == 1:
                            local_p = np.expand_dims(local_p, axis=1)
                        local_classes = model.classes_

                    with self._timer(t, 'class_remapping'):
                        full_p = np.zeros((n_samples, n_classes))
                        for i, cls in enumerate(local_classes):
                            global_idx = np.where(self.classes_ == cls)[0][0]
                            full_p[:, global_idx] = local_p[:, i]

                    cluster_model_probs.append(full_p)
                with self._timer(t, 'aggregation'):
                    avg_cluster_p = np.mean(cluster_model_probs, axis=0)
                    final_probs += avg_cluster_p * cluster_weight

        return final_probs

    def predict(self, X_test):
        """
        Inference:
        1. Get probabilities (Shape: n_samples, n_classes)
        2. Pick the index of the highest probability (Shape: n_samples,)
        3. Return actual class labels
        """
        # This returns a 2D array (e.g., 50x2 or 50x3)
        probs = self.predict_proba(X_test)

        # np.argmax(..., axis=1) collapses the matrix into a 1D array of indices
        best_indices = np.argmax(probs, axis=1)

        # Map indices back to the original labels (e.g., [0, 1, 2])
        # This ensures the output is a 1D array of shape (n_samples,)
        return self.classes_[best_indices]

    def evaluate(self, X_test, y_test):
        """
        Returns (accuracy, auc) for a held-out set.

        Kept separate from `score()` on purpose: sklearn's ClassifierMixin
        (which this class now inherits) supplies a standard `score(X, y)`
        that returns a single float (mean accuracy) via self.predict — that
        convention is what GridSearchCV, cross_val_score, and cross_validate
        rely on. This method keeps the original (accuracy, auc) tuple for
        direct/manual use; for AUC inside sklearn's CV tooling, pass
        `scoring=["accuracy", "roc_auc"]` (or a custom scorer) instead of
        calling this method.
        """
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
            # Handle Binary vs Multiclass
            if probs.shape[1] == 2:
                # Binary classification: pass only the probabilities of the positive class
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                # Multiclass classification: explicitly provide the labels
                # This prevents errors if a class is missing from this specific test fold
                labels = self.classes_ if hasattr(self, 'classes_') else np.unique(y_true)
                auc = roc_auc_score(y_true, probs, multi_class='ovr', labels=labels)

        except ValueError as e:
            print(f" AUC Warning in this fold: {e}")

            # If it's 1-class fold, AUC remains mathematically impossible.
            auc = np.nan

        return acc, auc