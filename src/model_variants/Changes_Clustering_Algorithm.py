"""
Variants of FuzzyEspecialistEnsembleClassifier for the comparative analysis.

Base_Model.py is used as a base class so we can alter just one method per variant.

Section 1 (clustering algorithm): each class changes only the fuzzy
clustering algorithm itself, by overriding `_fit_fuzzy_clustering`, the method
both `get_best_c` (evaluating candidate c values) and `run_fuzzy_clustering`
(the final clustering used for training) call through. Everything else is inherited unchanged.
All three still return a (centers, membership) pair in the base class's
contract shape:
    centers:    ndarray, shape (c, n_features)
    membership: ndarray, shape (c, n_samples)
"""

import numpy as np
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

from ..Base_Model import (
    FuzzyEspecialistEnsembleClassifier,
)


# 1. Gaussian Mixture Model (soft clustering via posterior probabilities)
class GMMEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with a Gaussian Mixture Model. A GMM's `predict_proba` 
    gives each point a softresponsibility (posterior probability) per component, 
    which plays the same role fuzzy c-means' membership degree does everywhere 
    downstream.

    Note: `fuzzy_m` (the fuzziness exponent) has no equivalent in a GMM and
    is ignored here. `fuzzy_error` is passed through as GaussianMixture's
    `tol` (convergence tolerance), a different quantity than fuzzy c-means'
    membership-shift tolerance, but serving the same "when do we stop
    iterating" role.
    """

    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        X = X_T.T  # GaussianMixture expects (n_samples, n_features)
        gmm = GaussianMixture(
            n_components=c,
            tol=error,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        gmm.fit(X)

        membership = gmm.predict_proba(X).T  # (n_samples, c) -> (c, n_samples)
        centers = gmm.means_                  # (c, n_features)
        return centers, membership


# 2. Fuzzy c-medoids (cluster prototypes are real data points, not means)
def _fuzzy_c_medoids(X, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    Function used for the c_medoids variant implemented in the 
    FuzzyKMedoidsEspecialistEnsembleClassifier class.

    Fuzzy c-medoids (FCMdd): same iterative structure as fuzzy c-means, 
    except each cluster's prototype is required to be an actual data 
    point (a medoid) rather than a computed mean.

    Returns (centers, membership, medoid_indices):
        centers:    ndarray, shape (c, n_features) -> the chosen medoids
        membership: ndarray, shape (c, n_samples)
        medoid_indices: ndarray, shape (c,) -> row indices into X
    """
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)

    dist_matrix = cdist(X, X, metric='euclidean')
    medoid_idx = rng.choice(n_samples, size=c, replace=False)

    u = np.full((c, n_samples), 1.0 / c)
    power = 2.0 / (m - 1)

    for _ in range(max_iter):
        d = np.fmax(dist_matrix[medoid_idx], 1e-10)  # (c, n_samples)
        inv_d = d ** (-power)
        u_new = inv_d / inv_d.sum(axis=0, keepdims=True)

        # Pick, for each cluster, the candidate point minimizing the
        # membership-weighted sum of distances to every other point.
        weights = u_new ** m
        costs = weights @ dist_matrix  # (c, n_samples); dist_matrix is symmetric
        new_medoid_idx = np.argmin(costs, axis=1)

        shift = np.max(np.abs(u_new - u))
        u = u_new
        medoids_changed = not np.array_equal(new_medoid_idx, medoid_idx)
        medoid_idx = new_medoid_idx

        if shift < error and not medoids_changed:
            break

    centers = X[medoid_idx]
    return centers, u, medoid_idx


class FuzzyKMedoidsEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with fuzzy c-medoids. `fuzzy_m` and `fuzzy_error`
    keep their usual meaning (fuzziness exponent, membership-shift
    convergence tolerance); medoid search uses `random_state` only for its
    initial medoid pick, since the update rule itself is deterministic given
    that start.
    """

    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        X = X_T.T  # (n_samples, n_features)
        centers, membership, _ = _fuzzy_c_medoids(
            X, c=c, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, membership


# 3. Possibilistic C-Means (typicalities, rows don't need
#    sum to 1; better suited to noisy data / outlier handling)
def _possibilistic_cmeans(X_T, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    Possibilistic C-Means. Bootstraps from a standard fuzzy c-means run 
    to get initial centers and to estimate each cluster's typicality scale (eta),
    the common way to avoid every cluster collapsing onto the same point, 
    which is PCM's well-known failure mode without a good initialization.

    Returns (centers, typicality):
        centers:    ndarray, shape (c, n_features)
        typicality: ndarray, shape (c, n_samples) -- NOT column-normalized;
                    unlike fuzzy c-means/c-medoids/GMM, a point's values
                    across clusters need not sum to 1.
    """
    cntr0, u0, _, _, _, _, _ = fuzz.cmeans(
        data=X_T, c=c, m=m, error=error, maxiter=max_iter, init=None
    )
    X = X_T.T  # (n_samples, n_features)

    d0_sq = cdist(X, cntr0, metric='sqeuclidean').T  # (c, n_samples)
    weights0 = u0 ** m
    eta = (weights0 * d0_sq).sum(axis=1) / weights0.sum(axis=1)
    eta = np.fmax(eta, 1e-10)

    centers = cntr0.copy()
    power = 1.0 / (m - 1)

    for _ in range(max_iter):
        d_sq = cdist(X, centers, metric='sqeuclidean').T  # (c, n_samples)
        t = 1.0 / (1.0 + (d_sq / eta[:, None]) ** power)

        weights = t ** m
        new_centers = (weights @ X) / weights.sum(axis=1, keepdims=True)

        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < error:
            break

    return centers, t


class PossibilisticCMeansEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with possibilistic c-means (PCM).

    Worth knowing before using this one: PCM's typicalities deliberately
    don't sum to 1 across clusters for a point, which is exactly what makes
    it more robust to noise/outliers -- but it also means the c-selection
    metrics that assume a normalized partition (FPC, MPC, FPE in
    FuzzyClusteringMetrics) are less meaningful here than the hard-label-
    based ones (Silhouette, Calinski-Harabasz, Davies-Bouldin, the elbow
    method), which only depend on each point's argmax cluster and stay valid
    regardless of normalization. `membership_threshold` (default 0.3) was
    tuned with fuzzy c-means' normalized memberships in mind and may need
    re-tuning for PCM's typicality scale.
    """

    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        centers, typicality = _possibilistic_cmeans(
            X_T, c=c, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, typicality


# 4. FCM-PFCV (fuzzy c-means with a single adaptive quadratic distance defined
#    by a pooled fuzzy covariance matrix)
def _fcm_pfcv(X_T, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    FCM-PFCV of de Carvalho et al. (2006, Fuzzy Sets and Systems 157, sec. 3.1.1).

    Fuzzy c-means where the distance is d_M^2(x, g) = (x - g)^T M (x - g), with
    one matrix M shared by every cluster. Each iteration:
        1. prototypes (Eq. 3):  g_i = sum_k u_ik^m x_k / sum_k u_ik^m
        2. matrix M   (Eq. 7):  Q = sum_i sum_k u_ik^m (x_k - g_i)(x_k - g_i)^T
                                M = det(Q)^(1/p) Q^-1          (so det(M) = 1)
        3. membership (Eq. 8):  u_ik = [sum_h (d_M^2(x_k, g_i) / d_M^2(x_k, g_h))^(1/(m-1))]^-1
    and stops when |J(t) - J(t-1)| <= error, with J = sum u_ik^m d_M^2(x_k, g_i),
    or after `max_iter` iterations. `error` is therefore a tolerance on J (the
    paper's criterion), not on the membership shift like the other variants.

    The initial membership is random (seeded by `random_state`). det(Q) goes
    through `slogdet` to avoid under/overflow, and Q gets a tiny ridge so a
    singular Q (redundant or constant features) can still be inverted.

    Returns (centers, membership):
        centers:    ndarray, shape (c, n_features)
        membership: ndarray, shape (c, n_samples), columns sum to 1
    """
    X = X_T.T  # (n_samples, n_features)
    n_samples, p = X.shape
    rng = np.random.default_rng(random_state)

    u = rng.random((c, n_samples)) + 1e-3
    u /= u.sum(axis=0, keepdims=True)

    exponent = 1.0 / (m - 1)
    j_prev = np.inf

    for _ in range(max_iter):
        w = u ** m
        centers = (w @ X) / w.sum(axis=1, keepdims=True)  # (c, p)

        diff = X[None, :, :] - centers[:, None, :]  # (c, n_samples, p)
        Q = np.einsum('ik,ikp,ikq->pq', w, diff, diff)
        Q += np.eye(p) * (1e-10 * np.trace(Q) / p)
        _, logdet = np.linalg.slogdet(Q)
        M = np.exp(logdet / p) * np.linalg.inv(Q)

        d_sq = np.fmax(np.einsum('ikp,pq,ikq->ik', diff, M, diff), 1e-10)  # (c, n_samples)
        inv_d = d_sq ** (-exponent)
        u = inv_d / inv_d.sum(axis=0, keepdims=True)

        j = float(((u ** m) * d_sq).sum())
        if abs(j_prev - j) <= error:
            break
        j_prev = j

    # Prototypes consistent with the final membership.
    w = u ** m
    centers = (w @ X) / w.sum(axis=1, keepdims=True)
    return centers, u


class FCM_PFCVEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with FCM-PFCV (see `_fcm_pfcv`): the clustering uses
    one adaptive quadratic distance shared by all clusters, learned from the
    pooled fuzzy covariance of the data. `fuzzy_m` is the fuzziness exponent
    and `fuzzy_error` the tolerance on the change of the criterion J.

    Only the clustering step uses the matrix M. The c-selection metrics and
    the cluster weights at inference time keep using the Euclidean distance
    to the prototypes, as in the base model.
    """

    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        centers, membership = _fcm_pfcv(
            X_T, c=c, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, membership


# 5. Gustafson-Kessel (fuzzy c-means with one adaptive quadratic distance per
#    cluster, defined by that cluster's fuzzy covariance matrix)
def _gustafson_kessel(X_T, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    Gustafson-Kessel as written in de Carvalho et al. (2006, Fuzzy Sets and
    Systems 157, sec. 3.2.1).

    Like `_fcm_pfcv`, but every cluster has its own matrix M_i, so clusters can
    take different shapes and orientations: d_Mi^2(x, g_i) = (x - g_i)^T M_i (x - g_i).
    Each iteration:
        1. prototypes (Eq. 3):  g_i = sum_k u_ik^m x_k / sum_k u_ik^m
        2. matrices   (Eq. 15): C_i = sum_k u_ik^m (x_k - g_i)(x_k - g_i)^T
                                M_i = det(C_i)^(1/p) C_i^-1        (so det(M_i) = 1)
        3. membership (Eq. 16): u_ik = [sum_h (d_Mi^2(x_k, g_i) / d_Mh^2(x_k, g_h))^(1/(m-1))]^-1
    The stopping rule is the one the paper gives for FCM-PFCV: |J(t) - J(t-1)| <= error,
    with J = sum u_ik^m d_Mi^2(x_k, g_i), or `max_iter` iterations. `error` is a
    tolerance on J, not on the membership shift like the other variants.

    The initial membership is random (seeded by `random_state`). det(C_i) goes
    through `slogdet`, and each C_i gets a tiny ridge so a singular C_i can
    still be inverted. The paper adds no other regularization: a cluster with
    fewer effective points than features gives a degenerate C_i.

    Returns (centers, membership):
        centers:    ndarray, shape (c, n_features)
        membership: ndarray, shape (c, n_samples), columns sum to 1
    """
    X = X_T.T  # (n_samples, n_features)
    n_samples, p = X.shape
    rng = np.random.default_rng(random_state)

    u = rng.random((c, n_samples)) + 1e-3
    u /= u.sum(axis=0, keepdims=True)

    exponent = 1.0 / (m - 1)
    identity = np.eye(p)
    j_prev = np.inf

    for _ in range(max_iter):
        w = u ** m
        centers = (w @ X) / w.sum(axis=1, keepdims=True)  # (c, p)

        diff = X[None, :, :] - centers[:, None, :]  # (c, n_samples, p)
        C = np.einsum('ik,ikp,ikq->ipq', w, diff, diff)  # (c, p, p)
        C += (1e-10 * np.trace(C, axis1=1, axis2=2) / p)[:, None, None] * identity
        _, logdet = np.linalg.slogdet(C)  # (c,)
        M = np.exp(logdet / p)[:, None, None] * np.linalg.inv(C)  # (c, p, p)

        d_sq = np.fmax(np.einsum('ikp,ipq,ikq->ik', diff, M, diff), 1e-10)  # (c, n_samples)
        inv_d = d_sq ** (-exponent)
        u = inv_d / inv_d.sum(axis=0, keepdims=True)

        j = float(((u ** m) * d_sq).sum())
        if abs(j_prev - j) <= error:
            break
        j_prev = j

    # Prototypes consistent with the final membership.
    w = u ** m
    centers = (w @ X) / w.sum(axis=1, keepdims=True)
    return centers, u


class GKEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with Gustafson-Kessel (see `_gustafson_kessel`): each
    cluster has its own adaptive quadratic distance, learned from the cluster's
    fuzzy covariance. `fuzzy_m` is the fuzziness exponent and `fuzzy_error` the
    tolerance on the change of the criterion J.

    Only the clustering step uses the matrices M_i. The c-selection metrics and
    the cluster weights at inference time keep using the Euclidean distance
    to the prototypes, as in the base model.
    """

    def _fit_fuzzy_clustering(self, X_T, c, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        centers, membership = _gustafson_kessel(
            X_T, c=c, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, membership
