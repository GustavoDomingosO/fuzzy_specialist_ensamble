# Fuzzy Specialist Ensemble

A research codebase for a fuzzy-clustering-based especialist ensemble classifier.

## Setup

```
uv sync
```

This creates the virtual environment, installs every dependency pinned in `pyproject.toml`, and installs `src`.

## The model

`FuzzyEspecialistEnsembleClassifier` (in `src/Base_Model.py`) is a clustering-based ensemble: instead of training one model on the whole dataset, it splits the training data into fuzzy, overlapping clusters, trains a small forest of a base classifier (Decision Trees as the default) per cluster, and combines their predictions weighted by how close a new point sits to each cluster. `fit()` runs four stages in order:

**1. Choosing the number of clusters (k).** For each candidate `k` in a configurable range (default 2–10), the data is fuzzy-clustered and ten different cluster-validity metrics are computed (Silhouette, Calinski-Harabasz, Dunn, Davies-Bouldin, the Fuzzy Partition Coefficient and its modified version, Fuzzy Partition Entropy, Xie-Beni, Fukuyama-Sugeno, plus an elbow read on the distortion curve). Whichever `k` the largest number of metrics agree on wins by majority vote; if there's no clear consensus, the Silhouette-preferred `k` is used instead; if nothing computes at all, a fixed fallback `k` is used.

**2. Fuzzy clustering.** The training data is clustered with that chosen `k` using fuzzy c-means. Unlike ordinary (hard) clustering, every point gets a *membership degree* between 0 and 1 for every cluster, rather than being assigned to exactly one, a point near the boundary between two clusters can meaningfully belong to both.

**3. Building per-cluster training subsamples.** For each cluster, the points whose membership degree clears a threshold (default 0.3) set the rough sample size; a stratified, weighted bootstrap sample is then drawn from that cluster (weighted by membership degree, and proportioned per class using each class's average membership, so class balance is preserved). This is repeated with a different random seed several times per cluster, so the total number of subsamples across all clusters adds up to `num_subsamples` (default 100), e.g. with 5 clusters, 20 subsamples come from each.

**4. Training the ensemble.** One Decision Tree is trained per subsample, grouped by the cluster it came from.

At inference time (`predict_proba`), a new point's distance to every cluster center is measured and converted into a normalized weight via inverse-squared-distance weighting, closer clusters are assigned a higher weight. The final prediction is a weighted average of each cluster's models' averaged probabilities; any cluster that ended up with no trained models (an empty cluster) is simply skipped.

### Hyperparameters

All tunable values are constructor arguments:

| Argument | Default | Meaning |
|---|---|---|
| `fuzzy_m` | 2 | Fuzzy clustering fuzziness exponent |
| `fuzzy_error` | 0.005 | Convergence threshold for fuzzy clustering |
| `fuzzy_max_iter` | 1000 | Max iterations for fuzzy clustering |
| `k_range` | `range(2, 11)` | Candidate k values tried during k-selection |
| `fallback_k` | 3 | k used if no heuristic reaches a consensus |
| `membership_threshold` | 0.3 | Minimum membership degree to count a point as "in" a cluster |
| `num_subsamples` | 100 | Total bootstrap subsamples (~= trees) across all clusters |
| `min_samples` | 10 | Minimum samples per bootstrap subsample |
| `random_state` | 42 | Seed, propagated through clustering, sampling, and tree training |
| `detailed_logs` | `False` | Print per-k metric scores and the k-selection reasoning |