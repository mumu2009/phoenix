# GNN-GA Learner Algorithm

## 1. Goal

The GNN-GA learner combines graph neural message passing and genetic search to optimize architecture parameters and training hyperparameters over text-induced graphs.

## 2. Graph Construction

From corpus token sequences:

1. Build token co-occurrence graph with sliding window.
2. Add bidirectional edges for local context coupling.
3. Compute normalized adjacency with self-loops.

Adjacency normalization:

$$
\hat{A}=D^{-1/2}(A+I)D^{-1/2}.
$$

## 3. Node Features

Node feature matrix $X\in\mathbb{R}^{N\times d}$ contains:

- Hash-based dense seed features.
- Degree-derived statistics (`log1p(degree)`).
- Optional structural signals from higher-order topology.

This gives stable embeddings without requiring external feature stores.

## 4. GNN Forward Model

Two-layer message passing model:

$$
H = \text{ReLU}(\hat{A}XW_1+b_1),
$$

$$
P = \text{softmax}(\hat{A}HW_2+b_2).
$$

`P` is node-level class distribution used for unsupervised quality scoring.

## 5. Fitness Function

Fitness combines smoothness and entropy regularization:

- Smoothness: cosine similarity on connected node predictions.
- Entropy penalty: avoid over-diffuse distributions.

$$
\text{fitness}(P)=\text{smooth}(P,\hat{A})-\lambda\,\text{entropy}(P).
$$

## 6. Genome Encoding

A genome stores:

- GNN weights `W1`, `W2`.
- Biases `b1`, `b2`.
- Learning-rate-like scalar.
- Dropout-like scalar.

Genetic operators:

1. Initialization from Gaussian priors.
2. Parent selection by fitness ranking.
3. Element-wise crossover.
4. Sparse Gaussian mutation.
5. Elite retention.

## 7. Evolution Loop

For each generation:

1. Evaluate all genomes with current graph and features.
2. Record best fitness and hyperparameters.
3. Keep top-K elites.
4. Fill population via crossover + mutation.

After final generation, re-evaluate and export best genome metrics.

## 8. Engineering Contract

`IGnnGaLearner` methods:

- `evolve(gens)` runs evolution rounds and returns JSON metrics.
- `latest()` returns last evolution result.
- `refreshTests(dir)` reloads corpus and rebuilds token graph basis.

This module is bound via factory registration and remains compatible with existing gateway orchestration.

## 9. Extension Direction

The current design is ready for:

- GraphScope/DGL backend feature enrichment.
- Multi-objective evolution (fitness, sparsity, latency).
- NSGA-II style Pareto front search.
- Hybrid gradient-evolution alternation.
