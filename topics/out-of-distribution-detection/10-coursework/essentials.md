# OOD Detection — Essentials

## The problem

A deployed classifier assigns a class to *any* input, including inputs unlike anything
in training — softmax confidence does not degrade gracefully; a network can be >99%
"confident" on pure noise. OOD detection asks a prior question: "is this input from the
training distribution?", so the system can abstain, flag, or route before trusting the
class prediction.

## Vocabulary

- **In-distribution (ID) / out-of-distribution (OOD)** — ID = drawn from (or like) the
  training distribution; OOD = not.
- **Covariate shift** — input distribution `p(x)` changes but label semantics `p(y|x)`
  don't (new camera, lighting, weather).
- **Semantic / label shift** — the input's true class isn't in the training label set.
- **Near-OOD vs far-OOD** — near-OOD is semantically close to ID (hardest, e.g.
  CIFAR-100 vs CIFAR-10); far-OOD is obviously unrelated (easiest, e.g. random noise).
- **Open-set recognition** — must classify ID inputs correctly *and* reject unseen
  classes at test time.
- **Anomaly / novelty detection** — sibling fields; anomaly = rare-but-known deviation,
  novelty = never-seen-before pattern.
- **Selective prediction / reject option** — a classifier may output "I don't know"
  instead of a forced label, trading coverage for accuracy.
- **Aleatoric vs epistemic uncertainty** — aleatoric = irreducible data noise (more data
  won't remove it); epistemic = model's own lack of knowledge (reducible; what OOD
  inputs spike).
- **Calibration** — whether stated confidence matches empirical accuracy (a
  70%-confidence bucket should be right ~70% of the time).

## Score-and-threshold framing

Every OOD detector reduces to a scalar score `s(x)` plus a threshold `τ`: predict OOD if
`s(x) < τ` (or `>`, by convention). Method families differ only in how `s(x)` is
computed; evaluation is threshold-free unless a specific operating point is chosen.

## Method families

1. **Output/logit-based** — Max Softmax Probability (MSP, Hendrycks & Gimpel 2017):
   `max_c p(y=c|x)`; ODIN (Liang et al. 2018) adds temperature scaling + input
   perturbation to widen the ID/OOD gap; energy score (Liu et al. 2020) uses
   `-logsumexp(logits)`, avoiding softmax's normalization-induced overconfidence.
2. **Feature/distance-based** — Mahalanobis (Lee et al. 2018) fits a per-class Gaussian
   in feature space, scores by distance to nearest class mean; KNN (Sun et al. 2022)
   scores by distance to the k-th nearest training feature, nonparametric.
3. **Density/generative** — score by model likelihood `p(x)` directly. Known failure:
   deep generative models (flows, VAEs, PixelCNN) can assign *higher* likelihood to OOD
   than ID (Nalisnick et al. 2019) — likelihood tracks low-level statistics, not
   semantic fit.
4. **Uncertainty-based** — deep ensembles (Lakshminarayanan et al. 2017) and MC-dropout
   (Gal & Ghahramani 2016) approximate a predictive distribution over models;
   predictive entropy flags total uncertainty; mutual information / BALD (Houlsby et
   al. 2011) isolates epistemic uncertainty (inter-model disagreement) — the
   OOD-relevant signal.
5. **Training-time** — outlier exposure (Hendrycks et al. 2019) trains against an
   auxiliary known-OOD dataset so the model learns low confidence directly, rather
   than relying on a post-hoc score.

## Evaluation metrics

- **AUROC** — area under the ROC curve (TPR vs FPR across thresholds); threshold-free;
  0.5 = random, 1.0 = perfect.
- **FPR@95TPR** — false-positive rate at the threshold keeping 95% of ID inputs; lower
  is better, more sensitive to near-OOD failure than AUROC.
- **AUPR** — area under precision-recall; more informative than AUROC under class
  imbalance (typically far more ID than OOD examples).
- **Standard protocol** — train on ID only; score a separate OOD test set (disjoint
  dataset or held-out classes) and the ID test set with `s(x)`; compute the metrics
  above (state which class is "positive").

## Math prereqs

- **Softmax / logits** — `p_c = exp(z_c) / sum_k exp(z_k)`; `z` are pre-softmax logits.
- **Log-sum-exp** — `logsumexp(z) = log(sum_k exp(z_k))`; energy = `-logsumexp(z)`;
  stabilized by subtracting `max(z)` before exponentiating.
- **Covariance + Mahalanobis distance** — `D(x) = sqrt((x-mu)^T Sigma^-1 (x-mu))`,
  generalizing Euclidean distance via covariance `Sigma`.
- **Entropy** — `H(p) = -sum_c p_c log p_c`; spread of a predictive distribution (high
  = uncertain).
- **Mutual information (BALD)** — `I = H(E[p]) - E[H(p)]`; gap between entropy of the
  averaged prediction and the average of individual entropies; isolates inter-model
  disagreement (epistemic) from per-model uncertainty (aleatoric).
- **Uncertainty decomposition** — total = aleatoric + epistemic; ensembles/MC-dropout
  estimate the split from within- vs between-pass variance.
