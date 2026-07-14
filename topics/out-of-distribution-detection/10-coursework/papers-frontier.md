# OOD Detection — Frontier / Applied Papers

More recent and applied works. Grouped by mechanism; see [essentials.md](essentials.md) for
vocabulary and the score-and-threshold framing these all instantiate.

## Post-hoc scoring (no retraining)

- **Sun, Guo & Li, 2021 — "ReAct: Out-of-distribution Detection With Rectified Activations"**
  Clips penultimate-layer activations at a fixed percentile before scoring. Drop-in wrapper
  on any already-trained classifier, no retraining — near-zero marginal cost per query.

- **Sun, Ming, Zhu & Li, 2022 — "Out-of-Distribution Detection with Deep Nearest Neighbors"**
  Non-parametric: score by distance to the k-th nearest training embedding, no Gaussian
  assumption. Strong, simple baseline against any embedding space.

- **Wang, Li, Feng & Zhang, 2022 — "ViM: Out-Of-Distribution with Virtual-logit Matching"**
  Fuses a feature-space residual (the part of `x` orthogonal to the principal subspace) with
  the existing logits into one virtual-class score — combines two signal families cheaply.

- **Djurisic, Bozanic, Ashok & Liu, 2023 — "Extremely Simple Activation Shaping for Out-of-Distribution Detection"**
  ASH: prune low-magnitude activations to zero and rescale survivors at inference time, no
  training change. Among the cheapest post-hoc methods in the literature.

## Distance / density refinements

- **Ren, Fort, Liu, Roy, Padhy & Lakshminarayanan, 2021 — "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"**
  Relative Mahalanobis: subtracts a background-model term so the score isolates
  class-conditional density from a shared background. Targets the harder near-OOD regime.

## Representation learning for OOD

- **Winkens, Bunel, Roy, Stanforth, Natarajan, et al., 2020 — "Contrastive Training for Improved Out-of-Distribution Detection"**
  Shows contrastive pretraining improves downstream separability of essentially any post-hoc
  score — representation quality is itself an OOD lever, independent of the scoring rule used.

## Vision-language / zero-shot

- **Ming, Cai, Gu, Sun, Li & Li, 2022 — "Delving into Out-of-Distribution Detection with Vision-Language Representations"**
  MCM: zero-shot OOD detection from CLIP's image-text alignment, no ID-specific training at
  all. Closest analogue here to zero-shot triage of an unfamiliar LLM input.

## Survey

- **Yang, Zhou, Li & Liu, 2021 — "Generalized Out-of-Distribution Detection: A Survey"**
  Unifies OOD, anomaly, open-set, and novelty detection under one taxonomy. Read this first
  to place any of the above within the broader field.

## LLM-adjacent (selective prediction / hallucination as OOD)

- **Kadavath et al., 2022 — "Language Models (Mostly) Know What They Know"**
  LLMs' self-reported P(True) on their own answers calibrates against actual correctness —
  a usable internal confidence signal with no external detector required.

- **Kuhn, Gal & Farquhar, 2023 — "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"**
  Clusters resampled generations by semantic equivalence before computing entropy, so
  paraphrases don't inflate the uncertainty estimate. Per-query hallucination/OOD signal.

## Routing / offload connection

For a switchboard-style router deciding whether a cheap local model should answer or escalate,
ReAct / ASH / KNN are the most directly transferable: cheap, post-hoc, no retraining, and they
wrap an embedding or activation layer already computed during a normal forward pass — the same
shape as gating an escalation decision on a distance-to-cluster or activation-shaping score.
Kadavath et al. and Kuhn et al. are the natural analogue on the pure-text side: instead of
scoring pixels or logits against a fixed embedding space, score the model's own stated
confidence or the semantic spread across resampled generations, and escalate to a stronger
model once that score crosses threshold — self-assessed uncertainty standing in for distance-
based OOD scoring when there is no fixed feature space to measure against.
