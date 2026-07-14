# OOD Detection — Related Topics

Adjacency map: how neighboring fields relate to OOD detection, not a tutorial on each.
See [essentials.md](essentials.md) for OOD detection's own vocabulary and method families.

### Anomaly / novelty detection (classical)

One-class SVM (Scholkopf et al. 2001), isolation forest (Liu et al. 2008), autoencoder
reconstruction error. OOD detection is the deep-classifier-era instance of the same
reject-the-unusual problem — classical methods score raw feature/input space directly;
OOD methods score a trained classifier's logits or learned representations instead.

### Open-set recognition

Classify known classes AND reject unknowns at test time (Scheirer et al. 2013). Same
test-time goal as OOD detection and the terms are often used interchangeably; open-set
recognition is typically posed as multi-class-with-rejection, with an OOD score plugged
in as the rejection signal. See [essentials.md](essentials.md)'s vocabulary section.

### Selective prediction / classification with a reject option

The downstream USE of an OOD score: abstain (output "I don't know") instead of a forced
label, trading coverage for accuracy (Geifman & El-Yaniv, coverage-risk curves). OOD
detection supplies one abstention signal, but selective prediction is broader — it also
covers thresholding on plain ID confidence, with no OOD notion at all.

### Model calibration

Temperature scaling, reliability diagrams, expected calibration error (ECE) (Guo et al.
2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Necessary but not
sufficient for OOD: a network calibrated to match empirical accuracy on ID test data can
still be confidently wrong on OOD inputs, since calibration only constrains behavior on
the distribution it was fit to — it says nothing about inputs from outside it.

### Uncertainty quantification

Aleatoric vs epistemic uncertainty; Bayesian NNs, deep ensembles, MC-dropout, evidential
deep learning. Epistemic uncertainty (the model's own lack of knowledge) is a natural
OOD signal — see [essentials.md](essentials.md)'s uncertainty-based method family for
the ensemble/MC-dropout/BALD treatment already covered there.

### Distribution shift / domain adaptation / covariate shift

Includes dataset-shift benchmarks ImageNet-C (Hendrycks & Dietterich 2019,
[arXiv:1903.12261](https://arxiv.org/abs/1903.12261)) and WILDS (Koh et al. 2021,
[arXiv:2012.07421](https://arxiv.org/abs/2012.07421)). OOD detection is detection — flag
and abstain; domain adaptation is coping — retrain or adjust the model to perform well
under the shift anyway. Same root cause (train/test mismatch), opposite response.

### Adversarial examples

Small, often imperceptible input perturbations that flip a classifier's prediction
(Goodfellow et al. 2015, [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)). Shares
detection machinery with OOD — the Mahalanobis distance paper (Lee et al. 2018, see
[essentials.md](essentials.md)) scores both — but a distinct threat model: adversarial
inputs are adversarially constructed to sit near the ID manifold, while OOD inputs occur
naturally and are often far from it.

### Conformal prediction

Distribution-free prediction sets with finite-sample coverage guarantees (Vovk,
Gammerman & Shafer). A complementary, more rigorous take on "how sure": it quantifies
calibrated uncertainty over the label space under an exchangeability assumption.
Standard conformal prediction assumes ID data, and its coverage guarantee degrades
silently under distribution shift — exactly OOD detection's territory.

### Outlier exposure / auxiliary-data training and generative-model likelihood

See [essentials.md](essentials.md)'s training-time (Hendrycks et al. 2019) and
density/generative (Nalisnick et al. 2019) method families. Both push the OOD notion
earlier in the pipeline: outlier exposure teaches low confidence directly on auxiliary
unfamiliar data at train time; generative likelihood instead tries to score the raw
"typicality" of an input, bypassing the classifier's features entirely.

### (Applied) Model routing / cascades and LLM hallucination / selective answering

Deciding whether a cheap or small model is "out of its depth" on a given query, and
should escalate to a larger model or refuse, is an OOD-flavored decision applied to LLM
serving — the query (or its embedding/output distribution) plays the role of the input
`x`. Hallucination-refusal training echoes outlier exposure's confidence-shaping; router
gating echoes a reject-option threshold.
