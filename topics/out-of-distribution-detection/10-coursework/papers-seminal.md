# Seminal papers — OOD detection

Reading order below is roughly historical/dependency order within each group, but **start with
MSP** (group 2) — everything downstream reacts to or extends it. Hands-on notebook mapping:
MSP -> `70-handson/notebooks/02-*`, ODIN/energy -> `03-*`, Mahalanobis -> `04-*`,
ensembles/MC-dropout -> `05-*` (only `01-ood-intuition-2d.ipynb` exists as of this writing;
later numbers are planned).

## 1. Output / logit scores — start here

- **Hendrycks & Gimpel, 2017** — "A Baseline for Detecting Misclassified and Out-of-Distribution
  Examples in Neural Networks" ([arxiv:1610.02136](https://arxiv.org/abs/1610.02136)).
  Introduces Maximum Softmax Probability (MSP): use the softmax confidence itself as the OOD
  score, no retraining needed. Also defines the AUROC/AUPR evaluation protocol nearly every later
  paper reuses. The baseline everything else is measured against. Notebook: `02-*`.

- **Liang, Li & Srikant, 2018** — "Enhancing the Reliability of Out-of-distribution Image
  Detection in Neural Networks" ([arxiv:1706.02690](https://arxiv.org/abs/1706.02690)). ODIN:
  temperature-scale the logits and add a small adversarial-style input perturbation before
  scoring. Both tricks widen the ID/OOD softmax gap with no retraining. Notebook: `03-*`.

- **Liu, Wang, Owens & Li, 2020** — "Energy-based Out-of-distribution Detection"
  ([arxiv:2010.03759](https://arxiv.org/abs/2010.03759)). Replaces softmax probability with the
  energy score E(x) = -T logsumexp(logits/T) — theoretically grounded, less prone to softmax's
  overconfidence saturation, and can be used as a training-time regularizer too. Notebook: `03-*`.

## 2. The overconfidence problem (motivating context)

- **Nguyen, Yosinski & Clune, 2015** — "Deep Neural Networks are Easily Fooled" (author+year only,
  no verified id). Shows nets emit >99% confidence on images unrecognizable to humans (static,
  fractal patterns). The empirical shock that motivates the whole field: softmax confidence is not
  a reliable "I don't know" signal.

## 3. Feature / distance

- **Lee, Lee, Lee & Shin, 2018** — "A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks" ([arxiv:1807.03888](https://arxiv.org/abs/1807.03888)). Fits a
  class-conditional Gaussian to penultimate-layer features and scores by Mahalanobis distance —
  moves detection from output space into feature space, also catches adversarial examples.
  Notebook: `04-*`.

## 4. Uncertainty

- **Lakshminarayanan, Pritzel & Blundell, 2017** — "Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles" ([arxiv:1612.01474](https://arxiv.org/abs/1612.01474)). Train
  several independently-initialized networks; disagreement across the ensemble is the uncertainty
  signal. Simple, strong, became the practical default over Bayesian NNs. Notebook: `05-*`.

- **Gal & Ghahramani, 2016** — "Dropout as a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning" ([arxiv:1506.02142](https://arxiv.org/abs/1506.02142)). Keeping
  dropout active at inference and sampling multiple forward passes approximates a Bayesian
  posterior — cheap uncertainty estimates from an already-trained net. Notebook: `05-*`.

## 5. Calibration & training-time

- **Guo, Pleiss, Sun & Weinberger, 2017** — "On Calibration of Modern Neural Networks"
  ([arxiv:1706.04599](https://arxiv.org/abs/1706.04599)). Shows modern nets (deeper, wider) are
  systematically overconfident despite better accuracy, and that single-parameter temperature
  scaling fixes most of the miscalibration cheaply. Underpins ODIN's temperature trick.

- **Hendrycks, Mazeika & Dietterich, 2019** — "Deep Anomaly Detection with Outlier Exposure"
  ([arxiv:1812.04606](https://arxiv.org/abs/1812.04606)). Trains against an auxiliary, disjoint
  outlier dataset so the model learns to output low-confidence/high-entropy predictions on
  "OOD-shaped" inputs directly, rather than hoping it generalizes from ID training alone.

## 6. Generative models & benchmarks

- **Nalisnick, Matsukawa, Teh, Gorur & Lakshminarayanan, 2019** — "Do Deep Generative Models Know
  What They Don't Know?" ([arxiv:1810.09136](https://arxiv.org/abs/1810.09136)). Counterintuitive
  finding: deep generative models (flows, VAEs, PixelCNN) can assign *higher* likelihood to OOD
  inputs than to their own training distribution — likelihood alone is not an OOD score.

- **Ren, Liu, Fertig, Snoek, Poplin, DePristo, Dillon & Lakshminarayanan, 2019** — "Likelihood
  Ratios for Out-of-Distribution Detection" (author+year only, no verified id). Fixes Nalisnick's
  failure mode with a likelihood-ratio correction against a background model that captures generic
  input statistics (e.g. image background), isolating the semantic signal.

- **Hendrycks & Dietterich, 2019** — "Benchmarking Neural Network Robustness to Common Corruptions
  and Perturbations" ([arxiv:1903.12261](https://arxiv.org/abs/1903.12261)). Introduces
  ImageNet-C/CIFAR-C: systematic corruption/perturbation benchmarks for measuring robustness to
  distribution shift, distinct from but closely related to OOD detection proper.
