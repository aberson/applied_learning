# OOD Detection — Courses & Resources

Where to go deeper once [essentials.md](essentials.md) and the paper lists ([papers-seminal.md](papers-seminal.md),
[papers-frontier.md](papers-frontier.md)) are read. Search terms given where a URL isn't certain enough to print.

## Benchmarks / toolkits

- **OpenOOD** — unified OOD-detection benchmark and codebase (Zhang, Yang et al., NeurIPS 2022
  Datasets & Benchmarks track) reimplementing most post-hoc, training-time, and density methods
  behind one CLI/config interface with shared dataloaders and metrics. The de facto reference
  implementation for reproducing numbers from the survey and frontier papers. Search GitHub for
  "OpenOOD" (maintained by the paper's authors).
- **PyTorch-OOD** — a PyTorch library packaging OOD/anomaly-detection scoring methods (MSP,
  ODIN, Mahalanobis, energy score, etc.) as drop-in wrappers around a trained classifier. Search
  GitHub for "pytorch-ood".
- **Common ID/OOD dataset pairs** — the standard benchmark combinations used across this
  literature: MNIST (ID) vs. FashionMNIST / KMNIST (OOD); CIFAR-10 (ID) vs. SVHN / CIFAR-100 /
  LSUN (OOD); ImageNet-1k (ID) vs. iNaturalist / SUN / Places / Textures (OOD, the "MOS"/large-scale
  suite). All are standard `torchvision.datasets` or easily-downloadable splits used in the
  seminal and frontier papers above.
- **ImageNet-C / CIFAR-10-C** — corruption-robustness benchmarks (Hendrycks & Dietterich, 2019,
  "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations") applying 15
  synthetic corruptions at 5 severities. Adjacent to OOD detection: covariate-shift robustness
  rather than semantic-shift detection, useful for contrasting the two failure modes. Search
  GitHub for "hendrycks/robustness".

## Surveys as entry points

- **Yang, Zhou, Li & Liu, 2021 — "Generalized Out-of-Distribution Detection: A Survey"** — the
  best single map of the field's sub-problems (OOD, anomaly, novelty, open-set) and how methods
  relate; full citation in [papers-frontier.md](papers-frontier.md).

## Blogs / explainers

- Project pages and author blog posts for OpenOOD and individual method papers (ReAct, ViM,
  KNN-OOD, energy-based OOD) typically accompany the paper on the authors' personal sites or lab
  pages — search the paper title + "project page" rather than trusting a guessed URL.
- Lilian Weng's ML blog (lilianweng.github.io) covers adjacent topics (uncertainty, anomaly
  detection framings) in her usual survey-style long-form posts — search her site's index for
  "anomaly" / "OOD" before assuming a specific post exists.

## Courses / lectures

- **Yarin Gal** (Oxford) — talks and tutorial material on Bayesian deep learning and uncertainty
  estimation (epistemic vs. aleatoric uncertainty, MC dropout), directly relevant background for
  why OOD detection is hard for point-estimate networks. Search for "Yarin Gal Bayesian Deep
  Learning" for his thesis, tutorials, and NeurIPS/ICML talk recordings.
- **Dan Hendrycks — "Introduction to ML Safety"** — a course covering robustness, monitoring
  (including OOD detection and anomaly detection as a safety-monitoring primitive), and alignment;
  Hendrycks is also a co-author of several seminal OOD papers ([papers-seminal.md](papers-seminal.md)).
  Search for "ML Safety course Hendrycks" for the syllabus and lecture material.

## Textbook-ish

- **Kevin Murphy — "Probabilistic Machine Learning: Advanced Topics" (2023)** — chapters on
  calibration, Bayesian deep learning, and uncertainty quantification give the probabilistic
  background (why softmax confidence miscalibrates, what a proper score is) underneath most
  OOD-detection scoring rules. Freely readable from the book's companion site — search
  "probml pml-book" for the current link.

## Practical / API references

- **`sklearn.metrics.roc_auc_score`** — https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
  — the standard metric for ID-vs-OOD separability used throughout this topic's notebooks.
- **`sklearn.neighbors.NearestNeighbors`** — https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
  — backs k-NN distance-based OOD scoring (the KNN-OOD approach in [papers-frontier.md](papers-frontier.md)).
- **`sklearn.neighbors.KernelDensity`** — https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
  — density-estimation baseline for the density-based scoring family in [essentials.md](essentials.md).
- **`torchvision.datasets`** — https://pytorch.org/vision/stable/datasets.html — provides
  MNIST/FashionMNIST/KMNIST/CIFAR-10/CIFAR-100/SVHN loaders used to build the ID/OOD pairs above.
