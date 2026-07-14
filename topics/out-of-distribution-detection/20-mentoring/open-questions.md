# Open Questions — OOD Detection

## Self-checks (answerable from the notebooks)

- Why does high softmax confidence NOT imply in-distribution? (See [`02-msp-baseline-mnist.ipynb`](../70-handson/notebooks/02-msp-baseline-mnist.ipynb) — softmax is a normalized ratio over *seen* classes; it says nothing about whether the input resembles training data at all.)
- What exactly does AUROC measure here, and why report FPR@95TPR alongside it? What failure mode does AUROC alone hide?
- ODIN uses a temperature of ~1000 — why does extreme softening help separate ID/OOD when it would look like a no-op on argmax predictions?
- What's the sign convention that keeps an OOD-score's AUROC above 0.5 (as opposed to below)? What happens if you flip it by accident?
- Why is the energy score theoretically preferable to MSP? (See [`03-odin-and-energy.ipynb`](../70-handson/notebooks/03-odin-and-energy.ipynb).) What does energy retain that softmax normalization throws away?
- What does mutual information (BALD) capture that predictive entropy alone does not? Give an example input where they'd disagree.

## Method-choice threads

- When would you reach for a feature-distance method (Mahalanobis/KNN) over a logit method (energy)? What's each cheap/expensive to compute, and what breaks each (e.g., Mahalanobis's Gaussian-per-class assumption, KNN's sensitivity to feature-space normalization)?
- Near-OOD vs far-OOD: [`01-ood-intuition-2d.ipynb`](../70-handson/notebooks/01-ood-intuition-2d.ipynb) makes far-OOD look easy. Which method families degrade first on near-OOD, and why (logit-based methods rely on decision-boundary geometry; feature-distance methods rely on class-conditional density — which assumption breaks first as the shift shrinks)?
- How much does the choice of OOD test set (e.g., MNIST vs FashionMNIST vs Gaussian noise as "OOD") bias a reported AUROC? If a paper cherry-picks an easy OOD set, what should you demand instead?

## Deeper / unsettled threads

- Is "OOD" even well-defined without specifying the shift type — covariate (same semantics, different style/domain) vs semantic (genuinely different classes)? Does a single detector make sense across both, or do they need different scores entirely?
- Nalisnick et al. 2019: deep generative models can assign *higher* likelihood to OOD inputs than to training data. What does that say about "density = detector" as an assumption? Does it indict likelihood specifically, or density estimation as a whole?
- Does calibration (temperature-scaled, well-calibrated ID accuracy) solve OOD? Why not — what's the distinction between "confidence is honest on ID" and "confidence degrades gracefully off-distribution"?
- Can any post-hoc score be adversarially defeated? If an attacker can query the detector, what's the asymmetry between attacking a classifier and attacking an OOD gate?
- How does OOD detection transfer from vision (this topic's notebooks) to LLM/agent routing — deciding a prompt is "out of depth" for a small model? What's the LLM analogue of a logit/energy score, and does it exist without an explicit closed label set?

## Applied thread — local-model router

- For a local-model router (see workspace `switchboard/`), what score + threshold + calibration protocol would you actually deploy? How would you set the threshold with zero labeled OOD data — proxy validation set, held-out ID quantile, human-in-the-loop bootstrapping?
- How do you evaluate an OOD-gated router honestly? Per [`measurement-validity.md`](../../../../.claude/rules/measurement-validity.md), the metric must match the decision (routing correctness, not proxy AUROC on an unrelated test set) and be scored on the production path, not a hand-built bench.
