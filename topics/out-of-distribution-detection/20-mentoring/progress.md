# Progress — out-of-distribution detection

Self-tracking checklist. Tick as you go; the order matches the suggested path in the [README](../README.md).

## 10% coursework
- [ ] Read [essentials.md](../10-coursework/essentials.md); can state the score-and-threshold framing and define AUROC + FPR@95TPR from memory
- [ ] Skimmed [papers-seminal.md](../10-coursework/papers-seminal.md); read the MSP paper (Hendrycks & Gimpel 2017) intro
- [ ] Skimmed [papers-frontier.md](../10-coursework/papers-frontier.md) and [related-topics.md](../10-coursework/related-topics.md); know how OOD relates to open-set recognition and selective prediction

## 70% hands-on
- [ ] Ran [01-ood-intuition-2d](../70-handson/notebooks/01-ood-intuition-2d.ipynb); can say why softmax confidence is NOT an ID indicator
- [ ] Ran [02-msp-baseline-mnist](../70-handson/notebooks/02-msp-baseline-mnist.ipynb); understand MSP, AUROC, and FPR@95TPR
- [ ] Ran [03-odin-and-energy](../70-handson/notebooks/03-odin-and-energy.ipynb); understand the energy score and why ODIN's T must match the net's logit scale
- [ ] Ran [04-mahalanobis-and-knn](../70-handson/notebooks/04-mahalanobis-and-knn.ipynb); understand scoring in feature space vs logit space
- [ ] Ran [05-ensembles-uncertainty](../70-handson/notebooks/05-ensembles-uncertainty.ipynb); understand entropy vs mutual information (aleatoric vs epistemic)
- [ ] Ran the standalone [ood_min.py](../70-handson/examples/ood_min.py)
- [ ] Exercises — Warm-up done ([exercises.md](../70-handson/exercises.md))
- [ ] Exercises — Core done
- [ ] Exercises — at least one Stretch done (near-OOD, outlier exposure, or label-free thresholding)

## 20% mentoring
- [ ] Answered the self-checks in [open-questions.md](open-questions.md) out loud
- [ ] Picked one idea from [seed-ideas.md](seed-ideas.md) to prototype (start: switchboard's energy-defer)
- [ ] Wrote down one "deeper thread" question to discuss with a mentor / Claude

## Milestone
- [ ] Can explain to someone else: why softmax is overconfident on OOD, when a feature-distance score beats a logit score, and how you'd set a threshold with no labeled OOD data.
