# Out-of-distribution (OOD) detection

A hands-on ramp on OOD detection — deciding whether an input is unlike anything a model was
trained on, so the model can abstain, flag, or route it elsewhere instead of confidently guessing
wrong — following the **70 / 20 / 10** applied-learning split.

The through-line: a classifier assigns a class to *every* input, and softmax confidence stays high
even far from the training data. An OOD detector is a scalar **score** `s(x)` plus a **threshold**;
the whole subject is finding a score that separates in-distribution (ID) from OOD, and measuring
that separation with **AUROC** and **FPR@95TPR**.

## Suggested learning path

1. **Skim the essentials** — [10-coursework/essentials.md](10-coursework/essentials.md) (the
   score-and-threshold framing, the five method families, the eval metrics). ~15 min.
2. **Work the notebooks in order** (the 70%) — each trains its own tiny model, so they stand alone;
   they build the method arc MSP -> better post-hoc scores -> feature-space -> uncertainty:
   1. [01-ood-intuition-2d](70-handson/notebooks/01-ood-intuition-2d.ipynb) — *numpy/sklearn only*;
      a 2D confidence heatmap shows softmax staying confident arbitrarily far from the data, next to
      a density score that correctly collapses. The motivating picture.
   2. [02-msp-baseline-mnist](70-handson/notebooks/02-msp-baseline-mnist.ipynb) — the Maximum Softmax
      Probability baseline (Hendrycks & Gimpel 2017) on MNIST (ID) vs FashionMNIST (OOD); score
      histograms, **AUROC + FPR@95TPR**. Deliberately weak — the baseline everything else beats.
   3. [03-odin-and-energy](70-handson/notebooks/03-odin-and-energy.ipynb) — ODIN (temperature +
      input perturbation) and the energy score; both beat MSP on the same split.
   4. [04-mahalanobis-and-knn](70-handson/notebooks/04-mahalanobis-and-knn.ipynb) — score in
      *feature* space instead of on logits: class-conditional Mahalanobis distance and KNN distance.
   5. [05-ensembles-uncertainty](70-handson/notebooks/05-ensembles-uncertainty.ipynb) — a deep
      ensemble; predictive entropy and mutual information (BALD) rise on OOD (epistemic uncertainty).
3. **Run the standalone example** — [70-handson/examples/ood_min.py](70-handson/examples/ood_min.py)
   (a synthetic, no-download MSP + energy scorer -> AUROC + `ood_scores.png`).
4. **Do the exercises** — [70-handson/exercises.md](70-handson/exercises.md) (swap OOD sets, sweep
   ODIN's T and KNN's k, near-OOD, outlier exposure, label-free threshold-setting).
5. **Read the mentoring notes** — [20-mentoring/seed-ideas.md](20-mentoring/seed-ideas.md) (wiring
   an OOD signal into switchboard / void_furnace / Alpha4Gate) and
   [20-mentoring/open-questions.md](20-mentoring/open-questions.md) (self-checks + deeper threads).
6. **Go deeper** — [papers-seminal.md](10-coursework/papers-seminal.md),
   [papers-frontier.md](10-coursework/papers-frontier.md),
   [related-topics.md](10-coursework/related-topics.md),
   [courses-resources.md](10-coursework/courses-resources.md).

Track yourself in [20-mentoring/progress.md](20-mentoring/progress.md).

## Layout (70 / 20 / 10)

```text
out-of-distribution-detection/
├── 10-coursework/   essentials, papers-seminal, papers-frontier, related-topics, courses-resources
├── 20-mentoring/    seed-ideas, open-questions, progress
└── 70-handson/      notebooks/ (01-05)  examples/ood_min.py  exercises.md
```

## Running

```bash
uv sync                                              # from the applied_learning repo root
uv run jupyter lab                                   # then open 70-handson/notebooks/
uv run python topics/out-of-distribution-detection/70-handson/examples/ood_min.py   # -> ood_scores.png
```

Every notebook is committed already-executed (graphics embedded). All five run headless on CPU in
under ~30s each; MNIST (ID) and FashionMNIST/KMNIST (OOD) are pre-downloaded under the repo-root
`data/`. Models are deliberately tiny — the reported AUROCs are illustrative of the *method
ordering*, not leaderboard numbers. A tiny net trained briefly is exactly why the MSP baseline
looks as weak as it does, which is the point.

## What next

From the goblin repo: `uv run goblin suggest applied_learning` ranks next-learning-steps grounded
in this folder.
