# Diffusion models

A hands-on ramp on diffusion models — generative models that learn to reverse a gradual noising process — following the **70 / 20 / 10** applied-learning split.

## Suggested learning path

1. **Skim the essentials** — [10-coursework/essentials.md](10-coursework/essentials.md) (forward/reverse process, the closed form, the noise-prediction objective). ~15 min.
2. **Work the notebooks in order** (the 70%):
   1. [01-forward-noising-intuition](70-handson/notebooks/01-forward-noising-intuition.ipynb) — *numpy only*; watch a 2D cloud melt into Gaussian noise via the closed form `x_t = sqrt(abar)*x_0 + sqrt(1-abar)*eps`.
   2. [02-ddpm-from-scratch-2d](70-handson/notebooks/02-ddpm-from-scratch-2d.ipynb) — train a tiny DDPM on a 2D distribution; real-vs-generated scatter + reverse trajectory.
   3. [03-ddpm-mnist](70-handson/notebooks/03-ddpm-mnist.ipynb) — the same idea on images (tiny UNet, 16x16 MNIST); sample a digit grid.
   4. [04-ddim-and-guidance](70-handson/notebooks/04-ddim-and-guidance.ipynb) — DDIM fast deterministic sampling + classifier-free guidance.
3. **Do the exercises** — [70-handson/exercises.md](70-handson/exercises.md) (schedule swaps, step-count sweeps, label conditioning).
4. **Read the mentoring notes** — [20-mentoring/seed-ideas.md](20-mentoring/seed-ideas.md) (how to use diffusion in your real projects) and [20-mentoring/open-questions.md](20-mentoring/open-questions.md) (self-checks + deeper threads).
5. **Go deeper** — [10-coursework/papers-seminal.md](10-coursework/papers-seminal.md) (start with DDPM), [papers-frontier.md](10-coursework/papers-frontier.md), [related-topics.md](10-coursework/related-topics.md), [courses-resources.md](10-coursework/courses-resources.md).

Track yourself in [20-mentoring/progress.md](20-mentoring/progress.md).

## Layout (70 / 20 / 10)

```text
diffusion-models/
├── 10-coursework/   essentials, papers-seminal, papers-frontier, related-topics, courses-resources
├── 20-mentoring/    seed-ideas, open-questions, progress
└── 70-handson/      notebooks/ (01-04)  examples/ddpm_min.py  exercises.md
```

## Running

```bash
uv sync                                              # from the applied_learning repo root
uv run jupyter lab                                   # then open 70-handson/notebooks/
uv run python topics/diffusion-models/70-handson/examples/ddpm_min.py   # standalone 2D DDPM -> samples.png
```

Every notebook is committed already-executed (graphics embedded). All four run headless on CPU in well under a minute each. Models are deliberately tiny — samples are recognizable, not photorealistic; the point is the mechanism.

## What next

From the goblin repo: `uv run goblin suggest applied_learning` ranks next-learning-steps grounded in this folder.
