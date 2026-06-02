# Model distillation

A hands-on ramp on knowledge distillation — training a small "student" to replicate a large "teacher" — following the **70 / 20 / 10** applied-learning split.

## Suggested learning path

1. **Skim the essentials** — [10-coursework/essentials.md](10-coursework/essentials.md) (soft targets, temperature, the distillation loss, the taxonomy). ~15 min.
2. **Work the notebooks in order** (the 70%):
   1. [01-soft-targets-intuition](70-handson/notebooks/01-soft-targets-intuition.ipynb) — *numpy only*; temperature softening, "dark knowledge", and what KL-on-soft-targets optimizes.
   2. [02-logit-distillation-mnist](70-handson/notebooks/02-logit-distillation-mnist.ipynb) — the headline result: a ~8x smaller distilled student beats the same student trained from scratch; temperature sweep.
   3. [03-feature-distillation-fitnets](70-handson/notebooks/03-feature-distillation-fitnets.ipynb) — hint/feature matching (FitNets) and when it helps over logit-only.
   4. [04-tradeoffs](70-handson/notebooks/04-tradeoffs.ipynb) — accuracy vs params vs latency across student sizes; scratch vs distilled curves.
3. **Do the exercises** — [70-handson/exercises.md](70-handson/exercises.md) (T/alpha sweeps, capacity-gap effect, latency measurement).
4. **Read the mentoring notes** — [20-mentoring/seed-ideas.md](20-mentoring/seed-ideas.md) (distillation in your real projects — goblin's grader, the toybox kiosk, the Alpha4Gate advisor) and [20-mentoring/open-questions.md](20-mentoring/open-questions.md).
5. **Go deeper** — [10-coursework/papers-seminal.md](10-coursework/papers-seminal.md) (start with Hinton et al. 2015), [papers-frontier.md](10-coursework/papers-frontier.md), [related-topics.md](10-coursework/related-topics.md), [courses-resources.md](10-coursework/courses-resources.md).

Track yourself in [20-mentoring/progress.md](20-mentoring/progress.md).

## Layout (70 / 20 / 10)

```text
model-distillation/
├── 10-coursework/   essentials, papers-seminal, papers-frontier, related-topics, courses-resources
├── 20-mentoring/    seed-ideas, open-questions, progress
└── 70-handson/      notebooks/ (01-04)  examples/distill_min.py  exercises.md
```

## Running

```bash
uv sync                                              # from the applied_learning repo root
uv run jupyter lab                                   # then open 70-handson/notebooks/
uv run python topics/model-distillation/70-handson/examples/distill_min.py   # standalone: load_digits teacher->student
```

Every notebook is committed already-executed (graphics embedded) and runs headless on CPU. An honest teaching note carried in the notebooks: distillation's win over from-scratch training **depends on the regime** — it shows up clearly when the teacher generalizes well and the student is genuinely under-capacity / data-limited (notebook 02 sets this up deliberately and explains why). On a too-easy task a tiny student already saturates from hard labels alone (the `distill_min.py` near-tie documents this).

## What next

From the goblin repo: `uv run goblin suggest applied_learning` ranks next-learning-steps grounded in this folder.
