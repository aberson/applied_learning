# applied_learning — Plan

## Goal

A personal applied-learning workspace where each topic gets a self-contained, hands-on ramp following the **70% hands-on / 20% mentoring / 10% coursework** method. Topic folders are produced by the global `/user-learn` skill; this repo stores the output and tracks which topics exist.

## Method (70 / 20 / 10)

- **70% hands-on** — runnable notebooks with graphics, standalone code examples, exercises. The bulk of learning.
- **20% mentoring** — concrete applications in my real projects, open questions to work through, a progress tracker.
- **10% coursework** — a compact knowledge base: essentials, seminal + frontier papers, related-topics map, courses.

## Topic-folder structure

```text
topics/<slug>/
├── README.md
├── 10-coursework/   essentials.md, papers-seminal.md, papers-frontier.md, related-topics.md, courses-resources.md
├── 20-mentoring/    seed-ideas.md, open-questions.md, progress.md
└── 70-handson/      notebooks/  examples/  exercises.md
```

## Build order

### Step 1: Topic — diffusion models

Populate `topics/diffusion-models/` via `/user-learn diffusion-models` (maximal depth): full knowledge base, runnable notebooks (forward noising intuition, DDPM from scratch on a 2D toy, DDPM on an MNIST subset, DDIM + classifier-free guidance), standalone example scripts, exercises, and project-grounded seed ideas. Every notebook executes clean headless on CPU. **Status:** DONE.

### Step 2: Topic — model distillation

Populate `topics/model-distillation/` via `/user-learn model-distillation` (maximal depth): full knowledge base, runnable notebooks (soft-targets intuition, logit distillation on MNIST, feature/FitNets distillation, size/accuracy/latency tradeoffs), standalone example scripts, exercises, and project-grounded seed ideas. Every notebook executes clean headless on CPU. **Status:** DONE.

### Step 3: Topic — out-of-distribution detection

Populate `topics/out-of-distribution-detection/` via `/user-learn out-of-distribution-detection` (maximal depth): full knowledge base, runnable notebooks (2D softmax-overconfidence intuition, MSP baseline on MNIST vs FashionMNIST, ODIN + energy scores, feature-space Mahalanobis + KNN, deep-ensemble/MC-dropout uncertainty), standalone `ood_min.py`, exercises, and project-grounded seed ideas (switchboard/void_furnace/Alpha4Gate routing gates). Every notebook executes clean headless on CPU. **Status:** DONE (2026-07-14).

### Step 4 (future): Goblin learning-companion

Build out `b2_project_goblin` support for driving learning over time — spaced-repetition prompts, quiz generation from a topic's knowledge base, next-exercise nudges, and progress-aware suggestions. Seeded at `b2_project_goblin/seeds/phase-g-learning-companion.md`. **Not in scope for the initial build** — goblin v1 is mid-UAT, so this stays a seed until a deliberate phase.

## Future topics

Anything else I want to learn: `/user-learn <topic>` adds a new self-contained folder under `topics/`.
