# Progress — diffusion models

Self-tracking checklist. Tick as you go; the order matches the suggested path in the [README](../README.md).

## 10% coursework
- [ ] Read [essentials.md](../10-coursework/essentials.md); can state the closed form for `q(x_t|x_0)` from memory
- [ ] Skimmed [papers-seminal.md](../10-coursework/papers-seminal.md); read the DDPM paper intro
- [ ] Skimmed [related-topics.md](../10-coursework/related-topics.md); know how score matching relates to eps-prediction

## 70% hands-on
- [ ] Ran [01-forward-noising-intuition](../70-handson/notebooks/01-forward-noising-intuition.ipynb)
- [ ] Ran [02-ddpm-from-scratch-2d](../70-handson/notebooks/02-ddpm-from-scratch-2d.ipynb); understand `L_simple`
- [ ] Ran [03-ddpm-mnist](../70-handson/notebooks/03-ddpm-mnist.ipynb)
- [ ] Ran [04-ddim-and-guidance](../70-handson/notebooks/04-ddim-and-guidance.ipynb); understand the CFG formula
- [ ] Ran the standalone [ddpm_min.py](../70-handson/examples/ddpm_min.py)
- [ ] Exercises — Warm-up done ([exercises.md](../70-handson/exercises.md))
- [ ] Exercises — Core done
- [ ] Exercises — at least one Stretch done

## 20% mentoring
- [ ] Answered the self-checks in [open-questions.md](open-questions.md) out loud
- [ ] Picked one idea from [seed-ideas.md](seed-ideas.md) to prototype in a real project
- [ ] Wrote down one "deeper thread" question to discuss with a mentor / Claude

## Milestone
- [ ] Can explain to someone else: why predict noise, why DDIM is faster, and what the guidance scale trades off.
