# Progress — model distillation

Self-tracking checklist. Tick as you go; the order matches the suggested path in the [README](../README.md).

## 10% coursework
- [ ] Read [essentials.md](../10-coursework/essentials.md); can write the distillation loss (KL + CE, T, alpha) from memory
- [ ] Skimmed [papers-seminal.md](../10-coursework/papers-seminal.md); read the Hinton et al. 2015 paper
- [ ] Skimmed [related-topics.md](../10-coursework/related-topics.md); know how distillation differs from quantization/pruning and from *dataset* distillation

## 70% hands-on
- [ ] Ran [01-soft-targets-intuition](../70-handson/notebooks/01-soft-targets-intuition.ipynb); can explain dark knowledge and the T^2 factor
- [ ] Ran [02-logit-distillation-mnist](../70-handson/notebooks/02-logit-distillation-mnist.ipynb); understand WHY distilled beat scratch here (the regime)
- [ ] Ran [03-feature-distillation-fitnets](../70-handson/notebooks/03-feature-distillation-fitnets.ipynb); understand the dimension-matching regressor
- [ ] Ran [04-tradeoffs](../70-handson/notebooks/04-tradeoffs.ipynb); read the accuracy-vs-params and accuracy-vs-latency plots
- [ ] Ran the standalone [distill_min.py](../70-handson/examples/distill_min.py)
- [ ] Exercises — Warm-up done ([exercises.md](../70-handson/exercises.md))
- [ ] Exercises — Core done (incl. scratch-vs-distilled at fixed size)
- [ ] Exercises — at least one Stretch done

## 20% mentoring
- [ ] Answered the self-checks in [open-questions.md](open-questions.md) out loud
- [ ] Picked one idea from [seed-ideas.md](seed-ideas.md) to prototype (goblin grader distillation is the highest-leverage)
- [ ] Wrote down one "deeper thread" question to discuss with a mentor / Claude

## Milestone
- [ ] Can explain to someone else: why soft targets beat hard labels, when distillation helps vs doesn't, and the three flavors (logit / feature / relation).
