# Model Distillation — Open Questions

Discussion prompts for mentoring sessions. Use these to check understanding and surface gaps before moving to experiments.

---

## Can you explain it? (self-check)

1. **Why soft targets instead of hard labels?** Hard labels assign all probability mass to one class. What information would you lose, and why does a teacher's probability distribution carry more signal per example?

2. **What is "dark knowledge"?** The teacher assigns small but non-zero probability to wrong classes. What does that distribution reveal about class relationships that the training label cannot?

3. **Why the T² factor in the loss?** When you scale logits by 1/T before softmax, the gradient magnitudes shrink. Where does T² come from, and why do Hinton et al. multiply the distillation loss term by T²?

4. **What does α trade off?** The combined loss is α · L_distill + (1−α) · L_CE. What happens at α=0 and α=1? When would you prefer a high α, and when a low one?

5. **Why might a very large teacher distill *worse* into a tiny student?** The teacher's decision boundary may be too complex for the student's capacity to approximate. What other failure modes arise when the capacity gap is extreme?

6. **Feature distillation vs. logit distillation — when does feature distillation win?** Logit distillation only supervises the final layer. In what network types or task regimes does matching intermediate representations outperform logit matching?

7. **Temperature T → ∞ approaches label smoothing. What does T → 1 recover?** Trace through the softmax formula and state what the soft targets look like at each extreme.

8. **Can a student ever outperform its teacher, and if so, how?** Under what data-augmentation or ensemble-distillation conditions has this been observed? What does it imply about the teacher's role?

---

## Deeper threads

1. **Is distillation just fancy label smoothing?** Both soften the target distribution. State the formal difference. When would the two produce identical gradients, and when would they diverge most?

2. **Online vs. offline distillation trade-offs.** Offline: teacher is fixed. Online: teacher and student train jointly (e.g., mutual learning). What does each buy you in terms of compute, stability, and final accuracy? When is online distillation impractical?

3. **Distilling a sequence model / LLM vs. a classifier.** Classifiers have a single softmax; autoregressive models have T successive softmaxes. How does exposure bias interact with off-policy teacher logits? What changes if you use on-policy student rollouts for supervision?

4. **Data-free distillation.** When the original training set is private or too large to store, can you reconstruct a usable dataset from the teacher alone? What are the practical limits — mode collapse, label imbalance, distribution shift?

5. **How do you pick a teacher?** Accuracy alone is not a sufficient criterion. What other properties — calibration, confidence gap, architecture family — affect how much knowledge transfers?

6. **Measuring dark knowledge transfer.** After distillation, how would you quantify how much inter-class structure the student actually absorbed? What proxy metrics exist, and what are their blind spots?

---

## Things to try next

1. **Implement temperature sweep.** In the [essentials notebook](../10-coursework/essentials.md), train a student at T ∈ {1, 2, 5, 10, 20} and plot validation accuracy vs. T. Find the optimum and explain the shape of the curve.

2. **Compare logit vs. feature distillation on a small CNN.** Use an intermediate feature map from the teacher as an additional supervision signal. Measure whether accuracy, calibration (ECE), or both improve.

3. **Read the DistilBERT paper.** Note which distillation losses they combine and why they add a cosine-embedding loss on hidden states. Compare their α choices to the image-classification defaults.

4. **Try self-distillation.** Train a single network, then distill it into itself (same architecture, random re-init). Does it improve over the baseline? What does this reveal about the role of teacher capacity?
