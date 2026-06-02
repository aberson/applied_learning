# Model Distillation — Core Concepts

## Core idea

A large, accurate **teacher** network supervises a smaller **student** network. Instead of training the student on one-hot hard labels, you train it to match the teacher's output distribution. A student trained this way consistently outperforms the same architecture trained on hard labels alone.

Why? Hard labels are maximally sparse — one class is right, every other class is equally wrong. The teacher's softmax output is dense: it assigns non-trivial probability to visually or semantically similar classes. That relative structure is extra signal the hard-label student never sees.

## Soft targets and temperature

Standard softmax: `p_i = exp(z_i) / sum_j exp(z_j)`

Temperature-scaled softmax: `p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)`

- `T = 1`: normal softmax (same as inference time)
- `T > 1`: distribution flattens; small logits become relatively larger
- `T >> 1`: approaches a uniform distribution

**Dark knowledge** (Hinton et al. 2015): the non-zero probability mass the teacher places on wrong classes. Example: a model trained on MNIST might assign cat images 0.001 probability to "dog" and 0.0001 to "car". That ratio encodes learned similarity structure invisible in a one-hot label.

Higher temperature amplifies these small probabilities, making the dark-knowledge gradient strong enough to influence training. At `T = 1` the signal from wrong classes is numerically negligible.

## Distillation loss

```
L = alpha * T^2 * KL( softmax(z_teacher/T) || softmax(z_student/T) )
  + (1 - alpha) * CE( y_hard, softmax(z_student) )
```

**Terms:**

- `alpha` — weight on soft-target (distillation) loss; `(1 - alpha)` weights hard-label cross-entropy
- `T^2` factor — when you differentiate `KL(softmax(z/T) || ...)` with respect to student logits `z`, the gradients scale as `1/T^2`. Multiplying the loss by `T^2` restores gradient magnitudes to the same order as the `T = 1` case, so the balance between soft and hard losses does not shift as you vary T.
- The hard-label term is sometimes dropped entirely (`alpha = 1`) when no ground-truth labels are available (e.g., distilling dark data).

**Alpha trade-off:** high alpha trusts the teacher's soft structure; low alpha keeps the student anchored to ground truth. In practice `alpha = 0.9, T = 4` is a common starting point; tune on a dev set.

## Taxonomy

- **Response/logit-based** (Hinton et al. 2015): match final output logits or softmax distributions. Simplest; requires no architecture alignment.
- **Feature-based** (Romero et al. 2015, FitNets): match intermediate feature maps via a hint layer. Transfers internal representations; student and teacher layers must be aligned or projected.
- **Relation-based** (Park et al. 2019, RKD; Tian et al. 2019, CRD): match pairwise or higher-order relationships between examples in embedding space, not individual activations. More architecture-agnostic than feature-based.

## Variants

- **Self-distillation**: a network distills into itself or a shallower version of itself; no external teacher needed.
- **Born-again networks** (Furlanello et al. 2018): train a student with the same architecture as the teacher, then iterate. Gains come purely from the softer training signal.
- **Online / mutual distillation** (Zhang et al. 2018, DML): two or more students train simultaneously, each using the other as a peer teacher. No pretrained teacher needed.
- **Ensemble distillation**: compress an ensemble of models into a single student, recovering most of the ensemble's accuracy at single-model cost.
- **Teacher-assistant distillation** (Mirzadeh et al. 2020): when teacher and student capacity gap is large, insert an intermediate-sized teacher-assistant; large gaps cause distillation to underperform.

## Why it works (intuition)

1. **Richer training signal**: a soft distribution over C classes carries more bits of information per example than a one-hot vector.
2. **Regularization**: soft targets act like label smoothing — the student is penalized for being overconfident, reducing overfitting.
3. **Inductive bias transfer**: the teacher's generalization behavior (what it thinks is similar to what) is encoded in its output distribution and propagates to the student.
4. **Relationship to label smoothing**: label smoothing replaces `y_i = 1` with `y_i = 1 - eps` and spreads `eps` uniformly. Teacher soft targets are a data-driven, non-uniform version of the same idea — the smoothing reflects actual learned structure rather than a flat prior.

## When to use

- **Model compression**: deploy a small model without retraining from scratch on hard labels.
- **On-device / low-latency inference**: fit within memory and latency budgets while preserving accuracy.
- **Cheaper LLM serving**: distill a large language model into a smaller one for the same task distribution.
- **Ensemble-to-single**: recover ensemble accuracy in a single forward pass.

## Math prerequisites

- Softmax and temperature scaling
- Cross-entropy loss `CE(y, p) = -sum_i y_i log p_i`
- KL divergence `KL(p || q) = sum_i p_i log(p_i / q_i)`
- Gradient of softmax with respect to logits (to understand the `T^2` correction)
