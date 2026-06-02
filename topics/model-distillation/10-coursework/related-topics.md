# Related Topics — Model Distillation Adjacency Map

Neighboring concepts a distillation learner should know. Each entry notes what the topic is and how it relates to distillation (complementary, alternative, prerequisite, or easily confused).

See [essentials](essentials.md) for core distillation mechanics.

---

## Model Compression (Umbrella)

**Model compression** is the parent category containing distillation, quantization, pruning, and factorization. Distillation is one compression strategy; the others below are siblings. In practice, techniques are combined: distill first to a smaller architecture, then quantize or prune that student further.

---

## Quantization

**Quantization** reduces the bit-width of weights and activations (e.g., float32 → int8 or int4). It compresses a fixed architecture without changing its depth or width. Complements distillation: quantization-aware training can be run on a student model post-distillation, or a quantized teacher can guide a student (though quantization degrades soft targets slightly).

---

## Pruning (Structured and Unstructured)

**Pruning** zeros out or removes weights. Unstructured pruning removes individual weights (sparse tensors, hardware-unfriendly). Structured pruning removes entire heads, channels, or layers (hardware-friendly). Both shrink a trained model; distillation instead trains a smaller model from scratch using the teacher's outputs. The two compose naturally: prune a large model, then distill its knowledge into the pruned result.

---

## Low-Rank Factorization and LoRA

**Low-rank factorization** decomposes a weight matrix W into two smaller matrices (e.g., UV^T). **LoRA** (Hu et al.) applies low-rank updates during fine-tuning, keeping the base model frozen. Both reduce parameter count or update cost. LoRA is an alternative to full fine-tuning and can be applied to students post-distillation; factorization is an alternative compression route to distillation.

---

## Neural Architecture Search (NAS)

**NAS** automates finding efficient architectures given a compute budget. Distillation and NAS often co-occur: NAS finds the student architecture, distillation trains it. Some NAS methods use a teacher's soft targets directly as training signal. NAS is a prerequisite to distillation when the student architecture is not hand-designed.

---

## Label Smoothing and Its Link to Soft Targets

**Label smoothing** replaces hard one-hot labels with a smoothed distribution (e.g., 0.9 correct class, 0.1/K elsewhere). It acts as a regularizer. Hinton et al. noted that distillation soft targets contain richer information than label smoothing — they encode relative class similarities, not just a uniform prior. Understanding label smoothing clarifies *why* soft targets help beyond mere regularization.

---

## Ensembles and Ensemble Distillation

**Ensembles** combine predictions from multiple independently trained models, improving accuracy and calibration at high inference cost. **Ensemble distillation** (Born Again Networks, Furlanello et al.) trains a single student to match the ensemble's output distribution, recovering much of the accuracy gain at single-model inference cost. Distillation is the canonical way to make ensembles practical.

---

## Transfer Learning and Fine-Tuning

**Transfer learning** initializes a model on a large upstream task, then fine-tunes on a downstream task. Distillation is complementary: the upstream pretrained model is often the teacher; the fine-tuned or compressed version is the student. Task-specific distillation (e.g., DistilBERT) combines both — pretrain a large teacher, then distill into a smaller student initialized from the teacher's weights.

---

## Mixture of Experts (MoE)

**MoE** models route each input to a sparse subset of expert sub-networks, scaling parameter count without scaling compute proportionally. MoE models are increasingly common teachers (e.g., Mixtral). Distilling an MoE teacher into a dense student is an active research area; the teacher's routing logits provide extra training signal beyond class probabilities.

---

## Speculative Decoding

**Speculative decoding** pairs a small draft model with a large verifier model at inference time: the draft proposes tokens, the large model accepts or rejects them. The small and large models must be from the same distribution family. Unlike distillation (which trains the student offline), speculative decoding uses both models jointly at runtime. The draft model can itself be a distilled student.

---

## Data-Free Distillation

**Data-free distillation** synthesizes training data for the student using only the teacher (e.g., inverting batch-norm statistics, or prompting a generative teacher). Used when the original training set is proprietary or unavailable. Contrasts with standard distillation, which assumes access to the original or a proxy dataset.

---

## Dataset Distillation

**Dataset distillation** (Wang et al.) compresses the *training dataset* into a small set of synthetic examples that produce nearly the same trained model as the full dataset. This is a *different problem from model distillation* — it compresses the data, not the model. The name overlap causes frequent confusion. Dataset distillation is used for continual learning, privacy, and NAS inner loops; model distillation is used for inference-time efficiency.

---

## Temperature Scaling and Calibration

**Temperature scaling** (Guo et al.) divides logits by a scalar T before softmax to sharpen or soften the output distribution. In distillation, high temperature softens the teacher's outputs to expose dark-knowledge class similarities. Calibration is the broader goal of aligning predicted probabilities with empirical frequencies; temperature scaling is the simplest calibration method. A well-calibrated teacher produces more informative soft targets.

---

## Student-Teacher Capacity Gap and Teacher Assistants

When the student is much smaller than the teacher, directly imitating the teacher degrades student performance — the teacher's distribution is too complex for the student to fit. **Teacher assistants** (Mirzadeh et al.) insert intermediate-capacity models in a distillation chain: teacher → assistant → student. Each step is a smaller capacity gap, improving the final student. Relevant when designing multi-stage compression pipelines.
