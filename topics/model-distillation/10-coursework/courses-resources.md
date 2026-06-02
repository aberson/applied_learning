# Courses & Resources — Model Distillation (10% coursework)

Curated list of real, verifiable resources. URLs omitted where not certain of exact path; title + org is enough to find them.

---

## Surveys

**"Distilling the Knowledge in a Neural Network"** — Hinton, Vinyals, Dean (arXiv:1503.02531, 2015)
The paper that named and formalized KD. Introduces temperature-scaled soft targets and the student-teacher framing. Read first; everything else builds on this vocabulary.

**"Knowledge Distillation: A Survey"** — Gou et al. (arXiv:2006.05525, 2021)
Comprehensive taxonomy: response-based, feature-based, and relation-based distillation; offline vs online vs self-distillation; application domains. Use as a map when you need to locate a technique by name.

---

## Blogs & tutorials

**Keras.io — "Knowledge Distillation"** (keras.io/examples/vision/knowledge_distillation/)
End-to-end Keras/TF example: trains a small student on MNIST using a teacher's soft logits. Minimal code; good first implementation to read before writing your own.

**Hugging Face blog — DistilBERT and task-specific distillation**
Covers how DistilBERT was trained (triple loss: soft cross-entropy + hard cross-entropy + cosine embedding), and how to fine-tune distilled models for downstream tasks. Find via huggingface.co/blog. Use this when working with transformer distillation or the `transformers` library.

**Hugging Face docs — Optimum / distillation pipeline**
`optimum` library exposes `DistillationTrainer`; docs walk through a BERT-to-student distillation with config. Practical starting point for production-grade NLP distillation without writing the loss from scratch.

---

## Code / hands-on

**PyTorch tutorial — "Knowledge Distillation Tutorial"** (pytorch.org/tutorials)
Teaches temperature scaling and soft-target loss in raw PyTorch. Walks through a ResNet teacher and a smaller student on image classification. Use this when you need to understand the loss implementation before using a higher-level library.

**"Awesome Knowledge Distillation"** — GitHub curated list (search: "Awesome Knowledge Distillation" on GitHub)
Community-maintained index of papers, code repos, and frameworks organized by method type and modality. Use as a discovery layer once you know what sub-topic you want (e.g., attention transfer, data-free KD, NLP-specific).

**Hugging Face `transformers` — DistilBERT model card and training script**
`huggingface/transformers` repo includes the original DistilBERT training script (`examples/research_projects/distillation/`). Reading the actual training loop (token-level soft-label loss + MLM) is more instructive than any tutorial. Use when studying how distillation is applied at pre-training scale.

---

## Talks / lectures

**Geoffrey Hinton — "Dark Knowledge" talk, NeurIPS 2014 Deep Learning Workshop**
The talk that introduced "dark knowledge" framing: what soft probability distributions over wrong classes reveal about the teacher's learned representation. Available via recorded video (search "Hinton dark knowledge NeurIPS 2014"). Watch before reading the 2015 paper; the intuition lands better in spoken form.

**Andrej Karpathy — Neural network lectures (cs231n / karpathy.github.io)**
Not distillation-specific, but the sections on softmax temperature and logit interpretation provide the necessary background. Use if temperature scaling feels opaque after the Hinton paper.

---

## Sequence recommendation

1. Hinton et al. 1503.02531 (paper) + "Dark Knowledge" talk (intuition)
2. PyTorch or Keras tutorial (first implementation)
3. Gou et al. survey (orientation)
4. HF DistilBERT blog + training script (transformer application)
5. Awesome KD list (discovery for next sub-topic)
