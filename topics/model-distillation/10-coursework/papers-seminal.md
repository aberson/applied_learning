# Seminal Papers — Model / Knowledge Distillation

**Start here: Hinton et al. 2015 (1503.02531)**

Suggested reading order: Bucilua 2006 → Hinton 2015 → Ba & Caruana 2014 → FitNets 2014 → Attention Transfer 2016 → Born-Again Networks 2018 → CRD 2019 → RKD 2019 → DistilBERT 2019 → Beyer et al. 2021.

---

## Logit-Based Distillation

**Model Compression** — Bucilua, Caruana, Niculescu-Mizil. KDD 2006.
The original idea: train a single compact model to mimic the output of a large ensemble, recovering most of the ensemble's accuracy at a fraction of the inference cost. Established the core paradigm that Hinton et al. later formalized.

**[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)** — Hinton, Vinyals, Dean. 2015.
Introduced soft targets and a temperature parameter to expose the full predictive distribution of a large teacher; the soft targets carry inter-class similarity information that hard labels discard. The foundational reference for the field.

**[Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)** — Ba & Caruana. 2014.
Showed that a shallow net trained on soft targets from a deep teacher can match the deep net's accuracy, even though the shallow net trained on hard labels cannot. Demonstrated that depth is partly a training artifact that distillation can circumvent.

---

## Feature- and Relation-Based Distillation

**[FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)** — Romero, Ballas, Kahou, Chassang, Gatta, Bengio. 2014.
Extended distillation beyond the output layer by supervising intermediate student activations ("hints") against teacher hidden representations, enabling training of networks that are simultaneously thinner and deeper than the teacher.

**[Paying More Attention to Attention](https://arxiv.org/abs/1612.03928)** — Zagoruyko & Komodakis. 2016.
Proposed transferring spatial attention maps from teacher to student rather than raw activations, yielding a compact and hardware-friendly supervision signal that consistently improves student accuracy across architectures.

**[Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699)** — Tian, Krishnan, Isola. 2019.
Reframed distillation as maximizing mutual information between teacher and student representations via contrastive learning, outperforming prior feature-matching methods especially across heterogeneous architectures.

**[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)** — Park, Kim, Lu, Cho. 2019.
Instead of transferring individual activation values, transfers structural relationships among examples (pairwise distances, triplet angles) in the teacher's embedding space, capturing relational geometry that pointwise methods miss.

---

## Applications and Training Dynamics

**[Born-Again Neural Networks](https://arxiv.org/abs/1805.04770)** — Furlanello, Lipton, Tschannen, Itti, Anandkumar. 2018.
Showed that a student trained to imitate a teacher of identical architecture can exceed the teacher's accuracy, and that ensembling successive generations further compounds gains. Established self-distillation and generational training as practical techniques.

**[DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)** — Sanh, Debut, Chaumond, Wolf. 2019.
Applied distillation to a large pretrained Transformer, producing a model 40% smaller and 60% faster that retains 97% of BERT's GLUE performance. The canonical demonstration of distillation for NLP and the blueprint for compressing large language models.

**[Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)** — Beyer, Zhai, Royer, Markeeva, Anil, Kolesnikov. 2021.
Identified that standard distillation recipes underperform because teacher and student see different augmentations; enforcing consistent views and training longer ("patient") closes the gap substantially. A practical corrective to common experimental pitfalls in distillation research.
