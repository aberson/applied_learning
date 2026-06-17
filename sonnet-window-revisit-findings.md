# Sonnet-Window Revisit — Findings

**Repo:** applied_learning
**Plan:** [sonnet-window-revisit-plan.md](sonnet-window-revisit-plan.md) · **Issue:** #4
**Reviewed:** 2026-06-17 (Opus 4.8 session; model re-pin confirmed before run)
**Method:** `/build-phase --plan sonnet-window-revisit-plan.md` (single step, `--reviewers code`).
Two independent adversarial reviewers (citation lens + technical-claims lens) verified every
correction against primary sources.

## Per-commit verdict

| Commit | Verdict | Note |
|--------|---------|------|
| a759017 | **needs-fix → fixed** | Sonnet-authored (`Co-Authored-By: Claude Sonnet 4.6`). 2 blocking SyntaxErrors + 2 wrong citation venues + 1 fabricated-by-confusion statistic. All fixed; runnability gate now passes, all citations verified, technical claims sound. |

## What a759017 changed

803 insertions across 4 files: NEW notebook `05-distribution-shift-detection.ipynb` (546 lines),
extensions to `02-logit-distillation-mnist.ipynb` (Step 5) and `04-tradeoffs.ipynb` (Step 6),
and a "Mechanism transfer and failure modes" section in `papers-frontier.md`.

The commit message itself confirms the contamination premise and the root cause of the
defects: *"Notebooks written without executed outputs; run scripts/run_all_notebooks.sh to
populate."* Sonnet never executed the notebooks, so syntax errors shipped undetected.

---

## Defects found and fixed

### 1. BLOCKING — two notebooks could not execute (SyntaxError)

Both new cells contained **literal newline characters inside single-quoted string literals**
where a `\n` escape was intended — an `unterminated string literal` SyntaxError. The
notebooks fail at import/execute time; this directly violates this repo's #1 convention
("Notebooks must run headless on CPU in seconds — a notebook that won't execute clean
doesn't ship").

| File | Cell | Broken statements |
|------|------|-------------------|
| `02-logit-distillation-mnist.ipynb` | Step 5 (cell 26) | `ax2.set_xlabel('…T=4⏎(lower…')`, `ax2.set_title('…accuracy⏎(more…')`, `print('⏎Best accuracy…')` |
| `04-tradeoffs.ipynb` | Step 6 (cell 24) | `print('⏎Note: WAKD…')` |

**Fix:** replaced each inner literal newline with the `\n` escape sequence (4 replacements
total). Every code cell now passes `compile()`, and `nbconvert --execute --inplace` exits 0
for all three touched notebooks (outputs + embedded graphics repopulated).

### 2. Citation venue/title errors (papers-frontier.md + nb05)

| Citation | Original (wrong) | Corrected |
|----------|------------------|-----------|
| Wu et al. | "**Mechanism Distillation**", "**ICML 2024**" | "**What Mechanisms Does Knowledge Distillation Distill?**", Wu/Lubana/Mlodozeniec/Kirk/Krueger, **UniReps Workshop (NeurIPS 2023), PMLR v243** |
| arXiv 2505.15442 | "(2025, **ACL**)", "replicated and extended" | "(Ramesh et al., 2025, **preprint**)", reframed as a *related* paradox in **LLM reasoning** distillation (not Stanton's CIFAR-100 image setting) |

The `proceedings.mlr.press/v243/wu24a` URL was correct — only the human-readable title and
venue were fabricated. The "ICML 2024" venue was also repeated in nb05's takeaways
("Jacobian matching (Wu et al., ICML 2024)") and corrected there to "UniReps/NeurIPS 2023".

### 3. Fabricated-by-confusion statistic (nb05 "Evidence" paragraph)

Original prose: *"teacher worst-group accuracy 94.6 % vs. student 66.1 % on CelebA."*
Against **Table 1 of arXiv 2506.02294** (verified by both reviewers from the PDF):

- 94.6 % is the teacher's **group-mean** accuracy, mislabeled here as **worst-group**.
- The teacher's true **worst-group** accuracy is **66.7 %**.
- 66.1 % is the worst-group result of the paper's **proposed ConfiG method**, not a plain
  distilled student. A standard distilled-student baseline is **38.9 %** worst-group.

**Fix:** corrected to "a robust CLIP teacher retains 66.7 % worst-group while a standard
distilled student collapses to 38.9 %", and flipped the causal/spurious feature labels to the
canonical CelebA setup (target = hair colour, spurious = gender). The qualitative claim
("student degrades more than teacher under spurious-correlation shift") is sound and now rests
on the correct figures.

---

## Verified sound — no change required

- **All 6 citations now resolve to real, on-topic papers** (Stanton NeurIPS 2021;
  arXiv 2506.02294 Popp et al.; arXiv 2309.11446 Berezovskiy & Morozov; arXiv 2505.10822
  Haskins & Adams; Wu et al. UniReps; arXiv 2505.15442 Ramesh et al.). No remaining
  fabrications (confirmed by the independent citation reviewer).
- **WAKD specific figures (+0.8 pp ResNet-50→ResNet-18, +1.6 pp DeiT-Small→DeiT-Tiny on
  PACS/OfficeHome)** — independently confirmed verbatim against §4.2 of arXiv 2309.11446.
  Left unchanged.
- **Technical/scientific claims — SOUND.** KD math correct everywhere (KL direction
  `KL(teacher‖student)`, T² gradient rescaling, α-blend); Color-MNIST spurious-correlation
  construction genuinely produces the claimed shift; per-group teacher-student **agreement**
  is a legitimate **label-free** shift-detection signal; the fidelity-generalisation paradox
  cell honestly reports when the effect fails to surface on MNIST ("try CIFAR-100"); WAKD
  checkpoint averaging is implemented correctly with honest IID-MNIST expectation-setting.
- **ASCII-only `print()`** — PASS. Zero non-ASCII characters in any code-cell source across
  all three notebooks (Windows cp1252 safe).

## Noted, non-blocking

- **nb05 effect size is modest and seed-fragile.** In the executed run the student's
  Group-A→B gap (0.221) is only ~1.7 pp wider than the teacher's (0.204). The general-mechanism
  prose ("fails *more severely*") is directionally correct and supported by the corrected
  CelebA evidence, and the notebook's own "Where to go next" already prescribes raising the
  correlation probability to amplify it. Left as-is; a one-line magnitude hedge would be a
  reasonable future polish but is not a correctness defect.
- **38.9 % label nuance.** Table 1's 38.9 % worst-group row is technically the *CutMix*
  distilled baseline (EDRM-alone is 15.2 %); "standard distilled student" is faithful for the
  pedagogical contrast but slightly less precise than "CutMix-distilled baseline".

## Runnability gate

`uv run jupyter nbconvert --to notebook --execute --inplace` (scoped to the 3 touched
notebooks — the repo `scripts/run_all_notebooks.sh` runs every topic inplace, which would
dirty unrelated notebooks including a concurrent uncommitted `diffusion-models/03-ddpm-mnist`
change kept out of scope):

| Notebook | Before fix | After fix |
|----------|-----------|-----------|
| 05-distribution-shift-detection | PASS (exit 0) | PASS (exit 0) |
| 02-logit-distillation-mnist | **FAIL — SyntaxError L55** | PASS (exit 0) |
| 04-tradeoffs | **FAIL — SyntaxError L84** | PASS (exit 0) |

## Files changed by this revisit

- `topics/model-distillation/70-handson/notebooks/02-logit-distillation-mnist.ipynb` (syntax fix + executed)
- `topics/model-distillation/70-handson/notebooks/04-tradeoffs.ipynb` (syntax fix + executed)
- `topics/model-distillation/70-handson/notebooks/05-distribution-shift-detection.ipynb` (CelebA numbers + Wu venue fix + executed)
- `topics/model-distillation/10-coursework/papers-frontier.md` (Wu title/venue + 2505.15442 venue fixes)
