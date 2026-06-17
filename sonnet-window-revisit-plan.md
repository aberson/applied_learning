# Sonnet-Window Revisit Plan — applied_learning

**Status:** Open — Opus diff re-review pending. **CORRECTED 2026-06-17:** the original "Closed — no revisit needed" verdict was WRONG (window was truncated to Jun 4–12 and `a759017` was mislabeled "Opus-era"). See correction note below.
**Created:** 2026-06-16
**Owner:** operator + Opus session

## Why this exists

A `latest`-channel CLI auto-update (~2026-06-05) silently reset the model from Opus to
Sonnet. The regression ran from ~Jun 5 until **Opus was re-pinned 2026-06-16** — so the true
contamination window is **Jun 4–16**, not Jun 4–12. Root cause + fix: `dev/CLAUDE.md`
Environment section, memory `feedback_model_pin_opus_autoupdate_reset`.

> **Verdict reversal (2026-06-17).** The first version of this plan closed as "no revisit
> needed," reasoning that `a759017` (Jun 15) was "Opus-era, after the fix." That is **false** —
> Opus was not restored until Jun 16, so `a759017` is **Sonnet-authored and in-window**. An
> adversarial re-inventory caught the mislabel. This repo DOES require a revisit.

## Commit inventory (Jun 4–16)

| Commit | Date | What | Re-review focus |
|--------|------|------|-----------------|
| a759017 | 06-15 | docs(model-distillation): KD under distribution shift — 4 files, 803 insertions | **Sonnet-authored ML pedagogy** — new 546-line notebook + edits to 2 notebooks + papers-frontier.md. Check against this repo's hard conventions (runnable on CPU, no fabricated citations, technically sound). |

Pre-window commits (`90b61ca`, `8275e3e`, both Jun 1) are Opus-era and out of scope.

## Shared deliverable

The step appends per-commit verdicts to **`sonnet-window-revisit-findings.md`** (repo root).
Row: `commit | verdict (OK | needs-fix | reverted) | note`.

## Build steps

### Step 1: Re-review a759017 — KD-under-distribution-shift notebooks + frontier doc

- **Problem:** `a759017` (Jun 15, Sonnet) added 803 lines of ML-pedagogy content under Sonnet: a brand-new 546-line notebook plus edits to two existing notebooks and the frontier-papers doc. Sonnet's known failure modes here are fabricated/wrong citations, incorrect distillation / distribution-shift claims, and notebooks that don't execute clean headless on CPU — each a direct violation of this repo's `CLAUDE.md` conventions.
- **Type:** code
- **Issue:** #
- **Files:** topics/model-distillation/70-handson/notebooks/05-distribution-shift-detection.ipynb (NEW, 546 lines), topics/model-distillation/70-handson/notebooks/02-logit-distillation-mnist.ipynb, topics/model-distillation/70-handson/notebooks/04-tradeoffs.ipynb, topics/model-distillation/10-coursework/papers-frontier.md
- **Done when:**
  - `git show a759017` diffed for all 4 files
  - `bash scripts/run_all_notebooks.sh` confirms all touched notebooks execute clean headless on CPU (per CLAUDE.md runnability gate)
  - NO fabricated citations in papers-frontier.md or notebook prose (author+year present; only certain arXiv ids — per CLAUDE.md)
  - distillation / distribution-shift technical claims sanity-checked against primary sources
  - any `.py` examples added are ASCII-only `print()` (Windows cp1252)
  - verdict appended to sonnet-window-revisit-findings.md; fixes applied where needed
- **Flags:** --reviewers code

## Notes

- The original revisit-plan commit (521933a) and this correction were both authored in a
  session that was partly Sonnet — trust this corrected version, not the original no-op reasoning.
- dev / toybox have their own revisit plans; void_furnace is handled by the operator separately.
