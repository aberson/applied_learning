# Sonnet-Window Revisit Plan — applied_learning

**Status:** Closed — no revisit needed
**Created:** 2026-06-16

## Finding

A `latest`-channel CLI auto-update (~2026-06-05) silently ran interactive sessions on Sonnet
instead of Opus from ~Jun 5 to ~Jun 11. One Sonnet interactive session touched this project
on **2026-06-07** (project setup / pyproject exploration).

**However, applied_learning has NO commits in the June 4–12 window.** Git history shows only:

- `90b61ca` 2026-06-01 — Initial commit: scaffold
- `8275e3e` 2026-06-01 — Add diffusion-models + model-distillation topics
- `a759017` 2026-06-15 — KD under distribution shift (Opus-era, after the fix)

The Jun 7 Sonnet session produced no committed work in the affected window. The Jun 15 commit
landed after the model was re-pinned to Opus.

## Verdict

**No revisit required.** Nothing authored by Sonnet was committed to this repo. If any
uncommitted Jun 7 scratch work exists in the working tree, discard or re-derive it under Opus.

Root cause + permanent fix recorded in `dev/CLAUDE.md` (Environment section) and memory
`feedback_model_pin_opus_autoupdate_reset`.
