# WildFrame docs

Contributor-facing reference material. This is additive to [`CLAUDE.md`](../CLAUDE.md), which
remains the source of truth for architecture, invariants, and the pipeline data flow — these
docs go one level deeper into things CLAUDE.md only summarizes.

- [`rules.md`](rules.md) — what each composition rule checks, its gate conditions, and its config knobs.
- [`glossary.md`](glossary.md) — shared vocabulary from `contracts.py` (`SubjectMode`, `RuleState`, `Subject`/`Member`/`SubjectSet`).
- [`calibration.md`](calibration.md) — the `# UNCALIBRATED` convention and how each threshold is meant to get a real value.
- [`study.md`](study.md) — status of the planned user-study harness (`study/`).

Related material that lives outside `docs/`:

- [`../plans/PLAN_multi_animal.md`](../plans/PLAN_multi_animal.md) — the implementation plan for single-animal → multi-animal support; background for why `SubjectSet`/`scene.py` are shaped the way they are.
- [`../paper/`](../paper/) — research write-up: lit review, introduction draft, deck, figures.
- [`../notebooks/`](../notebooks/) — calibration and quantization/optimization experiment notes (see `notebooks/README.md`).
