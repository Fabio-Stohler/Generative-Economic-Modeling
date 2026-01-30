# BM Model Cleanup Plan

Goal: make the Brock–Mirman core clear, side-effect light, and easy to reuse from scripts/notebooks/Colab.

Planned refactors (no code changes yet):
- **Separate concerns:** keep `BM.py` focused on model equations, shocks, simulation. Move dataset assembly/surrogate prep into a dedicated module later (e.g., `data_pipeline.py`).
- **Device/dtype hygiene:** avoid implicit device moves inside methods; ensure `Parameters`, `State`, `Shocks` accept `device` and `dtype` explicitly and keep them consistent.
- **Shock handling:** centralize AR(1) update math; document lognormal vs. normal treatment; remove duplicated logic between `step`, `simulate`, and helpers.
- **Dataset generation:** have `simulate_dataset` return a structured object (x, y, meta) with optional seed; validate and fail fast instead of retry loops for NaN detection.
- **Config, not globals:** move flags like `Force_CPU`, `retrain`, `rerun`, `truncate_length` into a config/CLI layer outside the model.
- **Naming/shape clarity:** standardize names (`Kp` vs `Kprime`), enforce (batch, features) shapes with small asserts, and document expected tensor layouts.
- **I/O boundaries:** keep pickle save/load minimal; document file formats; avoid training/plotting concerns inside model code.
- **Tests (later):** smoke tests for `simulate` (shape + NaN checks) and deterministic AR(1) step.

Next action after approval: start by carving out dataset/surrogate orchestration from `BM.py` into a new module without altering behavior; add small docstrings/comments for any clarified interfaces.
