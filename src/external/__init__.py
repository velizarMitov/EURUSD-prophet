"""Evaluation harnesses for EXTERNAL, third-party models.

Nothing in this package is part of the serving path for this project's own
models. `src/inference.py` may expose an external model behind its own
readiness gate, but no existing family, ledger or hypothesis log depends on
anything here, and every dependency in this package is optional.
"""
