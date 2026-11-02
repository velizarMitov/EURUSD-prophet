"""tsfm -- external time-series foundation models as a pinned second opinion.

openspec change `tsfm-second-opinion`. Chronos-2 (Amazon) and TimesFM 2.5
(Google), open weights under Apache-2.0, forecast the size of the next H1 move
for gold and EUR/USD. They are challengers with their own pre-registered
family (`external_tsfm`, 4 cells at alpha 0.05/4), judged only on forward data.
They are never an input to, or an ensemble with, the in-house models.

ISOLATION CONTRACT
  * The models run in a SEPARATE environment (.venv-tsfm, requirements-tsfm.txt)
    inside a worker subprocess. The app's .venv and requirements.txt never
    change, and nothing on the production path (src/inference.py,
    _train_pipeline.py, src/paper_trading.py, src/features.py) imports this
    package. api.py may import only `view`, which is read-only.
  * Writes only under results/external_tsfm/ and the gitignored
    research_models/external_tsfm/. Reads the gold and EUR/USD loggers' files,
    never writes them, and never writes under models/.
  * Nothing before CLEAN_START is scored: both models were published in autumn
    2025, and every earlier bar is in the fine-tuning by design.
  * Gross of cost, like the gold family: no spread, no pips, no P&L.

Layout (design D1-D10):
  vendor          pinned revisions, the SHA-256 manifest, verify()
  worker          SIDE environment only: series in, deciles out; fine-tuning
  client          main-environment caller of the worker, with the time limit
  forecast        deciles -> size (median |R|) and P(up)
  data            the frozen pre-window history and replay window
  finetune        --freeze the history, --train the LoRA adapters
  baseline        the frozen EUR/USD GARCH(1,1) x hour-of-day
  replay          the clean-window configuration choice (descriptive)
  forward_logger  hourly forecasts, gaps, settlement, health
  prereg          the registration and the phase rule
  forward_report  the one-shot score per cell
  view            read-only payload for /api/second-opinion
"""

from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CLEAN_START = '2025-11-01'
MODELS = ('chronos2', 'timesfm25')
INSTRUMENTS = ('gold', 'eurusd')
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

RESULTS = 'results/external_tsfm'
ARTIFACTS = 'research_models/external_tsfm'
MANIFEST = f'{RESULTS}/artifact_manifest.csv'
FORWARD = f'{RESULTS}/forward'
REPLAY = f'{RESULTS}/replay'
FINETUNE = f'{RESULTS}/finetune'
PREREG_MD = f'{RESULTS}/PRE_REGISTRATION.md'
PREREG_JSON = f'{RESULTS}/PRE_REGISTRATION.json'
VERDICTS = f'{RESULTS}/verdicts.json'
VENV_PYTHON = '.venv-tsfm/Scripts/python.exe'


def repo_path(rel: str, repo: str = REPO) -> str:
    """Absolute path for a repo-relative path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(repo, *rel.split('/'))
