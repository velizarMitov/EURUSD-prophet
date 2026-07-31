"""Load Kronos-base + its tokenizer at PINNED HF revisions.

Loading is LAZY. `probe()` answers "could this serve?" without pulling 400MB of
weights or claiming ~1.3GB of VRAM, so PredictionService can set kronos_ready at
construction without every server start paying for a 102M-parameter model that
most requests never touch. The weights load on the first prediction and are
cached for the process lifetime.
"""

import os
import threading

from .vendor import (KRONOS_COMMIT, KronosPinMismatch, KronosUnavailable,
                     assert_pinned, import_upstream)

# Immutable HF revisions. HEAD SHAs are pinned (not the weights-adding commits)
# because a full repo snapshot is guaranteed to carry config.json alongside
# model.safetensors; the weights blob itself has been unchanged since
# 31afc02ed0a3cfaa1a97238c8b948d663e45b597 (base) and
# 9ef143b98ee3c2488eebd85404e0c215c112b46a (tokenizer), both 2025-06-30 -- which
# is what makes the paper's June-2024 training cutoff apply to these weights.
KRONOS_BASE_REPO = 'NeoQuasar/Kronos-base'
KRONOS_BASE_REVISION = '2b554741eca47781b64468546e77fef3e85130e6'
KRONOS_TOKENIZER_REPO = 'NeoQuasar/Kronos-Tokenizer-base'
KRONOS_TOKENIZER_REVISION = '0e0117387f39004a9016484a186a908917e22426'
KRONOS_WEIGHTS_UPLOADED = '2025-06-30'

# FROZEN GENERATION PARAMETERS (Phase A.4), declared before any accuracy was
# seen. sample_count fixed at 30 by A.2: Monte-Carlo variance was 8.1% of the
# observed p_up spread, so 30 paths already resolve the model's window-to-window
# variation. A test asserts these are unchanged -- with a sampler, quietly
# sweeping them against the evaluation data is unusually easy and is exactly the
# data-snooping this project's methodology exists to prevent.
T = 1.0
TOP_P = 0.9
PRED_LEN = 1
SAMPLE_COUNT = 30
CONTEXT_BARS = 512

TRAINING_CUTOFF = '2024-06'
CLEAN_WINDOW_START = '2024-07-01'
# A.2, measured: mean |p_up(seed A) - p_up(seed B)| over 200 windows at 30 paths.
MC_NOISE_ESTIMATE = 0.0817

CONTAMINATION_NOTE = (
    'Kronos pre-training data extends to June 2024 (paper, arXiv 2508.02739). '
    'The published weights were uploaded 2025-06-30 and are unchanged since, so '
    'only 2024-07-01 onward is out-of-sample. Anything earlier is in-corpus and '
    'no accuracy measured there would mean anything.')

TOKENIZER_NOTE = (
    'The tokenizer normalises per window and reconstructs EURUSD H1 at '
    'MAE/range 0.206 vs 0.411 for an in-corpus equity control, so it represents '
    'the instrument well. It occupies only 6.5% of level-1 codes vs 18.1% for '
    'that control, so it operates at reduced resolution. Sampled paths land on '
    'roughly 6 distinct closes out of 30, and generated one-step dispersion is '
    'about 0.64x trailing realised volatility -- the paths are too tight, which '
    'inflates apparent confidence.')

WEEKEND_NOTE = (
    'Kronos does not encode elapsed time between bars, so the ~48h weekend break '
    'is a single step to the model. spans_weekend_gap flags those windows.')

DISCLAIMER = (
    'External foundation model, zero-shot. Evaluated once on 2024-07 to 2026-07, '
    'the only period after its June-2024 training cutoff. Observational, '
    'simulated only, not a trading instruction.')

_lock = threading.Lock()
_cached = None


def model_version() -> str:
    """Pinned upstream commit + both HF revision SHAs, short form."""
    return 'kronos-base@%s+tok@%s+src@%s' % (
        KRONOS_BASE_REVISION[:10], KRONOS_TOKENIZER_REVISION[:10], KRONOS_COMMIT[:10])


def probe() -> tuple:
    """(ready, reason). Cheap: verifies the pin and that torch imports, without
    downloading weights or touching the GPU. Never raises for an absent optional
    dependency; a PIN MISMATCH does raise, because that is a correctness failure
    rather than a missing-extra."""
    try:
        assert_pinned()
    except KronosPinMismatch:
        raise
    except KronosUnavailable as e:
        return False, str(e)
    try:
        import torch                                    # noqa: F401
    except ImportError as e:
        return False, 'torch not installed (%s); pip install -r requirements-kronos.txt' % e
    try:
        import_upstream()
    except KronosUnavailable as e:
        return False, str(e)
    return True, ''


def load(device: str = None):
    """Load (and cache) the predictor. Raises KronosUnavailable if it cannot."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        import_upstream()
        try:
            import torch
            from model import Kronos, KronosTokenizer, KronosPredictor
        except ImportError as e:
            raise KronosUnavailable('Kronos dependencies missing: %s' % e)
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        try:
            tok = KronosTokenizer.from_pretrained(
                KRONOS_TOKENIZER_REPO, revision=KRONOS_TOKENIZER_REVISION)
            mdl = Kronos.from_pretrained(
                KRONOS_BASE_REPO, revision=KRONOS_BASE_REVISION)
        except Exception as e:
            raise KronosUnavailable('Kronos checkpoint unavailable: %s' % e)
        predictor = KronosPredictor(mdl, tok, device=device, max_context=CONTEXT_BARS)
        _cached = (predictor, {'device': device, 'model_version': model_version()})
        return _cached


def reset_cache():
    """Drop the cached predictor. Tests only."""
    global _cached
    with _lock:
        _cached = None
