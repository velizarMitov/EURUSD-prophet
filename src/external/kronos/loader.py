"""Load Kronos at PINNED HF revisions, for the VOLATILITY channel.

WHAT CHANGED, 2026-08-04, and why the previous configuration was retired
-----------------------------------------------------------------------
This module used to serve ``PRED_LEN = 1`` and a direction probability from
Kronos-base. Both were wrong:

* ``pred_len=1`` takes ONE autoregressive step, so 30 samples are 30 draws from a
  single-step categorical distribution and collapse onto ~7.6 distinct closes out
  of 30. The authors never use it -- their demo uses 24, their README example 120.
  At 24 the endpoint count is 29.9/30.
* DIRECTION IS THE NULL CHANNEL. Measured dead at pred_len=1 (AUC 0.509, an
  out-of-sample isotonic recalibration fails to beat a constant), dead at
  pred_len=24 (AUC 0.509 base / 0.517 mini, and Brier skill got WORSE at -0.62),
  and dead cross-sectionally (RankIC +0.0199 entirely attributable to a one-line
  reversal ranking; orthogonalised CI [-0.00127, +0.01768] includes zero at a
  properly powered MDE of 0.0133). A served number with no information is worse
  than no number, because it looks like information.
* Kronos-MINI (4.1M) beat Kronos-base (102M) on every volatility measure, costs
  ~12.6x less, and the paper's own Forex column puts Kronos-S above Kronos-B.

So: mini, pred_len 24, volatility only. Base stays loadable behind
``which='base'`` for comparison but is not the default.

Loading is LAZY. ``probe()`` answers "could this serve?" without pulling weights
or claiming VRAM, so PredictionService can set kronos_ready at construction.
"""

import json
import os
import threading

from .vendor import (KRONOS_COMMIT, KronosPinMismatch, KronosUnavailable,
                     assert_pinned, import_upstream)

# Immutable HF revisions. HEAD SHAs are pinned (not the weights-adding commits)
# because a full repo snapshot is guaranteed to carry config.json alongside
# model.safetensors. The base weights blob has been unchanged since 2025-06-30,
# which is what makes the paper's June-2024 training cutoff apply to them.
KRONOS_MINI_REPO = 'NeoQuasar/Kronos-mini'
KRONOS_MINI_REVISION = 'f4e68697d9d5aed55cef5c96aabc3376bcad9f81'
KRONOS_TOKENIZER_2K_REPO = 'NeoQuasar/Kronos-Tokenizer-2k'
KRONOS_TOKENIZER_2K_REVISION = '26966d0035065a0cae0ebad7af8ece35bc1fb51c'

KRONOS_BASE_REPO = 'NeoQuasar/Kronos-base'
KRONOS_BASE_REVISION = '2b554741eca47781b64468546e77fef3e85130e6'
KRONOS_TOKENIZER_REPO = 'NeoQuasar/Kronos-Tokenizer-base'
KRONOS_TOKENIZER_REVISION = '0e0117387f39004a9016484a186a908917e22426'
KRONOS_WEIGHTS_UPLOADED = '2025-06-30'

MODELS = {
    'mini': (KRONOS_MINI_REPO, KRONOS_MINI_REVISION,
             KRONOS_TOKENIZER_2K_REPO, KRONOS_TOKENIZER_2K_REVISION),
    'base': (KRONOS_BASE_REPO, KRONOS_BASE_REVISION,
             KRONOS_TOKENIZER_REPO, KRONOS_TOKENIZER_REVISION),
}
DEFAULT_MODEL = 'mini'

# ---------------------------------------------------------------------------
# FROZEN GENERATION PARAMETERS
# ---------------------------------------------------------------------------
# These are the EVALUATION constants. Serving must use exactly these or the
# reported performance describes something the endpoint does not do; a test
# asserts equality. sample_count stays at 30 (the demo's value, and A.2 measured
# Monte-Carlo noise at 8.1% of the observed spread, so 30 paths already resolve
# window-to-window variation).
T = 1.0
TOP_P = 0.9
PRED_LEN = 24
SAMPLE_COUNT = 30
CONTEXT_BARS = 512

# The trailing window the amplification question is asked against. Matches the
# horizon, and matches how the clean-window evaluation defined it.
TRAILING_BARS = 24

TRAINING_CUTOFF = '2024-06'
CLEAN_WINDOW_START = '2024-07-01'
MC_NOISE_ESTIMATE = 0.0817

_CAL_PATH = os.path.join('models', 'external_kronos', 'vol_calibration.json')

CONTAMINATION_NOTE = (
    'Kronos pre-training data extends to June 2024 (paper, arXiv 2508.02739). '
    'The published weights were uploaded 2025-06-30 and are unchanged since, so '
    'only 2024-07-01 onward is out-of-sample. Anything earlier is in-corpus and '
    'no accuracy measured there would mean anything.')

TOKENIZER_NOTE = (
    'The tokenizer normalises per window and reconstructs EURUSD H1 at MAE/range '
    '0.206 vs 0.411 for an in-corpus equity control, so it represents the '
    'instrument well. At pred_len=24 the sampled paths land on 29.9 distinct '
    'endpoints out of 30 -- the single-step collapse that affected pred_len=1 is '
    'not present. Generated dispersion still runs BELOW realised (0.655x for '
    'mini), which is what pred_vol_pct_24h_scaled corrects for.')

WEEKEND_NOTE = (
    'Kronos does not encode elapsed time between bars, so the ~48h weekend break '
    'is a single step to the model. A 24-bar horizon spans a full trading day, so '
    'spans_weekend_gap is frequently true and was reported as a separate stratum '
    'in evaluation.')

DIRECTION_RETIRED_REASON = (
    'direction channel retired -- measured to carry no information; '
    'see /api/kronos-volatility')

DISCLAIMER = (
    'External foundation model, zero-shot, observational. Volatility '
    'discrimination measured once on 2024-07 to 2026-07 (AUC 0.689, n=519). Raw '
    'probabilities are poorly calibrated; the calibrated field applies our own '
    'correction. NOT tested against this project\'s own volatility ensemble. '
    'Simulated only, not a trading instruction.')

# What the single clean-window evaluation actually measured, for Kronos-mini at
# this exact configuration. Served verbatim so the endpoint cannot overstate it.
CLEAN_WINDOW_RESULT = {
    'window': '2024-07-30 to 2026-07-31',
    'n': 519,
    'sampling': 'non-overlapping, stride 24; label uniqueness 1.0 by construction',
    'p_vol_amp': {
        'auc': 0.6889, 'auc_ci': [0.6449, 0.7309], 'n': 519,
        'brier_raw': 0.36294, 'brier_constant': 0.25000, 'ece_raw': 0.36352,
        'brier_calibrated_oos': 0.23042, 'brier_constant_oos': 0.24999, 'n_oos': 260,
        'note': ('raw probability is NOT calibrated (ECE 0.364) -- it discriminates. '
                 'The calibrated field applies our frozen isotonic map, whose '
                 'out-of-sample Brier is the number quoted here.'),
    },
    'continuous_rv': {
        'corr_kronos': 0.4585, 'corr_kronos_ci': [0.309, 0.575],
        'corr_persistence': 0.4422, 'corr_persistence_ci': [0.297, 0.565],
        'corr_garch': 0.3734, 'corr_garch_ci': [0.247, 0.479],
        'paired_diff_vs_persistence': 0.0149,
        'paired_diff_vs_persistence_ci': [-0.0760, 0.0984],
        'incremental_over_persistence': 0.2193,
        'incremental_over_persistence_ci': [0.1356, 0.2935],
        'n': 519,
        'note': ('Kronos does NOT beat a trailing window alone -- the paired CI '
                 'includes zero. It does add information a trailing window does '
                 'not contain. That incremental result was found POST-HOC.'),
    },
    'direction_for_contrast': {
        'auc': 0.5172, 'auc_ci': [0.4672, 0.5680],
        'note': 'why the direction channel was retired',
    },
    'not_tested_against': ('this project\'s own volatility ensemble '
                           '(models/volatility/). That comparison has never been '
                           'made on any row and is what would decide whether this '
                           'is more than observational.'),
}

_lock = threading.Lock()
_cached = {}
_cal_cache = {}


def model_version(which: str = DEFAULT_MODEL) -> str:
    """Pinned upstream commit + both HF revision SHAs, short form."""
    repo, rev, _tok_repo, tok_rev = MODELS[which]
    return 'kronos-%s@%s+tok@%s+src@%s' % (
        which, rev[:10], tok_rev[:10], KRONOS_COMMIT[:10])


def load_calibration(path: str = None) -> dict:
    """The FROZEN scale constant and isotonic knots. Loaded, never refitted --
    fitting at request time would silently re-tune the served numbers against
    whatever data happened to be in the log. Cached per path.

    Raises KronosUnavailable if absent: serving an uncalibrated probability while
    labelling it calibrated would be worse than not serving it.
    """
    p = path or _CAL_PATH
    key = os.path.abspath(p)
    with _lock:
        if key in _cal_cache:
            return _cal_cache[key]
    if not os.path.exists(p):
        raise KronosUnavailable(
            'volatility calibration missing at %s; the serving path loads it and '
            'never refits, so it must be present.' % p)
    with open(p) as fh:
        cal = json.load(fh)
    for k in ('scale', 'isotonic', 'fit_window'):
        if k not in cal:
            raise KronosUnavailable('calibration %s is missing %r' % (p, k))
    # The calibration is only valid for the configuration it was fitted at.
    for field, expected in (('pred_len', PRED_LEN), ('sample_count', SAMPLE_COUNT),
                            ('T', T), ('top_p', TOP_P), ('context_bars', CONTEXT_BARS)):
        if cal.get(field) != expected:
            raise KronosUnavailable(
                'calibration was fitted at %s=%r but serving uses %r; refusing to '
                'apply a mapping from a different configuration.'
                % (field, cal.get(field), expected))
    with _lock:
        _cal_cache[key] = cal
    return cal


def probe() -> tuple:
    """(ready, reason). Cheap: verifies the pin, that torch imports, and that the
    frozen calibration is present -- without downloading weights or touching the
    GPU. Never raises for an absent optional dependency; a PIN MISMATCH does
    raise, because that is a correctness failure rather than a missing-extra."""
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
    try:
        load_calibration()
    except KronosUnavailable as e:
        return False, str(e)
    return True, ''


def load(device: str = None, which: str = DEFAULT_MODEL):
    """Load (and cache) the predictor for `which`. Raises KronosUnavailable if it
    cannot. Cached per model so the base comparison never evicts mini."""
    if which not in MODELS:
        raise KronosUnavailable('unknown Kronos model %r; expected one of %s'
                                % (which, sorted(MODELS)))
    with _lock:
        if which in _cached:
            return _cached[which]
    import_upstream()
    try:
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        raise KronosUnavailable('Kronos dependencies missing: %s' % e)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    repo, rev, tok_repo, tok_rev = MODELS[which]
    try:
        tok = KronosTokenizer.from_pretrained(tok_repo, revision=tok_rev)
        mdl = Kronos.from_pretrained(repo, revision=rev)
    except Exception as e:
        raise KronosUnavailable('Kronos checkpoint unavailable: %s' % e)
    predictor = KronosPredictor(mdl, tok, device=device, max_context=CONTEXT_BARS)
    entry = (predictor, {'device': device, 'model_version': model_version(which),
                         'model': which})
    with _lock:
        _cached[which] = entry
    return entry


def reset_cache():
    """Drop the cached predictors and calibration. Tests only."""
    global _cached, _cal_cache
    with _lock:
        _cached = {}
        _cal_cache = {}
