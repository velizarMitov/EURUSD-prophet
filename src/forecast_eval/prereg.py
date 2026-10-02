"""Pre-registration, power table, scoring gate and verdicts (tasks 12.1-12.5;
spec forward-arbiter; design D13).

  power_table / fixed_point_family
      Direction cells: time-to-decision for an edge of `edge` over breakeven at
      the family's Bonferroni alpha and 80 % power. Cells over the cap are
      UNDERPOWERED -- NO DECISION by design and leave the alpha count; admission
      is iterated to a fixed point, BEFORE any forward outcome exists.
  volatility_power
      Volatility cells: time-to-decision for a declared relative MAE improvement,
      from the block-bootstrap standard error measured on development data.
  scoring_allowed
      True only when PRE_REGISTRATION.md and registration.json are committed,
      unmodified, and their commit is an ancestor of HEAD.
  verdict
      KEEP / DROP / undecided, only once the registered sample size is reached.
      Before that, figures are labelled "interim, not adjudicating".
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy.stats import norm

from . import stats as ST

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FWD = os.path.join(REPO, 'results', 'forward_eval')
PREREG_MD = os.path.join(FWD, 'PRE_REGISTRATION.md')
PREREG_JSON = os.path.join(FWD, 'registration.json')

BARS_PER_YEAR = {'H1': 6170.0, 'D1': 260.0}
UNDERPOWERED = 'UNDERPOWERED — NO DECISION'
INTERIM = 'interim, not adjudicating'


# ── 12.1 power table and fixed-point family ────────────────────────────────

def z_total(alpha: float, power: float = 0.80) -> float:
    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))


def n_required(alpha: float, edge: float, power: float = 0.80, p: float = 0.5) -> float:
    """Independent forecasts needed to detect `edge` above a rate near p."""
    return (z_total(alpha, power) * np.sqrt(p * (1 - p)) / edge) ** 2


def years_to_decide(cadence: str, h: int, alpha: float, edge: float, coverage: float = 1.0,
                    power: float = 0.80) -> float:
    per_year = BARS_PER_YEAR[cadence] / h * coverage
    return n_required(alpha, edge, power) / per_year


def fixed_point_family(cells, edge=0.03, cap_years=3.0, coverage=1.0, power=0.80, family_alpha=0.05):
    """cells: [(model, cadence, h), ...]. Returns (admitted cells, alpha, table,
    iterations). Start from every cell; admit those decidable within the cap at
    the current family's alpha; repeat until the admitted set is stable."""
    cells = list(cells)
    F, history = len(cells), []
    while True:
        alpha = family_alpha / F
        admitted = [c for c in cells if years_to_decide(c[1], c[2], alpha, edge, coverage, power) <= cap_years]
        history.append((F, alpha, len(admitted)))
        if not admitted or len(admitted) == F:
            break
        F = len(admitted)
    alpha = family_alpha / max(len(admitted), 1)
    table = [{'model': m, 'cadence': c, 'horizon': h,
              'n_required': n_required(alpha, edge, power),
              'n_per_year': BARS_PER_YEAR[c] / h * coverage,
              'years': years_to_decide(c, h, alpha, edge, coverage, power),
              'status': 'ADMITTED' if (m, c, h) in admitted else UNDERPOWERED}
             for (m, c, h) in cells]
    return admitted, alpha, table, history


def direction_cells(record: dict):
    out = []
    for name, cfg in record['challengers'].items():
        if cfg['kind'] == 'direction':
            for h in cfg.get('horizons') or record['grid'][cfg['cadence']]:
                out.append((name, cfg['cadence'], int(h)))
    return out


# ── 12.2 volatility power ───────────────────────────────────────────────────

def volatility_power(n_scored, ci_low, ci_high, mae_baseline, cadence, h, alpha,
                     rel_improvement=0.05, coverage=1.0, power=0.80, ci_alpha=0.05):
    """Rows needed to detect a ΔMAE of rel_improvement * baseline MAE, using the
    per-row standard deviation implied by the development block-bootstrap CI
    (which already carries the autocorrelation): sd = SE * sqrt(n)."""
    se = (ci_high - ci_low) / (2 * norm.ppf(1 - ci_alpha / 2))
    sd = se * np.sqrt(n_scored)
    delta = rel_improvement * mae_baseline
    n_req = (z_total(alpha, power) * sd / delta) ** 2
    per_year = BARS_PER_YEAR[cadence] * coverage          # overlapping rows; sd accounts for overlap
    return {'n_required': float(n_req), 'years': float(n_req / per_year), 'sd_per_row': float(sd),
            'delta': float(delta)}


# ── 12.4 scoring gate ───────────────────────────────────────────────────────

def _git(args, cwd):
    return subprocess.run(['git', '-C', cwd] + args, capture_output=True, text=True)


def committed_and_clean(path: str, repo: str = REPO) -> str | None:
    """The commit that last touched `path`, if it exists, is unmodified in the
    working tree, and is an ancestor of HEAD; else None."""
    if not os.path.exists(path):
        return None
    rel = os.path.relpath(path, repo).replace('\\', '/')
    log = _git(['log', '-1', '--format=%H', '--', rel], repo)
    if log.returncode != 0 or not log.stdout.strip():
        return None
    commit = log.stdout.strip()
    if _git(['diff', '--quiet', 'HEAD', '--', rel], repo).returncode != 0:
        return None
    if _git(['merge-base', '--is-ancestor', commit, 'HEAD'], repo).returncode != 0:
        return None
    return commit


def scoring_allowed(repo: str = REPO, md: str = PREREG_MD, js: str = PREREG_JSON) -> bool:
    return committed_and_clean(md, repo) is not None and committed_and_clean(js, repo) is not None


def registration(js: str = PREREG_JSON) -> dict | None:
    if not os.path.exists(js):
        return None
    with open(js, encoding='utf-8') as fh:
        return json.load(fh)


def registered_server(js: str = PREREG_JSON):
    reg = registration(js)
    return reg.get('server') if reg else None


# ── 12.5 verdicts ───────────────────────────────────────────────────────────

def non_overlapping_rows(as_of_positions, h):
    keep, last = [], -10 ** 9
    for i, p in enumerate(as_of_positions):
        if p >= last + h:
            keep.append(i)
            last = p
    return np.asarray(keep, dtype=int)


def direction_verdict(correct, n_required_: float, breakeven: float, alpha: float, h: int,
                      cadence: str, admitted: bool = True) -> dict:
    """`correct` holds the 0/1 outcomes of NON-OVERLAPPING scorable forecasts,
    in time order."""
    x = np.asarray(correct, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    acc = float(x.mean()) if n else float('nan')
    base = {'n': n, 'n_required': n_required_, 'accuracy': acc, 'breakeven': breakeven}
    if not admitted:
        return {**base, 'verdict': UNDERPOWERED, 'label': UNDERPOWERED}
    if n < n_required_:
        return {**base, 'verdict': None, 'label': INTERIM}
    blen = ST.block_length(h, cadence)
    lo = ST.one_sided_lower_bound(x, blen, alpha)
    hi = ST.one_sided_upper_bound(x, blen, alpha)
    base.update({'lower': lo, 'upper': hi})
    if lo > breakeven:
        return {**base, 'verdict': 'KEEP', 'label': 'cost-viable edge'}
    if hi < breakeven:
        label = 'predictive, not cost-viable' if lo > 0.5 else 'no cost-viable edge'
        return {**base, 'verdict': 'DROP', 'label': label}
    return {**base, 'verdict': 'UNDECIDED', 'label': 'interval contains the breakeven at the registered n'}


def volatility_verdict(err_model, err_baseline, n_required_: float, alpha: float, h: int,
                       cadence: str, admitted: bool = True) -> dict:
    d = np.asarray(err_model, dtype=float) - np.asarray(err_baseline, dtype=float)
    d = d[np.isfinite(d)]
    base = {'n': len(d), 'n_required': n_required_, 'delta_mae': float(d.mean()) if len(d) else float('nan')}
    if not admitted:
        return {**base, 'verdict': UNDERPOWERED, 'label': UNDERPOWERED}
    if len(d) < n_required_:
        return {**base, 'verdict': None, 'label': INTERIM}
    blen = ST.block_length(h, cadence)
    lo, hi = ST.one_sided_lower_bound(d, blen, alpha), ST.one_sided_upper_bound(d, blen, alpha)
    base.update({'lower': lo, 'upper': hi})
    if hi < 0:
        return {**base, 'verdict': 'KEEP', 'label': 'beats GARCH x day-of-week'}
    if lo > 0:
        return {**base, 'verdict': 'DROP', 'label': 'worse than GARCH x day-of-week'}
    return {**base, 'verdict': 'UNDECIDED', 'label': 'interval contains zero at the registered n'}
