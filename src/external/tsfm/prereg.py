"""The `external_tsfm` family's pre-registration (design D7; spec "The family is
registered before it is scored").

    python -m src.external.tsfm.prereg --write

writes results/external_tsfm/PRE_REGISTRATION.md and .json after the replay.
They fix, before any forward row is scored:
  * the 4 cells, {Chronos-2, TimesFM 2.5} x {gold, EUR/USD}, next-hour size;
  * each cell's frozen configuration and the digests of every file it loads;
  * the baselines: gold's frozen h1_garch_hod, and the frozen EUR/USD
    GARCH(1,1) x hour-of-day, with their digests;
  * alpha = 0.05 / 4 = 0.0125, two-sided block bootstrap (block 24, 2,000
    resamples, seed 42) of (baseline |error| - model |error|) on paired,
    eligible, registered rows; KEEP when the lower bound is above 0;
  * N = 1,500 rows per cell, or the cap 2027-04-30; fewer than 500 at the cap is
    UNDERPOWERED -- NO DECISION;
  * the power statement.
Direction is logged and shown but never registered. The family never touches
the EUR/USD 0.05/9 bar or any gold registry.

The phase rule is the gold family's: rows are `registered` only once BOTH files
are committed and unmodified (src.forecast_eval.prereg.committed_and_clean);
until then every row is `dry_run`. Once committed, --write refuses to run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np

from . import CLEAN_START, INSTRUMENTS, MODELS, PREREG_JSON, PREREG_MD, REPLAY, REPO, repo_path
from . import baseline as B
from . import finetune as FT
from . import sources as S
from . import vendor as V

FAMILY = 'external_tsfm'
ALPHA = 0.05 / 4
BLOCK = 24
N_BOOT = 2000
SEED = 42
N_TARGET = 1500
N_MIN = 500
CAP = '2027-04-30'
EDGE_POWER = 0.80
LABEL = 'external model — research, no proven edge'


class AlreadyRegistered(RuntimeError):
    pass


def _z(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def power_statement(repo: str = REPO, replay: dict | None = None) -> dict:
    """Detectable edge at N for 80 % power, from gold h1_lstm's own forward-
    comparable edge over h1_garch_hod (the CI-implied SE, scaled by sqrt(n)).
    EUR/USD's moves are smaller, so its SE is scaled by the ratio of the two
    instruments' mean |return| in the replay window."""
    ref = S.gold_h1_lstm_volatility_result(repo)
    if not ref:
        return {'available': False}
    z_ref = _z(1 - ref['alpha_bonferroni'] / 2)
    se_ref = (ref['ci_high'] - ref['ci_low']) / (2 * z_ref)
    se_n = se_ref * np.sqrt(ref['n_rows'] / N_TARGET)
    mde = (_z(1 - ALPHA / 2) + _z(EDGE_POWER)) * se_n
    out = {'available': True, 'reference': f'gold h1_lstm vs h1_garch_hod, test block ({S.GOLD_TEST_REPORT})',
           'reference_edge_pp': ref['point_delta'], 'reference_n': ref['n_rows'], 'reference_se_pp': se_ref,
           'gold': {'se_at_n_pp': se_n, 'detectable_edge_pp': mde}}
    try:
        inst = (replay or {}).get('instruments', {})
        g = inst['gold']['comparators']['baseline']
        e = inst['eurusd']['comparators']['baseline']
        ratio = (e['first_half_mae'] + e['second_half_mae']) / (g['first_half_mae'] + g['second_half_mae'])
        out['eurusd'] = {'scale_vs_gold': ratio, 'se_at_n_pp': se_n * ratio, 'detectable_edge_pp': mde * ratio}
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    weeks = N_TARGET / 110
    out['weeks_to_n'] = round(weeks, 1)
    out['statement'] = (
        f'At N = {N_TARGET} the standard error of the paired MAE difference is about {se_n:.4f} pp for gold, so '
        f'an edge of {mde:.4f} pp is detected with {EDGE_POWER:.0%} power at alpha {ALPHA:.4f}. Gold h1_lstm\'s own '
        f'edge over h1_garch_hod was {ref["point_delta"]:.4f} pp: an external edge of that size would be '
        f'borderline. About 110 eligible hours a week means N arrives in roughly {weeks:.0f} weeks.')
    return out


def _gold_baseline_digest(repo: str) -> str:
    entries = S.gold_cell_files('h1_garch_hod', repo)
    if not entries:
        raise V.ArtifactMismatch('h1_garch_hod is not in the gold artifact manifest')
    return V.artifact_digest(entries)


def cells(repo: str = REPO, replay: dict | None = None) -> list[dict]:
    choice = (replay or {}).get('choice', {})
    out = []
    for model in MODELS:
        for inst in INSTRUMENTS:
            key = f'{model}_{inst}'
            cid = (choice.get(key) or {}).get('chosen')
            if not cid:
                raise ValueError(f'the replay chose no configuration for {key}: run the replay first')
            files = {model: V.verify(model, repo=repo)}
            if ':fine_tuned:' in cid:
                files[FT.adapter_artifact(model, inst)] = V.verify(FT.adapter_artifact(model, inst), repo=repo)
            base = ({'name': 'h1_garch_hod', 'source': 'gold family, frozen registered cell',
                     'digest': _gold_baseline_digest(repo)} if inst == 'gold' else
                    {'name': B.ARTIFACT, 'source': f'GARCH(1,1) x hour-of-day, fitted {B.FIT_START} .. before {CLEAN_START}',
                     'digest': V.verify(B.ARTIFACT, repo=repo)})
            out.append({'cell': f'{FAMILY}:{key}', 'model': model, 'instrument': inst, 'target': 'next-hour |return| %',
                        'score': 'MAE of size_pct vs realised |return|', 'config_id': cid, 'artifacts': files,
                        'baseline': base})
    return out


def build(repo: str = REPO) -> dict:
    with open(repo_path(f'{REPLAY}/replay.json', repo), encoding='utf-8') as fh:
        replay = json.load(fh)
    return {'family': FAMILY, 'written_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'cells': cells(repo, replay), 'alpha': ALPHA, 'family_size': 4,
            'test': {'statistic': 'mean of |baseline - realised| - |size - realised| on paired rows',
                     'bootstrap': {'block_len': BLOCK, 'n_boot': N_BOOT, 'seed': SEED, 'two_sided': True},
                     'keep_rule': f'lower bound of the {1 - ALPHA:.2%} interval > 0',
                     'rows': 'phase registered, eligible, model and baseline both present, in as-of order'},
            'n_target': N_TARGET, 'n_min': N_MIN, 'cap': CAP, 'clean_start': CLEAN_START,
            'direction': 'logged and shown, never registered, never judged',
            'untouched': ['EUR/USD 0.05/9 feature bar', 'gold registries', 'the horizon study record'],
            'replay': {'path': f'{REPLAY}/replay.json', 'label': replay.get('label')},
            'power': power_statement(repo, replay), 'label': LABEL}


def render(reg: dict) -> str:
    lines = ['# external_tsfm — pre-registration', '',
             'Written by `python -m src.external.tsfm.prereg --write`. Rows are `registered` only once this file and '
             '`PRE_REGISTRATION.json` are committed unmodified; every earlier row is `dry_run` and is never scored.', '',
             '## Cells', '', '| cell | configuration | baseline | baseline digest |', '|---|---|---|---|']
    for c in reg['cells']:
        lines.append(f'| `{c["cell"]}` | `{c["config_id"]}` | {c["baseline"]["name"]} | `{c["baseline"]["digest"]}` |')
    t = reg['test']
    lines += ['', '## Test', '',
              f'- Target: the next H1 bar\'s absolute log return in %; score: mean absolute error of `size_pct`.',
              f'- Statistic: {t["statistic"]}.',
              f'- Two-sided block bootstrap: block {t["bootstrap"]["block_len"]}, {t["bootstrap"]["n_boot"]} '
              f'resamples, seed {t["bootstrap"]["seed"]}.',
              f'- alpha = 0.05 / {reg["family_size"]} = {reg["alpha"]:.4f}. KEEP when the {t["keep_rule"]}; DROP otherwise.',
              f'- Each cell is scored ONCE: at its first {reg["n_target"]} rows, or at the cap {reg["cap"]} with what '
              f'it has. Fewer than {reg["n_min"]} rows at the cap: UNDERPOWERED — NO DECISION.',
              f'- Rows: {t["rows"]}.',
              f'- Direction: {reg["direction"]}.',
              f'- This family never changes: {", ".join(reg["untouched"])}.', '',
              '## Power', '', reg['power'].get('statement', 'reference unavailable'), '',
              '## Frozen artifacts', '']
    for c in reg['cells']:
        lines.append(f'- `{c["cell"]}`: ' + ', '.join(f'{k} `{v}`' for k, v in c['artifacts'].items()))
    lines += ['', f'Replay (descriptive, chose the configurations): `{reg["replay"]["path"]}`.', '',
              f'Label on every view until a verdict exists: "{reg["label"]}".', '']
    return '\n'.join(lines)


def tracked(path: str, repo: str = REPO) -> bool:
    rel = os.path.relpath(path, repo).replace('\\', '/')
    out = subprocess.run(['git', '-C', repo, 'log', '-1', '--format=%H', '--', rel], capture_output=True, text=True)
    return out.returncode == 0 and bool(out.stdout.strip())


def write(repo: str = REPO) -> dict:
    md, js = repo_path(PREREG_MD, repo), repo_path(PREREG_JSON, repo)
    if tracked(md, repo) or tracked(js, repo):
        raise AlreadyRegistered('the pre-registration is already committed; a change would alter registered '
                                'objects and needs a new registration')
    reg = build(repo)
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(js, 'w', encoding='utf-8') as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    with open(md, 'w', encoding='utf-8') as fh:
        fh.write(render(reg))
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write the external_tsfm pre-registration.')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args(argv)
    if a.write:
        reg = write()
        print(json.dumps({'cells': [(c['cell'], c['config_id']) for c in reg['cells']],
                          'power': reg['power'].get('statement')}, indent=1))


if __name__ == '__main__':
    main()
