"""The one-shot score per cell (design D7; spec "The family is registered
before it is scored").

    python -m src.external.tsfm.forward_report --score

Refuses before registration. A cell is scored ONCE -- at its first N paired,
eligible, registered rows in as-of order, or at the cap with what it has -- and
its verdict is APPENDED to results/external_tsfm/verdicts.json. A scored cell
is never re-scored, so later rows can never change it.

    KEEP          lower bound of the (1 - alpha) block-bootstrap interval of
                  mean(|baseline error| - |model error|) is above 0
    DROP          otherwise
    UNDERPOWERED  fewer than N_MIN rows at the cap: "UNDERPOWERED — NO DECISION"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import PREREG_JSON, REPO, VERDICTS, repo_path
from . import forward_logger as L

UNDERPOWERED = 'UNDERPOWERED — NO DECISION'


class NotRegistered(RuntimeError):
    pass


def read_verdicts(repo: str = REPO) -> list[dict]:
    path = repo_path(VERDICTS, repo)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def paired_rows(settlements: list[dict], model: str, instrument: str) -> pd.DataFrame:
    df = pd.DataFrame([r for r in settlements if r['model'] == model and r['instrument'] == instrument
                       and r.get('phase') == 'registered' and str(r.get('eligible')) == 'True'])
    if df.empty:
        return df
    for c in ('size_pct', 'baseline_size_pct', 'realised_abs_pct'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['size_pct', 'baseline_size_pct', 'realised_abs_pct'])
    df = df.drop_duplicates('key')
    df['t'] = pd.to_datetime(df['as_of_bar'], utc=True)
    return df.sort_values('t').reset_index(drop=True)


def score_cell(rows: pd.DataFrame, reg: dict, today) -> dict | None:
    """The verdict, or None when the cell is not due (fewer than N rows, cap not reached)."""
    from src.forecast_eval.stats import block_bootstrap_ci
    n_target, n_min = int(reg['n_target']), int(reg['n_min'])
    capped = pd.Timestamp(today).normalize() >= pd.Timestamp(reg['cap'], tz='UTC')
    if len(rows) >= n_target:
        use = rows.iloc[:n_target]
    elif capped:
        use = rows[rows['t'] < pd.Timestamp(reg['cap'], tz='UTC') + pd.Timedelta(days=1)]
    else:
        return None
    n = len(use)
    out = {'n': n, 'alpha': reg['alpha'], 'scored_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'reached': 'N' if len(rows) >= n_target else 'cap'}
    if n < n_min:
        return {**out, 'verdict': UNDERPOWERED, 'ci_low': None, 'ci_high': None, 'point': None}
    y = use['realised_abs_pct'].to_numpy()
    d = np.abs(use['baseline_size_pct'].to_numpy() - y) - np.abs(use['size_pct'].to_numpy() - y)
    b = reg['test']['bootstrap']
    lo, hi, point = block_bootstrap_ci(d, b['block_len'], reg['alpha'], b['n_boot'], b['seed'])
    return {**out, 'verdict': 'KEEP' if lo > 0 else 'DROP', 'ci_low': lo, 'ci_high': hi, 'point': point,
            'model_mae': float(np.mean(np.abs(use['size_pct'].to_numpy() - y))),
            'baseline_mae': float(np.mean(np.abs(use['baseline_size_pct'].to_numpy() - y))),
            'first_as_of': str(use['t'].iloc[0]), 'last_as_of': str(use['t'].iloc[-1])}


def score(repo: str = REPO, today=None, phase_fn=L.phase) -> list[dict]:
    if phase_fn(repo) != 'registered':
        raise NotRegistered('the pre-registration is not committed and unmodified: nothing is scored')
    with open(repo_path(PREREG_JSON, repo), encoding='utf-8') as fh:
        reg = json.load(fh)
    today = today or pd.Timestamp.now(tz='UTC')
    done = {v['cell'] for v in read_verdicts(repo)}
    setts = L.read_rows(L.out_path('settlements', repo))
    new = []
    for c in reg['cells']:
        if c['cell'] in done:
            continue
        v = score_cell(paired_rows(setts, c['model'], c['instrument']), reg, today)
        if v is not None:
            new.append({'cell': c['cell'], 'config_id': c['config_id'], **v})
    if new:
        path = repo_path(VERDICTS, repo)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(read_verdicts(repo) + new, fh, indent=1)
        os.replace(tmp, path)
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description='Score the external_tsfm cells once each.')
    ap.add_argument('--score', action='store_true')
    a = ap.parse_args(argv)
    if a.score:
        try:
            print(json.dumps(score(), indent=1))
        except NotRegistered as e:
            print(json.dumps({'refused': str(e)}))


if __name__ == '__main__':
    main()
