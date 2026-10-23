"""Round-two gold validation (openspec change gold-forecasting-v2).

One stage, `validation`, run ONCE, after PRE_REGISTRATION_V2.json is committed
and unmodified (prereg2.check_stage_allowed). The test block is spent and
refused. Every verdict is written with "needs forward confirmation" because the
validation block was already seen in round one.

Families and their scorers:
  range       ranges.score_range_cell on the saved round-one validation
              predictions: no refit, nothing new is fitted
  macro       macro.score_macro_cells: daily models with and without the macro
              block, fitted on [0:70] and scored on identical [70:80] rows
  direction2  direction2.score_direction2_cells: admitted cells only; the
              UNDERPOWERED ones are recorded without a score

Writes the three round-two registries, results/gold/validation_report_v2.md and
appends nothing to any round-one file.

    python -m src.gold.study2 --stage validation
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.forecast_eval import prereg as FP

from . import REPO, load_config, repo_path
from . import prereg as P
from . import prereg2 as P2
from . import study as ST

ARBITER_V2 = 'validation[70:80] round two (seen in round one)'


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def score_ranges(cfg, cfg2, reg2, frames=None) -> tuple[list, list]:
    from . import ranges as R
    fam = reg2['families']['range']
    st = cfg2['study']['range']
    results, failures = [], []
    for cell, spec in st['cells'].items():
        cad = spec['cadence']
        path = os.path.join(repo_path(cfg['paths']['results']), f'validation_predictions_{cad.lower()}.csv')
        try:
            pred = pd.read_csv(path)
            results.append(R.score_range_cell(cell, cad, pred, spec['model'], spec['baseline'], st,
                                              fam['alpha'], cfg['bootstrap']['n_boot']))
        except Exception as e:  # noqa: BLE001 -- a failed cell is reported, never silently dropped
            failures.append({'cell': cell, 'error': f'{type(e).__name__}: {e}'})
    return results, failures


def score_macro(cfg, cfg2, reg2, frames) -> tuple[list, list]:
    from . import macro as M
    return M.score_macro_cells(cfg, cfg2, reg2, frames)


def score_direction2(cfg, cfg2, reg2, frames) -> tuple[list, list]:
    from . import direction2 as D2
    return D2.score_direction2_cells(cfg, cfg2, reg2, frames)


SCORERS = {'range': score_ranges, 'macro': score_macro, 'direction2': score_direction2}


def append_registry(path: str, rows: list[dict]) -> None:
    import csv
    n = sum(1 for _ in open(path, encoding='utf-8')) - 1
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=P.REGISTRY_FIELDS, extrasaction='ignore')
        for i, r in enumerate(rows):
            verdict = r['verdict'] if 'UNDERPOWERED' in r['verdict'] else f'{r["verdict"]}; {P2.NEEDS_FORWARD}'
            w.writerow({**r, 'verdict': verdict, 'n': n + i + 1, 'arbiter': ARBITER_V2,
                        'date': datetime.now(timezone.utc).date().isoformat()})


def render_report(results, failures, meta) -> str:
    def f(x, d=4):
        return '' if x in ('', None) or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'
    L = ['# Gold validation report, round two', '',
         f'Arbiter: `{ARBITER_V2}`. **Gross of cost.** Every verdict **{P2.NEEDS_FORWARD}**.', '',
         P2.DISCLOSURE, '',
         f'- Registration: `results/gold/PRE_REGISTRATION_V2.md` (commit `{meta["registration_commit"]}`)',
         f'- Run: {meta["finished_at"]} ({meta["elapsed_s"]:.0f} s)', '',
         '| family | cell | n | model | baseline | delta | CI | p | alpha | verdict |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        ci = f'[{f(r.get("ci_low"))}, {f(r.get("ci_high"))}]' if r.get('ci_low') not in ('', None) else ''
        L.append(f'| {r["family"]} | {r["cell"]} | {r.get("n_rows", "")} | {f(r.get("metric_model"), 5)} | '
                 f'{f(r.get("metric_baseline"), 5)} | {f(r.get("point_delta"), 5)} | {ci} | '
                 f'{f(r.get("p_value"))} | {f(r.get("alpha_bonferroni"), 5)} | {r["verdict"]} |')
    L += ['', '## Notes', ''] + [f'- `{r["cell"]}`: {r.get("notes", "")}' for r in results]
    if failures:
        L += ['', '## Failures (unspent)', ''] + [f'- {x["cell"]}: {x["error"]}' for x in failures]
    return '\n'.join(L) + '\n'


def run(stage: str = 'validation', cfg: dict | None = None, cfg2: dict | None = None,
        frames: dict | None = None, families=('range', 'macro', 'direction2')) -> dict:
    cfg = cfg or load_config()
    cfg2 = cfg2 or P2.load_config_v2()
    t0 = time.time()
    reg2 = P2.check_stage_allowed(stage, cfg2)
    report_path = repo_path(cfg2['paths']['validation_report'])
    if os.path.exists(report_path):
        raise P2.StageRefused('the round-two validation report already exists: the stage runs once')
    if frames is None and any(f != 'range' for f in families):
        frames = ST.load_pinned(cfg, P.registration(cfg))
    results, failures = [], []
    for fam in families:
        r, fl = SCORERS[fam](cfg, cfg2, reg2, frames)
        results += r
        failures += fl
    meta = {'finished_at': _now(), 'elapsed_s': time.time() - t0,
            'registration_commit': FP.committed_and_clean(repo_path(cfg2['paths']['prereg_json']), REPO)}
    for fam, path in cfg2['paths']['registries'].items():
        rows = [r for r in results if r['family'] == fam]
        if rows:
            append_registry(repo_path(path), rows)
    with open(report_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(render_report(results, failures, meta))
    return {'results': ST._clean(results), 'failures': failures}


def main(argv=None):
    ap = argparse.ArgumentParser(description='round-two gold validation (runs once)')
    ap.add_argument('--stage', required=True, choices=['validation', 'test'])
    args = ap.parse_args(argv)
    out = run(args.stage)
    for r in out['results']:
        print(f"{r['family']:10s} {r['cell']:18s} {r['verdict']}")
    if out['failures']:
        print('FAILURES:', out['failures'])


if __name__ == '__main__':
    main()
