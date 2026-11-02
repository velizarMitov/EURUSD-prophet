"""Clean-window replay: chooses ONE configuration per model and instrument
(design D4; spec "The replay fixes one configuration before forward scoring").

    python -m src.external.tsfm.replay --freeze    freeze the clean-window bars
    python -m src.external.tsfm.replay             run the grid on the GPU

DESCRIPTIVE. The replay chooses a configuration and nothing else: it never
yields a verdict, and its comparisons with the baselines and with gold's
h1_lstm inform no choice. Verdicts come only from forward rows.

Window: as-of bars 2025-11-01 .. 2026-10-31, after both models' release, and
eligible by the market-break rule. Context may reach back before the window --
only the scored as-of bars must be clean.

Grid per model and instrument, declared before the run:
    weights  zero_shot | fine_tuned   (fine_tuned only where the fine-tune was kept)
    input    returns | closes
    context  2,048 | the model's long context (Chronos-2 8,192, TimesFM 16,256)
Rule: the lowest mean absolute error of the size forecast against the realised
|return| on the FIRST half (as-of before 2026-05-01). The second half is
reported as a check. Once results/external_tsfm/PRE_REGISTRATION.json exists,
the registered configuration is kept whatever a re-run finds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import ARTIFACTS, CLEAN_START, INSTRUMENTS, MODELS, PREREG_JSON, REPLAY, REPO, repo_path
from . import client as CL
from . import data as D
from . import finetune as FT
from . import forecast as Q
from . import sources as S
from . import vendor as V

FIRST_HALF_END = '2026-05-01'
WEIGHTS = ('zero_shot', 'fine_tuned')
INPUT_MODES = ('returns', 'closes')
SHORT_CONTEXT = 2048
WORK = f'{ARTIFACTS}/replay_work'
LABEL = 'DESCRIPTIVE: chooses a configuration only, never a verdict'
EURUSD_WINDOW_BARS = 9000


def long_context(model: str) -> int:
    return V.PINS[model]['long_context']


def config_id(model, weights, mode, context) -> str:
    return f'{model}:{weights}:{mode}:{int(context)}'


def parse_config(cid: str) -> dict:
    model, weights, mode, context = cid.split(':')
    return {'model': model, 'weights': weights, 'mode': mode, 'context': int(context)}


def grid(model: str, fine_tuned: bool) -> list[str]:
    return [config_id(model, w, m, c) for w in WEIGHTS if w == 'zero_shot' or fine_tuned
            for m in INPUT_MODES for c in (SHORT_CONTEXT, long_context(model))]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ── frozen bars ─────────────────────────────────────────────────────────────

def freeze(repo: str = REPO, gold=None, eurusd=None) -> dict:
    """Freeze both windows: gold from its append-only cache, EUR/USD from a
    fresh MT5 read, with its weekly-open clock checked."""
    out = {}
    g = gold if gold is not None else D.gold_h1(repo)
    out['gold'] = D.write_window('gold', g, f'{S.GOLD_H1_CACHE} (append-only cache)', repo=repo)
    if eurusd is None:
        from src.forecast_eval import m15_data as MD
        from src.forecast_eval.mt5_reader import MT5Reader
        with MT5Reader() as r:
            bars = r.bars(D.EURUSD_SYMBOL, 'H1', EURUSD_WINDOW_BARS, sync=False)
        eurusd = D._utc(bars[D.OHLC].iloc[:-1])
        ev = MD.verify_clock(eurusd.index)
        src = (f'MT5 {D.EURUSD_SYMBOL} H1 read {_now()}; weekly opens on the Berlin clock '
               f'{ev["match_rate"]:.1%} of {ev["n_weeks"]}')
    else:
        src = 'injected frame'
    out['eurusd'] = D.write_window('eurusd', eurusd, src, repo=repo)
    return out


def bars(instrument: str, repo: str = REPO) -> tuple[pd.DataFrame, int]:
    """(history + window, first window position), both files digest-checked."""
    V.verify(f'history_{instrument}', repo=repo)
    V.verify(f'window_{instrument}', repo=repo)
    hist = D.read_frozen(D.history_path(instrument, repo))
    win = D.read_frozen(D.window_path(instrument, repo))
    if not hist.index[-1] < win.index[0]:
        raise D.BoundaryViolation(f'{instrument}: history and window overlap')
    return pd.concat([hist, win]), len(hist)


def positions(full: pd.DataFrame, start=CLEAN_START) -> np.ndarray:
    D.check_window_request(start)
    end = pd.Timestamp(D.WINDOW_END, tz='UTC') + pd.Timedelta(days=1)
    return D.as_of_positions(full.index, start, end)


# ── running a configuration ─────────────────────────────────────────────────

def _digest(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def run_config(cid: str, instrument: str, full: pd.DataFrame, pos: np.ndarray, repo: str = REPO,
               call=CL.call) -> np.ndarray:
    """Return-deciles (n x 9, in %) for every as-of position. Cached per
    configuration and data digest, so an interrupted replay resumes."""
    c = parse_config(cid)
    work = repo_path(f'{WORK}/{instrument}', repo)
    os.makedirs(work, exist_ok=True)
    series = D.model_series(full['close'].to_numpy(), c['mode'])
    key = f'{cid}|{_digest(series)}|{_digest(pos)}'.replace(':', '_').replace('|', '__')
    out = os.path.join(work, f'{hashlib.sha256(key.encode()).hexdigest()[:20]}.npy')
    if not os.path.exists(out):
        V.verify(c['model'], repo=repo)
        adapter = None
        if c['weights'] == 'fine_tuned':
            V.verify(FT.adapter_artifact(c['model'], instrument), repo=repo)
            adapter = FT.adapter_dir(c['model'], instrument, repo)
        sp, pp = os.path.join(work, f'series_{c["mode"]}.npy'), os.path.join(work, 'positions.npy')
        np.save(sp, series)
        np.save(pp, pos.astype(np.int64))
        r = call({'op': 'rolling', 'model': c['model'], 'weights_dir': V.weights_dir(c['model'], repo),
                  'adapter_dir': adapter, 'context': c['context'], 'series_path': sp, 'positions_path': pp,
                  'out_path': out, 'batch_size': 32 if c['context'] > 4096 else 64, 'device': 'cuda'},
                 timeout=None, log_path=os.path.join(work, 'replay.log'))
        if not r.get('ok'):
            raise RuntimeError(f'{cid} on {instrument}: {r.get("error")}')
    dec = np.load(out)
    if c['mode'] == 'closes':
        dec = Q.closes_to_returns(dec, full['close'].to_numpy()[pos][:, None])
    return dec


def sizes(dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return Q.size_many(dec), Q.p_up_many(dec)


# ── scoring and the choice ──────────────────────────────────────────────────

def _mae(pred, real, mask):
    m = mask & np.isfinite(pred) & np.isfinite(real)
    return (float(np.mean(np.abs(pred[m] - real[m]))) if m.any() else None), int(m.sum())


def halves(index: pd.DatetimeIndex, pos: np.ndarray) -> np.ndarray:
    return index[pos] < pd.Timestamp(FIRST_HALF_END, tz='UTC')


def registered_configs(repo: str = REPO) -> dict | None:
    path = repo_path(PREREG_JSON, repo)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return {f'{c["model"]}_{c["instrument"]}': c['config_id'] for c in json.load(fh)['cells']}


def choose(results: dict, registered: dict | None) -> dict:
    """Lowest first-half MAE, unless a registration already froze the choice."""
    out = {}
    for key, cfgs in results.items():
        scored = {cid: r['first_half_mae'] for cid, r in cfgs.items() if r.get('first_half_mae') is not None}
        argmin = min(scored, key=scored.get) if scored else None
        if registered and key in registered:
            out[key] = {'chosen': registered[key], 'replay_argmin': argmin, 'frozen_by_registration': True}
        else:
            out[key] = {'chosen': argmin, 'replay_argmin': argmin, 'frozen_by_registration': False}
    return out


def comparators(instrument: str, full: pd.DataFrame, pos: np.ndarray, repo: str = REPO) -> dict:
    """Descriptive comparators on the same as-of rows (NaN where absent)."""
    stamps = [str(t) for t in full.index[pos]]
    if instrument == 'gold':
        study = S.gold_study_h1(repo)
        lstm = np.array([study.get(s, (np.nan, np.nan))[0] for s in stamps])
        garch = np.array([study.get(s, (np.nan, np.nan))[1] for s in stamps])
        return {'baseline': garch, 'h1_lstm': lstm}
    from . import baseline as B
    return {'baseline': B.size_pct(B.load(repo), full)[pos]}


def replay(repo: str = REPO, call=CL.call, instruments=INSTRUMENTS, models=MODELS) -> dict:
    meta = FT.read_meta(repo)
    trained = {k for k, e in meta.get('pairs', {}).items() if e.get('status') == 'trained'}
    report = {'label': LABEL, 'window': [CLEAN_START, D.WINDOW_END], 'first_half_end': FIRST_HALF_END,
              'rule': 'lowest first-half MAE of the size forecast vs realised |return| (%)',
              'grid': {m: grid(m, True) for m in models}, 'instruments': {}, 'run_at': _now()}
    results = {}
    for inst in instruments:
        full, _ = bars(inst, repo)
        pos = positions(full)
        real = np.abs(D.returns_pct(full['close'].to_numpy()))[pos + 1]
        first = halves(full.index, pos)
        comp = comparators(inst, full, pos, repo)
        block = {'n_rows': int(len(pos)), 'first_as_of': str(full.index[pos[0]]),
                 'last_as_of': str(full.index[pos[-1]]), 'n_first_half': int(first.sum()),
                 'comparators': {}, 'models': {}}
        for name, pred in comp.items():
            block['comparators'][name] = {'first_half_mae': _mae(pred, real, first)[0],
                                          'second_half_mae': _mae(pred, real, ~first)[0],
                                          'n_first': _mae(pred, real, first)[1],
                                          'n_second': _mae(pred, real, ~first)[1]}
        for model in models:
            key = f'{model}_{inst}'
            cfgs = {}
            for cid in grid(model, key in trained):
                size, pup = sizes(run_config(cid, inst, full, pos, repo, call))
                r = {'first_half_mae': _mae(size, real, first)[0], 'second_half_mae': _mae(size, real, ~first)[0],
                     'n_first': _mae(size, real, first)[1], 'n_second': _mae(size, real, ~first)[1],
                     'mean_size': float(np.nanmean(size)), 'mean_p_up': float(np.nanmean(pup))}
                for name, pred in comp.items():
                    both = np.isfinite(pred)
                    r[f'same_rows_vs_{name}'] = {'model_mae': _mae(size, real, both)[0],
                                                 f'{name}_mae': _mae(pred, real, both)[0],
                                                 'n': _mae(pred, real, both)[1]}
                cfgs[cid] = r
                print(f'{inst} {cid}: first-half MAE {r["first_half_mae"]:.5f}', flush=True)
            results[key] = cfgs
            block['models'][model] = {'fine_tuned_available': key in trained, 'configs': cfgs}
        report['instruments'][inst] = block
    report['choice'] = choose(results, registered_configs(repo))
    write(report, repo)
    return report


# ── outputs ─────────────────────────────────────────────────────────────────

def _f(x, nd=5):
    return '—' if x is None else f'{x:.{nd}f}'


def render(report: dict) -> str:
    lines = ['# External foundation models: clean-window replay', '',
             f'**{report["label"]}.** Window: as-of {report["window"][0]} .. {report["window"][1]}; '
             f'first half before {report["first_half_end"]}. Rule: {report["rule"]}. Verdicts come only from '
             'forward rows (results/external_tsfm/PRE_REGISTRATION.md).', '']
    for inst, b in report['instruments'].items():
        lines += [f'## {inst}', '', f'{b["n_rows"]} eligible as-of rows ({b["first_as_of"]} .. {b["last_as_of"]}), '
                  f'{b["n_first_half"]} in the first half.', '',
                  '| configuration | first-half MAE | second-half MAE | mean size % | mean P(up) |',
                  '|---|---|---|---|---|']
        for model, m in b['models'].items():
            chosen = report['choice'][f'{model}_{inst}']['chosen']
            for cid, r in m['configs'].items():
                mark = ' **(chosen)**' if cid == chosen else ''
                lines.append(f'| `{cid}`{mark} | {_f(r["first_half_mae"])} | {_f(r["second_half_mae"])} | '
                             f'{_f(r["mean_size"], 4)} | {_f(r["mean_p_up"], 3)} |')
            if not m['fine_tuned_available']:
                lines.append(f'| `{model}:fine_tuned:*` | not run: the fine-tune was not kept | | | |')
        lines += ['', 'Comparators on the same rows (descriptive; they inform no choice):', '',
                  '| comparator | first-half MAE | second-half MAE | rows |', '|---|---|---|---|']
        for name, c in b['comparators'].items():
            lines.append(f'| {name} | {_f(c["first_half_mae"])} | {_f(c["second_half_mae"])} | '
                         f'{c["n_first"]} + {c["n_second"]} |')
        lines.append('')
    lines += ['## Choice', '', '| model × instrument | chosen | replay argmin | frozen by registration |',
              '|---|---|---|---|']
    for key, c in report['choice'].items():
        lines.append(f'| {key} | `{c["chosen"]}` | `{c["replay_argmin"]}` | {c["frozen_by_registration"]} |')
    return '\n'.join(lines) + '\n'


def write(report: dict, repo: str = REPO) -> None:
    out = repo_path(REPLAY, repo)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'replay.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, 'REPLAY.md'), 'w', encoding='utf-8') as fh:
        fh.write(render(report))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Clean-window replay of the external models (descriptive).')
    ap.add_argument('--freeze', action='store_true', help='freeze the clean-window bars and stop')
    a = ap.parse_args(argv)
    if a.freeze:
        print(json.dumps(freeze(), indent=1))
        return
    rep = replay()
    print(json.dumps(rep['choice'], indent=1))


if __name__ == '__main__':
    main()
