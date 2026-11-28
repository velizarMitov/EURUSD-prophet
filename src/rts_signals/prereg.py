"""The pre-registration and its git gate (spec rts-signals/evaluation; design D8).

    python -m src.rts_signals.prereg --write    write PRE_REGISTRATION.md and .json
    python -m src.rts_signals.prereg --check    report whether both are committed and unchanged

No forward trade counts and nothing is scored until both files are committed unchanged, under the
current family config and rule book (both digests are recorded). The owner makes the commit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from statistics import NormalDist

from . import CELL, COST_HEADER, REPO, config_digest, load_config
from .registry import PREREG_JSON, PREREG_MD, REGISTRY, guard, path, read_json
from .rules import RULES_PATH, rules_digest


class PreregNotCommitted(RuntimeError):
    pass


ASSUMED_TRADES_PER_DAY = 4
ASSUMED_SAME_DAY_CORRELATION = 0.2

FACTS = [
    'RTS\'s own November 2026 spot track record (105 closed trades, 2026-11-02 .. 2026-11-26) states +151 points '
    'gross. On the broker\'s prices 101 trades are confirmed and 4 show a different exit; less the recorded spread '
    'at each entry it is about -214 points. This was seen before the cell was designed and decides nothing.',
    'The base trades of the reports received before the commit (2026-11-23, -25, -27) were also seen: 8 filled. '
    'They are not forward trades.',
    'RTS enters Rule 0 trades at 18:00 New York, the broker\'s daily rollover, when the spread is widest.',
    'The Morning report\'s positions table is read by OCR and checked arithmetically; it is information only and '
    'never enters the cell.',
    'By the owner\'s decision of 2026-11-28 today\'s levels are shown on the chart before any verdict, labelled '
    'unverified.',
]


def power(cfg: dict) -> dict:
    sc, n = cfg['scoring'], int(cfg['forward']['min_trades'])
    sd = float(sc['declared_sd_r'])
    deff = 1 + (ASSUMED_TRADES_PER_DAY - 1) * ASSUMED_SAME_DAY_CORRELATION
    se = sd / n ** 0.5 * deff ** 0.5
    z = NormalDist().inv_cdf
    za, zb = z(1 - float(sc['alpha'])), z(0.80)
    mde = (za + zb) * se
    return {'n_trades': n, 'declared_sd_r': sd, 'design_effect': deff, 'se': se, 'mde_80': mde,
            'calculation': f'SE = {sd} R / sqrt({n}) x sqrt(1 + ({ASSUMED_TRADES_PER_DAY} - 1) x '
                           f'{ASSUMED_SAME_DAY_CORRELATION}) = {se:.4f} R; MDE(80 %) = ({za:.3f} + {zb:.3f}) x SE = '
                           f'{mde:.3f} R per trade'}


def build(cfg: dict) -> dict:
    sc, fw = cfg['scoring'], cfg['forward']
    return {
        'change': 'rts-signals', 'cell': CELL, 'registry': f"{cfg['paths']['results']}/{REGISTRY}",
        'config_digest': config_digest(cfg), 'rules_digest': rules_digest(),
        'rules_file': os.path.relpath(RULES_PATH, REPO).replace('\\', '/'), 'cost': COST_HEADER,
        'hypothesis': 'the mean net R of forward base trades of RTS spot recommendations is above 0',
        'statistic': 'sum of net R / number of trades over the first scored trades',
        'alpha': sc['alpha'], 'family_size': sc['family_size'], 'one_sided': True,
        'test': f"circular block bootstrap over business days keyed by report date, {sc['block_days']}-day blocks, "
                f"{sc['resamples']} resamples, seed {sc['seed']} (fx_bb.score.test_ratio, side 'greater')",
        'verdict_rule': 'KEEP only if the bootstrap lower alpha-quantile of the mean net R is above 0, else DROP',
        'horizon': {'min_trades': fw['min_trades'], 'deadline': fw['deadline'],
                    'rule': f"scored once on the first {fw['min_trades']} forward trades as soon as they exist; with "
                            f"fewer on {fw['deadline']}, recorded UNDERPOWERED — NO DECISION. Never extended."},
        'forward_trade': 'a filled base trade of a spot report dated after the commit date of PRE_REGISTRATION.json '
                         '(Berlin label clock), received (Gmail arrival or drop-folder ingest) before its session '
                         'ended; reports received later are ledgered `received late` and excluded',
        'base_trade': 'src/rts_signals/simulate.py: Rule 0 with proportionality or Rule 1 guerilla, stop / target / '
                      'time stop 16:00 New York (Friday 12:00), a bar touching both is the stop, M1 bid bars; net = '
                      'gross less the recorded bar spread once (buy at entry, sell at exit); R = |entry - stop|',
        'power': power(cfg),
        'drop_is_not_failure': 'A DROP means no detectable edge net of spread at this sample size, not proof of none.',
        'scope': 'Net of the recorded spread only, not slippage, swap or commission. No verdict is a trading '
                 'instruction. The family never touches another family\'s registry, the production models '
                 'directory or config.json.',
        'facts': FACTS,
    }


def write(cfg: dict | None = None) -> tuple:
    import json
    cfg = cfg or load_config()
    guard(cfg)
    rec = build(cfg)
    os.makedirs(os.path.dirname(path(cfg, PREREG_JSON)), exist_ok=True)
    with open(path(cfg, PREREG_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, ensure_ascii=False)
    pw = rec['power']
    L = ['# rts-signals — pre-registration', '', f"> **{COST_HEADER}**", '',
         'Written before any forward trade. Commit this file and PRE_REGISTRATION.json unchanged; the ledger counts '
         'no forward trade and the scoring refuses otherwise.', '',
         f"Config digest `{rec['config_digest']}`. Rules digest `{rec['rules_digest']}` ({rec['rules_file']}).", '',
         '## The cell', '',
         f"- `{CELL}` in `{rec['registry']}`, alpha {rec['alpha']} (family size {rec['family_size']}), one-sided.",
         f"- Hypothesis: {rec['hypothesis']}.",
         f"- Statistic: {rec['statistic']}.",
         f"- Test: {rec['test']}. {rec['verdict_rule']}.",
         f"- Horizon: {rec['horizon']['rule']}",
         f"- A forward trade is {rec['forward_trade']}.",
         f"- The base trade: {rec['base_trade']}.", '',
         '## Power — stated before any forward trade', '',
         f"- Declared per-trade SD {pw['declared_sd_r']} R; about {ASSUMED_TRADES_PER_DAY} trades a day with an assumed "
         f"same-day correlation of {ASSUMED_SAME_DAY_CORRELATION} (design effect {pw['design_effect']:.2f}).",
         f"- {pw['calculation']}.",
         f"- **Minimum detectable mean net R at {pw['n_trades']} trades: about {pw['mde_80']:.2f} R.** "
         f"{rec['drop_is_not_failure']}", '',
         '## Scope', '', f"- {rec['scope']}", '',
         '## Facts recorded', ''] + [f'- {f}' for f in FACTS] + ['']
    with open(path(cfg, PREREG_MD), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path(cfg, PREREG_MD), path(cfg, PREREG_JSON)


def _git(args, repo):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True)


def gate(cfg: dict, repo: str = REPO) -> str:
    """Raise unless both files are committed and unchanged and match the config and the rule book.
    Returns the ISO commit date of PRE_REGISTRATION.json."""
    for name in (PREREG_MD, PREREG_JSON):
        p = path(cfg, name)
        if not os.path.exists(p):
            raise PreregNotCommitted(f'{name} does not exist: write it with --write and commit it')
        rel = os.path.relpath(p, repo)
        if _git(['ls-files', '--error-unmatch', '--', rel], repo).returncode != 0:
            raise PreregNotCommitted(f'{name} is not committed')
        if _git(['diff', '--quiet', 'HEAD', '--', rel], repo).returncode != 0:
            raise PreregNotCommitted(f'{name} differs from its committed version')
    rec = read_json(cfg)
    if rec['config_digest'] != config_digest(cfg):
        raise PreregNotCommitted('the committed pre-registration was written under another config')
    if rec['rules_digest'] != rules_digest():
        raise PreregNotCommitted('the committed pre-registration was written under another rule book')
    rel = os.path.relpath(path(cfg, PREREG_JSON), repo)
    return _git(['log', '-1', '--format=%cI', '--', rel], repo).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description='rts-signals pre-registration')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true')
    g.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    if a.write:
        print('\n'.join(write()))
    else:
        try:
            print(f'committed and unchanged (commit date {gate(load_config())})')
        except PreregNotCommitted as e:
            raise SystemExit(f'NOT READY: {e}')


if __name__ == '__main__':
    main()
