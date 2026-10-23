"""Round-two gold pre-registration (openspec change gold-forecasting-v2, spec
gold-forecasting/registration-amendments).

Round two adds three hypothesis families -- range, macro, direction2 -- with their
own registries and their own Bonferroni bars. Round one is untouched: its
registration, registries, alphas and verdicts are never read for writing, its
config (src/gold/gold_config.json, digest-pinned by round one) is never edited,
and every round-two setting lives in src/gold/gold_config_v2.json.

Disclosure, written into the registration: the validation block [70:80] was
already scored in round one and its results were seen. The test block [80:100]
was spent in round one and is never scored again. Every round-two verdict is
therefore labelled as needing forward confirmation.

The direction2 cells keep round one's power rule unchanged (power.edge,
power.power, power.family_alpha from gold_config.json). A cell whose eligible
validation count cannot reach it is UNDERPOWERED -- NO DECISION, and removing it
never raises the alpha of the cells that remain beyond what the fixed point gives.

    python -m src.gold.prereg2      write PRE_REGISTRATION_V2.{md,json} (once)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from src.forecast_eval import prereg as FP

from . import REPO, load_config, repo_path
from . import prereg as P

V2_CONFIG_PATH = os.path.join(REPO, 'src', 'gold', 'gold_config_v2.json')
UNDERPOWERED = P.UNDERPOWERED
NEEDS_FORWARD = 'needs forward confirmation'
DISCLOSURE = ('The validation block [70:80] was scored in round one (2026-10-21) and its results were '
              'seen before this registration. The test block [80:100] was spent in round one and is '
              'never scored in round two. Every round-two verdict needs forward confirmation.')

RANGE_CELLS = ('h1_range', 'd1_range')
MACRO_CELLS = ('d1_lstm_macro:vol', 'd1_lstm_macro:ret', 'd1_gbm_macro:ret', 'd1_range_macro')
DIRECTION2_CELLS = ('h1_selective', 'h1_barrier', 'h4_dir', 'd1_gbm_macro:dir', 'd1_lstm_macro:dir')

EXPECTED = {
    'h1_range': 'likely CLEARED: it rides on the h1_lstm volatility edge that cleared validation and test',
    'd1_range': 'uncertain: the d1_lstm volatility edge cleared validation but not the test block',
    'd1_lstm_macro:vol': 'likely DROP: 346 validation days leave little room to show a paired gain',
    'd1_lstm_macro:ret': 'DROP expected: no gold return cell has ever cleared',
    'd1_gbm_macro:ret': 'DROP expected: no gold return cell has ever cleared',
    'd1_range_macro': 'likely DROP, for the same reason as d1_lstm_macro:vol',
    'h1_selective': 'DROP expected: round-one H1 direction sat at a coin; abstaining rarely creates an edge',
    'h1_barrier': 'DROP expected, if admitted',
    'h4_dir': 'UNDERPOWERED expected',
    'd1_gbm_macro:dir': 'UNDERPOWERED expected: 346 validation days',
    'd1_lstm_macro:dir': 'UNDERPOWERED expected: 346 validation days',
}


class StageRefused(RuntimeError):
    pass


def load_config_v2(path: str = V2_CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def study_digest(cfg2: dict) -> str:
    """Digest of the settings that shape round-two scoring. The alert threshold
    and the retrain schedule are written later and are deliberately outside it."""
    return hashlib.sha256(json.dumps(cfg2['study'], sort_keys=True).encode()).hexdigest()


def assert_not_test(stage: str) -> None:
    if stage == 'test':
        raise StageRefused('the test block [80:100] was spent in round one; round two never scores it')


# ── counts that decide admission (data only, no model) ──────────────────────

def validation_counts(frames: dict, cfg: dict, cfg2: dict) -> dict:
    """Eligible validation rows per direction2 cell, from the pinned frames."""
    from . import challengers as C
    from . import direction2 as D2
    from . import splits as S
    from . import targets as G
    h1, d1 = frames['H1'], frames['D1']
    out = {}
    p = S.stage_positions(len(h1), 'validation', cfg)
    out['n_h1_dir'] = int(len(C.usable(C.build_inputs(h1, 'H1'), p['score'], G.build_targets(h1['close']), ('dir',))))
    b = cfg2['study']['direction2']['barrier']
    lab = D2.barrier_labels(h1, b['k_hours'], b['m'], b['ewma_span'])
    in_val = np.zeros(len(h1), dtype=bool)
    in_val[p['fit_end']:p['score_end']] = True
    out['n_barrier'] = int((in_val & lab['label'].notna().to_numpy()).sum())
    out['barrier_timeouts'] = int((in_val & lab['eligible'].to_numpy() & lab['label'].isna().to_numpy()).sum())
    w = D2.h4_windows(h1, cfg2['study']['direction2']['h4']['start_hours'])
    out['n_h4'] = int(((w['asof_pos'] >= p['fit_end']) & (w['asof_pos'] < p['score_end'])
                       & w['dir'].notna()).sum()) if len(w) else 0
    pd1 = S.stage_positions(len(d1), 'validation', cfg)
    out['n_d1_dir'] = int(len(C.usable(C.build_inputs(d1, 'D1'), pd1['score'], G.build_targets(d1['close']), ('dir',))))
    return out


def direction2_family(counts: dict, cfg2: dict, edge: float, power: float, family_alpha: float):
    """Round one's fixed point over the direction2 cells. The selective cell's
    covered count is floor x n_h1_dir, with the floor raised just enough to meet
    the requirement at the current alpha, never below floor_min."""
    floor_min = cfg2['study']['direction2']['selective']['floor_min']

    def n_eff(cell, need):
        if cell == 'h1_selective':
            floor = max(floor_min, min(1.0, need / max(counts['n_h1_dir'], 1)))
            return floor * counts['n_h1_dir'], floor
        return {'h1_barrier': counts['n_barrier'], 'h4_dir': counts['n_h4'],
                'd1_gbm_macro:dir': counts['n_d1_dir'], 'd1_lstm_macro:dir': counts['n_d1_dir']}[cell], None

    cells, size, history = list(DIRECTION2_CELLS), len(DIRECTION2_CELLS), []
    while True:
        alpha = family_alpha / size
        need = FP.n_required(alpha, edge, power)
        admitted = [c for c in cells if n_eff(c, need)[0] >= need - 1e-9]
        history.append({'family_size': size, 'alpha': alpha, 'n_required': need, 'admitted': len(admitted)})
        if not admitted or len(admitted) == size:
            break
        size = len(admitted)
    # With nothing admitted the bar stays at the last family size tried: an empty
    # family must not report the looser single-cell requirement.
    alpha = family_alpha / (len(admitted) if admitted else size)
    need = FP.n_required(alpha, edge, power)
    table, floor = [], None
    for c in cells:
        n, f = n_eff(c, need)
        if f is not None:
            floor = f
        table.append({'cell': c, 'n_validation': int(round(n)), 'n_required': round(need, 1),
                      'status': 'ADMITTED' if c in admitted else UNDERPOWERED})
    return admitted, alpha, table, history, floor


# ── the registration ────────────────────────────────────────────────────────

def build_registration(cfg: dict, cfg2: dict, frames: dict, reg1: dict, macro_pin: dict | None = None) -> dict:
    counts = validation_counts(frames, cfg, cfg2)
    if macro_pin is None:
        from . import macro as M
        macro_pin = M.pin_features(cfg2, frames['D1'])
    pw = cfg['power']
    admitted, alpha_d2, table, history, floor = direction2_family(
        counts, cfg2, pw['edge'], pw['power'], pw['family_alpha'])
    st = cfg2['study']
    alpha_range = pw['family_alpha'] / len(RANGE_CELLS)
    alpha_macro = pw['family_alpha'] / len(MACRO_CELLS)
    r1_json = repo_path(cfg['paths']['prereg_json'])
    return {
        'change': 'gold-forecasting-v2',
        'round': 2,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'disclosure': DISCLOSURE,
        'verdict_label': NEEDS_FORWARD,
        'round_one': {'registration_sha256': P.sha256_file(r1_json),
                      'registration_commit': FP.committed_and_clean(r1_json, REPO),
                      'config_sha256': reg1['config_sha256'],
                      'families_untouched': ['direction', 'return', 'volatility']},
        'config_v2_study_sha256': study_digest(cfg2),
        'data': reg1['data'],
        'arbiter': 'validation[70:80] of the round-one pinned rows; test[80:100] spent and never used',
        'cost_basis': 'GROSS of cost -- statistical skill only, never a trading result',
        'families': {
            'range': {
                'cells': list(RANGE_CELLS), 'alpha': alpha_range,
                'definition': ('sigma = E|r| x sqrt(pi/2) from the frozen LSTM volatility forecast; range = '
                               'as-of close x exp(+/- q_t x sigma / 100); q_t by adaptive conformal inference '
                               f'(gamma {st["range"]["gamma"]}, window {st["range"]["window"]}) from past '
                               'settled |z| only; the first `window` validation rows warm up and are not scored'),
                'baseline': 'GARCH sigma (h1_garch_hod / d1_garch) with the static normal quantile, same rows',
                'verdict': (f'CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap CI of '
                            f'mean(winkler_baseline - winkler_model) at the {st["range"]["scored_level"]:.0%} '
                            f'level has lower bound > 0 AND the {st["range"]["scored_level"]:.0%} coverage is '
                            f'within +/-{st["range"]["coverage_tolerance"]:.0%} of target; else DROP. The 95 % '
                            'level and a static-Gaussian LSTM range are reported, not decisive.'),
            },
            'macro': {
                'cells': list(MACRO_CELLS), 'alpha': alpha_macro,
                'inputs': {'fred': st['macro']['fred'], 'mt5': st['macro']['mt5'],
                           'timing': ('FRED values dated on or before the previous US business day; MT5 bars '
                                      'labelled on or before the gold D1 bar; forward-fill only')},
                'pinned': macro_pin,
                'baseline': 'the same model and hyperparameters without the macro block, on identical rows',
                'verdict': ('vol and range: CLEARED iff the paired bootstrap CI of mean(|err_without| - '
                            '|err_with|) (Winkler for range) has lower bound > 0; return: CLEARED iff the '
                            'one-sided Clark-West p < alpha against the model without macro; else DROP.'),
            },
            'direction2': {
                'cells': list(DIRECTION2_CELLS), 'admitted': admitted, 'alpha': alpha_d2,
                'edge': pw['edge'], 'power': pw['power'], 'power_table': table, 'fixed_point': history,
                'counts': counts, 'selective_floor': floor,
                'selective': {'tau': 'largest |p - 0.5| keeping inner-fold training coverage >= the floor',
                              'inner_splits': st['direction2']['selective']['inner_splits']},
                'barrier': st['direction2']['barrier'], 'h4': st['direction2']['h4'],
                'baselines': 'training-majority class and a 50 % coin, on the same scored rows',
                'verdict': ('CLEARED iff both paired bootstrap CIs (vs majority, vs coin) have lower bound > 0; '
                            'abstentions and time-outs are never scored; else DROP.'),
            },
        },
        'expected_outcomes': EXPECTED,
        'ci_method': reg1['ci_method'],
    }


def render_markdown(reg: dict) -> str:
    f = reg['families']
    L = ['# Gold pre-registration, round two', '',
         f'Created {reg["created_at"]} (openspec change `gold-forecasting-v2`). '
         f'Every verdict below is **{reg["verdict_label"]}**. GROSS of cost.', '',
         '## Disclosure', '', reg['disclosure'], '',
         '## Round one is untouched', '',
         f'- Round-one registration digest: `{reg["round_one"]["registration_sha256"]}` '
         f'(commit `{reg["round_one"]["registration_commit"]}`)',
         f'- Round-one config digest: `{reg["round_one"]["config_sha256"]}`; its families '
         'direction, return and volatility keep their registries and alphas.',
         f'- Round-two study settings digest (`src/gold/gold_config_v2.json` → `study`): '
         f'`{reg["config_v2_study_sha256"]}`', '',
         '## Families', '',
         f'### range (alpha {f["range"]["alpha"]:.4f})', '',
         f'- Cells: {", ".join(f["range"]["cells"])}', f'- Definition: {f["range"]["definition"]}',
         f'- Baseline: {f["range"]["baseline"]}', f'- Verdict: {f["range"]["verdict"]}', '',
         f'### macro (alpha {f["macro"]["alpha"]:.4f})', '',
         f'- Cells: {", ".join(f["macro"]["cells"])}', f'- Timing: {f["macro"]["inputs"]["timing"]}',
         f'- Pinned feature file: `{f["macro"]["pinned"].get("features_file")}` '
         f'(SHA-256 `{f["macro"]["pinned"].get("features_sha256")}`, '
         f'{f["macro"]["pinned"].get("complete_rows")} complete rows from '
         f'{f["macro"]["pinned"].get("first_complete")})',
         f'- Baseline: {f["macro"]["baseline"]}', f'- Verdict: {f["macro"]["verdict"]}', '',
         f'### direction2 (alpha {f["direction2"]["alpha"]:.4f})', '',
         f'Edge {f["direction2"]["edge"]:.0%} over 50 % at {f["direction2"]["power"]:.0%} power, '
         'the round-one rule unchanged.', '',
         '| cell | validation n | required n | status |', '|---|---|---|---|']
    for r in f['direction2']['power_table']:
        L.append(f'| {r["cell"]} | {r["n_validation"]} | {r["n_required"]:.0f} | {r["status"]} |')
    L += ['', f'- Selective coverage floor: {f["direction2"]["selective_floor"]:.3f}',
          f'- Barrier: k = {f["direction2"]["barrier"]["k_hours"]} h, m = {f["direction2"]["barrier"]["m"]}, '
          f'EWMA span {f["direction2"]["barrier"]["ewma_span"]}; validation time-outs: '
          f'{f["direction2"]["counts"]["barrier_timeouts"]}',
          f'- Verdict: {f["direction2"]["verdict"]}', '',
          '## Expected outcomes', '']
    L += [f'- `{k}`: {v}' for k, v in reg['expected_outcomes'].items()]
    return '\n'.join(L) + '\n'


def init_registries(cfg2: dict) -> list[str]:
    import csv
    made = []
    for path in cfg2['paths']['registries'].values():
        full = repo_path(path)
        if os.path.exists(full):
            continue
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(P.REGISTRY_FIELDS)
        made.append(path)
    return made


def generate(cfg: dict | None = None, cfg2: dict | None = None, frames: dict | None = None,
             macro_pin: dict | None = None) -> dict:
    cfg = cfg or load_config()
    cfg2 = cfg2 or load_config_v2()
    out_json = repo_path(cfg2['paths']['prereg_json'])
    if os.path.exists(out_json):
        raise StageRefused(f'{cfg2["paths"]["prereg_json"]} already exists: a registration is written once')
    reg1 = P.registration(cfg)
    if reg1 is None:
        raise StageRefused('round one has no registration')
    if frames is None:
        from . import study as ST
        frames = ST.load_pinned(cfg, reg1)
    reg = build_registration(cfg, cfg2, frames, reg1, macro_pin)
    with open(out_json, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(reg, fh, indent=1, default=str)
    with open(repo_path(cfg2['paths']['prereg_md']), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(render_markdown(reg))
    init_registries(cfg2)
    return reg


def registration(cfg2: dict | None = None) -> dict | None:
    cfg2 = cfg2 or load_config_v2()
    path = repo_path(cfg2['paths']['prereg_json'])
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def scoring_allowed_v2(cfg2: dict | None = None, repo: str = REPO) -> bool:
    """Committed, unmodified, and the study settings are still the registered ones."""
    cfg2 = cfg2 or load_config_v2()
    path = repo_path(cfg2['paths']['prereg_json'])
    if FP.committed_and_clean(path, repo) is None:
        return False
    reg = registration(cfg2)
    return reg is not None and reg.get('config_v2_study_sha256') == study_digest(cfg2)


def check_stage_allowed(stage: str, cfg2: dict | None = None) -> dict:
    assert_not_test(stage)
    cfg2 = cfg2 or load_config_v2()
    path = repo_path(cfg2['paths']['prereg_json'])
    if not os.path.exists(path):
        raise StageRefused('no round-two registration: run python -m src.gold.prereg2 and commit it first')
    if FP.committed_and_clean(path, REPO) is None:
        raise StageRefused('PRE_REGISTRATION_V2.json is uncommitted or modified: scoring refuses to run')
    reg = registration(cfg2)
    if reg.get('config_v2_study_sha256') != study_digest(cfg2):
        raise StageRefused('gold_config_v2.json study settings changed after registration')
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description='write the round-two gold pre-registration (once)')
    ap.parse_args(argv)
    reg = generate()
    d2 = reg['families']['direction2']
    print(f'direction2 alpha {d2["alpha"]:.4f}, admitted {d2["admitted"]}, selective floor {d2["selective_floor"]:.3f}')
    for r in d2['power_table']:
        print(f"  {r['cell']:18s} n_val={r['n_validation']:5d}  need {r['n_required']:.0f}  {r['status']}")


if __name__ == '__main__':
    main()
