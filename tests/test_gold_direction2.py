"""Gold direction, second attempt (openspec change gold-forecasting-v2, spec
gold-forecasting/direction-research). Synthetic data only: the round-two
registration must precede the first real score."""
import numpy as np
import pandas as pd
import pytest

from src.gold import direction2 as D2
from src.gold import load_config
from src.gold import prereg2 as P2

CFG = load_config()
CFG2 = P2.load_config_v2()


class Recorder:
    """A stand-in classifier that remembers every row it was trained on."""
    trained = []

    def fit(self, inputs, train_pos, targets, seed=None, h=1):
        Recorder.trained.append(np.asarray(train_pos))
        self.n = len(inputs)
        return self

    def predict(self, inputs):
        rng = np.random.default_rng(len(Recorder.trained))
        return {'p_up': rng.uniform(0.3, 0.7, len(inputs)), 'ret_pct': np.zeros(len(inputs)),
                'vol_pct': np.full(len(inputs), 0.1)}


def _h1(n=4000, seed=3):
    idx = pd.date_range('2023-01-02', periods=n, freq='h', tz='UTC')
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.002, n)))
    return pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close': c}, index=idx)


def _reg2(statuses):
    table = [{'cell': c, 'status': s, 'n_validation': 3000, 'n_required': 2640.3} for c, s in statuses.items()]
    return {'families': {'direction2': {'cells': list(statuses), 'alpha': 0.025, 'power_table': table,
                                        'selective_floor': 0.45}}}


def test_tau_never_sees_validation_rows():
    from src.gold import challengers as C
    from src.gold import splits as S
    from src.gold import targets as G
    h1 = _h1()
    inp = C.build_inputs(h1, 'H1')
    pos = S.stage_positions(len(h1), 'validation', CFG)
    Recorder.trained = []
    tau, predicted = D2.inner_tau(Recorder, inp, pos['fit'], G.build_targets(h1['close']), 0.45, 5, 42)
    assert len(Recorder.trained) == 4
    assert max(int(t.max()) for t in Recorder.trained) < pos['fit_end']
    assert max(predicted) < pos['fit_end'] and 0 < tau < 0.2


def test_abstentions_are_excluded_from_hit_counts():
    p = np.array([0.52, 0.70, 0.30, 0.49, 0.65])
    sel = D2.selective(p, 0.1)
    assert np.isnan(sel[0]) and np.isnan(sel[3])
    assert list(sel[[1, 2, 4]]) == [1.0, 0.0, 1.0]


def test_admitted_cells_are_scored_and_underpowered_ones_are_only_recorded():
    h1 = _h1()
    statuses = {'h1_selective': 'ADMITTED', 'h1_barrier': 'ADMITTED', 'h4_dir': P2.UNDERPOWERED,
                'd1_gbm_macro:dir': P2.UNDERPOWERED, 'd1_lstm_macro:dir': P2.UNDERPOWERED}
    Recorder.trained = []
    rows, failures = D2.score_direction2_cells(CFG, CFG2, _reg2(statuses), {'H1': h1},
                                               make=lambda name: Recorder())
    assert not failures
    assert sorted(r['cell'] for r in rows) == sorted(statuses)
    by = {r['cell']: r for r in rows}
    for c in ('h4_dir', 'd1_gbm_macro:dir', 'd1_lstm_macro:dir'):
        assert by[c]['verdict'] == P2.UNDERPOWERED and by[c]['metric_model'] == ''
    assert by['h1_selective']['family'] == 'direction2' and 'abstentions not scored' in by['h1_selective']['notes']
    assert by['h1_selective']['verdict'].startswith(('CLEARED', 'DROP'))
    assert 'purged training labels' in by['h1_barrier']['notes']


def test_barrier_training_rows_stay_inside_the_training_block():
    from src.gold import splits as S
    h1 = _h1()
    pos = S.stage_positions(len(h1), 'validation', CFG)
    Recorder.trained = []
    D2.score_direction2_cells(CFG, CFG2, _reg2({'h1_barrier': 'ADMITTED'}), {'H1': h1}, make=lambda n: Recorder())
    k = CFG2['study']['direction2']['barrier']['k_hours']
    assert int(Recorder.trained[-1].max()) + k < pos['fit_end']
