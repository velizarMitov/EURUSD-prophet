"""Large-move alerts (openspec change gold-forecasting-v2, spec gold-forecasting/move-alerts).

When the frozen h1_lstm expects the next hour to move unusually much, the logger
writes an alert row, raises one Windows notification and /gold shows a banner
while that hour is open. An alert names the SIZE of the move and says the
direction is unknown: the only gold forecast with proven skill is the size.

"Unusually much" is fixed once, from the model itself:

    python -m src.gold.alerts --calibrate

takes the configured percentile (90) of the frozen h1_lstm's forecasts on the
rows it was fitted on and stores percentile, value and the model digest in
src/gold/gold_config_v2.json under `alerts`. It is never tuned on live outcomes.
The logger raises no alert until it is calibrated.

Every settled alert records the realised move, so the leaderboard can compare
alerted with non-alerted hours.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np

from . import load_config, repo_path

ALERT_FIELDS = ['key', 'cell', 'as_of_bar', 'vol_pct', 'threshold_vol_pct', 'percentile', 'notified', 'logged_at']
ALERT_SETTLE_FIELDS = ['key', 'realised_vol_pct', 'settled_at']
ALERT_CELL = 'h1_lstm'
NOTIFY_TIMEOUT_S = 10

TEXT = {
    'title': {'bg': 'Злато: очаква се голямо движение', 'en': 'Gold: a large move is expected'},
    'body': {'bg': 'Следващия час: около ±{v:.2f} %. Посоката е неизвестна.',
             'en': 'Next hour: about ±{v:.2f} %. Direction unknown.'},
}

# PowerShell's own Windows Runtime toast API: no extra dependency. The texts come
# in through environment variables, so no quoting can break the command. A plain
# toast leaves the screen after ~5 s and was missed on 2026-10-27, so this one is a
# reminder: it stays on screen with a sound until its button closes it.
_TOAST_PS = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$t = New-Object Windows.Data.Xml.Dom.XmlDocument
$t.LoadXml('<toast scenario="reminder"><visual><binding template="ToastGeneric"><text/><text/></binding></visual><audio src="ms-winsoundevent:Notification.Reminder"/><actions><action activationType="system" arguments="dismiss" content=""/></actions></toast>')
$x = $t.GetElementsByTagName('text')
$x.Item(0).AppendChild($t.CreateTextNode($env:PS_TOAST_TITLE)) > $null
$x.Item(1).AppendChild($t.CreateTextNode($env:PS_TOAST_BODY)) > $null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
"""


def config_v2_path() -> str:
    from .prereg2 import V2_CONFIG_PATH
    return V2_CONFIG_PATH


def settings() -> dict:
    from .prereg2 import load_config_v2
    return load_config_v2()['alerts']


def texts(vol_pct: float) -> dict:
    """Title and body in both languages. Size only: never buy, sell, entry, stop or target."""
    return {lang: {'title': TEXT['title'][lang], 'body': TEXT['body'][lang].format(v=vol_pct)} for lang in ('bg', 'en')}


def notify(vol_pct: float, runner=subprocess.run) -> tuple[bool, str]:
    """One Windows notification, Bulgarian first then English. (ok, error)."""
    t = texts(vol_pct)
    env = dict(os.environ, PS_TOAST_TITLE=f"{t['bg']['title']} / {t['en']['title']}",
               PS_TOAST_BODY=f"{t['bg']['body']}\n{t['en']['body']}")
    try:
        r = runner(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', _TOAST_PS],
                   env=env, capture_output=True, text=True, timeout=NOTIFY_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 -- a toast must never cost the logger pass
        return False, f'{type(e).__name__}: {e}'
    if getattr(r, 'returncode', 1) != 0:
        return False, (getattr(r, 'stderr', '') or 'notification failed').strip()[:200]
    return True, ''


def check(new_preds: list, existing_keys: set, cfg_alerts: dict, now_iso: str) -> list:
    """Alert rows for new h1_lstm forecasts at or above the threshold, one per bar."""
    thr = cfg_alerts.get('threshold_vol_pct')
    if thr is None:
        return []
    out = []
    for p in new_preds:
        if p.get('cell') != ALERT_CELL or p['key'] in existing_keys or p.get('vol_pct') in ('', None):
            continue
        v = float(p['vol_pct'])
        if np.isfinite(v) and v >= float(thr):
            out.append({'key': p['key'], 'cell': ALERT_CELL, 'as_of_bar': p['as_of_bar'], 'vol_pct': v,
                        'threshold_vol_pct': float(thr), 'percentile': cfg_alerts.get('percentile'),
                        'notified': '', 'logged_at': now_iso})
    return out


def settle(alert_rows: list, settled_keys: set, setts: dict, now_iso: str) -> list:
    out = []
    for a in alert_rows:
        if a['key'] in settled_keys or a['key'] not in setts:
            continue
        out.append({'key': a['key'], 'realised_vol_pct': setts[a['key']].get('realised_vol_pct', ''),
                    'settled_at': now_iso})
    return out


def open_alert(cfg, now_label) -> dict | None:
    """The newest alert whose target hour has not closed on the label clock."""
    import csv
    import pandas as pd
    path = os.path.join(repo_path(cfg['paths']['forward']), 'alerts.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for r in reversed(rows):
        end = pd.Timestamp(r['as_of_bar']) + pd.Timedelta(hours=2)
        if end > now_label:
            return {'as_of_bar': r['as_of_bar'], 'window_end': str(end), 'vol_pct': float(r['vol_pct']),
                    'threshold_vol_pct': float(r['threshold_vol_pct']), 'percentile': r.get('percentile')}
        break
    return None


# ── calibration (once) ──────────────────────────────────────────────────────

def calibrate(cfg=None, percentile: float | None = None) -> dict:
    """The configured percentile of the frozen h1_lstm's forecasts on its own fit
    rows (in-sample by design: it defines "unusually large for this model")."""
    from . import challengers as C
    from . import forward_logger as FL
    from . import prereg as P
    from . import study as ST
    cfg = cfg or load_config()
    path = config_v2_path()
    with open(path, encoding='utf-8') as fh:
        cfg2 = json.load(fh)
    pct = float(percentile if percentile is not None else cfg2['alerts']['percentile'])
    reg = P.registration(cfg)
    h1 = ST.load_pinned(cfg, reg)['H1']
    digest = FL.verify_cell(ALERT_CELL, FL.read_manifest(cfg))
    model = FL.load_cell(ALERT_CELL, cfg)
    vol = model.predict(C.build_inputs(h1, 'H1'))['vol_pct']
    vol = vol[np.isfinite(vol)]
    thr = float(np.percentile(vol, pct))
    cfg2['alerts'] = {'percentile': pct, 'threshold_vol_pct': thr, 'n_forecasts': int(vol.size),
                      'calibrated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      'model_digest': digest}
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(cfg2, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    return cfg2['alerts']


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold large-move alerts')
    ap.add_argument('--calibrate', action='store_true', help='fix the threshold from the frozen h1_lstm')
    ap.add_argument('--test-notification', action='store_true', help='raise one sample notification')
    args = ap.parse_args(argv)
    if args.calibrate:
        print(json.dumps(calibrate(), indent=1))
    if args.test_notification:
        print(notify(0.35))


if __name__ == '__main__':
    main()
