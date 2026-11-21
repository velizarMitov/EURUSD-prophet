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
import urllib.request
from datetime import datetime, timezone

import numpy as np

from . import load_config, repo_path

ALERT_FIELDS = ['key', 'cell', 'as_of_bar', 'vol_pct', 'threshold_vol_pct', 'percentile', 'notified', 'logged_at']
ALERT_SETTLE_FIELDS = ['key', 'realised_vol_pct', 'settled_at']
ALERT_CELL = 'h1_lstm'
NOTIFY_TIMEOUT_S = 10

# Phone push via ntfy.sh: free, no account, subscribe a private topic in the ntfy
# phone app. The topic is a secret in practice -- anyone who knows it can post, so
# keep it in .env, never commit it. Unset => push is skipped (toast still fires).
NTFY_TOPIC_ENV = 'GOLD_ALERT_NTFY_TOPIC'
NTFY_SERVER_ENV = 'GOLD_ALERT_NTFY_SERVER'
NTFY_DEFAULT_SERVER = 'https://ntfy.sh'

TEXT = {
    'title': {'bg': 'Злато: очаква се голямо движение', 'en': 'Gold: a large move is expected'},
    'body': {'bg': 'Следващия час: около ±{v:.2f} %. Посоката е неизвестна.',
             'en': 'Next hour: about ±{v:.2f} %. Direction unknown.'},
    # Research direction line for the phone push only -- the SAME unvalidated
    # h1_lstm direction MT5 draws next to the alert marker. Marked as research
    # with no proven edge, so it is never read as a validated signal and the
    # alert's canonical "direction unknown" line still stands.
    'research_up': {'bg': 'нагоре', 'en': 'up'},
    'research_down': {'bg': 'надолу', 'en': 'down'},
    'research': {'bg': 'Research посока (без доказан edge): {d} {p} %',
                 'en': 'Research direction (no proven edge): {d} {p} %'},
    # Per-bar forecast push: fires on every new h1_lstm / d1_lstm / d1_gbm bar,
    # not gated by the alert threshold -- same cadence as MT5's arrows. Research
    # tagged, never a trade word.
    'forecast_title': {'h1_lstm': {'bg': 'Злато H1 lstm', 'en': 'Gold H1 lstm'},
                       'd1_lstm': {'bg': 'Злато D1 lstm', 'en': 'Gold D1 lstm'},
                       'd1_gbm': {'bg': 'Злато D1 gbm', 'en': 'Gold D1 gbm'}},
    'forecast_size': {'bg': 'Очакван размер: около ±{v:.2f} %',
                      'en': 'Expected size: about +-{v:.2f} %'},
    'forecast_tag': {'bg': 'Research — без доказан edge.',
                     'en': 'Research -- no proven edge.'},
}

FORECAST_CELLS = ('h1_lstm', 'd1_lstm', 'd1_gbm')
PUSH_LOG_FIELDS = ['key', 'pushed_at']

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


def _forecast_body(cell: str, direction: str | None, prob_pct: float | None,
                   vol_pct: float | None) -> tuple[bytes, str, str]:
    """Body / ASCII title / priority for a flavor='forecast' push."""
    cell = cell if cell in FORECAST_CELLS else 'h1_lstm'
    title = TEXT['forecast_title'][cell]
    r = research_line(direction, prob_pct)
    size_bg = size_en = None
    if vol_pct is not None:
        try:
            v = float(vol_pct)
            if np.isfinite(v):
                size_bg = TEXT['forecast_size']['bg'].format(v=v)
                size_en = TEXT['forecast_size']['en'].format(v=v)
        except (TypeError, ValueError):
            pass
    bg_lines = [f"{title['bg']}:"]
    en_lines = [f"{title['en']}:"]
    if r is not None:
        bg_lines.append(r['bg'])
        en_lines.append(r['en'])
    if size_bg is not None:
        bg_lines.append(size_bg)
        en_lines.append(size_en)
    bg_lines.append(TEXT['forecast_tag']['bg'])
    en_lines.append(TEXT['forecast_tag']['en'])
    body = ('\n'.join(bg_lines) + '\n\n' + '\n'.join(en_lines)).encode('utf-8')
    dir_tag = direction if direction in ('up', 'down') else '?'
    title_ascii = f'{title["en"]}: {dir_tag}'
    if prob_pct is not None:
        try:
            title_ascii += f' {float(prob_pct):.0f}%'
        except (TypeError, ValueError):
            pass
    return body, title_ascii, 'default'


def push_forecasts(new_preds: list, pushed_keys: set, pusher, now_iso: str) -> tuple[list, list]:
    """Per-bar forecast pushes -- every new h1_lstm / d1_lstm / d1_gbm bar, not
    gated by the alert threshold. Same cadence as MT5's arrows. Returns
    (log_rows, notes): log_rows get appended to forecast_pushes.csv for dedup,
    notes carry per-push failures."""
    log_rows, notes = [], []
    for p in new_preds:
        cell = p.get('cell')
        if cell not in FORECAST_CELLS or p.get('key') in pushed_keys:
            continue
        p_up = p.get('p_up')
        direction = None
        prob = None
        try:
            if p_up not in (None, ''):
                v = float(p_up)
                direction = 'up' if v >= 0.5 else 'down'
                prob = (v if direction == 'up' else 1.0 - v) * 100.0
        except (TypeError, ValueError):
            direction, prob = None, None
        vol = p.get('vol_pct')
        vol_val = None
        try:
            if vol not in (None, ''):
                vv = float(vol)
                if np.isfinite(vv):
                    vol_val = vv
        except (TypeError, ValueError):
            pass
        ok, err = pusher(vol_val, direction, prob, 'forecast', cell)
        if not ok:
            notes.append(f'forecast push failed ({cell} {p.get("as_of_bar", "")}): {err}')
        log_rows.append({'key': p['key'], 'pushed_at': now_iso})
    return log_rows, notes


def research_line(direction: str | None, prob_pct: float | None) -> dict | None:
    """Bilingual research-direction line, or None when direction is unknown or
    prob is missing. 'up' / 'down' only; anything else is treated as unknown."""
    if direction not in ('up', 'down') or prob_pct is None:
        return None
    try:
        p = float(prob_pct)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(p):
        return None
    out = {}
    for lang in ('bg', 'en'):
        d = TEXT[f'research_{direction}'][lang]
        out[lang] = TEXT['research'][lang].format(d=d, p=f'{p:.0f}')
    return out


def push_phone(vol_pct: float | None, direction: str | None = None,
               prob_pct: float | None = None, flavor: str = 'alert',
               cell: str = ALERT_CELL, http_post=None) -> tuple[bool, str]:
    """ntfy.sh push to the phone. (ok, err). Returns (True, 'skipped') when the
    topic env is unset -- a missing topic is not a failure, it is the opt-out.

    flavor='alert' (default): large-move alert body, size-only + optional
    research direction line. Priority: urgent. Used by the threshold-gated
    alert path -- same spec as the Windows toast.

    flavor='forecast': per-bar forecast push for one of FORECAST_CELLS, not
    gated by the alert threshold. Shows direction + prob + size, tagged as
    research with no proven edge -- same cadence as MT5's arrows. Priority:
    default (quiet-friendly), so hourly h1_lstm pushes do not scream."""
    topic = (os.environ.get(NTFY_TOPIC_ENV) or '').strip()
    if not topic:
        return True, 'skipped'
    server = ((os.environ.get(NTFY_SERVER_ENV) or '').strip() or NTFY_DEFAULT_SERVER).rstrip('/')
    if flavor == 'forecast':
        body, title_ascii, priority = _forecast_body(cell, direction, prob_pct, vol_pct)
    else:
        t = texts(float(vol_pct))
        lines = [t['bg']['body'], t['en']['body']]
        r = research_line(direction, prob_pct)
        if r is not None:
            lines += [r['bg'], r['en']]
        body = '\n'.join(lines).encode('utf-8')
        title_ascii = f'Gold: large move expected (+-{float(vol_pct):.2f}%)'
        priority = 'urgent'
    req = urllib.request.Request(
        f'{server}/{topic}',
        data=body,
        headers={'Title': title_ascii, 'Priority': priority,
                 'Tags': 'warning' if priority == 'urgent' else 'chart_with_upwards_trend',
                 'Content-Type': 'text/plain; charset=utf-8'},
        method='POST',
    )
    poster = http_post or (lambda r: urllib.request.urlopen(r, timeout=NOTIFY_TIMEOUT_S))
    try:
        resp = poster(req)
    except Exception as e:  # noqa: BLE001 -- a push must never cost the logger pass
        return False, f'{type(e).__name__}: {e}'
    code = getattr(resp, 'status', None) or getattr(resp, 'getcode', lambda: 0)()
    if 200 <= int(code or 0) < 300:
        return True, ''
    return False, f'HTTP {code}'


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
    ap.add_argument('--test-push', action='store_true', help='post one sample push to the configured ntfy topic')
    args = ap.parse_args(argv)
    if args.calibrate:
        print(json.dumps(calibrate(), indent=1))
    if args.test_notification:
        print(notify(0.35))
    if args.test_push:
        print(push_phone(0.35))


if __name__ == '__main__':
    main()
