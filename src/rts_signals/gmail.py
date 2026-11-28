"""Fetching RTS report PDFs from the owner's Gmail, read-only (spec rts-signals/report-ingest; design D2).

    python -m src.rts_signals.gmail --authorize   one-time browser consent (scripts/rts_signals/authorize_gmail.ps1)
    python -m src.rts_signals.gmail --status      'gmail authorised' or 'gmail not authorised'
    python -m src.rts_signals.gmail --fetch       save and ingest new report PDFs

Scope: gmail.readonly. The only API calls are messages.list, messages.get and messages.attachments.get;
nothing is sent, labelled, moved or deleted. The OAuth client file and the token live in the gitignored
research_models/rts_signals/secrets/ and are never printed, logged or committed. The token is refreshed
in place when it expires; if it is missing or revoked, the fetch reports `gmail not authorised` and the
drop folder still works.

Each attachment goes through ingest.ingest_bytes with the message's own received time, so a report
fetched after its session ended is recognisably late in the forward ledger.
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
from datetime import datetime, timezone

from . import art_dir, load_config, sub

SEEN = 'gmail_seen.csv'
NOT_AUTHORISED = 'gmail not authorised'


class NotAuthorised(RuntimeError):
    pass


def token_path(cfg: dict) -> str:
    return os.path.join(sub(cfg, 'secrets'), cfg['gmail']['token_file'])


def client_path(cfg: dict) -> str:
    return os.path.join(sub(cfg, 'secrets'), cfg['gmail']['client_file'])


def query(cfg: dict) -> str:
    g = cfg['gmail']
    return f"from:({' OR '.join(g['senders'])}) {g['query_extra']}"


def credentials(cfg: dict):
    """Stored credentials, refreshed if expired. Raises NotAuthorised without a usable token."""
    p = token_path(cfg)
    if not os.path.exists(p):
        raise NotAuthorised(NOT_AUTHORISED)
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    try:
        creds = Credentials.from_authorized_user_file(p, [cfg['gmail']['scope']])
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write(creds.to_json())
    except Exception as e:                       # revoked or corrupt token: never echo its content
        raise NotAuthorised(f'{NOT_AUTHORISED} ({type(e).__name__})') from None
    if not creds.valid:
        raise NotAuthorised(NOT_AUTHORISED)
    return creds


def service(cfg: dict):
    from googleapiclient.discovery import build
    return build('gmail', 'v1', credentials=credentials(cfg), cache_discovery=False)


def authorize(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    c = client_path(cfg)
    if not os.path.exists(c):
        raise SystemExit(f'put the OAuth client file at {c} first (see results/rts_signals/RUNBOOK.md)')
    from google_auth_oauthlib.flow import InstalledAppFlow
    creds = InstalledAppFlow.from_client_secrets_file(c, [cfg['gmail']['scope']]).run_local_server(port=0)
    os.makedirs(os.path.dirname(token_path(cfg)), exist_ok=True)
    with open(token_path(cfg), 'w', encoding='utf-8') as fh:
        fh.write(creds.to_json())
    return 'gmail authorised'


def _seen(cfg: dict) -> set:
    p = os.path.join(art_dir(cfg), SEEN)
    if not os.path.exists(p):
        return set()
    with open(p, newline='', encoding='utf-8') as fh:
        return {r['message_id'] for r in csv.DictReader(fh)}


def _mark_seen(cfg: dict, mid: str, n: int) -> None:
    p = os.path.join(art_dir(cfg), SEEN)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        if new:
            w.writerow(['message_id', 'pdfs', 'fetched_utc'])
        w.writerow([mid, n, datetime.now(timezone.utc).isoformat(timespec='seconds')])


def _pdf_parts(part: dict) -> list:
    out = []
    if (part.get('filename') or '').lower().endswith('.pdf') and part.get('body', {}).get('attachmentId'):
        out.append(part)
    for p in part.get('parts') or []:
        out += _pdf_parts(p)
    return out


def fetch(cfg: dict | None = None, svc=None, ocr=None) -> dict:
    """Save and ingest the PDFs of new RTS messages. Returns counts; never raises for a missing token."""
    from .ingest import ingest_bytes
    cfg = cfg or load_config()
    try:
        svc = svc or service(cfg)
    except NotAuthorised as e:
        return {'status': str(e), 'messages': 0, 'pdfs': 0, 'stored': 0}
    users = svc.users()
    msgs = users.messages().list(userId='me', q=query(cfg), maxResults=int(cfg['gmail']['max_messages'])).execute()
    seen = _seen(cfg)
    n_msg = n_pdf = n_new = 0
    for m in msgs.get('messages') or []:
        if m['id'] in seen:
            continue
        full = users.messages().get(userId='me', id=m['id'], format='full').execute()
        received = datetime.fromtimestamp(int(full['internalDate']) / 1000, tz=timezone.utc).isoformat(timespec='seconds')
        parts = _pdf_parts(full.get('payload') or {})
        for p in parts:
            att = users.messages().attachments().get(userId='me', messageId=m['id'],
                                                     id=p['body']['attachmentId']).execute()
            data = base64.urlsafe_b64decode(att['data'])
            r = ingest_bytes(cfg, data, 'gmail', p['filename'], received_utc=received, ocr=ocr)
            n_pdf += 1
            n_new += r['status'] not in ('already stored',)
        _mark_seen(cfg, m['id'], len(parts))
        n_msg += 1
    return {'status': 'gmail authorised', 'messages': n_msg, 'pdfs': n_pdf, 'stored': n_new}


def main(argv=None):
    ap = argparse.ArgumentParser(description='read-only Gmail fetch of RTS report PDFs')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--authorize', action='store_true')
    g.add_argument('--status', action='store_true')
    g.add_argument('--fetch', action='store_true')
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.authorize:
        print(authorize(cfg))
    elif a.status:
        try:
            credentials(cfg)
            print('gmail authorised')
        except NotAuthorised as e:
            print(e)
    else:
        print(fetch(cfg))


if __name__ == '__main__':
    main()
