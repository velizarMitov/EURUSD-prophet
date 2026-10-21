"""Run a gold module with its output appended to research_models/gold/<logname>.log.

Called by scripts/gold/run_hidden.vbs from the scheduled task. Doing the log
redirection here, in Python, means the task needs no .cmd wrapper (the
one-click-launcher change keeps the repo to a single tracked .cmd) and no
console window.

    python -m src.gold.run_task src.gold.forward_logger EURUSDProphet-GoldLogger
"""

from __future__ import annotations

import contextlib
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone

from . import load_config, repo_path


def main(argv=None) -> int:
    module, name = (argv or sys.argv[1:])[:2]
    if not module.startswith('src.gold.'):
        raise SystemExit(f'refusing to run {module!r}: only src.gold modules')
    log_dir = repo_path(load_config()['paths']['artifacts'])
    os.makedirs(log_dir, exist_ok=True)
    code = 0
    with open(os.path.join(log_dir, f'{name}.log'), 'a', encoding='utf-8') as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        print(f'--- {datetime.now(timezone.utc).isoformat(timespec="seconds")} {module}', flush=True)
        sys.argv = [module]
        try:
            runpy.run_module(module, run_name='__main__', alter_sys=True)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception:                                    # logged, then a failing exit code
            traceback.print_exc()
            code = 1
    return code


if __name__ == '__main__':
    sys.exit(main())
