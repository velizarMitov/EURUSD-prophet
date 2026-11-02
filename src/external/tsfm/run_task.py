"""Run an external-model module with its output appended to
research_models/external_tsfm/<logname>.log.

Called by scripts/external_tsfm/run_hidden.vbs from the scheduled task, the same
way src/gold/run_task.py serves the gold logger: no .cmd wrapper, no console
window.

    python -m src.external.tsfm.run_task src.external.tsfm.forward_logger EURUSDProphet-ExternalLogger
"""

from __future__ import annotations

import contextlib
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone

from . import ARTIFACTS, repo_path


def main(argv=None) -> int:
    module, name = (argv or sys.argv[1:])[:2]
    if not module.startswith('src.external.tsfm.'):
        raise SystemExit(f'refusing to run {module!r}: only src.external.tsfm modules')
    log_dir = repo_path(ARTIFACTS)
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
