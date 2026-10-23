"""Entry point of the monthly EURUSDProphet-GoldRetrain task (openspec change
gold-forecasting-v2, spec gold-forecasting/model-versions).

src/gold/run_task.py runs a module with no arguments, so the refit gets its own
module: one call to versions.refit, which writes a new shadow version and never
touches the registered models.
"""

import json

from .versions import refit

if __name__ == '__main__':
    print(json.dumps(refit(), default=str))
