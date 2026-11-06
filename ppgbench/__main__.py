"""``python -m ppgbench`` -> :func:`ppgbench.cli.main`."""
from __future__ import annotations

import sys

from ppgbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
