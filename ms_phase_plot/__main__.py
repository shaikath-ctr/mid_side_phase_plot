"""Module entry point for `python -m ms_phase_plot`."""

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
