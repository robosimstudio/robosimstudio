# SPDX-License-Identifier: Apache-2.0
"""``python -m robosimstudio_real2sim_agent <command>`` is the command line onto a live page."""

import sys

from .ctl import main

if __name__ == "__main__":
    sys.exit(main())
