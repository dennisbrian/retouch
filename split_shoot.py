#!/usr/bin/env python3
"""Split a shoot folder into sets by capture time (see retouch/shoot_split.py).

    python3 split_shoot.py ~/shoots/2026-09-20              # preview
    python3 split_shoot.py ~/shoots/2026-09-20 --move       # do it
    python3 split_shoot.py --undo ~/shoots/2026-09-20/split-manifest.json
"""

import sys

from retouch.shoot_split import main

if __name__ == "__main__":
    sys.exit(main())
