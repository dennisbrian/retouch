#!/bin/bash
# Dev server with auto-reload on file changes
# Usage: ./dev.sh
exec python3 -m watchfiles "python3 -u gui.py" . retouch/
