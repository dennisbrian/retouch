#!/bin/bash
# Dev server with auto-reload on file changes
# Usage: ./dev.sh
kill -9 $(lsof -ti :7860) 2>/dev/null
sleep 1
exec python3 -m watchfiles "python3 -u gui.py" . retouch/
