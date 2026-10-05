#!/bin/sh
# Linux / Steam Deck / macOS launcher.
cd "$(dirname "$0")"
python3 -c "import numpy, PIL, cryptography" 2>/dev/null || python3 -m pip install --user -r requirements.txt
exec python3 -m vroid2diva "$@"
