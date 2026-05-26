#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change this path if the macOS user keeps the virtual environment elsewhere.
PYTHON_EXE="${PYTHON_EXE:-${SCRIPT_DIR}/go2-eeg-venv/bin/python}"

"${PYTHON_EXE}" "${SCRIPT_DIR}/mac_go2_eeg_keyboard_control.py" \
  --participant test \
  --trials-per-target 20 \
  --trial-duration 5 \
  --iti-duration 3
