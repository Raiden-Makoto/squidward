#!/bin/bash
# Thin wrapper — delegates to python/run_dsv4.sh which has the canonical flags.
# Override model path: MODEL=/path/to/model ./start_server_dsv4_pro.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/python/run_dsv4.sh"
