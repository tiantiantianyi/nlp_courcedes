#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi
exec "${python_bin}" "${repository_root}/run.py" "$@"
