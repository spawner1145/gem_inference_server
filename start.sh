#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if [ -n "${GEM_PYTHON:-}" ]; then
    gem_python="$GEM_PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    gem_python="$VIRTUAL_ENV/bin/python"
elif [ -x .venv/bin/python ]; then
    gem_python=.venv/bin/python
else
    gem_python=python3
fi
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec "$gem_python" -m gem_server "$@"
