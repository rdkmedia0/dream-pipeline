#!/usr/bin/env bash
# Launches the Dream Pipeline web GUI on Linux/macOS.
#
# Dynamic on purpose: this project folder is meant to be shareable
# (e.g. over SMB) and run from multiple machines against the same data,
# so nothing here is a fixed absolute path. Everything is resolved
# relative to THIS script's own location, and the Python virtual
# environment lives in this machine's own per-user local-app-data
# directory (never inside the shared folder itself -- a venv bakes in
# absolute paths and OS-specific binaries, so one made on machine A is
# useless on machine B; see setup_installer.py's _default_venv_dir()
# for the full reasoning, this script just mirrors its OS-detection
# logic in shell).
#
# First run on a new machine: no venv exists yet at that per-user path,
# so this runs setup_installer.py first (installs Python packages into
# a fresh venv there) before launching. Every run after that just
# launches directly -- no reinstall, no prompts.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/_pipeline"

if [ ! -f "$PIPELINE_DIR/dream_step.py" ]; then
    echo "ERROR: $PIPELINE_DIR/dream_step.py not found -- this script must stay next to the _pipeline/ folder." >&2
    exit 1
fi

# Same per-OS local-app-data convention as setup_installer.py's
# _default_venv_dir() / secret_store.py's _local_appdata_dir().
case "$(uname -s)" in
    Darwin) APPDATA_DIR="$HOME/Library/Application Support" ;;
    *)      APPDATA_DIR="${XDG_CONFIG_HOME:-$HOME/.config}" ;;
esac
VENV_PY="$APPDATA_DIR/Dream Pipeline/venv/bin/python"

PYTHON_BOOTSTRAP="$(command -v python3 || command -v python || true)"

if [ ! -x "$VENV_PY" ]; then
    if [ -z "$PYTHON_BOOTSTRAP" ]; then
        echo "ERROR: no python3/python found on PATH -- install Python first (python.org or your OS package manager), then re-run this script." >&2
        exit 1
    fi
    echo "No pipeline environment found for this machine yet -- installing Python packages..."
    echo "(this only happens once per machine; it installs into $APPDATA_DIR/Dream Pipeline/venv, never into the shared folder)"
    # Only the venv/pip step -- NOT setup_installer.py's full interactive
    # main() (which also asks about Ollama/ComfyUI/model downloads via
    # input() prompts, needing a real attached terminal). Everything
    # after Python packages is handled by the GUI's own dependency
    # checks/install actions once it's actually running -- that's the
    # whole point of this session's earlier work on those screens, so
    # this wrapper's only job is to get to a launchable GUI, fast and
    # non-interactively.
    "$PYTHON_BOOTSTRAP" -c "
import sys
sys.path.insert(0, '$PIPELINE_DIR')
import setup_installer
setup_installer.install_pip_requirements()
"
fi

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: setup did not produce a usable environment (see messages above) -- fix the issue and re-run this script." >&2
    exit 1
fi

exec "$VENV_PY" "$PIPELINE_DIR/dream_step.py" --web "$@"
