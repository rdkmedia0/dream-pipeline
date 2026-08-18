#!/bin/sh
# First-run bootstrap: create a minimal config.json in the mounted state
# volume if one doesn't exist yet, so project data has somewhere sane to
# land out of the box. Everything else (ollama_url, comfyui_url,
# creative/vision models, ...) is left unset on purpose -- the GUI's own
# Settings screen is where those get defined, same as a bare install
# (see CLAUDE.md's config.json note: "Edited via the GUI's Settings
# screen, not by hand").
set -e

CONFIG_DIR="${DREAM_PIPELINE_CONFIG_DIR:-/state}"
CONFIG_FILE="$CONFIG_DIR/config.json"

mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" <<'EOF'
{
  "projects_root": "/data"
}
EOF
fi

exec python dream_step.py --web --host 0.0.0.0 --port 8420
