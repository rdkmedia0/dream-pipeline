#!/bin/sh
# First-run bootstrap: create a minimal config.json in the mounted state
# volume if one doesn't exist yet, so project data has somewhere sane to
# land out of the box. Everything else (creative/vision models, kf_backend,
# ...) is left unset on purpose -- the GUI's own Settings screen is where
# those get defined, same as a bare install (see CLAUDE.md's config.json
# note: "Edited via the GUI's Settings screen, not by hand").
#
# OLLAMA_URL/COMFYUI_URL/CREATIVE_MODEL/VISION_MODEL are the one
# exception: plain non-secret config (hostnames + model names), unlike
# the Gemini API key -- see secret_store.py's encrypted-at-rest design,
# deliberately NOT given an env-var path here, since a plain compose env
# var would put that key in cleartext in docker-compose.yml/.env,
# undoing the whole point of encrypting it. When set, these four are
# applied on EVERY container start (not just first-run), so editing
# docker-compose.yml and restarting always takes effect -- Settings can
# still override any of them afterward for the rest of that container's
# life, same as any other config.json field. CREATIVE_MODEL/VISION_MODEL
# exist specifically for the "complete" compose profile (see
# docker-compose.yml) -- without them, load_config()'s own auto-detect
# would default BOTH to whatever Ollama model happens to list first,
# which is wrong the moment more than one model is actually pulled.
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

if [ -n "$OLLAMA_URL" ] || [ -n "$COMFYUI_URL" ] || [ -n "$CREATIVE_MODEL" ] || [ -n "$VISION_MODEL" ]; then
    python - "$CONFIG_FILE" <<'PYEOF'
import json
import os
import sys

config_file = sys.argv[1]
with open(config_file, encoding="utf-8") as f:
    config = json.load(f)

ollama_url = os.environ.get("OLLAMA_URL")
comfyui_url = os.environ.get("COMFYUI_URL")
creative_model = os.environ.get("CREATIVE_MODEL")
vision_model = os.environ.get("VISION_MODEL")
if ollama_url:
    config["ollama_url"] = ollama_url
if comfyui_url:
    config["comfyui_url"] = comfyui_url
if creative_model:
    config["creative_model"] = creative_model
if vision_model:
    config["vision_model"] = vision_model

with open(config_file, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PYEOF
fi

exec python dream_step.py --web --host 0.0.0.0 --port 8420
