#!/usr/bin/env bash
# Launches the "complete" Docker profile: Dream Pipeline + Ollama +
# ComfyUI as sibling containers on THIS machine, with dream-pipeline's
# ollama_url/comfyui_url/creative_model/vision_model pre-wired to find
# them automatically -- no manual Settings entry needed.
#
# This is a convenience wrapper around `docker compose --profile
# complete up -d` with the right env vars set, nothing more -- see
# docker-compose.yml's own comments on each service for what actually
# gets started, and README.md's "Lite vs Complete" section for what
# this profile does and doesn't include (notably: NOT ComfyUI's own
# render checkpoints -- run setup_installer.py once the stack is up to
# fetch those, same as a lite/bare install).
#
# IMPORTANT -- read before running: this requires a real NVIDIA GPU on
# THIS machine (Ollama and ComfyUI's GPU work both run locally here,
# unlike "lite" mode which can point at hardware elsewhere) plus the
# NVIDIA Container Toolkit installed so Docker can actually reach it.
# See README.md's hardware/compatibility notes, especially if this
# machine is Windows -- Docker Desktop there runs containers inside
# WSL2, which needs its own WSL-aware NVIDIA driver setup, not just a
# normal Windows GPU driver.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
export COMFYUI_URL="${COMFYUI_URL:-http://comfyui:8188}"
export CREATIVE_MODEL="${CREATIVE_MODEL:-gemma4:12b}"
export VISION_MODEL="${VISION_MODEL:-qwen3-vl:8b}"

echo "Starting the complete profile (Dream Pipeline + Ollama + ComfyUI)..."
echo "First run downloads real model weights (several GB) -- this can take a while."
docker compose --profile complete up -d

echo
echo "Started. Open http://127.0.0.1:${WEB_UI_PORT:-8420} once the containers report healthy"
echo "(docker compose ps to check) -- Ollama's default models finish downloading in the"
echo "background via the ollama-model-init container, no action needed."
