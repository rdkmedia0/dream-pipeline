@echo off
REM Launches the "complete" Docker profile: Dream Pipeline + Ollama +
REM ComfyUI as sibling containers on THIS machine, with dream-pipeline's
REM ollama_url/comfyui_url/creative_model/vision_model pre-wired to find
REM them automatically -- no manual Settings entry needed.
REM
REM This is a convenience wrapper around `docker compose --profile
REM complete up -d` with the right env vars set, nothing more -- see
REM docker-compose.yml's own comments on each service for what actually
REM gets started, and README.md's "Lite vs Complete" section for what
REM this profile does and doesn't include (notably: NOT ComfyUI's own
REM render checkpoints -- run setup_installer.py once the stack is up
REM to fetch those, same as a lite/bare install).
REM
REM IMPORTANT -- read before running: this requires a real NVIDIA GPU
REM on THIS machine (Ollama and ComfyUI's GPU work both run locally
REM here, unlike "lite" mode which can point at hardware elsewhere)
REM plus the NVIDIA Container Toolkit installed so Docker can actually
REM reach it. Docker Desktop on Windows runs containers inside WSL2,
REM which needs its own WSL-aware NVIDIA driver setup, not just a
REM normal Windows GPU driver -- see README.md's hardware/compatibility
REM notes before running this.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not defined OLLAMA_URL set "OLLAMA_URL=http://ollama:11434"
if not defined COMFYUI_URL set "COMFYUI_URL=http://comfyui:8188"
if not defined CREATIVE_MODEL set "CREATIVE_MODEL=gemma4:12b"
if not defined VISION_MODEL set "VISION_MODEL=qwen3-vl:8b"
if not defined WEB_UI_PORT set "WEB_UI_PORT=8420"

echo Starting the complete profile (Dream Pipeline + Ollama + ComfyUI)...
echo First run downloads real model weights (several GB) -- this can take a while.
docker compose --profile complete up -d

echo.
echo Started. Open http://127.0.0.1:%WEB_UI_PORT% once the containers report healthy
echo (docker compose ps to check^) -- Ollama's default models finish downloading in the
echo background via the ollama-model-init container, no action needed.
