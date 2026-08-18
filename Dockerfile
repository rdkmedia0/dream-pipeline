# Dream Pipeline -- packaged image. The code is baked in at build time;
# persistent state lives in two volumes mounted at runtime (see
# docker-compose.yml): /data (project folders -- config.json's
# projects_root, set by entrypoint.sh) and /state (config.json itself,
# plus the Fernet key + every .enc secret -- see secret_store.py's
# _local_appdata_dir() and web_ui.py's _secrets_base_dir(), both of
# which respect DREAM_PIPELINE_CONFIG_DIR). Nothing secret or
# machine-specific (.venv, __pycache__, browsers/, *.enc) is copied in
# -- see .dockerignore.
#
# No GPU passthrough here on purpose: this image assumes config.json's
# ollama_url/comfyui_url point at reachable services (local or remote,
# doesn't matter which -- both are always accessed over HTTP, never a
# local binary/GPU from inside this container). If you actually need
# Ollama/ComfyUI running INSIDE this same container with GPU access,
# that's a different, separate image -- don't conflate the two.
FROM python:3.12-slim

WORKDIR /workspace/_pipeline

COPY _pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY _pipeline/ .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV DREAM_PIPELINE_CONFIG_DIR=/state

EXPOSE 8420

# --host 0.0.0.0 (baked into entrypoint.sh) is REQUIRED here --
# web_ui.py's default 127.0.0.1 bind would be the container's OWN
# loopback, unreachable via `docker run -p` no matter what you map.
# Keep the no-auth/local-only trust model intact by publishing to
# 127.0.0.1 on the HOST side instead (docker-compose.yml's ports:
# entry) -- Docker's network isolation is the boundary, not this bind,
# but only if you don't undo that by publishing to every interface.
ENTRYPOINT ["/entrypoint.sh"]
