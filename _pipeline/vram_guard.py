"""
VRAM Guard -- frees the GPU VRAM held by the local Claude Code session
(running on qwen3-coder-agent via Ollama) before running the ComfyUI
video-generation render.

WHY THIS EXISTS
----------------
This machine runs Claude Code two ways: (1) hosted sessions talking to
Anthropic's API, which use no local GPU VRAM at all, and (2) a local
session, started by "launch claude.cmd" in the Dreams root, that talks
to Ollama's qwen3-coder-agent model running on THIS GPU. ComfyUI's
LTX-2 video model needs nearly the entire card (measured: 16303 MiB
total, only 139 MiB free while qwen3-coder-agent was loaded). They
cannot run at the same time.

WHY THE LOCAL SESSION DOESN'T NEED TO BE KILLED
--------------------------------------------------
Killing and relaunching the local Claude Code process was tried and is
unnecessary complexity. The local session, while it's waiting on the
Bash tool call that invoked this script, is *blocked* -- it cannot
generate its next turn (and therefore cannot trigger a new Ollama
inference, and therefore cannot cause the model to reload) until this
script's process exits and the tool call returns. That means for the
entire time this script is running the render, there's no way for the
model to get reloaded out from under it, even without killing anything.
Unloading the model with `ollama stop` -- which does NOT touch the
Ollama server process, and does NOT touch the local Claude Code
process -- is sufficient by itself, and was confirmed live: it freed
this machine's VRAM from ~6.4GB to ~15.4GB free in under a second.

WHAT THIS SCRIPT DOES
----------------------
1. Tells Ollama to unload the local model (via its own HTTP API).
2. Polls Ollama's API until it confirms nothing is loaded.
3. Runs the render command and blocks until it finishes.
4. Returns. That's it -- there is nothing to relaunch, because
   nothing was killed. The calling Claude Code session (local or
   hosted) simply continues once this process exits, same as any
   other Bash tool call.

USAGE
-----
Run this INSTEAD OF calling generate_dream.py directly:

    python vram_guard.py -- python generate_dream.py --spec spec_NNN.json

Everything after the bare "--" is the actual render command; run it
from the Dreams/_pipeline folder either way.

HOST-AGNOSTIC BY DESIGN (2026-08-08)
--------------------------------------
Deliberately API-only now, not local-process-based: this used to shell
out to the `ollama` CLI and, as a last resort, use psutil to force-kill
a stuck worker process and query nvidia-smi for a numeric free-VRAM
threshold -- both of those are fundamentally local-machine-only (no
remote equivalent exists for "enumerate/kill a process on a different
host" or "query a GPU this process isn't attached to"), which breaks
entirely the moment this pipeline runs on a different machine than
Ollama/ComfyUI/the GPU (e.g. this orchestrator in a Linux container,
Ollama+ComfyUI+GPU on a Windows host reachable only over the network).
Replaced with Ollama's own HTTP API (GET/POST against ollama_url,
same as every other Ollama call this pipeline makes) for both checking
what's loaded and unloading it -- works identically whether Ollama is
on this machine or a remote one. Traded away: a hard numeric "X MB
actually free" guarantee (nvidia-smi) and a forceful last-resort kill
(psutil) for a stuck worker that wouldn't unload gracefully -- accepted
as the right tradeoff for portability; a model that genuinely won't
unload via its own API now surfaces as a clear failure instead of
being forced off the GPU.

CONFIG
------
Reads the same config.json as the rest of the pipeline (see
dream_step.load_config()) -- ollama_url, creative_model,
graceful_stop_timeout_s, comfyui_url, vision_model. This file talks to
Ollama purely over HTTP, see the host-agnostic-by-design note above
(there is no local-executable config anywhere in the pipeline anymore --
ollama_exe and the "start a local service" feature it supported were
removed 2026-08-16, per explicit direction: this pipeline only ever
depends on a URL being reachable, never on what's installed on this
particular machine). Edit it via the
web UI's Settings tab, or the file directly, if the local model's
name, timeout, or ComfyUI/Ollama location ever change -- do not
hardcode changes here. This used to be a separate
vram_guard.config.json; unified so relocating the pipeline to another
machine only ever means editing one file.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import dream_step

PIPELINE_DIR = Path(__file__).resolve().parent


def load_config():
    """Delegates to dream_step's unified config.json (see the Settings
    tab in the web UI)."""
    return dream_step.load_config()


def list_loaded_models(cfg):
    """Every model name Ollama's own /api/ps currently reports as loaded --
    the HTTP equivalent of `ollama ps`, used directly instead of shelling
    out to the CLI so this works against a remote Ollama exactly the same
    as a local one (same ollama_url every other Ollama call in this
    pipeline already uses). Returns [] on any failure to reach Ollama,
    same as the previous CLI-based check did."""
    try:
        req = urllib.request.Request(f"{cfg['ollama_url']}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        print(f"[vram_guard] WARNING: could not read Ollama's /api/ps ({e}); assuming nothing loaded", flush=True)
        return []


def ollama_stop_model(cfg, model_name=None):
    """Unload model(s) from Ollama via its own HTTP API (POST /api/generate
    with keep_alive=0 -- the documented API-level equivalent of `ollama
    stop`, no CLI needed) instead of shelling out to the `ollama` CLI, so
    this works identically against a remote Ollama instance.

    Confirmed bug (2026-08-08): this used to always stop
    cfg["creative_model"] specifically (the Settings-tab default) -- but
    the chat feature can load a DIFFERENT model per message when "Lock
    chat to this model" is off, and a Settings change can point
    creative_model at a new model without whatever was PREVIOUSLY loaded
    ever getting stopped. Freeing VRAM by name only works if that name
    matches whatever's actually loaded; it silently did nothing for a
    model loaded under a different name, leaving it resident with no
    error at all. Default (model_name=None) now stops EVERY model
    /api/ps currently reports as loaded, not one assumed name -- pass
    model_name for the one case that genuinely wants a single specific
    model stopped (e.g. the vision model after --review-images)."""
    if model_name is not None:
        names = [model_name]
    else:
        names = list_loaded_models(cfg)
        if not names:
            return  # nothing loaded -- nothing to do, no need to print anything
    for name in names:
        print(f"[vram_guard] telling Ollama to unload '{name}' "
              "(this does not stop the Ollama server, and does not touch the Claude Code process)",
              flush=True)
        try:
            payload = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                f"{cfg['ollama_url']}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=15)
        except Exception as e:
            print(f"[vram_guard] unload request for '{name}' failed ({e}) -- continuing anyway", flush=True)


def comfyui_free_vram(cfg):
    """Tell ComfyUI to unload its own cached models and free VRAM (its
    /free API endpoint) -- this is a SEPARATE cache from Ollama's, and
    'ollama stop' does nothing to release it. Hits comfyui_url directly
    -- local or remote, exactly like every other call this pipeline
    makes to ComfyUI. FIXED 2026-08-15: this used to hardcode
    http://127.0.0.1:<port>, discarding comfyui_url's actual host
    entirely, on the reasoning that "VRAM freeing only makes sense
    against a ComfyUI instance running on this machine" -- true when
    this pipeline only supported a local ComfyUI, but false now that a
    remote ComfyUI is a fully supported real configuration: for a
    remote setup this always failed (wasting a timeout on an
    unreachable localhost port, every single render) AND never actually
    freed anything on the GPU that matters, silently letting VRAM
    accumulate there across renders. Silently returns False if ComfyUI
    isn't reachable at all (not fatal -- wait_for_ready below still does
    the real check)."""
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    comfyui_url = cfg["comfyui_url"].rstrip("/")
    try:
        req = urllib.request.Request(
            f"{comfyui_url}/free", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status == 200:
                print(f"[vram_guard] told ComfyUI ({comfyui_url}) to free its cached VRAM", flush=True)
                return True
    except Exception as e:
        print(f"[vram_guard] WARNING: could not reach ComfyUI's /free endpoint at "
              f"{comfyui_url} ({e}) -- continuing anyway", flush=True)
    return False


def model_is_loaded(cfg):
    """True if ANY model is currently loaded in Ollama -- not just
    cfg["creative_model"] by name (see ollama_stop_model's docstring for
    why that assumption was wrong: the chat feature or a Settings change
    can leave a DIFFERENT model loaded than the configured default, and
    that model holds VRAM just as much as the "expected" one would)."""
    return bool(list_loaded_models(cfg))


def start_reload_guard(cfg, poll_interval_s=5):
    """Start a background thread that re-stops the local model for as long
    as a render is in progress, in case something reloads it mid-render
    (a stray Bash-tool backgrounding, an indirect wrapper with the wrong
    timeout, or any other trigger -- this guards against ALL of them
    instead of relying on a single stop-once-at-the-start check).

    Confirmed need for this: a session's render succeeded despite the
    model getting reloaded mid-render multiple times that same night,
    purely because the user was manually re-running `ollama stop` by
    hand in another terminal to compensate -- this makes that manual
    babysitting unnecessary by having the render process do it itself.

    Returns a threading.Event -- call .set() on it once the render
    subprocess has finished to stop the guard thread.

    Confirmed real bug (2026-08-13): this used to log the STOPPED model
    as cfg["creative_model"] unconditionally, regardless of what /api/ps
    actually reported -- with creative_backend set to something other
    than Ollama (e.g. Gemini), every log line named a model that was
    never even the one running, always the stale creative_model config
    value. Worse, the guard didn't distinguish an unwanted external
    reload from the render's OWN vision-QC step legitimately loading
    vision_model mid-render (generate_dream.py reviews each generated
    keyframe via Ollama while this guard is still active for the whole
    render) -- stopping that model out from under its own in-flight
    query forced pointless reload/retry cycles, visible as vision
    queries taking 20s+ for what should be a quick call. The guard now
    logs the real loaded model name(s) and leaves vision_model alone
    when vision_backend is "ollama" (that's expected, legitimate
    concurrent use, not something to fight), still stopping anything
    else exactly as before.
    """
    stop_event = threading.Event()
    expected_model = cfg.get("vision_model") if cfg.get("vision_backend") == "ollama" else None

    def _guard_loop():
        while not stop_event.is_set():
            loaded = list_loaded_models(cfg)
            unexpected = [m for m in loaded if m != expected_model]
            if unexpected:
                print(f"[vram_guard] GUARD: {unexpected} reloaded "
                      "mid-render -- re-stopping to protect the render", flush=True)
                for m in unexpected:
                    ollama_stop_model(cfg, model_name=m)
            stop_event.wait(poll_interval_s)

    thread = threading.Thread(target=_guard_loop, daemon=True)
    thread.start()
    return stop_event


def wait_for_ready(cfg, label, stop_retries=4):
    """Waits until Ollama reports nothing loaded (replaces the old numeric
    "X MB actually free" nvidia-smi check -- see the module docstring's
    host-agnostic-by-design note for why: no remote equivalent exists for
    querying a GPU this process isn't attached to, so this now trusts
    Ollama's own bookkeeping of what it has loaded instead of independently
    verifying VRAM). ComfyUI's own cache (a separate holder, freed via its
    /free API) is checked too, since that's equally API-only and equally
    portable.

    stop_retries: re-issue the unload request for whatever's actually
    loaded (see ollama_stop_model) this many times across the wait
    window, not just once up front. Confirmed live (2026-08-08): a
    graceful unload that reports success can still take multiple
    attempts before the model actually releases -- calling it exactly
    once and then just passively polling left real cases stuck for the
    full graceful_stop_timeout_s."""
    deadline = time.time() + cfg["graceful_stop_timeout_s"]
    retry_interval = cfg["graceful_stop_timeout_s"] / (stop_retries + 1)
    next_retry_at = time.time() + retry_interval
    while time.time() < deadline:
        if not model_is_loaded(cfg):
            print(f"[vram_guard] {label}: Ollama confirms nothing loaded -- enough to proceed", flush=True)
            return True
        if time.time() >= next_retry_at:
            print(f"[vram_guard] {label}: still loaded -- re-issuing unload "
                  "for whatever's currently loaded", flush=True)
            ollama_stop_model(cfg)
            next_retry_at = time.time() + retry_interval
        time.sleep(1)
    print(f"[vram_guard] {label}: Ollama still reports something loaded after "
          f"{cfg['graceful_stop_timeout_s']}s", flush=True)

    # ComfyUI's own model cache (separate from Ollama's) can independently
    # be holding VRAM -- only call this (which force-unloads ALL of
    # ComfyUI's cached models) as a fallback when a plain Ollama unload
    # genuinely wasn't enough, NOT unconditionally on every render.
    # Confirmed regression from calling it unconditionally: it forces a
    # full cold-reload of every node's cached state on every single
    # render, including whatever the FML2V graph's auto-voiceover node
    # relies on staying warm -- this silently broke narration (renders
    # came back with dead air / no narration) even though the render
    # itself still reported success. Only reaching here means a graceful
    # Ollama unload alone already proved insufficient.
    if comfyui_free_vram(cfg):
        retry_deadline = time.time() + 10
        while time.time() < retry_deadline:
            if not model_is_loaded(cfg):
                print(f"[vram_guard] {label} (after freeing ComfyUI cache): "
                      "Ollama confirms nothing loaded -- enough to proceed", flush=True)
                return True
            time.sleep(1)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Free local Claude Code / qwen3-coder-agent VRAM, then run a render command.")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                         help="The render command to run, after a bare --")
    parser.add_argument("--clear-only", action="store_true",
                         help="Unload Ollama's model and free ComfyUI's VRAM/cache, "
                              "then exit -- no render command needed. Confirmed both "
                              "can be left holding VRAM across sessions (ComfyUI "
                              "especially -- it isn't torn down just because the "
                              "previous render finished), so run this before starting "
                              "any new local-agent session, not just before a render.")
    args = parser.parse_args()

    cfg = load_config()

    if args.clear_only:
        comfyui_free_vram(cfg)
        ollama_stop_model(cfg)
        if cfg.get("vision_model"):
            ollama_stop_model(cfg, model_name=cfg["vision_model"])
        wait_for_ready(cfg, "after clearing ComfyUI + Ollama for a new session")
        sys.exit(0)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Usage: python vram_guard.py -- <render command and args>\n"
              "       python vram_guard.py --clear-only", flush=True)
        sys.exit(2)

    comfyui_free_vram(cfg)
    ollama_stop_model(cfg)
    wait_for_ready(cfg, "after freeing local-model + ComfyUI cache")

    print(f"[vram_guard] running render command (blocks until it finishes): {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    exit_code = result.returncode

    print(f"[vram_guard] render command finished with exit code {exit_code}. "
          "Nothing was killed, so there's nothing to resume -- you're still the same session.",
          flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
