"""
render_dream.py -- the ONE tool for turning a finished Dream spec into a
rendered, filed video. This is the only command you (the local model)
need once you and the user have agreed on a concept in conversation and
you've written spec_NNN.json.

USAGE (the entire interface -- one required flag, nothing else)
------------------------------------------------------------------
    python render_dream.py --spec spec_NNN.json

Internally this:
1. Frees your own VRAM (`ollama stop` on your model) so ComfyUI has room.
2. Waits until the GPU actually has enough free VRAM.
3. Runs the real render via generate_dream.py, which talks to ComfyUI,
   validates the result, writes the .mp4 + .txt into the Dream folder,
   and appends the entry to _pipeline/index.json.
4. Prints the same result JSON generate_dream.py would: {"ok": bool,
   "path": ..., "duration": ..., "error": ...}.

Do not call generate_dream.py directly -- always go through this script.
Do not add any other flags here. If a spec needs a non-default workflow,
set "workflow" inside the spec JSON itself (see the new-dream skill),
don't pass it as a command-line flag.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import vram_guard

PIPELINE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Render a Dream spec: frees local-model VRAM, then renders and files it.")
    parser.add_argument("--spec", required=True, help="Path to the spec_NNN.json to render")
    parser.add_argument("--randomize-seeds", action="store_true",
                         help="Use fresh random seeds instead of the graph's saved values -- "
                              "for re-rolling a render whose content didn't change but the "
                              "result was a bad roll.")
    args = parser.parse_args()

    cfg = vram_guard.load_config()
    vram_guard.ollama_stop_model(cfg)
    ready = vram_guard.wait_for_ready(cfg, "after freeing local-model VRAM")
    if not ready:
        print("[render_dream] ABORTING -- Ollama still reports something loaded after "
              "the timeout. Running the render anyway would let ComfyUI and the "
              "still-unloading local model fight over the same GPU, which can hang "
              "both (confirmed live: this caused an unrecoverable stall that needed "
              "a manual kill). Check Ollama's own status -- something else may still "
              "be loaded, or the model just needs more time. Try again once it's "
              "confirmed clear.", flush=True)
        print(json.dumps({"ok": False, "error": "Ollama model not unloaded before render", "path": None}), flush=True)
        sys.exit(1)

    cmd = [sys.executable, str(PIPELINE_DIR / "generate_dream.py"), "--spec", args.spec]
    if args.randomize_seeds:
        cmd.append("--randomize-seeds")
    print(f"[render_dream] running: {' '.join(cmd)}", flush=True)

    # Guard against the model getting reloaded mid-render for any reason
    # (a stray backgrounded turn, a mistaken indirect wrapper with a
    # short timeout, etc.) -- re-stops it automatically for as long as
    # the render is running, instead of only checking once at the start.
    guard_stop = vram_guard.start_reload_guard(cfg)
    try:
        result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
    finally:
        guard_stop.set()

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
