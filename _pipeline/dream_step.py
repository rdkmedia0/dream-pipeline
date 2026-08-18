"""
dream_step.py -- the ONE command for all Dream pipeline mechanics.

WHY THIS EXISTS
----------------
Every failure documented in CLAUDE.md tonight was a mechanical/procedural
mistake (wrong number, wrong file, skipped VRAM safety, wrong timeout,
hand-editing index.json, "regenerating" a spec without actually changing
its content, writing custom loop scripts) -- never a creative failure.
The model's only real value-add in this pipeline is originating concepts
and writing prompts. Everything else -- including looping through an
entire batch, not just one item at a time -- is a deterministic decision
a script makes strictly more reliably than a small model reasoning
through a multi-step chain turn after turn, noticing printed output, and
manually deciding to re-invoke this script for the next item. This
script now does that looping itself: one call processes as much of a
batch/rework as it can without creative input, and only returns control
to the model at the exact point it genuinely needs new prompt content
-- there is no other point at which the model should need to reason
about "what's next."

USAGE
-----
    cd _pipeline && python dream_step.py --project dreams --status
    cd _pipeline && python dream_step.py --project dreams --generate 83
    cd _pipeline && python dream_step.py --project dreams --rework 82
    cd _pipeline && python dream_step.py --project dreams --rework 82,84

--project selects which sibling project folder (e.g. "dreams", "animals")
this call operates on. _pipeline itself is shared, content-agnostic code;
each project folder holds its OWN spec_NNN.json files, index.json,
render_hashes.json, rework_history.json, and rendered output folders --
so multiple concurrent projects never collide on numbering or shared
state.

Always run this as its own Bash call with an explicit long timeout (e.g.
timeout: 600000 for 10 minutes) -- it blocks synchronously through
however many real renders it does in one call, each taking several
minutes. Hitting the Bash tool's 120s default backgrounds the call and
forces a response that reloads the local model into VRAM mid-render.
The process itself defends against that too (see the reload guard
below), but the explicit timeout remains the first line of defense and
should still always be used.

STATUS-FIRST, SCRIPT-DRIVEN MODEL
----------------------------------
`--status` is the mandatory first call of every session: it inspects
real project state (specs/renders/uploads on disk) and prints ONLY the
menu options that are actually valid right now, each paired with the
exact command to run. The agent's job is to relay that menu verbatim,
ask the human which option + number(s), then run EXACTLY the command
`--status` named -- never decide the next step from memory/prose.

Why: trusting the agent to decide scope itself from advisory text
invites exactly the kind of drift a deterministic menu prevents.
`--generate` and `--rework` only ever touch the EXACT numbers passed
to them -- there is no stale/carried-over range state anywhere.
AI-composed creative content (spec fields, keyframe prompts) is written
via the manage table's Run updates (see write_row_spec/
write_row_keyframes) -- `--write-spec` on the CLI is direct-content-only
(--spec-json/--spec-json-stdin) for scripted/manual use.

GENERATE MODE -- --generate N[,N...|N-N|all]:
  Renders EXACTLY the listed numbers (or all specced-but-unrendered ones
  for "all", echoed back before acting) -- each must already have a spec
  (write one first via --write-spec). Stops on the first render failure
  rather than continuing past it.

REWORK MODE -- --rework N[,N...|N-N|all]:
  For each listed number (or all rendered ones for "all"), in order,
  within this same call: re-renders from the spec's CURRENT content, no
  questions asked -- the human decided this needs a re-render by asking
  for it, so the script just does it, regardless of whether the spec's
  premise/positive_prompt/negative_prompt hash matches the last render.
  What's rendered is the spec's content; the concept-list markdown is
  the source of ideas; nothing else needs tracking to decide whether a
  requested rerun happens.
  Only the numbers you explicitly list are touched. It will not pull in
  a "related" number (e.g. a duplicate-pair partner) on its own.

RELOAD GUARD
------------
A background thread runs for the ENTIRE duration of this script's
execution (covering every render in a multi-item batch/rework, not just
one), continuously re-stopping the local model (via Ollama's own HTTP
API, see vram_guard.py) if anything reloads it mid-run -- a stray
Bash-tool backgrounding, an indirect wrapper with the wrong timeout, or
any other trigger. This exists so a human never needs to manually
babysit Ollama's up/down state around a render again.

You never call render_dream.py, generate_dream.py, or vram_guard.py
directly -- this script is the only entry point for both new dreams and
reworks.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import random
import re
import shutil
import socket
import string
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import vram_guard

# Vision-model review text routinely contains characters (em dashes, curly
# quotes, arrows) that Windows console's default cp1252 codepage can't
# encode -- confirmed crashing a real render mid-run on a lone "->" arrow
# in a PASS verdict's own text. Force UTF-8 stdout so print() never dies
# on vision-model output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Confirmed critical bug fix: qwen3-vl:8b (the default vision_model, see
# load_config) is a "thinking" model whose internal
# reasoning trace is NOT actually suppressed by "think": False (that flag just
# routes reasoning into a separate "thinking" response field instead of
# stopping it) -- combined with Ollama's small DEFAULT runtime context/predict
# limits, the reasoning alone was exhausting the token budget before any
# visible "response" text got generated at all, returning silently empty
# every time. num_predict/num_ctx below give enough room for the (mandatory,
# often 4000-6000 token) thinking phase AND a real answer -- necessary
# on every call, not just longer/multi-image ones.
VISION_OPTIONS = {"num_predict": 8192, "num_ctx": 16384}

PIPELINE_DIR = Path(__file__).resolve().parent

# Single consolidated source for every mechanical/render-quality rule
# this pipeline needs (see golden_rules.md's own header for why) --
# shared across every project; a project's CREATIVE.md holds only
# genuinely project-specific facts (visual style options, genre), not
# rules. See format_rules().
FORMAT_RULES_PATH = PIPELINE_DIR / "golden_rules.md"

# The strong-backend (Gemini) spec-generation prompt SKELETON --
# the literal wording/section layout Gemini itself chose in a two-round
# self-critique (see build_simple_spec_prompt's docstring), as a
# string.Template with $genre/$title/$duration/$style/$direction/$rules/
# $exclusions/$negative_baseline placeholders. This is the pipeline-wide
# DEFAULT/seed copy only -- do_new_project() copies it into every new
# project's own CREATIVE.md under PROMPT_TEMPLATE_SECTION_HEADER at
# creation time, and that per-project copy (editable from the Creative
# tab, same as genre/style) is what actually gets read at generation time
# (see project_prompt_template()). Editing THIS file only changes the
# default new projects start from, not any already-created project.
SPEC_PROMPT_TEMPLATE_PATH = PIPELINE_DIR / "spec_prompt_template.md"

PROMPT_TEMPLATE_SECTION_HEADER = "## Prompt template"


def default_spec_prompt_template():
    """The pipeline-wide seed copy of the spec-prompt template -- see
    SPEC_PROMPT_TEMPLATE_PATH. Falls back to a minimal inline copy if the
    file's gone missing, so project creation never hard-fails on this."""
    if SPEC_PROMPT_TEMPLATE_PATH.exists():
        return SPEC_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return ('# TASK\nGenerate an original $genre script titled "$title" '
            '($duration seconds, $style).$direction\n\n$rules\n\n'
            '# TOPIC EXCLUSIONS\nDo not repeat concepts, roles, or animal '
            'species from:\n$exclusions\n\n# OUTPUT FORMAT REQUIREMENTS\n'
            'Return ONLY a valid JSON object with keys: title, premise, '
            'positive_prompt, negative_prompt (CSV, starting with '
            '"$negative_baseline"), description, tags (CSV), '
            'fml2v_keyframe_prompts ({"first","middle","last"}).')


def project_prompt_template():
    """This project's own copy of the spec-prompt template -- read from
    its CREATIVE.md under PROMPT_TEMPLATE_SECTION_HEADER (the same file
    the Creative tab edits, seeded there by do_new_project() from
    default_spec_prompt_template()), so a human can tune the actual
    prompt wording/structure per project without touching code. Falls
    back to the pipeline-wide default for any project created before this
    existed (no section header yet in its CREATIVE.md).

    The human-facing explanatory sentence above the fenced block mentions
    "$genre/$title/..." as plain-prose documentation, which is also valid
    string.Template syntax -- so only text inside the ``` fence must be
    taken as the template, never everything after the header. And only a
    ``` that starts its own line (the actual markdown fence convention)
    counts as the fence, found via regex anchored to line start -- an
    inline mention of the word/characters (e.g. "keep it inside the ```
    fence") can't match this no matter how the surrounding prose is
    worded."""
    text = read_creative_md() or ""
    idx = text.find(PROMPT_TEMPLATE_SECTION_HEADER)
    if idx == -1:
        return default_spec_prompt_template()
    after_header = text[idx + len(PROMPT_TEMPLATE_SECTION_HEADER):]
    fences = list(re.finditer(r"(?m)^```\w*[ \t]*\n", after_header))
    if not fences:
        return default_spec_prompt_template()
    body_start = fences[0].end()
    close_idx = after_header.find("\n```", body_start)
    body = after_header[body_start:close_idx] if close_idx != -1 else after_header[body_start:]
    return body.strip()

# Overridable so a container can point config.json at a mounted volume
# (persisting Settings across image rebuilds/recreates) without changing
# anything for a normal install, where this env var is simply unset and
# config.json stays next to the pipeline code as before.
CONFIG_PATH = Path(os.environ.get("DREAM_PIPELINE_CONFIG_DIR", PIPELINE_DIR)) / "config.json"
DEFAULT_CONFIG = {
    # Where project/channel folders live --
    # empty string means "use PIPELINE_DIR.parent" (this pipeline's own
    # install location). A user may want project data (which can grow
    # very large -- rendered videos, keyframes) on a separate disk/path
    # from where the pipeline code itself is installed. See
    # projects_root() below -- every project-path resolution in this
    # codebase goes through that one function, never PIPELINE_DIR.parent
    # directly, so this setting actually takes effect everywhere
    # consistently.
    "projects_root": "",
    "ollama_url": "http://localhost:11434",
    "comfyui_url": "http://localhost:8000",
    # Where ComfyUI is actually installed on disk -- separate from
    # comfyui_url (which is just where its HTTP API listens, and says
    # nothing about whether ComfyUI is actually installed vs just not
    # started yet). No fixed fallback here, same reasoning as
    # creative_model/vision_model below -- load_config() auto-detects a
    # Comfy Desktop install (detect_comfyui_path()) each time this is
    # still unset, rather than a value that could go stale. Set for real
    # once Settings' "Download & Install" flow, setup_installer.py, or a
    # human typing it in saves a real choice.
    "comfyui_path": "",
    # No hardcoded model name here on purpose -- any specific model can be
    # deleted from Ollama at any time, so a fixed fallback would
    # eventually point at nothing. load_config() below fills these in
    # dynamically (first model in the live Ollama list) only when unset;
    # once Settings saves a real choice to config.json, this default is
    # never consulted again.
    "creative_model": None,
    "vision_model": None,
    # When on, chat always uses creative_model with no per-message
    # override -- the model-name dropdown in chat is hidden entirely
    # rather than shown-but-pointless. Off by default so the picker
    # stays available (e.g. for comparing models).
    "lock_creative_model": False,
    # Optional, off by default. When on, every manage-table spec
    # generation (both "S" per-row AI compose and the CLI --interactive
    # path) quietly checks this project's own YouTube Analytics cache for
    # top-performing titles/tags and, if found, gives the model that as
    # style/tag signal -- never allowed to override the row's own locked
    # concept (title/premise), only informs tone/word-choice in whatever
    # it's already writing. "Quiet" specifically means: if no analytics
    # data exists yet for this project, generation proceeds completely
    # normally with no trend context and no error -- this is meant to be
    # left on permanently without needing per-project setup first.
    "spec_trend_mode_enabled": False,
    # Only matters when spec_trend_mode_enabled is on. Off by default:
    # top performers are described by title/tags only. On: also pulls
    # each top performer's real premise (from index.json, durable) and an
    # excerpt of its actual rendered script (from that video's own .txt
    # file, if the render folder hasn't been cleaned up) -- genuinely
    # richer creative signal, but heavier (more file reads, more prompt
    # content) than most requests need.
    "spec_trend_include_script_excerpts": False,
    # vram_guard.py settings -- unified here (was its own separate
    # vram_guard.config.json) so relocating this pipeline to another
    # machine only ever means editing one file.
    "graceful_stop_timeout_s": 25,
    # Which Gemini model gemini_image.py uses for "Online photo" reference-
    # image generation. No hardcoded fallback here either, same reasoning
    # as creative_model/vision_model above -- Google renames/replaces
    # these fairly often, so a fixed default would eventually point
    # at a retired model. None means gemini_image.py's own MODEL constant
    # is used until Settings' "Refresh models" + a real choice sets this.
    "gemini_model": None,
    # A pause switch separate from the saved key itself -- toggled via
    # its own instant Enable/Disable button in Settings (not the
    # section's Remove button, which deletes the key entirely). Lets a
    # human temporarily stop all Gemini spend without losing the saved
    # credential, e.g. pausing for a while without having to re-paste
    # the key later. Defaults True so an already-saved, already-working
    # key isn't silently disabled the first time this field appears in
    # an existing config.json.
    "gemini_enabled": True,
    # Optional spend guard for gemini_image.py's PAID API calls (there is
    # no usable free tier for image generation on this account). Off by
    # default -- a pay-as-you-go Google Cloud billing account already
    # caps real spend on its own; this is a convenience stop for catching
    # a runaway/automated batch before it burns through a lot of calls,
    # not a required safety net. A call COUNT, not a $ figure --
    # per-image price varies by model/resolution and changes over time,
    # so a fixed $ rate baked in here would just go stale.
    "gemini_pay_guard_enabled": False,
    "gemini_pay_guard_monthly_limit": 200,
    # Which backend _creative_completion() (spec/keyframe text generation)
    # actually calls. "ollama" (default) uses creative_model above, fully
    # local/free. "gemini" reuses the SAME saved Gemini API key as
    # gemini_image.py (one key, two uses) -- only selectable once a key is
    # saved, see gemini_text.py.
    "creative_backend": "ollama",
    # Which backend _vision_query() (keyframe/image QC review) actually
    # calls -- same "ollama" (default, local/free) / "gemini" choice as
    # creative_backend. Kept as its own independent setting rather than
    # reusing creative_
    # backend's value: a user may want strong-model creative writing but
    # keep vision review on cheap/free local Ollama, or vice versa -- see
    # gemini_text.generate_vision_text.
    "vision_backend": "ollama",
    # fml2v/i2v keyframe image GENERATION (not review/QC -- see
    # vision_backend for that) -- distinct from the other *_backend
    # settings above since this picks between LOCAL ComfyUI generation and
    # a REAL BILLED Gemini image call, not a free/local vs API-key text-
    # completion choice. A pipeline-wide 2x2: first frame and middle/last
    # are each independently local or Gemini (see generate_dream.py's
    # generate_keyframes for the full per-combination behavior):
    #   "all_local" (default, cheapest): unchanged from before this
    #     setting existed -- first frame respects each Tale's OWN
    #     first_frame_source=="online" toggle (may still be Gemini-seeded
    #     per-Tale); middle/last always local I2I.
    #   "all_gemini": first frame ALWAYS via Gemini (every Tale, ignoring
    #     its own first_frame_source), middle/last always a Gemini image-
    #     EDIT call off that first frame.
    #   "first_local_rest_gemini": first respects the per-Tale toggle,
    #     same as all_local; middle/last always Gemini image-edit.
    #   "first_gemini_rest_local": first ALWAYS via Gemini (every Tale);
    #     middle/last always local I2I, conditioned on that Gemini first
    #     frame same as any other first-frame image.
    "kf_backend": "all_local",
    # No hardcoded fallback, same reasoning as creative_model -- Google
    # renames/retires text models too. None means gemini_text.py's own
    # MODEL constant is used until Settings' "Refresh models" sets this.
    # This is the CREATIVE-writing model specifically -- vision_backend
    # has its OWN separate model setting below (gemini_vision_model),
    # never silently reusing this one: forcing vision QC onto whatever
    # model creative writing happens to be using would mean a genuinely
    # vision-capable model can't be picked independently of (and at
    # different cost from) the creative-writing choice. Concept research
    # (build_concepts_request_payload's web-search-driven flow) always
    # follows creative_backend/this model directly instead -- research
    # feeds straight into writing, so a separate backend/model choice
    # would add a decision with no real payoff.
    "gemini_text_model": None,
    "gemini_vision_model": None,
    # Render output size/duration lives in per-project CREATIVE.md
    # (Duration:/Resolution: lines, see project_render_settings())
    # since it's genuinely a per-channel decision, not pipeline-wide.
}


def load_config():
    """Local pipeline settings -- ComfyUI/Ollama endpoints and which
    models to use, kept in one file instead of scattered hardcoded
    localhost URLs, so this whole pipeline can be relocated to another
    machine (or pointed at a remote ComfyUI/Ollama) by editing one file
    instead of the code. Read fresh on every call (not cached at import
    time) so a change made in the web UI's Settings tab takes effect on
    the next request, no restart needed. Falls back to localhost defaults
    if the file doesn't exist yet, and fills in any key an older/partial
    file is missing. creative_model/vision_model specifically have NO
    fixed fallback (see DEFAULT_CONFIG) -- if either is still unset after
    merging config.json, this queries the live Ollama model list and
    picks the first one, so there's never a hardcoded model name that
    could point at something deleted. Only triggers before Settings has
    ever saved a real choice; silently leaves the key unset if Ollama
    itself isn't reachable (nothing sensible to default to)."""
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    if not config.get("creative_model") or not config.get("vision_model"):
        try:
            models = list_ollama_models(config["ollama_url"])
        except Exception:
            models = []
        if models:
            if not config.get("creative_model"):
                config["creative_model"] = models[0]
            if not config.get("vision_model"):
                config["vision_model"] = models[0]
    if not config.get("comfyui_path"):
        try:
            detected = detect_comfyui_path()
        except Exception:
            detected = None
        if detected:
            config["comfyui_path"] = detected
    return config


def projects_root(config=None):
    """The single source of truth for where project folders live --
    EVERY project-path resolution in this codebase must go through this
    function, never PIPELINE_DIR.parent directly, so config.json's
    projects_root setting actually takes effect everywhere consistently
    (added 2026-08-15). Empty/unset resolves to PIPELINE_DIR.parent,
    the original behavior -- projects living alongside the pipeline
    code itself. A non-default path is created on first use (parents=
    True) if it doesn't exist yet -- the user explicitly chose it, so
    "doesn't exist yet" means "not set up yet", not "invalid"."""
    config = config or load_config()
    root = config.get("projects_root")
    if not root:
        return PIPELINE_DIR.parent
    path = Path(root).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(updates):
    config = load_config()
    config.update({k: v for k, v in (updates or {}).items() if k in DEFAULT_CONFIG})
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def reset_config_to_defaults():
    """Settings' "Load defaults" button -- overwrites config.json with a
    fresh copy of DEFAULT_CONFIG, discarding every saved value (URLs,
    model choices, backend picks, etc). Never touches secrets (Gemini
    key, YouTube client_secret) -- those live in their own encrypted
    .enc files via secret_store, not config.json, so a reset here can't
    accidentally wipe a saved credential."""
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return dict(DEFAULT_CONFIG)


CUSTOM_WORKFLOWS_PATH = PIPELINE_DIR / "custom_workflows.json"


def load_custom_workflows():
    """Registry of confirmed workflow-file wiring, keyed by filename
    (e.g. "workflow_api_myname_i2v.json") -- see workflow_introspect.py
    for how each entry's wiring gets detected, and web_ui.py's Settings
    "Workflow files" section for how an entry gets confirmed via a real
    test render before ever landing here. Same defensive
    exists()/try-except read as load_config() -- a missing or partially
    written file just means "no custom wiring registered yet", not an
    error. At most one entry per type should have "active": true (the
    file currently selected for that type); absence of any active entry
    for a type means "use the built-in default"."""
    if not CUSTOM_WORKFLOWS_PATH.exists():
        return {}
    try:
        return json.loads(CUSTOM_WORKFLOWS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_custom_workflows(registry):
    CUSTOM_WORKFLOWS_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def active_custom_workflow_for_type(type_):
    """The confirmed custom-workflow-file entry currently active for a
    given type (t2v/i2v/fml), or None if the built-in default should be
    used -- the single lookup generate_dream.load_workflow_template()
    needs to decide which file backs a given workflow name."""
    for filename, entry in load_custom_workflows().items():
        if entry.get("type") == type_ and entry.get("active"):
            return filename, entry
    return None, None


def list_ollama_models(ollama_url=None):
    """Models actually installed at the configured Ollama instance (GET
    /api/tags) -- lets the Settings UI offer a real dropdown instead of a
    model name typed in blind, and doubles as a reachability check for
    whatever URL is configured (local or remote)."""
    url = (ollama_url or load_config()["ollama_url"]).rstrip("/")
    with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


# External binaries this pipeline shells out to that are NOT covered by
# config.json (ollama/comfyui are already there). These are bare literal
# names with no override -- if one's missing on a machine this gets
# relocated to, the failure would otherwise surface as a raw, unexplained
# subprocess error
# deep inside a render. platform_note marks Windows-only tools that would
# always be reported missing (not actually broken) on another OS, since
# nothing in this codebase currently branches on sys.platform.
EXTERNAL_BINARIES = [
    # Neither ffmpeg nor ffprobe are listed here -- ffmpeg has no caller
    # (multi-shot/spliced-sides rendering is an unused, non-working
    # feature); ffprobe's job (every render's duration/stream check)
    # goes through PyAV (the `av` pip package) instead of a subprocess
    # call, so neither is a real external dependency anymore -- both
    # fully replaced by pip packages, see requirements.txt.
    #
    # nvidia-smi is not listed here either: VRAM guard is purely
    # API-based (asks Ollama/ComfyUI what THEY have loaded over HTTP, not
    # the local card's own free memory), specifically so this pipeline
    # can be relocated to a machine that
    # isn't the GPU host at all (e.g. this orchestrator in a Linux
    # container, Ollama/ComfyUI/the GPU on a separate Windows box) without
    # a dependency check falsely failing/warning about a local binary
    # that's genuinely irrelevant in that setup.
]


def _dep_status(defined, available, critical):
    """Three-state classification applied uniformly across every
    check_dependencies() entry: a
    check must never collapse "never configured" and "configured but
    broken right now" into the same generic MISSING/red -- they're
    different problems with different fixes. Returns (status, critical):
      - not defined at all  -> "undefined", red if critical else amber
      - defined but unreachable/absent -> "error", red if critical else
        amber (the caller decides criticality -- e.g. a local ollama
        binary missing is NOT critical when ollama_url is already a
        working remote instance, see check_dependencies())
      - defined and confirmed working -> "ok\""""
    if available:
        return "ok", False
    if not defined:
        return "undefined", critical
    return "error", critical


def local_machine_addresses():
    """Every hostname/IP that actually refers to THIS machine -- not just
    the literal strings "localhost"/"127.0.0.1"/"::1". A common setup
    points ollama_url/comfyui_url at the machine's own LAN IP (e.g.
    http://192.168.10.8:11434) rather than localhost -- same machine,
    just addressed differently, but a literal-string-only check would
    treat that as "remote", hiding the local-only Settings fields (Ollama
    executable, ComfyUI install path) AND the dependency-check popup's
    "appears to be installed -- start it?" offer even when Ollama/ComfyUI
    genuinely are installed right here. Best-
    effort: a sandboxed/offline environment where hostname resolution
    itself fails still gets the three literal fallbacks."""
    addrs = {"localhost", "127.0.0.1", "::1"}
    try:
        hostname = socket.gethostname()
        addrs.add(hostname.lower())
        _, _, ips = socket.gethostbyname_ex(hostname)
        addrs.update(ips)
    except Exception:
        pass
    return addrs


def check_dependencies(services=None):
    """Reports whether everything this pipeline needs is actually
    reachable right now, purely over HTTP -- Ollama's and ComfyUI's own
    APIs at their configured URLs, local or remote alike. No local-
    executable/binary check of any kind: this pipeline never shells out
    to Ollama/ComfyUI itself, so whether a binary happens to be on THIS
    machine's PATH is never actually the thing that matters -- only
    reachability is. Returns a
    list of {name, found, status, critical, path, note, platform_note} so
    both the CLI and the web UI's Settings tab can surface the same check
    -- `found` is kept for backward compatibility (found == status=="ok"),
    `status`/`critical` drive the undefined/error distinction above.
    Advisory only -- doesn't block anything.

    `services` (default None = both) restricts which service(s) actually
    get probed -- {"ollama"} or {"comfyui"}. This lets Settings' per-field
    refresh (editing just the Ollama URL, or clicking its own refresh
    icon) probe only that one service instead of always re-running the
    ComfyUI model-file check too."""
    services = set(services) if services else {"ollama", "comfyui"}
    results = []
    for name, note, platform_note in EXTERNAL_BINARIES:
        path = shutil.which(name)
        status, critical = _dep_status(True, path is not None, True)
        results.append({"name": name, "found": path is not None, "status": status, "critical": critical,
                         "path": path, "note": note, "platform_note": platform_note})
    import setup_installer
    config = load_config()
    ollama_selected = any(config.get(k) == "ollama" for k in
                           ("creative_backend", "vision_backend"))

    # Service reachability is checked FIRST, ahead of the local
    # binary/install/model-file rows below: if remote or local checks
    # pass the result should be green, with no warning for unused local
    # services that are reachable via one of the options available. A
    # local install is genuinely one of two equally-valid ways to satisfy Ollama/
    # ComfyUI, not a separate hard requirement layered on top of the
    # URL check -- so each local-fallback row below reads these results
    # and reports fully "ok" (not even the softer amber) whenever the
    # configured URL already works, local or remote alike.
    def _probe_service(service_key, label, url_key, endpoint):
        """One service's reachability probe -- factored out so both
        services can run concurrently instead of serially, each with its
        own up-to-5s timeout that otherwise stacked sequentially on every
        dependency check. Real time saved is largest exactly when it
        matters most: a slow/remote host."""
        url = config[url_key]
        error = None
        try:
            with urllib.request.urlopen(f"{url}{endpoint}", timeout=5) as resp:
                reachable = resp.status == 200
        except Exception as e:
            reachable = False
            error = str(e)
        note = f"HTTP reachability check (config.json: {url_key})"
        if not reachable:
            note += f" -- {error}"
        return service_key, label, url_key, url, reachable, note

    service_specs = tuple(spec for spec in (
        ("ollama", "Ollama service", "ollama_url", "/api/tags"),
        ("comfyui", "ComfyUI service", "comfyui_url", "/queue"),
    ) if spec[0] in services)
    # future.result(timeout=...) hard-bounds the WAIT even though the
    # underlying thread can't actually be killed: a garbage/unreachable
    # host's DNS lookup can hang well past urlopen's own timeout=
    # parameter (getaddrinfo doesn't
    # always respect it, especially on Windows), making a single bad URL
    # save silently take 20-30s with the button just saying "Checking..."
    # the whole time. This caps what the human actually waits to ~6s (5s
    # probe + 1s margin) no matter how badly the DNS lookup itself
    # misbehaves in the background.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(service_specs)) as pool:
        futures = {pool.submit(_probe_service, *spec): spec for spec in service_specs}
        probe_results = []
        for fut, spec in futures.items():
            try:
                probe_results.append(fut.result(timeout=6))
            except concurrent.futures.TimeoutError:
                service_key, label, url_key, *_ = spec
                probe_results.append((service_key, label, url_key, config[url_key], False,
                                       f"HTTP reachability check (config.json: {url_key}) -- timed out"))

    service_reachable = {}
    service_rows = {}
    for service_key, label, url_key, url, reachable, note in probe_results:
        service_reachable[service_key] = reachable
        # ollama_url/comfyui_url always have a non-empty default in
        # config.json -- this check is never really "undefined", only
        # ever ok or a defined-but-unreachable error (see _dep_status).
        service_critical = ollama_selected if service_key == "ollama" else True
        status, critical = _dep_status(bool(url), reachable, service_critical)
        service_rows[service_key] = {
            "name": label, "found": reachable, "status": status, "critical": critical,
            "path": url if reachable else None,
            "note": note, "platform_note": None,
            "service_key": service_key,
            # Shown any time the service isn't reachable -- the only
            # offered fix besides editing the URL above. No local-install
            # detection or "start it for me" option: the human is assumed
            # capable of installing/starting/exposing these themselves, per
            # help.html's install + network-exposure guides.
            "install_url": SERVICE_INSTALL_URLS.get(service_key) if not reachable else None,
        }

    # No standalone "ollama binary on PATH" row -- this pipeline ALWAYS
    # talks to Ollama over its HTTP API (ollama_url), identically whether
    # that's localhost or a remote host; the binary itself is never
    # invoked for anything. Reporting local-binary-presence as a separate
    # pass/fail dependency was misleading either way: false-red when a
    # reachable remote made it irrelevant, and even when genuinely local,
    # it was never really a distinct requirement from "is the service
    # reachable" -- that's the one check that actually matters, above.

    if "ollama" in service_rows:
        results.append(service_rows["ollama"])
    if "comfyui" in service_rows:
        results.append(service_rows["comfyui"])

    # No local-install-path check of any kind -- ComfyUI's own
    # reachability (above) and its model files (below) are both checked
    # purely through comfyui_url's live HTTP API, local or remote alike.
    # There is nothing left for a separate "is ComfyUI installed on THIS
    # machine" row to tell a human that the reachability check doesn't
    # already cover.

    # Model-file completeness is ALWAYS a real dependency, local or
    # remote ComfyUI alike: a
    # workflow graph fails identically either way if the model is
    # genuinely missing wherever ComfyUI actually runs, and
    # check_models_status() already answers this purely from ComfyUI's
    # own live /object_info API (comfyui_url), not a local directory
    # scan -- so this always runs, never skipped just because the
    # service happens to be remote. It naturally reports "unreachable"
    # (via meta['reason']/'stale' below) whenever ComfyUI itself isn't
    # connected yet, rather than needing a separate "only check once
    # connected" gate here -- and since this specific call site already
    # knows service_reachable["comfyui"] for free (the probe above),
    # skip check_models_status() entirely when it's already known to be
    # down rather than paying for its own internal reachability probe
    # again. Skipped ENTIRELY when comfyui wasn't even asked for (see
    # `services` param) -- an Ollama-only refresh has no reason to also
    # re-run ComfyUI's own (much heavier) model-file check.
    if "comfyui" in services:
        if service_reachable["comfyui"]:
            total, missing, meta = setup_installer.check_models_status(None)
        else:
            total, missing, meta = 0, [], {"stale": True, "checked_at": None,
                                            "reason": f"Could not reach ComfyUI at {config['comfyui_url']!r} to check required model files."}
        reason = meta.get("reason")
        note = "all present" if not missing else f"{len(missing)} of {total} required model file(s) missing"
        if reason:
            note = reason
        elif meta["stale"]:
            note += " -- COULD NOT VERIFY (ComfyUI unreachable during recheck), showing last known result"
        # meta["stale"] means "unconfirmed right now" (ComfyUI unreachable),
        # not "confirmed fine" -- with reason unset but stale=True and a
        # cached missing=[], reporting status "ok"/green would put
        # "all present -- COULD NOT VERIFY..." in the SAME breath as a
        # green badge, inconsistent with the sibling "ComfyUI service"/
        # "ComfyUI install" rows correctly going red the moment ComfyUI is
        # unreachable. An unverifiable claim must never render as a
        # confident green pass.
        if reason or meta["stale"]:
            status, critical = "error", True
        else:
            status, critical = _dep_status(True, not missing, True)
        results.append({
            "name": "ComfyUI models", "found": not missing and not reason and not meta["stale"],
            "status": status, "critical": critical,
            "path": None, "note": note, "platform_note": None, "stale": meta["stale"],
        })
    return results


# ComfyUI's real default is 8188 (a plain `python main.py` with no --port
# flag); Ollama's is 11434. Neither is necessarily what config.json's
# ollama_url/comfyui_url point at -- e.g. this project's own comfyui_url
# is a custom 8000 -- so these are only ever used as a fallback probe in
# check_dependencies(), never assumed to be correct on their own.
SERVICE_DEFAULT_PORTS = {"ollama": 11434, "comfyui": 8188}
# comfy.org/download is the official ComfyUI Desktop installer (Windows/
# macOS one-click app, same as ollama.com/download for Ollama) -- not the
# GitHub repo, which is the manual git-clone + Python venv path instead.
SERVICE_INSTALL_URLS = {"ollama": "https://ollama.com/download", "comfyui": "https://www.comfy.org/download"}


def _comfy_desktop_appdata_dir():
    """Where the Comfy Desktop Electron app keeps its own state
    (install manifest, logs) -- used by detect_comfyui_path()."""
    import os
    import platform
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", ""))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "Comfy Desktop"


def detect_comfyui_path():
    """Best-effort auto-detect of an existing ComfyUI Desktop install,
    for Settings' "Detect" button -- there's no Windows registry
    uninstall entry to query (Comfy Desktop doesn't add one), but the
    Desktop app keeps its own install manifest (installations.json)
    listing every environment it manages, which is more reliable than
    guessing well-known folder names. Returns the real ComfyUI source
    path (the one containing main.py) or None if nothing usable is
    found. Only ever reads local files -- never touches the network."""
    manifest = _comfy_desktop_appdata_dir() / "installations.json"
    if not manifest.is_file():
        return None
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    for entry in entries:
        install_path = entry.get("installPath")
        if not install_path or entry.get("status") != "installed":
            continue
        for candidate in (Path(install_path) / "ComfyUI", Path(install_path)):
            if (candidate / "main.py").is_file() and (candidate / "models").is_dir():
                return str(candidate)
    return None


# Fixed rather than a fresh uuid per render (see generate_dream.py's
# queue_prompt) -- ComfyUI only pushes "progress_state" websocket events
# to the connection whose clientId matches whichever prompt is currently
# executing (comfy_execution/progress.py, execution.py's
# `self.server.client_id = extra_data["client_id"]`), so
# query_comfyui_progress() needs a client_id it can always reconnect
# with, not one it would have to learn from a specific subprocess call.
# Safe as a shared constant: vram_guard's whole point is this pipeline
# only ever runs one render at a time, so there's never a second
# concurrent submitter to collide with.
COMFYUI_CLIENT_ID = "dream_pipeline"


def query_comfyui_progress(timeout=2.0):
    """Real render progress straight from ComfyUI's own websocket API --
    per ComfyUI's own source (comfy_execution/progress.py, server.py):
    there is no REST endpoint
    that exposes step-level percentage (/api/jobs/<id> only has
    pending/in_progress/completed/failed, no numeric progress), only a
    "progress_state" event pushed over /ws to whichever connection's
    clientId matches the currently-executing prompt's -- see
    COMFYUI_CLIENT_ID above for how the pipeline guarantees that match.
    Opens a short-lived connection (aiohttp -- already an installed
    transitive dependency, no new install needed), waits up to `timeout`
    seconds for one progress_state message, and returns the
    furthest-along running node's {percent, step, total_steps}, or None
    if nothing arrived in time (ComfyUI idle, unreachable, no render
    actually running, or an old ComfyUI version without this event)."""
    import asyncio

    async def _fetch():
        import aiohttp
        url = load_config()["comfyui_url"]
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"{ws_url}/ws?clientId={COMFYUI_CLIENT_ID}") as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if data.get("type") != "progress_state":
                            continue
                        nodes = data.get("data", {}).get("nodes", {}).values()
                        running = [n for n in nodes if n.get("state") == "running"]
                        if not running:
                            continue
                        node = max(running, key=lambda n: n.get("value") or 0)
                        step, total = node.get("value") or 0, node.get("max") or 0
                        return {"percent": int(step / total * 100) if total else 0,
                                "step": step, "total_steps": total}
        except Exception:
            return None
        return None

    try:
        return asyncio.run(asyncio.wait_for(_fetch(), timeout=timeout))
    except Exception:
        return None


# Set once in main() from --project, before any function below is called.
# Each project folder (a sibling of _pipeline, e.g. "dreams", "animals")
# holds its own spec_NNN.json files + tracking JSON, so projects never
# share/collide on numbering or state. Rendered "Dream #N Title" output
# folders live directly under PROJECT_DIR; everything else the workflow
# needs (specs + tracking JSON) lives under PROJECT_DIR/_data, kept
# separate so the project folder isn't a mix of video output and JSON.
PROJECT_DIR = None
DATA_DIR = None
DREAMS_ROOT = None
INDEX_PATH = None
HISTORY_PATH = None

REQUIRED_SPEC_FIELDS = [
    "number", "title", "premise", "positive_prompt",
    "negative_prompt", "description", "tags", "workflow",
]

# Confirmed working default, proven across many renders this session.
# Reworks may add to this (never remove from it) if a specific render
# needs additional negative terms -- but this baseline always applies
# unless the agent has a documented reason to deviate.
DEFAULT_NEGATIVE_PROMPT = ("blurry image, low detail, camera shake, "
                            "text overlay, subtitles, watermark, morphing")

# The Scene Setup / Timeline & Audio Sync structure produces reliable
# audio/lip-sync vs. a flowing paragraph, while staying flexible on beat
# count/timing and scaling from one voice to several without forcing an
# artificial split. Voice
# count follows the content: Voice A alone for a single narrator/
# character, Voice B/C/... added only when the piece genuinely has that
# many distinct voices trading lines -- never added just to fill a
# template. Required in positive_prompt for every FIRST-time render of a
# number. Reworks may adjust beat count/timing/voice count to fit the
# specific piece, but the section headers plus per-beat Video:/Audio:
# lines stay required by default -- start from this structure, don't
# skip it without --allow-custom-beats.
REQUIRED_POSITIVE_PROMPT_HEADERS = ["[Scene Setup]:", "[Timeline & Audio Sync]:"]

# Single source of truth for the render's target length -- must stay in
# sync with generate_dream.py's MIN_DURATION_S/MAX_DURATION_S accept band
# (currently 18-30s around this) and the ComfyUI workflow's actual frame
# count (currently 588 frames @ 24fps) if either is ever changed. Nothing
# in do_write_spec's validation mechanically requires the example's beat
# count/timing -- this only keeps the illustrative EXAMPLE below from
# quietly going stale (and misleading the model into anchoring on the old
# number) the next time the target duration changes; update this one
# constant rather than hand-editing beat timestamps in prose.
TARGET_DURATION_S = 24


def _build_beat_format_example():
    """BEAT_FORMAT_EXAMPLE, generated from TARGET_DURATION_S so its beat
    timestamps always match the pipeline's actual target length instead
    of a hand-maintained number that can drift from it."""
    beat_s = TARGET_DURATION_S // 4
    lines = ['[Scene Setup]: Cinematic shot of a barn owl perched on a rafter in a '
             'dim barn, moonlight cutting through a gap in the boards.\n',
             '[Timeline & Audio Sync]:']
    # A beat with Video:
    # but no Audio: line at all pushes that beat's audio to the WRONG
    # place in the render (audio from an earlier beat bled into it) --
    # every beat needs both lines, always, even when there's no dialogue.
    # A "silent" beat still gets an Audio: line, just one describing
    # ambient sound/music/sfx instead of a quoted Voice line.
    beats = [
        (0, "The owl sits still, looking around calmly, eyes wide open.",
         'Voice A [Deadpan Night-Watch Narrator]: "23:14. All quiet."'),
        (1, "The owl blinks once, unhurried, continuing its watch.",
         'Voice A [Deadpan Night-Watch Narrator]: "23:15. Still quiet."'),
        (2, "The owl tilts its head, surveying the branch below.",
         'Voice A [Deadpan Night-Watch Narrator]: "This job is mostly quiet."'),
        (3, "The owl settles back into its still, watchful pose as the camera "
            "stays steady.", "Low ambient barn hum, no dialogue."),
    ]
    for i, video, audio in beats:
        start, end = i * beat_s, min((i + 1) * beat_s, TARGET_DURATION_S)
        lines.append(f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}:")
        lines.append(f"- Video: {video}")
        lines.append(f"- Audio: {audio}")
        lines.append("")
    lines.append(
        '(This is a FORMAT template only -- invent your own scene/dialogue content '
        "for the actual piece, don't reuse this owl example. EVERY beat needs both "
        "a Video: line and an Audio: line -- never omit Audio: even when there's no "
        "dialogue; describe ambient sound/music/sfx/silence instead (see the last "
        "beat above). Beat count/timing should span this project's actual target "
        f"duration (~{TARGET_DURATION_S}s here, but match whatever this pipeline is "
        "actually configured to render), "
        "not necessarily 4 beats. Use ONE voice (Voice A only) for single-narrator/"
        "single-character content; add Voice B, Voice C, etc. only when the content "
        "genuinely needs more than one distinct voice trading lines.)")
    return "\n".join(lines)


BEAT_FORMAT_EXAMPLE = _build_beat_format_example()


TYPE_TO_WORKFLOW = {"t2v": "fp8_t2v", "i2v": "i2v", "fml": "fml2v"}
WORKFLOW_TO_TYPE = {v: k for k, v in TYPE_TO_WORKFLOW.items()}

# Fields each non-default workflow needs beyond the base REQUIRED_SPEC_FIELDS
# -- used to detect whether a rework/generate switching type already has
# everything it needs (see ensure_workflow_type).
TYPE_SPECIFIC_FIELDS = {
    "i2v": ["image_path"],
    "fml2v": ["fml2v_keyframe_prompts"],
}


def recent_titles_for_dedup(limit=15):
    """Last `limit` index.json entries' title/premise, embedded directly
    into a spec/concepts prompt payload so dedup checking doesn't need a
    separate lookup."""
    index = load_json(INDEX_PATH, [])
    entries = [e for e in index if isinstance(e, dict)]
    entries.sort(key=lambda e: e.get("number", 0))
    return [{"number": e.get("number"), "title": e.get("title"),
              "premise": e.get("premise")} for e in entries[-limit:]]


def format_rules():
    """The shared, pipeline-wide mechanical prompt-format rules (bracket
    beats, camera/shot composition, negative-prompt baseline, etc.) --
    read from FORMAT_RULES_PATH, one file every project's spec/keyframe
    requests pull from. Human-edited directly; a rule change here takes
    effect for every project immediately, no per-project duplication.
    Returns None if the file is missing (a from-scratch pipeline install
    before it's been created) so callers degrade gracefully instead of
    crashing on a request that's otherwise still perfectly usable."""
    if not FORMAT_RULES_PATH.exists():
        return None
    text = FORMAT_RULES_PATH.read_text(encoding="utf-8")
    _, _, duration_s = project_render_settings()
    if duration_s != 24:
        # format_rules.md's own text hardcodes "00:24" as the (default)
        # target -- substitute the user's actual configured duration so
        # story generation is told to hit the real render length, not a
        # stale default. A no-op string when duration_s is still the
        # default 24, so nothing changes for anyone who hasn't touched
        # this setting.
        text = text.replace("00:24", f"{duration_s // 60:02d}:{duration_s % 60:02d}")
    return text


def using_strong_creative_backend():
    """True when creative_backend is a genuinely strong external model
    (Gemini API) rather than the small local Ollama default."""
    return load_config().get("creative_backend", "ollama") != "ollama"


_NEGATIVE_BASELINE_MARKER = "Negative-prompt baseline (always include): "


def golden_rules_body():
    """golden_rules.md with its own header/intro paragraph stripped --
    just the rule sections, ready to drop into a prompt's $rules slot (the
    header explains the file to a human editing it, not to the model)."""
    text = format_rules()
    if not text:
        return ""
    # Drop everything up to (and including) the first '## ' section header
    # -- that's the "# Golden Rules" title + explanatory paragraph, meant
    # for whoever edits this file, not for the prompt itself.
    marker = "\n## "
    idx = text.find(marker)
    return ("## " + text[idx + len(marker):]) if idx != -1 else text


def golden_negative_baseline():
    """The negative-prompt baseline CSV, parsed out of golden_rules.md's
    own marker line -- one real source for this list, so it can't
    silently drift from the file a human actually edits."""
    text = format_rules() or ""
    for line in text.splitlines():
        if line.startswith(_NEGATIVE_BASELINE_MARKER):
            return line[len(_NEGATIVE_BASELINE_MARKER):].strip()
    return ("blurry image,low detail,camera shake,text overlay,subtitles,watermark,"
            "morphing,warping,melting,distorted,deformed,unnatural movement,"
            "inconsistent physics,objects merging")


# CREATIVE.md now holds ONLY project-specific facts (genre, visual style
# option(s)) -- everything else moved to golden_rules.md (see that file's
# own header). These markers are what the user-manageable Creative tab's
# CREATIVE.md is expected to contain; parsed live so editing that page
# actually changes what gets sent, rather than these being hardcoded
# Python constants a human editing the Creative tab has no way to reach.
_GENRE_MARKER = "Genre:"
_STYLE_MARKER = "Visual style:"
_DURATION_MARKER = "Duration:"
_RESOLUTION_MARKER = "Resolution:"
_FALLBACK_STYLE_OPTIONS = (
    "Warm, modern feature-film animated style",
    "Photorealistic nature-documentary style",
)


def style_negative_terms(style):
    """Contra-style exclusion terms for the negative-prompt baseline --
    an animated-style request with a negative baseline that never
    excludes photorealism/live-action risks the video model morphing
    between render styles mid-clip. This
    project's $style is chosen per-generation from two real options
    (project_genre_and_styles), so the same exclusion mechanism is
    applied to whichever style actually got picked THIS call, by keyword
    match, rather than hardcoding one direction."""
    style_lower = (style or "").lower()
    if "photorealistic" in style_lower or "documentary" in style_lower:
        return "3d animation, cartoon, cel-shaded, illustrated, stylized rendering"
    if "animat" in style_lower or "feature-film" in style_lower or "3d" in style_lower:
        return "photorealistic, live action, real footage"
    return ""
# Byte-identical to every current workflow_api_*.json's own baked-in
# defaults -- a project that's never set Duration:/Resolution: in its
# CREATIVE.md gets exactly the render behavior it always had, sourced
# per-project rather than from a global Settings field.
_FALLBACK_RENDER_WIDTH = 512
_FALLBACK_RENDER_HEIGHT = 896
_FALLBACK_RENDER_DURATION_S = 24

# Matches workflow_api_fml2v.json's own baked-in default strengths for
# nodes 2110 (first), 2278 (middle), 2108 (last) -- used whenever a spec
# has no fml2v_guide_strengths override, so the manage table always has
# something real to show/edit rather than a blank field.
_FALLBACK_GUIDE_STRENGTHS = {"first": 0.4, "middle": 0.7, "last": 1.0}


def project_genre_and_styles():
    """(genre, [style, ...]) parsed live from this project's own
    CREATIVE.md -- the same file the Creative tab edits -- so genre/style
    are genuinely project-editable, not baked into pipeline code. Falls
    back to sane defaults (comedy; the two styles this pipeline started
    from) if CREATIVE.md hasn't been given these lines yet, so an
    unconfigured project still works."""
    text = creative_guidance_pointer() or ""
    genre = None
    styles = []
    for line in text.splitlines():
        line = line.strip().lstrip("*-").strip()
        if line.startswith(_GENRE_MARKER):
            genre = line[len(_GENRE_MARKER):].strip() or None
        elif line.startswith(_STYLE_MARKER):
            style = line[len(_STYLE_MARKER):].strip()
            if style:
                styles.append(style)
    return genre or "Comedy", styles or list(_FALLBACK_STYLE_OPTIONS)


def project_render_settings():
    """(width, height, duration_s) parsed live from this project's own
    CREATIVE.md -- "Duration: 24" (seconds) and "Resolution: 512x896"
    lines, same marker-line pattern as Genre:/Visual style:. Render
    size/length is genuinely a per-PROJECT decision (a channel's own
    format), not a pipeline-wide one, so this lives next to the
    genre/style facts already in the same per-project file. Falls back to
    this pipeline's original baked-in defaults for a project that hasn't
    set these yet."""
    text = creative_guidance_pointer() or ""
    width, height, duration_s = None, None, None
    for line in text.splitlines():
        line = line.strip().lstrip("*-").strip()
        if line.startswith(_DURATION_MARKER):
            try:
                duration_s = int(line[len(_DURATION_MARKER):].strip().rstrip("s").strip())
            except ValueError:
                pass
        elif line.startswith(_RESOLUTION_MARKER):
            m = re.match(r"(\d+)\s*x\s*(\d+)", line[len(_RESOLUTION_MARKER):].strip(), re.I)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
    return (width or _FALLBACK_RENDER_WIDTH, height or _FALLBACK_RENDER_HEIGHT,
            duration_s or _FALLBACK_RENDER_DURATION_S)


_CONCEPT_DIRECTIVE_HEADER = "## Concept directive"

GENRE_OPTIONS = ("Comedy", "Drama", "Documentary", "Educational", "Heartwarming", "Satire")
STYLE_OPTIONS = (
    "Warm, modern feature-film animated style",
    "Photorealistic nature-documentary style",
    "Cinematic 3D animated style",
    "Flat, stylized 2D animated style",
)
# Most-used values as dropdown suggestions -- both fields stay a real
# <select> with an explicit "Custom..." fallback (see selectField in
# web_ui.py), so any value outside this list is still fully typeable.
# Conventional sizes up through short-form (15-60s), then minute
# increments up to a 30-minute cap.
DURATION_OPTIONS = (15, 24, 30, 45, 60, 90, 120, 180, 300, 600, 900, 1200, 1800)
RESOLUTION_OPTIONS = (
    "512x896",   # this pipeline's original default -- fastest to render
    "1080x1920", # YouTube Shorts/TikTok/Reels standard vertical
    "720x1280",  # vertical, lighter-weight than 1080x1920
    "1024x1024", # square
    "1920x1080", # standard landscape (16:9)
    "1280x720",  # landscape, lighter-weight than 1920x1080
)


def project_concept_directive():
    """This project's standing creative directive, from CREATIVE.md's
    '## Concept directive' section -- everything between that header and
    the next '##' (or end of file). REAL, functional input:
    fed into every build_simple_spec_prompt() call via the template's
    $concept_directive slot. Blank means exactly what the form says --
    the AI originates a new idea from scratch each call; non-blank is a
    standing instruction applied to every story in this project (distinct
    from a per-video note, which only applies to that one regen). Not to
    be confused with concepts.md, this project's master concept-LIST file
    (see find_concept_list_path/concept_list_stats) -- that's a separate,
    always-fixed-path mechanism this directive has no relationship to."""
    text = read_creative_md() or ""
    idx = text.find(_CONCEPT_DIRECTIVE_HEADER)
    if idx == -1:
        return ""
    after = text[idx + len(_CONCEPT_DIRECTIVE_HEADER):]
    next_idx = after.find("\n## ")
    return (after[:next_idx] if next_idx != -1 else after).strip()


def creative_fields():
    """Every field the Creative tab's FORM edits, parsed live from this
    project's own CREATIVE.md -- one call backing the form's GET load.
    The Creative tab is a real form (genre/style/duration/resolution as
    dropdown-plus-custom fields, concept directive as plain text, prompt
    template as its own textarea) -- a human editing this shouldn't need
    to know or preserve CREATIVE.md's exact markdown shape (marker
    lines, fence syntax) by hand; the form/compose_creative_md pair is
    the only thing that needs to agree on that shape."""
    genre, styles = project_genre_and_styles()
    width, height, duration_s = project_render_settings()
    concept_path, concept_total, concept_remaining = concept_list_stats()
    return {
        "genre": genre,
        "style1": styles[0] if len(styles) > 0 else "",
        "style2": styles[1] if len(styles) > 1 else "",
        "duration_s": duration_s,
        "resolution": f"{width}x{height}",
        "concept_directive": project_concept_directive(),
        # Live facts about concepts.md (always this project's DATA_DIR /
        # "concepts.md" -- there's no per-project custom path, see
        # find_concept_list_path) -- purely informational display, a
        # completely separate mechanism from concept_directive above. It's
        # checked unconditionally for every number regardless of anything
        # set here, whether it was ever manually written to, AI-populated
        # via Manage's "Research & add ideas", or left untouched.
        "concept_list_total": concept_total,
        "concept_list_remaining": concept_remaining,
        "template": project_prompt_template(),
    }


def compose_creative_md(display_name, genre, styles, duration_s, resolution, concept_directive, template_body):
    """Build a full CREATIVE.md from the Creative tab FORM's fields --
    the single place that knows this file's canonical shape (marker
    lines + '## Concept directive' + PROMPT_TEMPLATE_SECTION_HEADER's
    fenced block). Used by do_new_project's stub, the guided-questions
    fill, and the form's Save -- one composer, so all three stay in the
    same shape rather than each hand-building slightly different text."""
    styles = [s.strip() for s in styles if (s or "").strip()]
    if not styles:
        styles = list(_FALLBACK_STYLE_OPTIONS)
    style_lines = "\n".join(f"{_STYLE_MARKER} {s}" for s in styles)
    template_body = (template_body or "").strip() or default_spec_prompt_template().strip()
    return (
        f"# {display_name} Creative Guidelines\n\n"
        "This project's own creative facts -- editable from the Creative tab. "
        "Every mechanical/render-quality rule (lip sync, complexity budget, "
        "one fixed environment, negative-prompt baseline, copyright, etc.) is "
        "shared pipeline-wide and lives in _pipeline/golden_rules.md instead "
        "-- not repeated here.\n\n"
        f"{_GENRE_MARKER} {genre or 'Comedy'}\n\n"
        f"{style_lines}\n\n"
        f"{_DURATION_MARKER} {int(duration_s or _FALLBACK_RENDER_DURATION_S)}\n"
        f"{_RESOLUTION_MARKER} {resolution or f'{_FALLBACK_RENDER_WIDTH}x{_FALLBACK_RENDER_HEIGHT}'}\n\n"
        f"{_CONCEPT_DIRECTIVE_HEADER}\n\n"
        # Must never write a placeholder sentence here when blank (e.g.
        # "(Blank -- the AI originates...") -- project_concept_directive()
        # reads this section back as real, human-provided directive text,
        # and build_simple_spec_prompt() would silently include that
        # SENTENCE ITSELF in every prompt as a "standing directive."
        # Genuinely empty means genuinely empty; the explanation lives
        # only in the form's tooltip/placeholder, never in the file.
        f"{(concept_directive or '').strip()}\n\n"
        f"{PROMPT_TEMPLATE_SECTION_HEADER}\n\n"
        "This is the actual prompt sent to the AI for each story -- tweak its "
        "wording/structure freely inside the fenced code block below, just "
        "keep the placeholders intact (genre/title/duration/style/direction/"
        "rules/exclusions/negative_baseline get filled in automatically each "
        "call). Keep it inside that fence -- only the fenced block is read as "
        "the real template.\n\n"
        f"```\n{template_body}\n```\n"
    )


_INVENT_NEW_TITLE_PLACEHOLDER = ("<invent a brand-new, original title and premise for this "
                                  "slot -- do not reuse the previous one>")


def _simple_prompt_title_and_style(existing_spec, master_list_entry, style_options, title_locked=True):
    """Resolve $title/$style for build_simple_spec_prompt: a regen keeps
    the existing spec's own title and whichever of style_options its
    positive_prompt already used (so re-rolling a story doesn't silently
    flip its visual style); a new spec takes its title from the master
    list entry (format 'Tale #N: TITLE -- premise...') and picks a style
    at random, same as a human manually varying it call to call.

    title_locked=False means the manage table's title FIELD was cleared
    by the human, not just "still whatever this row already had" --
    build_simple_spec_prompt must not just reuse existing_spec's on-disk
    title regardless of what the browser form actually showed, since
    there's otherwise no way to tell "human cleared this on purpose,
    wants a genuinely new idea" apart from "field just happens to be
    blank because nothing's been generated yet." An explicit instruction
    substituted into $title itself asks the model
    to invent an all-new title/premise, ignoring both the existing spec
    AND the master list entry (which is itself just the OLD idea for
    this slot) -- see write_row_spec's own master-list sync for how the
    list gets updated with whatever the model actually invents."""
    if not title_locked:
        return _INVENT_NEW_TITLE_PLACEHOLDER, random.choice(style_options)
    if existing_spec:
        title = existing_spec.get("title") or "an animal doing something unexpected"
        positive = existing_spec.get("positive_prompt") or ""
        style = next((s for s in style_options if s in positive), None) \
            or random.choice(style_options)
        return title, style
    if master_list_entry:
        # "Tale #12: The Title Here -- A premise sentence..."
        after_colon = master_list_entry.split(":", 1)[-1].strip()
        title = after_colon.split("—")[0].strip() or after_colon
    else:
        title = "an animal doing something unexpected"
    return title, random.choice(style_options)


def build_simple_spec_prompt(number, note=None, workflow=None, title_locked=True):
    """The minimal spec-generation prompt for strong creative backends
    (Gemini API) -- a bare, hand-written template with no dedup-as-JSON,
    no reviewed-examples few-shot, no schema-hint prose, which produces
    funnier, more committed writing than a heavier JSON-context payload.

    This exact section layout is Gemini's OWN preference: shown its
    prompt, asked to critique it, then asked to rebuild it per its own
    critique and further tighten that rebuild. The dedup list excludes
    THIS spec's own current title -- including it would tell the model
    not to repeat the very thing it was just asked to write.

    $rules is golden_rules_body() -- the single consolidated rules file
    (see its own header) -- so every mechanical/render-quality fix this
    pipeline needs rides along here too, not just the
    motion-continuity note. $genre/$style come from THIS project's own
    CREATIVE.md (project_genre_and_styles()) -- the same file the
    Creative tab edits -- so that page actually controls what gets sent,
    not a hardcoded pipeline constant. The template ITSELF (the section
    layout/wording below this docstring) is also just the DEFAULT --
    project_prompt_template() reads the actual live copy from this
    project's own CREATIVE.md (seeded there at project creation from
    SPEC_PROMPT_TEMPLATE_PATH), so the wording/structure is human-
    editable per project too, not fixed in code.

    $direction is a PER-CALL note (only this one regen); $concept_directive
    is this PROJECT's standing directive (project_concept_directive(), the
    Creative tab's "Concept directive" field) -- applied to every story in
    this project, every call, until changed. Both are appended
    independently so either, neither, or both can be in effect at once."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing_spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
    workflow = workflow or (existing_spec or {}).get("workflow") or "fml2v"
    concept_entry = master_list_concept_entry(number)
    genre, style_options = project_genre_and_styles()
    title, style = _simple_prompt_title_and_style(existing_spec, concept_entry, style_options, title_locked)
    _, _, duration = project_render_settings()
    direction = f"\n\nCreative direction for this one: {note}" if note else ""
    concept_directive_text = project_concept_directive()
    if concept_directive_text:
        direction += f"\n\nStanding creative directive for every story in this project: {concept_directive_text}"
    # Plain title list, not the old full title+premise JSON blob -- enough
    # to dodge a repeat animal/role pairing without dragging the verbosity
    # the side-by-side test showed actively hurt the writing back in.
    # Excludes THIS number's own current title (see docstring) -- telling
    # the model not to repeat the very thing it's about to write.
    recent_titles = [e["title"] for e in recent_titles_for_dedup()
                      if e.get("title") and e.get("number") != number]
    # When inventing a genuinely new idea (title_locked=False), also
    # exclude THIS row's own previous title -- otherwise nothing stops
    # the model from just re-writing the same idea it's meant to replace,
    # since recent_titles_for_dedup only ever excludes OTHER numbers.
    if not title_locked and existing_spec and existing_spec.get("title"):
        recent_titles = [existing_spec["title"]] + recent_titles
    exclusions = "\n".join(recent_titles) if recent_titles else "(none yet)"
    duration_timestamp = f"{duration // 60:02d}:{duration % 60:02d}"
    negative_baseline = golden_negative_baseline()
    style_negative = style_negative_terms(style)
    if style_negative:
        negative_baseline = f"{negative_baseline}, {style_negative}"

    # golden_rules_body() itself contains a $duration_timestamp
    # placeholder (see its own "Timestamps" bullet) -- string.Template
    # does NOT recursively substitute inside an already-substituted
    # value, so that placeholder needs its own pass here BEFORE being
    # embedded as the outer template's `rules` value, or it would reach
    # the model as literal, un-filled-in "$duration_timestamp" text.
    rules_text = string.Template(golden_rules_body()).substitute(duration_timestamp=duration_timestamp)

    return string.Template(project_prompt_template()).substitute(
        genre=genre, title=title, duration=duration, style=style, direction=direction,
        rules=rules_text, exclusions=exclusions,
        negative_baseline=negative_baseline, duration_timestamp=duration_timestamp,
    )


def lean_spec_instructions(note, concept_entry, reviewed_examples, locked_fields=None):
    """Minimal instructions for a strong creative_backend (Gemini
    API) -- piling on CREATIVE.md's full rule
    set PLUS an explicit "you have creative freedom" caveat on top of it
    is more context than a genuinely strong model needs, and risks the
    model tripping over contradicting/redundant instructions rather
    than just writing a good story. This intentionally sends far LESS
    than the Ollama path: no
    CREATIVE.md dump, no numbered rules, no labeled-draft process -- just
    the required format (already in schema_hint), the human's own
    direction if given, and reviewed_examples as few-shot style/tone
    reference (the actual proven pattern: a bare template + real
    examples + "make it funny" outperformed heavy prescriptive rules)."""
    parts = []
    if note:
        parts.append(f"Creative direction for this one: {note}")
    if locked_fields:
        parts.append("locked_fields are already final (human-written) -- don't repeat "
                      "them, just stay consistent with them, and only write the keys "
                      "listed in schema_hint.")
    if concept_entry and not note:
        parts.append("Use master_list_entry's animal/role, don't invent a different one.")
    parts.append("Check recent_titles_for_dedup so this doesn't repeat a recent one.")
    if reviewed_examples:
        parts.append("reviewed_examples are real Tales from this channel that already "
                      "shipped -- use them as your style/tone/format reference.")
    parts.append("Write a genuinely funny, well-paced Tale filling the schema_hint format, "
                  f"with the final timestamp landing at EXACTLY {_target_duration_timestamp()} "
                  "(the rendered clip is this fixed length -- anything shorter renders as "
                  "dead air at the end -- do not stop early).")
    return " ".join(parts)


def creative_guidance_pointer():
    """This project's own CREATIVE.md -- its creative STYLE (tone, subject
    matter, content modes, worked examples, dedup rules, etc.), specific
    to this one channel. Mechanical/technical prompt-format rules
    (bracket beats, camera rules, negative-prompt baseline) live in
    format_rules() instead -- shared pipeline-wide, not duplicated per
    project (see format_rules.md).

    The 60000-character cap is a generous sanity cap, not an expected
    truncation point, now that the file is creative-style-only and much
    smaller -- hard-truncating to a low number would silently drop most
    of a channel's own guidance once its CREATIVE.md grows past that."""
    path = DATA_DIR / "CREATIVE.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return text[:60000]


SPEC_SCHEMA_HINT = {
    "title": "string",
    "premise": "string",
    "positive_prompt": "string -- MUST use the [Scene Setup]:/[Timeline & Audio Sync]: structure (see format rules), with at least 2 timestamped beats each carrying Video:/Audio: lines, and Voice A (plus Voice B/C/... only if the content genuinely needs more than one voice) tagging quoted dialogue in most beats",
    "negative_prompt": "string (optional -- a confirmed default is applied if omitted)",
    "description": "string, 1-2 sentence summary only, no channel boilerplate",
    "tags": "single comma-separated STRING, e.g. 'loris,night,alibi' -- never a JSON array",
    "workflow": "'fp8_t2v' (default) | 'i2v' | 'fml2v'",
}

# Swapped in for "positive_prompt" when using_strong_creative_backend() --
# reviewed_examples' own full positive_prompt text already demonstrates
# the exact [Scene Setup]/[Timeline & Audio Sync] structure via real
# examples, so restating it as prescriptive prose
# on top is redundant context a strong model doesn't need (same reasoning
# as lean_spec_instructions). Landing near 20-24s stays explicit since
# that's not something a model can infer from reading examples alone.
def _target_duration_timestamp():
    _, _, duration_s = project_render_settings()
    return f"{duration_s // 60:02d}:{duration_s % 60:02d}"


def lean_positive_prompt_hint():
    return ("string -- follow the exact [Scene Setup]/[Timeline & Audio "
            "Sync] structure shown in reviewed_examples' own positive_prompt. "
            f"The final beat's end timestamp MUST be exactly {_target_duration_timestamp()} "
            "(the actual rendered clip is this fixed length regardless of the "
            "scripted timeline -- anything shorter renders as dead air)")


# Fields the model is NEVER asked for and NEVER allowed to return -- these
# are facts about real files/decisions, owned entirely by code or a direct
# human answer, not creative content. Trusting the model to correctly echo
# back a real file path is fragile -- if that path is ever missing from
# its context, it will fabricate a plausible-looking fake (e.g.
# "existing_reference_photo.jpg") rather than say it doesn't know. Removing
# these fields from the model's schema entirely, and having the SCRIPT set
# them after the fact, makes that failure mode structurally impossible: the
# model is never in a position to invent a path because it's never asked
# for one.
CODE_OWNED_SPEC_FIELDS = ("number", "workflow", "image_path",
                           "fml2v_first_image", "fml2v_middle_image", "fml2v_last_image")


def find_concept_list_path():
    """The project's master concept-list file -- same filename in every
    project, differentiated only by DATA_DIR (which project it's in)."""
    return DATA_DIR / "concepts.md"


def concept_list_stats():
    """(path, total_entries, remaining_unspecced) for the project's master
    concept list -- remaining = entries whose number has no spec_NNN.json
    yet. Returns 0s if the list doesn't exist yet."""
    path = find_concept_list_path()
    if not path.exists():
        return path, 0, 0
    text = path.read_text(encoding="utf-8")
    numbers = [int(m) for m in re.findall(r"(?m)^Tale #(\d+):", text)]
    specced = {int(p.stem.split("_")[1]) for p in DATA_DIR.glob("spec_*.json")}
    remaining = [n for n in numbers if n not in specced]
    return path, len(numbers), len(remaining)


def build_spec_request_payload(number, note=None, workflow=None):
    """The full context+instructions payload for elaborating one spec --
    shared by write_row_spec (the manage table's real AI-generation path)
    and --interactive, both rendering it straight into a completion
    prompt, no file needed. Regen vs new
    is decided automatically -- a spec that already exists gets its
    current content embedded (so the model edits/regenerates it in
    context), one that doesn't gets the master-list entry (if any) plus
    dedup context instead. One shape either way -- there are no separate
    "new" and "regen" flags.

    note: optional free-text creative direction from the human, embedded
    directly in the payload. Exists because relying on a model to
    remember something said earlier in conversation is exactly the kind
    of unenforced, easy-to-lose input this was built to replace with
    something explicit.

    workflow: the graph type for this spec, decided by the human (or kept
    from the existing spec) BEFORE this payload is built -- never decided
    by the model. See CODE_OWNED_SPEC_FIELDS."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing_spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
    workflow = workflow or (existing_spec or {}).get("workflow")
    concept_entry = master_list_concept_entry(number)

    # Giving the model the OLD full creative content (premise/
    # positive_prompt/etc) alongside a human_direction note makes it
    # just copy the old content almost verbatim and ignore the note --
    # a small model anchors hard on a large, already-well-formed block
    # of text sitting right next to a much shorter instruction. So when
    # a note is given, the old creative content isn't shown at all
    # (nothing to anchor on) -- without a note (pure "fix what's wrong"
    # regen), it is shown, since the model genuinely needs to see the
    # old content to know what to fix. Either way, CODE_OWNED_
    # SPEC_FIELDS are stripped out entirely -- the model never sees or
    # returns them.
    if note and existing_spec:
        existing_spec_for_prompt = None
    elif existing_spec:
        existing_spec_for_prompt = {k: v for k, v in existing_spec.items()
                                     if k not in CODE_OWNED_SPEC_FIELDS}
    else:
        existing_spec_for_prompt = None

    schema_hint = {k: v for k, v in SPEC_SCHEMA_HINT.items() if k not in CODE_OWNED_SPEC_FIELDS}
    if using_strong_creative_backend():
        schema_hint["positive_prompt"] = lean_positive_prompt_hint()
    if workflow in TYPE_SPECIFIC_FIELDS:
        for field in TYPE_SPECIFIC_FIELDS[workflow]:
            if field in CODE_OWNED_SPEC_FIELDS:
                continue  # code-owned (a real file path), never part of the model's schema
            # fml2v_keyframe_prompts: skip asking for it at all once three
            # real reference images already exist -- write_row_keyframes
            # itself treats that as "nothing to write" (all-or-nothing on
            # the prompt side), so asking the model here too is pure
            # waste, and it doesn't reliably return the required
            # {"first"/"middle"/"last"} object shape when asked without
            # this skip, since nothing here tells it that shape is
            # required -- it would write one flat string instead, which
            # crashes every later read of this row (get_manage_row's
            # kf.get(...)).
            if field == "fml2v_keyframe_prompts" and fml2v_images_satisfied(number):
                continue
            # These are still-image T2I/I2I prompts (see generate_keyframes),
            # not video content. Wording that describes ANIMATION produces
            # motion verbs a single still frame can't depict ("stalking
            # across the snow"), which then fails its own post-generation
            # review every attempt, no matter how many retries run. Matches
            # build_keyframes_request_payload's schema_hint (the K chip's
            # own path to this same field) -- still POSE, never animation.
            schema_hint[field] = (f"required for workflow={workflow!r} -- an OBJECT "
                                   f"with keys \"first\", \"middle\", \"last\" (never a "
                                   f"plain string). Each is a STILL-IMAGE description of "
                                   f"that beat's held pose (appearance/pose/setting, NOT "
                                   f"motion or action -- e.g. 'crouched low, weight "
                                   f"forward' not 'stalking across the snow'); \"middle\"/"
                                   f"\"last\" as a delta off \"first\": 'Maintain "
                                   f"everything, but make X happen'. Write this FRESH to "
                                   f"match the new story/direction.")

    reviewed_examples = reviewed_spec_examples()
    strong_backend = using_strong_creative_backend()
    trend_context = _quiet_spec_trend_context()
    trend_clause = _spec_trend_clause(trend_context)

    payload = {
        "workflow": workflow,
        "mode": "regen" if existing_spec else "new",
        "human_direction": note,
        "existing_spec": existing_spec_for_prompt,
        "master_list_entry": concept_entry,
        "recent_titles_for_dedup": recent_titles_for_dedup(),
        "reviewed_examples": reviewed_examples or None,
        "trend_context": trend_context,
        "schema_hint": schema_hint,
    }
    if strong_backend:
        payload["instructions"] = lean_spec_instructions(note, concept_entry, reviewed_examples) + trend_clause
        return payload

    payload["creative_guidance"] = creative_guidance_pointer()
    payload["instructions"] = (
        (f"THE HUMAN GAVE THIS EXACT CREATIVE DIRECTION -- YOUR ANSWER MUST BE "
         f"BUILT FROM IT, not from anything else below: {note!r}\n\n"
         if note else "") +
        f"Write the spec's CREATIVE content as a JSON object matching schema_hint "
        f"above -- only the keys listed there, nothing else. The graph type "
        f"(workflow={workflow!r}) and any file paths are already decided and set "
        f"by the code, not something you're asked for or should mention. "
        + (f"existing_spec is empty here on purpose -- write everything fresh from "
           f"the direction above, don't try to guess or reuse old wording you "
           f"haven't been shown.\n\n" if note else
           f"If existing_spec is set, this is a REGEN -- keep whatever's still "
           f"good, fix what was wrong.\n\n") +
        f"If master_list_entry is set and there's no human direction above, use "
        f"its exact animal/role -- don't invent a different concept. If neither "
        f"is set, originate one following creative_guidance, checking "
        f"recent_titles_for_dedup for near-duplicates. "
        + (f"reviewed_examples are Tales a human actually approved -- the real bar, "
           f"not just what's been drafted recently. Use them for two things: don't "
           f"repeat an animal+role pairing or joke-type that already shipped there, "
           f"and match their comedic tightness (short, quotable punchline lines; a "
           f"specific voice played with real commitment, not flat/hedging "
           f"exposition)." if reviewed_examples else
           f"reviewed_examples is empty -- nothing approved yet for this channel, so "
           f"rely on creative_guidance alone.")
        + trend_clause
    )
    return payload


ROW_SPEC_FIELDS = tuple(k for k in SPEC_SCHEMA_HINT if k != "workflow")


def _quiet_spec_trend_context():
    """Shared by build_row_spec_payload and build_spec_request_payload --
    the config-driven, always-on-if-enabled trend lookup for creative-spec
    generation (distinct from concepts' explicit, per-request use_trends
    checkbox). Reads config.json's spec_trend_mode_enabled/
    spec_trend_include_script_excerpts; returns None whenever the setting
    is off OR no analytics data exists yet for the current project --
    NEVER raises, since this is meant to sit on permanently without
    requiring every project to have analytics set up first."""
    config = load_config()
    if not config.get("spec_trend_mode_enabled"):
        return None
    if not DREAMS_ROOT:
        return None
    return build_trend_context(
        DREAMS_ROOT.name, trend_projects=None,
        include_script_excerpts=bool(config.get("spec_trend_include_script_excerpts")))


def _spec_trend_clause(trend_context):
    """Shared instruction text for both spec-generation payload builders
    -- deliberately framed as STYLE/TAG SIGNAL ONLY: this must never let
    the model redirect or override the row's own concept (locked_fields,
    master_list_entry), which are fixed inputs decided before this trend
    data is even looked up."""
    if not trend_context:
        return ""
    return (
        f" trend_context lists this channel's own real top-performing video "
        f"titles/tags (and, when available, their premise/script excerpt) from "
        f"YouTube Analytics -- use it ONLY as style/tag/word-choice signal for "
        f"whatever you're already writing. It must NEVER change, redirect, or "
        f"merge into this row's own concept -- locked_fields/master_list_entry "
        f"above are the fixed subject of this specific video and always win."
    )


def build_row_spec_payload(number, locked_fields, note, workflow):
    """Field-level variant of build_spec_request_payload for the web UI's
    manage table: instead of an all-or-nothing "note vs existing content"
    choice, the caller already knows exactly which fields the human typed
    a value into directly this row (locked_fields) -- those are used
    verbatim, fed to the model as fixed context it must not contradict,
    and REMOVED from schema_hint entirely. Every base field the human left
    blank goes into schema_hint for the model to compose. Returns None if
    there's nothing left for the model to do (every base field already
    locked) -- the caller should skip the AI call entirely in that case.

    Also quietly attaches trend_context when config.json's
    spec_trend_mode_enabled is on and this project has analytics data --
    see _quiet_spec_trend_context. Off by default, and framed strictly as
    style signal, never allowed to override the concept -- see
    _spec_trend_clause."""
    schema_hint = {k: v for k, v in SPEC_SCHEMA_HINT.items()
                   if k not in CODE_OWNED_SPEC_FIELDS and k not in locked_fields}
    if using_strong_creative_backend() and "positive_prompt" in schema_hint:
        schema_hint["positive_prompt"] = lean_positive_prompt_hint()
    if workflow in TYPE_SPECIFIC_FIELDS:
        for field in TYPE_SPECIFIC_FIELDS[workflow]:
            if field in CODE_OWNED_SPEC_FIELDS or field in locked_fields:
                continue
            # See build_spec_request_payload's identical check -- three
            # real images already satisfy fml2v, so asking for this too
            # is both unnecessary and (confirmed 2026-08-09) prone to the
            # model returning a flat string instead of the required
            # {"first"/"middle"/"last"} object, which crashes every later
            # read of this row.
            if field == "fml2v_keyframe_prompts" and fml2v_images_satisfied(number):
                continue
            # See build_spec_request_payload's identical fix -- still
            # POSE, never animation (a single frame can't depict motion).
            schema_hint[field] = (f"required for workflow={workflow!r} -- an OBJECT "
                                   f"with keys \"first\", \"middle\", \"last\" (never a "
                                   f"plain string). Each is a STILL-IMAGE description of "
                                   f"that beat's held pose (appearance/pose/setting, NOT "
                                   f"motion or action -- e.g. 'crouched low, weight "
                                   f"forward' not 'stalking across the snow'); \"middle\"/"
                                   f"\"last\" as a delta off \"first\": 'Maintain "
                                   f"everything, but make X happen'. Write this FRESH to "
                                   f"match the story, consistent with locked_fields below.")
    if not schema_hint:
        return None

    concept_entry = master_list_concept_entry(number) if "title" not in locked_fields else None
    reviewed_examples = reviewed_spec_examples()
    strong_backend = using_strong_creative_backend()
    trend_context = _quiet_spec_trend_context()
    trend_clause = _spec_trend_clause(trend_context)

    payload = {
        "workflow": workflow,
        "human_direction": note,
        "locked_fields": locked_fields or None,
        "master_list_entry": concept_entry,
        "recent_titles_for_dedup": recent_titles_for_dedup(),
        "reviewed_examples": reviewed_examples or None,
        "trend_context": trend_context,
        "schema_hint": schema_hint,
    }
    if strong_backend:
        payload["instructions"] = lean_spec_instructions(
            note, concept_entry, reviewed_examples, locked_fields=locked_fields) + trend_clause
        return payload

    payload["creative_guidance"] = creative_guidance_pointer()
    payload["instructions"] = (
        (f"THE HUMAN GAVE THIS EXACT CREATIVE DIRECTION -- YOUR ANSWER MUST BE "
         f"BUILT FROM IT: {note!r}\n\n" if note else "") +
        f"locked_fields are already FINAL, written by the human directly -- "
        f"do not repeat them in your answer, only write the keys listed in "
        f"schema_hint, and make sure your answer is consistent with locked_fields "
        f"(e.g. if locked_fields has a title, your positive_prompt must match that "
        f"story, not a different one). "
        + (f"If master_list_entry is set and there's no human direction above, use "
           f"its exact animal/role -- don't invent a different concept. " if concept_entry else "") +
        f"Check recent_titles_for_dedup for near-duplicates. "
        + (f"reviewed_examples are the human-approved bar (Tales actually accepted, "
           f"not just drafted) -- avoid repeating their animal+role pairings/joke-"
           f"types, and match their comedic tightness (short, quotable punchlines; a "
           f"committed specific voice). " if reviewed_examples else
           f"reviewed_examples is empty -- nothing approved yet for this channel. ") +
        f"Follow creative_guidance."
        + trend_clause
    )
    return payload


def write_row_spec(number, workflow, fields, note, verbose=False):
    """One manage-table row's spec write -- fields is every editable base
    field as currently shown in the table (blank string for "not filled
    in"). Non-blank fields are locked verbatim; blank ones are always
    AI-composed -- there is no manual on/off switch. If every base field
    is already non-blank, nothing is generated and this degenerates to a
    plain verbatim save (see the `payload is None` branch below).

    do_write_spec does a full overwrite (spec_path.write_text), so
    building the new spec dict from scratch (just naive_locked_fields +
    code_owned) would silently drop any field not in ROW_SPEC_FIELDS or
    CODE_OWNED_SPEC_FIELDS (e.g. fml2v_guide_strengths, saved separately
    via its own per-slot weight input) the instant "Save content" ran for
    any other reason. Starting from the existing on-disk spec and
    layering locked fields + code_owned on top preserves anything this
    function doesn't itself know or care about."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing_on_disk = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    naive_locked_fields = {k: v for k, v in fields.items() if k in ROW_SPEC_FIELDS and (v or "").strip()}
    code_owned, error = determine_code_owned_spec_fields(number, workflow)
    if error:
        raise SystemExit(error)

    # With a note given, only a field that actually DIFFERS from what's on
    # disk counts as human-locked -- otherwise every pre-filled, untouched
    # field looks "locked" and the note gets silently ignored.
    if note:
        locked_fields = {k: v for k, v in naive_locked_fields.items()
                          if v.strip() != (existing_on_disk.get(k) or "").strip()}
    else:
        locked_fields = naive_locked_fields

    payload = build_row_spec_payload(number, locked_fields, note, workflow)
    if payload is None:
        spec = dict(existing_on_disk)
        spec.update(locked_fields)
        spec.update(code_owned)
        # payload is None means schema_hint ended up empty -- every base
        # field is in locked_fields, whether because AI is genuinely done
        # (no note: any non-blank field counts) or because the human
        # explicitly edited/typed every single one (a note: only dirty-
        # vs-disk fields count, so this only triggers here if ALL of them
        # were actually touched) -- either way, 100% human-authored.
        do_write_spec(number, json.dumps(spec), positive_prompt_is_human=True)
        return
    # The new minimal template (build_simple_spec_prompt) always asks for
    # every base field at once -- no mechanism to hold some locked -- so
    # it only replaces the old JSON-context prompt when nothing's locked
    # (a genuine full write/regen), the common case this was tested
    # against. A partial edit (some fields human-typed, others blank)
    # still needs the old payload, which DOES support that.
    if using_strong_creative_backend() and not locked_fields:
        prompt = build_simple_spec_prompt(number, note, workflow, title_locked="title" in locked_fields)
    else:
        prompt = _render_creative_prompt(payload)
    if verbose:
        print(f"[dream_step] #{number}: prompt sent to the model:\n{prompt}\n")
    ok = _generate_and_write_spec(number, prompt, code_owned, extra_locked_fields=locked_fields, verbose=verbose)
    if not ok:
        raise SystemExit(f"[dream_step] #{number}: the AI couldn't produce a spec that "
                          f"passed validation after several tries -- see the attempts "
                          f"above for exactly what it got wrong. Try a clearer creative "
                          f"direction, or fill in the field(s) yourself instead.")


def write_row_keyframes(number, workflow, fields, verbose=False):
    """One manage-table row's keyframe-prompt write. `fields` carries
    whatever prompt text the human typed for slots that don't already have
    an image (uploaded or found): i2v -- {"i2v_generate_image_prompt": ...};
    fml2v -- {"first":..., "middle":..., "last":...}. A row whose images
    already fully satisfy its workflow (1 for i2v, 3 named 1/2/3 for fml2v)
    needs nothing written here at all -- the image alone is enough.
    Non-blank fields save verbatim; any still-blank prompt needed is
    always AI-composed -- no manual on/off switch.

    fml2v is all-or-nothing on the prompt side: check_image_prerequisites
    only accepts three real images OR the complete fml2v_keyframe_prompts
    object, never a partial mix -- so unless all three images are already
    present, all three prompts are needed (each individually still locked
    verbatim if the human typed it; the model only fills the rest.

    Whenever this function is the one actually composing a first-frame
    prompt (never when the human typed it verbatim), it sets
    "first_frame_source": "online" on the spec, so a real CC0 photo seeds
    the same T2I/I2I graph instead of the blank placeholder -- fixes the
    species accuracy an AI-authored image prompt alone can't guarantee."""
    images = find_reference_images(number)

    if workflow == "i2v":
        text = (fields.get("i2v_generate_image_prompt") or "").strip()
        if len(images) == 1:
            if not text:
                return  # already satisfied, nothing new to record
            # Same stale-image-vs-rewritten-story fix as the fml2v branch
            # below -- fresh prompt text arriving while an image already
            # exists means an explicit new one was composed for this save.
            images[0].unlink(missing_ok=True)
            images = []
        if text:
            merge_and_write_spec(number, {"i2v_generate_image_prompt": text})
            return
        payload = build_keyframes_request_payload(number, 1)
        if payload is None:
            raise SystemExit(f"[dream_step] #{number}: no spec exists yet.")
        ok = _generate_and_write_keyframes(
            number, _render_creative_prompt(payload), verbose=verbose,
            extra_locked_fields={"first_frame_source": "online"})
        if not ok:
            raise SystemExit(f"[dream_step] #{number}: the AI couldn't produce an "
                              f"image prompt that passed validation -- see the attempts "
                              f"above. Try a clearer description, or type one in yourself.")
        return

    if workflow == "fml2v":
        typed = {k: (fields.get(k) or "").strip() for k in ("first", "middle", "last")
                 if (fields.get(k) or "").strip()}
        if len(images) == 3 and {p.stem for p in images} == {"1", "2", "3"}:
            if not typed:
                return  # already satisfied, nothing new to record
            # A story rewrite that leaves stale keyframe images in place
            # would otherwise silently drop freshly-typed/composed
            # keyframe text right here -- fields is never even looked at
            # once images satisfy the workflow, so the spec would keep
            # pointing at images from the OLD story with no record the
            # human/AI had just written new prompts for it. Any
            # real text arriving here means an explicit new set was
            # composed for this save -- delete the stale images so they
            # can't keep being silently reused, same as the manual
            # delete-image-then-retype-then-save GUI workaround this
            # mirrors, and fall through to the normal write path below.
            for img in images:
                img.unlink(missing_ok=True)
            images = []
        locked = typed
        if len(locked) == 3:
            updates = {"fml2v_keyframe_prompts": locked}
            # Deleting the FILES isn't enough on its own -- the spec's own
            # fml2v_first_image/middle/last fields (set by a prior render
            # to the old images' paths) survive a merge untouched
            # otherwise, since merge_and_write_spec only
            # overwrites keys actually present in `updates`. The next
            # render then tries to load a path that no longer exists and
            # fails outright ("spec's fml2v_first_image does not exist")
            # instead of falling back to generating fresh images from
            # fml2v_keyframe_prompts. Clear all three explicitly whenever
            # they don't match what's currently on disk.
            spec_path = DATA_DIR / f"spec_{number:03d}.json"
            if spec_path.exists():
                existing = json.loads(spec_path.read_text(encoding="utf-8"))
                for field in ("fml2v_first_image", "fml2v_middle_image", "fml2v_last_image"):
                    if existing.get(field) and not resolve_stored_rel_path(DREAMS_ROOT, existing[field]).exists():
                        updates[field] = None
            merge_and_write_spec(number, updates)
            return
        payload = build_keyframes_request_payload(number, 3)
        if payload is None:
            raise SystemExit(f"[dream_step] #{number}: no spec exists yet.")
        if locked:
            # Same verbatim-wins pattern as the spec row: only ask the
            # model for whichever of first/middle/last isn't already locked.
            payload["schema_hint"]["fml2v_keyframe_prompts"] = {
                k: v for k, v in payload["schema_hint"]["fml2v_keyframe_prompts"].items()
                if k not in locked}
            payload["instructions"] += (f" first/middle/last already locked by the human: "
                                         f"{list(locked)} -- write ONLY the remaining "
                                         f"sub-key(s) shown in schema_hint's "
                                         f"fml2v_keyframe_prompts, consistent with those.")
        prompt = _render_creative_prompt(payload)
        extra_locked_fields = {"fml2v_keyframe_prompts": locked} if locked else {}
        extra_locked_fields["first_frame_source"] = "online"
        ok = _generate_and_write_keyframes(number, prompt, verbose=verbose,
                                            extra_locked_fields=extra_locked_fields)
        if not ok:
            raise SystemExit(f"[dream_step] #{number}: the AI couldn't produce keyframe "
                              f"prompts that passed validation -- see the attempts above. "
                              f"Try clearer descriptions, or type them in yourself.")
        return


def resolve_slot_image_lenient(number, workflow, slot):
    """The actual reference-image file (if any) currently sitting in one
    manage-table image slot -- used by the web UI to serve a thumbnail.
    None if that slot isn't satisfied yet. Per-slot independent: an fml2v
    Dream with only 2 of 3 keyframe images still shows whichever ones
    actually exist, rather than requiring a complete triple.

    An all-or-nothing version (only counting a slot as filled once all
    three fml2v images are present, matching the render-time rule that
    fml2v needs a complete triple) would mean deleting just the 'first'
    slot's image (to force a regen after a story rewrite) makes the OTHER
    TWO still-good images ('middle'/'last', 2.png/3.png, physically still
    on disk) vanish from the manage table entirely -- nothing to look at,
    reuse, or manually reassign to a different slot. "Only show once
    render-ready" is right for deciding whether a render can proceed,
    wrong for deciding what a human editing this row gets to see -- so
    this lookup is the only one the manage table uses; render-readiness
    is checked separately where it actually matters."""
    if workflow == "i2v":
        images = find_reference_images(number)
        return images[0] if len(images) == 1 else None
    if workflow != "fml2v":
        return None
    stem = {"first": "1", "middle": "2", "last": "3"}.get(slot)
    if stem is None:
        return None
    for folder_name in existing_dream_folders(number):
        folder = DREAMS_ROOT / folder_name
        matches = sorted(p for p in folder.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if matches:
            return matches[0]
    return None


def rename_slot_image(number, workflow, from_slot, to_slot):
    """Reassigns which keyframe SLOT an already-existing image file
    represents -- e.g. reusing a 'middle' pose that already happens to
    match the new story's 'first' beat, instead of spending a fresh AI
    image generation on a pose that already exists elsewhere in this
    same Dream (2026-08-12: a real case -- a story rewrite changed the
    beat order but two of the three poses were still usable, just in the
    wrong slots). Only meaningful for fml2v (i2v has one slot, nothing to
    reassign against). SWAPS with whatever's currently in to_slot rather
    than overwriting it, so this can never silently destroy an existing
    image -- worst case a human runs it twice to swap back. No-op
    (returns False) if from_slot has nothing to move."""
    if workflow != "fml2v":
        raise SystemExit("[dream_step] rename_slot_image only applies to fml2v (first/middle/last).")
    if from_slot == to_slot:
        return False
    src = resolve_slot_image_lenient(number, workflow, from_slot)
    if src is None:
        return False
    dst = resolve_slot_image_lenient(number, workflow, to_slot)
    to_stem = {"first": "1", "middle": "2", "last": "3"}[to_slot]
    from_stem = {"first": "1", "middle": "2", "last": "3"}[from_slot]
    if dst is not None:
        # Swap via a temp name -- can't rename src->dst directly while
        # dst's own file still occupies that name.
        tmp = dst.with_name(f"_swap_tmp{dst.suffix}")
        dst.rename(tmp)
        src.rename(src.with_name(f"{to_stem}{src.suffix}"))
        tmp.rename(tmp.with_name(f"{from_stem}{dst.suffix}"))
    else:
        src.rename(src.with_name(f"{to_stem}{src.suffix}"))
    # The moved (and, if swapped, the displaced) file(s) may now sit at a
    # DIFFERENT path than whatever the spec's fml2v_*_image fields say --
    # clear them so the next render/rework re-derives them fresh from
    # disk (find_reference_images/check_image_prerequisites already do
    # this auto-repoint, see do_rework's own comment on the same pattern)
    # instead of pointing at a filename that no longer holds that pose.
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        changed = False
        for field in ("fml2v_first_image", "fml2v_middle_image", "fml2v_last_image"):
            if spec.get(field):
                spec[field] = ""
                changed = True
        if changed:
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    # The Dream folder's _fml2v_keyframe_prompts.json sidecar (see
    # generate_dream.py's generate_keyframes) tracks "this image FILE was
    # generated from this PROMPT TEXT," keyed by role name -- after a
    # rename/swap, that mapping is just wrong (the file sitting at '1.png'
    # is no longer whatever 'first' used to mean), and left stale it would
    # make role_changed('first') compare the NEW first-slot prompt against
    # the OLD role's recorded text, see them differ, and trigger a full
    # regeneration of every role -- silently undoing the whole point of
    # reassigning slots to avoid unnecessary regeneration. Deleting it
    # instead falls through to that function's own "no sidecar entry ->
    # trust the existing image as-is" rule, which is exactly what a
    # reassignment means: these files are good, don't touch them again
    # until a human's own edited prompt text says otherwise.
    for folder_name in existing_dream_folders(number):
        sidecar = DREAMS_ROOT / folder_name / "_fml2v_keyframe_prompts.json"
        if sidecar.exists():
            sidecar.unlink()
    return True


def master_list_concept_entry(number):
    """The raw 'Tale #N: Title — A animal role. "line"' line from this
    project's master concept list for `number`, or None if there isn't
    one / no list exists yet."""
    concept_list_path = find_concept_list_path()
    if not concept_list_path.exists():
        return None
    text = concept_list_path.read_text(encoding="utf-8")
    m = re.search(rf"(?m)^Tale #{number}:.*$", text)
    return m.group(0) if m else None


def parse_concept_entry(entry_text):
    """One master-list line -> (title, premise draft). The premise here is
    just the animal/role description + sample line stitched together, a
    starting point for the human to flesh out or for the AI to expand into
    a real positive_prompt -- not a finished spec premise itself."""
    m = re.match(r'^Tale #\d+:\s*(.+?)\s*—\s*(.+?)\.\s*"(.*)"\s*$', entry_text)
    if not m:
        return entry_text, ""
    title, description, line = m.groups()
    # description is already a full sentence (older entries start with "A
    # ..." themselves, newer ones from commit_concepts_response are just
    # "animal role") -- don't assume which, just append the line as-is.
    return title, f"{description}. \"{line}\""


def get_manage_row(number):
    """One manage-table row's full current state -- the web UI's single
    source of truth for what to pre-fill, built entirely from real files
    (spec_NNN.json, index.json, reference images), never guessed.

    Image status is reported independent of the spec's CURRENTLY STORED
    workflow (single-image-present / triple-image-present), since
    find_reference_images itself doesn't care what workflow is selected --
    this lets the web UI switch its graph-type dropdown client-side and
    immediately show the right slot state without a round trip.

    A number with no spec yet, but an entry in the master concept list,
    pre-fills title/premise from that entry instead of loading blank --
    "load 84 which has no files" should still show what's already decided
    for it, not nothing."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    workflow = spec.get("workflow") or "fp8_t2v"
    index = load_json(INDEX_PATH, [])
    entry = next((e for e in index if isinstance(e, dict) and e.get("number") == number), None)

    title, premise, from_concept_list = spec.get("title", ""), spec.get("premise", ""), False
    if not spec:
        concept_entry = master_list_concept_entry(number)
        if concept_entry:
            title, premise = parse_concept_entry(concept_entry)
            from_concept_list = True

    images = find_reference_images(number)
    single_ok = len(images) == 1
    triple_ok = fml2v_images_satisfied(number)
    # find_reference_images is all-or-nothing between "real rendered
    # Dream folder" and "uploads staging" -- it only ever falls back to
    # staging when the real folder returned NOTHING across every slot.
    # So this one flag is enough to tell the web UI whether `images`
    # above (what /slot-image serves as "current") came from a genuine
    # render or is itself just the staged file -- true means any ALSO-
    # staged upload found below is a real alternative worth showing
    # side by side; false means there's nothing to compare it against
    # yet (the "current" thumbnail already IS the staged file).
    # Only counts files actually relevant to the keyframe/reference set
    # this Dream uses -- properly-named keyframe files (stems "1"/"2"/
    # "3") when any exist, or any OTHER image when none of those do (the
    # i2v single-reference case). A flat "any image file in the folder"
    # scan would let an unrelated file sitting there (a debug artifact
    # like _online_reference_seed.png, or a human's own kept-around
    # reference photo) read as "real images present" even with zero
    # actual keyframes rendered yet -- wrong for what this flag exists to
    # answer. Uses the same "is this related to our selected keyframes"
    # test find_reference_images itself uses, not a naming-convention
    # (underscore-prefix) heuristic.
    _all_folder_images = [
        p for folder_name in existing_dream_folders(number)
        for p in (DREAMS_ROOT / folder_name).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    real_images_present = bool(
        [p for p in _all_folder_images if p.stem in ("1", "2", "3")]
        or [p for p in _all_folder_images if not p.name.startswith("_")])
    staged_slots = {slot: staged_upload_path(number, slot) is not None
                     for slot in ("image", "first", "middle", "last")}
    # Per-slot, independent of triple_ok -- lets the manage table show
    # (and offer to reassign) whichever fml2v keyframe images actually
    # exist even when the set isn't complete (see
    # resolve_slot_image_lenient's own docstring for the bug this fixes).
    slot_has_image = {slot: resolve_slot_image_lenient(number, workflow, slot) is not None
                       for slot in ("first", "middle", "last")}
    guide_strengths = spec.get("fml2v_guide_strengths") or {}
    if not isinstance(guide_strengths, dict):
        guide_strengths = {}
    guide_strengths = {
        role: guide_strengths.get(role, default)
        for role, default in _FALLBACK_GUIDE_STRENGTHS.items()
    }

    kf = spec.get("fml2v_keyframe_prompts") or {}
    # Legacy/malformed defense: this must be an object with first/middle/
    # last sub-keys (see do_write_spec's own shape check for new writes)
    # -- a row with a bad value here shouldn't crash every OTHER row's
    # table load too.
    if not isinstance(kf, dict):
        kf = {}

    return {
        "number": number,
        "exists": bool(spec),
        "from_concept_list": from_concept_list,
        "title": title,
        "premise": premise,
        "positive_prompt": spec.get("positive_prompt", ""),
        "negative_prompt": spec.get("negative_prompt", ""),
        "description": spec.get("description", ""),
        "tags": spec.get("tags", ""),
        "workflow": workflow,
        "image_status": {"single": single_ok, "triple": triple_ok},
        "slot_has_image": slot_has_image,
        "real_images_present": real_images_present,
        "staged_slots": staged_slots,
        "i2v_prompt": spec.get("i2v_generate_image_prompt", ""),
        "fml_prompts": {"first": kf.get("first", ""), "middle": kf.get("middle", ""), "last": kf.get("last", "")},
        "guide_strengths": guide_strengths,
        "rendered": entry is not None,
        "uploaded": bool(entry and entry.get("published")),
    }


def build_keyframes_request_payload(number, image_count):
    """Shared by write_row_keyframes (the manage table's real AI-generation
    path) and --interactive. Returns None if no spec exists yet for this
    number. image_count: 1 -> i2v_generate_image_prompt, 3 ->
    fml2v_keyframe_prompts."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    if not spec_path.exists():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if image_count == 1:
        fields_needed = ["i2v_generate_image_prompt"]
        # Without an explicit prohibition, the model tends to copy
        # positive_prompt's own "[Scene Setup]:" section label onto the
        # FRONT of this field too -- harmless to the T2I render itself
        # (just extra text), but sloppy and not what this field is.
        schema = {"i2v_generate_image_prompt": "string -- still-image description "
                  "of the opening frame (appearance/pose/setting, NOT animation). "
                  "Plain descriptive text ONLY -- do NOT prefix it with "
                  "\"[Scene Setup]:\" or any other section label from "
                  "positive_prompt's format; this is a separate, standalone field."}
    else:
        fields_needed = ["fml2v_keyframe_prompts"]
        # "full scene description" alone lets the model write ACTION verbs
        # into "first" ("stalking slowly and cautiously across the snow")
        # -- a single still frame can't actually depict "stalking across,"
        # so the T2I output gets judged a mismatch against its own prompt
        # no matter how many retries run. Same approach i2v's own schema
        # hint uses just below: force a still POSE, not a motion/action
        # description.
        schema = {"fml2v_keyframe_prompts": {
            "first": "string -- still-image description of this frame's held pose "
                     "(appearance/pose/setting, NOT motion or action -- e.g. 'crouched "
                     "low, weight forward, eyes fixed on the snow' not 'stalking across "
                     "the snow')",
            "middle": "delta off first: 'Maintain everything, but make X happen'",
            "last": "delta off first: 'Maintain everything, but make Y happen'"}}
    return {
        "number": number, "current_spec": spec, "image_count": image_count,
        "fields_needed": fields_needed, "schema_hint": schema,
        "instructions": f"Write ONLY {fields_needed} as a JSON object to the "
                         f"response path below, matching schema_hint. The actual "
                         f"image(s) generate automatically the next time this "
                         f"number is rendered (--generate or --rework).",
    }


CHAT_TOOL_BRIEFING = """This is the manage table's built-in help chat. Answer questions about
how the tool works, and where relevant, propose field content for numbers the human already
has loaded in the table.

CONCEPTS:
- A "spec" (spec_NNN.json) is one video's full content: title, premise (internal summary),
  positive_prompt (the actual animation prompt), negative_prompt, description (public-facing
  summary), tags, and workflow (t2v/i2v/fml).
- positive_prompt MUST use the [Scene Setup]:/[Timeline & Audio Sync]: structure (see
  format_rules.md) -- timestamped beats each with Video:/Audio: lines, most beats carrying
  real quoted spoken dialogue tagged with a voice (Voice A alone for one narrator/character;
  Voice B/C/... only if the content genuinely needs more than one distinct voice).
- workflow t2v = pure text-to-video, no reference image needed. i2v = needs exactly 1
  reference image (uploaded or already in the Dream's folder) -- positive_prompt/
  negative_prompt still render (same field as every workflow, no separate i2v-only
  version); since the image already fixes appearance, prefer describing the ANIMATION
  over re-describing appearance. fml = needs exactly 3 reference images (first/middle/
  last) OR a complete fml2v_keyframe_prompts object with all three sub-fields -- never
  a partial mix of some images and some prompts.
- tags and negative_prompt are both short comma-separated lists (pill-style in the UI).

TABLE MECHANICS (field-locking -- the most important rule to explain if asked):
- Any field the human typed content into is used VERBATIM, always -- never touched by AI,
  never inferred to be "probably fine to overwrite."
- Blank fields are AI-composed only if that row's AI chip is ticked (S = spec fields, K =
  keyframe/image prompts). If AI is off and a field is blank, saving it fails with a clear
  error telling the human to fill it in or turn AI on.
- "Run updates" writes spec/keyframe content for selected rows -- it never renders anything.
- "Run video gen" is the separate, explicit, GPU-spending step -- renders for the first time
  if a number has no video yet, or RE-RENDERS (overwriting the current file) if it does.
  Always asks for confirmation first.
- Uploading to YouTube is its own separate tab, never bundled into the above.

YOUR OWN LIMITS: you have no ability to write files, run code, or save anything directly --
you only ever produce chat text and a list of field proposals. A human reviews any proposals
and clicks Apply, then still has to click Run updates for anything to actually be saved."""


def build_chat_payload(project_name, message, history, numbers_context):
    """One turn of the manage table's help chat. numbers_context is the
    comma-separated list of numbers CURRENTLY loaded in the human's table
    -- field proposals are only meaningful for those (a number that isn't
    loaded has no row for the UI to drop a proposal into), so the model is
    told to stick to that set rather than guess at ones it can't see."""
    return {
        "project": project_name,
        "tool_briefing": CHAT_TOOL_BRIEFING,
        "numbers_currently_loaded_in_table": numbers_context or None,
        "conversation_history": history,
        "user_message": message,
        "schema_hint": {
            "reply": "string -- your conversational answer, shown directly to the human",
            "proposals": [{"number": "int -- must be one of numbers_currently_loaded_in_table",
                            "field": "string -- one of: title, premise, positive_prompt, "
                                     "negative_prompt, description, tags, note, type "
                                     "(type's value must be t2v/i2v/fml)",
                            "value": "string"}],
        },
        "instructions": (
            "Answer the user_message, using tool_briefing and conversation_history for "
            "context. If they're just asking a question, reply and leave proposals empty. "
            "If they ask you to draft/compose content for a specific number, only do so if "
            "that number is in numbers_currently_loaded_in_table -- otherwise tell them to "
            "load it into the table first (Number(s) field, Load button), don't invent "
            "content for a row that doesn't exist in their view. Reply with ONLY the JSON "
            "object matching schema_hint, nothing else."
        ),
    }


def chat_with_agent(project_name, message, history, numbers_context, model, model_name=None):
    """Dispatches one chat turn to the local Ollama model -- fast,
    offline, no account/CLI dependency beyond Ollama itself, with real
    web_search/wikipedia_search access via _ollama_tool_completion's
    local tool-calling loop. Never given file-write or code-execution
    tools -- the tool set is hardcoded to just web_search/wikipedia_search
    (see _OLLAMA_SEARCH_TOOLS), a structural guarantee, not just an
    instruction."""
    payload = build_chat_payload(project_name, message, history, numbers_context)
    prompt = _render_creative_prompt(payload)
    response, _history = _ollama_tool_completion(prompt, model=model_name)
    proposals = response.get("proposals")
    return {
        "reply": response.get("reply", ""),
        "proposals": proposals if isinstance(proposals, list) else [],
    }


def guess_animal_query(title):
    """Best-effort common-name extraction from a Tale's title, for
    reference_photo lookups when a spec has no explicit "animal" field
    of its own (true for every spec written before that field existed).
    Titles follow "The <Animal>'s <rest>" almost universally -- falls
    back to the title's first two words when that pattern doesn't match
    (e.g. Tale #3, "Otti Investigations", has no possessive at all),
    which is still a better search term than nothing."""
    if not title:
        return ""
    m = re.match(r"^(?:The\s+)?(.+?)'s\b", title.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return " ".join(title.strip().split()[:2])


def reviewed_spec_examples(limit=3):
    """title/premise/positive_prompt for a RANDOM sample of Tales a human
    has actually moved into DREAMS_ROOT/Reviewed -- i.e. the approved
    bar, not just whatever made it into concepts.md or got a spec
    written. Deduping concept generation only against existing_list_tail
    (the draft master list) can't catch a new concept re-treading an
    animal/role/joke-type that already shipped, and gives the model no
    signal on what tone actually cleared review vs. what merely got
    drafted -- this supplies that signal.

    Randomized rather than most-recent: always citing the same handful of
    most-recent Tales as the style reference risks the model's own output
    converging toward whatever those few happen to be, rather than the
    full breadth of what's actually shipped -- a fresh random draw each
    call keeps the style reference varied across a batch of generations.
    limit=3 keeps the prompt tight -- a strong model needs a few good
    examples, not an exhaustive dump.

    Uses list_media_folders' existing reviewed-vs-active split rather
    than re-deriving it. Returns [] if nothing's been reviewed yet."""
    reviewed_numbers = [e["number"] for e in list_media_folders(PROJECT_DIR.name)
                         if e["location"] == "reviewed" and e["number"] is not None]
    sample_numbers = random.sample(reviewed_numbers, min(limit, len(reviewed_numbers)))
    examples = []
    for number in sample_numbers:
        spec_path = DATA_DIR / f"spec_{number:03d}.json"
        if not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        examples.append({"number": number, "title": spec.get("title"),
                          "premise": spec.get("premise"),
                          "positive_prompt": spec.get("positive_prompt")})
    return examples


def list_projects_with_analytics_data(exclude=None):
    """Every existing project whose YouTube Analytics cache has actually
    been refreshed at least once (fetched_at set) -- used to offer trend-
    mode concept generation the option to pull in other projects' top
    performers alongside this project's own."""
    out = []
    for name in list_existing_projects():
        if name == exclude:
            continue
        cache_path = projects_root() / name / "_data" / "youtube" / "analytics_cache.json"
        if not cache_path.exists():
            continue
        if load_json(cache_path, {}).get("fetched_at"):
            out.append(name)
    return sorted(out)


def _project_top_titles(project_name, n=8, include_script_excerpts=True):
    """Top-performing video titles/tags for one project (current or
    another), or None if that project has no analytics data cached yet.
    Reads the cache file directly by path rather than through the
    module-level DATA_DIR global, so this also works for projects other
    than the currently active one.

    include_script_excerpts controls the heavier enrichment: index.json's
    own "premise" field (durable, never deleted) plus an excerpt of the
    actual rendered script's POSITIVE PROMPT section from that video's
    own .txt file (see generate_dream.write_txt) when that video's render
    folder hasn't been cleaned up -- spec_NNN.json itself is routinely
    deleted after upload, so this durable per-video record is the only
    place genuine past creative writing survives for an already-published
    video. False skips the index.json/.txt reads entirely and returns
    title/tags/views only -- for callers that want a lighter, faster
    check (e.g. spec_trend_mode_enabled's quiet per-generation lookup)."""
    import youtube_analytics
    project_dir = projects_root() / project_name
    cache_path = project_dir / "_data" / "youtube" / "analytics_cache.json"
    if not cache_path.exists():
        return None
    cache = load_json(cache_path, {})
    if not cache.get("fetched_at") or not cache.get("videos"):
        return None
    top = youtube_analytics.top_titles(cache, n=n)

    if include_script_excerpts:
        index = load_json(project_dir / "_data" / "index.json", [])
        entry_by_video_id = {e["youtube_video_id"]: e for e in index if e.get("youtube_video_id")}
        for item in top:
            entry = entry_by_video_id.get(item.pop("video_id", ""))
            if not entry:
                continue
            if entry.get("premise"):
                item["premise"] = entry["premise"]
            folder = entry.get("folder")
            txt_path = project_dir / folder / f"{folder}.txt" if folder else None
            if txt_path and txt_path.exists():
                text = txt_path.read_text(encoding="utf-8")
                marker = "POSITIVE PROMPT:\n\n"
                if marker in text:
                    excerpt = text.split(marker, 1)[1].split("\n\nNEGATIVE PROMPT:", 1)[0]
                    item["script_excerpt"] = excerpt[:1200]
    else:
        for item in top:
            item.pop("video_id", None)

    return {
        "project": project_name,
        "top_titles": top,
        "by_workflow": (cache.get("correlation") or {}).get("by_workflow", [])[:5],
        "by_tag": (cache.get("correlation") or {}).get("by_tag", [])[:5],
    }


def build_trend_context(project_name, trend_projects=None, include_script_excerpts=True):
    """Assembles the trend-mode payload: this project's own top performers
    plus, optionally, top performers from other projects the human
    explicitly opted to include. Returns None if nothing usable was found
    anywhere -- callers that treat trend mode as an explicit, deliberate
    request (concepts) should refuse outright on None rather than
    silently proceeding with no real data behind it; callers that treat
    it as a quiet, always-on enhancement (spec_trend_mode_enabled) should
    just skip trend context entirely on None, no error."""
    names = [project_name] + [p for p in (trend_projects or []) if p != project_name]
    per_project = [ctx for ctx in (_project_top_titles(name, include_script_excerpts=include_script_excerpts)
                                     for name in names) if ctx]
    if not per_project:
        return None
    return {"projects": per_project}


def build_concepts_request_payload(project_name, count, web_search_available,
                                    use_trends=False, trend_projects=None):
    """Shared by every concepts-research caller (the web UI's "Research &
    add ideas" button and the --interactive new-project flow).
    web_search_available controls whether the instructions ask for
    research that can actually happen --
    silently asking a tool-less model to "use web search" would just make
    it fabricate the appearance of having searched. As of 2026-08-07 the
    default dispatch (_ollama_tool_completion) genuinely has web_search/
    wikipedia_search access via Ollama's own tool-calling -- web_search_
    available=True is accurate for every current
    caller; the False branch stays available for any future caller that
    genuinely has no tool access. Numbering starts at 1 for a brand-new
    project, or continues after the highest number already in the list.

    use_trends (optional, off by default) feeds real YouTube Analytics
    top-performer data into the request instead of relying purely on
    external research -- see build_trend_context. trend_projects lets the
    human also pull in top performers from OTHER projects (each entry
    stays tagged with its own project name), so a new concept can draw on
    or even merge ideas across channels, not just within this one. Raises
    SystemExit if use_trends is set but no project (this one or any
    selected other) has analytics data yet -- silently falling back to
    plain research would defeat the point of asking for trend mode."""
    concept_list_path = find_concept_list_path()
    existing_text = concept_list_path.read_text(encoding="utf-8") if concept_list_path.exists() else ""
    existing_numbers = [int(m) for m in re.findall(r"(?m)^Tale #(\d+):", existing_text)]
    start_at = (max(existing_numbers) + 1) if existing_numbers else 1

    research_clause = (
        "Use web search to research what performs well in this content genre, then "
        if web_search_available else
        "Drawing on what you already know about what performs well in this content genre, "
    )
    reviewed_examples = reviewed_spec_examples()

    trend_context = None
    if use_trends:
        trend_context = build_trend_context(project_name, trend_projects)
        if trend_context is None:
            raise SystemExit(
                "[dream_step] trend mode is on but no performance data is available yet.\n"
                "EXPECTED: this project (or another project explicitly selected to "
                "include) needs a YouTube Analytics cache that's actually been refreshed.\n"
                "TO FIX: open that project's Analytics tab and click Refresh, or turn off "
                "trend mode for this request.")

    trend_clause = (
        f" trend_context lists this channel's own real top-performing video titles/tags "
        f"from YouTube Analytics (and, if selected, other projects' too -- each entry "
        f"tagged with its source project). Use it to favor patterns that have actually "
        f"performed well here (subject matter, tag themes, workflow style) instead of "
        f"guessing blind. When two listed top performers -- from the same project or "
        f"different ones -- could genuinely combine into one strong new concept, merging "
        f"them into a single idea is encouraged; only do this when it actually produces "
        f"something good, never force an awkward mashup just to use two entries."
        if trend_context else ""
    )

    return {
        "project": project_name, "count": count, "start_at": start_at,
        "existing_list_tail": existing_text[-2000:] if existing_text else None,
        "reviewed_examples": reviewed_examples or None,
        "creative_guidance": creative_guidance_pointer(),
        "trend_context": trend_context,
        "schema_hint": [{"number": "int", "title": "string", "animal": "string",
                          "role": "string", "line": "string -- one sample line"}],
        "instructions": (
            f"{research_clause}write exactly {count} concept entries as a JSON array "
            f"matching schema_hint, numbered {start_at} through {start_at + count - 1}. "
            f"Each needs a distinct animal AND human-role pairing (check "
            f"existing_list_tail for near-duplicates to avoid). "
            + (f"Also check reviewed_examples -- these are Tales a human has actually "
               f"approved, the real creative bar, not just whatever's been drafted -- for "
               f"two things: (1) don't repeat an animal+role pairing or joke-type that "
               f"already shipped there, and (2) match the comedic tightness/tone those "
               f"examples demonstrate (short, quotable punchlines; a specific mapped voice "
               f"played with real commitment) rather than just the broader style described "
               f"in creative_guidance. " if reviewed_examples else
               f"reviewed_examples is empty -- nothing's been reviewed/approved yet for "
               f"this channel, so rely on creative_guidance alone. ") +
            f"Use "
            f"creative_guidance (this channel's own CREATIVE.md) as the source of truth for "
            f"tone, format, and what kind of concept fits this channel -- research is for "
            f"what performs well in the genre, not for overriding the channel's own "
            f"established style."
            + trend_clause
        ),
    }


def commit_concepts_response(project_name, count, response):
    """Validate + write a concepts response to the project's own master
    concept list (concepts.md) -- shared by every concepts-research
    caller (h_concepts, --interactive's new-project flow)."""
    if not isinstance(response, list) or len(response) != count:
        raise SystemExit(
            f"[dream_step] concepts response is not a JSON array of exactly {count} "
            f"entries.\nEXPECTED: a JSON array matching the schema_hint from the "
            f"request.\nTO FIX: get a corrected response matching that shape.")

    concept_list_path = find_concept_list_path()
    lines = []
    for entry in response:
        lines.append(f"Tale #{entry['number']}: {entry['title']} — "
                      f"A {entry['animal']} {entry['role']}. \"{entry['line']}\"")
    with concept_list_path.open("a", encoding="utf-8") as f:
        if concept_list_path.stat().st_size > 0:
            f.write("\n")
        f.write("\n".join(lines) + "\n")
    print(f"[dream_step] wrote {len(response)} concept entries to {concept_list_path}")


def do_write_spec(number, spec_json_str, allow_custom_beats=False, positive_prompt_is_human=False):
    """Write (or overwrite) spec_{number:03d}.json from a JSON string,
    validating required fields and forcing the correct filename/path/
    "number" value -- the agent supplies data, this function owns
    where and how it lands on disk, so a wrong path, wrong zero-
    padding, or mismatched "number" field is structurally impossible.

    allow_custom_beats: the CLI's explicit --allow-custom-beats opt-in --
    NOT the default, a deliberate human decision to fully skip the beat
    check for this one call, no warning printed (unchanged from its
    original contract).

    positive_prompt_is_human: separate signal, set by write_row_spec /
    _generate_and_write_spec when positive_prompt specifically was typed
    directly into the manage table (not composed by the model) -- prints
    a WARNING on a beat-structure mismatch but still writes the spec,
    rather than either silently skipping the check (that's what
    allow_custom_beats is for, and conflating the two made a human's
    typed content trigger the CLI flag's silent-bypass behavior, which
    isn't what was wanted) or hard-blocking a human's own deliberate
    wording choice the way AI-composed content should be."""
    try:
        spec = json.loads(spec_json_str)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"[dream_step] --spec-json is not valid JSON: {e}\n"
            f"EXPECTED: a single valid JSON object with the spec's fields.\n"
            f"TO FIX: this usually means shell-escaping broke quotes/apostrophes in "
            f"dialogue. Use --spec-json-stdin with a heredoc instead of a quoted "
            f"--spec-json argument -- a heredoc needs no escaping at all and avoids "
            f"this entire class of error.")

    spec = _validate_and_normalize_spec(number, spec, allow_custom_beats, positive_prompt_is_human)

    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[dream_step] wrote {spec_path}", flush=True)
    sync_master_list_entry(number, spec.get("title"), spec.get("premise"))


def sync_master_list_entry(number, title, premise):
    """Keep the master concept list's 'Tale #N: ...' line for this number
    matching whatever spec actually just got written to disk -- called
    from the ONE place a spec is ever written (do_write_spec), so every
    save path (human-typed, AI-composed, either backend) goes through
    this the same way.

    Clearing a row's title/premise and letting the AI (or a human) invent
    a genuinely new idea would otherwise leave the OLD list line
    untouched -- the list would keep advertising the old idea, silently
    falling out of sync with what actually got rendered. Overwrites the
    existing line if this number already has
    one, appends a new line if not -- always reality-matches, no matter
    which direction the change came from.

    Deliberately does NOT try to reproduce commit_concepts_response's
    strict 'TITLE — A animal role. "line"' format (that needs an
    animal/role/quoted-line breakdown this function was never given) --
    a plainer 'TITLE — premise' line still round-trips fine through
    parse_concept_entry (it falls back to the whole remainder as-is when
    the strict pattern doesn't match), just without that extra
    structure. Silently no-ops if no master list exists yet -- syncing
    into a list that was never created isn't this function's job to
    start."""
    if not title:
        return
    concept_list_path = find_concept_list_path()
    if not concept_list_path.exists():
        return
    text = concept_list_path.read_text(encoding="utf-8")
    premise_line = next((l for l in (premise or "").strip().splitlines() if l.strip()), "(no premise)")
    new_line = f"Tale #{number}: {title} — {premise_line}"
    pattern = re.compile(rf"(?m)^Tale #{number}:.*$")
    if pattern.search(text):
        text = pattern.sub(lambda m: new_line, text, count=1)
    else:
        text = text.rstrip("\n") + f"\n{new_line}\n"
    concept_list_path.write_text(text, encoding="utf-8")


def _validate_and_normalize_spec(number, spec, allow_custom_beats=False, positive_prompt_is_human=False):
    """The validation core of do_write_spec, split out so AI-composed
    content can be validated the exact same way as anything else written
    through do_write_spec, which is the only place that actually
    persists; this function just enforces every rule a spec has to pass
    either way. Mutates and returns spec (sets "number", applies the
    default negative_prompt) -- raises SystemExit on any failure,
    same as do_write_spec always has."""
    spec["number"] = number

    if not spec.get("negative_prompt"):
        spec["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
        print(f"[dream_step] no negative_prompt given -- applied the confirmed "
              f"default: {DEFAULT_NEGATIVE_PROMPT!r}", flush=True)

    missing = [f for f in REQUIRED_SPEC_FIELDS if f not in spec]
    if missing:
        raise SystemExit(f"[dream_step] --spec-json is missing required fields: "
                          f"{missing}. Required (besides \"number\", which is set "
                          f"automatically from --write-spec): {REQUIRED_SPEC_FIELDS}")

    if not isinstance(spec.get("workflow"), str) or spec["workflow"] not in (
            "fp8_t2v", "i2v", "fml2v"):
        raise SystemExit(f"[dream_step] \"workflow\" must be one of "
                          f"\"fp8_t2v\"/\"i2v\"/\"fml2v\" as a plain string, "
                          f"got: {spec.get('workflow')!r}")

    # Confirmed failure (2026-08-09): a model asked for
    # fml2v_keyframe_prompts returned a single flat string instead of the
    # required {"first"/"middle"/"last"} object -- written to disk
    # unchecked, it crashed every later read of this row (get_manage_row's
    # kf.get(...)). Reject the shape here, at the one place all spec
    # writes funnel through, rather than trusting every caller/model to
    # get it right.
    if "fml2v_keyframe_prompts" in spec:
        kf = spec["fml2v_keyframe_prompts"]
        if not isinstance(kf, dict) or set(kf) != {"first", "middle", "last"} or \
                not all(isinstance(v, str) and v.strip() for v in kf.values()):
            raise SystemExit(
                f"[dream_step] \"fml2v_keyframe_prompts\" must be a JSON object with "
                f"exactly the keys \"first\", \"middle\", \"last\", each a non-empty "
                f"string -- got: {kf!r}")

    if not allow_custom_beats:
        positive_prompt = spec.get("positive_prompt", "")
        missing_headers = [h for h in REQUIRED_POSITIVE_PROMPT_HEADERS if h not in positive_prompt]
        # Beats are found by splitting on the "HH:MM-HH:MM:" timestamp
        # lines under [Timeline & Audio Sync]: -- unlike the old fixed
        # 4x[00:00-00:05] format, beat count/timing is flexible now, so
        # this is a plain count rather than a fixed marker list.
        segments = re.split(r"\d{2}:\d{2}-\d{2}:\d{2}:", positive_prompt)[1:]
        # Confirmed failure (2026-08-08, live render test): a beat with
        # Video: but no Audio: line at all pushed that beat's audio to the
        # WRONG place in the render (an earlier beat's audio bled into
        # it). Every beat needs BOTH lines, always -- a beat with nothing
        # to say still needs an Audio: line describing ambient sound/
        # music/sfx/silence, never an omitted one.
        beats_missing_lines = [i + 1 for i, seg in enumerate(segments)
                                if "- Video:" not in seg or "- Audio:" not in seg]
        has_voice_tag = bool(re.search(r"Voice [A-Z]\s*\[", positive_prompt))
        # The char class includes the plain apostrophe ' because smaller
        # local models often quote dialogue with straight single quotes
        # ('like this') rather than curly/double ones; omitting it scores
        # real dialogue as 0 beats and gets the model told to "fix"
        # prompts that are already fine. The lookarounds keep
        # contractions (it's, don't) from being
        # miscounted as a quote pair -- a bare ' next to a letter on the
        # inward side is a contraction, not a quote boundary.
        beats_with_dialogue = sum(
            1 for seg in segments
            if re.search(r'(?<![A-Za-z])[\'"‘’“”].{3,}[\'"‘’“”](?![A-Za-z])', seg))
        # Confirmed failure (Tale #81): beats present, but only ONE beat
        # carried any actual spoken dialogue -- the rest were pure silent
        # visual description, and most of the original line got dropped
        # rather than spread across beats. "Has the structure" isn't the
        # same as "has enough narrative content." Proportional (60%,
        # rounded up) rather than a flat floor of 3 -- a flat floor was
        # fine for the old fixed 4-beat format (3 of 4 = 75%) but silently
        # gets too lax as videos get longer and beat count grows with them
        # (3 of 10 beats having dialogue would leave most of a longer
        # piece silent). ceil(0.6n) reduces to the original 3-of-4 ratio
        # exactly at n=4, and scales from there instead of staying fixed.
        required_dialogue_beats = -(-6 * len(segments) // 10) if segments else 0

        if missing_headers or len(segments) < 2:
            problem = (f"missing required section header(s): {missing_headers}"
                       if missing_headers else
                       f"only {len(segments)} timestamped beat(s) found under "
                       f"[Timeline & Audio Sync]: -- need at least 2 (format: "
                       f"'HH:MM-HH:MM:' on its own line, followed by '- Video:' "
                       f"and '- Audio:' lines).")
            message = (
                f"positive_prompt is missing the required Scene Setup / "
                f"Timeline & Audio Sync structure -- confirmed to produce "
                f"reliable audio/lip-sync vs. a flowing paragraph (which has "
                f"caused confused/garbled audio).\n"
                f"PROBLEM: {problem}\n"
                f"REQUIRED FORMAT (match this exactly, character for character, "
                f"just with your own content):\n{BEAT_FORMAT_EXAMPLE}\n"
                f"If this specific rework genuinely needs a different "
                f"structure (documented reason, not just convenience), pass "
                f"--allow-custom-beats.")
            if positive_prompt_is_human:
                print(f"[dream_step] WARNING (proceeding -- positive_prompt was "
                      f"typed directly, not AI-composed): {message}", flush=True)
            else:
                raise SystemExit(f"[dream_step] {message}")
        elif beats_missing_lines:
            message = (
                f"positive_prompt has beat(s) missing a Video: or Audio: line -- "
                f"beat number(s) {beats_missing_lines} (counting from 1, in order) "
                f"are missing one or the other. Confirmed to break the render: a "
                f"beat with no Audio: line pushes that beat's audio to the WRONG "
                f"place in the output (an earlier beat's audio bleeds into it).\n"
                f"REQUIRED: EVERY beat needs both lines, always. A beat with "
                f"nothing to say still needs an Audio: line -- describe ambient "
                f"sound/music/sfx/silence instead of omitting it (see the last "
                f"beat in the example, which has no dialogue but still has an "
                f"Audio: line).\n"
                f"REQUIRED FORMAT:\n{BEAT_FORMAT_EXAMPLE}\n"
                f"Pass --allow-custom-beats only if this specific rework has a "
                f"documented reason to skip this.")
            if positive_prompt_is_human:
                print(f"[dream_step] WARNING (proceeding -- positive_prompt was "
                      f"typed directly, not AI-composed): {message}", flush=True)
            else:
                raise SystemExit(f"[dream_step] {message}")
        elif not has_voice_tag:
            message = (
                f"positive_prompt has the required structure, but no "
                f"'Voice A [role]:' (or Voice B/C/...) tag was found before any "
                f"dialogue line -- every spoken line needs a voice tag, even "
                f"when there's only one voice in the piece.\n"
                f"REQUIRED FORMAT:\n{BEAT_FORMAT_EXAMPLE}\n"
                f"Pass --allow-custom-beats only if this specific rework has a "
                f"documented reason to skip this.")
            if positive_prompt_is_human:
                print(f"[dream_step] WARNING (proceeding -- positive_prompt was "
                      f"typed directly, not AI-composed): {message}", flush=True)
            else:
                raise SystemExit(f"[dream_step] {message}")
        elif beats_with_dialogue < required_dialogue_beats:
            message = (
                f"positive_prompt has the required structure, "
                f"but only {beats_with_dialogue}/{len(segments)} beats contain "
                f"actual quoted dialogue (a real \"line like this\" inside the "
                f"beat's own Audio: line) -- confirmed too thin (Tale #81 had "
                f"dialogue in only 1 of 4 beats and needed far more narrative).\n"
                f"REQUIRED: at least {required_dialogue_beats} of {len(segments)} "
                f"beats need a real spoken line, not just silent visual "
                f"description. If your dialogue is one long line, SPLIT it "
                f"across beats rather than compressing it into one -- see how "
                f"each beat below carries its own piece of the "
                f"line:\n{BEAT_FORMAT_EXAMPLE}\n"
                f"Pass --allow-custom-beats only if this specific rework has a "
                f"documented reason to skip this.")
            if positive_prompt_is_human:
                print(f"[dream_step] WARNING (proceeding -- positive_prompt was "
                      f"typed directly, not AI-composed): {message}", flush=True)
            else:
                raise SystemExit(f"[dream_step] {message}")

    return spec


def _log_ai_call(label, backend, model, fn):
    """Runs fn() (a zero-arg callable wrapping one AI backend call) with
    a start/elapsed-time log line naming the backend and model actually
    used. Without this, the render log would show only generic stage
    names ("t2i (keyframe: first)") with no indication of which
    backend/model answered or how long it took -- impossible to tell
    from the log alone whether a slow step was local Ollama, a real
    Gemini API round-trip (with real cost), or a retry loop.
    Every dispatcher (_vision_query, _creative_completion,
    tool_completion) routes its actual network call through this."""
    model_part = f" ({model})" if model else ""
    print(f"[dream_step] -> {label} via {backend}{model_part}...", flush=True)
    start = time.time()
    try:
        result = fn()
    except Exception as e:
        print(f"[dream_step] <- {label} via {backend} FAILED after {time.time() - start:.1f}s: {e}", flush=True)
        raise
    print(f"[dream_step] <- {label} via {backend} done in {time.time() - start:.1f}s", flush=True)
    return result


def _vision_query(prompt, b64_images):
    """Image-in/text-out vision query -- backend picked by config.json's
    vision_backend (2026-08-12), same "ollama" (default, local/free) /
    "gemini" choice as creative_backend, but tracked independently (see
    DEFAULT_CONFIG's own comment on why). "gemini" delegates to
    gemini_text.py's generate_vision_text, which implements this exact
    same (prompt, b64_images) -> response_text contract, so callers
    (keyframe QC review, generate_dream.py's
    review_image_against_description/review_keyframe_pair) never need to
    know which backend answered."""
    config = load_config()
    backend = config.get("vision_backend", "ollama")
    if backend == "gemini":
        import gemini_text
        return _log_ai_call("vision query", "gemini", config.get("gemini_vision_model"),
                             lambda: gemini_text.generate_vision_text(prompt, b64_images))

    def _ollama_call():
        req = urllib.request.Request(
            f"{config['ollama_url']}/api/generate",
            data=json.dumps({"model": config["vision_model"], "prompt": prompt, "images": b64_images,
                              "stream": False, "think": False,
                              "options": VISION_OPTIONS}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "(no response)")
    return _log_ai_call("vision query", "ollama", config.get("vision_model"), _ollama_call)


CREATIVE_OPTIONS = {"num_predict": 12288, "num_ctx": 49152}


def _creative_completion(prompt, retries=3, model=None):
    """Direct, single-shot call for creative content -- modeled on
    _vision_query above (same project, same reliable pattern): the model
    is a pure text-in/JSON-out function, never an agent with tool use or
    its own judgment about what to do next. Built 2026-08-07 to replace
    dispatching a full Claude Code agent to drive this workflow, after
    that agent twice fabricated an answer (a number that was never given,
    a "note" nobody actually provided) rather than genuinely stopping to
    ask a human -- a CLI flag requiring a value doesn't stop a model from
    inventing that value itself, but a function that only ever returns
    what the model produced, with no ability to call any other tool,
    closes that hole structurally.

    Backend picked by config.json's creative_backend (2026-08-11, added
    after the default local model -- gemma4:E4B, a small 4B-class model
    -- proved technically compliant but consistently weak on genuine
    comedic writing no matter how much prompt guidance it was given):
    "ollama" (default, local/free) stays the function below; "gemini"
    delegates to gemini_text.py, which implements this exact same
    (prompt, retries, model) -> (parsed, history) contract, so the
    caller never needs to know which backend answered. Requires its own
    saved API key -- see that module's is_enabled()."""
    config = load_config()
    backend = config.get("creative_backend", "ollama")
    if backend == "gemini":
        import gemini_text
        return _log_ai_call("creative completion", "gemini", model or config.get("gemini_text_model"),
                             lambda: gemini_text.generate_json(prompt, retries=retries, model=model))
    attempt_prompt = prompt
    history = []
    for attempt in range(1, retries + 1):
        attempt_start = time.time()
        req = urllib.request.Request(
            f"{config['ollama_url']}/api/generate",
            data=json.dumps({
                "model": model or config["creative_model"], "prompt": attempt_prompt, "stream": False,
                "think": False, "format": "json", "options": CREATIVE_OPTIONS,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8")).get("response", "")
        except Exception as e:
            print(f"[dream_step] creative completion via ollama attempt {attempt} FAILED "
                  f"after {time.time() - attempt_start:.1f}s: {e}", flush=True)
            history.append(f"attempt {attempt}: request failed ({e})")
            continue
        print(f"[dream_step] creative completion via ollama ({model or config['creative_model']}) "
              f"attempt {attempt} done in {time.time() - attempt_start:.1f}s", flush=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            history.append(f"attempt {attempt}: invalid JSON ({e}): {raw[:300]!r}")
            attempt_prompt = (f"{prompt}\n\nYour previous answer was not valid JSON "
                               f"({e}). Reply with ONLY the JSON object, nothing else.")
            continue
        return parsed, history
    raise RuntimeError(
        f"[dream_step] _creative_completion failed after {retries} attempts:\n" +
        "\n".join(history))


def _extract_json_from_text(text, label, attempt, history):
    """Used by _ollama_tool_completion: a model's actual answer may have
    prose/markdown fences around the JSON despite being told not to, so
    this searches for the first embedded {...}/[...] block rather than
    requiring the whole response to be pure JSON, parses it, and on any
    failure appends a description to history. Returns (parsed, None) on
    success, or (None, retry_note) on failure, where retry_note is a
    short string describing what went wrong for the model --
    _ollama_tool_completion restarts the whole tool-calling conversation
    fresh each attempt, so it just discards retry_note and retries."""
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not m:
        history.append(f"attempt {attempt}: no JSON object/array found in {label}: {text[:300]!r}")
        return None, ("Your previous answer didn't contain a JSON object/array. "
                       "Reply with ONLY the JSON, nothing else.")
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        history.append(f"attempt {attempt}: {label} JSON invalid ({e}): {m.group(0)[:300]!r}")
        return None, (f"Your previous answer was not valid JSON ({e}). "
                       f"Reply with ONLY the JSON, nothing else.")


_OLLAMA_SEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use for anything requiring "
                        "up-to-date or external facts you don't already know.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "max results, default 5"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for factual/reference information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "max results, default 3"},
        }, "required": ["query"]},
    }},
]


def tool_completion(prompt, retries=3, model=None):
    """Web-search-capable completion for concept generation -- backend
    picked by config.json's creative_backend -- always aligned with
    Creative writing's own backend choice, since concept research feeds
    straight into creative writing and letting the two diverge would
    just add a decision with no real payoff.

    "gemini" delegates to gemini_text.py's generate_json_with_search,
    which uses Gemini's own NATIVE server-side web search tool (google_
    search grounding) -- doesn't need the local tool-execution loop
    _ollama_tool_completion below implements for itself, since Gemini's
    own servers run the search and fold results in before responding.
    Same (prompt, retries, model) -> (parsed, history) contract either
    way, so callers (build_concepts_request_payload's web-research flow)
    never need to know which backend answered. This function is only
    reachable from concept generation."""
    config = load_config()
    backend = config.get("creative_backend", "ollama")
    if backend == "gemini":
        import gemini_text
        return _log_ai_call("tool completion (web search)", "gemini", model or config.get("gemini_text_model"),
                             lambda: gemini_text.generate_json_with_search(prompt, retries=retries, model=model))
    # Concept research shares Creative writing's own model choice --
    # both are lightweight text tasks, so a separate model per role would
    # just add a dropdown with no real benefit.
    resolved_model = model or config.get("creative_model")
    return _log_ai_call("tool completion (web search)", "ollama", resolved_model,
                         lambda: _ollama_tool_completion(prompt, retries=retries, model=resolved_model))


def _ollama_tool_completion(prompt, retries=3, max_tool_rounds=4, model=None):
    """Local tool-calling loop against Ollama's OWN /api/chat 'tools'
    parameter -- gives a fully local model real web_search/
    wikipedia_search access with no external API and no Ollama account
    of any kind involved. A plain curl to /api/chat with a custom tool
    schema returns a genuine tool_calls response using nothing but a
    locally running Ollama instance -- Ollama's own account-gated
    web-search PRODUCT is a separate,
    unrelated feature this never touches. We execute the tool calls
    ourselves in plain Python (importing web_search_mcp's functions
    directly as plain callables, not through the MCP stdio protocol),
    then feed the result back as a 'tool' message and let the model
    continue, same request/response shape as any OpenAI-style
    tool-calling loop. Bounded to max_tool_rounds so a model that keeps
    calling tools instead of answering can't loop forever."""
    import web_search_mcp
    tool_fns = {"web_search": web_search_mcp.web_search, "wikipedia_search": web_search_mcp.wikipedia_search}
    config = load_config()
    history = []
    for attempt in range(1, retries + 1):
        messages = [{"role": "user", "content": prompt}]
        final_content = None
        request_failed = False
        for _round in range(max_tool_rounds):
            req = urllib.request.Request(
                f"{config['ollama_url']}/api/chat",
                data=json.dumps({
                    "model": model or config["creative_model"], "messages": messages,
                    "stream": False, "tools": _OLLAMA_SEARCH_TOOLS, "think": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                history.append(f"attempt {attempt}: request failed ({e})")
                request_failed = True
                break
            message = data.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                break
            messages.append(message)
            for call in tool_calls:
                fn_name = call.get("function", {}).get("name")
                fn_args = call.get("function", {}).get("arguments") or {}
                fn = tool_fns.get(fn_name)
                if fn is None:
                    result_text = f"ERROR: unknown tool '{fn_name}'"
                else:
                    try:
                        result_text = fn(**fn_args)
                    except Exception as e:
                        result_text = f"ERROR running {fn_name}: {e}"
                messages.append({"role": "tool", "content": str(result_text)})
        else:
            history.append(f"attempt {attempt}: exceeded {max_tool_rounds} tool-call rounds "
                            f"without a final answer")
        if request_failed or final_content is None:
            continue
        parsed, _retry_note = _extract_json_from_text(final_content, "final answer", attempt, history)
        if parsed is None:
            continue
        return parsed, history
    raise RuntimeError(
        f"[dream_step] _ollama_tool_completion failed after {retries} attempts:\n" +
        "\n".join(history))


def sanitize_review_text_for_log(text):
    """Display-only cleanup for advisory vision-review text before it hits
    a job's log stream. The model is deliberately instructed to end its
    reply with the literal word PASS or FAIL (see do_review_images /
    review_image_against_description / review_keyframe_pair -- that exact
    wording is calibrated by direct A/B testing and must NOT change, or
    the model's actual verdict accuracy could regress). But a job whose
    render genuinely succeeded, possibly after an advisory check or an
    automatic retry caught and fixed something along the way, should
    never show the bare word FAIL in its log -- a human/agent skimming
    for what went wrong reads FAIL as fatal, when here it's just one
    non-blocking signal (or an attempt that got retried and fixed).
    Rewrites only the display copy: the raw text used for the actual
    pass/fail decision elsewhere is untouched by this function."""
    text = re.sub(r"\bFAIL\b", "NOT MATCHED (advisory only)", text)
    text = re.sub(r"\bPASS\b", "MATCHED", text)
    return text


def do_review_images(number):
    """Send this number's rendered reference image(s) to a local vision
    model for a quality check -- the local qwen3-coder-agent driving
    this pipeline is text-only (confirmed: no "vision" capability), so
    it cannot look at its own generated images at all without this.
    qwen3-vl:8b confirmed more reliable than minicpm-v for this
    (consistent, specific species identification vs. minicpm-v naming a
    different animal per frame for the same actual images).

    Checks each image against its OWN specific intended description
    (from the fml2v keyframe-prompts sidecar, or i2v_generate_image_prompt)
    -- confirmed necessary, not optional: a generic "does this look like
    a natural, consistent animal" check PASSED Tale #81's images even
    though the actual middle frame showed an isolated trapped paw
    instead of the weasel's body squeezing through as its own prompt
    described. Only checking against the SPECIFIC intended content
    caught that.

    Advisory, not a hard gate -- prints a report for a human or the
    orchestrating session to weigh, since vision-model verdicts on a
    small local model are a useful signal, not an infallible one.
    Unloads the vision model when done, same load/unload discipline as
    every other model this pipeline uses -- it doesn't sit holding VRAM
    after its job is finished.

    Looks up each image's intended description by mapping filename ->
    role via the same 1/2/3 = first/middle/last convention
    generate_dream.py uses, since the sidecar's own keys are "first"/
    "middle"/"last" rather than the actual filenames ("1", "2", "3").
    The generic no-description fallback prompt below only checks natural
    anatomy, not a specific art style, when no sidecar description is
    available -- both photorealistic and animated/CGI are legitimate,
    Tale-by-Tale choices per CREATIVE.md, so the fallback must not
    hardcode "photorealistic nature-documentary style" as the only
    acceptable art style."""
    images = find_reference_images(number)
    if not images:
        print(f"[dream_step] no reference images found for #{number} to review.", flush=True)
        return

    sidecar_path = images[0].parent / "_fml2v_keyframe_prompts.json"
    keyframe_prompts = load_json(sidecar_path, {}) if len(images) > 1 else {}
    role_by_index = {"1": "first", "2": "middle", "3": "last"}
    spec = load_json(DATA_DIR / f"spec_{number:03d}.json", {})
    i2v_prompt = spec.get("i2v_generate_image_prompt")

    try:
        for img in images:
            role = role_by_index.get(img.stem)
            intended = (keyframe_prompts.get(role) if role else None) or \
                (i2v_prompt if len(images) == 1 else None)
            b64 = [base64.b64encode(img.read_bytes()).decode("utf-8")]
            if intended:
                prompt = (f"This image is supposed to show exactly this: {intended!r} "
                          f"Does the image actually match that description? Describe "
                          f"exactly what you see, note any mismatch (wrong/missing body "
                          f"parts, wrong action, deformed or hybrid anatomy, wrong "
                          f"trajectory/side), and end with PASS or FAIL.")
            else:
                prompt = ("Review this reference image for a video generation pipeline. "
                          "Describe the animal/subject and whether its anatomy looks "
                          "natural and correct for whatever art style is used (not "
                          "deformed or a hybrid of multiple animals). Do not judge "
                          "whether the style itself is photorealistic or animated/CGI -- "
                          "either is acceptable, only judge internal consistency and "
                          "anatomical correctness. End with PASS or FAIL.")
            try:
                response = _vision_query(prompt, b64)
            except Exception as e:
                print(f"[dream_step] >>> vision review of {img.name} failed to run ({e}) -- "
                      f"{load_config()['vision_model']} may not be loaded/available. Not a "
                      f"hard failure, just means no automated check happened for this image.", flush=True)
                continue
            # This check is advisory (see docstring) -- it never affects the
            # job's own success/failure. The model's own response text still
            # ends with the literal word "FAIL" it was asked for (that
            # instruction is calibrated, see the module comment above; don't
            # change the model-facing wording), but a human/agent skimming
            # the log for what actually went wrong shouldn't read this as
            # the RENDER having failed -- labeled WARNING, not FAIL, and
            # explicitly marked non-blocking right in the line.
            verdict_label = "WARNING" if re.search(r"\bFAIL\b", response) else "OK"
            print(f"[dream_step] image QC ({verdict_label}, advisory -- does not affect "
                  f"render result) of #{number}'s {img.name} via "
                  f"{load_config()['vision_model']}:\n{sanitize_review_text_for_log(response)}\n", flush=True)

        if len(images) > 1:
            # Confirmed (Tale #81): a vague "same animal throughout" question
            # PASSED three keyframes with visibly different fences/hole
            # sizes and geometry the video render then had to physically
            # reconcile (the animal passing through solid wood in one
            # beat). Ask specifically about the SETTING/OBJECTS, not just
            # the animal -- that's what actually drifted and what the
            # vague version missed.
            # A holistic "are these consistent" question misses real
            # defects that an itemized YES/NO checklist catches -- use
            # the checklist structure, not a holistic question.
            b64_all = [base64.b64encode(p.read_bytes()).decode("utf-8") for p in images]
            prompt = (f"These {len(images)} keyframes ({', '.join(p.stem for p in images)}) "
                      f"are meant to be the SAME continuous physical scene -- same animal, "
                      f"same setting, same objects -- at different moments of one action.\n"
                      f"Answer each question with YES or NO, then a final verdict:\n"
                      f"1. Is it the same animal/species throughout, with natural "
                      f"(not deformed/hybrid) anatomy in every image?\n"
                      f"2. Is every fixed object (fence, gap/hole, furniture, background "
                      f"structures) identical in position, size, and shape across all images?\n"
                      f"3. Is the background/setting identical across all images?\n"
                      f"If you answered NO to ANY question, the final verdict is FAIL. If all "
                      f"YES, the verdict is PASS. List every specific difference you notice, "
                      f"however small, then end with the single word PASS or FAIL.")
            try:
                response = _vision_query(prompt, b64_all)
                verdict_label = "WARNING" if re.search(r"\bFAIL\b", response) else "OK"
                print(f"[dream_step] image QC ({verdict_label}, advisory -- does not affect "
                      f"render result) of #{number}'s overall consistency via "
                      f"{load_config()['vision_model']}:\n{sanitize_review_text_for_log(response)}", flush=True)
            except Exception as e:
                print(f"[dream_step] >>> consistency review failed to run ({e})", flush=True)
    finally:
        cfg = vram_guard.load_config()
        if cfg.get("vision_model"):
            vram_guard.ollama_stop_model(cfg, model_name=cfg["vision_model"])


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def rel_path_str(path, base):
    """The ONLY way a spec's image_path/fml2v_*_image fields should ever
    be written -- ALWAYS forward-slash (.as_posix()), never OS-native
    separators. A spec written while this pipeline runs on Windows would
    store these fields with backslashes if written via bare str(Path)
    (which is OS-native); read later on Linux, "...Faceoff\\1.png" is
    treated as ONE literal filename (backslash is a normal filename
    character on Linux, not a separator) instead of two path segments,
    so a real, existing file gets reported as "does not exist". Forcing
    forward slashes on write means the stored value is valid on either
    platform regardless of which one wrote it."""
    return path.relative_to(base).as_posix()


def resolve_stored_rel_path(base, stored):
    """Turns a spec's stored image_path/fml2v_*_image string back into a
    real Path, tolerating a LEGACY value that was written on the other
    platform before rel_path_str existed (still has backslashes) --
    normalizes to forward slashes before joining. See rel_path_str's
    docstring for the exact failure this prevents."""
    return base / str(stored).replace("\\", "/")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def uploaded_images_dir(number):
    """A stable, code-owned location for reference images the web UI's
    manage table lets a human upload directly -- independent of the
    Dream's own render folder, which may not exist yet for a number
    that's never been rendered, or whose name changes if the title does.
    image_path (etc.) can point at ANY path under DREAMS_ROOT, not just
    inside the render folder, so this needs no special-casing elsewhere."""
    return DATA_DIR / "uploads" / str(number)


# slot name -> filename stem, shared between the upload handler and
# find_reference_images' sort order (1/2/3 sorts first/middle/last correctly).
IMAGE_SLOT_STEMS = {"image": "1", "first": "1", "middle": "2", "last": "3"}


def fml2v_images_satisfied(number):
    """True when three real, correctly-named (1/2/3) reference images
    already exist for this number -- fml2v_keyframe_prompts is only
    needed from the spec/keyframe-writing steps when this is False (see
    write_row_keyframes's own "nothing to write" case)."""
    images = find_reference_images(number)
    return len(images) == 3 and {p.stem for p in images} == {"1", "2", "3"}


def find_reference_images(number):
    """Look inside this number's existing rendered Dream folder(s) for image
    files the user dropped in alongside the .mp4/.txt -- the signal that
    they want the NEXT rework to use image-to-video conditioning instead of
    a fresh text-to-video render. Returns a list sorted by filename (so
    naming them "01_intro.png", "02_reaction.png", "03_conclusion.png"
    controls scene order): empty if none, one entry for a single-image i2v
    rework, multiple entries for a multi-keyframe rework (one generation per
    image, stitched together -- see run_multi_shot_i2v).

    Also checks uploaded_images_dir(number) -- the manage table's own
    upload location -- but only if the Dream folder itself has none, so a
    human manually dropping a file into the real render folder still wins
    (closer to canonical / more recently touched by a human on purpose)."""
    # A leading underscore marks a pipeline-internal file everywhere else
    # in this codebase (e.g. _fml2v_keyframe_prompts.json, _swap_tmp*), so
    # this loop must not count generate_dream.py's own
    # "_online_reference_seed.png" debug artifact as a fourth REAL
    # reference image for an online-sourced fml2v Tale -- doing so would
    # break resolve_slot_image's all-or-nothing len(images)==3 check for
    # every slot (breaking manage-table image delete), and would send
    # do_rework's {"1","2","3"} stem-equality check to the "ambiguous,
    # refusing to render" branch instead of recognizing a perfectly
    # normal complete triple.
    #
    # More generally: any OTHER unrelated photo sitting in a
    # Dream folder alongside a genuine 1/2/3 keyframe set (e.g. a human
    # keeping a reference/inspiration image nearby for their own use)
    # hit the exact same confusion -- 4+ images made the "exactly 3,
    # stems {1,2,3}" check fail even though the real keyframe set was
    # completely fine. Once real, properly-named keyframe files (stems
    # "1"/"2"/"3") are found, they're returned ALONE -- anything else in
    # the folder is just "another project image," not a competing
    # reference, and is ignored for this purpose entirely.
    folders = existing_dream_folders(number)
    images = []
    for folder_name in folders:
        folder = DREAMS_ROOT / folder_name
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("_"):
                images.append(p)
    keyframe_named = [p for p in images if p.stem in ("1", "2", "3")]
    if keyframe_named:
        images = keyframe_named
    if not images:
        updir = uploaded_images_dir(number)
        if updir.is_dir():
            for p in updir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(p)
    return sorted(images, key=lambda p: p.name)


def save_uploaded_image(number, slot, data, ext):
    """Write one manage-table image upload to its stable slot location.
    Overwrites whatever was there before for that slot (any extension --
    a human replacing a .png with a .jpg for the same slot shouldn't leave
    the old file behind to confuse find_reference_images' count)."""
    if slot not in IMAGE_SLOT_STEMS:
        raise SystemExit(f"[dream_step] invalid image slot: {slot!r}")
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in IMAGE_EXTENSIONS:
        raise SystemExit(f"[dream_step] invalid image extension: {ext!r}")
    updir = uploaded_images_dir(number)
    updir.mkdir(parents=True, exist_ok=True)
    stem = IMAGE_SLOT_STEMS[slot]
    for p in updir.glob(f"{stem}.*"):
        p.unlink()
    path = updir / f"{stem}{ext}"
    path.write_bytes(data)
    return path


def staged_upload_path(number, slot):
    """The staged (not-yet-rendered) upload for one slot, if any --
    checks ONLY uploaded_images_dir, unlike find_reference_images which
    ignores staging entirely once a real Dream folder has its own
    image. Lets the web UI show "current (rendered)" and "new (staged,
    not rendered yet)" as two distinct thumbnails instead of one
    resolved image silently winning, so a human can compare before
    deciding whether the staged replacement is actually better. Returns
    None if this slot has nothing staged."""
    if slot not in IMAGE_SLOT_STEMS:
        return None
    updir = uploaded_images_dir(number)
    if not updir.is_dir():
        return None
    stem = IMAGE_SLOT_STEMS[slot]
    matches = sorted(updir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def clear_staged_upload(number, slot):
    """Discards a staged (not-yet-rendered) upload for one slot without
    rendering anything -- the manage table's "undo" for a photo fetch/
    upload a human decides they don't want after seeing the "new"
    thumbnail, without forcing a real render+migrate_uploaded_images
    cycle just to get rid of it. No-op if nothing is staged for that
    slot."""
    path = staged_upload_path(number, slot)
    if path is None:
        return
    path.unlink()
    updir = uploaded_images_dir(number)
    try:
        updir.rmdir()  # only succeeds once empty -- leaves it if other slots still have staged files
    except OSError:
        pass


def delete_slot_image(number, slot):
    """Deletes the exact file currently shown/loaded for this slot --
    resolve_slot_image_lenient's per-slot exact-stem lookup (1.*/2.*/3.*
    for fml2v, the sole image for i2v), NOT resolve_slot_image's
    all-or-nothing "only counts as real once all three exist" check.
    Using resolve_slot_image (the stricter check) for delete would
    silently delete NOTHING whenever that check fails to resolve a path
    for any reason (e.g. an unrelated extra file in the Dream folder
    makes the "exactly 3" count wrong) -- the confirm dialog would fire,
    the API call would report success, but nothing on disk would
    change. Delete must only ever act on the one file actually
    displayed for this slot, never re-derive "is this a complete,
    render-ready set" as a precondition for removing it.

    Clears whichever spec field pointed at it (image_path /
    fml2v_first_image / etc). Deliberately does NOT try to regenerate
    anything itself -- this just removes what's there so the NEXT
    render/rework has nothing to reuse and falls through to its own
    generate-prompt path (i2v_generate_image_prompt /
    fml2v_keyframe_prompts) instead, same as if that image had never
    existed. No-op (returns False) if nothing is actually there to
    delete. Confirming with the human BEFORE calling this is the
    caller's job (the web UI does, via confirmModal) -- this is a real,
    permanent deletion with no undo."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    if not spec_path.exists():
        return False
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    workflow = spec.get("workflow") or ""
    path = resolve_slot_image_lenient(number, workflow, slot)
    changed = False
    if path and path.exists():
        path.unlink()
        changed = True
    field = {"image": "image_path", "first": "fml2v_first_image",
              "middle": "fml2v_middle_image", "last": "fml2v_last_image"}.get(slot)
    if field and spec.get(field):
        spec[field] = ""
        changed = True
    if changed:
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return changed


def generate_reference_image_to_slot(number, slot, query, scene_prompt=None):
    """The manage table's "Online photo" button (Gemini-only since
    2026-08-09 -- see gemini_image.py's own docstring on why the free
    CC0-photo lookup and Hugging Face were both removed): generates a
    reference image via Gemini and saves it through save_uploaded_image,
    the exact same staging location a human dragging a file onto that
    slot uses -- so everything downstream (find_reference_images,
    migrate_uploaded_images, rendering) treats it identically to a
    manual upload, no separate code path needed. Overwrite-if-one-
    already-exists is the caller's call to confirm with the human
    BEFORE calling this (the web UI does, via confirmModal) -- this
    function just does it, unconditionally, same as save_uploaded_image
    always has.

    scene_prompt: the row's own still-image description for this slot
    (i2v_generate_image_prompt / fml2v_keyframe_prompts["first"]) if the
    web UI had one to send -- used verbatim (prefixed with an explicit
    size instruction) instead of a generic "a photo of the animal"
    description, same fix as generate_dream.py's try_online_first_frame
    and for the same confirmed-live reason: a generic species-only
    prompt gives Gemini no framing/pose guidance and real renders came
    back with parts of the subject cut off. Falls back to a generic
    (still size-prefixed) description when no scene prompt is available
    yet (e.g. a brand new row with nothing typed in that field).

    No "try a different one" exclusion logic needed here (unlike the
    old photo-search version) -- each Gemini call is already a fresh
    generation, a repeat click naturally gets a different result on its
    own. Raises RuntimeError (propagated from
    gemini_image.generate_reference_image) if Gemini isn't configured
    or the call fails -- this is a real, possibly-billed API call, not
    a free lookup, so failures surface plainly rather than silently
    falling back to anything."""
    import gemini_image
    scene_prompt = (scene_prompt or "").strip() or (
        f"A photorealistic photo of a {query}, accurate anatomy and "
        f"coloring, natural habitat, high detail.")
    prompt = f"512x896 portrait image: {scene_prompt}"
    tmp_path = DATA_DIR / f"_gemini_tmp_{number}_{slot}.png"
    try:
        gemini_image.generate_reference_image(prompt, tmp_path)
        data = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
    path = save_uploaded_image(number, slot, data, ".png")
    model = load_config().get("gemini_model") or gemini_image.MODEL
    return {"path": str(path), "query": query, "model": model}


def generate_keyframe_image_to_slot(number, workflow, slot, prompt_text):
    """The manage table's "Generate new" button for a keyframe IMAGE (as
    opposed to the prompt TEXT written by write_row_keyframes) -- one
    on-demand candidate, staged the exact same way
    generate_reference_image_to_slot (Gemini "Online photo") and manual
    uploads both already are, via save_uploaded_image -- so the result
    shows up in the same Current/New comparison, never touching the
    active image until the human deletes or replaces it. Local-vs-Gemini
    for this slot follows config.json's kf_backend, the same decision a
    real render makes (see generate_dream.generate_one_keyframe_candidate)
    -- never a per-click override, so this can't drift from what Render/
    Rework would actually produce for this role.

    slot 'image'/'first': no prerequisite. slot 'middle'/'last': the
    first-frame image must already exist (it's the I2I/image-edit base
    either way) -- raises SystemExit with a clear message if not, rather
    than silently generating from nothing."""
    import generate_dream
    role = {"image": "first", "first": "first", "middle": "middle", "last": "last"}.get(slot)
    if role is None:
        raise SystemExit(f"[dream_step] #{number}: invalid slot {slot!r} for keyframe generation.")

    first_frame_path = None
    if role != "first":
        first_frame_path = resolve_slot_image_lenient(number, workflow, "first")
        if first_frame_path is None:
            raise SystemExit(
                f"[dream_step] #{number}: the '{slot}' slot needs the first-frame image "
                f"to already exist -- it's the base every regeneration (local I2I or "
                f"Gemini image-edit) is conditioned on.\nTO FIX: generate/upload a first "
                f"frame for this row before regenerating '{slot}'.")

    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {"number": number}

    tmp_path = DATA_DIR / f"_kf_tmp_{number}_{slot}.png"
    try:
        generate_dream.generate_one_keyframe_candidate(
            spec, role, prompt_text, tmp_path, first_frame_path=first_frame_path)
        data = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
    path = save_uploaded_image(number, slot, data, ".png")
    return {"path": str(path)}


def migrate_uploaded_images(number, spec):
    """After a render creates/confirms this number's real Dream folder,
    move any images still sitting in the uploads staging location
    (uploaded_images_dir) into it, and repoint image_path if it was
    pointing at the staged copy. The uploads dir only ever exists as
    staging for a number that had no Dream folder yet when the image was
    uploaded through the manage table -- once a real folder exists,
    everything for that Dream belongs colocated in it, same as an image a
    human drops in directly."""
    updir = uploaded_images_dir(number)
    if not updir.is_dir():
        return
    staged = sorted(p for p in updir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not staged:
        return
    folders = existing_dream_folders(number)
    if not folders:
        return  # no real folder yet -- leave staged where it is
    dest_folder = DREAMS_ROOT / folders[0]
    # Normalized to match moved_map's own forward-slash keys (rel_path_str)
    # -- see rel_path_str's docstring for why a legacy backslash value
    # would otherwise silently fail this lookup.
    old_image_path = (spec.get("image_path") or "").replace("\\", "/") or None
    moved_map = {}
    for p in staged:
        dest = dest_folder / p.name
        # A staged upload REPLACING an existing slot (e.g. re-uploading a
        # new "first"/"middle" keyframe for a Dream that's already been
        # rendered once) must not be silently skipped here -- a naive
        # "don't clobber" would treat the OLD, now-outdated folder image
        # as the winner, so the new upload would never make it into the
        # real folder, never get used by the next render (which would
        # then "succeed" using the stale images, looking like nothing
        # happened), and never get cleared out of staging either (the
        # manage table would keep showing it as "new"). The whole
        # point of staging a same-named upload IS to replace whatever's
        # currently in that slot -- remove the stale file first so the
        # move actually lands.
        #
        # This must also cover the case where the replacement's file
        # FORMAT differs (e.g. the old slot is "2.png", the new upload is
        # "2.jpg") -- dest.exists() only checks the exact same filename,
        # so a differently-named old file would never be removed, leaving
        # BOTH "2.png" and "2.jpg" in the folder ("found 4 image files...
        # ambiguous"). The GUI is the source of truth for what a SLOT
        # (stem) contains, not whatever extension happens to already be
        # on disk -- clear every existing file sharing this upload's
        # STEM (any extension), not just an exact filename match, before
        # moving the new one in.
        for stale in dest_folder.glob(f"{p.stem}.*"):
            if stale.suffix.lower() in IMAGE_EXTENSIONS:
                stale.unlink()
        shutil.move(str(p), str(dest))
        moved_map[rel_path_str(p, DREAMS_ROOT)] = rel_path_str(dest, DREAMS_ROOT)
    try:
        updir.rmdir()  # only succeeds once empty -- leaves it if something didn't move
    except OSError:
        pass
    changed = False
    if old_image_path in moved_map:
        spec["image_path"] = moved_map[old_image_path]
        changed = True
    # Moving a staged upload into place must also update the spec's OWN
    # fml2v_first_image/middle/last fields to point at it -- relying on a
    # narrower special case elsewhere to do this isn't enough on its own.
    # A staged upload only ever exists because the human used the
    # GUI's per-slot upload feature (unlike a human manually dropping a
    # raw file into the folder, which is genuinely ambiguous) -- that's
    # already unambiguous confirmed intent, so this shouldn't need a
    # separate manual JSON edit to "confirm" what the upload already
    # confirmed. dest_folder / "1.png" (etc) is always named by slot stem
    # (see IMAGE_SLOT_STEMS/staged_upload_path), so the stem alone is
    # enough to know which fml2v_*_image field a given moved file
    # belongs to, regardless of its extension.
    if spec.get("workflow") == "fml2v":
        stem_to_field = {"1": "fml2v_first_image", "2": "fml2v_middle_image", "3": "fml2v_last_image"}
        for p in staged:
            field = stem_to_field.get(p.stem)
            dest_rel = moved_map.get(rel_path_str(p, DREAMS_ROOT))
            if field and dest_rel and spec.get(field) != dest_rel:
                spec[field] = dest_rel
                changed = True
    if changed:
        spec_path = DATA_DIR / f"spec_{number:03d}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def find_title_collision(number, title):
    """Check every OTHER spec_NNN.json for the exact same title (the
    hash-guard only catches whether a spec's content changed from ITS OWN
    history -- it has no idea whether the new content duplicates a
    DIFFERENT spec entirely). Returns the colliding number, or None.
    Confirmed failure mode: asked to make #90 unique from #82 (both
    "mirror" duplicates), a rework produced #90 with the exact title
    "The Mirror of Echoes" -- which #82 ALREADY had from its own earlier,
    separate rework. The result was a brand-new duplicate of #82's fixed
    version, the opposite of what was asked."""
    norm_title = (title or "").strip().lower()
    if not norm_title:
        return None
    for p in DATA_DIR.glob("spec_*.json"):
        try:
            other_number = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if other_number == number:
            continue
        try:
            other = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (other.get("title") or "").strip().lower() == norm_title:
            return other_number
    return None


def get_episode_label():
    """Same lookup as generate_dream.py's -- output folders are named
    '<label> #N <title>', label configurable per-project via that
    project's upload_template.json ('episode_label'), defaulting to
    'Dream'. Kept in sync with generate_dream.py's copy deliberately
    (small enough that a shared import isn't worth the coupling)."""
    template_path = DATA_DIR / "youtube" / "upload_template.json"
    if template_path.exists():
        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
            return template.get("episode_label", "Dream")
        except Exception:
            pass
    return "Dream"


def existing_dream_folders(number):
    """All '<label> #N <title>' folders on disk for this number -- normally
    there should be at most one, since a same-titled render overwrites its
    own folder. More than one means the title changed across renders and
    an earlier render got orphaned under its old name instead of replaced."""
    label = get_episode_label()
    return sorted(p.name for p in DREAMS_ROOT.glob(f"{label} #{number} *") if p.is_dir())


def list_media_folders(project_name):
    """Every render folder for this project in either of the two places a
    human moves a finished render between by hand today: 'active' (directly
    under DREAMS_ROOT, same folders run_render writes to) and 'reviewed'
    (under DREAMS_ROOT/Reviewed, purely human-managed -- dream_step.py
    itself never writes there). One entry per folder: number/title parsed
    from its '<label> #N <title>' name (falls back to the raw folder name
    if it doesn't match, so a human's own renamed/dropped-in folder still
    shows up rather than vanishing), and the folder's own named .mp4
    (falling back to the first .mp4 found if that exact file isn't there)."""
    label = get_episode_label()
    pattern = re.compile(rf"^{re.escape(label)} #(\d+) (.+)$")
    entries = []
    for location, base in (("active", DREAMS_ROOT), ("reviewed", DREAMS_ROOT / "Reviewed")):
        if not base.is_dir():
            continue
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name.startswith("_") or folder.name == "Reviewed":
                continue  # pipeline-internal (_data) or the Reviewed folder itself
            m = pattern.match(folder.name)
            number = int(m.group(1)) if m else None
            title = m.group(2) if m else folder.name
            video_files = sorted(p.name for p in folder.iterdir()
                                  if p.is_file() and p.suffix.lower() == ".mp4")
            # A stray extra .mp4 dropped into the
            # folder (e.g. a manual ComfyUI test render, or a raw
            # LTX_2.3_i2v_NNNNN_.mp4 the pipeline itself left behind before
            # copying the real result over it) can sort alphabetically
            # before the folder's own "<folder name>.mp4" and get picked
            # instead -- the player then showed the wrong video for this
            # Dream. Prefer the file matching the folder's own name
            # exactly; only fall back to "first found" if that exact file
            # isn't present at all.
            expected_name = f"{folder.name}.mp4"
            if expected_name in video_files:
                video_file = expected_name
            else:
                video_file = video_files[0] if video_files else None
            entries.append({
                "location": location,
                "folder": folder.name,
                "number": number,
                "title": title,
                "video_file": video_file,
            })
    entries.sort(key=lambda e: (e["number"] is None, e["number"] or 0, e["title"]))
    return entries


def _media_base(location):
    if location not in ("active", "reviewed"):
        raise SystemExit(f"[dream_step] invalid location {location!r} -- must be "
                          f"\"active\" or \"reviewed\"")
    return DREAMS_ROOT if location == "active" else DREAMS_ROOT / "Reviewed"


def _resolve_media_folder(folder_name, location):
    """Resolve a folder name the caller supplied against real folders on
    disk, refusing anything that isn't a direct, existing child of the
    expected base -- the only path-traversal defense this needs, since the
    caller is HTTP request data, not trusted code."""
    base = _media_base(location)
    if not folder_name or "/" in folder_name or "\\" in folder_name or folder_name in (".", ".."):
        raise SystemExit(f"[dream_step] invalid folder name: {folder_name!r}")
    path = (base / folder_name)
    if not path.is_dir() or path.resolve().parent != base.resolve():
        raise SystemExit(f"[dream_step] folder not found: {folder_name!r} in {location}")
    return path


def resolve_media_file(folder_name, location, filename):
    folder = _resolve_media_folder(folder_name, location)
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise SystemExit(f"[dream_step] invalid filename: {filename!r}")
    path = folder / filename
    if not path.is_file() or path.resolve().parent != folder.resolve():
        raise SystemExit(f"[dream_step] file not found: {filename!r} in {folder_name!r}")
    return path


def move_media_folder(folder_name, from_location, to_location):
    """Move a render folder between 'active' (DREAMS_ROOT) and 'reviewed'
    (DREAMS_ROOT/Reviewed) -- the same move a human does today by dragging
    the folder in Explorer, just exposed through the web UI."""
    if from_location == to_location:
        raise SystemExit(f"[dream_step] already in {to_location}")
    src = _resolve_media_folder(folder_name, from_location)
    dest_base = _media_base(to_location)
    dest_base.mkdir(exist_ok=True)
    dest = dest_base / folder_name
    if dest.exists():
        raise SystemExit(f"[dream_step] a folder named {folder_name!r} already exists "
                          f"in {to_location} -- resolve the conflict by hand first")
    shutil.move(str(src), str(dest))
    return dest


def delete_media_folder(folder_name, location):
    """Permanently delete a render folder (and everything in it -- the
    .mp4, its .txt sidecar, any reference images). No undo; the web UI is
    expected to confirm with the human before calling this."""
    path = _resolve_media_folder(folder_name, location)
    shutil.rmtree(path)


def record_history(number, event_type):
    """Append one entry to rework_history.json -- called ONLY after a real
    successful render, automatically, by this script. Never written to by
    the model directly; this is deliberate audit trail infrastructure, not
    a place for freeform notes (see CLAUDE.md's file-write allowlist)."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        spec = {}
    history = load_json(HISTORY_PATH, [])
    history.append({
        "number": number,
        "title": spec.get("title", ""),
        "premise": spec.get("premise", ""),
        "event": event_type,  # "origin" (first-ever render) or "rework"
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def check_image_prerequisites(number, spec):
    """For i2v/fml2v, verify a usable reference image exists (or a
    generate-prompt is provided) BEFORE launching the render subprocess,
    so a missing image surfaces as a clean, expected question here in
    dream_step.py instead of a generate_dream.py subprocess failure
    further down the pipeline. Checks the Dream's own project folder
    (find_reference_images) for an already-available image first."""
    workflow = spec.get("workflow")

    if workflow == "i2v":
        stored = spec.get("image_path") or ""
        img_path = Path(stored.replace("\\", "/")) if stored else Path("")
        if img_path.name and not img_path.is_absolute():
            img_path = DREAMS_ROOT / img_path
        if img_path.name and img_path.exists():
            return True
        existing = find_reference_images(number)
        if len(existing) == 1:
            # Auto-repoint rather than refuse -- same reasoning as
            # do_rework's own stale-image_path handling: a human/pipeline
            # replacing the image, or a folder rename on title change,
            # both leave image_path pointing at a file that no longer
            # exists even though workflow is already correctly "i2v" --
            # refusing here would just report "clicking render does
            # nothing" a second time despite do_rework's own check
            # already handling this.
            new_rel = rel_path_str(existing[0], DREAMS_ROOT)
            spec["image_path"] = new_rel
            spec_path = DATA_DIR / f"spec_{number:03d}.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            print(f"[dream_step] #{number}: image_path was stale -- auto-updated to "
                  f"the real file found in its Dream folder: {new_rel}", flush=True)
            return True
        if spec.get("i2v_generate_image_prompt"):
            return True  # generate_dream.py auto-generates it via T2I
        print(f"[dream_step] >>> no first image found in the project path for "
              f"#{number}. ASK THE USER: \"no first image found in the project "
              f"path, would you like to create one?\" If yes, set "
              f"\"i2v_generate_image_prompt\" (a still-image description of the "
              f"opening frame -- appearance/pose/setting, not animation) via "
              f"--write-spec and run this again -- that generates it via T2I "
              f"automatically. If no, stop and report.", flush=True)
        return False

    if workflow == "fml2v":
        existing = find_reference_images(number)
        if len(existing) == 3:
            return True
        if spec.get("fml2v_keyframe_prompts"):
            return True  # generate_dream.py auto-generates all three via T2I/I2I
        print(f"[dream_step] >>> no keyframe images found in the project path for "
              f"#{number}. ASK THE USER: \"no keyframe images found in the project "
              f"path, would you like to create them?\" If yes, set "
              f"\"fml2v_keyframe_prompts\" ({{\"first\": ..., \"middle\": ..., "
              f"\"last\": ...}}, still-image descriptions of each beat) via "
              f"--write-spec and run this again -- that generates all three via "
              f"T2I/I2I automatically. If no, stop and report.", flush=True)
        return False

    return True


def run_render(number, event_type, randomize_seeds=False, verbose=False, cancel_check=None):
    """Call render_dream.py for one number. Returns True on success.
    On success, automatically records a history entry -- this is the only
    place rework_history.json ever gets written.

    cancel_check: an optional zero-arg callable the web UI's Cancel
    button flips to True (see h_cancel_job) -- checked here after the
    render subprocess itself finishes, to skip the (several-minute,
    purely advisory) post-render vision QC pass when the user has
    already asked to stop. ComfyUI's own /interrupt already kills the
    render subprocess promptly; this only covers the QC tail some users
    mistook for the cancel "hanging" (2026-08-12)."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not check_image_prerequisites(number, spec):
        return False
    # True when the spec already points at real, already-existing image
    # file(s) BEFORE this render even starts (image_path / fml2v_first_
    # image+middle+last are only ever set from real files on disk --
    # see determine_code_owned_spec_fields) -- i.e. a human picked these
    # images directly. False means the workflow itself is about to
    # generate them fresh via T2I this render (i2v_generate_image_prompt /
    # fml2v_keyframe_prompts). The post-render vision review exists to
    # catch the WORKFLOW's own generation mistakes, not to second-guess a
    # human's own deliberate choice of image -- see the call site below.
    images_are_human_provided = (
        bool(spec.get("image_path")) if spec.get("workflow") == "i2v"
        else all(spec.get(f) for f in
                 ("fml2v_first_image", "fml2v_middle_image", "fml2v_last_image"))
        if spec.get("workflow") == "fml2v" else False)
    print(f"[dream_step] rendering #{number} via render_dream.py --spec {spec_path}", flush=True)
    cmd = [sys.executable, str(PIPELINE_DIR / "render_dream.py"), "--spec", str(spec_path)]
    if randomize_seeds:
        cmd.append("--randomize-seeds")
    if verbose:
        print(f"[dream_step] full command: {' '.join(cmd)}", flush=True)
        print(f"[dream_step] spec content:\n{json.dumps(spec, indent=2)}", flush=True)
    # subprocess.run() with no stdout capture would write straight to
    # this process's inherited file descriptor, which bypasses the web
    # UI's own sys.stdout redirect entirely (_LiveLog) -- nothing
    # render_dream.py/generate_dream.py/vram_guard prints, including the
    # ACTUAL REASON a render failed, would reach the GUI's job log (which
    # would show status "done" even for a real failure -- see _run_job --
    # with zero evidence why). Popen + reading stdout line by
    # line makes each line a real print() in THIS process instead, so it
    # flows through whatever sys.stdout currently is (the GUI's _LiveLog
    # mid-job, or the real console for the CLI), live, not just the two or
    # three lines dream_step itself happens to print directly.
    # text=True with no explicit encoding defaults to
    # locale.getpreferredencoding(), which on Windows is the console's
    # ANSI codepage (cp1252 here) -- NOT
    # UTF-8. A vision-model response (real Ollama JSON, genuinely UTF-8)
    # containing so much as one curly quote/em-dash/etc. outside cp1252's
    # range crashes this read outright ('charmap' codec can't decode byte
    # ...), killing an otherwise-successful multi-minute render at
    # whatever random point that byte happens to appear. errors="replace"
    # rather than strict UTF-8, since a still-wrong byte from some other
    # source shouldn't be able to repeat this failure mode ever again --
    # worst case a mangled character in the log, never a crashed render.
    # with-block (not a bare Popen()) so an exception mid-stream (e.g. the
    # print() below failing) still closes stdout and waits on the process
    # instead of leaking the pipe/subprocess handle.
    with subprocess.Popen(cmd, cwd=str(PIPELINE_DIR), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1,
                           encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    ok = proc.returncode == 0
    if ok:
        record_history(number, event_type)
        if spec.get("workflow") in ("i2v", "fml2v"):
            migrate_uploaded_images(number, spec)
            if images_are_human_provided:
                print(f"[dream_step] #{number}: reference image(s) were provided "
                      f"directly, not AI-generated -- skipping the vision QC review "
                      f"(that check exists to catch the WORKFLOW's own generation "
                      f"mistakes, not to second-guess an image a human already chose).",
                      flush=True)
            elif cancel_check is not None and cancel_check():
                print(f"[dream_step] #{number}: cancelled -- skipping the (advisory, "
                      f"non-essential) post-render vision QC pass.", flush=True)
                backfill_generated_image_path(number, spec)
            else:
                # Same gap as migrate_uploaded_images closes for the human-
                # upload case, just for images the WORKFLOW just generated
                # via T2I this render -- without this, the real file(s) sit
                # in the Dream folder but the spec's own image_path/
                # fml2v_first_image/etc never point at them, so a later
                # rework's "is this already configured for the image
                # that's there" check (do_rework) fails and skips instead
                # of reworking -- e.g. a render that goes clean via
                # i2v_generate_image_prompt can still leave the spec with
                # no image_path afterward.
                backfill_generated_image_path(number, spec)
                # Automatic, not dependent on the agent remembering to run
                # --review-images separately -- the local coding agent is
                # text-only and has repeatedly not thought to check its own
                # generated images without being told to every single time.
                do_review_images(number)
    return ok


def backfill_generated_image_path(number, spec):
    """Write the real, just-AI-generated image path(s) back into the spec
    after a successful render -- see run_render's call site for why."""
    images = find_reference_images(number)
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    if spec.get("workflow") == "i2v" and len(images) == 1:
        rel = rel_path_str(images[0], DREAMS_ROOT)
        if spec.get("image_path") != rel:
            spec["image_path"] = rel
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    elif spec.get("workflow") == "fml2v" and len(images) == 3 \
            and {p.stem for p in images} == {"1", "2", "3"}:
        images_by_stem = {p.stem: p for p in images}
        updates = {
            "fml2v_first_image": rel_path_str(images_by_stem["1"], DREAMS_ROOT),
            "fml2v_middle_image": rel_path_str(images_by_stem["2"], DREAMS_ROOT),
            "fml2v_last_image": rel_path_str(images_by_stem["3"], DREAMS_ROOT),
        }
        if any(spec.get(k) != v for k, v in updates.items()):
            spec.update(updates)
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def do_rework(numbers, randomize_seeds=False, type_arg=None, verbose=False, cancel_check=None):
    """type_arg: None/"keep" = use whatever the spec's own "workflow" field
    already says (the default -- decided back when the spec was written).
    Any other value ("t2v"/"i2v"/"fml") is an explicit override for just
    this rework -- if the spec doesn't already have that type's fields,
    skips this number and reports what's missing (ensure_workflow_type)
    rather than rendering with stale/wrong-type content.

    cancel_check: see do_generate's own docstring -- same "stop before
    the next number" check, for the same reason."""
    any_rendered = False
    for number in numbers:
        if cancel_check is not None and cancel_check():
            print(f"[dream_step] >>> cancelled -- stopping before #{number} "
                  f"(remaining: {numbers[numbers.index(number):]}).", flush=True)
            return True
        spec_path = DATA_DIR / f"spec_{number:03d}.json"
        if not spec_path.exists():
            print(f"[dream_step] >>> spec_{number:03d}.json does not exist -- "
                  f"nothing to rework. Write it first if this is meant to be a new dream.")
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        if type_arg and type_arg != "keep" and not ensure_workflow_type(number, spec, type_arg, kind="rework"):
            continue  # missing fields reported, skip this number for now

        # Running this only AFTER a successful render (see
        # migrate_uploaded_images' own docstring) would leave every case
        # other than the special-cased 1-of-3 fml2v branch below --
        # most commonly, replacing 2 of 3 (or all 3) keyframe images on a
        # Dream that's already been rendered once -- completely
        # unmigrated: find_reference_images below only looks at what's
        # actually IN the Dream folder already, so it would keep finding
        # the OLD images (all 3 still present from the previous render),
        # take the "already configured, nothing to do" branch, and
        # render with the stale images -- reporting success ("done")
        # while visually nothing had changed, since the newly staged
        # uploads were never even looked at. Migrating unconditionally
        # here, before find_reference_images runs, means any pending
        # staged upload always wins over what's currently in the folder,
        # for every branch below, not just the 1-of-3 special case.
        migrate_uploaded_images(number, spec)
        ref_images = find_reference_images(number)
        if (len(ref_images) == 1 and spec.get("workflow") == "fml2v"
                and ref_images[0].stem in ("1", "2", "3")
                and all((spec.get("fml2v_keyframe_prompts") or {}).get(r) for r in ("first", "middle", "last"))):
            # A single fml2v keyframe image provided (uploaded or manually
            # dropped in) with the other two roles still covered by
            # written keyframe prompts -- same legitimate-partial-set
            # reasoning as the 2-of-3 branch below, just for 1-of-3.
            # Without this branch, a human uploading just the
            # 'first' image (intending 'middle'/'last' to be AI-generated
            # off it) would hit the len==1 branch below instead, which
            # assumes ANY single image found means "switch this to i2v"
            # and refuses to render at all -- even though this is exactly
            # what fml2v with a partially-provided keyframe set looks like.
            #
            # generate_dream.py's generate_keyframes() decides whether to
            # skip regenerating a role purely by whether that role's file
            # already sits in the REAL Dream folder (dest_paths[role]),
            # not by any spec field -- and nothing in the render pipeline
            # ever migrates a still-staged upload into that real folder
            # BEFORE the render starts (migrate_uploaded_images only runs
            # AFTER a successful render). So this image needs migrating
            # into place, and its spec field set, right here, before
            # falling through to run_render below -- otherwise the
            # T2I/I2I pass would silently regenerate this role from
            # scratch, discarding the human-provided image entirely.
            migrate_uploaded_images(number, spec)
            ref_images = find_reference_images(number)
            role = {"1": "first", "2": "middle", "3": "last"}[ref_images[0].stem]
            field = {"first": "fml2v_first_image", "middle": "fml2v_middle_image",
                      "last": "fml2v_last_image"}[role]
            rel = rel_path_str(ref_images[0], DREAMS_ROOT)
            if spec.get(field) != rel:
                spec[field] = rel
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        elif len(ref_images) == 1:
            ref_image = ref_images[0]
            expected_path = Path(spec.get("image_path", "").replace("\\", "/"))
            if not expected_path.is_absolute():
                expected_path = DREAMS_ROOT / expected_path
            configured_for_this_image = (
                spec.get("workflow") == "i2v"
                and expected_path.resolve() == ref_image.resolve()
            )
            if not configured_for_this_image and spec.get("workflow") == "i2v":
                # workflow is ALREADY i2v -- the human already confirmed
                # image-conditioning intent when the spec was written;
                # image_path just went stale -- manually replacing the
                # image with a different filename/extension -- e.g.
                # dropping in a downloaded .jpg where the spec still
                # points at an old .png -- leaves image_path pointing at
                # a file that no longer exists. Refusing to render here
                # while the web UI's job still reports "done" would read
                # as "clicking render does nothing". Auto-repoint to
                # the real file and proceed -- same auto-detection
                # determine_code_owned_spec_fields already does at
                # spec-write time, just also applied here at render
                # time since a human replacing an image rarely goes
                # through a spec-write step first.
                spec["image_path"] = rel_path_str(ref_image, DREAMS_ROOT)
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                print(f"[dream_step] #{number}: image_path was stale (pointed at a file "
                      f"that no longer exists) -- auto-updated to the real file found in "
                      f"its Dream folder: {spec['image_path']}", flush=True)
            elif not configured_for_this_image:
                print(f"[dream_step] >>> #{number}: found a reference image at {ref_image} in "
                      f"its Dream folder -- this means the user wants the next render to use "
                      f"image-to-video conditioning on it instead of a fresh text-to-video "
                      f"render. To do that, edit spec_{number:03d}.json: set \"workflow\" to "
                      f"\"i2v\" and set \"image_path\" to \"{ref_image}\" -- positive_prompt/"
                      f"negative_prompt (already on the spec) are what render, same as any "
                      f"other workflow; since the image already fixes appearance, prefer "
                      f"describing the ANIMATION happening (dialogue, motion, camera "
                      f"behaviour) over re-describing appearance, and drop appearance/"
                      f"framing/anatomy terms from the negative prompt that the image "
                      f"already locks in, keeping only terms about behavior the video model "
                      f"still controls (see session notes / CREATIVE.md for the reasoning). "
                      f"Then run 'dream_step.py --rework {number}' again. Not rendering yet.", flush=True)
                continue
        elif len(ref_images) == 3 and {p.stem for p in ref_images} == {"1", "2", "3"}:
            # First-Middle-Last multi-keyframe i2v (fml2v workflow): the user
            # dropped exactly three images named 1/2/3 (first/middle/last
            # frame stills) into the Dream's folder -- use them as FML2V
            # guide images instead of a fresh render.
            images_by_stem = {p.stem: p for p in ref_images}
            keyframe_prompts = spec.get("fml2v_keyframe_prompts") or {}
            is_auto_generated = (
                spec.get("workflow") == "fml2v"
                and all(keyframe_prompts.get(r) for r in ("first", "middle", "last"))
            )
            configured_for_fml2v = (
                spec.get("workflow") == "fml2v"
                and spec.get("fml2v_first_image") and spec.get("fml2v_middle_image")
                and spec.get("fml2v_last_image")
            )
            # workflow is ALREADY explicitly "fml2v" -- that's unambiguous
            # confirmed intent already on record (same reasoning as the
            # i2v auto-repoint above), it just needs the three fields
            # synced to match what's actually in the folder. Refusing and
            # asking for a manual JSON edit here would be wrong whenever
            # workflow is already fml2v and the images are sitting right
            # there correctly named -- e.g. after migrate_uploaded_images
            # moved a staged upload in on an earlier render attempt,
            # leaving nothing left in staging to re-trigger a sync on a
            # later attempt. A human uploading through the GUI's own
            # first/middle/last slots, with the spec already marked
            # fml2v, has nothing left to confirm.
            if spec.get("workflow") == "fml2v" and not configured_for_fml2v and not is_auto_generated:
                role_field = {"1": "fml2v_first_image", "2": "fml2v_middle_image", "3": "fml2v_last_image"}
                changed = False
                for stem, field in role_field.items():
                    rel = rel_path_str(images_by_stem[stem], DREAMS_ROOT)
                    if spec.get(field) != rel:
                        spec[field] = rel
                        changed = True
                if changed:
                    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                    print(f"[dream_step] #{number}: fml2v_first_image/middle/last were stale "
                          f"or unset -- auto-synced to the real files found in its Dream "
                          f"folder.", flush=True)
                configured_for_fml2v = True
            # If the spec is set up for T2I/I2I auto-generation
            # (fml2v_keyframe_prompts), the 1/2/3 images sitting here are
            # this pipeline's OWN output from a previous render, not
            # something the user manually placed -- proceed straight to
            # regeneration (generate_dream.py's generate_keyframes will
            # overwrite them) instead of refusing, so iterating on keyframe
            # prompt wording doesn't get stuck behind this check forever.
            if not configured_for_fml2v and not is_auto_generated:
                print(f"[dream_step] >>> #{number}: found three images named 1/2/3 in its Dream "
                      f"folder ({images_by_stem['1'].name}, {images_by_stem['2'].name}, "
                      f"{images_by_stem['3'].name}) -- this means the user wants the next render "
                      f"to use first-middle-last multi-keyframe conditioning. To do that, edit "
                      f"spec_{number:03d}.json: set \"workflow\" to \"fml2v\" and set "
                      f"\"fml2v_first_image\"/\"fml2v_middle_image\"/\"fml2v_last_image\" to "
                      f"\"{images_by_stem['1']}\"/\"{images_by_stem['2']}\"/\"{images_by_stem['3']}\" "
                      f"-- positive_prompt/negative_prompt (already on the spec) are what "
                      f"render, same as any other workflow; since the images already fix "
                      f"appearance, prefer describing the ANIMATION happening across all "
                      f"three beats (dialogue, motion, camera behaviour) over re-describing "
                      f"appearance. Then run 'dream_step.py --rework {number}' again. Not "
                      f"rendering yet.",
                      flush=True)
                continue
        elif (len(ref_images) == 2 and spec.get("workflow") == "fml2v"
              and {p.stem for p in ref_images} <= {"1", "2", "3"}
              and all((spec.get("fml2v_keyframe_prompts") or {}).get(r) for r in ("first", "middle", "last"))):
            # A partial fml2v set -- exactly 2 of the 3 named 1/2/3 files
            # present, the missing one covered by a written keyframe
            # prompt. The manage table's "Use as..." slot reassignment
            # moves/swaps existing keyframe images between roles, which
            # can legitimately leave exactly one slot needing a fresh
            # generation while the other two (still-good, already-approved
            # poses) stay untouched -- that must not fall into the generic
            # "ambiguous, refuse" branch below since it's a perfectly
            # resolvable state.
            # generate_dream.py's generate_keyframes() already handles
            # this correctly per-role (only regenerates a role whose
            # image file is actually missing, via its own sidecar-based
            # role_changed() check) -- nothing more to do here, just
            # don't refuse.
            pass
        elif len(ref_images) > 1:
            # Multi-keyframe i2v (one generation per scene, stitched together)
            # was tried and abandoned -- stitching separate generations
            # together produced a different voice/style per clip that never
            # tied together, no matter how the prompts or frame allocation
            # were tuned. Only single-image i2v and three-image (1/2/3)
            # first-middle-last fml2v are supported now.
            print(f"[dream_step] >>> #{number}: found {len(ref_images)} image files in its Dream "
                  f"folder(s) ({[p.name for p in ref_images]}) -- ambiguous which are meant "
                  f"for conditioning (only a single image, for i2v, or exactly three named "
                  f"1/2/3, for fml2v, are supported). Remove the extras, then run this again.",
                  flush=True)
            continue

        collision = find_title_collision(number, spec.get("title"))
        if collision is not None:
            print(f"[dream_step] >>> REFUSING TO RENDER #{number} -- its title "
                  f"'{spec.get('title')}' is an EXACT match for spec_{collision:03d}.json's "
                  f"title. This would create a brand-new duplicate instead of a unique "
                  f"concept. Rewrite #{number}'s title AND premise/positive_prompt to be "
                  f"genuinely different from #{collision} (not just re-check the title -- "
                  f"the underlying idea must differ too), then run "
                  f"'dream_step.py --rework {number}' again.", flush=True)
            continue

        prior_folders = existing_dream_folders(number)
        if len(prior_folders) == 1:
            # A title change since the last render would otherwise make
            # run_render (generate_dream.py) create a BRAND NEW folder
            # under the new title, orphaning the old one under its stale
            # name instead of replacing it in place -- a rework batch that
            # only changes titles/premises (images untouched) would
            # otherwise leave every renamed Tale with two folders, one
            # dead. Rename the existing folder to match
            # up front so run_render's own folder_name computation lands
            # on it and overwrites cleanly, same as when the title didn't
            # change at all.
            import generate_dream
            label = get_episode_label()
            new_name = generate_dream.sanitize_filename(f"{label} #{number} {spec.get('title', '')}")
            old_name = prior_folders[0]
            if new_name != old_name:
                old_path = DREAMS_ROOT / old_name
                new_path = DREAMS_ROOT / new_name
                if not new_path.exists():
                    old_path.rename(new_path)
                    # The rename above moves the FOLDER, not the output
                    # files inside it -- the previous render's .mp4/.txt are
                    # still named after the OLD title (run_render/
                    # generate_dream.py names them from the folder's own
                    # name at render time), so they just ride along under
                    # their old filenames instead of being replaced. The
                    # render that's about to happen writes fresh
                    # {new_name}.mp4/.txt alongside them, leaving both the
                    # old-titled output AND the new one sitting in the same
                    # folder if left alone. Delete the
                    # old-named leftovers now so the folder only ever holds
                    # the current title's output once the render finishes.
                    for suffix in (".mp4", ".txt"):
                        stale = new_path / f"{old_name}{suffix}"
                        if stale.exists():
                            stale.unlink()
                    # Any code-owned image path field pointing INSIDE the
                    # renamed folder (image_path for i2v; fml2v_first_image/
                    # middle/last for fml2v) still has the OLD folder name
                    # baked into its stored path string -- left as-is, the
                    # very next render step fails with "does not exist"
                    # even though the file is right there under its new
                    # folder name. Rewrite each such field's leading
                    # folder-name segment
                    # to match, same rename, not a content change.
                    # Normalizes to forward-slash for BOTH the comparison
                    # and the rewritten value (matches rel_path_str's own
                    # convention) -- a value that happens to still have
                    # legacy backslashes (see rel_path_str's docstring)
                    # gets fixed to forward slashes here too, not just
                    # repointed to the new folder name.
                    repointed = False
                    for field in ("image_path", "fml2v_first_image",
                                  "fml2v_middle_image", "fml2v_last_image"):
                        value = spec.get(field)
                        if value:
                            norm_value = str(value).replace("\\", "/")
                            if norm_value.startswith(old_name + "/"):
                                spec[field] = new_name + norm_value[len(old_name):]
                                repointed = True
                    if repointed:
                        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                    print(f"[dream_step] #{number}: title changed -- renamed its Dream "
                          f"folder from {old_name!r} to {new_name!r} instead of leaving "
                          f"the old one orphaned."
                          + (" Also repointed its image path field(s) to the new folder "
                             "name." if repointed else ""), flush=True)
        elif len(prior_folders) > 1:
            print(f"[dream_step] WARNING: #{number} already has {len(prior_folders)} "
                  f"different Dream folders on disk from earlier reworks: {prior_folders}. "
                  f"This usually means it's already been reworked at least once before "
                  f"(possibly in an earlier session you don't have visibility into). "
                  f"Rendering again anyway since the content genuinely differs from the "
                  f"last recorded render, but consider whether the most recent one already "
                  f"satisfies the request before assuming another change is needed.",
                  flush=True)
        ok = run_render(number, event_type="rework", randomize_seeds=randomize_seeds, verbose=verbose,
                         cancel_check=cancel_check)
        if ok:
            any_rendered = True
        else:
            print(f"[dream_step] >>> #{number} render FAILED (see output above). Stopping.")
            return False
    if not any_rendered:
        print("[dream_step] Nothing was rendered this run.")
    else:
        print("[dream_step] Rework call complete.")
    return True


ALL_NUMBERS = object()  # sentinel: spec_str was "all"/"*" -- caller must
                         # resolve this against ITS OWN valid set and echo
                         # back what it resolved to, never act on it silently.


def parse_number_spec(spec_str):
    """Parse 'x', 'x-y', a comma-separated mix (e.g. '82,84-86'), or
    'all'/'*' into a sorted list of unique ints -- or the ALL_NUMBERS
    sentinel, which the caller must resolve against whatever set is valid
    for ITS OWN action (e.g. "all un-rendered specs" for --generate, "all
    rendered specs" for --rework) and print what it resolved to, so a
    typed 'all' is never a silent surprise."""
    if spec_str.strip().lower() in ("all", "*"):
        return ALL_NUMBERS
    numbers = set()
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return sorted(numbers)


def resolve_all(numbers, all_candidates, label):
    """If numbers is the ALL_NUMBERS sentinel, resolve it to all_candidates
    and print what it resolved to. Otherwise pass through unchanged."""
    if numbers is ALL_NUMBERS:
        numbers = sorted(all_candidates)
        print(f"[dream_step] 'all' resolved to {label}: {format_number_ranges(numbers)}", flush=True)
    return numbers


def format_number_ranges(numbers):
    """Compact display for a sorted number list -- '1-83' instead of
    spelling out 83 individual numbers, which is unreadable in a --status
    menu meant to be relayed straight to a human."""
    numbers = sorted(numbers)
    if not numbers:
        return "none"
    parts = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def do_upload(numbers, force):
    """Call upload_dream.py for each listed number, in order, stopping on
    the first failure -- same discipline as do_rework. upload_dream.py
    itself refuses to re-upload an already-published number unless --force
    is passed, so accidentally re-listing an already-uploaded number is
    safe by default (prints a message, does not create a duplicate video)."""
    any_uploaded = False
    for number in numbers:
        print(f"[dream_step] uploading #{number} via upload_dream.py", flush=True)
        cmd = [sys.executable, str(PIPELINE_DIR / "upload_dream.py"),
               "--project", PROJECT_DIR.name, "--number", str(number)]
        if force:
            cmd.append("--force")
        result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
        if result.returncode == 0:
            any_uploaded = True
        else:
            print(f"[dream_step] >>> #{number} upload did not succeed (see output above) "
                  f"-- this may be an intentional skip (already published) rather than a "
                  f"real failure; check the JSON line it printed. Stopping here either way "
                  f"rather than continuing past it silently.")
            return
    if not any_uploaded:
        print("[dream_step] Nothing was uploaded this run.")
    else:
        print("[dream_step] Upload call complete.")


UPLOAD_TEMPLATE_REQUIRED_FIELDS = (
    "channel_handle", "episode_label", "category_id", "privacy_status", "default_language")


def upload_template_path():
    return DATA_DIR / "youtube" / "upload_template.json"


def load_upload_template():
    """Read this project's upload_template.json, if present. Returns
    (template_dict_or_None, error_or_None) -- error is set whenever the
    file is missing, isn't valid JSON, or is missing a field upload_dream.py
    actually needs. The web UI's upload tab uses this to decide whether to
    show the editable form (pre-filled) or a "needs setup" prompt instead of
    letting someone attempt an upload with a broken/absent template."""
    path = upload_template_path()
    if not path.exists():
        return None, "no upload_template.json exists yet for this project"
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"upload_template.json is not valid JSON: {e}"
    missing = [f for f in UPLOAD_TEMPLATE_REQUIRED_FIELDS if not template.get(f)]
    schedule = template.get("schedule")
    if schedule is not None and not isinstance(schedule, dict):
        missing.append("schedule (must be an object)")
    elif isinstance(schedule, dict) and schedule.get("enabled") and not schedule.get("days_of_week"):
        missing.append("schedule.days_of_week (required while schedule.enabled is true)")
    if missing:
        return template, f"upload_template.json is missing/invalid fields: {missing}"
    return template, None


def write_upload_template(fields):
    """Write this project's upload_template.json from a flat fields dict
    (as gathered by the web UI's upload-template form) -- same shape
    do_new_project writes when scaffolding a brand-new project, so
    upload_dream.py needs no changes to read either. Used both to create a
    missing/malformed template and to edit an existing one -- always
    rewrites the whole file from the form's current values rather than
    patching, so there's never a stale field left over from hand-editing."""
    path = upload_template_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "_comment": "Channel-level defaults for uploads to THIS project only. "
                    "Title/description/tags per video come from that video's own "
                    "spec_NNN.json -- do not put per-video content here.",
        "channel_handle": fields["channel_handle"],
        "episode_label": fields["episode_label"],
        "category_id": fields["category_id"],
        "privacy_status": fields["privacy_status"],
        "privacy_status_note": "private | unlisted | public. Applies to any number NOT "
                                "covered by the schedule below. Scheduled numbers are "
                                "forced to private+publishAt regardless of this value.",
        "made_for_kids": bool(fields.get("made_for_kids")),
        "embeddable": bool(fields.get("embeddable", True)),
        "license": fields.get("license") or "youtube",
        "default_language": fields["default_language"],
        "contains_synthetic_media": bool(fields.get("contains_synthetic_media")),
        "contains_synthetic_media_note": "YouTube's mandatory AI-content disclosure. Only "
                                          "required when content depicts a real person, "
                                          "event, or place in a way that could be mistaken "
                                          "for reality -- confirm this with the user per "
                                          "video, don't assume it copies from another project.",
        "description_footer": fields.get("description_footer") or "",
        "default_tags": [t.strip() for t in (fields.get("default_tags") or "").split(",") if t.strip()],
        "schedule": {
            "enabled": bool(fields.get("schedule_enabled", True)),
            "anchor_number": int(fields.get("schedule_anchor_number") or 1),
            "anchor_date": fields.get("schedule_anchor_date") or "",
            "days_of_week": [d.strip().capitalize()
                              for d in (fields.get("schedule_days") or "").split(",") if d.strip()],
            "time_of_day_local": fields.get("schedule_time_of_day") or "00:00:00",
            "timezone": fields.get("schedule_timezone") or "Europe/Zurich",
        },
    }
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template


def do_check(numbers):
    """Call upload_dream.py --check-only for each listed number -- fetches
    the live video and diffs it against what the spec/template currently
    intend, without uploading anything. Does not stop on a mismatch (that's
    the expected way this reports problems); only stops on a hard error."""
    any_checked = False
    all_ok = True
    for number in numbers:
        print(f"[dream_step] checking #{number} via upload_dream.py --check-only", flush=True)
        cmd = [sys.executable, str(PIPELINE_DIR / "upload_dream.py"),
               "--project", PROJECT_DIR.name, "--number", str(number), "--check-only"]
        result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
        any_checked = True
        if result.returncode != 0:
            all_ok = False
    if not any_checked:
        print("[dream_step] Nothing was checked this run.")
    elif all_ok:
        print("[dream_step] Check call complete -- all clean.")
    else:
        print("[dream_step] Check call complete -- see mismatches above.")


def do_update_metadata(numbers):
    """Call upload_dream.py --update-metadata for each listed number --
    pushes freshly-built title/description/tags/status (from that number's
    CURRENT spec_NNN.json + the project's upload_template.json) to the
    already-uploaded video via videos.update. Does not touch the video
    file itself, so this is the right tool for "fix/change the title,
    description, or tags of an already-uploaded number" -- edit the
    relevant field(s) in spec_NNN.json first, then run this. Editing
    title/description/tags does NOT change the render content hash
    (only premise/positive_prompt/negative_prompt do), so this never
    triggers or requires a re-render."""
    any_updated = False
    for number in numbers:
        print(f"[dream_step] updating #{number}'s live metadata via upload_dream.py --update-metadata", flush=True)
        cmd = [sys.executable, str(PIPELINE_DIR / "upload_dream.py"),
               "--project", PROJECT_DIR.name, "--number", str(number), "--update-metadata"]
        result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
        if result.returncode == 0:
            any_updated = True
        else:
            print(f"[dream_step] >>> #{number} metadata update did not succeed (see output above). "
                  f"Stopping here rather than continuing past it silently.")
            return
    if not any_updated:
        print("[dream_step] Nothing was updated this run.")
    else:
        print("[dream_step] Metadata update call complete.")


REQUIRED_NEW_PROJECT_FIELDS = [
    "channel_handle", "schedule_anchor_date", "schedule_days", "episode_label",
]


# 2026-08-12: CREATIVE.md was cut down to project-FACTS only (genre,
# visual style) -- every rule/process/worked-example section that used to
# live here (content modes, complexity budget, voice/role pool, dedup
# approach, worked examples, etc.) moved to the shared, pipeline-wide
# golden_rules.md instead (see that file's own header for why: several of
# those "channel-specific" rules turned out to be general render-quality
# fixes, not actually per-project). A fresh project's CREATIVE.md now only
# needs these two facts -- project_genre_and_styles() parses them back out
# by the exact marker lines shown here.
def read_creative_md():
    """This project's current, live CREATIVE.md. No size cap (unlike
    creative_guidance_pointer, which caps for prompt-embedding purposes)
    -- callers here need the exact full text to parse fields/template out
    of, not a truncated preview."""
    path = DATA_DIR / "CREATIVE.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def build_creative_draft_payload(project_name, concept):
    """AI drafts genre/visual-style FIELDS (not raw markdown, and not
    concept_directive -- see below) from a one-line concept -- 2026-08-12:
    the Creative tab moved from a free-text markdown editor to a real
    FORM (see creative_fields/compose_creative_md), so drafting now needs
    to fill that form's fields directly, not hand back a whole document a
    human then has to read/reconcile against the form. Used right after a
    brand-new project is created, and later from the Creative tab to
    redraft an existing project's facts from scratch -- the result only
    ever fills the FORM (never written to CREATIVE.md itself until the
    human clicks Save), same human-approved-only guarantee as before.

    Deliberately does NOT draft concept_directive: that field is a real,
    functional standing instruction fed into every story generated for
    this project (see project_concept_directive/$concept_directive) --
    the human should write that intentionally if they want one, not have
    the AI invent a directive that then silently governs everything it
    writes afterward.

    concept can be blank -- same "blank means the AI has full creative
    freedom" pattern as concept_directive/build_simple_spec_prompt's
    title, not an error case (the button that triggers this used to
    hard-block with an alert on empty input; that was a real
    inconsistency with how every other blank-is-valid field behaves)."""
    if concept:
        concept_clause = f"this concept: {concept!r}"
        tailored_clause = f"tailored specifically to {concept!r}, not generic"
    else:
        concept_clause = "a concept of your own choosing -- invent something original and specific, not generic"
        tailored_clause = "specific and opinionated, not generic filler"
    return {
        "project": project_name,
        "concept": concept,
        "schema_hint": {
            "genre": "string -- one word or short phrase, e.g. 'Comedy'",
            "style1": "string -- a full visual-style description sentence",
            "style2": "string, optional -- a second, genuinely different visual "
                      "style a human can pick between per Tale; empty string if "
                      "the concept clearly calls for only one",
        },
        "instructions": (
            f"Propose genre/visual-style facts for a new short-form AI-animated-video "
            f"channel with {concept_clause}. This is a FIRST DRAFT a human will review "
            f"before it governs anything -- keep every field short and concrete, "
            f"{tailored_clause}. Do NOT write rules, process, worked examples, or a "
            f"voice pool -- every mechanical/render-quality rule already applies "
            f"pipeline-wide via a shared golden_rules.md, not something this draft "
            f"needs to state or reinvent."
        ),
    }


def draft_creative_fields(project_name, concept):
    payload = build_creative_draft_payload(project_name, concept)
    # Drafting these facts is a style task -- format_rules.md's mechanical
    # rules already apply to every project regardless of what's drafted
    # here, so they don't belong duplicated into the draft itself either.
    prompt = _render_creative_prompt(payload, include_format_rules=False)
    content, history = _creative_completion(prompt)
    if not (content.get("genre") or content.get("style1")):
        raise SystemExit("[dream_step] model returned an empty creative draft.")
    return {
        "genre": (content.get("genre") or "Comedy").strip(),
        "style1": (content.get("style1") or "").strip(),
        "style2": (content.get("style2") or "").strip(),
    }


def save_creative_fields(project_name, genre, style1, style2, duration_s, resolution, concept_directive, template_body):
    """The only path that writes the REAL CREATIVE.md -- takes the
    Creative tab FORM's current field values (a human clicking Save in
    their own tool, not an agent writing the file directly -- CREATIVE.md
    stays outside any agent's write permissions everywhere else) and
    composes+writes it via compose_creative_md, the single source for
    this file's shape."""
    display_name = project_name
    text = compose_creative_md(display_name, genre, [style1, style2], duration_s,
                                resolution, concept_directive, template_body)
    target = DATA_DIR / "CREATIVE.md"
    target.write_text(text, encoding="utf-8")
    return target


def do_new_project(name, args):
    """Create a new sibling project folder + its upload_template.json.
    Every value in the template comes from an explicit CLI flag (which the
    model gathers from the user in conversation first) -- this function
    never invents a channel handle, schedule, or policy choice on its own.
    Required flags are checked up front and the whole thing refuses to
    create a half-configured project if any are missing, rather than
    writing a template full of placeholder guesses."""
    project_dir = (projects_root() / name).resolve()
    if project_dir.exists():
        print(f"[dream_step] >>> {project_dir} already exists -- refusing to overwrite. "
              f"If this is really a fresh project, pick a different name or remove the "
              f"existing folder yourself first.")
        return

    missing = [f for f in REQUIRED_NEW_PROJECT_FIELDS
               if getattr(args, f) is None]
    if missing:
        print(f"[dream_step] >>> Cannot create project '{name}' yet -- missing required "
              f"details: {missing}. Ask the user for these specifically (channel handle, "
              f"first scheduled upload date, which days of the week it publishes on, and "
              f"what to call each episode/entry -- e.g. 'Dream' or 'Tale', used for output "
              f"folder/file naming as '<label> #N <title>'), then run this again with all "
              f"values supplied. Do not guess or default them.")
        return

    data_dir = project_dir / "_data"
    youtube_dir = data_dir / "youtube"
    youtube_dir.mkdir(parents=True)

    def as_bool(s):
        return str(s).strip().lower() in ("1", "true", "yes")

    template = {
        "_comment": "Channel-level defaults for uploads to THIS project only. "
                    "Title/description/tags per video come from that video's own "
                    "spec_NNN.json -- do not put per-video content here.",
        "channel_handle": args.channel_handle,
        "episode_label": args.episode_label,
        "category_id": args.category_id,
        "privacy_status": args.privacy_status,
        "privacy_status_note": "private | unlisted | public. Applies to any number NOT "
                                "covered by the schedule below. Scheduled numbers are "
                                "forced to private+publishAt regardless of this value.",
        "made_for_kids": as_bool(args.made_for_kids),
        "embeddable": True,
        "license": "youtube",
        "default_language": args.default_language,
        "contains_synthetic_media": as_bool(args.contains_synthetic_media),
        "contains_synthetic_media_note": "YouTube's mandatory AI-content disclosure. Only "
                                          "required when content depicts a real person, "
                                          "event, or place in a way that could be mistaken "
                                          "for reality -- confirm this with the user per "
                                          "video, don't assume it copies from another project.",
        "description_footer": args.description_footer or "",
        "default_tags": [t.strip() for t in (args.default_tags or "").split(",") if t.strip()],
        "schedule": {
            "enabled": True,
            "anchor_number": args.schedule_anchor_number,
            "anchor_date": args.schedule_anchor_date,
            "days_of_week": [d.strip().capitalize() for d in args.schedule_days.split(",") if d.strip()],
            "time_of_day_local": args.time_of_day,
            "timezone": args.timezone,
        },
    }
    (youtube_dir / "upload_template.json").write_text(
        json.dumps(template, indent=2), encoding="utf-8")

    # Scaffold the rest of a real project: empty index.json (machine-owned
    # from here on), Reviewed/ for finalized videos, and a CREATIVE.md STUB.
    # 2026-08-12: CREATIVE.md's job narrowed to project-FACTS (genre,
    # visual style) plus its own live copy of the spec-prompt template
    # (see PROMPT_TEMPLATE_SECTION_HEADER/project_prompt_template) --
    # every mechanical/render-quality RULE moved to the shared
    # golden_rules.md instead, so a fresh project's stub no longer needs
    # to scaffold sections for those. Genre/visual style come pre-filled
    # with sensible defaults (this pipeline's own proven starting point)
    # that a human is expected to edit, not blank placeholders -- a new
    # project should work immediately, tweak from there. Never
    # agent-authored: matches the existing "never write/edit CREATIVE.md"
    # rule, this just gives the human a real, already-working starting
    # point via code, not the model.
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("[]", encoding="utf-8")
    (project_dir / "Reviewed").mkdir(exist_ok=True)
    creative_stub_path = data_dir / "CREATIVE.md"
    if not creative_stub_path.exists():
        creative_stub_path.write_text(
            compose_creative_md(
                name, genre="Comedy", styles=list(_FALLBACK_STYLE_OPTIONS),
                duration_s=_FALLBACK_RENDER_DURATION_S,
                resolution=f"{_FALLBACK_RENDER_WIDTH}x{_FALLBACK_RENDER_HEIGHT}",
                concept_directive="", template_body=default_spec_prompt_template()),
            encoding="utf-8")

    print(f"[dream_step] >>> Created project '{name}' at {project_dir}, with "
          f"{youtube_dir / 'upload_template.json'} configured, plus an empty index.json, "
          f"a Reviewed/ folder, and a CREATIVE.md stub for a human to fill in. Still needed "
          f"before this project can actually upload: client_secret.json.enc is already shared "
          f"at _pipeline/youtube/ (nothing to do there, once set up in Settings), but this "
          f"project has no token.json.enc yet -- the first upload or check for this project "
          f"will trigger a one-time "
          f"browser authorization that a HUMAN must approve. Tell the user this is next, "
          f"then run --project {name} --status.")


def compute_status(project_name):
    """Ground-truth state inspection, shared by --status (prints it for an
    agent to relay) and --interactive (uses it directly to build its own
    menu in-process, no printing/parsing round trip needed). Returns None
    if project_name is falsy (caller should list projects instead) --
    otherwise a dict: specced/rendered/uploaded/not_rendered/
    image_workflow_specs/rendered_not_uploaded (all sorted lists)."""
    if not project_name:
        return None
    index = load_json(INDEX_PATH, [])
    specced = sorted(int(p.stem.split("_")[1]) for p in DATA_DIR.glob("spec_*.json"))
    rendered, uploaded = set(), set()
    for e in index:
        if not isinstance(e, dict) or e.get("number") is None:
            continue
        rendered.add(e["number"])
        if e.get("published"):
            uploaded.add(e["number"])
    not_rendered = [n for n in specced if n not in rendered]
    image_workflow_specs = []
    for n in specced:
        spec = load_json(DATA_DIR / f"spec_{n:03d}.json", {})
        if spec.get("workflow") in ("i2v", "fml2v"):
            image_workflow_specs.append(n)
    rendered_not_uploaded = sorted(n for n in rendered if n not in uploaded)
    concept_path, concept_total, concept_remaining = concept_list_stats()
    _, upload_template_error = load_upload_template()
    return {
        "specced": specced,
        "rendered": sorted(rendered),
        "uploaded": sorted(uploaded),
        "not_rendered": not_rendered,
        "image_workflow_specs": image_workflow_specs,
        "rendered_not_uploaded": rendered_not_uploaded,
        "concept_list_name": concept_path.name,
        "concept_list_path": str(concept_path.relative_to(projects_root())),
        "concept_list_total": concept_total,
        "concept_list_remaining": concept_remaining,
        "upload_template_error": upload_template_error,
    }


def list_existing_projects():
    return sorted(p.name for p in projects_root().iterdir()
                  if p.is_dir() and p.name != "_pipeline" and (p / "_data").is_dir())


def _validate_project_folder_name(name):
    """Rejects anything that could resolve outside the projects root or
    collide with the pipeline's own directory -- shared by rename_project/
    delete_project so a malformed name can't escape where projects are
    supposed to live. Returns the resolved Path."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or name == "_pipeline":
        raise SystemExit(f"[dream_step] {name!r} is not a valid project name.")
    root = projects_root()
    resolved = (root / name).resolve()
    if resolved.parent != root:
        raise SystemExit(f"[dream_step] {name!r} is not a valid project name.")
    return resolved


def _retry_on_windows_lock(fn, attempts=6, delay=0.3):
    """Windows can transiently deny a rename/delete on a directory with a
    PermissionError ([WinError 5]) even when nothing in THIS process has
    anything under it open -- a rename that fails here can succeed
    instantly from a brand-new process moments later, no code change.
    That points at an OS-level transient hold
    (search indexer, AV real-time scan, a just-closed handle not yet
    released) rather than a real conflict -- short retries absorb it
    instead of failing outright on what's usually a sub-second race."""
    last_err = None
    for _ in range(attempts):
        try:
            return fn()
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def rename_project(old_name, new_name):
    """Renames a project's folder on disk. Every path in this pipeline is
    resolved from the project name at call time (resolve_project_globals),
    never hardcoded into a stored file, so a plain folder rename is
    sufficient -- nothing inside needs editing."""
    old_dir = _validate_project_folder_name(old_name)
    new_dir = _validate_project_folder_name(new_name)
    if not old_dir.is_dir():
        raise SystemExit(f"[dream_step] project folder does not exist: {old_dir}")
    if new_dir.exists():
        raise SystemExit(f"[dream_step] {new_name!r} already exists -- pick a different name.")
    _retry_on_windows_lock(lambda: old_dir.rename(new_dir))
    return new_dir


def delete_project(name):
    """Permanently deletes a project's entire folder -- every spec,
    render, upload record, and credential for it. No undo; the web UI is
    expected to demand a strong (typed-name) confirmation before ever
    calling this."""
    project_dir = _validate_project_folder_name(name)
    if not project_dir.is_dir():
        raise SystemExit(f"[dream_step] project folder does not exist: {project_dir}")
    _retry_on_windows_lock(lambda: shutil.rmtree(project_dir))


def resolve_project_globals(name):
    """Validate a project name and set the module-level PROJECT_DIR/
    DATA_DIR/etc globals from it -- shared by main()'s direct-flag
    dispatch and run_interactive(), so both paths validate identically."""
    global PROJECT_DIR, DATA_DIR, DREAMS_ROOT, INDEX_PATH, HISTORY_PATH
    PROJECT_DIR = (projects_root() / name).resolve()
    if not PROJECT_DIR.is_dir() or name == "_pipeline":
        raise SystemExit(
            f"[dream_step] project folder does not exist: {PROJECT_DIR}\n"
            f"EXPECTED: --project must exactly match a real sibling folder name.\n"
            f"TO FIX: use one of the existing projects: {list_existing_projects()}. "
            f"Never create a folder literally called \"Project\" -- that is a "
            f"placeholder label, not a real folder name.")
    DATA_DIR = PROJECT_DIR / "_data"
    DATA_DIR.mkdir(exist_ok=True)
    DREAMS_ROOT = PROJECT_DIR
    INDEX_PATH = DATA_DIR / "index.json"
    HISTORY_PATH = DATA_DIR / "rework_history.json"


def do_status(project_name):
    """The mandatory first call of every session for the agent-dispatch
    path (superseded as the PRIMARY way to run this pipeline by
    --interactive, 2026-08-07 -- see the plan's Phase 2 -- but kept as a
    read-only inspection command and for any headless/scripted dispatch
    to a separate agent process). Prints ONLY the menu options that are
    actually possible right now, from ground truth on disk, not from an
    agent's own judgment."""
    if not project_name:
        existing = list_existing_projects()
        if not existing:
            print("[dream_step] STATUS: no projects exist yet.")
            print("ONLY VALID OPTION: 1. New project")
            print("NEXT: ask the user for a project name, then run:")
            print("  python dream_step.py --new-project <name> --channel-handle <handle> "
                  "--episode-label <label> --schedule-anchor-date <YYYY-MM-DD> "
                  "--schedule-days <comma-separated days>")
            return
        print("[dream_step] STATUS: existing projects:")
        for i, pname in enumerate(existing, 1):
            print(f"  {i}. {pname}")
        print(f"  {len(existing) + 1}. New project")
        print("NEXT: ask the user which one (by number or name), then run:")
        print("  python dream_step.py --project <name> --status")
        return

    s = compute_status(project_name)
    print(f"[dream_step] STATUS for project '{project_name}':")
    print(f"  specs written: {format_number_ranges(s['specced'])}")
    print(f"  rendered: {format_number_ranges(s['rendered'])}")
    print(f"  uploaded: {format_number_ranges(s['uploaded'])}")
    print(f"  concept list: {s['concept_list_path']} "
          f"({s['concept_list_total']} entries, {s['concept_list_remaining']} remaining)")

    print()
    print("VALID OPTIONS -- relay this menu to the user verbatim, do not offer "
          "anything not listed here:")
    opt = 1
    print(f"  {opt}. Gen/Regen spec -- AI-composed or direct spec content, one or "
          f"more numbers. This is done through the web UI's manage table (Run "
          f"updates), not a CLI flag -- run:")
    print(f"       python dream_step.py --project {project_name} --web")
    print(f"     (--write-spec --spec-json-stdin is also available for direct/"
          f"scripted single-number writes, no AI involved.)")
    opt += 1
    if s["not_rendered"]:
        print(f"  {opt}. Generate video -- render number(s) that have a spec but no "
              f"render yet: {format_number_ranges(s['not_rendered'])}. Ask which "
              f"number(s); ask whether to use the spec's own graph type (default) or "
              f"override it, then run:")
        print(f"       python dream_step.py --project {project_name} --generate <numbers> [--type <t2v|i2v|fml>]")
        opt += 1
    if s["rendered"]:
        print(f"  {opt}. Rework video -- re-render already-rendered number(s): "
              f"{format_number_ranges(s['rendered'])}. Ask which number(s); ask "
              f"whether to use the spec's own graph type (default) or override it, "
              f"then run:")
        print(f"       python dream_step.py --project {project_name} --rework <numbers> [--type <t2v|i2v|fml>]")
        opt += 1
    if s["image_workflow_specs"]:
        print(f"  {opt}. Gen keyframe images -- (re)generate reference image(s) for "
              f"i2v/fml specs: {format_number_ranges(s['image_workflow_specs'])}. Done "
              f"through the web UI's manage table (Run updates, K chip) -- run:")
        print(f"       python dream_step.py --project {project_name} --web")
        opt += 1
    if s["rendered_not_uploaded"]:
        print(f"  {opt}. Upload to YouTube -- rendered but not yet uploaded: "
              f"{format_number_ranges(s['rendered_not_uploaded'])}. Ask which "
              f"number(s), then run:")
        print(f"       python dream_step.py --project {project_name} --upload <numbers>")
        opt += 1
    print(f"  {opt}. New project -- ask for a name, then run --new-project.")


def with_vram_guard(fn, *args, **kwargs):
    """Wrap a render-triggering call (do_generate/do_rework) with the
    reload guard + post-call VRAM cleanup, so --interactive
    gets the exact same VRAM discipline on every render it triggers
    inside its loop, not just once at process exit."""
    cfg = vram_guard.load_config()
    guard_stop = vram_guard.start_reload_guard(cfg)
    try:
        return fn(*args, **kwargs)
    finally:
        guard_stop.set()
        # Confirmed: ComfyUI can sit holding several GB of VRAM after a
        # render finishes even though nothing is queued anymore -- left
        # unaddressed, that VRAM stays held into whatever session starts
        # next instead of being available to it. Free it here, every
        # time a render/rework call ends, not just at the next session's
        # own startup.
        print("[dream_step] freeing VRAM now that rendering is done for this call...", flush=True)
        vram_guard.comfyui_free_vram(cfg)
        vram_guard.ollama_stop_model(cfg)
        if cfg.get("vision_model"):
            vram_guard.ollama_stop_model(cfg, model_name=cfg["vision_model"])


def do_generate(numbers, type_arg, verbose=False, cancel_check=None):
    """Replaces the old --range-end continuation mode. Renders EXACTLY the
    numbers given -- nothing implicit, nothing continued from a persisted
    file. Confirmed root cause this removes: the old do_batch() silently
    used agent_memory.json's stored range_end instead of the CLI argument
    once that file existed once -- a stale range_end: 365 from early in
    this project's history caused --range-end 83 to render #84 and #85 on
    top of it, unasked, before being caught (2026-08-06).

    cancel_check: see run_render's own docstring -- checked here too,
    BEFORE starting each number, so a mid-batch Cancel actually stops
    the batch instead of just interrupting the render that happened to
    be in flight and then silently continuing on to the next number --
    without this check, a mid-batch Cancel keeps rendering later numbers
    since nothing else in the loop checks the flag between numbers."""
    for number in numbers:
        if cancel_check is not None and cancel_check():
            print(f"[dream_step] >>> cancelled -- stopping before #{number} "
                  f"(remaining: {numbers[numbers.index(number):]}).", flush=True)
            return True
        spec_path = DATA_DIR / f"spec_{number:03d}.json"
        if not spec_path.exists():
            print(f"[dream_step] >>> #{number}: no spec exists yet.\n"
                  f"EXPECTED: a spec must exist before generating.\n"
                  f"TO FIX: run: python dream_step.py --project {PROJECT_DIR.name} --write-spec {number}")
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        if type_arg and not ensure_workflow_type(number, spec, type_arg, kind="generate"):
            continue  # missing fields reported, skip this number for now

        ok = run_render(number, event_type="origin", randomize_seeds=True, verbose=verbose,
                         cancel_check=cancel_check)
        if not ok:
            print(f"[dream_step] >>> #{number} render FAILED (see output above). Stopping.")
            return False
    print(f"[dream_step] --generate complete for: {numbers}")
    return True


def ensure_workflow_type(number, spec, type_arg, kind):
    """Shared by --generate and --rework's --type handling. If the spec
    already matches the requested type and has all its type-specific
    fields, returns True (nothing to do). Otherwise returns False --
    caller should skip this number (so a batch with a mix of numbers
    still processes the ones that ARE ready) and report that the missing
    fields need to be composed first via the manage table's "Run
    updates" (AI-composed with the K/S chip, or typed in directly), not
    silently rendered with stale/wrong-type content."""
    if type_arg == "keep":
        return True
    target_workflow = TYPE_TO_WORKFLOW[type_arg]
    needed_fields = TYPE_SPECIFIC_FIELDS.get(target_workflow, [])
    already_set = (spec.get("workflow") == target_workflow
                   and all(spec.get(f) for f in needed_fields))
    if already_set:
        return True

    print(f"[dream_step] >>> #{number}: switching to {type_arg} needs {needed_fields}, "
          f"not yet set. TO FIX: load #{number} into the manage table, fill those "
          f"fields in (or tick the K/S AI chip to compose them), click Run updates, "
          f"then retry this render.", flush=True)
    return False


def _merge_and_validate_spec(number, updates):
    """The validation core of merge_and_write_spec, split out so it can
    also validate an AI-composed update WITHOUT writing it to disk (see
    _generate_keyframes_content). Returns the merged, validated spec
    dict (mutates nothing on disk)."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    existing.update(updates)
    existing.pop("number", None)
    return _validate_and_normalize_spec(number, existing, allow_custom_beats=True)


def merge_and_write_spec(number, updates):
    """Apply a partial field update to an existing spec, re-validating
    through the exact same path as a full --write-spec -- used by
    write_row_keyframes (AI-composed or locked-in keyframe prompts) so a
    partial update can never bypass the bracket-format/tags-string/
    required-field checks a full rewrite gets."""
    spec = _merge_and_validate_spec(number, updates)
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[dream_step] wrote {spec_path}", flush=True)


def save_guide_strengths(number, strengths):
    """Write fml2v_guide_strengths for a row, independent of
    write_row_keyframes' image-already-satisfied early return -- these
    weights need to be editable even when all 3 keyframe images already
    exist (the normal case), so they get their own always-writable path
    straight through merge_and_write_spec."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    if spec.get("workflow") != "fml2v":
        raise SystemExit("[dream_step] save_guide_strengths only applies to fml2v.")
    existing = spec.get("fml2v_guide_strengths") or {}
    if not isinstance(existing, dict):
        existing = {}
    clean = dict(existing)
    for role in ("first", "middle", "last"):
        if role in strengths and strengths[role] is not None:
            value = float(strengths[role])
            if not (0.0 <= value <= 1.0):
                raise SystemExit(f"[dream_step] guide strength for '{role}' must be between 0 and 1, got {value}")
            clean[role] = value
    merge_and_write_spec(number, {"fml2v_guide_strengths": clean})


def ask(prompt, default=None):
    """input() with a printed default hint and EOF handling -- used
    throughout run_interactive(). If stdin hits EOF (e.g. run
    non-interactively by mistake), exits cleanly with a clear message
    instead of crashing or, worse, silently proceeding with no answer."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise SystemExit(
            "\n[dream_step] >>> stdin closed while waiting for an answer.\n"
            "EXPECTED: --interactive needs a real human (or something piping real "
            "answers) typing responses -- it will not proceed on a guess.\n"
            "TO FIX: run this in an actual terminal, or pipe genuine answers to "
            "each prompt in order.")
    return answer if answer else (default or "")


def ask_multiline(prompt):
    """For long-form/pasted creative content (scripts, lyrics, anything
    with line breaks) -- plain input() only reads ONE line, so pasting
    multi-line text into a single ask() cuts it off after the first line
    and the rest bleeds into whatever prompt comes next as garbage answers.
    Reads lines until a lone line
    containing just EOF, joins them with real newlines preserved -- this
    is how you paste a whole script/lyrics block safely: paste it all,
    then on its own line type EOF and press enter."""
    print(f"{prompt}\n(paste as much as you want, including blank lines -- when "
          f"done, type a line with just EOF and press enter):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_interactive(project_name):
    """The primary way to run this pipeline (2026-08-07) -- a deterministic
    Python REPL, not an agent. Every branch below is decided by a real
    input() call; the local model is only ever invoked (via
    _creative_completion) as a bounded text-in/JSON-out function for the
    one leaf step that genuinely needs creative content, never to decide
    what happens next. See the plan's Phase 2 context for why: an agent
    given the equivalent choice twice fabricated an answer (an unasked-for
    number, an invented "note") rather than genuinely stopping to ask."""
    if not project_name:
        existing = list_existing_projects()
        if not existing:
            print("[dream_step] No projects exist yet.")
            name = ask("Project name to create")
            _interactive_new_project(name)
            return
        print("[dream_step] Existing projects:")
        for i, pname in enumerate(existing, 1):
            print(f"  {i}. {pname}")
        print(f"  {len(existing) + 1}. New project")
        choice = ask("Which project?")
        if choice == str(len(existing) + 1) or choice.lower() in ("new", "new project"):
            name = ask("Project name to create")
            _interactive_new_project(name)
            return
        if choice.isdigit() and 1 <= int(choice) <= len(existing):
            project_name = existing[int(choice) - 1]
        elif choice in existing:
            project_name = choice
        else:
            raise SystemExit(f"[dream_step] '{choice}' isn't one of {existing} or "
                              f"'new project'. Run --interactive again.")

    resolve_project_globals(project_name)

    while True:
        s = compute_status(project_name)
        print(f"\n[dream_step] --- {project_name} --- "
              f"specs: {format_number_ranges(s['specced'])} | "
              f"rendered: {format_number_ranges(s['rendered'])} | "
              f"uploaded: {format_number_ranges(s['uploaded'])}")
        options = [("Gen/Regen spec", "spec")]
        if s["not_rendered"]:
            options.append(("Generate video", "generate"))
        if s["rendered"]:
            options.append(("Rework video", "rework"))
        if s["image_workflow_specs"]:
            options.append(("Gen keyframe images", "keyframes"))
        if s["rendered_not_uploaded"]:
            options.append(("Upload to YouTube", "upload"))
        options.append(("Switch project", "switch"))
        options.append(("Quit", "quit"))

        for i, (label, _) in enumerate(options, 1):
            print(f"  {i}. {label}")
        choice = ask("Which option?")
        if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
            print(f"[dream_step] '{choice}' isn't a valid option number. Try again.")
            continue
        _, action = options[int(choice) - 1]

        if action == "quit":
            return
        if action == "switch":
            run_interactive(None)
            return
        if action == "spec":
            _interactive_spec(s)
        elif action == "generate":
            _interactive_generate_or_rework(s, is_rework=False)
        elif action == "rework":
            _interactive_generate_or_rework(s, is_rework=True)
        elif action == "keyframes":
            _interactive_keyframes(s)
        elif action == "upload":
            _interactive_upload(s)


def _ask_numbers(prompt, valid_candidates, strict=True):
    """Ask for a number list, parse it, and resolve 'all' against the
    SPECIFIC set of numbers valid for this action (never the whole
    project) -- echoes what 'all' resolved to so it's never a silent
    surprise, same discipline as the non-interactive resolve_all().

    strict=False (spec origination): valid_candidates only controls what
    'all' resolves to (the existing specced numbers) -- a number OUTSIDE
    that set is legitimate (writing a brand-new spec for an unspecced
    number is the whole point) and is NOT filtered out."""
    raw = ask(f"{prompt} (a number, 'N1,N2', 'N1-N2', or 'all')")
    if not raw:
        print("[dream_step] No numbers given -- nothing to do.")
        return []
    numbers = parse_number_spec(raw)
    numbers = resolve_all(numbers, valid_candidates, "your selection")
    if not strict:
        return numbers
    invalid = [n for n in numbers if n not in valid_candidates]
    if invalid:
        print(f"[dream_step] {invalid} aren't valid for this action "
              f"(valid: {format_number_ranges(valid_candidates)}) -- skipping them.")
    return [n for n in numbers if n in valid_candidates]


def _generate_spec_content(number, prompt, code_owned, max_validation_retries=3,
                            extra_locked_fields=None, verbose=False):
    """Call the model, strip/overwrite CODE_OWNED_SPEC_FIELDS, and validate
    through _validate_and_normalize_spec -- on a validation failure
    (bracket markers, dialogue count, etc.), feed the EXACT error back to
    the model and retry, same self-correction pattern _creative_completion
    already uses for malformed JSON. Confirmed necessary (2026-08-07):
    without this, a single validation failure just got reported to the
    human with no attempt to self-correct, even though the error message
    is specific and actionable enough for the model to fix on its own.

    Does NOT write anything to disk -- see _generate_and_write_spec (the
    real 'Save content' path, the only thing that calls this).

    extra_locked_fields: manage-table fields the human typed in directly
    (build_row_spec_payload already excluded these from what the model
    was asked for) -- merged back in here since the model's JSON answer
    only covers the fields it was actually asked to write.

    Returns the validated content dict on success, None if every attempt
    failed -- callers that need to distinguish "produced something" from
    "gave up" (the web table's Run updates, which otherwise reported
    ok:true even when nothing was actually written) should check this
    rather than assume a normal return means success."""
    # Same fix as write_row_spec, same reason (2026-08-17): the model's
    # JSON answer only ever covers the base fields it was asked to write
    # -- anything else already on disk (fml2v_guide_strengths, saved
    # separately via its own per-slot weight input) would otherwise be
    # silently dropped the instant an AI-composed save landed, since
    # _generate_and_write_spec writes whatever this function returns as
    # a full overwrite.
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing_on_disk = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    attempt_prompt = prompt
    for attempt in range(1, max_validation_retries + 1):
        try:
            content, history = _creative_completion(attempt_prompt)
        except RuntimeError as e:
            print(f"[dream_step] >>> #{number}: {e}")
            return None
        if verbose:
            print(f"[dream_step] #{number}: attempt {attempt} raw model response:\n"
                  f"{json.dumps(content, indent=2)}\n")
        content = {**existing_on_disk, **content}
        for field in CODE_OWNED_SPEC_FIELDS:
            content.pop(field, None)  # never trust the model's copy, even if present
        if extra_locked_fields:
            content.update(extra_locked_fields)  # human's own words win, verbatim
        content.update(code_owned)
        # Only soften the beat-structure check to a warning when
        # positive_prompt ITSELF was human-locked (typed directly, not
        # asked of the model) -- the check still hard-blocks whatever the
        # model composed on its own, but a human who deliberately wrote
        # their own dialogue/timing doesn't need it second-guessed just
        # because some OTHER field on the same row was AI-composed.
        positive_prompt_is_human = bool(extra_locked_fields and "positive_prompt" in extra_locked_fields)
        try:
            content = _validate_and_normalize_spec(number, content, positive_prompt_is_human=positive_prompt_is_human)
            return content
        except SystemExit as e:
            if attempt == max_validation_retries:
                print(f"[dream_step] >>> #{number}: model's answer failed validation "
                      f"even after {max_validation_retries} attempts:\n{e}")
                return None
            print(f"[dream_step] #{number}: attempt {attempt} failed validation, "
                  f"retrying with the error fed back...", flush=True)
            attempt_prompt = (f"{prompt}\n\nYour previous answer failed validation "
                               f"with this exact error -- fix it and answer again, "
                               f"full JSON object, same schema:\n{e}")


def _generate_and_write_spec(number, prompt, code_owned, max_validation_retries=3,
                              extra_locked_fields=None, verbose=False):
    """The real 'Save content' path -- generates via _generate_spec_content,
    then writes to disk. Returns True on a real write, False if every
    attempt failed validation (see _generate_spec_content)."""
    content = _generate_spec_content(number, prompt, code_owned, max_validation_retries,
                                      extra_locked_fields, verbose)
    if content is None:
        return False
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"[dream_step] #{number}: spec written.")
    return True


def determine_code_owned_spec_fields(number, workflow):
    """Figures out image_path (etc.) from what's actually on disk -- shared
    by the CLI (_interactive_spec) and the web UI, so this real-file-system
    detection logic exists exactly once. Returns (code_owned_dict, error)
    -- error is reserved for genuine failures; kept in the return shape
    for callers, but nothing below actually produces one anymore.

    Must not hard-block writing an i2v/fml2v spec until a reference image
    already exists on disk -- that would be a real deadlock, since the
    thing that GENERATES that image (write_row_keyframes) requires a spec
    to already exist first ("no spec exists yet"). A fresh i2v/fml2v row
    could never get past either step: no spec without an image, no image
    without a spec. image_path
    is only ever set here when a real file is already on disk; when none
    exists yet, it's simply left unset (never asked of the AI --
    CODE_OWNED_SPEC_FIELDS already keeps it out of schema_hint either
    way). check_image_prerequisites is the actual gate that blocks
    RENDERING (not spec-writing) until a real image or a generate-prompt
    exists -- it already handles "no image yet, but i2v_generate_image_
    prompt is set" as a normal, expected case, so spec-write time doesn't
    need its own separate, stricter copy of that same requirement.

    Calling find_reference_images straight away, without migrating first,
    would -- like migrate_uploaded_images' own docstring explains --
    ignore the uploads staging dir entirely once the real Dream folder
    already has SOME image for that slot. "Save content" (write_row_spec)
    then does a full spec replace using these fields, so that would
    silently overwrite a correctly-migrated fml2v_first_image/middle
    back to whatever the STALE folder contents implied -- from the
    human's side, clicking Save would appear to make a just-uploaded
    replacement image vanish. Migrating first means this always sees the
    current, real state, the same approach do_rework uses."""
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    existing_spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    migrate_uploaded_images(number, existing_spec)
    code_owned = {"workflow": workflow}
    if workflow == "i2v":
        ref_images = find_reference_images(number)
        if len(ref_images) == 1:
            code_owned["image_path"] = rel_path_str(ref_images[0], DREAMS_ROOT)
    elif workflow == "fml2v" and fml2v_images_satisfied(number):
        # Three real, correctly-named (1/2/3) images being present is
        # already enough for check_image_prerequisites to greenlight
        # rendering, but nothing else copies that into
        # fml2v_first_image/middle/last -- the only fields
        # generate_dream.py itself actually reads (it has no idea
        # find_reference_images or the uploads staging dir exist).
        # Without this, a first-time fml render with directly-uploaded
        # keyframe images fails at the subprocess with "neither existing
        # keyframe images nor a complete fml2v_keyframe_prompts", even
        # though the images are right there. Mirrors image_path's i2v handling
        # above exactly.
        images_by_stem = {p.stem: p for p in find_reference_images(number)}
        code_owned["fml2v_first_image"] = rel_path_str(images_by_stem["1"], DREAMS_ROOT)
        code_owned["fml2v_middle_image"] = rel_path_str(images_by_stem["2"], DREAMS_ROOT)
        code_owned["fml2v_last_image"] = rel_path_str(images_by_stem["3"], DREAMS_ROOT)
    return code_owned, None


def _interactive_spec(s):
    numbers = _ask_numbers("Which number(s) to write/overwrite the spec for?",
                            s["specced"], strict=False)
    if not numbers:
        return
    note = ask_multiline("Any specific creative direction for this? (blank for none)")
    note = note or None
    for number in numbers:
        existing_spec_path = DATA_DIR / f"spec_{number:03d}.json"
        existing_workflow = None
        if existing_spec_path.exists():
            existing_workflow = json.loads(existing_spec_path.read_text(encoding="utf-8")).get("workflow")

        # Graph type is a human decision, never the model's -- ask directly,
        # default to whatever's already there for a regen.
        type_choice = ask("Graph type -- t2v, i2v, or fml?",
                           default=WORKFLOW_TO_TYPE.get(existing_workflow, "t2v"))
        workflow = TYPE_TO_WORKFLOW.get(type_choice.strip().lower(), "fp8_t2v")

        code_owned, error = determine_code_owned_spec_fields(number, workflow)
        if error:
            print(f"[dream_step] >>> #{number}: {error} Skipping for now.")
            continue

        if using_strong_creative_backend():
            prompt = build_simple_spec_prompt(number, note, workflow)
        else:
            payload = build_spec_request_payload(number, note, workflow=workflow)
            prompt = _render_creative_prompt(payload)
        _generate_and_write_spec(number, prompt, code_owned)


def _generate_keyframes_content(number, prompt, max_validation_retries=3,
                                 extra_locked_fields=None, verbose=False):
    """Same self-correcting retry pattern as _generate_spec_content, for
    keyframe prompt fields -- shared by the CLI and the web UI. Does NOT
    write anything to disk; see _generate_and_write_keyframes (the real
    'Save content' path).

    extra_locked_fields: manage-table sub-fields (e.g. fml2v_keyframe_prompts'
    "first"/"middle") the human typed in directly -- merged into whichever
    of the model's answer is a dict at the same key (locked values win on
    conflict), since the model was only asked for the remaining sub-keys.

    Returns (merged_spec, update_fields) on success -- merged_spec is the
    full validated spec (for a preview to show), update_fields is just
    the keys this call was actually asked to produce (what a real write
    would merge in). Returns (None, None) if every attempt failed."""
    if verbose:
        print(f"[dream_step] #{number}: prompt sent to the model:\n{prompt}\n")
    attempt_prompt = prompt
    for attempt in range(1, max_validation_retries + 1):
        try:
            content, history = _creative_completion(attempt_prompt)
        except RuntimeError as e:
            print(f"[dream_step] >>> #{number}: {e}")
            return None, None
        if verbose:
            print(f"[dream_step] #{number}: attempt {attempt} raw model response:\n"
                  f"{json.dumps(content, indent=2)}\n")
        for field, locked_value in (extra_locked_fields or {}).items():
            if isinstance(locked_value, dict) and isinstance(content.get(field), dict):
                content[field] = {**content[field], **locked_value}
            else:
                content[field] = locked_value
        try:
            merged = _merge_and_validate_spec(number, content)
            return merged, content
        except SystemExit as e:
            if attempt == max_validation_retries:
                print(f"[dream_step] >>> #{number}: model's answer failed validation "
                      f"even after {max_validation_retries} attempts:\n{e}")
                return None, None
            print(f"[dream_step] #{number}: attempt {attempt} failed validation, "
                  f"retrying with the error fed back...", flush=True)
            attempt_prompt = (f"{prompt}\n\nYour previous answer failed validation "
                               f"with this exact error -- fix it and answer again, "
                               f"full JSON object, same schema:\n{e}")


def _generate_and_write_keyframes(number, prompt, max_validation_retries=3,
                                   extra_locked_fields=None, verbose=False):
    """The real 'Save content' path for keyframes -- generates via
    _generate_keyframes_content, then writes to disk. Returns True on a
    real write, False if every attempt failed validation."""
    merged, update_fields = _generate_keyframes_content(
        number, prompt, max_validation_retries, extra_locked_fields, verbose)
    if merged is None:
        return False
    spec_path = DATA_DIR / f"spec_{number:03d}.json"
    spec_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[dream_step] #{number}: keyframe prompts written. Run Generate/"
          f"Rework video on this number to actually produce the image(s).")
    return True


def _interactive_keyframes(s):
    numbers = _ask_numbers("Which number(s) to (re)generate keyframe prompts for?",
                            s["image_workflow_specs"])
    if not numbers:
        return
    image_count = ask("1 image (i2v) or 3 images (fml)?", default="1")
    image_count = 1 if image_count.strip() == "1" else 3
    for number in numbers:
        payload = build_keyframes_request_payload(number, image_count)
        if payload is None:
            print(f"[dream_step] >>> #{number}: no spec exists yet -- skipping.")
            continue
        prompt = _render_creative_prompt(payload)
        _generate_and_write_keyframes(number, prompt)


def _interactive_generate_or_rework(s, is_rework):
    candidates = s["rendered"] if is_rework else s["not_rendered"]
    verb = "re-render" if is_rework else "render"
    numbers = _ask_numbers(f"Which number(s) to {verb}?", candidates)
    if not numbers:
        return
    type_choice = ask("Graph type -- 'keep' (use the spec's own), t2v, i2v, or fml?",
                       default="keep")
    type_arg = None if type_choice.strip().lower() in ("keep", "default", "") else type_choice.strip().lower()
    if is_rework:
        with_vram_guard(do_rework, numbers, randomize_seeds=False, type_arg=type_arg)
    else:
        with_vram_guard(do_generate, numbers, type_arg)


def _interactive_upload(s):
    numbers = _ask_numbers("Which number(s) to upload?", s["rendered_not_uploaded"])
    if not numbers:
        return
    do_upload(numbers, force=False)


def _interactive_new_project(name):
    print(f"[dream_step] Creating project '{name}' -- a few required details:")
    channel_handle = ask("YouTube channel handle")
    episode_label = ask("Episode label (e.g. 'Tale', 'Dream')")
    schedule_anchor_date = ask("First scheduled upload date (YYYY-MM-DD)")
    schedule_days = ask("Days of week it publishes on (comma-separated)")
    args = argparse.Namespace(
        channel_handle=channel_handle, episode_label=episode_label,
        category_id="24", privacy_status="private", made_for_kids="false",
        default_language="en", contains_synthetic_media="false",
        description_footer="", default_tags="", schedule_anchor_number=1,
        schedule_anchor_date=schedule_anchor_date, schedule_days=schedule_days,
        timezone="Europe/Zurich", time_of_day="00:00:00",
    )
    do_new_project(name, args)
    resolve_project_globals(name)

    count_str = ask("How many concept ideas to generate now? (blank to skip)")
    if count_str.strip().isdigit():
        count = int(count_str.strip())
        payload = build_concepts_request_payload(name, count, web_search_available=True)
        # Concepts are title/premise/animal/role/line only -- no
        # positive_prompt involved, so format_rules.md's mechanical rules
        # have nothing to compose here.
        prompt = _render_creative_prompt(payload, include_format_rules=False)
        print(f"[dream_step] researching {count} concepts (this dispatches a real "
              f"web-search-capable request via the local Ollama model, may take a "
              f"while)...", flush=True)
        try:
            response, history = tool_completion(prompt)
            commit_concepts_response(name, count, response)
        except (RuntimeError, SystemExit) as e:
            print(f"[dream_step] >>> concept generation failed: {e}\nYou can retry "
                  f"this from the main menu later.")

    run_interactive(name)


def _render_creative_prompt(payload, include_format_rules=True):
    """Turn a request payload (build_spec_request_payload/
    build_keyframes_request_payload/build_concepts_request_payload's
    shape) into a single completion prompt -- no file, no separate
    process, the model just gets the full context in one string and
    returns JSON directly.

    format_rules (the shared, pipeline-wide mechanical prompt-format
    rules) and creative_guidance (this project's own CREATIVE.md, its
    STYLE) are pulled out of the JSON context blob and given their own
    explicit, clearly labeled sections up front instead -- a small model
    is more likely to actually follow a short standalone instruction
    block than a large doc string buried as one more value inside a big
    JSON object. include_format_rules=False for calls that never touch
    positive_prompt (e.g. concepts, which only ask for title/premise) --
    there's nothing in format_rules.md relevant to compose, and every
    token spent on it is one the model isn't spending on the actual task.
    A human's own typed field content always wins verbatim regardless --
    see write_row_spec's locked_fields / positive_prompt_is_human."""
    shape = "array" if isinstance(payload.get("schema_hint"), list) else "object"
    instructions = payload.pop("instructions")
    creative_style = payload.pop("creative_guidance", None)
    sections = []
    if include_format_rules:
        rules = format_rules()
        if rules:
            sections.append(
                "=== FORMAT RULES -- mechanical, apply to every project, do not "
                f"deviate unless a human typed the field content directly ===\n{rules}")
    if creative_style:
        sections.append(
            f"=== THIS PROJECT'S CREATIVE STYLE (its own CREATIVE.md) ===\n{creative_style}")
    sections.append(f"=== TASK ===\n{instructions}")
    sections.append(f"Context (JSON):\n{json.dumps(payload, indent=2)}")
    sections.append(
        f"Reply with ONLY the JSON {shape} described above, matching schema_hint "
        f"exactly. No prose, no markdown fences, just the JSON {shape}."
    )
    return "\n\n".join(sections)


def main():
    global PROJECT_DIR, DATA_DIR, DREAMS_ROOT, INDEX_PATH, HISTORY_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None,
                         help="Sibling project folder under video-projects/ to operate on, e.g. 'dreams'. "
                              "Not required when using --new-project.")
    parser.add_argument("--new-project", type=str, default=None,
                         help="Create a new sibling project folder + its upload_template.json. "
                              "Requires --channel-handle, --schedule-anchor-date, --schedule-days, "
                              "--episode-label (gathered from the user, never guessed).")
    parser.add_argument("--channel-handle", type=str, default=None)
    parser.add_argument("--episode-label", type=str, default=None,
                         help="What to call each entry, e.g. 'Dream' or 'Tale' -- used for output "
                              "folder/file naming as '<label> #N <title>'.")
    parser.add_argument("--category-id", type=str, default="24")
    parser.add_argument("--privacy-status", type=str, default="private")
    parser.add_argument("--made-for-kids", type=str, default="false")
    parser.add_argument("--default-language", type=str, default="en")
    parser.add_argument("--contains-synthetic-media", type=str, default="false")
    parser.add_argument("--description-footer", type=str, default="")
    parser.add_argument("--default-tags", type=str, default="")
    parser.add_argument("--schedule-anchor-number", type=int, default=1)
    parser.add_argument("--schedule-anchor-date", type=str, default=None)
    parser.add_argument("--schedule-days", type=str, default=None)
    parser.add_argument("--timezone", type=str, default="Europe/Zurich")
    parser.add_argument("--time-of-day", type=str, default="00:00:00")
    parser.add_argument("--status", action="store_true",
                         help="Inspect real project state and print ONLY the menu options "
                              "that are actually valid right now, each paired with its exact "
                              "command. The mandatory first call of every session -- relay "
                              "its output to the user verbatim, never decide the next step "
                              "yourself. Works with or without --project (without, lists "
                              "existing projects).")
    parser.add_argument("--web", action="store_true",
                         help="Start the local web UI (127.0.0.1 only) and open it in the "
                              "default browser -- the primary way a human runs this pipeline. "
                              "Same underlying functions as --interactive/direct flags, just "
                              "with a real page (project browser, status, forms, a results "
                              "panel) instead of a terminal.")
    parser.add_argument("--port", type=int, default=8420,
                         help="With --web: which localhost port to serve on.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                         help="With --web: interface to bind to. Leave as 127.0.0.1 "
                              "(localhost-only, no auth) for a normal install; a Docker "
                              "container passes 0.0.0.0 here since Docker's own network "
                              "isolation is the boundary in that case, not this bind.")
    parser.add_argument("--check-deps", action="store_true",
                         help="Report whether Ollama/ComfyUI are reachable at their "
                              "configured URLs and whether required model files are "
                              "present. Read-only, doesn't require --project -- useful "
                              "right after moving this pipeline to a new machine, before "
                              "running anything real.")
    parser.add_argument("--review-images", type=int, default=None,
                         help="Send this number's reference image(s) to a local vision "
                              "model (qwen3-vl:8b) for a quality check -- the local "
                              "coding agent driving this pipeline is text-only and "
                              "cannot look at its own generated images without this. "
                              "Advisory only, prints a report; doesn't block anything.")
    parser.add_argument("--write-spec", type=str, default=None,
                         help="Number to write/overwrite spec_{N:03d}.json for, with "
                              "--spec-json/--spec-json-stdin supplying the content directly "
                              "(single number only). For AI-composed content, use the manage "
                              "table's Run updates instead -- this flag is for direct/scripted "
                              "content only.")
    parser.add_argument("--spec-json", type=str, default=None,
                         help="JSON object with the spec's fields, as a single-quoted shell "
                              "argument, for a single --write-spec number. Dialogue text "
                              "with apostrophes/quotes makes this argument genuinely fragile "
                              "to shell-escape correctly -- PREFER --spec-json-stdin instead.")
    parser.add_argument("--spec-json-stdin", action="store_true",
                         help="Read the --spec-json content from stdin instead of a command-line "
                              "argument -- use a heredoc: python dream_step.py --project P "
                              "--write-spec N --spec-json-stdin <<'EOF'\n{...}\nEOF. This is the "
                              "PREFERRED way to pass spec JSON directly: a heredoc handles "
                              "embedded quotes/apostrophes in dialogue natively, no shell-"
                              "escaping needed at all.")
    parser.add_argument("--allow-custom-beats", action="store_true",
                         help="With --write-spec: skip the required four-beat positive_prompt "
                              "bracket-marker check. Use ONLY when a specific rework has a "
                              "documented reason the default beat structure isn't working for "
                              "that Tale -- never as a default.")
    parser.add_argument("--generate", type=str, default=None,
                         help="Number(s) to render for the first time: 'x', 'x-y', comma-mix, "
                              "or 'all' (all specced-but-unrendered numbers). Renders EXACTLY "
                              "these numbers, nothing implicit/continued. Replaces the old "
                              "--range-end continuation mode.")
    parser.add_argument("--type", type=str, default=None,
                         help="With --generate/--rework: 't2v'/'i2v'/'fml' to override the "
                              "spec's own workflow for just this call, or omit/'keep'/'default' "
                              "to use whatever the spec's 'workflow' field already says (the "
                              "default -- decided when the spec itself was written). If the "
                              "target type's fields aren't set yet, compose them first via the "
                              "manage table's Run updates.")
    parser.add_argument("--rework", type=str, default=None,
                         help="Number(s) to re-render from their CURRENT spec content -- 'x', "
                              "'x-y', comma-mix, or 'all' (all rendered numbers). Does NOT "
                              "re-elaborate content -- fix content via --write-spec first. "
                              "--type overrides the graph for just this call (see --type).")
    parser.add_argument("--upload", type=str, default=None,
                         help="Numbers to upload to YouTube: 'x', 'x-y', comma-mix, or 'all' "
                              "(all rendered-but-not-yet-uploaded numbers).")
    parser.add_argument("--check", type=str, default=None,
                         help="Numbers to verify against YouTube (no upload) -- same number-"
                              "list syntax as --upload. Confirms the live video still "
                              "matches what the spec/template currently intend.")
    parser.add_argument("--update-metadata", type=str, default=None,
                         help="Numbers to push fresh title/description/tags/status to their "
                              "ALREADY-uploaded video (videos.update, no file re-upload, no "
                              "duplicate) -- same number-list syntax. Edit the relevant "
                              "field(s) via --write-spec first, then run this.")
    parser.add_argument("--force", action="store_true",
                         help="With --upload: re-upload even if already marked published.")
    parser.add_argument("--randomize-seeds", action="store_true",
                         help="With --rework: use fresh random seeds instead of the graph's "
                              "saved values, for re-rolling a render whose content didn't "
                              "change but the result was a bad roll. Combine with --force "
                              "when the content is genuinely unchanged (the normal case for "
                              "a re-roll request).")
    args = parser.parse_args()

    if args.check_deps:
        results = check_dependencies()
        print("[dream_step] dependency check:")
        for r in results:
            status = "OK  " if r["found"] else "MISSING"
            where = f"({r['path']})" if r["found"] else ""
            plat = f" [{r['platform_note']}]" if r["platform_note"] else ""
            print(f"  {status} {r['name']:<12} {where} -- {r['note']}{plat}")
        missing = [r["name"] for r in results if not r["found"]]
        if missing:
            print(f"\n[dream_step] missing: {', '.join(missing)}. Install these (or make sure "
                  f"they're on PATH) before relying on the features noted above.")
        return

    if args.web:
        import web_ui
        results = check_dependencies()
        missing = [r["name"] for r in results if not r["found"] and not r["platform_note"]]
        if missing:
            print(f"[dream_step] NOTE: {', '.join(missing)} not found on PATH -- some features "
                  f"will fail until installed. Run --check-deps for details.", flush=True)
        web_ui.serve(port=args.port, host=args.host, initial_project=args.project)
        return

    if args.new_project:
        do_new_project(args.new_project, args)
        return

    if args.status and not args.project:
        do_status(None)
        return

    action_given = any(v not in (None, False) for v in (
        args.status, args.write_spec, args.generate, args.rework,
        args.upload, args.check, args.update_metadata, args.review_images))
    if not action_given:
        # No action flag at all -- the primary, recommended way to run this
        # pipeline (2026-08-07): a bare `python dream_step.py` (or with just
        # --project to skip the first question) drives a deterministic
        # input()-based menu itself. No flags to know, none to construct --
        # this is deliberate: an agent (or human) relaying answers into this
        # loop only ever needs to type plain text, never build a CLI
        # invocation, which is exactly the thing that let an agent fabricate
        # a number/answer on 2026-08-07 (see the plan's Phase 2 context).
        run_interactive(args.project)
        return

    if not args.project:
        root = projects_root()
        existing = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and p.name != "_pipeline" and (p / "_data").is_dir())
        raise SystemExit(
            "[dream_step] --project is required (not using --new-project).\n"
            f"EXPECTED: --project must be one of the real sibling project folders "
            f"under {root}, never assumed/omitted.\n"
            f"TO FIX: add --project <name>. Existing projects found: {existing}")

    resolve_project_globals(args.project)

    if args.status:
        do_status(args.project)
        return

    if args.write_spec is not None:
        if not (args.spec_json_stdin or args.spec_json):
            raise SystemExit(
                "[dream_step] --write-spec needs --spec-json or --spec-json-stdin.\n"
                "EXPECTED: direct spec content for scripted/manual writes -- for "
                "AI-composed content, use the manage table's Run updates instead.\n"
                "TO FIX: add --spec-json-stdin with a heredoc, or --spec-json '<json>'.")
        numbers = parse_number_spec(args.write_spec)
        if numbers is ALL_NUMBERS or len(numbers) != 1:
            raise SystemExit(
                "[dream_step] --spec-json/--spec-json-stdin only support a SINGLE "
                "number.\nEXPECTED: one number with direct content.\n"
                "TO FIX: run once per number.")
        spec_json_str = sys.stdin.read() if args.spec_json_stdin else args.spec_json
        do_write_spec(numbers[0], spec_json_str, allow_custom_beats=args.allow_custom_beats)
        return

    if args.upload:
        # No GPU/VRAM involvement in an upload -- skip the reload guard
        # entirely, it exists only to protect renders from local-model
        # VRAM contention.
        index = load_json(INDEX_PATH, [])
        rendered = {e["number"] for e in index if isinstance(e, dict) and e.get("number") is not None}
        not_uploaded = {n for n in rendered
                        if not next((e.get("published") for e in index
                                     if isinstance(e, dict) and e.get("number") == n), False)}
        numbers = resolve_all(parse_number_spec(args.upload), not_uploaded, "all rendered-but-not-uploaded")
        do_upload(numbers, args.force)
        return

    if args.check:
        numbers = parse_number_spec(args.check)
        if numbers is ALL_NUMBERS:
            index = load_json(INDEX_PATH, [])
            numbers = resolve_all(numbers, {e["number"] for e in index if isinstance(e, dict)}, "all rendered")
        do_check(numbers)
        return

    if args.update_metadata:
        numbers = parse_number_spec(args.update_metadata)
        if numbers is ALL_NUMBERS:
            index = load_json(INDEX_PATH, [])
            numbers = resolve_all(numbers, {e["number"] for e in index if isinstance(e, dict)}, "all rendered")
        do_update_metadata(numbers)
        return

    if args.review_images is not None:
        do_review_images(args.review_images)
        return

    type_arg = args.type
    if type_arg in (None, "keep", "default"):
        type_arg = None

    if args.rework:
        index = load_json(INDEX_PATH, [])
        rendered = {e["number"] for e in index if isinstance(e, dict) and e.get("number") is not None}
        numbers = resolve_all(parse_number_spec(args.rework), rendered, "all rendered")
        with_vram_guard(do_rework, numbers, randomize_seeds=args.randomize_seeds, type_arg=type_arg)
    elif args.generate:
        all_specced = sorted(int(p.stem.split("_")[1]) for p in DATA_DIR.glob("spec_*.json"))
        index = load_json(INDEX_PATH, [])
        rendered = {e["number"] for e in index if isinstance(e, dict) and e.get("number") is not None}
        not_rendered = [n for n in all_specced if n not in rendered]
        numbers = resolve_all(parse_number_spec(args.generate), not_rendered, "all specced-but-unrendered")
        with_vram_guard(do_generate, numbers, type_arg)
    else:
        raise SystemExit(
            "[dream_step] no action given.\nEXPECTED: one of --status, --write-spec, "
            "--generate, --rework, --upload, --check, --update-metadata, "
            "--review-images, --new-project, --web.\nTO FIX: run --status first if "
            "unsure what's valid right now.")


if __name__ == "__main__":
    main()
