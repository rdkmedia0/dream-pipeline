"""
Render one Dream video through the locally running ComfyUI LTX-2.3 T2V fp8
"default" workflow, validate it, and file it into the Dreams folder
following the Dream #47 convention.

Usage:
    python generate_dream.py --spec spec.json

spec.json: {
  "number": 48,
  "title": "Some Title",
  "positive_prompt": "full script text with [NARRATION] \"...\" lines",
  "negative_prompt": "comma separated negative terms",
  "description": "two short paragraphs for the YouTube description",
  "tags": "comma,separated,tag,list",
  "workflow": "fp8_t2v"
}

"workflow" is optional -- defaults to "fp8_t2v" (see DEFAULT_WORKFLOW
below), the fp8-scaled text-to-video graph. The other supported values are
"i2v" (image-to-video, same GGUF checkpoint stack, rectified from a fp8
ComfyUI template): when set, the spec must also have "image_path" (a
reference start-frame image, path relative to the project folder).
positive_prompt/negative_prompt are used for every workflow, i2v included
-- there is no separate i2v-only prompt field. See dream_step.py's
do_rework / find_reference_image for how a spec gets switched into i2v
mode: the user drops an image into an already-rendered Dream's folder and
asks for a rework.

The graph is submitted EXACTLY as authored; only the two prompt text fields
are replaced. Seeds default to the graph's saved values; pass --randomize-seeds
for variation or --seeds A,B to reproduce a specific render.

Prints one JSON line to stdout: {"ok": bool, "path": str, "duration": float, "error": str|null}
"""
import argparse
import base64
import copy
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

import av

import dream_step
import gemini_image

# Vision-model review text routinely contains characters (em dashes, curly
# quotes, arrows) that Windows console's default cp1252 codepage can't
# encode -- confirmed crashing a real render mid-run on a lone "->" arrow
# in a PASS verdict's own text. Force UTF-8 stdout so print() never dies
# on vision-model output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Both vision review functions below delegate to dream_step._vision_query
# (2026-08-12), which picks the actual backend (Ollama/Gemini/Claude API)
# from config.json's vision_backend -- the qwen3-vl thinking-model context-
# sizing workaround (VISION_OPTIONS) now lives there, next to the Ollama
# call it applies to, not duplicated here.
MAX_IMAGE_RETRIES = 2  # extra attempts beyond the first, so 3 total tries max per image


def review_image_against_description(image_path, intended_description):
    """Ask the local vision model whether a generated still image actually
    matches what it was supposed to show. Returns (passed, response_text)
    -- passed is best-effort (the PASS/FAIL label itself is noisy on this
    small local model, confirmed by direct testing: identical image
    content got different verdicts depending on prompt phrasing), but the
    descriptive text is consistently accurate, so a caller that logs the
    full response (not just the boolean) still gets the real signal even
    when the label is wrong. Returns (None, error_message) if the vision
    model can't be reached -- callers should treat that as "couldn't
    check," not as a failure."""
    prompt = (f"This image is supposed to show exactly this: {intended_description!r} "
              f"Does the image actually match that description? Describe exactly what "
              f"you see, note any mismatch (wrong/missing body parts, wrong action, "
              f"wrong side of any barrier/divide, deformed or hybrid anatomy, wrong art "
              f"style), and end your reply with the single word PASS or FAIL.")
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    try:
        text = dream_step._vision_query(prompt, [b64])
    except Exception as e:
        return None, f"vision review unreachable ({e})"
    # Last non-empty line is where the model was told to put its verdict.
    last_line = next((l for l in reversed(text.strip().splitlines()) if l.strip()), "")
    passed = "FAIL" not in last_line.upper()
    return passed, text


def review_keyframe_pair(reference_path, reference_role, new_path, new_role, delta_description, story_context=None):
    """Compare a newly generated keyframe against the reference (first)
    frame it's conditioned on, structured as a pairwise challenge rather
    than a vague "are these consistent" question.

    Calibrated by direct A/B testing against 3 known-bad real keyframes
    (confirmed by eye to have a drifted fence gap, no clear side-of-
    barrier transition, and inconsistent background): a vague "are
    these consistent" question scored 0/6 (0%) FAIL on images with
    confirmed real defects -- it always said PASS. A structured,
    itemized yes/no checklist per named element scored 6/6 (100%)
    across repeated trials on the same images. This prompt uses that
    calibrated checklist structure -- do not simplify it back to a
    holistic "are these consistent" question, that specific phrasing
    is confirmed unreliable on this model.

    Also confirmed: this model's /api/generate can return an EMPTY
    response for 3 images + a long prompt in one call -- pairwise (2
    images at a time) avoids that failure mode entirely.

    Also confirmed (by re-running this checklist against 3 already-
    approved, published Tales as a false-positive check): a terse
    delta_description alone ("maintain everything, but the goat
    argues more") makes the model flag STORY-INTENDED changes (a
    fence breaking, a duck's secret being revealed mid-scene) as
    defects, since nothing told it those changes were expected.
    story_context -- the full positive_prompt/story beat for
    this Dream -- gives the model enough to tell "intended story
    progression" apart from "unwanted drift," so pass it whenever
    it's available instead of the delta line alone.

    Also confirmed: even WITH story_context, question 1's original
    wording ("is every fixed object identical") auto-failed a Tale
    where the story's own action destroys a fixed object (a goat
    shattering a fence) -- the object itself is the thing that's
    SUPPOSED to change, but the question didn't carve out that
    exception, so it always answered NO regardless of context. Fixed
    by explicitly excluding "objects the delta/story names as
    changing" from question 1's identical-object check. That same
    re-test also caught genuine unrelated drift the model was right
    to flag (goat's fur color shifting brown->gray, a house/flower
    beds appearing that neither the delta nor story mentioned) --
    so the fix is narrowing question 1's scope, not loosening the
    checklist overall.

    Also confirmed (real production render, Tale #81, 2026-08-06): the
    original 4 questions only check "did anything UNWANTED change,"
    never "did the WANTED change actually happen." A weasel that was
    supposed to end up on the opposite side of a fence instead stayed
    on the same side in all 3 keyframes (just posed differently at the
    gap each time) -- every question above still answered cleanly
    PASS, because nothing else drifted and no unwanted object showed
    up. Added question 5 to explicitly verify the delta's own
    positional/spatial change actually occurred, not just that nothing
    extra broke."""
    context_line = (
        f"For context, here is the full story action across all three keyframes: "
        f"{story_context!r} -- treat anything explicitly described there as an "
        f"EXPECTED change, not a defect.\n" if story_context else "")
    prompt = (
        f"The FIRST image attached is keyframe '{reference_role}' of an fml2v video "
        f"generation. The SECOND image attached is keyframe '{new_role}' of the SAME "
        f"video, which was supposed to: {delta_description!r} -- meaning everything "
        f"else (the setting, any fixed objects like a fence or gap, the animal's "
        f"identity and appearance) should be UNCHANGED from the first image, ONLY "
        f"that specific thing (and anything the story context below also names) "
        f"should differ.\n"
        f"{context_line}"
        f"Answer each question with YES or NO, then a final verdict:\n"
        f"1. EXCLUDING any object that the delta or story context above explicitly "
        f"says changes (e.g. a fence that's stated to break/disintegrate), is every "
        f"OTHER fixed object (background structures, furniture, unrelated scenery) "
        f"identical in position, size, and shape between the two images?\n"
        f"2. Is the background/setting (plants, terrain, other scenery not named as "
        f"changing) identical between the two images?\n"
        f"3. Is it clearly the exact same individual animal, with the same fur/"
        f"coloring/markings, in both images?\n"
        f"4. Beyond what the delta/story explicitly describes, did anything else "
        f"shift (new objects appearing, unrelated background changes)?\n"
        f"5. Does the SECOND image actually show the specific change described in "
        f"{delta_description!r}? (If the delta describes a change in POSITION or "
        f"SIDE of something, e.g. moving to the opposite side of a barrier -- check "
        f"this literally and strictly: if the subject still looks like it's in the "
        f"same place/side as the first image, or the pose changed but the location "
        f"didn't, answer NO here even if everything else looks fine.)\n"
        f"If you answered NO to question 1, 2, 3, or 5, or YES to question 4, the "
        f"final verdict is FAIL. Otherwise the verdict is PASS. List what you "
        f"observed for each question, then end your answer with the single word "
        f"PASS or FAIL.")
    b64_ref = base64.b64encode(reference_path.read_bytes()).decode("utf-8")
    b64_new = base64.b64encode(new_path.read_bytes()).decode("utf-8")
    try:
        text = dream_step._vision_query(prompt, [b64_ref, b64_new])
    except Exception as e:
        return None, f"vision review unreachable ({e})"
    if not text.strip():
        return None, "vision review returned an empty response"
    last_line = next((l for l in reversed(text.strip().splitlines()) if l.strip()), "")
    passed = "FAIL" not in last_line.upper()
    return passed, text


PIPELINE_DIR = Path(__file__).resolve().parent
# DREAMS_DIR/INDEX_PATH are project-specific and NOT known until --spec is
# parsed in main() -- the project folder is the GRANDPARENT of the spec file
# (spec lives in <project>/_data/spec_NNN.json; index.json lives alongside it
# in _data/; rendered output folders go directly in <project>/, not in
# _data/, so the project folder isn't a mix of video output and JSON). This
# is what lets _pipeline stay shared/content-agnostic across multiple
# concurrent projects without any of them colliding.
DREAMS_DIR = None
INDEX_PATH = None

# The only supported workflow -- the user's own proven, tested graph.
WORKFLOWS = {
    # A "gguf12gb" entry (Q4_K_M GGUF unet) used to live here -- removed,
    # its workflow_api_gguf12gb.json no longer exists on disk (deleted at
    # some point after fp8_t2v replaced it as the default below, for a
    # severe unexplained per-step slowdown -- renders that normally took
    # ~3 minutes started taking 15-155 minutes). The dict entry survived
    # the file's deletion and was still a valid `--workflow gguf12gb` CLI
    # choice (via choices=sorted(WORKFLOWS)) that would crash on load --
    # confirmed dead, removed rather than left as a landmine. Comments
    # elsewhere in this file still reference "gguf12gb" by name as
    # historical context for other workflows' hardcoded values (they
    # matched its node layout); those references are accurate history,
    # not a sign the workflow itself still exists.
    #
    # Image-to-video: same GGUF checkpoint/CLIP/VAE stack as gguf12gb (the
    # user's own graph, rectified from a fp8-based ComfyUI template -- see
    # session notes), output size/length hardcoded to match gguf12gb exactly
    # (512x896, 588 frames @24fps) so a Dream looks the same regardless of
    # which workflow rendered it. Used ONLY when a spec has an "image_path"
    # (see do_rework's image-detection in dream_step.py) -- renders from the
    # spec's normal positive_prompt/negative_prompt, same as every other
    # workflow (a separate i2v_positive_prompt/i2v_negative_prompt used to
    # exist here, removed 2026-08-08: in practice it was byte-identical to
    # positive_prompt on 29 of 37 real specs, and an incomplete/truncated
    # copy of it on the rest -- never a deliberately different, better-tuned
    # version, just a second place the same content could silently go
    # stale. See Tale #83's postmortem in session notes.).
    "i2v": {
        "path": PIPELINE_DIR / "workflow_api_i2v.json",
        "positive": "320:319", "negative": "320:313", "seeds": ["320:276", "320:277"],
        # 320:319 is a PrimitiveStringMultiline ("value"), NOT a
        # CLIPTextEncode ("text") like every other prompt node in this
        # pipeline -- confirmed missing this caused every i2v render so far
        # to silently ignore the real prompt (see build_prompt).
        "positive_field": "value",
        "image_node": "269", "image_field": "image",
        # Node "320:323" is a ComfyMathExpression hardcoded to the constant
        # "588" (see the size/length rework in session notes) -- the ONE
        # isolated frame-count control point in this graph, same role as
        # gguf12gb's node "112" but a different field type (see
        # build_prompt's length_field handling). Not currently used by any
        # render path (multi-keyframe i2v was tried and abandoned -- see
        # session notes -- stitching separate generations together produced
        # a different voice/style per clip that never tied together), kept
        # only in case a single-image i2v render ever needs a non-default
        # length.
        "length_node": "320:323", "length_field": "expression", "fps": 24,
        "width_node": "320:312", "height_node": "320:299",
    },
    # fp8 text-to-video: same LTX-2.3 model family as gguf12gb, but using the
    # fp8-scaled unet (no GGUF dequantization overhead) instead of the
    # Q4_K_M GGUF checkpoint. Added to sidestep a severe, unexplained
    # per-step slowdown observed in gguf12gb this session (renders that
    # normally took ~3 minutes were taking 15-155 minutes, with nvidia-smi
    # showing VRAM maxed but low power draw -- a memory-bound thrashing
    # signature) that a full reboot and disabling newly-installed custom
    # node packs did not resolve. Only the single distilled LoRA at
    # strength 0.5 is active (matching gguf12gb's convention); output
    # size/length hardcoded to match gguf12gb exactly (512x896, 588 frames
    # @24fps) so a Dream looks the same regardless of which workflow
    # rendered it. Confirmed working via a direct ComfyUI API test at full
    # resolution/length before being wired in here.
    "fp8_t2v": {
        "path": PIPELINE_DIR / "workflow_api_fp8_t2v.json",
        "positive": "121", "negative": "110", "seeds": ["114", "115"],
        "length_node": "112", "fps": 24,
        # Both width and height live on the SAME node (EmptyImage "111"),
        "width_node": "111", "width_field": "width",
        "height_node": "111", "height_field": "height",
    },
    # First-Middle-Last multi-keyframe i2v: three reference stills (first
    # frame, middle/action frame, last frame) instead of one -- lets the
    # model see both the "before" and "after" state of a hard-to-animate
    # action (e.g. a chameleon's tongue strike, a fence breaking) instead
    # of inferring it from text alone. Same fp8-distilled checkpoint family
    # as fp8_t2v (used directly as the base checkpoint here, no separate
    # LoRA needed -- confirmed the Power Lora Loader node in this graph is
    # intentionally empty). Only reached via dream_step.py's do_rework: the
    # user drops exactly three images named 1/2/3 into an already-rendered
    # Dream's folder (see find_reference_images) and the spec is set up
    # with "workflow": "fml2v" plus the fml2v_* fields below. The graph's
    # own prompt-enhancer switch is baked OFF (node 2082) so the spec's
    # exact wording is used verbatim, not silently rewritten by an LLM.
    "fml2v": {
        "path": PIPELINE_DIR / "workflow_api_fml2v.json",
        "positive": "2103", "positive_field": "value",
        "negative": "11", "seeds": ["14", "15"],
        # Three separate LoadImage nodes, one per keyframe role -- see
        # build_prompt's image_filenames handling (plural, unlike i2v's
        # single image_node/image_field).
        "image_nodes": {"first": "45", "middle": "47", "last": "2172"},
        "image_field": "image",
        # Per-keyframe guide strength (PrimitiveFloat "value" nodes feeding
        # LTXVAddGuideMulti's strength_1/2/3). "first"/"last" each drive both
        # sampling stages; "middle" only feeds the second (refinement) stage
        # -- that's a property of the graph itself, not this mapping. Spec's
        # optional "fml2v_guide_strengths" (partial dict, e.g. {"middle": 0.7})
        # overrides the graph's own saved defaults; any role left unset keeps
        # the graph's value untouched, same convention as seeds.
        "strength_nodes": {"first": "2110", "middle": "2278", "last": "2108"},
        "strength_field": "value",
        # This graph's length control (node 2078) is in SECONDS directly
        # (unlike i2v/fp8_t2v's raw frame count) -- length_unit="seconds"
        # tells build_prompt to write the target duration straight through
        # rather than computing a frame count via the granularity formula.
        "length_node": "2078", "length_unit": "seconds",
        "width_node": "2080", "height_node": "2079",
    },
    # Flux2-based still-image generator used ONLY by generate_keyframes to
    # auto-create the fml2v workflow's three keyframe images from a script,
    # when the user hasn't manually dropped in 1/2/3 images themselves.
    # Structurally an image-reference/I2I graph (always needs a loaded
    # source image for its ReferenceLatent conditioning) rather than a pure
    # text-only T2I -- for the first frame, generate_keyframes feeds it a
    # neutral blank placeholder image instead of a real prior frame, which
    # lets the text prompt dominate. No negative-prompt node (Flux-family
    # models don't use classifier-free negative guidance the way LTX does).
    # Output size hardcoded to 512x896 to match the video workflows exactly
    # (originally derived from whatever reference image was loaded, scaled
    # to ~1 megapixel -- decoupled from that so it's always our standard).
    "t2i_i2i": {
        "path": PIPELINE_DIR / "workflow_api_t2i_flux2.json",
        "positive": "68:6", "negative": None,
        "seeds": ["68:25"],
        "image_node": "118", "image_field": "image",
    },
}
# fp8_t2v replaced gguf12gb as the default after the GGUF slowdown (see
# WORKFLOWS above) -- workflow_api_fp8_t2v.json, do not "improve" it.
DEFAULT_WORKFLOW = "fp8_t2v"


def load_workflow_template(workflow_name):
    """Load a workflow's JSON graph AND confirm every node ID this
    pipeline's WORKFLOWS config expects (positive/negative/image_node/
    length_node/seeds/image_nodes/strength_nodes) actually exists as a
    top-level key in it. Node IDs are fixed, hand-recorded strings (see
    WORKFLOWS above) -- if the graph is ever re-exported from ComfyUI
    (even a trivial resave in a newer version), those IDs can silently
    shift, and prompt[stale_id] would previously either KeyError deep
    inside build_prompt with no context, or -- if a stale ID happened to
    collide with an unrelated node -- write the prompt into the WRONG
    node and render successfully with silently wrong content. Failing
    here, once, with every mismatched ID named up front, replaces both
    failure modes with one clear message pointing at the actual cause."""
    if workflow_name not in WORKFLOWS:
        raise SystemExit(f"[generate_dream] unknown workflow '{workflow_name}', "
                          f"choose from {sorted(WORKFLOWS)}")
    workflow_cfg = WORKFLOWS[workflow_name]
    # A user can point Settings' "Workflow files" section at their own
    # workflow_api_*.json for this workflow's type instead of the
    # built-in default -- custom_workflows.json records which file (if
    # any) is currently "active" for each type, with wiring that was
    # already detected via workflow_introspect.py and confirmed with a
    # real test render before being saved there (see web_ui.py). Falls
    # through to the hardcoded WORKFLOWS entry above when no custom file
    # is active for this workflow's type, so nothing changes for anyone
    # who never touches that Settings section.
    workflow_type = dream_step.WORKFLOW_TO_TYPE.get(workflow_name)
    if workflow_type:
        active_filename, active_entry = dream_step.active_custom_workflow_for_type(workflow_type)
        if active_entry:
            workflow_cfg = {k: v for k, v in active_entry.items() if k not in ("type", "active")}
            workflow_cfg["path"] = PIPELINE_DIR / active_filename
    path = workflow_cfg["path"]
    if not path.exists():
        raise SystemExit(
            f"[generate_dream] workflow '{workflow_name}' is configured to load "
            f"{path}, but that file doesn't exist. If this pipeline was just moved "
            f"to a new machine, make sure every workflow_api_*.json referenced in "
            f"WORKFLOWS (generate_dream.py) was copied along with it.")
    template = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = []
    for key in ("positive", "negative", "image_node", "length_node"):
        v = workflow_cfg.get(key)
        if v:
            expected_ids.append(v)
    expected_ids.extend(workflow_cfg.get("seeds") or [])
    expected_ids.extend((workflow_cfg.get("image_nodes") or {}).values())
    expected_ids.extend((workflow_cfg.get("strength_nodes") or {}).values())
    missing = [nid for nid in expected_ids if nid not in template]
    if missing:
        raise SystemExit(
            f"[generate_dream] workflow '{workflow_name}' ({path.name}) is missing "
            f"node ID(s) {missing} that this pipeline's WORKFLOWS config expects. "
            f"This usually means the graph was re-exported/resaved from ComfyUI and "
            f"its node IDs shifted -- re-run convert_workflow_to_api.py on a fresh "
            f"export and update the matching WORKFLOWS entry's node IDs in "
            f"generate_dream.py, or restore the previously-working {path.name}.")
    return workflow_cfg, template

# NOTE: there is deliberately no auto-appended negative prompt. The spec's
# negative_prompt is submitted verbatim. Anything the render needs belongs in
# the spec where it is visible, not injected here.

CANDIDATE_PORTS = [8188, 8000, 8300]
MIN_DURATION_S = 18.0
MAX_DURATION_S = 30.0
MAX_ATTEMPTS = 2


def find_comfyui_base_url():
    """Returns a full working base URL ("http://host:port"), not just a
    bare port -- confirmed real bug (2026-08-08): every ComfyUI call in
    this file used to hardcode "http://127.0.0.1:{port}", completely
    ignoring the configured comfyui_url from config.json (which every
    OTHER part of this pipeline -- web_ui.py, vram_guard.py -- already
    honors). That silently only worked because ComfyUI happened to
    always be on this same machine; it would have made remote ComfyUI
    (comfyui_url pointing at a different host) unreachable outright,
    regardless of what Settings said. Derives the host from the
    configured comfyui_url, then probes CANDIDATE_PORTS against THAT
    host (configured port tried first) the same way this always probed
    localhost, so it stays robust to the exact port drifting without
    needing comfyui_url to be byte-perfect."""
    cfg = dream_step.load_config()
    parsed = urllib.parse.urlparse(cfg["comfyui_url"])
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    ports = ([parsed.port] if parsed.port else []) + [p for p in CANDIDATE_PORTS if p != parsed.port]
    for port in ports:
        base = f"{scheme}://{host}:{port}"
        try:
            with urllib.request.urlopen(f"{base}/system_stats", timeout=3) as r:
                if r.status == 200:
                    return base
        except Exception:
            continue
    raise RuntimeError(f"ComfyUI not reachable on {host} at any of {ports}")


def get_episode_label(data_dir):
    """Output folders are named '<label> #N <title>' -- the label defaults
    to 'Dream' but is configurable per-project via that project's
    upload_template.json ('episode_label'), since not every project is
    about dreams (e.g. a comedy project might use 'Tale')."""
    template_path = data_dir / "youtube" / "upload_template.json"
    if template_path.exists():
        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
            return template.get("episode_label", "Dream")
        except Exception:
            pass
    return "Dream"


def sanitize_filename(name):
    # Confirmed defensive gap (2026-08-18): every CURRENT call site passes a
    # string that already includes a numbered prefix (e.g.
    # f"Dream #{number} {title}"), so in practice the combined result can't
    # collapse to empty even when title itself is nothing but forbidden/dot
    # characters -- but this function has no guarantee of that shape, and
    # nothing stops a future caller from passing just a raw title. Without
    # this fallback, an all-bad-character input strips to "", and
    # DREAMS_ROOT / "" resolves to DREAMS_ROOT itself -- the same
    # empty-folder-name collapse _validate_project_folder_name (dream_step.py)
    # already guards against for project names, just missing here.
    bad = '<>:"/\\|?*'
    out = "".join(c for c in name if c not in bad)
    out = out.strip().rstrip(".")
    return out or "untitled"


# Must match dream_step.py's own _FALLBACK_RENDER_WIDTH/HEIGHT/DURATION_S
# -- these are what every current workflow_api_*.json already has baked
# in. apply_render_settings only ever writes to a node when a project's
# own Duration:/Resolution: (CREATIVE.md) differs from these, so a
# project that's never set them changes NOTHING about existing render
# behavior (confirmed requirement, 2026-08-11: "Set the default to what
# it is already").
_DEFAULT_RENDER_WIDTH = 512
_DEFAULT_RENDER_HEIGHT = 896
_DEFAULT_RENDER_DURATION_S = 24


def apply_render_settings(prompt, workflow_cfg):
    """Overrides a workflow graph's own width/height/length nodes to match
    the current PROJECT's own Duration:/Resolution: settings (see
    dream_step.project_render_settings() -- moved 2026-08-12 from a global
    config.json Settings field to per-project CREATIVE.md, since render
    size/length is a per-channel decision, not a pipeline-wide one) -- but
    ONLY when a value differs from this pipeline's long-standing default
    (_DEFAULT_RENDER_*), so a project that's never set these gets
    byte-identical behavior to before this existed. workflow_cfg's
    width_node/height_node/length_node (see WORKFLOWS above) are optional
    per workflow -- silently does nothing for a dimension a given
    workflow hasn't wired up.

    Frame-count math for length_node when length_unit isn't "seconds"
    (i2v/fp8_t2v, which take a raw frame count, not seconds) uses the
    SAME granularity formula fml2v's own graph already relies on
    internally (round to the nearest 8k+1 frames) -- the one formula in
    this codebase actually validated against this LTX build, rather than
    inventing a second one. NOTE: this does NOT reproduce the exact
    frame count (588) these graphs shipped with for their own ~24s
    default -- that's fine, since the default-duration case never
    reaches this branch at all (returns before computing anything)."""
    width, height, duration_s_int = dream_step.project_render_settings()
    duration_s = float(duration_s_int)

    width_node = workflow_cfg.get("width_node")
    if width_node and width != _DEFAULT_RENDER_WIDTH:
        prompt[width_node]["inputs"][workflow_cfg.get("width_field", "value")] = width

    height_node = workflow_cfg.get("height_node")
    if height_node and height != _DEFAULT_RENDER_HEIGHT:
        prompt[height_node]["inputs"][workflow_cfg.get("height_field", "value")] = height

    length_node = workflow_cfg.get("length_node")
    if length_node and duration_s != _DEFAULT_RENDER_DURATION_S:
        length_field = workflow_cfg.get("length_field", "value")
        if workflow_cfg.get("length_unit") == "seconds":
            prompt[length_node]["inputs"][length_field] = duration_s
        else:
            fps = workflow_cfg.get("fps", 24)
            frames = ((round((duration_s * fps - 1) / 8)) * 8) + 1
            value = str(frames) if length_field == "expression" else frames
            prompt[length_node]["inputs"][length_field] = value


def build_prompt(template, workflow_cfg, positive_prompt, negative_prompt,
                 seeds=None, randomize_seeds=False,
                 image_filename=None, image_filenames=None, guide_strengths=None):
    """Submit the workflow graph EXACTLY as authored, changing only the two
    prompt text fields (and, only for i2v, the single isolated LoadImage
    node -- see run_once). Nothing else is touched -- no appended negative
    terms, no seed randomisation by default -- because silently altering
    the graph made render results impossible to reason about.

    seeds:           explicit list of ints to use (overrides the graph's own)
    randomize_seeds: opt in to fresh random seeds for variation
    image_filename:  filename (already present in ComfyUI's input/ folder,
                      see run_once) to feed into an i2v workflow's LoadImage
                      node. Only valid when workflow_cfg has "image_node".
    image_filenames: dict of {role: filename} (e.g. {"first": "...",
                      "middle": "...", "last": "..."}) to feed an fml2v
                      workflow's THREE separate LoadImage nodes. Only valid
                      when workflow_cfg has "image_nodes" (plural).
    guide_strengths:  optional dict of {role: float} (subset of "first"/
                      "middle"/"last") overriding an fml2v workflow's
                      per-keyframe guide strength nodes. Only valid when
                      workflow_cfg has "strength_nodes". Any role left out
                      keeps the graph's own saved value.
    Default (both seeds args unset): use the seed values saved in the graph.
    """
    prompt = copy.deepcopy(template)
    # gguf12gb's positive/negative nodes are both CLIPTextEncode ("text");
    # the i2v graph's positive node is a PrimitiveStringMultiline ("value")
    # instead -- positive_field/negative_field let each workflow say which
    # input field its own prompt nodes actually take (same pattern as
    # length_field below). Confirmed bug this fixes: assuming "text" for
    # i2v's positive node silently wrote to a field that node doesn't read,
    # leaving its real "value" field stuck on the graph's placeholder text
    # for every i2v render so far -- the script was never reaching the model.
    positive_field = workflow_cfg.get("positive_field", "text")
    negative_field = workflow_cfg.get("negative_field", "text")
    prompt[workflow_cfg["positive"]]["inputs"][positive_field] = positive_prompt
    # "negative" is optional -- some graphs (e.g. Flux2 T2I/I2I keyframe
    # generation) have no negative-prompt node at all (Flux-family models
    # don't use classifier-free negative guidance the way LTX does).
    if workflow_cfg.get("negative") is not None:
        prompt[workflow_cfg["negative"]]["inputs"][negative_field] = negative_prompt
    if image_filename is not None:
        image_node = workflow_cfg.get("image_node")
        if image_node is None:
            raise RuntimeError("image_filename given but this workflow has no image_node configured")
        prompt[image_node]["inputs"][workflow_cfg["image_field"]] = image_filename
    if image_filenames is not None:
        image_nodes = workflow_cfg.get("image_nodes")
        if image_nodes is None:
            raise RuntimeError("image_filenames given but this workflow has no image_nodes configured")
        for role, filename in image_filenames.items():
            if role not in image_nodes:
                raise RuntimeError(f"image_filenames has unknown role '{role}', expected one of {list(image_nodes)}")
            prompt[image_nodes[role]]["inputs"][workflow_cfg["image_field"]] = filename
    if guide_strengths is not None:
        strength_nodes = workflow_cfg.get("strength_nodes")
        if strength_nodes is None:
            raise RuntimeError("guide_strengths given but this workflow has no strength_nodes configured")
        strength_field = workflow_cfg.get("strength_field", "value")
        for role, value in guide_strengths.items():
            if role not in strength_nodes:
                raise RuntimeError(f"guide_strengths has unknown role '{role}', expected one of {list(strength_nodes)}")
            prompt[strength_nodes[role]]["inputs"][strength_field] = float(value)
    apply_render_settings(prompt, workflow_cfg)
    used = []
    for i, nid in enumerate(workflow_cfg["seeds"]):
        if seeds and i < len(seeds):
            prompt[nid]["inputs"]["noise_seed"] = int(seeds[i])
        elif randomize_seeds:
            prompt[nid]["inputs"]["noise_seed"] = random.randint(0, 2**48)
        # else: leave the graph's own value untouched
        used.append(prompt[nid]["inputs"]["noise_seed"])
    return prompt, used


def queue_prompt(comfyui_base, prompt):
    # Fixed, not a fresh uuid per call -- ComfyUI only pushes "progress_state"
    # websocket events to the connection whose clientId matches whichever
    # prompt is currently executing (comfy_execution/progress.py,
    # execution.py's `self.server.client_id = extra_data["client_id"]`), so
    # dream_step.query_comfyui_progress() needs a client_id it can reliably
    # reconnect with at any time, not one it would have to learn from this
    # specific subprocess call. Safe as a shared constant: this pipeline
    # only ever runs one render at a time (vram_guard's whole point), so
    # there's never a second concurrent submitter to collide with.
    client_id = dream_step.COMFYUI_CLIENT_ID
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{comfyui_base}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected prompt: {e.code} {body}")
    if "prompt_id" not in resp:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {resp}")
    return resp["prompt_id"]


def wait_for_history(comfyui_base, prompt_id, timeout_s=3600, poll_s=5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with urllib.request.urlopen(f"{comfyui_base}/history/{prompt_id}", timeout=10) as r:
            hist = json.loads(r.read().decode("utf-8"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") is True:
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error: {json.dumps(status)}")
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id} after {timeout_s}s")


def _find_output_item(history_entry, extensions):
    """Scans a ComfyUI history entry's outputs for the first item whose
    filename ends in one of `extensions` -- shared by find_output_video
    (a single ".mp4") and find_output_image (the keyframe still-image
    extensions), which previously duplicated this exact scan."""
    outputs = history_entry.get("outputs", {})
    for node_id, out in outputs.items():
        for key, items in out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(extensions):
                    return item
    raise RuntimeError(f"No matching output found in history outputs: {json.dumps(outputs)[:500]}")


def find_output_video(history_entry):
    return _find_output_item(history_entry, (".mp4",))


def find_output_image(history_entry):
    """Same scan as find_output_video, but for a still-image output (used by
    the T2I/I2I keyframe-generation graphs -- see generate_keyframes)."""
    return _find_output_item(history_entry, (".png", ".jpg", ".jpeg", ".webp"))


def download_or_locate(comfyui_base, item, comfyui_output_dir=None):
    """Shared by the video and image output paths -- previously duplicated
    byte-for-byte as download_or_locate_video/download_or_locate_image."""
    subfolder = item.get("subfolder", "")
    filename = item["filename"]
    itype = item.get("type", "output")
    if comfyui_output_dir:
        local_path = Path(comfyui_output_dir) / subfolder / filename
        if local_path.exists():
            return local_path
    # Fall back to HTTP view endpoint -- this is what actually kicks in
    # when comfyui_output_dir is a path on a DIFFERENT machine than this
    # process (see find_comfyui_base_url's docstring): the local_path
    # check above just won't exist, and this download path already works
    # correctly regardless of host.
    q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": itype})
    url = f"{comfyui_base}/view?{q}"
    tmp_path = PIPELINE_DIR / f"_tmp_{filename}"
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path


def upload_image_to_comfyui(comfyui_base, src_path, dest_filename):
    """Uploads a local file's bytes to ComfyUI's own /upload/image endpoint
    (standard ComfyUI API, saves into ComfyUI's own input/ folder) instead
    of a local shutil.copy2() into comfyui_input_dir -- confirmed real gap
    (2026-08-08): a plain file copy only works when this process and
    ComfyUI share a filesystem (comfyui_input_dir is actually the same
    physical folder ComfyUI reads from, e.g. same machine or an NFS/SMB
    mount). Uploading the bytes over HTTP instead works identically
    whether ComfyUI is on this machine or a remote one, matching how
    download_or_locate already falls back to HTTP when the
    local path doesn't exist. overwrite=true so the filename we ask for
    is always the filename that lands (no silent ComfyUI-side rename on
    collision), matching the deterministic naming this pipeline already
    relies on (e.g. "i2v_dream_83.png") when reading it back into the
    graph's LoadImage node. Returns dest_filename on success."""
    boundary = f"----geoformboundary{dream_step.COMFYUI_CLIENT_ID}"
    image_bytes = Path(src_path).read_bytes()
    parts = []
    parts.append(f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="image"; filename="{dest_filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8"))
    parts.append(image_bytes)
    parts.append(b"\r\n")
    for field, value in (("type", "input"), ("overwrite", "true")):
        parts.append(f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{field}"\r\n\r\n'
                     f"{value}\r\n".encode("utf-8"))
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{comfyui_base}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected image upload: {e.code} {body_text}")
    if resp.get("name") != dest_filename:
        raise RuntimeError(f"ComfyUI saved the upload under an unexpected name: {resp}")
    return dest_filename


def try_online_first_frame(spec, dest_dir, scene_prompt, dest_path, force=False):
    """Generate the first frame directly via Gemini and use it AS the
    first keyframe -- when spec["first_frame_source"] == "online"
    (opt-in per Tale) or force=True (kf_backend's global "first frame
    via Gemini" options, all_gemini/first_gemini_rest_local, which apply
    to EVERY Tale regardless of its own toggle), AND Gemini is actually
    enabled (an API key is saved -- see gemini_image.is_enabled()).
    Either gate failing is a silent, ordinary skip straight to local
    generation, not an error.

    2026-08-12, two rounds of simplification: first removed a local
    T2I/I2I redraw pass that used to run on top of the Gemini image
    before accepting it as the first frame (confirmed live: Gemini
    already honors the requested 512x896 size from the prompt text, and
    every downstream stage resizes/composes on top of the first frame
    anyway, so the redraw was pure extra generation cost with a real
    risk of undoing exactly the species/subject accuracy online sourcing
    exists to get right). Then removed a SEPARATE cached
    "_online_reference_seed.png" file this function used to write the
    Gemini image to first, with its own independent existence-based
    reuse check -- confirmed real bug: the caller (generate_keyframes's
    role_changed()) already decides whether a fresh first frame is
    needed by comparing prompt text; a second, independent cache with no
    awareness of that decision could go stale (an old cached image
    surviving a genuinely new prompt) or look "stuck" (deleting the
    real 1.png didn't clear this file, so the next render silently
    reused the old image anyway). Generates fresh every time this
    function is actually called, writing directly to dest_path -- no
    intermediate file, nothing to fall out of sync.

    scene_prompt: the SAME still-image description already written for
    this beat (fml2v's keyframe_prompts["first"], or i2v's own
    prompt_text) -- used verbatim as the Gemini prompt (prefixed with
    an explicit size instruction), not a generic "a photo of the
    animal" description. Confirmed necessary (2026-08-09): a generic
    species-only prompt gives Gemini no framing/pose guidance, and
    parts of the subject came back cut off in real renders -- Gemini
    has no separate width/height parameter, sizing is inferred from the
    prompt text itself, so "512x896 portrait image: <the real scene
    description>" both grounds the composition in the actual story beat
    AND tells the model the target aspect ratio up front.

    Still goes through the same vision review as a locally-generated
    frame would, so a bad Gemini result doesn't silently ship -- on
    failure this returns None and the caller falls back to the ordinary
    local T2I generation loop, exactly as if online sourcing had never
    been attempted."""
    if not force and spec.get("first_frame_source") != "online":
        return None
    if not gemini_image.is_enabled():
        return None
    model = dream_step.load_config().get("gemini_model") or gemini_image.MODEL
    prompt = f"512x896 portrait image: {scene_prompt}"
    max_block_retries = 2
    attempt = 0
    while True:
        attempt += 1
        print(f"[generate_dream] -> first-frame image via gemini ({model}), attempt {attempt}...", flush=True)
        start = time.time()
        try:
            gemini_image.generate_reference_image(prompt, dest_path)
            break
        except gemini_image.GeminiContentBlocked as e:
            print(f"[generate_dream] <- first-frame image via gemini BLOCKED after "
                  f"{time.time() - start:.1f}s: {e}", flush=True)
            dest_path.unlink(missing_ok=True)
            if attempt > max_block_retries:
                print(f"[generate_dream] gemini content-blocked {attempt} times in a row -- "
                      f"falling back to local T2I generation", flush=True)
                return None
            continue
        except Exception as e:
            print(f"[generate_dream] <- first-frame image via gemini FAILED after "
                  f"{time.time() - start:.1f}s: {e} -- falling back to local T2I generation", flush=True)
            dest_path.unlink(missing_ok=True)
            return None
    print(f"[generate_dream] <- first-frame image via gemini done in {time.time() - start:.1f}s", flush=True)
    passed, review_text = review_image_against_description(dest_path, scene_prompt)
    print(f"[generate_dream] online first-frame vision review:\n"
          f"{dream_step.sanitize_review_text_for_log(review_text)}", flush=True)
    if passed is False:
        print(f"[generate_dream] online first frame failed review -- "
              f"falling back to local T2I generation", flush=True)
        dest_path.unlink(missing_ok=True)
        return None
    return dest_path


def generate_keyframes(spec, keyframe_prompts, comfyui_base, comfyui_output_dir, dest_dir):
    story_context = spec.get("positive_prompt")
    """Auto-generate the three first/middle/last still images an fml2v
    render needs, when the user hasn't dropped them in manually: T2I for
    the first frame (conditioned on a blank placeholder image, since this
    graph's reference-conditioning mechanism always expects SOME source
    image), then I2I for the middle and last frames, BOTH conditioned on
    that same first frame (not chained sequentially) -- keeps middle/last
    anchored to one stable, good result instead of compounding drift
    across a longer first->middle->last chain. Saves each result into the
    Dream's own folder as 1.png/2.png/3.png (matching the manual-drop
    convention find_reference_images looks for) and returns their paths.

    keyframe_prompts: {"first": "...", "middle": "...", "last": "..."} --
                       still-image descriptions of each beat's visual
                       state, NOT positive_prompt (that describes the
                       ANIMATION across all three, used separately for
                       the actual video render once these images exist).
                       "middle"/"last" use this graph's ReferenceLatent
                       conditioning, which is a loose/soft style of
                       reference-following, not a strict img2img denoise
                       lock -- confirmed it can drift noticeably from the
                       reference if the prompt just re-describes the whole
                       scene fresh (competes with the reference instead of
                       reinforcing it). Confirmed working phrasing pattern
                       (worked example: a chameleon's tongue-strike
                       sequence): "Maintain everything in the image, but
                       make [X happen]" / "Keep everything the same, but
                       [Y]" -- lead with the maintain/keep instruction,
                       then describe ONLY the pose/state change, never
                       re-describe the full scene from scratch.

    Only regenerates a role whose prompt actually changed since the last
    time this Dream's keyframes were generated (tracked in a small sidecar
    file, _fml2v_keyframe_prompts.json, next to the images in dest_dir) --
    a rework that only edits e.g. the "last" prompt reuses the existing
    1.png/2.png untouched instead of re-running all three T2I/I2I passes.
    If "first" itself changed, middle/last are also regenerated even if
    their own prompt text didn't change, since they're conditioned on
    first's actual pixel output, not just its prompt.
    """
    workflow_cfg, template = load_workflow_template("t2i_i2i")

    sidecar_path = dest_dir / "_fml2v_keyframe_prompts.json"
    previous_prompts = {}
    if sidecar_path.exists():
        try:
            previous_prompts = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            previous_prompts = {}

    dest_paths = {"first": dest_dir / "1.png", "middle": dest_dir / "2.png", "last": dest_dir / "3.png"}

    def role_changed(role):
        # No sidecar entry for this role (e.g. bootstrapping on a Dream
        # rendered before this sidecar existed) is NOT the same as "prompt
        # changed" -- if the image is already on disk, trust it as-is and
        # only regenerate when the sidecar actually recorded a DIFFERENT
        # prompt, or the image file itself is missing.
        if not dest_paths[role].exists():
            return True
        if role not in previous_prompts:
            return False
        return keyframe_prompts[role] != previous_prompts[role]

    first_changed = role_changed("first")

    def generate_one_attempt(role, reference_filename, dest_index):
        prompt, _ = build_prompt(
            template, workflow_cfg, keyframe_prompts[role], None,
            randomize_seeds=True, image_filename=reference_filename)
        print(f"[generate_dream] -> keyframe '{role}' via local ComfyUI T2I/I2I...", flush=True)
        start = time.time()
        prompt_id = queue_prompt(comfyui_base, prompt)
        history_entry = wait_for_history(comfyui_base, prompt_id)
        image_item = find_output_image(history_entry)
        tmp_image_path = download_or_locate(comfyui_base, image_item, comfyui_output_dir)
        dest_path = dest_dir / f"{dest_index}.png"
        try:
            shutil.copy2(tmp_image_path, dest_path)
        finally:
            # try/finally so a failed copy (disk full, permissions, etc.)
            # still cleans up the downloaded temp file instead of leaking
            # it in PIPELINE_DIR forever.
            if tmp_image_path.parent == PIPELINE_DIR:
                tmp_image_path.unlink(missing_ok=True)
        print(f"[generate_dream] <- keyframe '{role}' via local ComfyUI done in "
              f"{time.time() - start:.1f}s", flush=True)
        return dest_path

    def generate_one_attempt_gemini(role, dest_index, reference_image_path):
        """kf_middle_last_backend="gemini_middle_last" variant of
        generate_one_attempt -- a real billed Gemini image-EDIT call
        (image+text-in, image-out) conditioned on the actual first-frame
        image, replacing the local ComfyUI I2I pass for this role. Same
        "maintain everything, but X" prompt phrasing local I2I already
        relies on works here too -- it's an instruction to an image
        model either way, just a different one."""
        model = dream_step.load_config().get("gemini_model") or gemini_image.MODEL
        print(f"[generate_dream] -> keyframe '{role}' via gemini image-edit ({model})...", flush=True)
        start = time.time()
        dest_path = dest_dir / f"{dest_index}.png"
        gemini_image.edit_image(keyframe_prompts[role], [reference_image_path], dest_path)
        print(f"[generate_dream] <- keyframe '{role}' via gemini image-edit done in "
              f"{time.time() - start:.1f}s", flush=True)
        return dest_path

    def generate_one(role, reference_filename, dest_index, compare_against=None,
                      gemini_reference_image_path=None):
        """Generate one keyframe, reviewing it before accepting it --
        retries (fresh random seed each time, same prompt) up to
        MAX_IMAGE_RETRIES extra times if the review says it doesn't
        match, so a bad frame gets caught and fixed here, BEFORE the
        expensive full video render runs on top of it -- not after.

        compare_against: (reference_path, reference_role) for a pairwise
        comparison against that specific frame (used for middle/last,
        checked against first) -- confirmed more reliable than either a
        vague multi-image consistency question or a self-only check,
        since it catches things like an artificially widened fence gap
        that only shows up when directly compared to the reference frame.
        Omit for 'first' itself, which has nothing to compare against yet.

        gemini_reference_image_path: when given, generates via
        generate_one_attempt_gemini (a real billed Gemini image-edit
        call) instead of the local ComfyUI I2I pass -- see
        kf_middle_last_backend."""
        last_path, last_review = None, None
        for attempt in range(1, MAX_IMAGE_RETRIES + 2):
            path = (generate_one_attempt_gemini(role, dest_index, gemini_reference_image_path)
                    if gemini_reference_image_path is not None
                    else generate_one_attempt(role, reference_filename, dest_index))
            if compare_against:
                ref_path, ref_role = compare_against
                passed, review_text = review_keyframe_pair(
                    ref_path, ref_role, path, role, keyframe_prompts[role],
                    story_context=story_context)
            else:
                passed, review_text = review_image_against_description(path, keyframe_prompts[role])
            # sanitize_review_text_for_log strips the model's own literal
            # PASS/FAIL verdict word from the DISPLAYED text only (the
            # `passed` boolean above, used for the actual retry decision,
            # already came from the raw, unsanitized text) -- an attempt
            # that gets retried and then succeeds means the job overall is
            # fine, and a bare "FAIL" sitting in a successful job's log
            # reads as fatal when it was just this self-correcting loop
            # doing its job.
            print(f"[generate_dream] keyframe '{role}' attempt {attempt}/"
                  f"{MAX_IMAGE_RETRIES + 1} vision review:\n"
                  f"{dream_step.sanitize_review_text_for_log(review_text)}", flush=True)
            last_path, last_review = path, review_text
            if passed is not False:  # True (passed) or None (couldn't check) -- accept
                return path
            print(f"[generate_dream] '{role}' keyframe failed review, "
                  f"{'retrying with a fresh seed' if attempt <= MAX_IMAGE_RETRIES else 'out of retries'}...",
                  flush=True)
        print(f"[generate_dream] >>> '{role}' keyframe still failed review after "
              f"{MAX_IMAGE_RETRIES + 1} attempts -- proceeding with the last attempt "
              f"anyway (see review text above), since this signal isn't fully "
              f"reliable and a human/agent should make the final call, not an "
              f"infinite retry loop.", flush=True)
        return last_path

    # kf_backend (2026-08-12): a pipeline-wide 2x2 choice -- first frame
    # and middle/last are each independently local or Gemini:
    #   "all_local" (default, cheapest): unchanged -- first respects this
    #     Tale's OWN first_frame_source=="online" toggle (may still be
    #     Gemini-seeded per-Tale); middle/last always local I2I.
    #   "all_gemini": first ALWAYS via Gemini (force=True, ignoring this
    #     Tale's own first_frame_source), middle/last always Gemini edit.
    #   "first_local_rest_gemini": first respects the per-Tale toggle
    #     same as all_local; middle/last always Gemini edit.
    #   "first_gemini_rest_local": first ALWAYS via Gemini (force=True);
    #     middle/last always local I2I, conditioned on that Gemini first
    #     frame same as any other first-frame image.
    kf_backend = dream_step.load_config().get("kf_backend", "all_local")
    force_first_gemini = kf_backend in ("all_gemini", "first_gemini_rest_local")
    use_gemini_for_rest = kf_backend in ("all_gemini", "first_local_rest_gemini")

    if first_changed:
        first_path = try_online_first_frame(spec, dest_dir, keyframe_prompts["first"],
                                             dest_paths["first"], force=force_first_gemini)
        if first_path is None:
            blank_filename = f"t2i_blank_{spec['number']}.png"
            upload_image_to_comfyui(comfyui_base, PIPELINE_DIR / "blank_placeholder_512x896.png", blank_filename)
            first_path = generate_one("first", blank_filename, 1)
    else:
        first_path = dest_paths["first"]

    # Middle/last both reference the first frame's actual result, not the
    # blank placeholder or each other -- upload it under a fresh filename
    # so both LoadImage calls pick it up. Skipped entirely when both
    # roles are Gemini-generated -- nothing local ever reads this upload
    # in that case.
    first_reference_filename = f"t2i_frame1_{spec['number']}.png"
    if not use_gemini_for_rest:
        upload_image_to_comfyui(comfyui_base, first_path, first_reference_filename)

    if first_changed or role_changed("middle"):
        middle_path = generate_one("middle", first_reference_filename, 2,
                                    compare_against=(first_path, "first"),
                                    gemini_reference_image_path=first_path if use_gemini_for_rest else None)
    else:
        middle_path = dest_paths["middle"]

    if first_changed or role_changed("last"):
        # Reviewed against MIDDLE, not first (unlike middle's own review
        # just above, and unlike the actual I2I generation a few lines up,
        # which still conditions both middle AND last on first -- kept
        # that way deliberately, see this function's docstring on
        # compounding generation drift). The review's job is different:
        # it's protecting the video's actual transitions, which run
        # first->middle->last IN ORDER, so what matters for a smooth
        # middle->last cut is whether THOSE TWO agree with each other --
        # last matching first fine while drifting from middle would still
        # produce a jarring cut the old first-anchored review had no way
        # to catch.
        last_path = generate_one("last", first_reference_filename, 3,
                                  compare_against=(middle_path, "middle"),
                                  gemini_reference_image_path=first_path if use_gemini_for_rest else None)
    else:
        last_path = dest_paths["last"]

    sidecar_path.write_text(json.dumps(keyframe_prompts, indent=2), encoding="utf-8")
    return {"first": first_path, "middle": middle_path, "last": last_path}


def generate_i2v_first_frame(spec, prompt_text, comfyui_base, comfyui_output_dir, dest_dir):
    """Auto-generate the single still image an i2v render needs, when the
    spec has no usable image_path yet. Same T2I-from-blank-placeholder
    mechanism generate_keyframes uses for fml2v's first frame -- reused
    here rather than duplicated, since it's the same graph and the same
    "no source image exists yet" problem for a single frame instead of
    three. Saves as 1.png in the Dream's own folder (matching the
    manual-drop convention find_reference_images looks for) and returns
    the path.

    prompt_text: a still-image description of the opening frame (what
    the animal/scene looks like before the i2v animation starts) -- NOT
    positive_prompt (that describes the ANIMATION, used separately for
    the actual video render once this image exists).
    """
    workflow_cfg, template = load_workflow_template("t2i_i2i")

    # kf_backend's "force Gemini for the first frame" options apply here
    # too -- i2v has no middle/last, but this IS that workflow's
    # equivalent single "first frame" decision. See generate_keyframes'
    # own comment for the full 2x2 explanation.
    force_first_gemini = dream_step.load_config().get("kf_backend", "all_local") in (
        "all_gemini", "first_gemini_rest_local")
    online_path = try_online_first_frame(spec, dest_dir, prompt_text, dest_dir / "1.png",
                                          force=force_first_gemini)
    if online_path is not None:
        return online_path

    blank_filename = f"t2i_blank_i2v_{spec['number']}.png"
    upload_image_to_comfyui(comfyui_base, PIPELINE_DIR / "blank_placeholder_512x896.png", blank_filename)

    def attempt():
        prompt, _ = build_prompt(
            template, workflow_cfg, prompt_text, None,
            randomize_seeds=True, image_filename=blank_filename)
        print("[generate_dream] stage: t2i (i2v first frame)", flush=True)
        prompt_id = queue_prompt(comfyui_base, prompt)
        history_entry = wait_for_history(comfyui_base, prompt_id)
        image_item = find_output_image(history_entry)
        tmp_image_path = download_or_locate(comfyui_base, image_item, comfyui_output_dir)
        dest_path = dest_dir / "1.png"
        try:
            shutil.copy2(tmp_image_path, dest_path)
        finally:
            if tmp_image_path.parent == PIPELINE_DIR:
                tmp_image_path.unlink(missing_ok=True)
        return dest_path

    # Review against the intended description before accepting, same as
    # fml2v's keyframes -- catch a bad frame here, before the expensive
    # full video render runs on top of it.
    last_path = None
    for n in range(1, MAX_IMAGE_RETRIES + 2):
        last_path = attempt()
        passed, review_text = review_image_against_description(last_path, prompt_text)
        # See the identical comment in generate_keyframes -- display-only
        # sanitizing, the retry decision above already used the raw text.
        print(f"[generate_dream] i2v first-frame attempt {n}/{MAX_IMAGE_RETRIES + 1} "
              f"vision review:\n{dream_step.sanitize_review_text_for_log(review_text)}", flush=True)
        if passed is not False:
            return last_path
        print(f"[generate_dream] first frame failed review, "
              f"{'retrying with a fresh seed' if n <= MAX_IMAGE_RETRIES else 'out of retries'}...",
              flush=True)
    print(f"[generate_dream] >>> first frame still failed review after "
          f"{MAX_IMAGE_RETRIES + 1} attempts -- proceeding with the last attempt "
          f"anyway (see review text above), since this signal isn't fully reliable "
          f"and a human/agent should make the final call, not an infinite retry loop.",
          flush=True)
    return last_path


def ffprobe_check(path):
    """Confirms a render actually produced a real, playable video within
    the expected duration band. Uses PyAV (a Python binding to ffmpeg's
    own decoding libraries, pip-installed -- see requirements.txt)
    instead of shelling out to a separate ffprobe binary -- confirmed
    live 2026-08-07 reading the same stream/duration info a real
    ffprobe call would, with no subprocess call or system PATH
    dependency at all."""
    path = Path(path)
    try:
        container = av.open(str(path))
        try:
            has_video = any(s.type == "video" for s in container.streams)
            duration = float(container.duration) / 1_000_000 if container.duration else None
        finally:
            container.close()
    except Exception as e:
        raise RuntimeError(f"video probe failed: {e}")
    if not has_video:
        raise RuntimeError("no video stream found in output")
    if duration is None:
        raise RuntimeError("could not read video duration")
    if path.stat().st_size <= 0:
        raise RuntimeError("output file is empty")
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise RuntimeError(f"duration {duration:.2f}s outside expected [{MIN_DURATION_S},{MAX_DURATION_S}]s band")
    return duration


def write_txt(dest_path, spec, final_negative_prompt, keyframe_prompts=None):
    """keyframe_prompts: the fml2v_keyframe_prompts dict (first/middle/last
    still-image descriptions), if this was an fml2v render -- these are
    NOT the same as spec['positive_prompt'] (the video animation prompt)
    and were previously missing from this file entirely, leaving no
    written record of what each keyframe still was actually asked to
    show."""
    sections = [
        "POSITIVE PROMPT:\n\n"
        f"{spec['positive_prompt']}\n\n"
        "NEGATIVE PROMPT:\n\n"
        f"{final_negative_prompt}\n\n"
    ]
    if keyframe_prompts:
        sections.append(
            "KEYFRAME IMAGE PROMPTS (first/middle/last stills):\n\n"
            f"First: {keyframe_prompts.get('first', '')}\n\n"
            f"Middle: {keyframe_prompts.get('middle', '')}\n\n"
            f"Last: {keyframe_prompts.get('last', '')}\n\n"
        )
    sections.append(
        "DESCRIPTION:\n\n"
        f"{spec['description']}\n\n"
        "TAGS:\n\n"
        f"{spec['tags']}\n"
    )
    dest_path.write_text("".join(sections), encoding="utf-8")


def update_index(spec, folder_name, workflow_name, used_seeds=None):
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else []
    entry = {
        "number": spec["number"],
        "title": spec["title"],
        "folder": folder_name,
        "published": False,
        "premise": spec.get("premise", ""),
        "workflow": workflow_name,
        "seeds": used_seeds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Replace any existing entry for this number (e.g. a re-render) instead of
    # appending a duplicate.
    index = [e for e in index if e.get("number") != spec["number"]]
    index.append(entry)
    index.sort(key=lambda e: e["number"])
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def run_once(spec, template, workflow_cfg, comfyui_base, comfyui_output_dir, seeds=None,
             randomize_seeds=False, positive_prompt=None, negative_prompt=None,
             image_filename=None, image_filenames=None, guide_strengths=None):
    prompt, used_seeds = build_prompt(
        template, workflow_cfg,
        positive_prompt if positive_prompt is not None else spec["positive_prompt"],
        negative_prompt if negative_prompt is not None else spec["negative_prompt"],
        seeds, randomize_seeds, image_filename=image_filename, image_filenames=image_filenames,
        guide_strengths=guide_strengths)
    prompt_id = queue_prompt(comfyui_base, prompt)
    history_entry = wait_for_history(comfyui_base, prompt_id)
    video_item = find_output_video(history_entry)
    tmp_video_path = download_or_locate(comfyui_base, video_item, comfyui_output_dir)
    duration = ffprobe_check(tmp_video_path)
    return tmp_video_path, duration, used_seeds


TEST_RENDER_POSITIVE_PROMPT = "a red apple sitting on a plain white table, soft studio lighting"
TEST_RENDER_NEGATIVE_PROMPT = "blurry, low quality, distorted"


def run_test_render(graph_path, wiring, comfyui_output_dir, test_image_paths=None):
    """Settings' "Test this wiring?" step (see workflow_introspect.py) --
    a real, standalone render through ComfyUI using a candidate wiring
    config that has NOT been added to WORKFLOWS/custom_workflows.json
    yet, with a fixed placeholder prompt so a human can visually judge
    whether the detected positive/negative/image nodes actually do what
    their names suggest before the wiring is trusted. Deliberately
    bypasses load_workflow_template()/run_once() (which both expect an
    already-registered workflow name and a real spec) -- this is a
    one-off probe, not a real render.

    test_image_paths: None for t2v, a single Path for i2v, or a dict
    {"first"/"middle"/"last": Path} for fml -- uploaded to ComfyUI under
    fixed test filenames before the render, matching however "wiring"
    says the image node(s) are wired.

    Returns (tmp_video_path, used_seeds). Raises on any ComfyUI-side
    failure (missing node, execution error, timeout) -- callers should
    show that exception's message directly as the "why it failed"
    explanation, not swallow it."""
    template = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    comfyui_base = find_comfyui_base_url()

    image_filename = None
    image_filenames = None
    if test_image_paths is not None:
        if isinstance(test_image_paths, dict):
            image_filenames = {}
            for role, src in test_image_paths.items():
                dest_name = f"_test_wiring_{role}{Path(src).suffix}"
                upload_image_to_comfyui(comfyui_base, src, dest_name)
                image_filenames[role] = dest_name
        else:
            dest_name = f"_test_wiring_image{Path(test_image_paths).suffix}"
            upload_image_to_comfyui(comfyui_base, test_image_paths, dest_name)
            image_filename = dest_name

    prompt, used_seeds = build_prompt(
        template, wiring, TEST_RENDER_POSITIVE_PROMPT, TEST_RENDER_NEGATIVE_PROMPT,
        randomize_seeds=True, image_filename=image_filename, image_filenames=image_filenames)
    prompt_id = queue_prompt(comfyui_base, prompt)
    history_entry = wait_for_history(comfyui_base, prompt_id)
    video_item = find_output_video(history_entry)
    tmp_video_path = download_or_locate(comfyui_base, video_item, comfyui_output_dir)
    return tmp_video_path, used_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workflow", choices=sorted(WORKFLOWS), default=None,
                     help="Overrides the spec's \"workflow\" field, if any.")
    ap.add_argument("--randomize-seeds", action="store_true",
                     help="Use fresh random seeds instead of the graph's saved values.")
    ap.add_argument("--seeds", default=None,
                     help="Comma-separated noise seeds to reproduce an exact render "
                          "(e.g. --seeds 10,0). Overrides the spec's \"seeds\" field. "
                          "Omit for random. Lock these before iterating on wording, or "
                          "you cannot tell a prompt change from a re-roll.")
    ap.add_argument("--comfyui-output-dir", default=None,
                     help="Same-machine fast path: if ComfyUI's real output folder is "
                          "reachable at this local path, its files are read directly "
                          "instead of downloaded over HTTP. Omit entirely (the default) "
                          "when ComfyUI is on a different machine/container -- "
                          "download_or_locate() already falls back to HTTP over the "
                          "network in that case, so no default path is needed.")
    args = ap.parse_args()

    global DREAMS_DIR, INDEX_PATH
    spec_path = Path(args.spec).resolve()
    DREAMS_DIR = spec_path.parent.parent  # spec lives in <project>/_data/, output goes in <project>/
    INDEX_PATH = spec_path.parent / "index.json"
    # Confirmed real bug (2026-08-12): this script has always derived its
    # own DREAMS_DIR/INDEX_PATH directly from --spec rather than calling
    # dream_step.resolve_project_globals() -- harmless while nothing here
    # touched dream_step's own module-level DATA_DIR, but
    # apply_render_settings() now calls dream_step.project_render_settings()
    # -> creative_guidance_pointer() -> `DATA_DIR / "CREATIVE.md"`, and
    # DATA_DIR was still None in this subprocess -- "unsupported operand
    # type(s) for /: 'NoneType' and 'str'", a real render-breaking crash.
    # spec_path.parent IS this project's _data dir already (see DREAMS_DIR
    # above), exactly what resolve_project_globals() would have set.
    dream_step.DATA_DIR = spec_path.parent

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    workflow_name = args.workflow or spec.get("workflow") or DEFAULT_WORKFLOW
    workflow_cfg, template = load_workflow_template(workflow_name)

    raw_seeds = args.seeds if args.seeds is not None else spec.get("seeds")
    if isinstance(raw_seeds, str):
        seeds = [int(s) for s in raw_seeds.split(",") if s.strip()]
    elif isinstance(raw_seeds, (list, tuple)):
        seeds = [int(s) for s in raw_seeds]
    else:
        seeds = None

    episode_label = get_episode_label(spec_path.parent)
    folder_name = f"{episode_label} #{spec['number']} {spec['title']}"
    folder_name_safe = sanitize_filename(folder_name)
    dest_dir = DREAMS_DIR / folder_name_safe
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_mp4 = dest_dir / f"{folder_name_safe}.mp4"
    dest_txt = dest_dir / f"{folder_name_safe}.txt"

    result = {"ok": False, "path": None, "duration": None, "error": None}
    last_error = None
    try:
        comfyui_base = find_comfyui_base_url()
    except Exception as e:
        result["error"] = str(e)
        print(json.dumps(result))
        sys.exit(1)

    is_i2v = bool(workflow_cfg.get("image_node"))
    is_fml2v = bool(workflow_cfg.get("image_nodes"))

    i2v_image_filename = None
    if is_i2v:
        # Confirmed on Tale #83 (2026-08-08): positive_prompt/negative_prompt
        # are the ONE source of truth for every workflow now -- image_path is
        # the only thing i2v needs beyond the base spec fields (see run_once,
        # which renders straight from spec["positive_prompt"]/
        # spec["negative_prompt"] when no override is given). A separate
        # i2v_positive_prompt/i2v_negative_prompt used to exist "to avoid
        # re-describing appearance the image already fixes," but in practice
        # 29 of 37 real i2v specs had it byte-identical to positive_prompt
        # anyway, and the other 8 were just incomplete/truncated copies of
        # it -- never a deliberately different, better-tuned version. #83's
        # i2v_positive_prompt was one of those incomplete copies (no
        # dialogue at all), silently used instead of its own good, complete
        # positive_prompt, producing a real render with no scripted lip-sync
        # content. Two fields that are supposed to hold the same content but
        # can silently drift apart is worse than one field that's sometimes
        # more detailed than strictly necessary.
        # Bug fix (2026-08-09): this used to hard-require image_path be
        # PRE-SET on the spec, full stop -- before ever even looking at
        # i2v_generate_image_prompt. That's inconsistent with fml2v just
        # below, which correctly treats "no images yet, but a generate-
        # prompt is set" as a normal case needing nothing pre-computed.
        # Confirmed real: a spec written with ONLY i2v_generate_image_
        # prompt set (image_path never given -- nothing to point it at
        # yet, since generate_i2v_first_frame decides that path itself)
        # always failed here, even though everything needed to auto-
        # generate the image was right there. image_path is relative to
        # the project folder (DREAMS_DIR) unless already absolute --
        # keeps specs portable across machines the same way every other
        # path in this pipeline is project-relative.
        src_image = None
        image_path_str = spec.get("image_path")
        if image_path_str:
            # Normalizes legacy backslash-separated values (written while
            # this pipeline ran on Windows) -- see dream_step.rel_path_str's
            # docstring for the exact "does not exist" failure this
            # prevents on Linux, where '\\' is a normal filename character,
            # not a path separator.
            candidate = Path(image_path_str.replace("\\", "/"))
            if not candidate.is_absolute():
                candidate = DREAMS_DIR / candidate
            if candidate.exists():
                src_image = candidate
        if src_image is None:
            gen_prompt = spec.get("i2v_generate_image_prompt")
            if not gen_prompt:
                raise SystemExit(
                    f"workflow is 'i2v' but spec has neither an existing reference "
                    f"image nor an image-generation prompt.\n"
                    f"EXPECTED: either (a) image_path pointing to an already-existing "
                    f"image, OR (b) i2v_generate_image_prompt (a still-image "
                    f"description of the opening frame) to auto-generate one via T2I.\n"
                    f"TO FIX: add one of these to the spec via "
                    f"'dream_step.py --write-spec {spec.get('number', 'N')} "
                    f"--spec-json-stdin' with the rest of the spec's fields unchanged, "
                    f"then run this again.")
            src_image = generate_i2v_first_frame(
                spec, gen_prompt, comfyui_base, args.comfyui_output_dir, dest_dir)
        i2v_image_filename = f"i2v_dream_{spec['number']}{src_image.suffix}"
        upload_image_to_comfyui(comfyui_base, src_image, i2v_image_filename)

    fml2v_image_filenames = None
    fml2v_guide_strengths = None
    if is_fml2v:
        # If the three keyframe images aren't already provided (the user
        # dropped 1/2/3 into the Dream's folder manually), but the spec has
        # fml2v_keyframe_prompts, auto-generate them via T2I/I2I first (see
        # generate_keyframes) and use those results in their place.
        images_already_given = all(spec.get(f) for f in
                                    ("fml2v_first_image", "fml2v_middle_image", "fml2v_last_image"))
        if not images_already_given:
            keyframe_prompts = spec.get("fml2v_keyframe_prompts")
            if not keyframe_prompts or not all(keyframe_prompts.get(r) for r in ("first", "middle", "last")):
                raise SystemExit(
                    "workflow is 'fml2v' but spec has neither existing keyframe images "
                    "nor a complete fml2v_keyframe_prompts to generate them.\n"
                    "EXPECTED: either (a) fml2v_first_image/fml2v_middle_image/"
                    "fml2v_last_image pointing to three already-existing images, OR "
                    "(b) fml2v_keyframe_prompts as {\"first\": ..., \"middle\": ..., "
                    "\"last\": ...} (three still-image descriptions) to auto-generate "
                    "them via T2I/I2I.\n"
                    f"TO FIX: add fml2v_keyframe_prompts to the spec via "
                    f"'dream_step.py --write-spec {spec.get('number', 'N')} "
                    f"--spec-json-stdin' with the rest of the spec's fields unchanged, "
                    f"then run this again.")
            generated = generate_keyframes(
                spec, keyframe_prompts, comfyui_base, args.comfyui_output_dir, dest_dir)
            spec = {**spec,
                    "fml2v_first_image": str(generated["first"]),
                    "fml2v_middle_image": str(generated["middle"]),
                    "fml2v_last_image": str(generated["last"])}
        for field in ("fml2v_first_image", "fml2v_middle_image", "fml2v_last_image"):
            if not spec.get(field):
                raise SystemExit(
                    f"workflow is 'fml2v' but spec is missing required field '{field}'.\n"
                    f"EXPECTED: fml2v needs all three fml2v_first_image/"
                    f"fml2v_middle_image/fml2v_last_image (image paths) -- "
                    f"positive_prompt/negative_prompt (the animation across all "
                    f"three) are already required of every spec.\n"
                    f"TO FIX: add '{field}' to the spec via "
                    f"'dream_step.py --write-spec {spec.get('number', 'N')} "
                    f"--spec-json-stdin' with the rest of the spec's fields "
                    f"unchanged, then run this again.")
        fml2v_image_filenames = {}
        for role, field in (("first", "fml2v_first_image"), ("middle", "fml2v_middle_image"),
                            ("last", "fml2v_last_image")):
            # Normalizes legacy backslash-separated values -- see
            # dream_step.rel_path_str's docstring.
            src_image = Path(spec[field].replace("\\", "/"))
            if not src_image.is_absolute():
                src_image = DREAMS_DIR / src_image
            if not src_image.exists():
                # Confirmed real, repeated failure (2026-08-13): a human
                # manually replacing a keyframe image with a different file
                # TYPE (e.g. swapping in a .jpg where the spec's stored
                # path still says .png -- GUI upload/drag-drop keeps
                # whatever extension the source file actually had) left
                # the exact stored path pointing at nothing, even though
                # the real, intended image is sitting right there under
                # the same stem with a different suffix. find_reference_
                # images already treats a slot as satisfied by matching
                # the STEM alone (1/2/3), not the extension -- this now
                # does the same fallback here instead of hard-failing, so
                # a manually-swapped file actually gets used on the very
                # next render instead of silently continuing to be
                # ignored call after call.
                fallback = None
                for candidate in src_image.parent.glob(f"{src_image.stem}.*"):
                    if candidate.is_file():
                        fallback = candidate
                        break
                if fallback is not None:
                    print(f"[generate_dream] {field}'s stored path ({src_image.name}) doesn't "
                          f"exist, but found {fallback.name} at the same stem -- using it instead.",
                          flush=True)
                    src_image = fallback
                else:
                    raise SystemExit(
                        f"spec's {field} does not exist: {src_image}\n"
                        f"EXPECTED: this must point to a real, already-existing image file.\n"
                        f"TO FIX: either place the image at that exact path, or correct "
                        f"'{field}' via 'dream_step.py --write-spec {spec.get('number', 'N')} "
                        f"--spec-json-stdin' to point at wherever the real image actually is.")
            dest_filename = f"fml2v_dream_{spec['number']}_{role}{src_image.suffix}"
            upload_image_to_comfyui(comfyui_base, src_image, dest_filename)
            fml2v_image_filenames[role] = dest_filename
        fml2v_guide_strengths = spec.get("fml2v_guide_strengths")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Only reuse the requested seeds on the first attempt; a retry
            # exists because something failed, so re-roll rather than repeat it.
            attempt_seeds = seeds if attempt == 1 else None
            print(f"[generate_dream] stage: {workflow_name} (video)", flush=True)
            tmp_video_path, duration, used_seeds = run_once(
                spec, template, workflow_cfg, comfyui_base, args.comfyui_output_dir,
                attempt_seeds, args.randomize_seeds,
                image_filename=i2v_image_filename, image_filenames=fml2v_image_filenames,
                guide_strengths=fml2v_guide_strengths)
            try:
                shutil.copy2(tmp_video_path, dest_mp4)
            finally:
                # try/finally so a failed copy still cleans up the
                # downloaded temp file instead of leaking it in
                # PIPELINE_DIR forever.
                if tmp_video_path.parent == PIPELINE_DIR:
                    tmp_video_path.unlink(missing_ok=True)
            write_txt(dest_txt, spec, spec["negative_prompt"],
                      keyframe_prompts=spec.get("fml2v_keyframe_prompts") if is_fml2v else None)
            update_index(spec, folder_name_safe, workflow_name, used_seeds)
            result.update({"ok": True, "path": str(dest_mp4), "duration": duration,
                           "attempt": attempt, "workflow": workflow_name, "seeds": used_seeds})
            break
        except Exception as e:
            last_error = f"attempt {attempt}: {e}"
            continue
    else:
        result["error"] = last_error

    if not result["ok"] and result["error"] is None:
        result["error"] = last_error

    print(json.dumps(result))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
