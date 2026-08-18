"""Gemini image generation -- the sole "Online photo" reference-image
source for animals the local Flux2 checkpoint tends to hallucinate
(confirmed live 2026-08-09: a real gemini-3.1-flash-image generation of
a flying squirrel showed a correct patagium/flattened tail, something
neither the free CC0-photo lookup -- 5 of 6 wrong species matches --
nor a same-class hosted diffusion model, Hugging Face's SD3-medium, got
right on the same species). A large foundation model has genuinely seen
more of a rare species' real appearance than a diffusion checkpoint,
local or hosted.

PAID ONLY -- confirmed live across multiple models and a fresh key/
project that the Gemini API's free tier is not usable for image
generation on this account (every image model returned a hard
`limit: 0` quota error; billing must be linked to unlock ANY image
generation, free-tier-priced or not). Settings' Gemini section states
this plainly rather than implying a free option exists.

Only used when explicitly enabled (a saved API key present) -- when
not, or when a call fails, the caller (generate_dream.py's
try_online_first_frame) falls back to ordinary local blank-placeholder-
seeded generation. There is no other online fallback: the free CC0-
photo lookup (reference_photo.py) and Hugging Face (huggingface_image.py)
were both removed after this session's testing -- the photo lookup for
unreliable species matching, Hugging Face for no better accuracy than
local generation plus its own licensing notice requirement.

MODEL defaults to gemini-3.1-flash-image (confirmed genuinely GA, not a
preview build in disguise -- see list_image_models' own docstring on
why Gemini's naming can't be trusted at face value) but Settings'
"Gemini model" dropdown (config.json's gemini_model) is the real source
of truth, populated LIVE via list_image_models() rather than a
hardcoded guess, since Google renames/retires these often.

The API key is never handled by this module directly -- see
web_ui.py's Settings-tab Gemini section (mirrors the YouTube
client_secret encrypted-storage pattern via secret_store.py) and
load_api_key() below, which reads whatever that saved.
"""
import base64
import datetime
import json
import urllib.error
import urllib.request

MODEL = "gemini-3.1-flash-image"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _resolve_model():
    import dream_step
    return dream_step.load_config().get("gemini_model") or MODEL


def list_image_models(api_key=None):
    """Every model this key can see that supports image OUTPUT via
    generateContent -- lets Settings offer a real, live dropdown
    instead of a hardcoded guess at exactly which Gemini model ID is
    current right now (see the module docstring on why that guess keeps
    going stale). Gemini's /v1beta/models list has no single explicit
    "generates images" flag -- "image"/"imagen" appearing in the model's
    own ID is the only signal every current image-generation model
    (2.5-flash-image, 3.1-flash-image, imagen-*) reliably shares, so
    that's what this filters on, alongside requiring generateContent
    support at all (excludes embedding-only/deprecated entries). Note:
    a clean-looking ID is NOT proof a model is actually GA -- confirmed
    live that "gemini-2.5-flash-image" is internally backed by
    "gemini-2.5-flash-preview-image" (its own /models/{id} metadata
    literally describes it as "Preview"), which is why this list alone
    isn't enough; check each candidate's own description before
    trusting it as stable."""
    api_key = api_key or load_api_key()
    url = f"{API_BASE}/models?key={api_key}&pageSize=1000"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body[:500]}")
    models = []
    for m in data.get("models", []):
        model_id = (m.get("name") or "").split("/")[-1]
        if not model_id:
            continue
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        if "image" not in model_id.lower() and "imagen" not in model_id.lower():
            continue
        models.append(model_id)
    return sorted(set(models))


def _gemini_key_enc_path():
    """Local import to avoid a hard dependency on dream_step at module
    import time (this module is also usable standalone/from a scratch
    test script with no project resolved). Must match web_ui.py's own
    _secrets_base_dir() -- both need to agree on where this file lives,
    or Settings can report a key as saved (green) while this module
    still can't find it. Same DREAM_PIPELINE_CONFIG_DIR override as
    dream_step.CONFIG_PATH / secret_store._local_appdata_dir(), so a
    Docker container's /state volume is honored here too."""
    import os
    import dream_step
    override = os.environ.get("DREAM_PIPELINE_CONFIG_DIR")
    base = dream_step.Path(override) if override else dream_step.PIPELINE_DIR
    return base / "gemini" / "gemini_api_key.enc"


def is_enabled():
    """Whether Gemini sourcing should be tried at all -- true iff a key
    is saved AND Settings' Enable/Disable toggle (config.json's
    gemini_enabled, separate from the key itself -- see the config
    default's own comment on why: pausing spend shouldn't require
    deleting and later re-pasting the credential) is on. The single
    gate generate_dream.py's try_online_first_frame checks before
    calling this module at all, so either condition being false reads
    as a normal, silent skip straight to local generation, not a
    RuntimeError a caller has to specifically catch."""
    if not _gemini_key_enc_path().is_file():
        return False
    import dream_step
    return dream_step.load_config().get("gemini_enabled", True)


def load_api_key():
    """The saved Gemini API key, decrypted -- see web_ui.py's Settings
    Gemini section for how it gets there (secret_store.py, same
    encrypted-at-rest mechanism as the YouTube client_secret). Raises
    RuntimeError with a clear, actionable message if none is saved yet,
    rather than a bare FileNotFoundError a caller would have to
    translate itself."""
    path = _gemini_key_enc_path()
    if not path.is_file():
        raise RuntimeError(
            "no Gemini API key saved -- add one in Settings > Gemini "
            "(get one at https://aistudio.google.com/apikey -- note: PAID "
            "only, billing must be linked to the Google Cloud project, "
            "confirmed no usable free tier for image generation)")
    import secret_store
    return secret_store.decrypt_text(path.read_bytes()).strip()


def _usage_path():
    """Monthly Gemini call-count tracker -- not a secret, but real
    stateful usage data, so it follows the same DREAM_PIPELINE_CONFIG_DIR
    override as config.json/the .enc secrets (see web_ui._secrets_base_dir()):
    persisted in the Docker image's /state volume, never baked into the
    image itself as a stale snapshot of THIS environment's real usage."""
    import os
    import dream_step
    from pathlib import Path
    override = os.environ.get("DREAM_PIPELINE_CONFIG_DIR")
    base = Path(override) if override else dream_step.PIPELINE_DIR
    return base / "gemini" / "usage.json"


def _current_month_key():
    return datetime.date.today().strftime("%Y-%m")


def _load_usage():
    path = _usage_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def monthly_call_count(month_key=None):
    """Successful Gemini image generations recorded so far this
    calendar month (or `month_key`, e.g. "2026-08", if given) -- a
    simple call count, not a dollar figure, since per-image price
    varies by model/resolution and has already changed multiple times
    within this project's own lifetime (confirmed: 2.5 Flash Image,
    3.1 Flash Image, and 3 Pro Image all price differently) -- a fixed
    $ rate baked in here would just go stale. Settings shows an
    approximate $ estimate alongside the call-count limit instead."""
    return _load_usage().get(month_key or _current_month_key(), 0)


def _record_usage_call():
    path = _usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = _load_usage()
    key = _current_month_key()
    usage[key] = usage.get(key, 0) + 1
    path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    return usage[key]


def _check_pay_guard():
    """Settings' optional "Pay guard" -- off by default (the user is on
    a pay-as-you-go plan with its own spend cap already, so this is a
    convenience notification/stop, not a required safety net). When
    enabled with a positive monthly_limit, blocks further calls once
    this calendar month's successful-call count reaches it, with a
    clear, actionable RuntimeError (the caller then falls back to local
    generation, same as any other Gemini failure) rather than a silent
    stop or a surprise bill."""
    import dream_step
    config = dream_step.load_config()
    if not config.get("gemini_pay_guard_enabled"):
        return
    limit = config.get("gemini_pay_guard_monthly_limit") or 0
    if limit <= 0:
        return
    count = monthly_call_count()
    if count >= limit:
        raise RuntimeError(
            f"Gemini pay guard: {count} image generation(s) already made this "
            f"month, at or past the configured limit of {limit} "
            f"(Settings > Gemini > Pay guard). Raise the limit, disable the "
            f"guard, or wait until next month.")


def normalize_and_resize(path, target_size=(512, 896)):
    """Re-encodes an already-downloaded/generated image to real PNG
    bytes and center-crop+lanczos-resizes it to exactly target_size
    (default 512x896, matching blank_placeholder_512x896.png) -- both
    steps in place. Required before using ANY externally-sourced image
    as a T2I/I2I seed: workflow_api_t2i_flux2.json derives its OWN
    output latent size from the size of whatever reference image gets
    loaded (GetImageSize -> EmptyFlux2LatentImage), normally always the
    512x896 blank placeholder. Confirmed live (2026-08-09, originally
    against reference_photo.py's CC0-photo fetches, same underlying
    graph applies here): skipping this step let an arbitrarily-sized
    external image carry its own dimensions straight through, breaking
    the pipeline's portrait framing and compounding into fml2v's
    middle/last (each conditions on the PREVIOUS frame's actual pixel
    size the same way)."""
    from PIL import Image
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    with Image.open(path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_w = round(src_h * target_ratio)
            left = (src_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, src_h))
        else:
            new_h = round(src_w / target_ratio)
            top = (src_h - new_h) // 2
            img = img.crop((0, top, src_w, top + new_h))
        img = img.resize((target_w, target_h), Image.LANCZOS)
        img.save(path, "PNG")


class GeminiContentBlocked(RuntimeError):
    """Raised specifically when Gemini answered normally (a real HTTP 200,
    the API itself is up and the key/billing are fine) but declined to
    produce an image for this specific prompt -- a safety-filter block
    (finishReason PROHIBITED_CONTENT/SAFETY/etc, or promptFeedback.
    blockReason), or any other "responded but no image" shape. Distinct
    from a plain RuntimeError (HTTPError -- the API call itself failed:
    bad key, quota, network, service down) so callers can tell "this
    exact request got refused, retrying the same call might well work"
    apart from "the service itself isn't answering, retrying won't
    help, fall back to something that doesn't depend on it" -- see
    generate_dream.py's try_online_first_frame, which retries Gemini
    itself on this, and only falls back to local generation on a plain
    RuntimeError or after exhausting those retries."""


def generate_reference_image(prompt, dest_path, api_key=None, model=None):
    """Calls Gemini's image-generation model with `prompt`, saves the
    first image part returned to dest_path (bytes, overwriting), then
    normalizes+resizes it to the pipeline's standard seed dimensions
    (see normalize_and_resize). api_key defaults to load_api_key().
    model defaults to _resolve_model() (Settings' saved choice, falling
    back to MODEL only if nothing's been picked yet). This is a REAL
    BILLED call -- see h_gemini_key_test/list_image_models for the
    free, unbilled connectivity check Settings' "Test" button actually
    uses instead (note: that check passing does NOT prove image
    generation itself will work -- listing models is a free metadata
    call available even without billing, but actually generating an
    image requires a billing account linked to the project, confirmed
    live 2026-08-09 across multiple models on a fresh key/project).

    Raises GeminiContentBlocked (a RuntimeError subclass) when the API
    call itself succeeded but no image came back -- see that class's own
    docstring for why this is split out from a plain RuntimeError (bad
    key, quota, network, service down). Never silently falls back to
    anything; that decision belongs to the caller (see generate_dream.
    py's try_online_first_frame)."""
    _check_pay_guard()
    api_key = api_key or load_api_key()
    model = model or _resolve_model()
    url = f"{API_BASE}/models/{model}:generateContent?key={api_key}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body[:500]}")

    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        raise GeminiContentBlocked(
            f"Gemini returned no candidates"
            + (f" (blocked: {block_reason})" if block_reason else "")
            + f" -- full response: {json.dumps(data)[:500]}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            dest_path.write_bytes(base64.b64decode(inline["data"]))
            _record_usage_call()
            normalize_and_resize(dest_path)
            return
    raise GeminiContentBlocked(
        f"Gemini response had no image data -- full response: {json.dumps(data)[:500]}")


def edit_image(prompt, source_image_paths, dest_path, api_key=None, model=None):
    """Image-EDIT variant of generate_reference_image -- same model/
    endpoint (Gemini's image model handles image+text-in, image-out
    natively, no separate "edit" API), just with one or more real source
    images attached as inlineData parts alongside the text prompt,
    instead of text-only generation from scratch.

    Added 2026-08-12 for kf_middle_last_backend="gemini": generating an
    fml2v Tale's middle/last keyframes as a REMOTE image-edit off the
    real first-frame image, conditioned the same "maintain everything,
    but X changes" way local I2I already is -- kept as its own function
    rather than a source_image_paths=None branch bolted onto
    generate_reference_image, since a caller passing real source images
    is a meaningfully different (billed the same, but semantically
    distinct: edit vs from-scratch) operation worth naming separately.

    source_image_paths: list of Path -- read and base64-encoded here,
    in the order given (this pipeline always passes the first-frame
    image alone, matching local I2I's own single-reference conditioning
    -- see generate_dream.py's generate_keyframes). Raises RuntimeError
    on any failure, same as generate_reference_image -- no silent
    fallback, that decision belongs to the caller."""
    _check_pay_guard()
    api_key = api_key or load_api_key()
    model = model or _resolve_model()
    url = f"{API_BASE}/models/{model}:generateContent?key={api_key}"
    parts = [{"inlineData": {"mimeType": "image/png",
                              "data": base64.b64encode(p.read_bytes()).decode("utf-8")}}
             for p in source_image_paths] + [{"text": prompt}]
    body = json.dumps({"contents": [{"parts": parts}]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body[:500]}")

    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        raise GeminiContentBlocked(
            f"Gemini returned no candidates"
            + (f" (blocked: {block_reason})" if block_reason else "")
            + f" -- full response: {json.dumps(data)[:500]}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            dest_path.write_bytes(base64.b64decode(inline["data"]))
            _record_usage_call()
            normalize_and_resize(dest_path)
            return
    raise GeminiContentBlocked(
        f"Gemini response had no image data -- full response: {json.dumps(data)[:500]}")
