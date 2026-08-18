"""Gemini text generation -- an optional backend for
dream_step.py's _creative_completion() (spec/keyframe JSON composition),
alongside the default local Ollama backend. Selected via config.json's
creative_backend.

Reuses the SAME saved Gemini API key as gemini_image.py (one Gemini
account, two uses -- image generation and text generation are separate
Gemini models, but the same key authenticates both) rather than asking
for a second key. Only usable once that key is saved -- see
gemini_image.is_enabled()/load_api_key(), imported directly rather than
duplicated here.

MODEL defaults to a real, current Gemini text model -- unlike image
generation, Gemini's plain text models are usable on the free tier, so
this is a genuinely free option once a key is saved (confirmed by
gemini_image.py's own docstring: it's specifically IMAGE generation that
has no free tier). See list_text_models() for why the exact model ID is
never hardcoded as the sole source of truth.
"""
import json
import urllib.error
import urllib.request

import gemini_image

# A pinned model id like "gemini-2.5-flash" can become retired-for-new-users
# while still appearing in models.list, causing generateContent calls to
# 404. "-latest" is a stable alias Google itself maintains to always point
# at the current recommended flash model, avoiding that failure mode.
MODEL = "gemini-flash-latest"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _resolve_model():
    """The CREATIVE-writing model specifically -- generate_vision_text/
    generate_json_with_search below each have their OWN resolver reading
    their own config key, never falling back to this one. See
    DEFAULT_CONFIG's own comment on why they're independent."""
    import dream_step
    return dream_step.load_config().get("gemini_text_model") or MODEL


def _resolve_vision_model():
    import dream_step
    return dream_step.load_config().get("gemini_vision_model") or MODEL


def _resolve_search_model():
    import dream_step
    return dream_step.load_config().get("gemini_text_model") or MODEL


def list_text_models(api_key=None):
    """Every model this key can see that supports plain text generateContent
    output and ISN'T an image-generation model (those are listed
    separately by gemini_image.list_image_models()) -- same live-lookup
    reasoning as that function's own docstring: a hardcoded model ID goes
    stale as Google renames/retires them."""
    api_key = api_key or gemini_image.load_api_key()
    url = f"{API_BASE}/models?pageSize=1000"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
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
        if "image" in model_id.lower() or "imagen" in model_id.lower():
            continue
        # embedding/tts-only models also support generateContent in some
        # listings but aren't usable as a chat/JSON-composition model.
        if "embedding" in model_id.lower() or "tts" in model_id.lower():
            continue
        models.append(model_id)
    return sorted(set(models))


def generate_vision_text(prompt, b64_images, model=None, api_key=None):
    """Single-shot image-in/text-out call -- Gemini alternative to
    dream_step._vision_query's default Ollama implementation, same
    (prompt, b64_images) -> raw response text contract so callers (keyframe
    QC review, image-vs-description matching) never need to know which
    backend answered. Selected via config.json's vision_backend.

    Uses the same generateContent endpoint as generate_json, just with
    inlineData image parts mixed into the request alongside the text
    prompt -- Gemini's multimodal input, not a separate vision-specific
    endpoint. b64_images are assumed PNG (this pipeline's own generated
    keyframes/renders), same assumption dream_step._vision_query already
    makes for Ollama."""
    api_key = api_key or gemini_image.load_api_key()
    model = model or _resolve_vision_model()
    url = f"{API_BASE}/models/{model}:generateContent"
    parts = [{"text": prompt}] + [
        {"inlineData": {"mimeType": "image/png", "data": b64}} for b64 in b64_images]
    body = json.dumps({"contents": [{"parts": parts}]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body[:500]}")
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        return "(no response" + (f" -- blocked: {block_reason})" if block_reason else ")")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts) or "(no response)"


def generate_json_with_search(prompt, retries=3, model=None, api_key=None):
    """Same contract as generate_json (JSON in, JSON out), but with
    Gemini's native google_search grounding tool enabled -- the server
    itself decides whether/when to search and folds results into its
    answer, so this needs no local tool-execution loop the way
    dream_step._ollama_tool_completion does for the local-model path.
    Selected via config.json's tool_calling_backend.

    responseMimeType="application/json" (generate_json's usual strict-
    JSON mode) is NOT combined with tools here -- Gemini rejects that
    combination -- so this relies on the same permissive prompt-
    instruction + first-{...}/[...]-block extraction generate_json
    already uses for that reason."""
    api_key = api_key or gemini_image.load_api_key()
    model = model or _resolve_search_model()
    url = f"{API_BASE}/models/{model}:generateContent"
    attempt_prompt = prompt
    history = []
    for attempt in range(1, retries + 1):
        body = json.dumps({
            "contents": [{"parts": [{"text": attempt_prompt}]}],
            "tools": [{"google_search": {}}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST",
                                      headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            history.append(f"attempt {attempt}: Gemini API error {e.code}: {error_body[:300]}")
            continue
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason")
            history.append(f"attempt {attempt}: no candidates"
                            + (f" (blocked: {block_reason})" if block_reason else ""))
            continue
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        raw = "".join(p.get("text", "") for p in parts)
        import re
        m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if not m:
            history.append(f"attempt {attempt}: no JSON object/array found: {raw[:300]!r}")
            attempt_prompt = (f"{prompt}\n\nYour previous answer didn't contain a JSON "
                               f"object/array. Reply with ONLY the JSON, nothing else.")
            continue
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            history.append(f"attempt {attempt}: invalid JSON ({e}): {raw[:300]!r}")
            attempt_prompt = (f"{prompt}\n\nYour previous answer was not valid JSON "
                               f"({e}). Reply with ONLY the JSON object, nothing else.")
            continue
        return parsed, history
    raise RuntimeError(
        f"[gemini_text] generate_json_with_search failed after {retries} attempts:\n" +
        "\n".join(history))


def generate_json(prompt, retries=3, model=None, api_key=None):
    """Direct, single-shot call to Gemini for creative content -- same
    contract as dream_step._creative_completion (JSON in, JSON out,
    retries with the parse error fed back on failure) so the two backends
    are interchangeable from that function's point of view.

    Does NOT set responseMimeType="application/json". That setting forces
    constrained/structured decoding -- the model has to keep every token
    choice valid against a JSON grammar, which is a well-documented
    creativity tax on generation quality, distinct from and additional
    to whatever effect the prompt's own wording has. Ollama's parallel
    format="json" constraint is the same mechanism and is a likely
    contributor to its weaker output too.

    Uses free-form generation + a permissive first-{...}/[...]-block
    extraction instead -- the prompt's own existing "Return ONLY a valid
    JSON object" instruction is what keeps output parseable, same as it
    already does for Ollama's prompts."""
    api_key = api_key or gemini_image.load_api_key()
    model = model or _resolve_model()
    url = f"{API_BASE}/models/{model}:generateContent"
    attempt_prompt = prompt
    history = []
    for attempt in range(1, retries + 1):
        body = json.dumps({
            "contents": [{"parts": [{"text": attempt_prompt}]}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST",
                                      headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            history.append(f"attempt {attempt}: Gemini API error {e.code}: {error_body[:300]}")
            continue
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason")
            history.append(f"attempt {attempt}: no candidates"
                            + (f" (blocked: {block_reason})" if block_reason else ""))
            continue
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        raw = "".join(p.get("text", "") for p in parts)
        import re
        m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if not m:
            history.append(f"attempt {attempt}: no JSON object/array found: {raw[:300]!r}")
            attempt_prompt = (f"{prompt}\n\nYour previous answer didn't contain a JSON "
                               f"object/array. Reply with ONLY the JSON, nothing else.")
            continue
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            history.append(f"attempt {attempt}: invalid JSON ({e}): {raw[:300]!r}")
            attempt_prompt = (f"{prompt}\n\nYour previous answer was not valid JSON "
                               f"({e}). Reply with ONLY the JSON object, nothing else.")
            continue
        return parsed, history
    raise RuntimeError(
        f"[gemini_text] generate_json failed after {retries} attempts:\n" +
        "\n".join(history))
