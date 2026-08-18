"""
Guided setup for Dream Pipeline -- installs/verifies everything this
pipeline needs: Python packages (always), Ollama, ComfyUI, and the
ComfyUI model files the bundled workflow graphs reference (see
install_manifest.py for exactly which files and where they come from).

SCOPE: Windows and Linux, NVIDIA GPUs only -- matches this pipeline's
actual tested configuration. AMD/Intel support is a real, separate
problem (different ComfyUI backends, and the fp8-quantized models this
pipeline uses have narrow hardware support even on NVIDIA -- Ada/
Hopper-class or newer) not attempted here; see help.html's FAQ.

Every step is skippable -- if you already have Ollama/ComfyUI/some
models installed, say so and this won't touch them. Existing model
files are detected by filename and never re-downloaded.

USAGE
-----
    python setup_installer.py
"""
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
import install_manifest  # noqa: E402


def ask_yes_no(prompt, default=True):
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            ans = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def _default_venv_dir():
    """Where the pipeline's own venv lives by default -- deliberately
    OUTSIDE PIPELINE_DIR, in this machine's own per-user local-app-data
    directory (same convention as secret_store._local_appdata_dir(),
    reused directly here rather than duplicated).

    A venv is NOT portable: it bakes in absolute paths and OS-specific
    binaries, so one created on Linux is useless on Windows and vice
    versa. Confirmed live 2026-08-15: this pipeline's own project
    folder is meant to be shareable over a network mount and run from
    multiple machines/OSes against the same data -- if .venv lived
    inside that shared folder (the old default), every machine would
    fight over or silently break the same directory. It also flatly
    can't be created there at all in one confirmed real case: a GVFS
    SMB-mounted path containing a literal ':' character makes
    `python -m venv` refuse outright ("Refusing to create a venv...
    because it contains the PATH separator :")."""
    import secret_store
    return secret_store._local_appdata_dir() / "venv"


def venv_python_path(venv_dir=None):
    """Path to the dedicated pipeline venv's own python -- the launcher
    every other entry point (web_ui.py, dream_step.py, etc) should be run
    with after setup. Confirmed live 2026-08-15: install_pip_requirements()
    used to install into sys.executable's own environment directly, which
    on modern Debian/Ubuntu is PEP-668 "externally managed" and refuses
    `pip install` outright (real error, blocks every Linux user on such a
    distro, not just this test) -- and separately, on a Python missing the
    `pip` module entirely, raised an unhandled CalledProcessError instead
    of any actionable message. A dedicated venv sidesteps both: it's never
    externally-managed, and `python -m venv` bundles its own pip."""
    venv_dir = Path(venv_dir) if venv_dir else _default_venv_dir()
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_pip_requirements():
    print("\n=== Python packages ===")
    req = PIPELINE_DIR / "requirements.txt"
    venv_dir = _default_venv_dir()
    venv_python = venv_python_path(venv_dir)
    if not venv_python.exists():
        print(f"Creating a dedicated virtual environment at {venv_dir} ...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to create the virtual environment: {e}")
            print("On Debian/Ubuntu this usually means python3-venv isn't installed -- "
                  "try 'sudo apt install python3-venv' and re-run this installer.")
            return
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(req)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Installing requirements into {venv_dir} failed: {e}")
        print("Check requirements.txt / network access, then re-run this installer.")
        return
    print(f"Python packages installed into {venv_dir}.")
    print(f"From now on, run the pipeline with THIS python, not the system one, e.g.:\n"
          f"  {venv_python} dream_step.py --web")


def _load_pipeline_config():
    try:
        return json.loads((PIPELINE_DIR / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_remote_url(url):
    """True if url's host isn't this machine -- used so the installer's
    local-binary checks (Ollama below, same idea applies to ComfyUI)
    don't force a local install when config.json already points at a
    reachable remote instance, which is this pipeline's actual supported
    topology (see dream_step.py's URL-driven ollama_url/comfyui_url)."""
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host not in ("", "localhost", "127.0.0.1", "::1")


def install_ollama():
    print("\n=== Ollama ===")
    ollama_url = _load_pipeline_config().get("ollama_url", "")
    if _is_remote_url(ollama_url):
        print(f"config.json's ollama_url already points at a non-local address ({ollama_url}).")
        if ask_yes_no("Treat this as an intentional remote Ollama instance and skip local install?"):
            print("Skipped -- using the remote Ollama instance from config.json.")
            return True
        print("Continuing with local Ollama install/check anyway.")
    if shutil.which("ollama"):
        print("Ollama already found on PATH -- skipping install.")
        return True
    if not ask_yes_no("Ollama not found. Install it now?"):
        print("Skipped -- install Ollama yourself (https://ollama.com/download) before using this pipeline.")
        return False
    system = platform.system()
    if system == "Windows":
        url = "https://ollama.com/download/OllamaSetup.exe"
        dest = PIPELINE_DIR / "OllamaSetup.exe"
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print("Launching the installer -- follow its prompts, then come back here...")
            subprocess.run([str(dest)], check=True)
        except Exception as e:
            print(f"Ollama install failed: {e}")
            print("Install it yourself instead: https://ollama.com/download")
            return False
        finally:
            dest.unlink(missing_ok=True)
    elif system == "Linux":
        print("Running the official install script (curl | sh)...")
        # Confirmed live 2026-08-15: fails with an unhandled traceback
        # when the script needs sudo and no password is available
        # non-interactively (e.g. this session) -- same crash pattern
        # install_pip_requirements() had, fixed the same way here.
        try:
            subprocess.run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Automated Ollama install failed (exit {e.returncode}) -- likely needs sudo "
                  "and none was available non-interactively.")
            print("Install it yourself instead: run 'curl -fsSL https://ollama.com/install.sh | sh' "
                  "in a terminal where sudo can prompt you, or see https://ollama.com/download")
            return False
    else:
        print(f"No automated installer for {system!r} -- install Ollama manually: https://ollama.com/download")
        return False
    print("Ollama install finished.")
    return True


def install_comfyui():
    print("\n=== ComfyUI ===")
    existing = input("Path to an EXISTING ComfyUI install (blank if you don't have one yet): ").strip()
    if existing:
        p = Path(existing)
        if not (p / "models").is_dir():
            print(f"WARNING: {p} doesn't look like a ComfyUI install (no models/ folder) -- "
                  f"model downloads may go to the wrong place.")
        else:
            print(f"Using existing ComfyUI at {p}")
        return p

    default_dir = PIPELINE_DIR.parent / "ComfyUI"
    if not ask_yes_no(f"Install ComfyUI now via git clone into {default_dir}?"):
        print("Skipped -- the model-download step below will ask for a path to target instead.")
        return None
    try:
        return Path(install_comfyui_noninteractive(default_dir))
    except Exception as e:
        print(f"ComfyUI install failed: {e}")
        return None


def install_comfyui_noninteractive(target_dir=None):
    """Non-interactive git-clone install used by both this CLI's
    automated paths and web_ui.py's "Download & Install" job (which
    can't prompt for input) -- always installs to target_dir (defaults
    to PIPELINE_DIR.parent / "ComfyUI") without asking, raises on any
    failure (git missing, clone/pip errors) so the caller's job status
    reports it as failed rather than silently continuing. Returns the
    installed path as a string on success."""
    target_dir = Path(target_dir) if target_dir else (PIPELINE_DIR.parent / "ComfyUI")
    if shutil.which("git") is None:
        raise RuntimeError("git not found -- install git first: https://git-scm.com/downloads")
    print(f"Cloning ComfyUI into {target_dir} ...")
    subprocess.run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI.git", str(target_dir)], check=True)
    print("Installing ComfyUI's Python requirements ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(target_dir / "requirements.txt")], check=True)
    print(f"ComfyUI installed at {target_dir}.")
    print("NOTE: GPU-specific PyTorch (the CUDA build) may need installing separately -- "
          "see https://pytorch.org/get-started/locally/ if renders fail with a CUDA error.")
    return str(target_dir)


def pull_ollama_models():
    print("\n=== Ollama models ===")
    if not shutil.which("ollama"):
        print("Ollama not on PATH -- skipping model pulls.")
        return
    import dream_step as ds
    config = ds.load_config()
    models = {config.get("creative_model"), config.get("vision_model")}
    models.discard(None)
    if not models:
        print("No models configured yet -- set Creative/Vision model in Settings first, "
              "or run 'ollama pull <model>' yourself.")
        return
    for m in sorted(models):
        if ask_yes_no(f"Pull Ollama model '{m}'?"):
            subprocess.run(["ollama", "pull", m], check=True)


def _object_info(class_type):
    """ComfyUI's own declared input schema for one node class_type,
    straight from GET /object_info/<class_type> -- the live,
    authoritative description of every input (including which are
    COMBO/dropdown-from-folder, with their current live option list),
    covering ANY node type ComfyUI currently has registered -- core or
    custom, including ones this codebase has never seen -- not a list
    this codebase maintains. A COMBO input's option list IS ComfyUI's
    real folder listing already (folder_paths.get_filename_list(),
    merged across any extra_model_paths.yaml redirect -- this project's
    own real setup redirects every model folder to a separate drive,
    confirmed live 2026-08-08), so no separate /models/<folder> call is
    needed either. Returns None if ComfyUI isn't reachable or doesn't
    know that class_type (e.g. a custom node not currently installed)
    -- callers must treat that as "can't determine", never as "not a
    model field"."""
    import urllib.parse
    import urllib.request
    import dream_step as ds
    comfyui_url = ds.load_config()["comfyui_url"]
    try:
        quoted = urllib.parse.quote(class_type, safe="")
        with urllib.request.urlopen(f"{comfyui_url}/object_info/{quoted}", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get(class_type)
    except Exception:
        return None


MODEL_FILE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


def _looks_like_model_options(options):
    """Whether a COMBO input's live option list actually looks like
    model filenames, not some other dropdown-from-folder-shaped enum --
    confirmed live 2026-08-08: KSamplerSelect's sampler_name field
    matches the "_name" naming convention AND is COMBO-typed (e.g.
    ["euler", "euler_ancestral", ...]) but isn't a file picker at all,
    so naming + COMBO-type together still aren't sufficient on their
    own. Empty option lists (a genuinely empty models/ folder -- exactly
    the state a brand-new install is in, which is the actual purpose of
    this whole check) can't be judged this way -- there's nothing to
    look at -- so an empty list is treated as "can't rule it out",
    erring toward over-reporting rather than silently missing a real
    requirement on a fresh install."""
    if not options:
        return True
    matches = sum(1 for o in options if isinstance(o, str) and Path(o).suffix.lower() in MODEL_FILE_EXTENSIONS)
    return matches / len(options) > 0.5


def _combo_options(field_spec):
    """Extracts the live option list from a COMBO input's declared spec
    -- confirmed live 2026-08-08 that this ComfyUI instance returns TWO
    different shapes for different nodes: the older bare-list style
    `[[opt1, opt2, ...], {meta}]` (UNETLoader/VAELoader/... -- the first
    element IS the option list), and a newer explicit type-string style
    `["COMBO", {"options": [...], meta}]` (LatentUpscaleModelLoader --
    options live inside the second element's "options" key). Missing
    either shape silently drops real model fields (confirmed live: this
    is exactly what made LatentUpscaleModelLoader's file vanish from the
    required list entirely before this fix). Returns None if field_spec
    isn't a COMBO input in either shape."""
    if not isinstance(field_spec, list) or not field_spec:
        return None
    first = field_spec[0]
    if isinstance(first, list):
        return first
    if first == "COMBO" and len(field_spec) > 1 and isinstance(field_spec[1], dict):
        options = field_spec[1].get("options")
        if isinstance(options, list):
            return options
    return None


def _confirm_model_candidates(candidates):
    """Cross-checks install_manifest.required_model_candidates_from_
    workflows()'s output (fields merely NAMED like a model picker)
    against ComfyUI's own /object_info to confirm each one is genuinely
    a COMBO-typed input whose live options actually look like model
    filenames (_looks_like_model_options()) -- filters out anything
    that just happens to match the naming convention without being a
    real file picker -- and determines presence directly from that same
    COMBO's live option list. Returns (confirmed_required, missing,
    resolved_class_types) -- the third value is how many of the
    candidates' DISTINCT class_types ComfyUI actually answered, out of
    how many were asked; a candidate whose class_type ComfyUI can't
    currently describe (unreachable, or a custom node not installed
    right now) is skipped from both required/missing lists, but still
    counted in the "asked" half of this ratio -- callers use
    resolved_class_types to distinguish "confirmed zero requirements"
    from "couldn't ask ComfyUI at all" (see check_models_status()'s
    last-known-good fallback)."""
    info_cache = {}
    required, missing = [], []
    for cand in candidates:
        class_type = cand["class_type"]
        if class_type not in info_cache:
            info_cache[class_type] = _object_info(class_type)
        info = info_cache[class_type]
        if not info:
            continue
        fields = {**(info.get("input", {}).get("required") or {}),
                  **(info.get("input", {}).get("optional") or {})}
        options = _combo_options(fields.get(cand["field"]))
        if options is None or not _looks_like_model_options(options):
            continue
        required.append(cand)
        if cand["filename"] not in options:
            missing.append(cand)
    asked = len(info_cache)
    resolved = sum(1 for v in info_cache.values() if v is not None)
    return required, missing, (resolved, asked)



def _huggingface_search_url(filename):
    """A real HuggingFace full-text search URL for a missing file with
    no known download source -- a human-review link, NOT an automated
    download: a filename match alone isn't enough to trust for a
    multi-GB weights file (wrong quantization, a tampered mirror, etc.),
    so this only ever opens a search page for a person to look at and
    decide, never fetches anything itself."""
    import urllib.parse
    return f"https://huggingface.co/search/full-text?q={urllib.parse.quote(filename)}&type=model"


# check_models_status()'s cache -- IN-MEMORY ONLY, never written to
# disk. Naturally empty on every fresh process start (satisfies
# "checked at least on load of the tool" with no extra bookkeeping) and
# can never survive a port to a different machine/OS the way the old
# on-disk model_check_cache.json did (that was the actual root cause of
# the 2026-08-15 Windows->Linux port bug: a stale cross-OS "0 missing"
# served from disk). Set to None by check_models_status() itself on any
# failed check, so a later success never serves pre-failure data.
_MODELS_STATUS_CACHE = None


def _workflow_hash():
    """Hash of every workflow_api_*.json's content -- part of the cache
    key for check_models_status() above. Changes the moment any workflow
    graph changes (a new/removed/renamed model file reference), which is
    exactly when a recheck is actually needed -- unrelated config/code
    changes don't invalidate it."""
    import hashlib
    h = hashlib.sha256()
    for wf_path in sorted(PIPELINE_DIR.glob("workflow_api_*.json")):
        h.update(wf_path.read_bytes())
    return h.hexdigest()


def check_models_status(comfyui_dir, force=False):
    """Which required model files ComfyUI doesn't currently know about --
    "required" is never a fixed hand-maintained list: candidates come
    from scanning the workflow_api_*.json graphs for fields that merely
    LOOK like a model picker by naming convention (install_manifest.
    required_model_candidates_from_workflows()), then each one is
    confirmed for real against ComfyUI's own live /object_info
    (_confirm_model_candidates()) -- which also gives presence directly
    from that same COMBO's live option list (already merged across any
    extra_model_paths.yaml redirect, no separate /models/<folder> call
    needed). This combination is what fixed three real problems this
    project hit in turn: false "missing" positives from files on a
    redirected drive a plain directory scan can't see, a manifest that
    could silently drift out of sync with what the graphs actually
    reference, and a hardcoded node-type list that would've silently
    missed any new loader a future workflow/custom node introduces.

    Cached IN-MEMORY ONLY, for this process's lifetime (see
    _MODELS_STATUS_CACHE below) -- per explicit direction 2026-08-15:
    caching is fine, but must always be re-checked at least once per
    tool launch (a fresh process starts with an empty cache, so this is
    automatic -- no on-disk file to go stale across a restart or a port
    to a different machine, which is exactly the bug an on-disk cache
    caused before), and a failed check must invalidate any cached good
    result rather than let a later success silently serve pre-failure
    data or a since-changed answer. `force` bypasses a valid cache hit
    on demand (the "Re-check" button).

    Returns (required_total, missing, meta). missing is a list of
    {filename, target_dir, size_gb, source, search_url, can_auto_download}
    -- target_dir/size_gb/source are None when the filename isn't in
    install_manifest.KNOWN_SOURCES (found missing by the live scan, but
    with nowhere known to download it from automatically); search_url is
    always present as a fallback for that case. can_auto_download is
    False whenever comfyui_dir isn't a real, existing LOCAL directory on
    THIS machine (see below -- the file might be genuinely missing on a
    remote ComfyUI, but this machine has nowhere to put a download for
    it). meta is {stale, checked_at, reason}: stale=True means ComfyUI
    couldn't be reached to actually confirm anything during a rebuild
    the workflow-hash change called for, so this is the LAST
    successfully-confirmed result, not a fresh one -- the cache file
    itself is deliberately left untouched in that case (never silently
    overwritten with an unverified/empty result), and the caller is
    expected to surface that plainly rather than treat it as equivalent
    to a real check.

    IMPORTANT (per explicit direction 2026-08-15): the actual "is this
    model file present" answer comes ENTIRELY from ComfyUI's own live
    /object_info API (_confirm_model_candidates/_object_info below,
    which already reads config.json's comfyui_url itself) -- it was
    NEVER a local directory scan, local or remote alike. A workflow
    graph fails identically either way if a model is genuinely missing
    on whatever machine actually runs ComfyUI, so this check must run
    (and be trusted) purely from API reachability, regardless of
    whether comfyui_dir is a valid, existing LOCAL path. comfyui_dir is
    used ONLY to decide can_auto_download above -- this machine can only
    offer to download a file into a models/ folder it can actually see."""
    comfyui_dir = Path(comfyui_dir) if comfyui_dir else None
    local_models_dir = comfyui_dir / "models" if comfyui_dir else None
    # Confirmed live 2026-08-15: Path.is_dir() is NOT safe to assume
    # always returns a plain bool -- a foreign-OS path (e.g. a leftover
    # Windows "C:\comfyui\..." value in comfyui_path) can make the
    # underlying os.stat() raise OSError instead of just reporting
    # "not found", especially over a network filesystem (confirmed on
    # a GVFS SMB mount: os.stat() raised EINVAL for a path containing
    # literal backslashes, which the SMB layer treats specially, rather
    # than a clean ENOENT). This crashed the entire --web startup, since
    # check_dependencies() calls this at boot. Any such error just means
    # "not usable," same as any other reason it isn't a real local dir.
    try:
        local_models_dir_usable = bool(local_models_dir and local_models_dir.is_dir())
    except OSError:
        local_models_dir_usable = False
    import dream_step as ds
    comfyui_url = ds.load_config().get("comfyui_url", "")

    global _MODELS_STATUS_CACHE
    cache = _MODELS_STATUS_CACHE
    if (not force and cache and cache["workflow_hash"] == _workflow_hash()
            and cache["comfyui_url"] == comfyui_url):
        return cache["required_total"], cache["missing"], {"stale": False, "checked_at": cache["checked_at"], "reason": None}

    # Fast upfront reachability gate (2026-08-16, per explicit direction:
    # "the model check can only happen once connected") -- BEFORE the
    # expensive per-class_type loop below, not after it. Confirmed real
    # bug: with ComfyUI genuinely down, _confirm_model_candidates() used
    # to still fire one urlopen(timeout=5) per DISTINCT class_type across
    # every workflow graph (candidates can easily span 10+ node types)
    # entirely sequentially, so a single unreachable ComfyUI could hang
    # this call for a minute or more -- check_dependencies() calls this
    # at boot, so that hang blocked the whole web UI's startup dependency
    # check. One cheap probe here catches the down case in ~3s instead.
    import urllib.request
    try:
        with urllib.request.urlopen(f"{comfyui_url}/queue", timeout=3):
            pass
    except Exception:
        return 0, [], {"stale": True, "checked_at": None,
                        "reason": f"Could not reach ComfyUI at {comfyui_url!r} to check required model files."}

    candidates = install_manifest.required_model_candidates_from_workflows()
    required, missing_candidates, (resolved, asked) = _confirm_model_candidates(candidates)

    if candidates and asked and resolved == 0:
        # ComfyUI couldn't answer a single class_type -- unreachable.
        # A failed check invalidates any existing cache outright (per
        # explicit direction 2026-08-15) -- never let a LATER success
        # serve a result that predates this failure, and never serve
        # this failure's absence-of-data as if it were a cache hit
        # either. The next successful check starts clean.
        _MODELS_STATUS_CACHE = None
        return 0, [], {"stale": True, "checked_at": None,
                        "reason": f"Could not reach ComfyUI at {comfyui_url!r} to check required model files."}

    import time
    missing = []
    for cand in missing_candidates:
        source_entry = install_manifest.KNOWN_SOURCES.get(cand["filename"])
        missing.append({
            "filename": cand["filename"],
            "target_dir": source_entry.get("target_dir") if source_entry else None,
            "size_gb": source_entry.get("size_gb") if source_entry else None,
            "source": source_entry.get("source") if source_entry else None,
            "search_url": _huggingface_search_url(cand["filename"]),
            "can_auto_download": local_models_dir_usable,
        })
    checked_at = time.time()
    _MODELS_STATUS_CACHE = {
        "workflow_hash": _workflow_hash(), "comfyui_url": comfyui_url,
        "required_total": len(required), "missing": missing, "checked_at": checked_at,
    }
    return len(required), missing, {"stale": False, "checked_at": checked_at, "reason": None}


def missing_models(comfyui_dir):
    """Thin compat wrapper around check_models_status() for callers that
    only need the missing list, not the required-total count."""
    return check_models_status(comfyui_dir)[1]


def pull_comfyui_models_noninteractive(comfyui_dir):
    """Non-interactive counterpart to pull_comfyui_models() -- downloads
    every missing manifest file without prompting, used by web_ui.py's
    "Download missing models" background job (which can't prompt for
    input, same reasoning as install_comfyui_noninteractive()). Raises if
    comfyui_dir doesn't look like a real ComfyUI install; otherwise keeps
    going past individual file failures (one bad URL/network blip
    shouldn't abort the other 16 files) and raises at the end if anything
    failed, so the job status still reports failure."""
    comfyui_dir = Path(comfyui_dir)
    models_dir = comfyui_dir / "models"
    if not models_dir.is_dir():
        raise RuntimeError(f"{models_dir} doesn't exist -- comfyui_dir doesn't look like a ComfyUI install")

    to_fetch = missing_models(comfyui_dir)
    total = len(to_fetch)
    if not total:
        print("All model files already present -- nothing to download.")
        return
    failed = []
    for i, entry in enumerate(to_fetch, 1):
        if not entry.get("source"):
            print(f"[{i}/{total}] {entry['filename']} -- SKIPPED: no known download source. "
                  f"Search: {entry.get('search_url', '(none)')}")
            failed.append(entry["filename"])
            continue
        target_dir = models_dir / entry["target_dir"]
        target_path = target_dir / entry["filename"]
        size = f"{entry['size_gb']} GB" if entry.get("size_gb") else "size unknown"
        target_dir.mkdir(parents=True, exist_ok=True)
        url = install_manifest.resolve_url(entry["source"])
        print(f"[{i}/{total}] downloading {entry['filename']} ({size}) from {url} ...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"[{i}/{total}] {entry['filename']} -- done.")
        except Exception as e:
            print(f"[{i}/{total}] {entry['filename']} -- FAILED: {e}")
            target_path.unlink(missing_ok=True)
            failed.append(entry["filename"])
    if failed:
        raise RuntimeError(f"{len(failed)} of {total} model file(s) failed to download: {', '.join(failed)}")


def pull_comfyui_models(comfyui_dir):
    print("\n=== ComfyUI models ===")
    if comfyui_dir is None:
        existing = input("Path to your ComfyUI install (for model downloads, blank to skip entirely): ").strip()
        if not existing:
            print("Skipped.")
            return
        comfyui_dir = Path(existing)

    models_dir = comfyui_dir / "models"
    if not models_dir.is_dir():
        print(f"WARNING: {models_dir} doesn't exist -- skipping model downloads.")
        return

    to_fetch = missing_models(comfyui_dir)
    total = len(to_fetch)
    if not total:
        print("All required model files already present -- nothing to download.")
        return
    for i, entry in enumerate(to_fetch, 1):
        if not entry.get("source"):
            print(f"[{i}/{total}] {entry['filename']} -- no known download source. "
                  f"Search: {entry.get('search_url', '(none)')}")
            continue
        target_dir = models_dir / entry["target_dir"]
        target_path = target_dir / entry["filename"]
        size = f"{entry['size_gb']} GB" if entry.get("size_gb") else "size unknown"
        if not ask_yes_no(f"[{i}/{total}] Download {entry['filename']} ({size}) "
                           f"to models/{entry['target_dir']}/?"):
            print("  skipped.")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        url = install_manifest.resolve_url(entry["source"])
        print(f"  downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, target_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            target_path.unlink(missing_ok=True)
            continue
        print("  done.")


def main():
    print("Dream Pipeline setup")
    print("Scope: Windows/Linux, NVIDIA GPUs. Every step below is optional -- say no to "
          "anything you already have.\n")

    install_pip_requirements()
    install_ollama()
    comfyui_dir = install_comfyui()
    pull_ollama_models()
    pull_comfyui_models(comfyui_dir)

    print("\n=== Setup finished ===")
    vpy = venv_python_path()
    if vpy.exists():
        print(f"Run '{vpy} dream_step.py --check-deps' to confirm everything the pipeline "
              "needs is actually in place -- use this venv python, not the system one, for "
              "every pipeline command from now on (web_ui.py, dream_step.py, generate_dream.py, ...).")
    else:
        print("Python packages weren't installed into a virtual environment (see above) -- "
              "fix that first, then re-run this installer.")


if __name__ == "__main__":
    main()
