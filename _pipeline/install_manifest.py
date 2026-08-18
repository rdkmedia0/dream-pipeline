"""
Known download sources for Dream Pipeline's ComfyUI model files
(sourced directly against HuggingFace's/Civitai's own APIs, not
guessed from filenames) -- a lookup table, NOT the definitive
list of what's actually required.

This list is not hand-maintained here (a fixed list would drift the
moment a workflow_api_*.json graph changes -- a swapped checkpoint, an
added/removed LoRA -- with nothing forcing this file to be updated to
match). required_models_from_workflows() below derives the real,
current requirement list by reading the workflow JSON graphs
themselves every time, which can never go stale relative to what those
graphs actually reference. KNOWN_SOURCES here only answers "if this
filename turns out to be missing, where would I download it from" --
a filename the live scan finds that isn't in KNOWN_SOURCES is reported
as missing with no known source (see setup_installer.py), not silently
skipped or invented.

Files are NOT bundled/redistributed here -- several are individually
20-40GB -- this only stores where to fetch each one from on demand.
"""
import json
import re
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

# Field names matching this pattern are, by near-universal convention
# across ComfyUI core AND third-party custom nodes, "pick a file from a
# models/ subfolder" inputs (ckpt_name, vae_name, unet_name, lora_name,
# clip_name/clip_name1/clip_name2, model_name, control_net_name, ...).
# The trailing \d* handles multi-slot loaders like DualCLIPLoader's
# clip_name1/clip_name2 -- a plain endswith("_name") would silently
# miss those entirely (a real, dangerous gap: it would mean
# DualCLIPLoader's text-encoder files never get flagged as required
# at all, not even reported missing).
#
# This is a NAMING CONVENTION, not a list of specific node types or
# files -- it covers any loader node that follows it, including ones
# this codebase has never seen (a new custom node, a future workflow),
# with no new entry needed here every time. The convention alone isn't
# proof by itself: KSamplerSelect's sampler_name field matches this
# pattern and IS COMBO-typed, but lists sampler algorithm names, not
# files -- neither the naming convention nor COMBO-type alone is
# sufficient; setup_installer.py's
# check_models_status() adds a third check, that the live option list
# actually looks like filenames, before trusting a candidate).
MODEL_FIELD_PATTERN = re.compile(r"_name\d*$")


def required_model_candidates_from_workflows(pipeline_dir=None):
    """Every {class_type, field, filename} that MIGHT be a required
    model file, scanned live from the workflow_api_*.json graphs --
    offline, no ComfyUI needed. Deliberately a superset (candidates, not
    confirmed requirements) -- setup_installer.check_models_status()
    cross-checks each one against ComfyUI's own /object_info to confirm
    it's really a model-file field before trusting it. Handles
    rgthree's Power Lora Loader specially -- its LoRA slots are
    dynamically-named dict-valued inputs (e.g. "lora_1": {"on": true,
    "lora": "name.safetensors", ...}), not a fixed field name matching
    MODEL_FIELD_PATTERN."""
    pipeline_dir = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    seen = set()
    candidates = []
    for wf_path in sorted(pipeline_dir.glob("workflow_api_*.json")):
        try:
            data = json.loads(wf_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in data.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            inputs = node.get("inputs") or {}
            if not class_type:
                continue
            if "Power Lora Loader" in class_type:
                for value in inputs.values():
                    if isinstance(value, dict) and isinstance(value.get("lora"), str) and value["lora"]:
                        key = (class_type, "lora", value["lora"])
                        if key not in seen:
                            seen.add(key)
                            candidates.append({"class_type": class_type, "field": "lora", "filename": value["lora"]})
                continue
            for field, value in inputs.items():
                if MODEL_FIELD_PATTERN.search(field) and isinstance(value, str) and value:
                    key = (class_type, field, value)
                    if key not in seen:
                        seen.add(key)
                        candidates.append({"class_type": class_type, "field": field, "filename": value})
    return candidates


def resolve_url(source):
    """Turn one KNOWN_SOURCES entry's `source` dict into a real download URL."""
    if source["type"] == "huggingface":
        return f"https://huggingface.co/{source['repo']}/resolve/main/{source['path']}"
    if source["type"] == "civitai":
        return source["url"]
    raise ValueError(f"unknown source type: {source['type']!r}")


# Kept as MODEL_MANIFEST (rather than renamed) for backwards compat with
# any existing callers, but its "target_dir"/"used_by" fields are no
# longer treated as authoritative -- only "filename" and "source" are
# actually used now (via KNOWN_SOURCES below), everything else is
# historical annotation from when this list was written.
MODEL_MANIFEST = [
    # --- LTX-2.3 (fp8_t2v / i2v / fml2v workflows) ---
    {
        "filename": "ltx-2-3-22b-dev_transformer_only_fp8_input_scaled.safetensors",
        "target_dir": "diffusion_models", "size_gb": 25, "used_by": ["fp8_t2v", "i2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy",
                   "path": "diffusion_models/ltx-2-3-22b-dev_transformer_only_fp8_input_scaled.safetensors"},
    },
    {
        "filename": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
        "target_dir": "diffusion_models", "size_gb": 23, "used_by": ["fml2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy",
                   "path": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"},
    },
    {
        "filename": "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
        "target_dir": "loras", "size_gb": None, "used_by": ["fp8_t2v", "i2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy",
                   "path": "loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"},
    },
    {
        "filename": "ltx-2.3_text_projection_bf16.safetensors",
        "target_dir": "text_encoders", "size_gb": None, "used_by": ["fp8_t2v", "i2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy",
                   "path": "text_encoders/ltx-2.3_text_projection_bf16.safetensors"},
    },
    {
        "filename": "LTX23_video_vae_bf16.safetensors",
        "target_dir": "vae", "size_gb": None, "used_by": ["fp8_t2v", "i2v", "fml2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy", "path": "vae/LTX23_video_vae_bf16.safetensors"},
    },
    {
        "filename": "LTX23_audio_vae_bf16.safetensors",
        "target_dir": "vae", "size_gb": None, "used_by": ["fp8_t2v", "i2v", "fml2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy", "path": "vae/LTX23_audio_vae_bf16.safetensors"},
    },
    {
        "filename": "taeltx2_3.safetensors",
        "target_dir": "vae", "size_gb": None, "used_by": ["fp8_t2v", "fml2v"],
        "source": {"type": "huggingface", "repo": "Kijai/LTX2.3_comfy", "path": "vae/taeltx2_3.safetensors"},
    },
    {
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "target_dir": "latent_upscale_models", "size_gb": 1, "used_by": ["fp8_t2v", "i2v", "fml2v"],
        "source": {"type": "huggingface", "repo": "Lightricks/LTX-2.3",
                   "path": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
    },
    {
        "filename": "ltx-2-19b-ic-lora-detailer.safetensors",
        "target_dir": "loras", "size_gb": 2.6, "used_by": [],
        "source": {"type": "huggingface", "repo": "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
                   "path": "ltx-2-19b-ic-lora-detailer.safetensors"},
    },
    {
        "filename": "gemma_3_12B_it_fp4_mixed.safetensors",
        "target_dir": "text_encoders", "size_gb": 9.45, "used_by": ["fp8_t2v", "i2v"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/ltx-2",
                   "path": "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"},
    },
    {
        "filename": "gemma_3_12B_it_fp8_scaled.safetensors",
        "target_dir": "text_encoders", "size_gb": 13.2, "used_by": ["fml2v"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/ltx-2",
                   "path": "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"},
    },
    {
        "filename": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
        "target_dir": "loras", "size_gb": 0.6, "used_by": ["i2v"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/ltx-2",
                   "path": "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"},
    },
    {
        "filename": "LTX-2.3_Cinematic hardcut.safetensors",
        "target_dir": "loras", "size_gb": 0.63, "used_by": ["fp8_t2v"],
        # HF mirror of the same file (confirmed byte-identical size to the
        # Civitai original) -- used instead of Civitai for a consistent
        # source type across the whole manifest. The repo's own filename
        # differs (ltx-2.3_-cinematic-hardcut.safetensors); the installer
        # always saves using this entry's "filename", not the source's.
        "source": {"type": "huggingface", "repo": "Muapi/ltx-2.3_-cinematic-hardcut",
                   "path": "ltx-2.3_-cinematic-hardcut.safetensors"},
    },
    # --- Flux2 (t2i_flux2 keyframe-generation workflow) ---
    {
        "filename": "flux2_dev_fp8mixed.safetensors",
        "target_dir": "diffusion_models", "size_gb": 35.5, "used_by": ["t2i_i2i"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/flux2-dev",
                   "path": "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"},
    },
    {
        "filename": "mistral_3_small_flux2_bf16.safetensors",
        "target_dir": "text_encoders", "size_gb": None, "used_by": ["t2i_i2i"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/flux2-dev",
                   "path": "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors"},
    },
    {
        "filename": "Flux_2-Turbo-LoRA_comfyui.safetensors",
        "target_dir": "loras", "size_gb": None, "used_by": ["t2i_i2i"],
        "source": {"type": "huggingface", "repo": "Comfy-Org/flux2-dev",
                   "path": "split_files/loras/Flux_2-Turbo-LoRA_comfyui.safetensors"},
    },
    {
        "filename": "full_encoder_small_decoder.safetensors",
        "target_dir": "vae", "size_gb": None, "used_by": ["t2i_i2i"],
        "source": {"type": "huggingface", "repo": "black-forest-labs/FLUX.2-small-decoder",
                   "path": "full_encoder_small_decoder.safetensors"},
    },
]

# The actual lookup used by setup_installer.py: filename -> full entry
# (source/size_gb). Built from MODEL_MANIFEST purely as a source-of-
# download-URLs table now -- required_models_from_workflows() above is
# what decides whether a given filename is actually needed.
KNOWN_SOURCES = {entry["filename"]: entry for entry in MODEL_MANIFEST}
