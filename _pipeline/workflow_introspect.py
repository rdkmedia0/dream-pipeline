"""
Structural introspection for user-supplied workflow_api_*.json graphs --
figures out which node/field holds the positive prompt, the negative
prompt, the image input(s), and the seed(s), using ComfyUI's own live
GET /object_info/<class_type> (the same technique install_manifest.py /
setup_installer.py use for model-file detection), extended with a
backward link-walk through the graph.

This does NOT try to be certain on its own -- prompt wiring detection
is inherently less reliable than model-file detection (it requires
tracing CONDITIONING links back through however many pass-through
nodes a particular graph happens to use, not just reading one node's
declared schema). When detection can't converge on exactly one
candidate per role, callers must treat the result as unconfirmed and
fall back to a real test render + human confirmation (see web_ui.py's
workflow-file Settings flow) rather than guessing.
"""
import json
from pathlib import Path

import setup_installer

PIPELINE_DIR = Path(__file__).resolve().parent

# Order matters only for readability of "which types matched" messages --
# detection itself checks all three independently.
TYPE_SUBSTRINGS = ("t2v", "i2v", "fml")


def categorize_workflow_files(pipeline_dir=None):
    """Buckets every workflow_api_*.json in pipeline_dir by type, using a
    case-insensitive substring match on the filename for t2v/i2v/fml --
    the only way to know a graph's behavioral category, since that can't
    be reliably read off the graph's structure itself. Returns
    {"t2v": [filenames], "i2v": [...], "fml": [...],
     "unrecognized": [...], "ambiguous": [...]} -- a file matching none
    of the three substrings lands in "unrecognized" (needs a rename to be
    usable at all); a file matching more than one lands in "ambiguous"
    (its filename doesn't unambiguously declare a single category)."""
    pipeline_dir = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    buckets = {"t2v": [], "i2v": [], "fml": [], "unrecognized": [], "ambiguous": []}
    for path in sorted(pipeline_dir.glob("workflow_api_*.json")):
        name_lower = path.name.lower()
        matches = [t for t in TYPE_SUBSTRINGS if t in name_lower]
        if len(matches) == 1:
            buckets[matches[0]].append(path.name)
        elif len(matches) == 0:
            buckets["unrecognized"].append(path.name)
        else:
            buckets["ambiguous"].append(path.name)
    return buckets


def _load_graph(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merged_fields(info):
    if not info:
        return {}
    inp = info.get("input") or {}
    return {**(inp.get("required") or {}), **(inp.get("optional") or {})}


def _field_type(field_spec):
    """First element of a field's declared spec is always its type name
    (e.g. "STRING", "INT", "CONDITIONING", or a bare options list for
    COMBO) -- returns that type name, or None if unrecognized shape."""
    if isinstance(field_spec, list) and field_spec:
        first = field_spec[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list):
            return "COMBO"
    return None


def _is_link(value):
    """ComfyUI API-format graphs represent a wired-up input as
    [source_node_id, output_index]; a literal (user-set) value is
    anything else -- a string, number, bool, etc."""
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[1], int) and not isinstance(value[1], bool))


class _ObjectInfoCache:
    def __init__(self, comfyui_url):
        self.comfyui_url = comfyui_url
        self._cache = {}

    def get(self, class_type):
        if class_type not in self._cache:
            self._cache[class_type] = setup_installer._object_info(class_type)
        return self._cache[class_type]


def _walk_back_to_literal(link, graph, cache, wanted_types, visited=None):
    """Follows a [node_id, output_index] link backward through the graph
    until it reaches a node with a literal (non-link) input whose
    declared type is in wanted_types -- e.g. a CLIPTextEncode fed by a
    PrimitiveStringMultiline: the sampler's "positive" link points at
    CLIPTextEncode, whose own "text" input is itself a link to the
    primitive node that actually holds the literal string. Stops at the
    first literal match; returns (node_id, field_name) or None if the
    chain dead-ends (missing /object_info, no matching field, or a
    cycle)."""
    visited = visited or set()
    if not isinstance(link, list) or len(link) != 2:
        return None
    node_id = str(link[0])
    if node_id in visited:
        return None
    visited = visited | {node_id}
    node = graph.get(node_id)
    if not isinstance(node, dict):
        return None
    class_type = node.get("class_type")
    fields = _merged_fields(cache.get(class_type))
    inputs = node.get("inputs") or {}
    matching_fields = [f for f, spec in fields.items() if _field_type(spec) in wanted_types]
    literal_hits = [f for f in matching_fields if f in inputs and not _is_link(inputs[f])]
    if literal_hits:
        return node_id, literal_hits[0]
    for f in matching_fields:
        val = inputs.get(f)
        if _is_link(val):
            hop = _walk_back_to_literal(val, graph, cache, wanted_types, visited)
            if hop:
                return hop
    return None


def _find_sampler_nodes(graph, cache):
    """Every node instance whose /object_info schema declares BOTH a
    "positive" and "negative" CONDITIONING input -- the near-universal
    KSampler-family convention, independent of which specific sampler
    class_type a graph actually uses."""
    sampler_nodes = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        fields = _merged_fields(cache.get(node.get("class_type")))
        if (_field_type(fields.get("positive")) == "CONDITIONING"
                and _field_type(fields.get("negative")) == "CONDITIONING"):
            sampler_nodes.append(node_id)
    return sampler_nodes


def detect_prompt_nodes(path_or_graph, comfyui_url, cache=None):
    """Detects the (node_id, field) that hold the positive prompt and
    negative prompt for an arbitrary workflow graph. Looks for node
    instances whose /object_info schema declares BOTH a "positive" and a
    "negative" CONDITIONING input -- the near-universal KSampler-family
    convention across ComfyUI core and custom sampler nodes, independent
    of which specific sampler class_type a given graph uses -- then walks
    each link backward to the literal STRING field that actually needs
    the prompt text written into it.

    cache: an _ObjectInfoCache to reuse across multiple detect_* calls
    for the same graph -- each unique class_type otherwise costs a real
    /object_info HTTP round trip (~1-2s), and a graph commonly repeats
    the same handful of class_types across many nodes; callers doing
    more than one detect_* call on the same graph (see
    detect_workflow_wiring) should always share one cache rather than
    let each call build its own from scratch.

    Returns {"confident": bool, "positive": {"node":.., "field":..} or
    None, "negative": {...} or None, "explanation": str}. Multiple
    sampler-like nodes that all agree on the same source are fine
    (common with two-stage samplers sharing conditioning); disagreement
    or zero matches makes the result unconfident, with the candidates
    found listed in "explanation" so a human can see why."""
    graph = path_or_graph if isinstance(path_or_graph, dict) else _load_graph(path_or_graph)
    cache = cache or _ObjectInfoCache(comfyui_url)
    sampler_nodes = _find_sampler_nodes(graph, cache)

    if not sampler_nodes:
        return {"confident": False, "positive": None, "negative": None,
                "explanation": "No node was found with both a \"positive\" and \"negative\" "
                                "CONDITIONING input -- couldn't identify the sampler node(s) "
                                "this graph uses."}

    positive_hits, negative_hits = set(), set()
    for node_id in sampler_nodes:
        inputs = graph[node_id].get("inputs") or {}
        for role, hits in (("positive", positive_hits), ("negative", negative_hits)):
            link = inputs.get(role)
            hop = _walk_back_to_literal(link, graph, cache, {"STRING"}) if link else None
            if hop:
                hits.add(hop)

    explanation_parts = []
    result = {"positive": None, "negative": None}
    for role, hits in (("positive", positive_hits), ("negative", negative_hits)):
        if len(hits) == 1:
            node_id, field = next(iter(hits))
            result[role] = {"node": node_id, "field": field}
        elif len(hits) == 0:
            explanation_parts.append(f"couldn't trace the {role} prompt back to a literal text field")
        else:
            explanation_parts.append(
                f"found {len(hits)} conflicting candidates for the {role} prompt: "
                + ", ".join(f"node {n} field {f!r}" for n, f in sorted(hits)))

    confident = result["positive"] is not None and result["negative"] is not None
    explanation = "; ".join(explanation_parts) if explanation_parts else "detected unambiguously"
    return {"confident": confident, **result, "explanation": explanation}


def detect_image_nodes(path_or_graph, type_):
    """Detects LoadImage node(s) for a graph, given its declared type
    (t2v/i2v/fml). t2v expects none, i2v expects exactly one, fml expects
    exactly three -- assigned to first/middle/last by JSON key iteration
    order (the order the nodes appear as keys in the graph file; this is
    an assumption, not a guarantee -- ComfyUI doesn't promise export
    order matches visual/logical left-to-right order, so a wrong guess
    here is exactly what the test-render confirmation step exists to
    catch before anything is trusted). Returns {"confident": bool,
    "wiring": {...WORKFLOWS-shaped keys...} or None, "explanation": str}."""
    graph = path_or_graph if isinstance(path_or_graph, dict) else _load_graph(path_or_graph)
    load_image_nodes = [node_id for node_id, node in graph.items()
                         if isinstance(node, dict) and node.get("class_type") == "LoadImage"]

    expected = {"t2v": 0, "i2v": 1, "fml": 3}.get(type_)
    if expected is None:
        return {"confident": False, "wiring": None,
                "explanation": f"unknown type {type_!r}"}

    if len(load_image_nodes) != expected:
        return {"confident": False, "wiring": None,
                "explanation": f"type {type_!r} expects {expected} LoadImage node(s), "
                                f"found {len(load_image_nodes)}: {load_image_nodes}"}

    if type_ == "t2v":
        return {"confident": True, "wiring": {}, "explanation": "no image input needed"}
    if type_ == "i2v":
        return {"confident": True,
                "wiring": {"image_node": load_image_nodes[0], "image_field": "image"},
                "explanation": "detected unambiguously"}
    # fml
    first, middle, last = load_image_nodes
    return {"confident": True,
            "wiring": {"image_nodes": {"first": first, "middle": middle, "last": last},
                        "image_field": "image"},
            "explanation": "assigned first/middle/last by JSON key order -- verify with a test render"}


_SEED_FIELD_NAMES = ("noise_seed", "seed")


def detect_seed_nodes(path_or_graph, comfyui_url=None, cache=None):
    """Detects every node holding a literal noise seed -- a direct
    whole-graph scan for a literal (non-link) "noise_seed"/"seed" input,
    NOT tied to whatever detect_prompt_nodes() identified as a "sampler"
    (i.e. a node with both CONDITIONING "positive"/"negative" inputs).
    In the bundled fp8_t2v graph
    those two concepts are NOT the same node: fp8_t2v's actual seed
    holders are two RandomNoise nodes (114/115) feeding a
    SamplerCustomAdvanced node's "noise" input, while
    SamplerCustomAdvanced itself takes "guider" (built upstream by a
    CFGGuider from positive/negative) rather than positive/negative
    directly -- so a link-walk anchored on the CONDITIONING-based
    "sampler" heuristic never reaches the real seed nodes at all. A
    direct scan matches this project's own convention (see
    generate_dream.build_prompt: every workflow_cfg["seeds"] entry is a
    node ID whose inputs["noise_seed"] gets written directly, always a
    literal on that node itself, never reached via a link) and needs no
    /object_info calls. comfyui_url/cache are accepted
    but unused -- kept for a stable signature alongside the other
    detect_* functions."""
    graph = path_or_graph if isinstance(path_or_graph, dict) else _load_graph(path_or_graph)
    seeds = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for field in _SEED_FIELD_NAMES:
            if field in inputs and isinstance(inputs[field], int) and not isinstance(inputs[field], bool):
                seeds.append(node_id)
                break
    return seeds


def detect_workflow_wiring(path, type_, comfyui_url):
    """Ties detect_prompt_nodes/detect_image_nodes/detect_seed_nodes
    together into one WORKFLOWS-shaped candidate wiring config (see
    generate_dream.py's WORKFLOWS dict for the target shape), plus an
    overall confidence flag and human-readable explanation. Does NOT
    touch model-file detection -- install_manifest.py's existing glob
    already covers any workflow_api_*.json placed in this directory."""
    graph = _load_graph(path)
    cache = _ObjectInfoCache(comfyui_url)
    prompt_result = detect_prompt_nodes(graph, comfyui_url, cache=cache)
    image_result = detect_image_nodes(graph, type_)

    if not (prompt_result["confident"] and image_result["confident"]):
        parts = []
        if not prompt_result["confident"]:
            parts.append(f"prompt wiring: {prompt_result['explanation']}")
        if not image_result["confident"]:
            parts.append(f"image wiring: {image_result['explanation']}")
        return {"confident": False, "wiring": None, "explanation": "; ".join(parts)}

    seeds = detect_seed_nodes(graph)

    wiring = {
        "positive": prompt_result["positive"]["node"],
        "positive_field": prompt_result["positive"]["field"],
        "negative": prompt_result["negative"]["node"],
        "negative_field": prompt_result["negative"]["field"],
        "seeds": seeds,
        **image_result["wiring"],
    }
    return {"confident": True, "wiring": wiring,
            "explanation": image_result["explanation"]}
