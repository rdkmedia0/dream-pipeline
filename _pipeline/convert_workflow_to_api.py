"""
Convert a ComfyUI UI-graph workflow export (nodes/links) into API-prompt format
(the format POSTed to /prompt as the "prompt" field).

Handles:
- widget-input value resolution (link overrides widgets_values, else takes the
  next positional value from widgets_values, in the order widget-inputs appear)
- bypassed nodes (mode == 4): recursively resolved by following the bypassed
  node's own input at the same slot index, since every bypass node in this
  workflow is a single-in/single-out pass-through
- UI-only node types with no backend class (e.g. MarkdownNote) are dropped
"""
import json
import sys

UI_ONLY_TYPES = {"MarkdownNote", "Note"}
# Single-input/single-output passthrough node types with no backend class --
# resolved through exactly like a bypassed node, not just dropped.
PASSTHROUGH_TYPES = {"Reroute"}


def flatten_subgraphs(workflow):
    """Inline any subgraph-instance nodes (ComfyUI's newer 'group node' feature)
    into a single flat nodes/links list, resolving the -10/-20 boundary
    sentinels ComfyUI uses for subgraph input/output ports. Only handles one
    level of nesting, which is all that's needed here."""
    subgraphs = {sg["id"]: sg for sg in workflow.get("definitions", {}).get("subgraphs", [])}
    if not subgraphs:
        return workflow["nodes"], workflow["links"]

    nodes = list(workflow["nodes"])
    links = list(workflow["links"])
    changed = True
    while changed:
        changed = False
        instance_nodes = [n for n in nodes if n["type"] in subgraphs]
        for inst in instance_nodes:
            sg = subgraphs[inst["type"]]
            nodes = [n for n in nodes if n["id"] != inst["id"]]
            nodes.extend(sg["nodes"])

            new_links = []
            for l in links:
                origin_id, target_id = l[1], l[3]
                # Any link that used the subgraph instance as its origin (i.e.
                # something outside consumed the subgraph's output) gets
                # rewired to the subgraph's real internal source node.
                if origin_id == inst["id"]:
                    out_link_id = sg["outputs"][l[2]]["linkIds"][0]
                    inner = next(x for x in sg["links"] if x["id"] == out_link_id)
                    l = [l[0], inner["origin_id"], inner["origin_slot"], l[3], l[4], l[5]]
                new_links.append(l)
            links = new_links

            # Subgraph-internal links: boundary ones (origin -10 / target -20)
            # are dropped -- input boundaries have nothing feeding them from
            # outside in every workflow we use this on (the instance node
            # itself has no connected inputs), so the target just keeps its
            # own internal default widget value; output boundaries were
            # already spliced into the outer link above.
            for l in sg["links"]:
                if l["origin_id"] == -10 or l["target_id"] == -20:
                    continue
                links.append([l["id"], l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l["type"]])
            changed = True
    return nodes, links


def convert(workflow):
    flat_nodes, flat_links = flatten_subgraphs(workflow)
    nodes = {n["id"]: n for n in flat_nodes}
    links = {l[0]: {"origin_id": l[1], "origin_slot": l[2], "target_id": l[3], "target_slot": l[4], "type": l[5]} for l in flat_links}
    bypass_ids = {n["id"] for n in flat_nodes if n.get("mode") == 4}
    passthrough_ids = {n["id"] for n in flat_nodes if n["type"] in PASSTHROUGH_TYPES}
    skip_resolution_ids = bypass_ids | passthrough_ids

    def resolve(link_id, _seen=None):
        if _seen is None:
            _seen = set()
        if link_id in _seen:
            raise RuntimeError(f"cycle resolving link {link_id}")
        _seen.add(link_id)
        link = links[link_id]
        origin = nodes[link["origin_id"]]
        if origin["id"] in skip_resolution_ids:
            slot = link["origin_slot"]
            inputs = origin.get("inputs", [])
            if slot >= len(inputs):
                raise RuntimeError(f"passthrough node {origin['id']} has no input at slot {slot}")
            inp = inputs[slot]
            if inp.get("link") is None:
                raise RuntimeError(f"passthrough node {origin['id']} input slot {slot} is unconnected")
            return resolve(inp["link"], _seen)
        return [str(origin["id"]), link["origin_slot"]]

    prompt = {}
    skipped = []
    for node in flat_nodes:
        if node["id"] in skip_resolution_ids:
            continue
        node_type = node["type"]
        if node_type in UI_ONLY_TYPES:
            skipped.append((node["id"], node_type))
            continue

        widgets_values = node.get("widgets_values") or []
        widget_idx = 0
        inputs = {}
        for inp in node.get("inputs", []):
            name = inp["name"]
            has_widget = "widget" in inp
            link_id = inp.get("link")
            # A link pointing at a dropped subgraph input-boundary link (see
            # flatten_subgraphs) is treated as if unconnected.
            if link_id is not None and link_id not in links:
                link_id = None
            if has_widget:
                if link_id is not None:
                    inputs[name] = resolve(link_id)
                else:
                    if widget_idx >= len(widgets_values):
                        raise RuntimeError(f"node {node['id']} ({node_type}) missing widgets_values[{widget_idx}] for '{name}'")
                    inputs[name] = widgets_values[widget_idx]
                widget_idx += 1
            else:
                if link_id is not None:
                    inputs[name] = resolve(link_id)
                # else: optional unconnected input, omit

        prompt[str(node["id"])] = {
            "class_type": node_type,
            "_meta": {"title": node.get("title", node_type)},
            "inputs": inputs,
        }

    return prompt, skipped, bypass_ids


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    workflow = json.load(open(src, encoding="utf-8"))
    prompt, skipped, bypass_ids = convert(workflow)
    json.dump(prompt, open(dst, "w", encoding="utf-8"), indent=1)
    print(f"wrote {len(prompt)} nodes to {dst}")
    print(f"skipped UI-only nodes: {skipped}")
    print(f"bypassed nodes excluded: {sorted(bypass_ids)}")
