#!/usr/bin/env python3
"""Static guard for Comfy/Swarm workflow topology fail-loud behavior.

The current executor lowers typed graphs into one flat JobParams object. It must
reject multi-sampler or multi-output Comfy graphs until the runtime owns real
per-node tensor values; otherwise first-writer-wins flattening silently runs the
wrong graph.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "serenitymojo/serve/workflow_graph.mojo"


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = [
        "def _workflow_is_denoise_node(type_id: String) -> Bool:",
        'type_id == "KSampler"',
        'or type_id == "KSamplerAdvanced"',
        'or type_id == "LanPaint_KSampler"',
        'or type_id == "LanPaint_KSamplerAdvanced"',
        'or type_id == "SamplerCustom"',
        'or type_id == "SamplerCustomAdvanced"',
        'or type_id == "LanPaint_SamplerCustomAdvanced"',
        "def _workflow_reject_multi_output_topology(nodes_json: JSONValue) raises:",
        "Multi-denoise or multi-SaveImage Comfy graphs need real per-node tensor",
        "denoise_count > 1",
        "save_count > 1",
        "workflow graph has multiple sampler/output branches",
        "workflow graph has multiple SaveImage outputs",
        "workflow graph body needs edges for typed execution",
        "_workflow_reject_multi_output_topology(nodes_json)",
    ]
    missing = [needle for needle in required if needle not in text]
    if missing:
        print("[workflow-topology] FAIL")
        for needle in missing:
            print(f"  missing: {needle}")
        return 1

    forbidden = [
        "field_only_graph_adapter",
    ]
    present = [needle for needle in forbidden if needle in text]
    if present:
        print("[workflow-topology] FAIL")
        for needle in present:
            print(f"  forbidden fallback still present: {needle}")
        return 1

    call_pos = text.find("_workflow_reject_multi_output_topology(nodes_json)")
    setnode_pos = text.find("var setnode_names = List[String]()")
    loop_pos = text.find("while remaining > 0:")
    if call_pos < 0 or setnode_pos < 0 or loop_pos < 0:
        print("[workflow-topology] FAIL missing expected execution landmarks")
        return 1
    if not (call_pos < setnode_pos < loop_pos):
        print("[workflow-topology] FAIL topology guard must run before execution")
        return 1

    print("[workflow-topology] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
