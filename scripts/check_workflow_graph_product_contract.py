#!/usr/bin/env python3
"""Product-path smoke for the supported typed Comfy/Swarm workflow graph subset.

This is a development checker. It starts the compiled Mojo daemon, submits a
linked `workflow.nodes`/`workflow.edges` graph through `/v1/generate`, and
inspects the product artifact metadata. Runtime generation remains pure Mojo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON = REPO / "output/bin/serenity_daemon"
CHECKS_DIR = REPO / "output/checks"
SERENITYFLOW_WORKFLOWS = Path("/home/alex/serenityflow-v2/serenityflow/workflows")
IDEOGRAM4_BASIC_TXT2IMG = Path("/home/alex/Downloads/ideogram4_basic_txt2img_workflow_by_AI_Characters_v4.json")
GENPARAMS_KEY = "serenity.genparams.v1"
IDEOGRAM4_PROMPT = "a surreal streetwear collage poster with blue sky and large COMFY letters"

SERENITYFLOW_T2I_CASES: dict[str, dict[str, Any]] = {
    "zimage_t2i": {
        "template": SERENITYFLOW_WORKFLOWS / "zimage_t2i.json",
        "model": "z_image_turbo_bf16.safetensors",
        "prompt": "a stunning landscape photograph",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 8,
        "seed": 42,
        "cfg": 1.5,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "creativity": 1.0,
        "sigma_shift": 3,
        "workflow_node_count": 10,
        "workflow_edge_count": 10,
    },
    "qwen_image_t2i": {
        "template": SERENITYFLOW_WORKFLOWS / "qwen_image_t2i.json",
        "model": "qwen_image.safetensors",
        "prompt": "a beautiful landscape photograph",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 20,
        "seed": 42,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 1.0,
        "sigma_shift": 3,
        "workflow_node_count": 10,
        "workflow_edge_count": 10,
    },
    "klein9b_t2i": {
        "template": SERENITYFLOW_WORKFLOWS / "klein9b_t2i.json",
        "model": "flux2-klein-9b.safetensors",
        "prompt": "a beautiful landscape",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 35,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 1.0,
        "sigma_shift": 3,
        "workflow_node_count": 9,
        "workflow_edge_count": 9,
    },
    "klein4b_t2i": {
        "template": SERENITYFLOW_WORKFLOWS / "klein4b_t2i.json",
        "model": "flux2-klein-4b.safetensors",
        "prompt": "a beautiful landscape",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 35,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 1.0,
        "sigma_shift": 3,
        "workflow_node_count": 9,
        "workflow_edge_count": 9,
    },
    "flux2_dev_t2i": {
        "template": SERENITYFLOW_WORKFLOWS / "flux2_dev_t2i.json",
        "model": "flux2-dev.safetensors",
        "prompt": "A serene mountain lake with golden sunset light reflecting off the calm water, surrounded by snow-capped peaks",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 20,
        "seed": 42,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 1.0,
        "sigma_shift": 3,
        "workflow_node_count": 9,
        "workflow_edge_count": 9,
    },
}

SERENITYFLOW_EDIT_CASES: dict[str, dict[str, Any]] = {
    "klein9b_edit": {
        "template": SERENITYFLOW_WORKFLOWS / "klein9b_edit.json",
        "model": "flux2-klein-9b.safetensors",
        "prompt": "change the dress to blue",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 35,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "creativity": 1.0,
        "init_image": "input.png",
        "reference_image": "input.png",
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_node_count": 18,
        "workflow_edge_count": 21,
    },
    "klein4b_edit": {
        "template": SERENITYFLOW_WORKFLOWS / "klein4b_edit.json",
        "model": "flux2-klein-4b.safetensors",
        "prompt": "change the dress to blue",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 35,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "creativity": 1.0,
        "init_image": "input.png",
        "reference_image": "input.png",
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_node_count": 18,
        "workflow_edge_count": 21,
    },
}

SERENITYFLOW_QWEN_EDIT_CASES: dict[str, dict[str, Any]] = {
    "qwen_edit": {
        "template": SERENITYFLOW_WORKFLOWS / "qwen_edit.json",
        "model": "qwen_image_edit.safetensors",
        "prompt": "change the background to a beach",
        "negative": "",
        "steps": 20,
        "seed": 42,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 0.75,
        "sigma_shift": 3,
        "init_image": "input.png",
        "qwen_edit_conditioning_image": "input.png",
        "workflow_save_prefix": "qwen_edit",
        "workflow_node_count": 12,
        "workflow_edge_count": 14,
        "lora": [],
    },
    "qwen_edit_lora": {
        "template": SERENITYFLOW_WORKFLOWS / "qwen_edit_lora.json",
        "model": "qwen_image_edit.safetensors",
        "prompt": "change the background to a beach",
        "negative": "",
        "steps": 20,
        "seed": 42,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 0.75,
        "sigma_shift": 3,
        "init_image": "input.png",
        "qwen_edit_conditioning_image": "input.png",
        "workflow_save_prefix": "qwen_edit_lora",
        "workflow_node_count": 13,
        "workflow_edge_count": 15,
        "lora": [{"name": "lora.safetensors", "weight": 1.0}],
    },
}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, Any, str]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(text) if text else None, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        return int(exc.code), parsed, text
    except urllib.error.URLError as exc:
        return 0, None, str(exc)


def wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        status, data, text = http_json("GET", f"{base_url}/v1/health", timeout=2.0)
        last = text
        if status == 200 and isinstance(data, dict):
            return data
        time.sleep(0.1)
    raise RuntimeError(f"daemon did not become healthy: {last}")


def poll_job(base_url: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data, _ = http_json("GET", f"{base_url}/v1/job/{job_id}", timeout=5.0)
        if status == 200 and isinstance(data, dict):
            last = data
            if data.get("state") in {"done", "failed", "cancelled", "interrupted"}:
                return data
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {job_id}: {last}")


def read_png_text(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"not a PNG: {path}")
    out: dict[str, str] = {}
    idat_hash = hashlib.sha256()
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        typ = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if typ == b"tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            out[key.decode("latin1", errors="replace")] = value.decode("latin1", errors="replace")
        elif typ == b"IDAT":
            idat_hash.update(payload)
        elif typ == b"IEND":
            break
    out["_idat_sha256"] = idat_hash.hexdigest()
    return out


def linked_workflow_request() -> dict[str, Any]:
    # Deliberately shuffled node order: typed execution must follow links, not
    # array order or title heuristics.
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 5, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 5, "port": "negative"}},
                {"from": {"node": 4, "port": "LATENT"}, "to": {"node": 5, "port": "latent_image"}},
                {"from": {"node": 5, "port": "LATENT"}, "to": {"node": 6, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 6, "port": "vae"}},
                {"from": {"node": 6, "port": "IMAGE"}, "to": {"node": 7, "port": "images"}},
            ],
            "nodes": [
                {
                    "id": 5,
                    "type_id": "comfy/KSampler",
                    "title": "Sampler",
                    "fields": {
                        "steps": 7,
                        "seed": 12345,
                        "cfg": 3.5,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.75,
                    },
                },
                {"id": 7, "type_id": "comfy/SaveImage", "title": "Save", "fields": {"filename_prefix": "typed-graph"}},
                {"id": 6, "type_id": "comfy/VAEDecode", "title": "Decode", "fields": {}},
                {
                    "id": 2,
                    "type_id": "comfy/CLIPTextEncode",
                    "title": "Text node without negative title",
                    "fields": {"text": "linked negative prompt"},
                },
                {"id": 4, "type_id": "comfy/EmptyLatentImage", "title": "Latent", "fields": {"width": 640, "height": 512, "batch_size": 1}},
                {
                    "id": 3,
                    "type_id": "comfy/CLIPTextEncode",
                    "title": "Text node without positive title",
                    "fields": {"text": "linked positive prompt"},
                },
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "title": "Load Model", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def unsupported_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "nodes": [{"id": 1, "type_id": "comfy/ControlNetApply", "fields": {}}],
            "edges": [],
        }
    }


def wrong_type_workflow_request() -> dict[str, Any]:
    body = linked_workflow_request()
    edges = body["workflow"]["edges"]
    edges[3] = {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "positive"}}
    return body


def comfy_api_prompt_request() -> dict[str, Any]:
    # Native ComfyUI API prompt shape: node id keys, class_type, and inputs links
    # as [source_node_id, output_index]. This lowers into the typed executor.
    return {
        "workflow": {
            "prompt": {
                "5": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["3", 0],
                        "negative": ["2", 0],
                        "latent_image": ["4", 0],
                        "steps": 6,
                        "seed": 23456,
                        "cfg": 4.25,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.625,
                    },
                },
                "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "comfy-api"}},
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "api negative prompt"}},
                "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 704, "height": 512, "batch_size": 1}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "api positive prompt"}},
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "stub"}},
            }
        }
    }


def outpaint_threshold_comfy_api_prompt_request() -> dict[str, Any]:
    return {
        "workflow": {
            "prompt": {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "stub"}},
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 1], "text": "outpaint threshold negative prompt"},
                },
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 1], "text": "outpaint threshold positive prompt"},
                },
                "4": {"class_type": "LoadImage", "inputs": {"image": "/tmp/serenity_graph_init.png"}},
                "5": {
                    "class_type": "ImagePadForOutpaint",
                    "inputs": {
                        "image": ["4", 0],
                        "left": 16,
                        "top": 8,
                        "right": 16,
                        "bottom": 8,
                        "feathering": 0,
                    },
                },
                "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "ThresholdMask", "inputs": {"mask": ["5", 1], "value": 0.5}},
                "8": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["6", 0], "mask": ["7", 0]}},
                "9": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["3", 0],
                        "negative": ["2", 0],
                        "latent_image": ["8", 0],
                        "steps": 4,
                        "seed": 55678,
                        "cfg": 2.25,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.45,
                    },
                },
                "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
                "11": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["10", 0], "filename_prefix": "outpaint-threshold-graph"},
                },
            }
        }
    }


def inpaint_conditioning_comfy_api_prompt_request(noise_mask: bool = True) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "positive": ["3", 0],
        "negative": ["2", 0],
        "vae": ["1", 2],
        "pixels": ["4", 0],
        "mask": ["4", 1],
    }
    if not noise_mask:
        inputs["noise_mask"] = False
    prefix = "inpaint-conditioning-graph" if noise_mask else "inpaint-conditioning-no-noise-mask"
    seed = 66789 if noise_mask else 66790
    return {
        "workflow": {
            "prompt": {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "stub"}},
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 1], "text": "inpaint conditioning negative prompt"},
                },
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 1], "text": "inpaint conditioning positive prompt"},
                },
                "4": {"class_type": "LoadImage", "inputs": {"image": "/tmp/serenity_inpaint_conditioning.png"}},
                "5": {"class_type": "InpaintModelConditioning", "inputs": inputs},
                "6": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["5", 0],
                        "negative": ["5", 1],
                        "latent_image": ["5", 2],
                        "steps": 4,
                        "seed": seed,
                        "cfg": 2.75,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 0.6,
                    },
                },
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
            }
        }
    }


def inpaint_conditioning_missing_mask_comfy_api_prompt_request() -> dict[str, Any]:
    request = inpaint_conditioning_comfy_api_prompt_request()
    prompt = request["workflow"]["prompt"]
    del prompt["5"]["inputs"]["mask"]
    return request


def serenityflow_template_request(template: Path) -> dict[str, Any]:
    return {"workflow": json.loads(template.read_text(encoding="utf-8"))}


def ideogram4_visual_export_request() -> dict[str, Any]:
    return {
        "prompt": IDEOGRAM4_PROMPT,
        "seed": 424242,
        "workflow": json.loads(IDEOGRAM4_BASIC_TXT2IMG.read_text(encoding="utf-8")),
    }


def img2img_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 4, "port": "IMAGE"}, "to": {"node": 5, "port": "pixels"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 5, "port": "vae"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 6, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 6, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 6, "port": "negative"}},
                {"from": {"node": 5, "port": "LATENT"}, "to": {"node": 6, "port": "latent_image"}},
                {"from": {"node": 6, "port": "LATENT"}, "to": {"node": 7, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 7, "port": "vae"}},
                {"from": {"node": 7, "port": "IMAGE"}, "to": {"node": 8, "port": "images"}},
            ],
            "nodes": [
                {"id": 8, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "img2img-graph"}},
                {"id": 7, "type_id": "comfy/VAEDecode", "fields": {}},
                {
                    "id": 6,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 5,
                        "seed": 34567,
                        "cfg": 2.75,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.4,
                    },
                },
                {"id": 5, "type_id": "comfy/VAEEncode", "fields": {}},
                {"id": 4, "type_id": "comfy/LoadImage", "fields": {"image": "/tmp/serenity_graph_init.png"}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "img2img negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "img2img positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def lora_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 5, "port": "clip"}},
                {"from": {"node": 5, "port": "MODEL"}, "to": {"node": 6, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 6, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 6, "port": "negative"}},
                {"from": {"node": 4, "port": "LATENT"}, "to": {"node": 6, "port": "latent_image"}},
                {"from": {"node": 6, "port": "LATENT"}, "to": {"node": 8, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 8, "port": "vae"}},
                {"from": {"node": 8, "port": "IMAGE"}, "to": {"node": 9, "port": "images"}},
            ],
            "nodes": [
                {"id": 9, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "lora-graph"}},
                {"id": 8, "type_id": "comfy/VAEDecode", "fields": {}},
                {
                    "id": 6,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 3,
                        "seed": 56789,
                        "cfg": 2.0,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 1.0,
                    },
                },
                {
                    "id": 5,
                    "type_id": "comfy/LoraLoader",
                    "fields": {
                        "lora_name": "graph_lora.safetensors",
                        "strength_model": 0.8,
                        "strength_clip": 0.0,
                    },
                },
                {"id": 4, "type_id": "comfy/EmptyLatentImage", "fields": {"width": 576, "height": 512, "batch_size": 1}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "lora negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "lora positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def lora_clip_unsupported_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 5, "port": "clip"}},
                {"from": {"node": 5, "port": "MODEL"}, "to": {"node": 6, "port": "model"}},
                {"from": {"node": 5, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 6, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 6, "port": "negative"}},
                {"from": {"node": 4, "port": "LATENT"}, "to": {"node": 6, "port": "latent_image"}},
            ],
            "nodes": [
                {
                    "id": 6,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 3,
                        "seed": 56789,
                        "cfg": 2.0,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 1.0,
                    },
                },
                {
                    "id": 5,
                    "type_id": "comfy/LoraLoader",
                    "fields": {
                        "lora_name": "clip_lora.safetensors",
                        "strength_model": 0.8,
                        "strength_clip": 1.0,
                    },
                },
                {"id": 4, "type_id": "comfy/EmptyLatentImage", "fields": {"width": 576, "height": 512, "batch_size": 1}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "clip lora negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "clip lora positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def zimage_lora_model_only_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 5, "port": "MODEL"}, "to": {"node": 6, "port": "model"}},
                {"from": {"node": 6, "port": "MODEL"}, "to": {"node": 7, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 7, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 7, "port": "negative"}},
                {"from": {"node": 4, "port": "LATENT"}, "to": {"node": 7, "port": "latent_image"}},
                {"from": {"node": 7, "port": "LATENT"}, "to": {"node": 8, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 8, "port": "vae"}},
                {"from": {"node": 8, "port": "IMAGE"}, "to": {"node": 9, "port": "images"}},
            ],
            "nodes": [
                {"id": 9, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "zimage-lora-alias"}},
                {"id": 8, "type_id": "comfy/VAEDecode", "fields": {}},
                {
                    "id": 7,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 3,
                        "seed": 66789,
                        "cfg": 1.25,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 0.95,
                    },
                },
                {
                    "id": 6,
                    "type_id": "comfy/ZImageLoraModelOnly",
                    "fields": {
                        "lora_name": "zimage_second.safetensors",
                        "strength_model": 0.4,
                    },
                },
                {
                    "id": 5,
                    "type_id": "comfy/ZImageLoraModelOnly",
                    "fields": {
                        "lora_name": "zimage_first.safetensors",
                        "strength_model": 0.65,
                    },
                },
                {"id": 4, "type_id": "comfy/EmptySD3LatentImage", "fields": {"width": 640, "height": 512, "batch_size": 1}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "zimage alias negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "zimage alias positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def mask_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 4, "port": "IMAGE"}, "to": {"node": 5, "port": "pixels"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 5, "port": "vae"}},
                {"from": {"node": 5, "port": "LATENT"}, "to": {"node": 7, "port": "samples"}},
                {"from": {"node": 6, "port": "MASK"}, "to": {"node": 7, "port": "mask"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 8, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 8, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 8, "port": "negative"}},
                {"from": {"node": 7, "port": "LATENT"}, "to": {"node": 8, "port": "latent_image"}},
                {"from": {"node": 8, "port": "LATENT"}, "to": {"node": 9, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 9, "port": "vae"}},
                {"from": {"node": 9, "port": "IMAGE"}, "to": {"node": 10, "port": "images"}},
            ],
            "nodes": [
                {"id": 10, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "mask-graph"}},
                {"id": 9, "type_id": "comfy/VAEDecode", "fields": {}},
                {
                    "id": 8,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 4,
                        "seed": 45678,
                        "cfg": 2.25,
                        "sampler_name": "euler",
                        "scheduler": "karras",
                        "denoise": 0.35,
                    },
                },
                {"id": 7, "type_id": "comfy/SetLatentNoiseMask", "fields": {}},
                {"id": 6, "type_id": "comfy/LoadImage", "fields": {"image": "/tmp/serenity_graph_mask.png"}},
                {"id": 5, "type_id": "comfy/VAEEncode", "fields": {}},
                {"id": 4, "type_id": "comfy/LoadImage", "fields": {"image": "/tmp/serenity_graph_init.png"}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "mask negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "mask positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def outpaint_preprocess_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 3, "port": "clip"}},
                {"from": {"node": 4, "port": "IMAGE"}, "to": {"node": 5, "port": "image"}},
                {"from": {"node": 5, "port": "IMAGE"}, "to": {"node": 6, "port": "pixels"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 6, "port": "vae"}},
                {"from": {"node": 5, "port": "MASK"}, "to": {"node": 7, "port": "mask"}},
                {"from": {"node": 6, "port": "LATENT"}, "to": {"node": 8, "port": "samples"}},
                {"from": {"node": 7, "port": "MASK"}, "to": {"node": 8, "port": "mask"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 9, "port": "model"}},
                {"from": {"node": 3, "port": "CONDITIONING"}, "to": {"node": 9, "port": "positive"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 9, "port": "negative"}},
                {"from": {"node": 8, "port": "LATENT"}, "to": {"node": 9, "port": "latent_image"}},
                {"from": {"node": 9, "port": "LATENT"}, "to": {"node": 10, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 10, "port": "vae"}},
                {"from": {"node": 5, "port": "IMAGE"}, "to": {"node": 11, "port": "image1"}},
                {"from": {"node": 10, "port": "IMAGE"}, "to": {"node": 11, "port": "image2"}},
                {"from": {"node": 7, "port": "MASK"}, "to": {"node": 11, "port": "mask"}},
                {"from": {"node": 11, "port": "IMAGE"}, "to": {"node": 12, "port": "images"}},
            ],
            "nodes": [
                {"id": 12, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "outpaint-preprocess-graph"}},
                {"id": 11, "type_id": "comfy/LanPaint_MaskBlend", "fields": {"blend_overlap": 9}},
                {"id": 10, "type_id": "comfy/VAEDecode", "fields": {}},
                {
                    "id": 9,
                    "type_id": "comfy/KSampler",
                    "fields": {
                        "steps": 4,
                        "seed": 77889,
                        "cfg": 2.5,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 0.45,
                    },
                },
                {"id": 8, "type_id": "comfy/SetLatentNoiseMask", "fields": {}},
                {"id": 7, "type_id": "comfy/ThresholdMask", "fields": {"value": 0.01}},
                {"id": 6, "type_id": "comfy/VAEEncode", "fields": {}},
                {
                    "id": 5,
                    "type_id": "comfy/ImagePadForOutpaint",
                    "fields": {"left": 200, "top": 200, "right": 200, "bottom": 200, "feathering": 20},
                },
                {"id": 4, "type_id": "comfy/LoadImage", "fields": {"image": "/tmp/serenity_outpaint_source.png"}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "outpaint negative prompt"}},
                {"id": 3, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "outpaint positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def basic_scheduler_workflow_request() -> dict[str, Any]:
    return {
        "workflow": {
            "version": 1,
            "edges": [
                {"from": {"node": 1, "port": "CLIP"}, "to": {"node": 2, "port": "clip"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 4, "port": "model"}},
                {"from": {"node": 2, "port": "CONDITIONING"}, "to": {"node": 4, "port": "conditioning"}},
                {"from": {"node": 1, "port": "MODEL"}, "to": {"node": 5, "port": "model"}},
                {"from": {"node": 6, "port": "NOISE"}, "to": {"node": 8, "port": "noise"}},
                {"from": {"node": 4, "port": "GUIDER"}, "to": {"node": 8, "port": "guider"}},
                {"from": {"node": 7, "port": "SAMPLER"}, "to": {"node": 8, "port": "sampler"}},
                {"from": {"node": 5, "port": "SIGMAS"}, "to": {"node": 8, "port": "sigmas"}},
                {"from": {"node": 3, "port": "LATENT"}, "to": {"node": 8, "port": "latent_image"}},
                {"from": {"node": 8, "port": "LATENT"}, "to": {"node": 9, "port": "samples"}},
                {"from": {"node": 1, "port": "VAE"}, "to": {"node": 9, "port": "vae"}},
                {"from": {"node": 9, "port": "IMAGE"}, "to": {"node": 10, "port": "images"}},
            ],
            "nodes": [
                {"id": 10, "type_id": "comfy/SaveImage", "fields": {"filename_prefix": "basic-scheduler-graph"}},
                {"id": 9, "type_id": "comfy/VAEDecode", "fields": {}},
                {"id": 8, "type_id": "comfy/SamplerCustomAdvanced", "fields": {}},
                {"id": 7, "type_id": "comfy/KSamplerSelect", "fields": {"sampler_name": "euler"}},
                {"id": 6, "type_id": "comfy/RandomNoise", "fields": {"noise_seed": 67890}},
                {"id": 5, "type_id": "comfy/BasicScheduler", "fields": {"scheduler": "simple", "steps": 8, "denoise": 0.33}},
                {"id": 4, "type_id": "comfy/BasicGuider", "fields": {}},
                {"id": 3, "type_id": "comfy/EmptySD3LatentImage", "fields": {"width": 768, "height": 512, "batch_size": 1}},
                {"id": 2, "type_id": "comfy/CLIPTextEncode", "fields": {"text": "basic scheduler positive prompt"}},
                {"id": 1, "type_id": "comfy/CheckpointLoaderSimple", "fields": {"ckpt_name": "stub"}},
            ],
        }
    }


def unsupported_comfy_api_prompt_request() -> dict[str, Any]:
    return {
        "workflow": {
            "prompt": {
                "1": {"class_type": "ControlNetApply", "inputs": {}},
            }
        }
    }


def require(condition: bool, msg: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(msg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    port = args.port if args.port else find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = args.log or CHECKS_DIR / f"workflow_graph_product_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.daemon), "stub", str(port)]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        report: dict[str, Any] = {
            "schema": "serenity.workflow_graph_product.v1",
            "command": command,
            "log_path": str(log_path),
            "blockers": blockers,
        }
        try:
            report["health"] = wait_health(base_url, args.startup_timeout)

            unsupported_status, unsupported_data, unsupported_text = http_json("POST", f"{base_url}/v1/generate", unsupported_workflow_request())
            report["unsupported_node"] = {"status": unsupported_status, "body": unsupported_data}
            require(unsupported_status == 501, "unsupported graph node did not return HTTP 501", blockers)
            require("ControlNetApply" in unsupported_text, "unsupported graph response did not name ControlNetApply", blockers)

            wrong_status, wrong_data, wrong_text = http_json("POST", f"{base_url}/v1/generate", wrong_type_workflow_request())
            report["wrong_type_link"] = {"status": wrong_status, "body": wrong_data}
            require(wrong_status == 501, "wrong typed link did not return HTTP 501", blockers)
            require("expected CONDITIONING" in wrong_text, "wrong typed link response did not name expected type", blockers)

            lora_clip_status, lora_clip_data, lora_clip_text = http_json(
                "POST", f"{base_url}/v1/generate", lora_clip_unsupported_workflow_request()
            )
            report["lora_clip_unsupported"] = {"status": lora_clip_status, "body": lora_clip_data}
            require(lora_clip_status == 501, "LoraLoader CLIP-side graph did not return HTTP 501", blockers)
            require(
                "CLIP_LORA_UNSUPPORTED" in lora_clip_text,
                "LoraLoader CLIP-side graph response did not name unsupported CLIP LoRA semantics",
                blockers,
            )

            unsupported_api_status, unsupported_api_data, unsupported_api_text = http_json(
                "POST", f"{base_url}/v1/generate", unsupported_comfy_api_prompt_request()
            )
            report["unsupported_comfy_api_node"] = {"status": unsupported_api_status, "body": unsupported_api_data}
            require(unsupported_api_status == 501, "unsupported Comfy API graph node did not return HTTP 501", blockers)
            require("ControlNetApply" in unsupported_api_text, "unsupported Comfy API response did not name ControlNetApply", blockers)

            missing_inpaint_status, missing_inpaint_data, missing_inpaint_text = http_json(
                "POST", f"{base_url}/v1/generate", inpaint_conditioning_missing_mask_comfy_api_prompt_request()
            )
            report["inpaint_conditioning_missing_mask"] = {
                "status": missing_inpaint_status,
                "body": missing_inpaint_data,
            }
            require(missing_inpaint_status == 501, "InpaintModelConditioning missing-mask graph did not return HTTP 501", blockers)
            require(
                "InpaintModelConditioning missing required typed input" in missing_inpaint_text,
                "InpaintModelConditioning missing-mask response did not name the missing typed input",
                blockers,
            )

            request = linked_workflow_request()
            gen_status, gen_data, gen_text = http_json("POST", f"{base_url}/v1/generate", request)
            report["generate"] = {"status": gen_status, "body": gen_data}
            if gen_status != 200 or not isinstance(gen_data, dict) or not gen_data.get("job_id"):
                blockers.append(f"linked workflow generate failed HTTP {gen_status}: {gen_text}")
            else:
                job_id = str(gen_data["job_id"])
                job = poll_job(base_url, job_id, args.timeout)
                report["job"] = job
                require(job.get("state") == "done", f"linked workflow job state was {job.get('state')}", blockers)
                png_path = Path(str(job.get("output_path") or ""))
                require(png_path.is_file(), f"linked workflow PNG missing: {png_path}", blockers)
                if png_path.is_file():
                    text = read_png_text(png_path)
                    genparams = json.loads(text.get(GENPARAMS_KEY, "{}"))
                    report["png"] = {"path": str(png_path), "idat_sha256": text.get("_idat_sha256"), "genparams": genparams}
                    require(genparams.get("prompt") == "linked positive prompt", "linked positive prompt was not consumed from positive edge", blockers)
                    require(genparams.get("negative") == "linked negative prompt", "linked negative prompt was not consumed from negative edge", blockers)
                    require(genparams.get("model") == "stub", "checkpoint model did not flow through MODEL edge", blockers)
                    require(genparams.get("width") == 640 and genparams.get("height") == 512, "latent dimensions did not flow into sampler request", blockers)
                    require(genparams.get("steps") == 7, "KSampler steps missing from genparams", blockers)
                    require(genparams.get("seed") == 12345, "KSampler seed missing from genparams", blockers)
                    require(genparams.get("cfg") == 3.5, "KSampler cfg missing from genparams", blockers)
                    require(genparams.get("sampler") == "euler", "KSampler sampler_name missing from genparams", blockers)
                    require(genparams.get("scheduler") == "karras", "KSampler scheduler missing from genparams", blockers)
                    require(genparams.get("creativity") == 0.75, "KSampler denoise missing from genparams", blockers)
                    require(genparams.get("workflow_schema") == "serenity.workflow_graph.v1", "workflow schema missing from genparams", blockers)
                    require(genparams.get("workflow_executor") == "serenity.workflow_graph.executor.v1", "workflow executor missing from genparams", blockers)
                    require(genparams.get("workflow_source") == "typed_linked_graph", "workflow source missing from genparams", blockers)
                    require(genparams.get("workflow_save_prefix") == "typed-graph", "SaveImage filename_prefix missing from genparams", blockers)
                    require(genparams.get("workflow_node_count") == 7, "workflow node count missing from genparams", blockers)
                    require(genparams.get("workflow_edge_count") == 9, "workflow edge count missing from genparams", blockers)

            img_status, img_data, img_text = http_json("POST", f"{base_url}/v1/generate", img2img_workflow_request())
            report["img2img_generate"] = {"status": img_status, "body": img_data}
            if img_status != 200 or not isinstance(img_data, dict) or not img_data.get("job_id"):
                blockers.append(f"img2img workflow generate failed HTTP {img_status}: {img_text}")
            else:
                img_job_id = str(img_data["job_id"])
                img_job = poll_job(base_url, img_job_id, args.timeout)
                report["img2img_job"] = img_job
                require(img_job.get("state") == "done", f"img2img workflow job state was {img_job.get('state')}", blockers)
                img_png_path = Path(str(img_job.get("output_path") or ""))
                require(img_png_path.is_file(), f"img2img workflow PNG missing: {img_png_path}", blockers)
                if img_png_path.is_file():
                    img_text_chunks = read_png_text(img_png_path)
                    img_genparams = json.loads(img_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["img2img_png"] = {
                        "path": str(img_png_path),
                        "idat_sha256": img_text_chunks.get("_idat_sha256"),
                        "genparams": img_genparams,
                    }
                    require(img_genparams.get("prompt") == "img2img positive prompt", "img2img positive prompt was not consumed", blockers)
                    require(img_genparams.get("negative") == "img2img negative prompt", "img2img negative prompt was not consumed", blockers)
                    require(img_genparams.get("init_image") == "/tmp/serenity_graph_init.png", "LoadImage path did not flow into init_image", blockers)
                    require(img_genparams.get("creativity") == 0.4, "img2img KSampler denoise missing from creativity", blockers)
                    require(img_genparams.get("steps") == 5, "img2img KSampler steps missing", blockers)
                    require(img_genparams.get("seed") == 34567, "img2img KSampler seed missing", blockers)
                    require(img_genparams.get("workflow_source") == "typed_linked_graph", "img2img workflow source missing", blockers)
                    require(img_genparams.get("workflow_save_prefix") == "img2img-graph", "img2img SaveImage filename_prefix missing", blockers)
                    require(img_genparams.get("workflow_node_count") == 8, "img2img workflow node count missing", blockers)
                    require(img_genparams.get("workflow_edge_count") == 11, "img2img workflow edge count missing", blockers)

            lora_status, lora_data, lora_text = http_json("POST", f"{base_url}/v1/generate", lora_workflow_request())
            report["lora_generate"] = {"status": lora_status, "body": lora_data}
            if lora_status != 200 or not isinstance(lora_data, dict) or not lora_data.get("job_id"):
                blockers.append(f"LoRA workflow generate failed HTTP {lora_status}: {lora_text}")
            else:
                lora_job_id = str(lora_data["job_id"])
                lora_job = poll_job(base_url, lora_job_id, args.timeout)
                report["lora_job"] = lora_job
                require(lora_job.get("state") == "done", f"LoRA workflow job state was {lora_job.get('state')}", blockers)
                lora_png_path = Path(str(lora_job.get("output_path") or ""))
                require(lora_png_path.is_file(), f"LoRA workflow PNG missing: {lora_png_path}", blockers)
                if lora_png_path.is_file():
                    lora_text_chunks = read_png_text(lora_png_path)
                    lora_genparams = json.loads(lora_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["lora_png"] = {
                        "path": str(lora_png_path),
                        "idat_sha256": lora_text_chunks.get("_idat_sha256"),
                        "genparams": lora_genparams,
                    }
                    loras = lora_genparams.get("lora")
                    require(lora_genparams.get("prompt") == "lora positive prompt", "LoRA positive prompt was not consumed", blockers)
                    require(lora_genparams.get("negative") == "lora negative prompt", "LoRA negative prompt was not consumed", blockers)
                    require(isinstance(loras, list) and len(loras) == 1, "LoRA graph did not emit one flat lora entry", blockers)
                    if isinstance(loras, list) and loras and isinstance(loras[0], dict):
                        require(loras[0].get("name") == "graph_lora.safetensors", "LoRA graph name missing", blockers)
                        require(loras[0].get("weight") == 0.8, "LoRA graph strength_model missing", blockers)
                    require(lora_genparams.get("workflow_source") == "typed_linked_graph", "LoRA workflow source missing", blockers)
                    require(lora_genparams.get("workflow_save_prefix") == "lora-graph", "LoRA SaveImage filename_prefix missing", blockers)
                    require(lora_genparams.get("workflow_node_count") == 8, "LoRA workflow node count missing", blockers)
                    require(lora_genparams.get("workflow_edge_count") == 11, "LoRA workflow edge count missing", blockers)

            z_lora_status, z_lora_data, z_lora_text = http_json(
                "POST", f"{base_url}/v1/generate", zimage_lora_model_only_workflow_request()
            )
            report["zimage_lora_alias_generate"] = {"status": z_lora_status, "body": z_lora_data}
            if z_lora_status != 200 or not isinstance(z_lora_data, dict) or not z_lora_data.get("job_id"):
                blockers.append(f"ZImageLoraModelOnly workflow generate failed HTTP {z_lora_status}: {z_lora_text}")
            else:
                z_lora_job_id = str(z_lora_data["job_id"])
                z_lora_job = poll_job(base_url, z_lora_job_id, args.timeout)
                report["zimage_lora_alias_job"] = z_lora_job
                require(
                    z_lora_job.get("state") == "done",
                    f"ZImageLoraModelOnly workflow job state was {z_lora_job.get('state')}",
                    blockers,
                )
                z_lora_png_path = Path(str(z_lora_job.get("output_path") or ""))
                require(z_lora_png_path.is_file(), f"ZImageLoraModelOnly workflow PNG missing: {z_lora_png_path}", blockers)
                if z_lora_png_path.is_file():
                    z_lora_text_chunks = read_png_text(z_lora_png_path)
                    z_lora_genparams = json.loads(z_lora_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["zimage_lora_alias_png"] = {
                        "path": str(z_lora_png_path),
                        "idat_sha256": z_lora_text_chunks.get("_idat_sha256"),
                        "genparams": z_lora_genparams,
                    }
                    z_loras = z_lora_genparams.get("lora")
                    require(
                        z_lora_genparams.get("prompt") == "zimage alias positive prompt",
                        "ZImageLoraModelOnly positive prompt was not consumed",
                        blockers,
                    )
                    require(isinstance(z_loras, list) and len(z_loras) == 2, "ZImageLoraModelOnly graph did not emit two flat lora entries", blockers)
                    if isinstance(z_loras, list) and len(z_loras) == 2 and all(isinstance(item, dict) for item in z_loras):
                        require(
                            z_loras[0].get("name") == "zimage_first.safetensors"
                            and z_loras[0].get("weight") == 0.65,
                            "first ZImageLoraModelOnly metadata missing",
                            blockers,
                        )
                        require(
                            z_loras[1].get("name") == "zimage_second.safetensors"
                            and z_loras[1].get("weight") == 0.4,
                            "second ZImageLoraModelOnly metadata missing",
                            blockers,
                        )
                    require(z_lora_genparams.get("workflow_save_prefix") == "zimage-lora-alias", "ZImageLoraModelOnly SaveImage filename_prefix missing", blockers)
                    require(z_lora_genparams.get("workflow_node_count") == 9, "ZImageLoraModelOnly workflow node count missing", blockers)
                    require(z_lora_genparams.get("workflow_edge_count") == 11, "ZImageLoraModelOnly workflow edge count missing", blockers)

            mask_status, mask_data, mask_text = http_json("POST", f"{base_url}/v1/generate", mask_workflow_request())
            report["mask_generate"] = {"status": mask_status, "body": mask_data}
            if mask_status != 200 or not isinstance(mask_data, dict) or not mask_data.get("job_id"):
                blockers.append(f"mask workflow generate failed HTTP {mask_status}: {mask_text}")
            else:
                mask_job_id = str(mask_data["job_id"])
                mask_job = poll_job(base_url, mask_job_id, args.timeout)
                report["mask_job"] = mask_job
                require(mask_job.get("state") == "done", f"mask workflow job state was {mask_job.get('state')}", blockers)
                mask_png_path = Path(str(mask_job.get("output_path") or ""))
                require(mask_png_path.is_file(), f"mask workflow PNG missing: {mask_png_path}", blockers)
                if mask_png_path.is_file():
                    mask_text_chunks = read_png_text(mask_png_path)
                    mask_genparams = json.loads(mask_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["mask_png"] = {
                        "path": str(mask_png_path),
                        "idat_sha256": mask_text_chunks.get("_idat_sha256"),
                        "genparams": mask_genparams,
                    }
                    require(mask_genparams.get("prompt") == "mask positive prompt", "mask positive prompt was not consumed", blockers)
                    require(mask_genparams.get("negative") == "mask negative prompt", "mask negative prompt was not consumed", blockers)
                    require(mask_genparams.get("init_image") == "/tmp/serenity_graph_init.png", "mask graph init image missing", blockers)
                    require(mask_genparams.get("mask_image") == "/tmp/serenity_graph_mask.png", "SetLatentNoiseMask path did not flow into mask_image", blockers)
                    require(mask_genparams.get("creativity") == 0.35, "mask KSampler denoise missing from creativity", blockers)
                    require(mask_genparams.get("steps") == 4, "mask KSampler steps missing", blockers)
                    require(mask_genparams.get("seed") == 45678, "mask KSampler seed missing", blockers)
                    require(mask_genparams.get("workflow_source") == "typed_linked_graph", "mask workflow source missing", blockers)
                    require(mask_genparams.get("workflow_save_prefix") == "mask-graph", "mask SaveImage filename_prefix missing", blockers)
                    require(mask_genparams.get("workflow_node_count") == 10, "mask workflow node count missing", blockers)
                    require(mask_genparams.get("workflow_edge_count") == 13, "mask workflow edge count missing", blockers)

            outpaint_status, outpaint_data, outpaint_text = http_json(
                "POST", f"{base_url}/v1/generate", outpaint_preprocess_workflow_request()
            )
            report["outpaint_preprocess_generate"] = {"status": outpaint_status, "body": outpaint_data}
            if outpaint_status != 200 or not isinstance(outpaint_data, dict) or not outpaint_data.get("job_id"):
                blockers.append(f"outpaint preprocess workflow generate failed HTTP {outpaint_status}: {outpaint_text}")
            else:
                outpaint_job_id = str(outpaint_data["job_id"])
                outpaint_job = poll_job(base_url, outpaint_job_id, args.timeout)
                report["outpaint_preprocess_job"] = outpaint_job
                require(outpaint_job.get("state") == "done", f"outpaint preprocess workflow job state was {outpaint_job.get('state')}", blockers)
                outpaint_png_path = Path(str(outpaint_job.get("output_path") or ""))
                require(outpaint_png_path.is_file(), f"outpaint preprocess workflow PNG missing: {outpaint_png_path}", blockers)
                if outpaint_png_path.is_file():
                    outpaint_text_chunks = read_png_text(outpaint_png_path)
                    outpaint_genparams = json.loads(outpaint_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["outpaint_preprocess_png"] = {
                        "path": str(outpaint_png_path),
                        "idat_sha256": outpaint_text_chunks.get("_idat_sha256"),
                        "genparams": outpaint_genparams,
                    }
                    require(outpaint_genparams.get("prompt") == "outpaint positive prompt", "outpaint positive prompt was not consumed", blockers)
                    require(outpaint_genparams.get("negative") == "outpaint negative prompt", "outpaint negative prompt was not consumed", blockers)
                    require(outpaint_genparams.get("init_image") == "/tmp/serenity_outpaint_source.png", "ImagePadForOutpaint image path did not flow into init_image", blockers)
                    require(outpaint_genparams.get("mask_image") == "/tmp/serenity_outpaint_source.png", "ImagePadForOutpaint mask source did not flow into mask_image", blockers)
                    require(outpaint_genparams.get("lanpaint_mask_channel") == "image_pad_for_outpaint", "ImagePadForOutpaint mask source token missing", blockers)
                    require(outpaint_genparams.get("outpaint_left") == 200, "ImagePadForOutpaint left padding missing", blockers)
                    require(outpaint_genparams.get("outpaint_top") == 200, "ImagePadForOutpaint top padding missing", blockers)
                    require(outpaint_genparams.get("outpaint_right") == 200, "ImagePadForOutpaint right padding missing", blockers)
                    require(outpaint_genparams.get("outpaint_bottom") == 200, "ImagePadForOutpaint bottom padding missing", blockers)
                    require(outpaint_genparams.get("outpaint_feathering") == 20, "ImagePadForOutpaint feathering missing", blockers)
                    require(outpaint_genparams.get("threshold_mask_value") == 0.01, "ThresholdMask value missing", blockers)
                    require(outpaint_genparams.get("threshold_mask_operator") == "gt", "ThresholdMask strict Comfy operator missing", blockers)
                    require(outpaint_genparams.get("lanpaint_mask_blend_overlap") == 9, "LanPaint_MaskBlend overlap missing from outpaint graph", blockers)
                    require(outpaint_genparams.get("creativity") == 0.45, "outpaint KSampler denoise missing from creativity", blockers)
                    require(outpaint_genparams.get("steps") == 4, "outpaint KSampler steps missing", blockers)
                    require(outpaint_genparams.get("seed") == 77889, "outpaint KSampler seed missing", blockers)
                    require(outpaint_genparams.get("workflow_source") == "typed_linked_graph", "outpaint workflow source missing", blockers)
                    require(outpaint_genparams.get("workflow_save_prefix") == "outpaint-preprocess-graph", "outpaint SaveImage filename_prefix missing", blockers)
                    require(outpaint_genparams.get("workflow_node_count") == 12, "outpaint workflow node count missing", blockers)
                    require(outpaint_genparams.get("workflow_edge_count") == 18, "outpaint workflow edge count missing", blockers)

            basic_status, basic_data, basic_text = http_json("POST", f"{base_url}/v1/generate", basic_scheduler_workflow_request())
            report["basic_scheduler_generate"] = {"status": basic_status, "body": basic_data}
            if basic_status != 200 or not isinstance(basic_data, dict) or not basic_data.get("job_id"):
                blockers.append(f"BasicScheduler workflow generate failed HTTP {basic_status}: {basic_text}")
            else:
                basic_job_id = str(basic_data["job_id"])
                basic_job = poll_job(base_url, basic_job_id, args.timeout)
                report["basic_scheduler_job"] = basic_job
                require(basic_job.get("state") == "done", f"BasicScheduler workflow job state was {basic_job.get('state')}", blockers)
                basic_png_path = Path(str(basic_job.get("output_path") or ""))
                require(basic_png_path.is_file(), f"BasicScheduler workflow PNG missing: {basic_png_path}", blockers)
                if basic_png_path.is_file():
                    basic_text_chunks = read_png_text(basic_png_path)
                    basic_genparams = json.loads(basic_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["basic_scheduler_png"] = {
                        "path": str(basic_png_path),
                        "idat_sha256": basic_text_chunks.get("_idat_sha256"),
                        "genparams": basic_genparams,
                    }
                    require(basic_genparams.get("prompt") == "basic scheduler positive prompt", "BasicScheduler prompt was not consumed", blockers)
                    require(basic_genparams.get("model") == "stub", "BasicScheduler model did not flow through MODEL edge", blockers)
                    require(
                        basic_genparams.get("width") == 768 and basic_genparams.get("height") == 512,
                        "BasicScheduler latent dimensions missing",
                        blockers,
                    )
                    require(basic_genparams.get("steps") == 8, "BasicScheduler steps missing from SIGMAS metadata", blockers)
                    require(basic_genparams.get("seed") == 67890, "BasicScheduler RandomNoise seed missing", blockers)
                    require(basic_genparams.get("sampler") == "euler", "BasicScheduler sampler selection missing", blockers)
                    require(basic_genparams.get("scheduler") == "simple", "BasicScheduler scheduler missing from SIGMAS metadata", blockers)
                    require(basic_genparams.get("creativity") == 0.33, "BasicScheduler denoise missing from creativity", blockers)
                    require(basic_genparams.get("workflow_source") == "typed_linked_graph", "BasicScheduler workflow source missing", blockers)
                    require(basic_genparams.get("workflow_save_prefix") == "basic-scheduler-graph", "BasicScheduler SaveImage filename_prefix missing", blockers)
                    require(basic_genparams.get("workflow_node_count") == 10, "BasicScheduler workflow node count missing", blockers)
                    require(basic_genparams.get("workflow_edge_count") == 12, "BasicScheduler workflow edge count missing", blockers)

            api_status, api_data, api_text = http_json("POST", f"{base_url}/v1/generate", comfy_api_prompt_request())
            report["comfy_api_generate"] = {"status": api_status, "body": api_data}
            if api_status != 200 or not isinstance(api_data, dict) or not api_data.get("job_id"):
                blockers.append(f"Comfy API prompt generate failed HTTP {api_status}: {api_text}")
            else:
                api_job_id = str(api_data["job_id"])
                api_job = poll_job(base_url, api_job_id, args.timeout)
                report["comfy_api_job"] = api_job
                require(api_job.get("state") == "done", f"Comfy API prompt job state was {api_job.get('state')}", blockers)
                api_png_path = Path(str(api_job.get("output_path") or ""))
                require(api_png_path.is_file(), f"Comfy API prompt PNG missing: {api_png_path}", blockers)
                if api_png_path.is_file():
                    api_text_chunks = read_png_text(api_png_path)
                    api_genparams = json.loads(api_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["comfy_api_png"] = {
                        "path": str(api_png_path),
                        "idat_sha256": api_text_chunks.get("_idat_sha256"),
                        "genparams": api_genparams,
                    }
                    require(api_genparams.get("prompt") == "api positive prompt", "Comfy API positive prompt was not consumed", blockers)
                    require(api_genparams.get("negative") == "api negative prompt", "Comfy API negative prompt was not consumed", blockers)
                    require(api_genparams.get("model") == "stub", "Comfy API checkpoint model did not flow through MODEL edge", blockers)
                    require(api_genparams.get("width") == 704 and api_genparams.get("height") == 512, "Comfy API latent dimensions missing", blockers)
                    require(api_genparams.get("steps") == 6, "Comfy API KSampler steps missing", blockers)
                    require(api_genparams.get("seed") == 23456, "Comfy API KSampler seed missing", blockers)
                    require(api_genparams.get("cfg") == 4.25, "Comfy API KSampler cfg missing", blockers)
                    require(api_genparams.get("sampler") == "euler", "Comfy API sampler missing", blockers)
                    require(api_genparams.get("scheduler") == "karras", "Comfy API scheduler missing", blockers)
                    require(api_genparams.get("creativity") == 0.625, "Comfy API denoise missing", blockers)
                    require(api_genparams.get("workflow_source") == "comfy_api_prompt_graph", "Comfy API workflow source missing", blockers)
                    require(api_genparams.get("workflow_save_prefix") == "comfy-api", "Comfy API SaveImage filename_prefix missing", blockers)
                    require(api_genparams.get("workflow_node_count") == 7, "Comfy API workflow node count missing", blockers)
                    require(api_genparams.get("workflow_edge_count") == 9, "Comfy API workflow edge count missing", blockers)

            outpaint_api_status, outpaint_api_data, outpaint_api_text = http_json(
                "POST", f"{base_url}/v1/generate", outpaint_threshold_comfy_api_prompt_request()
            )
            report["outpaint_threshold_api_generate"] = {"status": outpaint_api_status, "body": outpaint_api_data}
            if outpaint_api_status != 200 or not isinstance(outpaint_api_data, dict) or not outpaint_api_data.get("job_id"):
                blockers.append(f"outpaint ThresholdMask Comfy API prompt generate failed HTTP {outpaint_api_status}: {outpaint_api_text}")
            else:
                outpaint_api_job_id = str(outpaint_api_data["job_id"])
                outpaint_api_job = poll_job(base_url, outpaint_api_job_id, args.timeout)
                report["outpaint_threshold_api_job"] = outpaint_api_job
                require(
                    outpaint_api_job.get("state") == "done",
                    f"outpaint ThresholdMask Comfy API prompt job state was {outpaint_api_job.get('state')}",
                    blockers,
                )
                outpaint_api_png_path = Path(str(outpaint_api_job.get("output_path") or ""))
                require(outpaint_api_png_path.is_file(), f"outpaint ThresholdMask Comfy API prompt PNG missing: {outpaint_api_png_path}", blockers)
                if outpaint_api_png_path.is_file():
                    outpaint_api_text_chunks = read_png_text(outpaint_api_png_path)
                    outpaint_api_genparams = json.loads(outpaint_api_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["outpaint_threshold_api_png"] = {
                        "path": str(outpaint_api_png_path),
                        "idat_sha256": outpaint_api_text_chunks.get("_idat_sha256"),
                        "genparams": outpaint_api_genparams,
                    }
                    require(outpaint_api_genparams.get("prompt") == "outpaint threshold positive prompt", "outpaint API positive prompt was not consumed", blockers)
                    require(outpaint_api_genparams.get("negative") == "outpaint threshold negative prompt", "outpaint API negative prompt was not consumed", blockers)
                    require(outpaint_api_genparams.get("model") == "stub", "outpaint API model missing", blockers)
                    require(outpaint_api_genparams.get("init_image") == "/tmp/serenity_graph_init.png", "outpaint API init_image missing", blockers)
                    require(outpaint_api_genparams.get("mask_image") == "/tmp/serenity_graph_init.png", "outpaint API mask_image missing", blockers)
                    require(outpaint_api_genparams.get("lanpaint_mask_channel") == "image_pad_for_outpaint", "outpaint API mask source token missing", blockers)
                    require(outpaint_api_genparams.get("outpaint_left") == 16, "outpaint API left padding missing", blockers)
                    require(outpaint_api_genparams.get("outpaint_top") == 8, "outpaint API top padding missing", blockers)
                    require(outpaint_api_genparams.get("outpaint_right") == 16, "outpaint API right padding missing", blockers)
                    require(outpaint_api_genparams.get("outpaint_bottom") == 8, "outpaint API bottom padding missing", blockers)
                    require(outpaint_api_genparams.get("outpaint_feathering") == 0, "outpaint API feathering missing", blockers)
                    require(outpaint_api_genparams.get("threshold_mask_value") == 0.5, "outpaint API ThresholdMask value missing", blockers)
                    require(outpaint_api_genparams.get("threshold_mask_operator") == "gt", "outpaint API strict ThresholdMask operator missing", blockers)
                    require(outpaint_api_genparams.get("steps") == 4, "outpaint API KSampler steps missing", blockers)
                    require(outpaint_api_genparams.get("seed") == 55678, "outpaint API KSampler seed missing", blockers)
                    require(outpaint_api_genparams.get("cfg") == 2.25, "outpaint API KSampler cfg missing", blockers)
                    require(outpaint_api_genparams.get("sampler") == "euler", "outpaint API sampler missing", blockers)
                    require(outpaint_api_genparams.get("scheduler") == "karras", "outpaint API scheduler missing", blockers)
                    require(outpaint_api_genparams.get("creativity") == 0.45, "outpaint API denoise missing", blockers)
                    require(outpaint_api_genparams.get("workflow_source") == "comfy_api_prompt_graph", "outpaint API workflow source missing", blockers)
                    require(outpaint_api_genparams.get("workflow_save_prefix") == "outpaint-threshold-graph", "outpaint API SaveImage filename_prefix missing", blockers)
                    require(outpaint_api_genparams.get("workflow_node_count") == 11, "outpaint API workflow node count missing", blockers)
                    require(outpaint_api_genparams.get("workflow_edge_count") == 15, "outpaint API workflow edge count missing", blockers)

            for inpaint_name, inpaint_noise_mask in (
                ("inpaint_conditioning_api", True),
                ("inpaint_conditioning_no_noise_mask_api", False),
            ):
                inpaint_status, inpaint_data, inpaint_text = http_json(
                    "POST",
                    f"{base_url}/v1/generate",
                    inpaint_conditioning_comfy_api_prompt_request(inpaint_noise_mask),
                )
                report[f"{inpaint_name}_generate"] = {"status": inpaint_status, "body": inpaint_data}
                if inpaint_status != 200 or not isinstance(inpaint_data, dict) or not inpaint_data.get("job_id"):
                    blockers.append(f"{inpaint_name} generate failed HTTP {inpaint_status}: {inpaint_text}")
                    continue
                inpaint_job_id = str(inpaint_data["job_id"])
                inpaint_job = poll_job(base_url, inpaint_job_id, args.timeout)
                report[f"{inpaint_name}_job"] = inpaint_job
                require(inpaint_job.get("state") == "done", f"{inpaint_name} job state was {inpaint_job.get('state')}", blockers)
                inpaint_png_path = Path(str(inpaint_job.get("output_path") or ""))
                require(inpaint_png_path.is_file(), f"{inpaint_name} PNG missing: {inpaint_png_path}", blockers)
                if inpaint_png_path.is_file():
                    inpaint_text_chunks = read_png_text(inpaint_png_path)
                    inpaint_genparams = json.loads(inpaint_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report[f"{inpaint_name}_png"] = {
                        "path": str(inpaint_png_path),
                        "idat_sha256": inpaint_text_chunks.get("_idat_sha256"),
                        "genparams": inpaint_genparams,
                    }
                    require(
                        inpaint_genparams.get("prompt") == "inpaint conditioning positive prompt",
                        f"{inpaint_name} positive prompt was not consumed",
                        blockers,
                    )
                    require(
                        inpaint_genparams.get("negative") == "inpaint conditioning negative prompt",
                        f"{inpaint_name} negative prompt was not consumed",
                        blockers,
                    )
                    require(inpaint_genparams.get("model") == "stub", f"{inpaint_name} model missing", blockers)
                    require(
                        inpaint_genparams.get("init_image") == "/tmp/serenity_inpaint_conditioning.png",
                        f"{inpaint_name} init image missing",
                        blockers,
                    )
                    require(
                        inpaint_genparams.get("inpaint_conditioning_image") == "/tmp/serenity_inpaint_conditioning.png",
                        f"{inpaint_name} conditioning image missing",
                        blockers,
                    )
                    require(
                        inpaint_genparams.get("inpaint_conditioning_mask") == "/tmp/serenity_inpaint_conditioning.png",
                        f"{inpaint_name} conditioning mask missing",
                        blockers,
                    )
                    require(
                        inpaint_genparams.get("inpaint_conditioning_noise_mask") is inpaint_noise_mask,
                        f"{inpaint_name} noise_mask metadata missing",
                        blockers,
                    )
                    if inpaint_noise_mask:
                        require(
                            inpaint_genparams.get("mask_image") == "/tmp/serenity_inpaint_conditioning.png",
                            f"{inpaint_name} latent noise mask image missing",
                            blockers,
                        )
                        require(
                            inpaint_genparams.get("lanpaint_mask_channel") == "load_image_mask",
                            f"{inpaint_name} mask source missing",
                            blockers,
                        )
                        require(
                            inpaint_genparams.get("workflow_save_prefix") == "inpaint-conditioning-graph",
                            f"{inpaint_name} SaveImage filename_prefix missing",
                            blockers,
                        )
                        require(inpaint_genparams.get("seed") == 66789, f"{inpaint_name} seed missing", blockers)
                    else:
                        require(inpaint_genparams.get("mask_image") == "", f"{inpaint_name} must not set latent mask_image when noise_mask=false", blockers)
                        require(inpaint_genparams.get("lanpaint_mask_channel") == "", f"{inpaint_name} must not set mask channel when noise_mask=false", blockers)
                        require(
                            inpaint_genparams.get("workflow_save_prefix") == "inpaint-conditioning-no-noise-mask",
                            f"{inpaint_name} SaveImage filename_prefix missing",
                            blockers,
                        )
                        require(inpaint_genparams.get("seed") == 66790, f"{inpaint_name} seed missing", blockers)
                    require(inpaint_genparams.get("steps") == 4, f"{inpaint_name} KSampler steps missing", blockers)
                    require(inpaint_genparams.get("cfg") == 2.75, f"{inpaint_name} KSampler cfg missing", blockers)
                    require(inpaint_genparams.get("sampler") == "euler", f"{inpaint_name} sampler missing", blockers)
                    require(inpaint_genparams.get("scheduler") == "simple", f"{inpaint_name} scheduler missing", blockers)
                    require(inpaint_genparams.get("creativity") == 0.6, f"{inpaint_name} denoise missing", blockers)
                    require(inpaint_genparams.get("workflow_source") == "comfy_api_prompt_graph", f"{inpaint_name} workflow source missing", blockers)
                    require(inpaint_genparams.get("workflow_node_count") == 8, f"{inpaint_name} workflow node count missing", blockers)
                    require(inpaint_genparams.get("workflow_edge_count") == 14, f"{inpaint_name} workflow edge count missing", blockers)

            report["serenityflow_t2i"] = {}
            for sf_name, expected in SERENITYFLOW_T2I_CASES.items():
                sf_request = serenityflow_template_request(expected["template"])
                sf_status, sf_data, sf_text = http_json("POST", f"{base_url}/v1/generate", sf_request)
                sf_case: dict[str, Any] = {
                    "generate": {"status": sf_status, "body": sf_data},
                    "template": str(expected["template"]),
                }
                report["serenityflow_t2i"][sf_name] = sf_case
                if sf_name == "zimage_t2i":
                    report["serenityflow_zimage_t2i_generate"] = sf_case["generate"]
                if sf_status != 200 or not isinstance(sf_data, dict) or not sf_data.get("job_id"):
                    blockers.append(f"SerenityFlow {sf_name} generate failed HTTP {sf_status}: {sf_text}")
                    continue

                sf_job_id = str(sf_data["job_id"])
                sf_job = poll_job(base_url, sf_job_id, args.timeout)
                sf_case["job"] = sf_job
                if sf_name == "zimage_t2i":
                    report["serenityflow_zimage_t2i_job"] = sf_job
                require(sf_job.get("state") == "done", f"SerenityFlow {sf_name} job state was {sf_job.get('state')}", blockers)
                sf_png_path = Path(str(sf_job.get("output_path") or ""))
                require(sf_png_path.is_file(), f"SerenityFlow {sf_name} PNG missing: {sf_png_path}", blockers)
                if sf_png_path.is_file():
                    sf_text_chunks = read_png_text(sf_png_path)
                    sf_genparams = json.loads(sf_text_chunks.get(GENPARAMS_KEY, "{}"))
                    sf_png = {
                        "path": str(sf_png_path),
                        "idat_sha256": sf_text_chunks.get("_idat_sha256"),
                        "genparams": sf_genparams,
                    }
                    sf_case["png"] = sf_png
                    if sf_name == "zimage_t2i":
                        report["serenityflow_zimage_t2i_png"] = sf_png
                    require(sf_genparams.get("workflow_source") == "comfy_api_prompt_graph", f"SerenityFlow {sf_name} workflow source missing", blockers)
                    require(sf_genparams.get("model") == expected["model"], f"SerenityFlow {sf_name} model missing", blockers)
                    require(sf_genparams.get("prompt") == expected["prompt"], f"SerenityFlow {sf_name} prompt missing", blockers)
                    require(sf_genparams.get("negative") == expected["negative"], f"SerenityFlow {sf_name} negative conditioning missing", blockers)
                    require(
                        sf_genparams.get("width") == expected["width"] and sf_genparams.get("height") == expected["height"],
                        f"SerenityFlow {sf_name} latent dimensions missing",
                        blockers,
                    )
                    for field in (
                        "images",
                        "steps",
                        "seed",
                        "cfg",
                        "sampler",
                        "scheduler",
                        "creativity",
                        "sigma_shift",
                        "workflow_node_count",
                        "workflow_edge_count",
                    ):
                        require(
                            sf_genparams.get(field) == expected[field],
                            f"SerenityFlow {sf_name} {field} missing",
                            blockers,
                        )

            report["serenityflow_edit"] = {}
            for sf_name, expected in SERENITYFLOW_EDIT_CASES.items():
                sf_request = serenityflow_template_request(expected["template"])
                sf_status, sf_data, sf_text = http_json("POST", f"{base_url}/v1/generate", sf_request)
                sf_case: dict[str, Any] = {
                    "generate": {"status": sf_status, "body": sf_data},
                    "template": str(expected["template"]),
                }
                report["serenityflow_edit"][sf_name] = sf_case
                if sf_status != 200 or not isinstance(sf_data, dict) or not sf_data.get("job_id"):
                    blockers.append(f"SerenityFlow {sf_name} generate failed HTTP {sf_status}: {sf_text}")
                    continue

                sf_job_id = str(sf_data["job_id"])
                sf_job = poll_job(base_url, sf_job_id, args.timeout)
                sf_case["job"] = sf_job
                require(sf_job.get("state") == "done", f"SerenityFlow {sf_name} job state was {sf_job.get('state')}", blockers)
                sf_png_path = Path(str(sf_job.get("output_path") or ""))
                require(sf_png_path.is_file(), f"SerenityFlow {sf_name} PNG missing: {sf_png_path}", blockers)
                if sf_png_path.is_file():
                    sf_text_chunks = read_png_text(sf_png_path)
                    sf_genparams = json.loads(sf_text_chunks.get(GENPARAMS_KEY, "{}"))
                    sf_case["png"] = {
                        "path": str(sf_png_path),
                        "idat_sha256": sf_text_chunks.get("_idat_sha256"),
                        "genparams": sf_genparams,
                    }
                    require(sf_genparams.get("workflow_source") == "comfy_api_prompt_graph", f"SerenityFlow {sf_name} workflow source missing", blockers)
                    require(sf_genparams.get("model") == expected["model"], f"SerenityFlow {sf_name} model missing", blockers)
                    require(sf_genparams.get("prompt") == expected["prompt"], f"SerenityFlow {sf_name} prompt missing", blockers)
                    require(sf_genparams.get("negative") == expected["negative"], f"SerenityFlow {sf_name} negative conditioning missing", blockers)
                    require(sf_genparams.get("init_image") == expected["init_image"], f"SerenityFlow {sf_name} reference image path missing", blockers)
                    require(
                        sf_genparams.get("width") == expected["width"] and sf_genparams.get("height") == expected["height"],
                        f"SerenityFlow {sf_name} latent dimensions missing",
                        blockers,
                    )
                    for field in (
                        "images",
                        "steps",
                        "seed",
                        "cfg",
                        "sampler",
                        "scheduler",
                        "creativity",
                        "reference_image",
                        "reference_latent_method",
                        "reference_latent_count",
                        "workflow_node_count",
                        "workflow_edge_count",
                    ):
                        require(
                            sf_genparams.get(field) == expected[field],
                            f"SerenityFlow {sf_name} {field} missing",
                            blockers,
                        )

            report["serenityflow_qwen_edit"] = {}
            for sf_name, expected in SERENITYFLOW_QWEN_EDIT_CASES.items():
                sf_request = serenityflow_template_request(expected["template"])
                sf_status, sf_data, sf_text = http_json("POST", f"{base_url}/v1/generate", sf_request)
                sf_case: dict[str, Any] = {
                    "generate": {"status": sf_status, "body": sf_data},
                    "template": str(expected["template"]),
                }
                report["serenityflow_qwen_edit"][sf_name] = sf_case
                if sf_status != 200 or not isinstance(sf_data, dict) or not sf_data.get("job_id"):
                    blockers.append(f"SerenityFlow {sf_name} generate failed HTTP {sf_status}: {sf_text}")
                    continue

                sf_job_id = str(sf_data["job_id"])
                sf_job = poll_job(base_url, sf_job_id, args.timeout)
                sf_case["job"] = sf_job
                require(sf_job.get("state") == "done", f"SerenityFlow {sf_name} job state was {sf_job.get('state')}", blockers)
                sf_png_path = Path(str(sf_job.get("output_path") or ""))
                require(sf_png_path.is_file(), f"SerenityFlow {sf_name} PNG missing: {sf_png_path}", blockers)
                if sf_png_path.is_file():
                    sf_text_chunks = read_png_text(sf_png_path)
                    sf_genparams = json.loads(sf_text_chunks.get(GENPARAMS_KEY, "{}"))
                    sf_case["png"] = {
                        "path": str(sf_png_path),
                        "idat_sha256": sf_text_chunks.get("_idat_sha256"),
                        "genparams": sf_genparams,
                    }
                    require(sf_genparams.get("workflow_source") == "comfy_api_prompt_graph", f"SerenityFlow {sf_name} workflow source missing", blockers)
                    for field in (
                        "model",
                        "prompt",
                        "negative",
                        "steps",
                        "seed",
                        "cfg",
                        "sampler",
                        "scheduler",
                        "creativity",
                        "sigma_shift",
                        "init_image",
                        "qwen_edit_conditioning_image",
                        "workflow_save_prefix",
                        "workflow_node_count",
                        "workflow_edge_count",
                    ):
                        require(
                            sf_genparams.get(field) == expected[field],
                            f"SerenityFlow {sf_name} {field} missing",
                            blockers,
                        )
                    expected_loras = expected["lora"]
                    if expected_loras:
                        require(sf_genparams.get("lora") == expected_loras, f"SerenityFlow {sf_name} LoRA metadata missing", blockers)
                    else:
                        require(sf_genparams.get("lora") == [], f"SerenityFlow {sf_name} unexpected LoRA metadata", blockers)

            ideogram_status, ideogram_data, ideogram_text = http_json(
                "POST", f"{base_url}/v1/generate", ideogram4_visual_export_request()
            )
            report["ideogram4_visual_export_generate"] = {
                "status": ideogram_status,
                "body": ideogram_data,
                "template": str(IDEOGRAM4_BASIC_TXT2IMG),
            }
            if ideogram_status != 200 or not isinstance(ideogram_data, dict) or not ideogram_data.get("job_id"):
                blockers.append(f"Ideogram4 visual export generate failed HTTP {ideogram_status}: {ideogram_text}")
            else:
                ideogram_job_id = str(ideogram_data["job_id"])
                ideogram_job = poll_job(base_url, ideogram_job_id, args.timeout)
                report["ideogram4_visual_export_job"] = ideogram_job
                require(
                    ideogram_job.get("state") == "done",
                    f"Ideogram4 visual export job state was {ideogram_job.get('state')}",
                    blockers,
                )
                ideogram_png_path = Path(str(ideogram_job.get("output_path") or ""))
                require(ideogram_png_path.is_file(), f"Ideogram4 visual export PNG missing: {ideogram_png_path}", blockers)
                if ideogram_png_path.is_file():
                    ideogram_text_chunks = read_png_text(ideogram_png_path)
                    ideogram_genparams = json.loads(ideogram_text_chunks.get(GENPARAMS_KEY, "{}"))
                    report["ideogram4_visual_export_png"] = {
                        "path": str(ideogram_png_path),
                        "idat_sha256": ideogram_text_chunks.get("_idat_sha256"),
                        "genparams": ideogram_genparams,
                    }
                    require(
                        ideogram_genparams.get("workflow_source") == "ideogram4_comfy_ui_export",
                        "Ideogram4 visual export workflow source missing",
                        blockers,
                    )
                    require(ideogram_genparams.get("model") == "ideogram-4-fp8", "Ideogram4 visual export model missing", blockers)
                    require(ideogram_genparams.get("prompt") == IDEOGRAM4_PROMPT, "Ideogram4 visual export prompt missing", blockers)
                    require(
                        ideogram_genparams.get("width") == 1024 and ideogram_genparams.get("height") == 1024,
                        "Ideogram4 visual export latent dimensions missing",
                        blockers,
                    )
                    require(ideogram_genparams.get("images") == 1, "Ideogram4 visual export batch size missing", blockers)
                    require(ideogram_genparams.get("seed") == 424242, "Ideogram4 visual export seed missing", blockers)
                    require(ideogram_genparams.get("steps") == 48, "Ideogram4 visual export quality steps missing", blockers)
                    require(ideogram_genparams.get("sampler") == "euler", "Ideogram4 visual export sampler missing", blockers)
                    require(ideogram_genparams.get("scheduler") == "simple", "Ideogram4 visual export scheduler missing", blockers)
                    require(ideogram_genparams.get("sigma_shift") == 5, "Ideogram4 visual export AuraFlow shift missing", blockers)
                    require(ideogram_genparams.get("cfg") == 7, "Ideogram4 visual export DualModelGuider cfg missing", blockers)
                    require(ideogram_genparams.get("cfg_override") == 3, "Ideogram4 visual export CFGOverride missing", blockers)
                    require(
                        ideogram_genparams.get("cfg_override_start_percent") == 0.7,
                        "Ideogram4 visual export CFGOverride start missing",
                        blockers,
                    )
                    require(
                        ideogram_genparams.get("cfg_override_end_percent") == 1,
                        "Ideogram4 visual export CFGOverride end missing",
                        blockers,
                    )
                    require(ideogram_genparams.get("workflow_node_count") == 28, "Ideogram4 visual export node count missing", blockers)
                    require(ideogram_genparams.get("workflow_edge_count") == 16, "Ideogram4 visual export edge count missing", blockers)
        except Exception as exc:
            blockers.append(str(exc))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            report["daemon_returncode"] = proc.returncode
    report["ready"] = not blockers
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write-readiness", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.daemon.is_file():
        raise SystemExit(f"[workflow-graph-product] FAIL daemon missing: {args.daemon}; run `pixi run build-daemon`")
    report = run(args)
    if args.write_readiness:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[workflow-graph-product] wrote readiness report: {args.write_readiness}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        print("[workflow-graph-product] FAIL")
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
        print(f"[workflow-graph-product] daemon log: {report['log_path']}")
        return 2
    job = report["job"]
    png = report["png"]
    print("[workflow-graph-product] PASS")
    print(f"  job_id: {job['id']}")
    print(f"  png: {png['path']}")
    print(f"  idat_sha256: {png['idat_sha256']}")
    print("  unsupported_node: HTTP 501")
    print("  unsupported_comfy_api_node: HTTP 501")
    print("  wrong_type_link: HTTP 501")
    print("  lora_clip_unsupported: HTTP 501")
    print(f"  img2img_job_id: {report['img2img_job']['id']}")
    print(f"  img2img_png: {report['img2img_png']['path']}")
    print(f"  lora_job_id: {report['lora_job']['id']}")
    print(f"  lora_png: {report['lora_png']['path']}")
    print(f"  zimage_lora_alias_job_id: {report['zimage_lora_alias_job']['id']}")
    print(f"  zimage_lora_alias_png: {report['zimage_lora_alias_png']['path']}")
    print(f"  mask_job_id: {report['mask_job']['id']}")
    print(f"  mask_png: {report['mask_png']['path']}")
    print(f"  outpaint_preprocess_job_id: {report['outpaint_preprocess_job']['id']}")
    print(f"  outpaint_preprocess_png: {report['outpaint_preprocess_png']['path']}")
    print(f"  basic_scheduler_job_id: {report['basic_scheduler_job']['id']}")
    print(f"  basic_scheduler_png: {report['basic_scheduler_png']['path']}")
    print(f"  comfy_api_job_id: {report['comfy_api_job']['id']}")
    print(f"  comfy_api_png: {report['comfy_api_png']['path']}")
    print(f"  outpaint_threshold_api_job_id: {report['outpaint_threshold_api_job']['id']}")
    print(f"  outpaint_threshold_api_png: {report['outpaint_threshold_api_png']['path']}")
    print(f"  inpaint_conditioning_api_job_id: {report['inpaint_conditioning_api_job']['id']}")
    print(f"  inpaint_conditioning_api_png: {report['inpaint_conditioning_api_png']['path']}")
    print(f"  inpaint_conditioning_no_noise_mask_api_job_id: {report['inpaint_conditioning_no_noise_mask_api_job']['id']}")
    print(f"  inpaint_conditioning_no_noise_mask_api_png: {report['inpaint_conditioning_no_noise_mask_api_png']['path']}")
    for sf_name, sf_case in report["serenityflow_t2i"].items():
        print(f"  serenityflow_{sf_name}_job_id: {sf_case['job']['id']}")
        print(f"  serenityflow_{sf_name}_png: {sf_case['png']['path']}")
    for sf_name, sf_case in report["serenityflow_edit"].items():
        print(f"  serenityflow_{sf_name}_job_id: {sf_case['job']['id']}")
        print(f"  serenityflow_{sf_name}_png: {sf_case['png']['path']}")
    for sf_name, sf_case in report["serenityflow_qwen_edit"].items():
        print(f"  serenityflow_{sf_name}_job_id: {sf_case['job']['id']}")
        print(f"  serenityflow_{sf_name}_png: {sf_case['png']['path']}")
    print(f"  ideogram4_visual_export_job_id: {report['ideogram4_visual_export_job']['id']}")
    print(f"  ideogram4_visual_export_png: {report['ideogram4_visual_export_png']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
