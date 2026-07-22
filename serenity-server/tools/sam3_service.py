#!/usr/bin/env python3
"""Local SAM3 mask service for the Serenity Canvas.

This is an accessory segmentation process, not a diffusion backend.  The model
loads lazily on CUDA and is released after a short idle interval so it does not
hold VRAM while a Mojo generation worker runs.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import Sam3Model, Sam3Processor


MODEL_DIR = os.environ.get("SERENITY_SAM3_MODEL_DIR", "/home/alex/.serenity/models/sam3")
IDLE_UNLOAD_SECONDS = float(os.environ.get("SERENITY_SAM3_IDLE_UNLOAD_SECONDS", "8"))
DEVICE = os.environ.get("SERENITY_SAM3_DEVICE", "cuda")
COLORS = ("#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4")

app = FastAPI(title="Serenity SAM3", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model: Sam3Model | None = None
_processor: Sam3Processor | None = None
_model_lock = asyncio.Lock()
_last_used = 0.0
_unload_task: asyncio.Task[None] | None = None


def _device() -> torch.device:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM3 requested CUDA but torch.cuda.is_available() is false")
    return torch.device(DEVICE)


def _load_model() -> tuple[Sam3Model, Sam3Processor]:
    global _model, _processor
    if _model is None or _processor is None:
        dtype = torch.bfloat16 if _device().type == "cuda" else torch.float32
        _processor = Sam3Processor.from_pretrained(MODEL_DIR, local_files_only=True)
        _model = Sam3Model.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            dtype=dtype,
        ).to(_device())
        _model.eval()
    return _model, _processor


def _release_model() -> None:
    global _model, _processor
    _model = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _idle_unload(generation: float) -> None:
    await asyncio.sleep(IDLE_UNLOAD_SECONDS)
    async with _model_lock:
        if _last_used <= generation and time.monotonic() - _last_used >= IDLE_UNLOAD_SECONDS:
            _release_model()


def _schedule_unload() -> None:
    global _unload_task
    if _unload_task and not _unload_task.done():
        _unload_task.cancel()
    _unload_task = asyncio.create_task(_idle_unload(_last_used))


def _decode_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
        return image
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid image: {error}") from error


def _to_device(inputs: dict[str, Any]) -> dict[str, Any]:
    device = _device()
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _mask_png(mask: torch.Tensor) -> str:
    array = mask.detach().to(torch.uint8).mul(255).cpu().numpy()
    image = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _instances(
    result: dict[str, torch.Tensor], label: str, max_instances: int
) -> list[dict[str, Any]]:
    scores = result["scores"].detach().float().cpu()
    boxes = result["boxes"].detach().float().cpu()
    masks = result["masks"]
    order = torch.argsort(scores, descending=True)[:max_instances]
    rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(order.tolist()):
        box = boxes[source_index].tolist()
        rows.append(
            {
                "instance_id": uuid.uuid4().hex,
                "label": label,
                "confidence": float(scores[source_index]),
                "bbox": {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                },
                "color": COLORS[output_index % len(COLORS)],
                "mask_png": _mask_png(masks[source_index]),
            }
        )
    return rows


async def _segment(
    image: Image.Image,
    text: str,
    threshold: float,
    boxes: list[list[float]] | None = None,
    box_labels: list[int] | None = None,
    max_instances: int = 24,
) -> list[dict[str, Any]]:
    global _last_used
    async with _model_lock:
        try:
            model, processor = _load_model()
            kwargs: dict[str, Any] = {
                "images": image,
                "text": text,
                "return_tensors": "pt",
            }
            if boxes:
                kwargs["input_boxes"] = [boxes]
                kwargs["input_boxes_labels"] = [box_labels or [1] * len(boxes)]
            inputs = _to_device(dict(processor(**kwargs)))
            with torch.inference_mode(), torch.autocast(
                device_type=_device().type,
                dtype=torch.bfloat16,
                enabled=_device().type == "cuda",
            ):
                outputs = model(**inputs)
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=0.5,
                target_sizes=[image.size[::-1]],
            )[0]
            return _instances(result, text or "selected object", max_instances)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {error}") from error
        finally:
            _last_used = time.monotonic()
            _schedule_unload()


@app.get("/status")
async def status() -> dict[str, Any]:
    return {
        "available": os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")),
        "model": "facebook/sam3",
        "device": DEVICE,
        "loaded": _model is not None,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "modes": ["text", "points", "exemplar"],
    }


@app.post("/text")
async def text_mask(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.3),
) -> dict[str, Any]:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="text prompt is required")
    source = _decode_image(await image.read())
    return {"instances": await _segment(source, prompt.strip(), threshold)}


@app.post("/exemplar")
async def exemplar_mask(
    image: UploadFile = File(...),
    bbox: str = Form(...),
    threshold: float = Form(0.3),
) -> dict[str, Any]:
    source = _decode_image(await image.read())
    try:
        item = json.loads(bbox)
        x, y = float(item["x"]), float(item["y"])
        width, height = float(item["width"]), float(item["height"])
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid bbox JSON: {error}") from error
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="bbox width and height must be positive")
    box = [x, y, x + width, y + height]
    return {
        "instances": await _segment(
            source, "object", threshold, [box], [1], max_instances=1
        )
    }


@app.post("/points")
async def points_mask(
    image: UploadFile = File(...),
    points: str = Form(...),
    threshold: float = Form(0.3),
) -> dict[str, Any]:
    source = _decode_image(await image.read())
    try:
        items = json.loads(points)
        if not isinstance(items, list) or not items:
            raise ValueError("at least one point is required")
        radius = max(3.0, min(source.size) * 0.004)
        boxes = [
            [
                max(0.0, float(item["x"]) - radius),
                max(0.0, float(item["y"]) - radius),
                min(float(source.width), float(item["x"]) + radius),
                min(float(source.height), float(item["y"]) + radius),
            ]
            for item in items
        ]
        labels = [1 if int(item.get("label", 1)) else 0 for item in items]
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid points JSON: {error}") from error
    return {
        "instances": await _segment(
            source, "object", threshold, boxes, labels, max_instances=1
        )
    }


@app.post("/release")
async def release() -> dict[str, Any]:
    async with _model_lock:
        _release_model()
    return {"released": True}
