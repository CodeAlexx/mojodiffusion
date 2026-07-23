"use strict";
/**
 * Compositor — SerenityFlow Canvas v2
 *
 * Analyzes canvas state, auto-detects generation mode, flattens layers,
 * builds the generation payload. User never picks a mode manually.
 */
// ── Generation Mode ──
var GenMode = {
    Txt2Img: 'txt2img',
    Img2Img: 'img2img',
    Inpaint: 'inpaint',
    Outpaint: 'outpaint',
    Regional: 'regional',
    VideoInpaint: 'video_inpaint',
};
// ── Compositor ──
var Compositor = (function () {
    'use strict';
    // ── Mode Detection ──
    function detectMode(ctx) {
        var hasDrawContent = false;
        var hasMask = false;
        var hasGuidance = false;
        var hasVideoMasks = false;
        ctx.canvasLayers.forEach(function (cl) {
            if (!cl.data.visible)
                return;
            switch (cl.data.type) {
                case 'draw':
                    if (cl.konvaLayer.getChildren().length > 0)
                        hasDrawContent = true;
                    break;
                case 'mask':
                    if (cl.konvaLayer.getChildren().length > 0)
                        hasMask = true;
                    break;
                case 'guidance':
                    var gd = cl.data;
                    if (gd.positivePrompt.trim() && cl.konvaLayer.getChildren().length > 0)
                        hasGuidance = true;
                    break;
            }
        });
        // Check video masks
        if (typeof CanvasVideo !== 'undefined' && CanvasVideo.isLoaded()) {
            var masks = CanvasVideo.getAllMasks();
            if (masks.size > 0)
                hasVideoMasks = true;
        }
        if (hasVideoMasks)
            return GenMode.VideoInpaint;
        if (hasGuidance)
            return GenMode.Regional;
        if (hasDrawContent && hasMask) {
            // Check for outpaint: bbox extends beyond draw content
            if (isBboxExtendsBeyondContent(ctx))
                return GenMode.Outpaint;
            return GenMode.Inpaint;
        }
        if (hasDrawContent)
            return GenMode.Img2Img;
        return GenMode.Txt2Img;
    }
    function isBboxExtendsBeyondContent(ctx) {
        // Quick check: see if any draw layer content extends to bbox edges
        // If bbox is larger than content, it's outpaint
        var bb = ctx.boundingBox;
        var bx = bb.x(), by = bb.y(), bw = bb.width(), bh = bb.height();
        var contentBounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
        var foundContent = false;
        ctx.canvasLayers.forEach(function (cl) {
            if (!cl.data.visible || cl.data.type !== 'draw')
                return;
            cl.konvaLayer.getChildren().forEach(function (child) {
                var rect = child.getClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    foundContent = true;
                    contentBounds.minX = Math.min(contentBounds.minX, rect.x);
                    contentBounds.minY = Math.min(contentBounds.minY, rect.y);
                    contentBounds.maxX = Math.max(contentBounds.maxX, rect.x + rect.width);
                    contentBounds.maxY = Math.max(contentBounds.maxY, rect.y + rect.height);
                }
            });
        });
        if (!foundContent)
            return false;
        // If bbox extends more than 32px beyond content on any side, it's outpaint
        var margin = 32;
        return (bx < contentBounds.minX - margin) ||
            (by < contentBounds.minY - margin) ||
            (bx + bw > contentBounds.maxX + margin) ||
            (by + bh > contentBounds.maxY + margin);
    }
    // ── Layer Flattening ──
    function flattenDrawLayers(ctx) {
        /**
         * Flatten all visible draw layers into a single composite image.
         * Returns base64 PNG data (without data: prefix).
         */
        return new Promise(function (resolve) {
            // Hide non-draw layers, UI, background
            ctx.uiLayer.hide();
            ctx.backgroundLayer.hide();
            ctx.canvasLayers.forEach(function (cl) {
                if (cl.data.type !== 'draw' || !cl.data.visible) {
                    cl.konvaLayer.hide();
                }
            });
            var dataURL = ctx.stage.toDataURL({
                x: ctx.boundingBox.x(),
                y: ctx.boundingBox.y(),
                width: ctx.boundingBox.width(),
                height: ctx.boundingBox.height(),
                pixelRatio: 1,
            });
            // Restore visibility
            ctx.canvasLayers.forEach(function (cl) {
                if (cl.data.visible)
                    cl.konvaLayer.show();
            });
            ctx.uiLayer.show();
            ctx.backgroundLayer.show();
            ctx.stage.batchDraw();
            resolve(dataURL.split(',')[1] || '');
        });
    }
    function flattenMaskLayers(ctx) {
        /**
         * Export each visible mask layer as a B&W PNG.
         * White = masked area, Black = unmasked.
         */
        var results = [];
        var promises = [];
        ctx.canvasLayers.forEach(function (cl) {
            if (cl.data.type !== 'mask' || !cl.data.visible)
                return;
            if (cl.konvaLayer.getChildren().length === 0)
                return;
            promises.push(new Promise(function (resolve) {
                var bb = ctx.boundingBox;
                var bw = Math.round(bb.width());
                var bh = Math.round(bb.height());
                // Export the mask layer directly so the stage background cannot
                // be flattened into an opaque full-frame mask.
                var oldPosition = { x: ctx.stage.x(), y: ctx.stage.y() };
                var oldScale = { x: ctx.stage.scaleX(), y: ctx.stage.scaleY() };
                var maskDataURL = '';
                try {
                    ctx.stage.position({ x: 0, y: 0 });
                    ctx.stage.scale({ x: 1, y: 1 });
                    ctx.stage.draw();
                    maskDataURL = cl.konvaLayer.toDataURL({
                        x: bb.x(), y: bb.y(), width: bw, height: bh, pixelRatio: 1,
                    });
                }
                finally {
                    ctx.stage.position(oldPosition);
                    ctx.stage.scale(oldScale);
                    ctx.stage.draw();
                }
                // Convert to B&W
                var img = new window.Image();
                img.onload = function () {
                    var offscreen = document.createElement('canvas');
                    offscreen.width = bw;
                    offscreen.height = bh;
                    var oc = offscreen.getContext('2d');
                    oc.clearRect(0, 0, bw, bh);
                    oc.drawImage(img, 0, 0, bw, bh);
                    var imageData = oc.getImageData(0, 0, bw, bh);
                    var d = imageData.data;
                    for (var i = 0; i < d.length; i += 4) {
                        var hasContent = d[i + 3] > 127;
                        d[i] = hasContent ? 255 : 0;
                        d[i + 1] = hasContent ? 255 : 0;
                        d[i + 2] = hasContent ? 255 : 0;
                        d[i + 3] = 255;
                    }
                    oc.putImageData(imageData, 0, 0);
                    results.push({
                        base64: offscreen.toDataURL('image/png').split(',')[1],
                        data: cl.data,
                    });
                    resolve();
                };
                img.onerror = function () { resolve(); };
                img.src = maskDataURL;
            }));
        });
        return Promise.all(promises).then(function () { return results; });
    }
    function collectGuidanceRegions(ctx) {
        var regions = [];
        ctx.canvasLayers.forEach(function (cl) {
            if (cl.data.type !== 'guidance' || !cl.data.visible)
                return;
            var gd = cl.data;
            if (!gd.positivePrompt.trim())
                return;
            if (cl.konvaLayer.getChildren().length === 0)
                return;
            var bb = ctx.boundingBox;
            // Export guidance region mask
            ctx.uiLayer.hide();
            ctx.backgroundLayer.hide();
            ctx.canvasLayers.forEach(function (other) {
                if (other !== cl)
                    other.konvaLayer.hide();
            });
            var dataURL = cl.konvaLayer.toDataURL({
                x: bb.x(), y: bb.y(),
                width: bb.width(), height: bb.height(),
                pixelRatio: 1,
            });
            ctx.canvasLayers.forEach(function (other) {
                if (other.data.visible)
                    other.konvaLayer.show();
            });
            ctx.uiLayer.show();
            ctx.backgroundLayer.show();
            regions.push({
                maskBase64: dataURL.split(',')[1] || '',
                data: gd,
            });
        });
        return regions;
    }
    function collectControlInputs(ctx) {
        var controls = [];
        ctx.canvasLayers.forEach(function (cl) {
            if (cl.data.type !== 'control' || !cl.data.visible)
                return;
            var cd = cl.data;
            if (!cd.refImageSrc)
                return;
            var base64 = cd.refImageSrc.split(',')[1] || '';
            controls.push({ refBase64: base64, data: cd });
        });
        return controls;
    }
    function applyAdjustments(ctx, imageBase64) {
        /**
         * Apply visible adjustment layers to the composite image.
         * Returns adjusted base64 PNG.
         */
        var adjustments = ctx.canvasLayers.filter(function (cl) {
            return cl.data.type === 'adjustment' && cl.data.visible;
        });
        if (adjustments.length === 0)
            return Promise.resolve(imageBase64);
        return new Promise(function (resolve) {
            var img = new window.Image();
            img.onload = function () {
                var c = document.createElement('canvas');
                c.width = img.width;
                c.height = img.height;
                var gc = c.getContext('2d');
                // Stack adjustments as CSS filters
                var filters = [];
                adjustments.forEach(function (cl) {
                    var ad = cl.data;
                    if (ad.brightness !== 0)
                        filters.push('brightness(' + (1 + ad.brightness) + ')');
                    if (ad.contrast !== 0)
                        filters.push('contrast(' + (1 + ad.contrast) + ')');
                    if (ad.saturation !== 0)
                        filters.push('saturate(' + (1 + ad.saturation) + ')');
                    if (ad.temperature !== 0)
                        filters.push('hue-rotate(' + (ad.temperature * 30) + 'deg)');
                });
                gc.filter = filters.length > 0 ? filters.join(' ') : 'none';
                gc.drawImage(img, 0, 0);
                gc.filter = 'none';
                resolve(c.toDataURL('image/png').split(',')[1]);
            };
            img.onerror = function () { resolve(imageBase64); };
            img.src = 'data:image/png;base64,' + imageBase64;
        });
    }
    // ── Build Payload ──
    function compose(ctx) {
        var mode = detectMode(ctx);
        var bb = ctx.boundingBox;
        var isVideo = ctx.genState.arch === 'ltxv' || ctx.genState.arch === 'wan' || ctx.genState.arch === 'bernini';
        var seed = ctx.genState.seed === -1
            ? Math.floor(Math.random() * 4294967296)
            : ctx.genState.seed;
        var payload = {
            mode: mode,
            model: ctx.genState.model || '',
            positive_prompt: ctx.genState.prompt,
            negative_prompt: '',
            image: null,
            masks: [],
            guidance_regions: [],
            control_inputs: [],
            reference_images: [],
            lanpaint: { enabled: false, thinking_steps: 5 },
            params: {
                seed: seed,
                steps: ctx.genState.steps,
                cfg: ctx.genState.cfg,
                guidance: ctx.genState.guidance,
                sampler: 'euler',
                scheduler: 'normal',
                width: Math.round(bb.width()),
                height: Math.round(bb.height()),
                denoise: ctx.genState.denoise,
            },
            batch_size: 1,
        };
        if (isVideo) {
            payload.params.frames = ctx.genState.frames;
            payload.params.fps = ctx.genState.fps;
        }
        // txt2img: just prompt + params
        if (mode === 'txt2img') {
            return Promise.resolve(payload);
        }
        // Everything else needs the composite image
        return flattenDrawLayers(ctx)
            .then(function (drawBase64) {
            return applyAdjustments(ctx, drawBase64);
        })
            .then(function (adjustedBase64) {
            if (!adjustedBase64)
                return payload;
            return ctx.uploadImage(adjustedBase64).then(function (imageId) {
                payload.image = imageId;
                return payload;
            });
        })
            .then(function (p) {
            // Masks
            return flattenMaskLayers(ctx).then(function (maskResults) {
                var maskUploads = maskResults.map(function (mr) {
                    return ctx.uploadImage(mr.base64).then(function (maskId) {
                        p.masks.push({
                            id: 'mask_' + mr.data.id,
                            imageId: maskId,
                            denoise: mr.data.denoiseStrength,
                            noiseLevel: mr.data.noiseLevel,
                        });
                    });
                });
                return Promise.all(maskUploads).then(function () { return p; });
            });
        })
            .then(function (p) {
            // Guidance regions
            var regions = collectGuidanceRegions(ctx);
            var regionUploads = regions.map(function (r) {
                return ctx.uploadImage(r.maskBase64).then(function (maskId) {
                    p.guidance_regions.push({
                        maskId: maskId,
                        positive: r.data.positivePrompt,
                        negative: r.data.negativePrompt,
                        refs: r.data.referenceImages,
                    });
                });
            });
            return Promise.all(regionUploads).then(function () { return p; });
        })
            .then(function (p) {
            // Control inputs
            var controls = collectControlInputs(ctx);
            var controlUploads = controls.map(function (c) {
                return ctx.uploadImage(c.refBase64).then(function (imageId) {
                    p.control_inputs.push({
                        imageId: imageId,
                        model: c.data.controlModel,
                        weight: c.data.weight,
                        stepRange: [c.data.beginStep, c.data.endStep],
                    });
                });
            });
            return Promise.all(controlUploads).then(function () { return p; });
        })
            .then(function (p) {
            // Video inpaint: attach per-frame mask data
            if (mode === 'video_inpaint' && typeof CanvasVideo !== 'undefined') {
                p.video_masks = CanvasVideo.getMasksAsArray();
            }
            return p;
        });
    }
    // ── Batch Generation ──
    function composeBatch(ctx, batchSize) {
        return compose(ctx).then(function (basePayload) {
            var payloads = [];
            for (var i = 0; i < batchSize; i++) {
                var p = JSON.parse(JSON.stringify(basePayload));
                if (i > 0) {
                    // Different seed for each batch item
                    p.params.seed = Math.floor(Math.random() * 4294967296);
                }
                p.batch_size = 1;
                payloads.push(p);
            }
            return payloads;
        });
    }
    // ── Public API ──
    return {
        detectMode: detectMode,
        compose: compose,
        composeBatch: composeBatch,
        flattenDrawLayers: flattenDrawLayers,
        flattenMaskLayers: flattenMaskLayers,
        collectGuidanceRegions: collectGuidanceRegions,
        collectControlInputs: collectControlInputs,
        applyAdjustments: applyAdjustments,
    };
})();
//# sourceMappingURL=canvas-compositor.js.map
