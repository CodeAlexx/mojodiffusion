"use strict";
/**
 * Reference Images & Smart Features — SerenityFlow Canvas v2
 *
 * Reference image panel, smart crop, context menu, clipboard paste,
 * zoom controls, layer thumbnails, solo mode, layer groups.
 */
// ── Reference Image Panel ──
var CanvasRefImages = (function () {
    'use strict';
    var _refs = [];
    var _panelEl = null;
    var _nextId = 0;
    var _compatibility = { supported: false, reason: 'Reference-image generation is not advertised by the selected backend' };
    function notifyStyleSelected(ref) {
        if (!ref || ref.method !== 'style')
            return;
        document.dispatchEvent(new CustomEvent('sf-style-reference-selected', {
            detail: { ref: JSON.parse(JSON.stringify(ref)) }
        }));
    }
    function add(src) {
        var ref = {
            id: 'ref_' + (++_nextId),
            src: src,
            weight: 1.0,
            method: 'style',
            model: 'ip-adapter',
            stepRange: [0, 1],
        };
        _refs.push(ref);
        renderPanel();
        notifyStyleSelected(ref);
        return ref;
    }
    function remove(id) {
        _refs = _refs.filter(function (r) { return r.id !== id; });
        renderPanel();
    }
    function getAll() { return _refs; }
    function serialize() { return JSON.parse(JSON.stringify(_refs)); }
    function restore(refs) {
        _refs = Array.isArray(refs) ? refs.map(function (ref) {
            var range = Array.isArray(ref.stepRange) ? ref.stepRange : [0, 1];
            return {
                id: String(ref.id || ('ref_' + (++_nextId))),
                src: String(ref.src || ''),
                weight: Math.max(0, Math.min(2, Number(ref.weight == null ? 1 : ref.weight))),
                method: String(ref.method || 'style'),
                model: String(ref.model || 'ip-adapter'),
                stepRange: [Math.max(0, Math.min(1, Number(range[0] || 0))), Math.max(0, Math.min(1, Number(range[1] == null ? 1 : range[1])))],
                imageName: ref.imageName || ''
            };
        }).filter(function (ref) { return !!ref.src; }) : [];
        _nextId = Math.max(_nextId, _refs.length);
        renderPanel();
    }
    function getForPayload() {
        return _refs.map(function (r) {
            return { imageId: r.src, weight: r.weight, method: r.method, model: r.model, stepRange: r.stepRange.slice() };
        });
    }
    function setCompatibility(supported, reason) {
        _compatibility = { supported: supported === true, reason: String(reason || '') };
        renderPanel();
    }
    function showPanel() {
        if (!_panelEl) {
            _panelEl = document.createElement('div');
            _panelEl.id = 'ref-images-panel';
            _panelEl.className = 'ref-panel';
            var rightPanel = document.querySelector('.cv-right');
            if (rightPanel) {
                rightPanel.insertBefore(_panelEl, rightPanel.firstChild);
            }
        }
        _panelEl.style.display = 'block';
        renderPanel();
    }
    function renderPanel() {
        if (!_panelEl)
            return;
        var html = '<div class="ref-header">' +
            '<span class="ref-title">Reference Images</span>' +
            '<button class="ref-add-btn" id="ref-add-btn"' + (_compatibility.supported ? '' : ' title="Local project reference; generation is capability-gated"') + '>+ Add</button>' +
            '<input type="file" id="ref-file-input" accept="image/*" style="display:none" multiple>' +
            '</div>';
        if (!_compatibility.supported) {
            html += '<div class="ref-capability-note">Local project only: ' + escapeHtml(_compatibility.reason) + '</div>';
        }
        if (_refs.length === 0) {
            html += '<div class="ref-empty">No reference images. Click + Add or drag & drop.</div>';
        }
        _refs.forEach(function (ref) {
            html += '<div class="ref-item" data-id="' + ref.id + '">' +
                '<img class="ref-thumb" src="' + ref.src + '">' +
                '<div class="ref-controls">' +
                '<div class="ref-row"><span>Weight</span><input type="range" class="ref-weight" data-id="' + ref.id + '" min="0" max="2" step="0.05" value="' + ref.weight + '"><span class="ref-weight-val">' + ref.weight.toFixed(2) + '</span></div>' +
                '<div class="ref-row"><span>Method</span><select class="ref-method" data-id="' + ref.id + '">' +
                '<option value="style"' + (ref.method === 'style' ? ' selected' : '') + '>Style</option>' +
                '<option value="face"' + (ref.method === 'face' ? ' selected' : '') + '>Face</option>' +
                '<option value="composition"' + (ref.method === 'composition' ? ' selected' : '') + '>Composition</option>' +
                '<option value="full"' + (ref.method === 'full' ? ' selected' : '') + '>Full</option>' +
                '</select></div>' +
                '<div class="ref-row"><span>Range</span><input type="range" class="ref-step-start" data-id="' + ref.id + '" min="0" max="1" step="0.05" value="' + ref.stepRange[0] + '"><span class="ref-range-val">' + ref.stepRange[0].toFixed(2) + '</span></div>' +
                '<div class="ref-row"><span>to</span><input type="range" class="ref-step-end" data-id="' + ref.id + '" min="0" max="1" step="0.05" value="' + ref.stepRange[1] + '"><span class="ref-range-val">' + ref.stepRange[1].toFixed(2) + '</span></div>' +
                '<button class="ref-remove" data-id="' + ref.id + '" title="Remove">&times;</button>' +
                '</div>' +
                '</div>';
        });
        _panelEl.innerHTML = html;
        bindPanelEvents();
    }
    function escapeHtml(value) {
        var div = document.createElement('div');
        div.textContent = value;
        return div.innerHTML;
    }
    function bindPanelEvents() {
        var addBtn = document.getElementById('ref-add-btn');
        var fileInput = document.getElementById('ref-file-input');
        if (addBtn && fileInput) {
            addBtn.addEventListener('click', function () { fileInput.click(); });
            fileInput.addEventListener('change', function () {
                if (!fileInput.files)
                    return;
                Array.from(fileInput.files).forEach(function (file) {
                    var reader = new FileReader();
                    reader.onload = function (ev) { add(ev.target.result); };
                    reader.readAsDataURL(file);
                });
                fileInput.value = '';
            });
        }
        // Weight sliders
        document.querySelectorAll('.ref-weight').forEach(function (el) {
            var input = el;
            input.addEventListener('input', function () {
                var targetId = input.dataset.id;
                var ref = _refs.find(function (r) { return r.id === targetId; });
                if (ref) {
                    ref.weight = parseFloat(input.value);
                    var valEl = input.parentElement.querySelector('.ref-weight-val');
                    if (valEl)
                        valEl.textContent = ref.weight.toFixed(2);
                }
            });
        });
        // Method selects
        document.querySelectorAll('.ref-method').forEach(function (el) {
            var select = el;
            select.addEventListener('change', function () {
                var targetId = select.dataset.id;
                var ref = _refs.find(function (r) { return r.id === targetId; });
                if (ref) {
                    ref.method = select.value;
                    notifyStyleSelected(ref);
                }
            });
        });
        document.querySelectorAll('.ref-thumb').forEach(function (el) {
            el.addEventListener('click', function () {
                var item = this.closest('.ref-item');
                var ref = item && _refs.find(function (candidate) { return candidate.id === item.dataset.id; });
                notifyStyleSelected(ref);
            });
        });
        document.querySelectorAll('.ref-step-start, .ref-step-end').forEach(function (el) {
            var input = el;
            input.addEventListener('input', function () {
                var ref = _refs.find(function (candidate) { return candidate.id === input.dataset.id; });
                if (!ref)
                    return;
                var index = input.classList.contains('ref-step-start') ? 0 : 1;
                ref.stepRange[index] = parseFloat(input.value);
                if (ref.stepRange[0] > ref.stepRange[1]) {
                    ref.stepRange[1 - index] = ref.stepRange[index];
                    var peer = input.parentElement.parentElement.querySelector(index === 0 ? '.ref-step-end' : '.ref-step-start');
                    if (peer)
                        peer.value = String(ref.stepRange[1 - index]);
                }
                var label = input.parentElement.querySelector('.ref-range-val');
                if (label)
                    label.textContent = ref.stepRange[index].toFixed(2);
            });
        });
        // Remove buttons
        document.querySelectorAll('.ref-remove').forEach(function (el) {
            el.addEventListener('click', function () {
                remove(this.dataset.id);
            });
        });
    }
    return {
        add: add,
        remove: remove,
        getAll: getAll,
        serialize: serialize,
        restore: restore,
        getForPayload: getForPayload,
        setCompatibility: setCompatibility,
        showPanel: showPanel,
    };
})();
// ── Context Menu ──
var CanvasContextMenu = (function () {
    'use strict';
    var _menuEl = null;
    var _ctx = null;
    function show(x, y, ctx) {
        _ctx = ctx;
        if (!_menuEl) {
            _menuEl = document.createElement('div');
            _menuEl.id = 'canvas-ctx-menu';
            _menuEl.className = 'canvas-ctx-menu';
            document.body.appendChild(_menuEl);
        }
        var al = ctx.getActiveLayer();
        var type = al ? al.data.type : null;
        var items = [];
        if (al) {
            items.push({ label: 'Rename Layer', action: 'rename' });
            items.push({ label: 'Transform Layer', action: 'transform' });
            items.push({ label: 'Duplicate Layer', action: 'duplicate' });
            items.push({ label: 'Bring Forward', action: 'forward' });
            items.push({ label: 'Send Backward', action: 'backward' });
            items.push({ label: al.data.locked ? 'Unlock Layer' : 'Lock Layer', action: 'togglelock' });
            if (type === 'draw')
                items.push({ label: al.data.lockTransparency ? 'Unlock Transparency' : 'Lock Transparency', action: 'toggle_transparency' });
            items.push({ label: 'Copy Layer', action: 'copy' });
            items.push({ label: 'Save Layer as PNG', action: 'save_layer' });
            items.push({ label: 'Crop Bbox to Layer', action: 'crop_layer' });
            items.push({ label: 'Run Current Workflow', action: 'run_workflow', divider: true });
            items.push({ label: 'Add Adjustment / Filters', action: 'add_adjustment' });
        }
        if (type === 'draw' || type === 'text' || type === 'adjustment') {
            if (type === 'draw') {
                items.push({ label: 'Convert to Inpaint Mask', action: 'convert_mask' });
                items.push({ label: 'Convert to Regional Guidance', action: 'convert_guidance' });
            }
            items.push({ label: 'Merge Down', action: 'merge' });
            items.push({ label: 'Flatten Visible', action: 'flatten' });
            items.push({ label: 'Delete Layer', action: 'delete', divider: true });
        }
        else if (type === 'mask') {
            items.push({ label: 'Convert to Raster Layer', action: 'convert_draw' });
            items.push({ label: 'Invert Mask', action: 'invert_mask' });
            items.push({ label: 'Expand Mask (5px)', action: 'expand_mask' });
            items.push({ label: 'Shrink Mask (5px)', action: 'shrink_mask' });
            items.push({ label: 'Feather Mask (3px)', action: 'feather_mask' });
            items.push({ label: 'Delete Layer', action: 'delete', divider: true });
        }
        else if (al) {
            items.push({ label: 'Delete Layer', action: 'delete', divider: true });
        }
        else {
            items.push({ label: 'Paste from Clipboard', action: 'paste' });
            items.push({ label: 'Import Image', action: 'import' });
            items.push({ label: 'Add Text Layer', action: 'add_text' });
        }
        var html = '';
        items.forEach(function (item) {
            if (item.divider)
                html += '<div class="ctx-divider"></div>';
            html += '<div class="ctx-item" data-action="' + item.action + '">' + item.label + '</div>';
        });
        _menuEl.innerHTML = html;
        _menuEl.style.left = x + 'px';
        _menuEl.style.top = y + 'px';
        _menuEl.style.display = 'block';
        // Bind actions
        _menuEl.querySelectorAll('.ctx-item').forEach(function (el) {
            el.addEventListener('click', function () {
                handleAction(this.dataset.action);
                hide();
            });
        });
        // Close on outside click
        setTimeout(function () {
            document.addEventListener('click', _outsideClick, { once: true });
        }, 0);
    }
    function _outsideClick() { hide(); }
    function hide() {
        if (_menuEl)
            _menuEl.style.display = 'none';
    }
    function handleAction(action) {
        if (!_ctx)
            return;
        var al = _ctx.getActiveLayer();
        switch (action) {
            case 'rename':
                _ctx.renameActiveLayer();
                break;
            case 'transform':
                _ctx.transformActiveLayer();
                break;
            case 'duplicate':
                _ctx.duplicateActiveLayer();
                break;
            case 'forward':
                _ctx.moveActiveLayerForward();
                break;
            case 'backward':
                _ctx.moveActiveLayerBackward();
                break;
            case 'delete':
                _ctx.deleteActiveLayer();
                break;
            case 'merge':
                if (al) {
                    // mergeDown is on CanvasTab — dispatch event
                    document.dispatchEvent(new CustomEvent('sf-merge-down', { detail: al.data.id }));
                }
                break;
            case 'flatten':
                _ctx.flattenVisible();
                break;
            case 'togglelock':
                _ctx.toggleActiveLayerLock();
                break;
            case 'toggle_transparency':
                _ctx.toggleActiveLayerTransparency();
                break;
            case 'add_adjustment':
                _ctx.addLayer('Adjustment', 'adjustment');
                break;
            case 'copy':
                _ctx.copyActiveLayer();
                break;
            case 'save_layer':
                _ctx.saveActiveLayer();
                break;
            case 'crop_layer':
                _ctx.cropToActiveLayer();
                break;
            case 'run_workflow':
                _ctx.runActiveLayerWorkflow();
                break;
            case 'convert_mask':
                _ctx.convertActiveLayerTo('mask');
                break;
            case 'convert_guidance':
                _ctx.convertActiveLayerTo('guidance');
                break;
            case 'convert_draw':
                _ctx.convertActiveLayerTo('draw');
                break;
            case 'invert_mask':
                invertMask(_ctx);
                break;
            case 'expand_mask':
                morphMask(_ctx, 'dilate', 5);
                break;
            case 'shrink_mask':
                morphMask(_ctx, 'erode', 5);
                break;
            case 'feather_mask':
                morphMask(_ctx, 'blur', 3);
                break;
            case 'paste':
                document.dispatchEvent(new CustomEvent('sf-paste'));
                break;
            case 'import':
                document.dispatchEvent(new CustomEvent('sf-import-image'));
                break;
            case 'add_text':
                _ctx.addLayer('Text', 'text');
                break;
        }
    }
    function invertMask(ctx) {
        var al = ctx.getActiveLayer();
        if (!al || al.data.type !== 'mask')
            return;
        var bb = ctx.boundingBox;
        var bw = Math.round(bb.width()), bh = Math.round(bb.height());
        var url = al.konvaLayer.toDataURL({ x: bb.x(), y: bb.y(), width: bw, height: bh, pixelRatio: 1 });
        var img = new window.Image();
        img.onload = function () {
            var c = document.createElement('canvas');
            c.width = bw;
            c.height = bh;
            var gc = c.getContext('2d');
            gc.drawImage(img, 0, 0);
            var imageData = gc.getImageData(0, 0, bw, bh);
            var d = imageData.data;
            for (var i = 0; i < d.length; i += 4) {
                d[i + 3] = 255 - d[i + 3]; // Invert alpha
            }
            gc.putImageData(imageData, 0, 0);
            var resultImg = new window.Image();
            resultImg.onload = function () {
                al.konvaLayer.destroyChildren();
                al.konvaLayer.add(new Konva.Image({ image: resultImg, x: bb.x(), y: bb.y(), width: bw, height: bh, listening: false }));
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            };
            resultImg.src = c.toDataURL();
        };
        img.src = url;
    }
    function morphMask(ctx, op, radius) {
        var al = ctx.getActiveLayer();
        if (!al || al.data.type !== 'mask')
            return;
        var bb = ctx.boundingBox;
        var bw = Math.round(bb.width()), bh = Math.round(bb.height());
        var url = al.konvaLayer.toDataURL({ x: bb.x(), y: bb.y(), width: bw, height: bh, pixelRatio: 1 });
        var img = new window.Image();
        img.onload = function () {
            var c = document.createElement('canvas');
            c.width = bw;
            c.height = bh;
            var gc = c.getContext('2d');
            gc.drawImage(img, 0, 0);
            if (op === 'blur') {
                // Feather: multiple gaussian-like passes
                gc.filter = 'blur(' + radius + 'px)';
                gc.drawImage(c, 0, 0);
                gc.filter = 'none';
            }
            else if (op === 'dilate') {
                // Expand: draw slightly offset copies
                for (var dx = -radius; dx <= radius; dx++) {
                    for (var dy = -radius; dy <= radius; dy++) {
                        if (dx * dx + dy * dy <= radius * radius) {
                            gc.drawImage(c, dx, dy);
                        }
                    }
                }
            }
            else if (op === 'erode') {
                // Shrink: invert, dilate, invert
                var imageData = gc.getImageData(0, 0, bw, bh);
                var d = imageData.data;
                for (var i = 0; i < d.length; i += 4)
                    d[i + 3] = 255 - d[i + 3];
                gc.putImageData(imageData, 0, 0);
                var tmpC = document.createElement('canvas');
                tmpC.width = bw;
                tmpC.height = bh;
                var tc = tmpC.getContext('2d');
                tc.drawImage(c, 0, 0);
                for (var dx2 = -radius; dx2 <= radius; dx2++) {
                    for (var dy2 = -radius; dy2 <= radius; dy2++) {
                        if (dx2 * dx2 + dy2 * dy2 <= radius * radius) {
                            gc.drawImage(tmpC, dx2, dy2);
                        }
                    }
                }
                imageData = gc.getImageData(0, 0, bw, bh);
                d = imageData.data;
                for (var j = 0; j < d.length; j += 4)
                    d[j + 3] = 255 - d[j + 3];
                gc.putImageData(imageData, 0, 0);
            }
            var resultImg = new window.Image();
            resultImg.onload = function () {
                al.konvaLayer.destroyChildren();
                al.konvaLayer.add(new Konva.Image({ image: resultImg, x: bb.x(), y: bb.y(), width: bw, height: bh, listening: false }));
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            };
            resultImg.src = c.toDataURL();
        };
        img.src = url;
    }
    return {
        show: show,
        hide: hide,
    };
})();
// ── Smart Crop ──
var SmartCrop = (function () {
    'use strict';
    function checkBboxCutsObjects(ctx) {
        // Check if SAM instances exist and bbox cuts through any
        if (typeof CanvasSAM === 'undefined')
            return { cuts: false };
        var instances = CanvasSAM.getInstances();
        if (instances.length === 0)
            return { cuts: false };
        var bb = ctx.boundingBox;
        var bx = bb.x(), by = bb.y(), bw = bb.width(), bh = bb.height();
        var cuts = false;
        var unionBbox = { x: bx, y: by, w: bw, h: bh };
        instances.forEach(function (inst) {
            var ib = inst.bbox;
            // Object partially inside bbox
            var inside = ib.x < bx + bw && ib.x + ib.width > bx &&
                ib.y < by + bh && ib.y + ib.height > by;
            var fullyInside = ib.x >= bx && ib.y >= by &&
                ib.x + ib.width <= bx + bw &&
                ib.y + ib.height <= by + bh;
            if (inside && !fullyInside) {
                cuts = true;
                // Expand union to include this object
                var minX = Math.min(unionBbox.x, ib.x);
                var minY = Math.min(unionBbox.y, ib.y);
                var maxX = Math.max(unionBbox.x + unionBbox.w, ib.x + ib.width);
                var maxY = Math.max(unionBbox.y + unionBbox.h, ib.y + ib.height);
                unionBbox = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
            }
        });
        return { cuts: cuts, suggestion: cuts ? unionBbox : undefined };
    }
    function snapToObjects(ctx) {
        var result = checkBboxCutsObjects(ctx);
        if (!result.cuts || !result.suggestion)
            return;
        var s = result.suggestion;
        ctx.boundingBox.x(s.x);
        ctx.boundingBox.y(s.y);
        ctx.boundingBox.width(s.w);
        ctx.boundingBox.height(s.h);
    }
    return {
        checkBboxCutsObjects: checkBboxCutsObjects,
        snapToObjects: snapToObjects,
    };
})();
// ── Zoom Controls ──
var CanvasZoom = (function () {
    'use strict';
    var _displayEl = null;
    function fitToScreen(ctx) {
        var bb = ctx.boundingBox;
        var sw = ctx.stage.width();
        var sh = ctx.stage.height();
        var pad = 80;
        var scale = Math.min((sw - pad * 2) / bb.width(), (sh - pad * 2) / bb.height(), 2);
        ctx.stage.scale({ x: scale, y: scale });
        ctx.stage.position({
            x: sw / 2 - (bb.x() + bb.width() / 2) * scale,
            y: sh / 2 - (bb.y() + bb.height() / 2) * scale,
        });
        ctx.stage.batchDraw();
        updateDisplay(ctx);
    }
    function zoomTo100(ctx) {
        var bb = ctx.boundingBox;
        var sw = ctx.stage.width();
        var sh = ctx.stage.height();
        ctx.stage.scale({ x: 1, y: 1 });
        ctx.stage.position({
            x: sw / 2 - (bb.x() + bb.width() / 2),
            y: sh / 2 - (bb.y() + bb.height() / 2),
        });
        ctx.stage.batchDraw();
        updateDisplay(ctx);
    }
    function getZoomPercent(ctx) {
        if (!ctx || !ctx.stage)
            return 100;
        return Math.round(ctx.stage.scaleX() * 100);
    }
    function updateDisplay(ctx) {
        if (!_displayEl) {
            _displayEl = document.getElementById('cv-zoom-display');
        }
        if (_displayEl) {
            _displayEl.textContent = getZoomPercent(ctx) + '%';
        }
    }
    return {
        fitToScreen: fitToScreen,
        zoomTo100: zoomTo100,
        getZoomPercent: getZoomPercent,
        updateDisplay: updateDisplay,
    };
})();
// ── Layer Thumbnails ──
var LayerThumbnails = (function () {
    'use strict';
    var THUMB_SIZE = 36;
    function generateThumbnail(cl) {
        try {
            return cl.konvaLayer.toDataURL({
                pixelRatio: THUMB_SIZE / Math.max(cl.konvaLayer.width() || 100, 1),
            });
        }
        catch (e) {
            return '';
        }
    }
    function getSize() { return THUMB_SIZE; }
    return {
        generateThumbnail: generateThumbnail,
        getSize: getSize,
    };
})();
// ── Layer Solo Mode ──
var LayerSolo = (function () {
    'use strict';
    var _soloActive = false;
    var _soloLayerId = null;
    var _savedVisibility = new Map();
    function toggle(layerId, canvasLayers) {
        if (_soloActive && _soloLayerId === layerId) {
            // Un-solo: restore all visibility
            canvasLayers.forEach(function (cl) {
                var saved = _savedVisibility.get(cl.data.id);
                if (saved !== undefined) {
                    cl.data.visible = saved;
                    saved ? cl.konvaLayer.show() : cl.konvaLayer.hide();
                }
            });
            _soloActive = false;
            _soloLayerId = null;
            _savedVisibility.clear();
        }
        else {
            // Solo: save current state, hide all except target
            _savedVisibility.clear();
            canvasLayers.forEach(function (cl) {
                _savedVisibility.set(cl.data.id, cl.data.visible);
                if (cl.data.id === layerId) {
                    cl.data.visible = true;
                    cl.konvaLayer.show();
                }
                else {
                    cl.data.visible = false;
                    cl.konvaLayer.hide();
                }
            });
            _soloActive = true;
            _soloLayerId = layerId;
        }
    }
    function isSoloed(layerId) {
        return _soloActive && _soloLayerId === layerId;
    }
    function isActive() { return _soloActive; }
    return {
        toggle: toggle,
        isSoloed: isSoloed,
        isActive: isActive,
    };
})();
