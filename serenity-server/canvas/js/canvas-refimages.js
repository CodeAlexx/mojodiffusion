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
        return ref;
    }
    function remove(id) {
        _refs = _refs.filter(function (r) { return r.id !== id; });
        renderPanel();
    }
    function getAll() { return _refs; }
    function getForPayload() {
        return _refs.map(function (r) {
            return { imageId: r.src, weight: r.weight, method: r.method, model: r.model };
        });
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
            '<button class="ref-add-btn" id="ref-add-btn">+ Add</button>' +
            '<input type="file" id="ref-file-input" accept="image/*" style="display:none" multiple>' +
            '</div>';
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
                '<button class="ref-remove" data-id="' + ref.id + '" title="Remove">&times;</button>' +
                '</div>' +
                '</div>';
        });
        _panelEl.innerHTML = html;
        bindPanelEvents();
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
                if (ref)
                    ref.method = select.value;
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
        getForPayload: getForPayload,
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
        if (type === 'draw' || type === 'text') {
            items.push({ label: 'Duplicate Layer', action: 'duplicate' });
            items.push({ label: 'Delete Layer', action: 'delete' });
            items.push({ label: 'Merge Down', action: 'merge' });
            items.push({ label: 'Flatten Visible', action: 'flatten' });
            items.push({ label: al.data.locked ? 'Unlock' : 'Lock', action: 'togglelock' });
        }
        else if (type === 'mask') {
            items.push({ label: 'Invert Mask', action: 'invert_mask' });
            items.push({ label: 'Expand Mask (5px)', action: 'expand_mask' });
            items.push({ label: 'Shrink Mask (5px)', action: 'shrink_mask' });
            items.push({ label: 'Feather Mask (3px)', action: 'feather_mask' });
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
            case 'duplicate':
                _ctx.duplicateActiveLayer();
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
                if (al) {
                    al.data.locked = !al.data.locked;
                }
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
//# sourceMappingURL=canvas-refimages.js.map