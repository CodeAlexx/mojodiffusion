"use strict";
/**
 * Staging Area — SerenityFlow Canvas v2
 *
 * Generated results appear here for review before committing to canvas.
 * Supports batch cycling, compare mode, partial accept, accept/reject.
 */
// ── Staging Module ──
var CanvasStaging = (function () {
    'use strict';
    var _state = {
        active: false,
        results: [],
        currentIndex: 0,
        compareMode: false,
        partialMaskMode: false,
    };
    var _stagingLayer = null;
    var _stagingImage = null;
    var _stagingBorder = null;
    var _stagingLabel = null;
    var _partialMaskLayer = null;
    var _panelEl = null;
    // Reference to original canvas state for compare mode
    var _originalSnapshot = null;
    var _originalImage = null;
    var _showingOriginal = false;
    // Partial mask drawing state
    var _partialDrawing = false;
    var _partialLine = null;
    // ── Context reference (set when staging activates) ──
    var _ctx = null;
    // ── Core ──
    function activate(results, ctx) {
        if (_state.active)
            deactivate();
        _ctx = ctx;
        _state.active = true;
        _state.results = results;
        _state.currentIndex = 0;
        _state.compareMode = false;
        _state.partialMaskMode = false;
        // Capture original canvas state for compare mode
        captureOriginal(ctx);
        // Create staging Konva layer (above content, below UI)
        if (!_stagingLayer) {
            _stagingLayer = new Konva.Layer();
            ctx.stage.add(_stagingLayer);
        }
        // Position: above content layers, below uiLayer
        _stagingLayer.moveToTop();
        ctx.uiLayer.moveToTop();
        // Create partial mask layer
        if (!_partialMaskLayer) {
            _partialMaskLayer = new Konva.Layer();
            ctx.stage.add(_partialMaskLayer);
        }
        _partialMaskLayer.moveToTop();
        ctx.uiLayer.moveToTop();
        // Show staging border
        var bb = ctx.boundingBox;
        _stagingBorder = new Konva.Rect({
            x: bb.x() - 3,
            y: bb.y() - 3,
            width: bb.width() + 6,
            height: bb.height() + 6,
            stroke: '#f59e0b',
            strokeWidth: 3,
            dash: [8, 4],
            cornerRadius: 4,
            listening: false,
            name: 'staging-border',
        });
        _stagingLayer.add(_stagingBorder);
        // Batch counter label
        _stagingLabel = new Konva.Text({
            x: bb.x(),
            y: bb.y() - 24,
            text: '',
            fontSize: 13,
            fontFamily: 'Inter, system-ui, sans-serif',
            fill: '#f59e0b',
            listening: false,
        });
        _stagingLayer.add(_stagingLabel);
        showResult(0, ctx);
        showPanel();
    }
    function deactivate() {
        _state.active = false;
        _state.results = [];
        _state.currentIndex = 0;
        _state.compareMode = false;
        _state.partialMaskMode = false;
        _showingOriginal = false;
        if (_stagingImage) {
            _stagingImage.destroy();
            _stagingImage = null;
        }
        if (_stagingBorder) {
            _stagingBorder.destroy();
            _stagingBorder = null;
        }
        if (_stagingLabel) {
            _stagingLabel.destroy();
            _stagingLabel = null;
        }
        if (_originalImage) {
            _originalImage.destroy();
            _originalImage = null;
        }
        if (_stagingLayer) {
            _stagingLayer.destroyChildren();
            _stagingLayer.batchDraw();
        }
        if (_partialMaskLayer) {
            _partialMaskLayer.destroyChildren();
            _partialMaskLayer.batchDraw();
        }
        _originalSnapshot = null;
        hidePanel();
        _ctx = null;
    }
    // ── Result Display ──
    function showResult(index, ctx) {
        if (index < 0 || index >= _state.results.length)
            return;
        _state.currentIndex = index;
        _showingOriginal = false;
        var result = _state.results[index];
        var bb = ctx.boundingBox;
        if (_stagingImage) {
            _stagingImage.destroy();
            _stagingImage = null;
        }
        if (_originalImage) {
            _originalImage.destroy();
            _originalImage = null;
        }
        if (result.isVideo) {
            // Video results: show placeholder text, actual video shown in panel
            _stagingLabel.text('Video Result ' + (index + 1) + ' of ' + _state.results.length);
            _stagingLayer.batchDraw();
            updatePanel();
            return;
        }
        var img = new window.Image();
        img.crossOrigin = 'anonymous';
        img.onload = function () {
            _stagingImage = new Konva.Image({
                image: img,
                x: bb.x(),
                y: bb.y(),
                width: bb.width(),
                height: bb.height(),
                listening: false,
                name: 'staging-result',
            });
            _stagingLayer.add(_stagingImage);
            // Keep border and label on top
            if (_stagingBorder)
                _stagingBorder.moveToTop();
            if (_stagingLabel)
                _stagingLabel.moveToTop();
            _stagingLayer.batchDraw();
        };
        img.src = result.src;
        updateLabel();
        updatePanel();
    }
    function updateLabel() {
        if (!_stagingLabel)
            return;
        var total = _state.results.length;
        var current = _state.currentIndex + 1;
        var text = 'Result ' + current + ' of ' + total;
        if (_showingOriginal)
            text = 'Original (compare)';
        if (_state.partialMaskMode)
            text += ' — Paint to accept';
        _stagingLabel.text(text);
        if (_stagingLayer)
            _stagingLayer.batchDraw();
    }
    // ── Navigation ──
    function nextResult() {
        if (!_state.active || !_ctx)
            return;
        var next = (_state.currentIndex + 1) % _state.results.length;
        showResult(next, _ctx);
    }
    function prevResult() {
        if (!_state.active || !_ctx)
            return;
        var prev = (_state.currentIndex - 1 + _state.results.length) % _state.results.length;
        showResult(prev, _ctx);
    }
    // ── Accept / Reject ──
    function accept() {
        if (!_state.active || !_ctx)
            return;
        var result = _state.results[_state.currentIndex];
        if (!result)
            return;
        if (_state.partialMaskMode && _partialMaskLayer) {
            // Partial accept: blend staged result through painted mask
            acceptPartial(_ctx, result);
        }
        else {
            // Full accept: place result on active draw layer
            acceptFull(_ctx, result);
        }
    }
    function acceptFull(ctx, result) {
        if (result.isVideo) {
            // Video: delegate to existing video overlay handler
            deactivate();
            return;
        }
        var bb = ctx.boundingBox;
        var img = new window.Image();
        img.crossOrigin = 'anonymous';
        img.onload = function () {
            var kImg = new Konva.Image({
                image: img,
                x: bb.x(),
                y: bb.y(),
                width: bb.width(),
                height: bb.height(),
                draggable: false,
                listening: false,
            });
            var activeLayer = ctx.getActiveKonvaLayer();
            if (activeLayer) {
                activeLayer.add(kImg);
                activeLayer.batchDraw();
            }
            ctx.pushHistory();
            deactivate();
        };
        img.src = result.src;
    }
    function acceptPartial(ctx, result) {
        // Composite the staged result through the partial mask onto the active layer
        var bb = ctx.boundingBox;
        var bw = Math.round(bb.width());
        var bh = Math.round(bb.height());
        // Get the partial mask
        var maskUrl = _partialMaskLayer.toDataURL({
            x: bb.x(), y: bb.y(), width: bw, height: bh, pixelRatio: 1,
        });
        var resultImg = new window.Image();
        var maskImg = new window.Image();
        var loaded = 0;
        function onBothLoaded() {
            var c = document.createElement('canvas');
            c.width = bw;
            c.height = bh;
            var gc = c.getContext('2d');
            // Draw result
            gc.drawImage(resultImg, 0, 0, bw, bh);
            // Apply mask as alpha: only keep result where mask is painted
            gc.globalCompositeOperation = 'destination-in';
            gc.drawImage(maskImg, 0, 0, bw, bh);
            gc.globalCompositeOperation = 'source-over';
            var finalImg = new window.Image();
            finalImg.onload = function () {
                var kImg = new Konva.Image({
                    image: finalImg,
                    x: bb.x(), y: bb.y(),
                    width: bw, height: bh,
                    listening: false,
                });
                var activeLayer = ctx.getActiveKonvaLayer();
                if (activeLayer) {
                    activeLayer.add(kImg);
                    activeLayer.batchDraw();
                }
                ctx.pushHistory();
                deactivate();
            };
            finalImg.src = c.toDataURL();
        }
        resultImg.onload = function () { loaded++; if (loaded === 2)
            onBothLoaded(); };
        maskImg.onload = function () { loaded++; if (loaded === 2)
            onBothLoaded(); };
        resultImg.src = result.src;
        maskImg.src = maskUrl;
    }
    function reject() {
        deactivate();
    }
    function regenerate() {
        // Signal to canvas-tab that we want a new generation with a different seed
        deactivate();
        // Dispatch custom event that canvas-tab listens for
        var event = new CustomEvent('sf-staging-regenerate');
        document.dispatchEvent(event);
    }
    // ── Compare Mode ──
    function captureOriginal(ctx) {
        var bb = ctx.boundingBox;
        try {
            ctx.uiLayer.hide();
            _originalSnapshot = ctx.stage.toDataURL({
                x: bb.x(), y: bb.y(),
                width: bb.width(), height: bb.height(),
                pixelRatio: 1,
            });
            ctx.uiLayer.show();
        }
        catch (e) {
            _originalSnapshot = null;
        }
    }
    function toggleCompare() {
        if (!_state.active || !_ctx || !_originalSnapshot)
            return;
        _state.compareMode = !_state.compareMode;
        _showingOriginal = _state.compareMode;
        if (_showingOriginal) {
            // Show original
            if (_stagingImage)
                _stagingImage.visible(false);
            if (!_originalImage) {
                var bb = _ctx.boundingBox;
                var img = new window.Image();
                img.onload = function () {
                    _originalImage = new Konva.Image({
                        image: img,
                        x: bb.x(), y: bb.y(),
                        width: bb.width(), height: bb.height(),
                        listening: false,
                        name: 'staging-original',
                    });
                    _stagingLayer.add(_originalImage);
                    if (_stagingBorder)
                        _stagingBorder.moveToTop();
                    if (_stagingLabel)
                        _stagingLabel.moveToTop();
                    _stagingLayer.batchDraw();
                };
                img.src = _originalSnapshot;
            }
            else {
                _originalImage.visible(true);
                _stagingLayer.batchDraw();
            }
        }
        else {
            // Show result
            if (_originalImage)
                _originalImage.visible(false);
            if (_stagingImage)
                _stagingImage.visible(true);
            _stagingLayer.batchDraw();
        }
        updateLabel();
        updatePanel();
    }
    // ── Partial Accept (paint mask on staged result) ──
    function togglePartialMask() {
        _state.partialMaskMode = !_state.partialMaskMode;
        if (!_state.partialMaskMode && _partialMaskLayer) {
            _partialMaskLayer.destroyChildren();
            _partialMaskLayer.batchDraw();
        }
        updateLabel();
        updatePanel();
    }
    function handlePartialMouseDown(ctx, pos) {
        if (!_state.partialMaskMode || !_partialMaskLayer)
            return;
        _partialDrawing = true;
        _partialLine = new Konva.Line({
            stroke: 'rgba(255, 255, 255, 0.8)',
            strokeWidth: ctx.getBrushSize(),
            globalCompositeOperation: 'source-over',
            lineCap: 'round',
            lineJoin: 'round',
            points: [pos.x, pos.y, pos.x, pos.y],
            listening: false,
        });
        _partialMaskLayer.add(_partialLine);
    }
    function handlePartialMouseMove(pos) {
        if (!_partialDrawing || !_partialLine)
            return;
        _partialLine.points(_partialLine.points().concat([pos.x, pos.y]));
        if (_partialMaskLayer)
            _partialMaskLayer.batchDraw();
    }
    function handlePartialMouseUp() {
        _partialDrawing = false;
        _partialLine = null;
    }
    // ── Panel UI ──
    function showPanel() {
        if (!_panelEl) {
            _panelEl = document.createElement('div');
            _panelEl.id = 'staging-panel';
            _panelEl.className = 'staging-panel';
            document.body.appendChild(_panelEl);
        }
        _panelEl.style.display = 'flex';
        updatePanel();
    }
    function hidePanel() {
        if (_panelEl)
            _panelEl.style.display = 'none';
    }
    function updatePanel() {
        if (!_panelEl)
            return;
        var total = _state.results.length;
        var current = _state.currentIndex + 1;
        var result = _state.results[_state.currentIndex];
        var html = '<div class="staging-header">' +
            '<span class="staging-title">Staging</span>' +
            '<span class="staging-counter">' + current + ' / ' + total + '</span>' +
            '</div>';
        // Video preview
        if (result && result.isVideo) {
            html += '<div class="staging-video-preview">' +
                '<video src="' + result.src + '" autoplay loop muted playsinline controls style="width:100%;border-radius:4px;"></video>' +
                '</div>';
        }
        // Navigation (batch cycling)
        if (total > 1) {
            html += '<div class="staging-nav">' +
                '<button class="staging-btn" id="stg-prev" title="Previous">&larr; Prev</button>' +
                '<button class="staging-btn" id="stg-next" title="Next">Next &rarr;</button>' +
                '</div>';
        }
        // Actions
        html += '<div class="staging-actions">' +
            '<button class="staging-btn staging-accept" id="stg-accept">Accept</button>' +
            '<button class="staging-btn staging-reject" id="stg-reject">Reject</button>' +
            '<button class="staging-btn" id="stg-regen" title="Regenerate with new seed">Regenerate</button>' +
            '</div>';
        // Mode toggles
        html += '<div class="staging-modes">' +
            '<button class="staging-btn staging-toggle' + (_state.compareMode ? ' active' : '') + '" id="stg-compare">Compare</button>' +
            '<button class="staging-btn staging-toggle' + (_state.partialMaskMode ? ' active' : '') + '" id="stg-partial">Partial Accept</button>' +
            '</div>';
        _panelEl.innerHTML = html;
        // Bind events
        var prevBtn = document.getElementById('stg-prev');
        var nextBtn = document.getElementById('stg-next');
        var acceptBtn = document.getElementById('stg-accept');
        var rejectBtn = document.getElementById('stg-reject');
        var regenBtn = document.getElementById('stg-regen');
        var compareBtn = document.getElementById('stg-compare');
        var partialBtn = document.getElementById('stg-partial');
        if (prevBtn)
            prevBtn.addEventListener('click', prevResult);
        if (nextBtn)
            nextBtn.addEventListener('click', nextResult);
        if (acceptBtn)
            acceptBtn.addEventListener('click', accept);
        if (rejectBtn)
            rejectBtn.addEventListener('click', reject);
        if (regenBtn)
            regenBtn.addEventListener('click', regenerate);
        if (compareBtn)
            compareBtn.addEventListener('click', toggleCompare);
        if (partialBtn)
            partialBtn.addEventListener('click', togglePartialMask);
    }
    // ── Public API ──
    function isActive() { return _state.active; }
    function getCurrentIndex() { return _state.currentIndex; }
    function getResults() { return _state.results; }
    function isCompareMode() { return _state.compareMode; }
    function isPartialMaskMode() { return _state.partialMaskMode; }
    return {
        activate: activate,
        deactivate: deactivate,
        accept: accept,
        reject: reject,
        regenerate: regenerate,
        nextResult: nextResult,
        prevResult: prevResult,
        toggleCompare: toggleCompare,
        togglePartialMask: togglePartialMask,
        handlePartialMouseDown: handlePartialMouseDown,
        handlePartialMouseMove: handlePartialMouseMove,
        handlePartialMouseUp: handlePartialMouseUp,
        isActive: isActive,
        getCurrentIndex: getCurrentIndex,
        getResults: getResults,
        isCompareMode: isCompareMode,
        isPartialMaskMode: isPartialMaskMode,
    };
})();
//# sourceMappingURL=canvas-staging.js.map