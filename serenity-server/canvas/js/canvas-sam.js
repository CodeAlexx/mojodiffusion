"use strict";
/**
 * SAM 3 Integration — SerenityFlow Canvas v2
 *
 * SelectObjectTool with three modes: Text, Click, Exemplar.
 * Results overlay with per-instance mask display, apply/cancel.
 * Auto-detect convenience buttons.
 */
// ── SAM State & API ──
var CanvasSAM = (function () {
    'use strict';
    var _mode = 'text';
    var _instances = [];
    var _overlayImages = [];
    var _clickPoints = [];
    var _clickMarkers = [];
    var _exemplarRect = null;
    var _exemplarStart = null;
    var _loading = false;
    var _debounceTimer = null;
    var _resultsPanel = null;
    // ── API calls ──
    function getApiBase() {
        return window.location.protocol + '//' + window.location.host;
    }
    function captureCanvasComposite(ctx) {
        return new Promise(function (resolve) {
            var bb = ctx.boundingBox;
            var dataUrl = ctx.stage.toDataURL({
                x: bb.x(), y: bb.y(),
                width: bb.width(), height: bb.height(),
                pixelRatio: 1,
            });
            // Convert data URL to Blob
            var byteString = atob(dataUrl.split(',')[1]);
            var mimeString = dataUrl.split(',')[0].split(':')[1].split(';')[0];
            var ab = new ArrayBuffer(byteString.length);
            var ia = new Uint8Array(ab);
            for (var i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            resolve(new Blob([ab], { type: mimeString }));
        });
    }
    function sendTextRequest(ctx, prompt, threshold) {
        if (_loading)
            return;
        _loading = true;
        updateLoadingState();
        captureCanvasComposite(ctx).then(function (blob) {
            var form = new FormData();
            form.append('image', blob, 'canvas.png');
            form.append('prompt', prompt);
            form.append('threshold', String(threshold));
            return fetch(getApiBase() + '/canvas/sam3/text', { method: 'POST', body: form });
        }).then(function (res) { return res.json(); })
            .then(function (data) {
            _loading = false;
            if (data.error) {
                showSamError(data.error);
            }
            else {
                handleResults(ctx, data.instances || []);
            }
        }).catch(function (err) {
            _loading = false;
            showSamError(err.message || 'Network error');
        });
    }
    function sendPointsRequest(ctx) {
        if (_loading || _clickPoints.length === 0)
            return;
        _loading = true;
        updateLoadingState();
        captureCanvasComposite(ctx).then(function (blob) {
            var form = new FormData();
            form.append('image', blob, 'canvas.png');
            form.append('points', JSON.stringify(_clickPoints));
            return fetch(getApiBase() + '/canvas/sam3/points', { method: 'POST', body: form });
        }).then(function (res) { return res.json(); })
            .then(function (data) {
            _loading = false;
            if (data.error) {
                showSamError(data.error);
            }
            else {
                handleResults(ctx, data.instances || []);
            }
        }).catch(function (err) {
            _loading = false;
            showSamError(err.message || 'Network error');
        });
    }
    function sendExemplarRequest(ctx, bbox) {
        if (_loading)
            return;
        _loading = true;
        updateLoadingState();
        captureCanvasComposite(ctx).then(function (blob) {
            var form = new FormData();
            form.append('image', blob, 'canvas.png');
            form.append('bbox', JSON.stringify(bbox));
            return fetch(getApiBase() + '/canvas/sam3/exemplar', { method: 'POST', body: form });
        }).then(function (res) { return res.json(); })
            .then(function (data) {
            _loading = false;
            if (data.error) {
                showSamError(data.error);
            }
            else {
                handleResults(ctx, data.instances || []);
            }
        }).catch(function (err) {
            _loading = false;
            showSamError(err.message || 'Network error');
        });
    }
    // ── Results handling ──
    function handleResults(ctx, instances) {
        clearOverlays(ctx);
        _instances = instances.map(function (inst) {
            inst.selected = true;
            return inst;
        });
        if (_instances.length === 0) {
            showSamError('No objects detected');
            return;
        }
        // Render mask overlays on canvas
        var bb = ctx.boundingBox;
        _instances.forEach(function (inst) {
            var img = new window.Image();
            img.onload = function () {
                // Create colored semi-transparent overlay from grayscale mask
                var tmpCanvas = document.createElement('canvas');
                tmpCanvas.width = img.width;
                tmpCanvas.height = img.height;
                var tctx = tmpCanvas.getContext('2d');
                // Draw mask
                tctx.drawImage(img, 0, 0);
                var imageData = tctx.getImageData(0, 0, img.width, img.height);
                var d = imageData.data;
                // Parse hex color
                var hex = inst.color || '#ef4444';
                var r = parseInt(hex.slice(1, 3), 16);
                var g = parseInt(hex.slice(3, 5), 16);
                var b = parseInt(hex.slice(5, 7), 16);
                // Replace white pixels with colored overlay
                for (var i = 0; i < d.length; i += 4) {
                    var alpha = d[i]; // grayscale value = mask intensity
                    d[i] = r;
                    d[i + 1] = g;
                    d[i + 2] = b;
                    d[i + 3] = Math.round(alpha * 0.4); // 40% opacity
                }
                tctx.putImageData(imageData, 0, 0);
                var coloredImg = new window.Image();
                coloredImg.onload = function () {
                    var kImg = new Konva.Image({
                        image: coloredImg,
                        x: bb.x(), y: bb.y(),
                        width: bb.width(), height: bb.height(),
                        listening: false,
                        name: 'sam-overlay-' + inst.instance_id,
                    });
                    ctx.uiLayer.add(kImg);
                    ctx.uiLayer.batchDraw();
                    _overlayImages.push({ img: kImg, id: inst.instance_id });
                };
                coloredImg.src = tmpCanvas.toDataURL();
            };
            img.src = 'data:image/png;base64,' + inst.mask_png;
        });
        showResultsPanel(ctx);
    }
    function clearOverlays(ctx) {
        _overlayImages.forEach(function (o) { o.img.destroy(); });
        _overlayImages = [];
        _clickMarkers.forEach(function (m) { m.destroy(); });
        _clickMarkers = [];
        if (_exemplarRect) {
            _exemplarRect.destroy();
            _exemplarRect = null;
        }
        ctx.uiLayer.batchDraw();
    }
    function toggleInstance(ctx, instanceId, selected) {
        _instances.forEach(function (inst) {
            if (inst.instance_id === instanceId)
                inst.selected = selected;
        });
        // Show/hide overlay
        _overlayImages.forEach(function (o) {
            if (o.id === instanceId) {
                o.img.visible(selected);
            }
        });
        ctx.uiLayer.batchDraw();
    }
    // ── Apply results to mask layers ──
    function applySelected(ctx) {
        var selected = _instances.filter(function (i) { return i.selected; });
        if (selected.length === 0)
            return;
        selected.forEach(function (inst) {
            var maskLayer = ctx.addLayer('Mask: ' + inst.label, 'mask');
            var img = new window.Image();
            img.onload = function () {
                var bb = ctx.boundingBox;
                var kImg = new Konva.Image({
                    image: img,
                    x: bb.x(), y: bb.y(),
                    width: bb.width(), height: bb.height(),
                    listening: false,
                });
                // Tint the mask red for visualization
                maskLayer.konvaLayer.add(kImg);
                maskLayer.konvaLayer.batchDraw();
            };
            img.src = 'data:image/png;base64,' + inst.mask_png;
        });
        clearOverlays(ctx);
        hideResultsPanel();
        _instances = [];
        ctx.pushHistory();
    }
    function applyAll(ctx) {
        _instances.forEach(function (i) { i.selected = true; });
        applySelected(ctx);
    }
    function cancelResults(ctx) {
        clearOverlays(ctx);
        hideResultsPanel();
        _instances = [];
        _clickPoints = [];
    }
    // ── Results Panel UI ──
    function showResultsPanel(ctx) {
        if (!_resultsPanel) {
            _resultsPanel = document.createElement('div');
            _resultsPanel.id = 'sam-results-panel';
            _resultsPanel.className = 'sam-results-panel';
            document.body.appendChild(_resultsPanel);
        }
        _resultsPanel.style.display = 'block';
        renderResultsList(ctx);
    }
    function hideResultsPanel() {
        if (_resultsPanel)
            _resultsPanel.style.display = 'none';
    }
    function renderResultsList(ctx) {
        if (!_resultsPanel)
            return;
        var html = '<div class="sam-results-header">SAM Results (' + _instances.length + ')</div>';
        html += '<div class="sam-results-list">';
        _instances.forEach(function (inst) {
            html += '<label class="sam-result-item">' +
                '<input type="checkbox" class="sam-result-check" data-id="' + inst.instance_id + '"' +
                (inst.selected ? ' checked' : '') + '>' +
                '<span class="sam-result-swatch" style="background:' + inst.color + '"></span>' +
                '<span class="sam-result-label">' + inst.label + '</span>' +
                '<span class="sam-result-conf">' + (inst.confidence * 100).toFixed(0) + '%</span>' +
                '</label>';
        });
        html += '</div>';
        html += '<div class="sam-results-actions">' +
            '<button class="sam-btn sam-btn-apply" id="sam-apply">Apply Selected</button>' +
            '<button class="sam-btn sam-btn-all" id="sam-apply-all">Apply All</button>' +
            '<button class="sam-btn sam-btn-cancel" id="sam-cancel">Cancel</button>' +
            '</div>';
        _resultsPanel.innerHTML = html;
        // Bind events
        _resultsPanel.querySelectorAll('.sam-result-check').forEach(function (cb) {
            cb.addEventListener('change', function () {
                toggleInstance(ctx, this.dataset.id, this.checked);
            });
        });
        document.getElementById('sam-apply').addEventListener('click', function () { applySelected(ctx); });
        document.getElementById('sam-apply-all').addEventListener('click', function () { applyAll(ctx); });
        document.getElementById('sam-cancel').addEventListener('click', function () { cancelResults(ctx); });
    }
    function updateLoadingState() {
        var statusEl = document.getElementById('sam-status');
        if (statusEl) {
            statusEl.textContent = _loading ? 'Detecting...' : '';
            statusEl.style.display = _loading ? 'inline' : 'none';
        }
    }
    function showSamError(msg) {
        updateLoadingState();
        console.warn('[SAM]', msg);
        // Brief toast
        var toast = document.createElement('div');
        toast.className = 'sam-toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(function () { toast.remove(); }, 3000);
    }
    // ── SelectObject Tool ──
    var SelectObjectTool = {
        name: 'sam',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onActivate: function (ctx) {
            showSamToolbar();
        },
        onDeactivate: function (ctx) {
            cancelResults(ctx);
            hideSamToolbar();
        },
        onMouseDown: function (ctx, e) {
            if (_mode === 'click') {
                // Green = include (left click), Red = exclude (alt+click)
                var pos = ctx.getRelativePointerPosition();
                var bb = ctx.boundingBox;
                var relX = pos.x - bb.x();
                var relY = pos.y - bb.y();
                var label = e.evt.altKey ? 0 : 1;
                _clickPoints.push({ x: relX, y: relY, label: label });
                // Visual marker
                var marker = new Konva.Circle({
                    x: pos.x, y: pos.y,
                    radius: 5 / ctx.stage.scaleX(),
                    fill: label === 1 ? '#22c55e' : '#ef4444',
                    stroke: '#fff',
                    strokeWidth: 1.5 / ctx.stage.scaleX(),
                    listening: false,
                });
                ctx.uiLayer.add(marker);
                ctx.uiLayer.batchDraw();
                _clickMarkers.push(marker);
                // Debounced request
                if (_debounceTimer)
                    clearTimeout(_debounceTimer);
                _debounceTimer = setTimeout(function () {
                    sendPointsRequest(ctx);
                }, 200);
            }
            if (_mode === 'exemplar') {
                _exemplarStart = ctx.getRelativePointerPosition();
                if (_exemplarRect) {
                    _exemplarRect.destroy();
                    _exemplarRect = null;
                }
                _exemplarRect = new Konva.Rect({
                    x: _exemplarStart.x, y: _exemplarStart.y,
                    width: 0, height: 0,
                    stroke: '#f59e0b',
                    strokeWidth: 2 / ctx.stage.scaleX(),
                    dash: [6, 3],
                    listening: false,
                });
                ctx.uiLayer.add(_exemplarRect);
            }
        },
        onMouseMove: function (ctx, pos) {
            if (_mode === 'exemplar' && _exemplarStart && _exemplarRect) {
                var x = Math.min(_exemplarStart.x, pos.x);
                var y = Math.min(_exemplarStart.y, pos.y);
                _exemplarRect.x(x);
                _exemplarRect.y(y);
                _exemplarRect.width(Math.abs(pos.x - _exemplarStart.x));
                _exemplarRect.height(Math.abs(pos.y - _exemplarStart.y));
                ctx.uiLayer.batchDraw();
            }
        },
        onMouseUp: function (ctx) {
            if (_mode === 'exemplar' && _exemplarStart && _exemplarRect) {
                var bb = ctx.boundingBox;
                var bboxData = {
                    x: _exemplarRect.x() - bb.x(),
                    y: _exemplarRect.y() - bb.y(),
                    width: _exemplarRect.width(),
                    height: _exemplarRect.height(),
                };
                _exemplarStart = null;
                if (bboxData.width > 5 && bboxData.height > 5) {
                    sendExemplarRequest(ctx, bboxData);
                }
            }
        },
    };
    // ── SAM Toolbar (mode selector + text input) ──
    function showSamToolbar() {
        var existing = document.getElementById('sam-toolbar');
        if (existing) {
            existing.style.display = 'flex';
            return;
        }
        var toolbar = document.createElement('div');
        toolbar.id = 'sam-toolbar';
        toolbar.className = 'sam-toolbar';
        toolbar.innerHTML =
            '<div class="sam-mode-group">' +
                '<button class="sam-mode-btn active" data-mode="text">Text</button>' +
                '<button class="sam-mode-btn" data-mode="click">Click</button>' +
                '<button class="sam-mode-btn" data-mode="exemplar">Exemplar</button>' +
                '</div>' +
                '<input type="text" id="sam-text-input" class="sam-text-input" placeholder="e.g. face, person, dog...">' +
                '<button id="sam-detect-btn" class="sam-detect-btn">Detect</button>' +
                '<span id="sam-status" class="sam-status" style="display:none"></span>' +
                '<div class="sam-presets">' +
                '<button class="sam-preset-btn" data-prompt="face" title="Detect Faces">Faces</button>' +
                '<button class="sam-preset-btn" data-prompt="person" title="Detect People">People</button>' +
                '<button class="sam-preset-btn" data-prompt="background" title="Detect Background">BG</button>' +
                '</div>';
        // Insert above the canvas stage
        var stageContainer = document.getElementById('canvas-stage-container');
        if (stageContainer && stageContainer.parentElement) {
            stageContainer.parentElement.insertBefore(toolbar, stageContainer);
        }
        else {
            document.body.appendChild(toolbar);
        }
        // Mode switching
        toolbar.querySelectorAll('.sam-mode-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                _mode = this.dataset.mode;
                toolbar.querySelectorAll('.sam-mode-btn').forEach(function (b) {
                    b.classList.remove('active');
                });
                this.classList.add('active');
                var textInput = document.getElementById('sam-text-input');
                var detectBtn = document.getElementById('sam-detect-btn');
                if (textInput)
                    textInput.style.display = _mode === 'text' ? '' : 'none';
                if (detectBtn)
                    detectBtn.style.display = _mode === 'text' ? '' : 'none';
            });
        });
        // Detect button and Enter key
        var detectBtn = document.getElementById('sam-detect-btn');
        var textInput = document.getElementById('sam-text-input');
        if (detectBtn) {
            detectBtn.addEventListener('click', function () {
                triggerTextDetect();
            });
        }
        if (textInput) {
            textInput.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    triggerTextDetect();
                }
            });
        }
        // Preset buttons
        toolbar.querySelectorAll('.sam-preset-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var prompt = this.dataset.prompt;
                if (textInput)
                    textInput.value = prompt;
                _mode = 'text';
                toolbar.querySelectorAll('.sam-mode-btn').forEach(function (b) {
                    b.classList.toggle('active', b.dataset.mode === 'text');
                });
                triggerTextDetect(prompt);
            });
        });
    }
    function triggerTextDetect(prompt) {
        var textInput = document.getElementById('sam-text-input');
        var text = prompt || (textInput ? textInput.value.trim() : '');
        if (!text)
            return;
        // We need the tool context — get it from CanvasTab
        var ctx = window._samToolContext;
        if (ctx) {
            sendTextRequest(ctx, text, 0.3);
        }
    }
    function hideSamToolbar() {
        var toolbar = document.getElementById('sam-toolbar');
        if (toolbar)
            toolbar.style.display = 'none';
    }
    // ── Public API ──
    function getMode() { return _mode; }
    function setMode(m) { _mode = m; }
    function isLoading() { return _loading; }
    function getInstances() { return _instances; }
    function getTool() { return SelectObjectTool; }
    return {
        getMode: getMode,
        setMode: setMode,
        isLoading: isLoading,
        getInstances: getInstances,
        getTool: getTool,
        applySelected: applySelected,
        applyAll: applyAll,
        cancelResults: cancelResults,
        triggerTextDetect: triggerTextDetect,
    };
})();
//# sourceMappingURL=canvas-sam.js.map