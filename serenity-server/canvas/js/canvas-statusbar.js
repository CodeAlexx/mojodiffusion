"use strict";
/**
 * Status Bar — SerenityFlow Canvas v2
 *
 * Bottom bar: cursor position, zoom%, active layer, generation status,
 * VRAM usage, loaded model. Updated in real-time.
 */
var CanvasStatusBar = (function () {
    'use strict';
    var _barEl = null;
    var _cursorEl = null;
    var _zoomEl = null;
    var _layerEl = null;
    var _genStatusEl = null;
    var _vramEl = null;
    var _modelEl = null;
    var _genState = 'idle';
    var _vramMb = 0;
    var _modelName = '';
    function create(container) {
        if (_barEl)
            return;
        _barEl = document.createElement('div');
        _barEl.id = 'cv-status-bar';
        _barEl.className = 'cv-status-bar';
        _barEl.innerHTML =
            '<span class="cv-sb-item" id="cv-sb-cursor" title="Cursor position">0, 0</span>' +
                '<span class="cv-sb-sep">|</span>' +
                '<span class="cv-sb-item" id="cv-sb-zoom" title="Zoom">100%</span>' +
                '<span class="cv-sb-sep">|</span>' +
                '<span class="cv-sb-item cv-sb-layer" id="cv-sb-layer" title="Active layer">—</span>' +
                '<span class="cv-sb-spacer"></span>' +
                '<span class="cv-sb-item cv-sb-gen" id="cv-sb-gen" title="Generation status">Idle</span>' +
                '<span class="cv-sb-sep">|</span>' +
                '<span class="cv-sb-item" id="cv-sb-vram" title="VRAM usage">—</span>' +
                '<span class="cv-sb-sep">|</span>' +
                '<span class="cv-sb-item cv-sb-model" id="cv-sb-model" title="Loaded model">No model</span>';
        container.appendChild(_barEl);
        _cursorEl = document.getElementById('cv-sb-cursor');
        _zoomEl = document.getElementById('cv-sb-zoom');
        _layerEl = document.getElementById('cv-sb-layer');
        _genStatusEl = document.getElementById('cv-sb-gen');
        _vramEl = document.getElementById('cv-sb-vram');
        _modelEl = document.getElementById('cv-sb-model');
    }
    function updateCursor(x, y) {
        if (_cursorEl)
            _cursorEl.textContent = Math.round(x) + ', ' + Math.round(y);
    }
    function updateZoom(percent) {
        if (_zoomEl)
            _zoomEl.textContent = percent + '%';
    }
    function updateActiveLayer(name, type) {
        if (!_layerEl)
            return;
        var rawLabel = LAYER_TYPE_LABELS[type] || type || 'Layer';
        _layerEl.textContent = rawLabel.toUpperCase() + ': ' + (name || 'Untitled');
    }
    function updateGenStatus(status) {
        _genState = status || 'idle';
        if (!_genStatusEl)
            return;
        var labels = {
            idle: 'Idle',
            queued: 'Queued',
            generating: 'Generating...',
            complete: 'Complete',
        };
        _genStatusEl.textContent = labels[_genState] || String(_genState);
        _genStatusEl.className = 'cv-sb-item cv-sb-gen cv-sb-gen-' + _genState;
    }
    function updateVram(usedMb) {
        _vramMb = usedMb;
        if (_vramEl) {
            if (usedMb <= 0) {
                _vramEl.textContent = '—';
            }
            else {
                var gb = (usedMb / 1024).toFixed(1);
                _vramEl.textContent = gb + ' GB';
            }
        }
    }
    function updateModel(name) {
        _modelName = name;
        if (_modelEl) {
            _modelEl.textContent = name ? name.split('/').pop().replace(/\.\w+$/, '') : 'No model';
        }
    }
    function getGenStatus() { return _genState; }
    function getVramMb() { return _vramMb; }
    function getModelName() { return _modelName; }
    return {
        create: create,
        updateCursor: updateCursor,
        updateZoom: updateZoom,
        updateActiveLayer: updateActiveLayer,
        updateGenStatus: updateGenStatus,
        updateVram: updateVram,
        updateModel: updateModel,
        getGenStatus: getGenStatus,
        getVramMb: getVramMb,
        getModelName: getModelName,
    };
})();
//# sourceMappingURL=canvas-statusbar.js.map