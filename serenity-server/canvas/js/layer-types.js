"use strict";
/**
 * Layer System Data Model — SerenityFlow Canvas v2
 *
 * Typed layer definitions, blend modes, drawing primitives, validation,
 * serialization. All types are global (no modules) to match SF conventions.
 */
// ── Blend Modes ──
var BlendMode = {
    Normal: 'source-over',
    Multiply: 'multiply',
    Screen: 'screen',
    Overlay: 'overlay',
    SoftLight: 'soft-light',
    HardLight: 'hard-light',
    ColorDodge: 'color-dodge',
    ColorBurn: 'color-burn',
    Darken: 'darken',
    Lighten: 'lighten',
    Difference: 'difference',
    Exclusion: 'exclusion',
};
var BLEND_MODE_LABELS = {
    Normal: 'Normal',
    Multiply: 'Multiply',
    Screen: 'Screen',
    Overlay: 'Overlay',
    SoftLight: 'Soft Light',
    HardLight: 'Hard Light',
    ColorDodge: 'Color Dodge',
    ColorBurn: 'Color Burn',
    Darken: 'Darken',
    Lighten: 'Lighten',
    Difference: 'Difference',
    Exclusion: 'Exclusion',
};
// ── Layer Type Enum ──
var LayerType = {
    Draw: 'draw',
    Mask: 'mask',
    Guidance: 'guidance',
    Control: 'control',
    Adjustment: 'adjustment',
    Text: 'text',
};
var LAYER_TYPE_LABELS = {
    draw: 'Draw',
    mask: 'Mask',
    guidance: 'Guidance',
    control: 'Control',
    adjustment: 'Adjustment',
    text: 'Text',
};
// ── Default Factories ──
var LayerDefaults = (function () {
    'use strict';
    var _nextId = 0;
    function nextId() {
        return ++_nextId;
    }
    function setIdCounter(n) {
        _nextId = n;
    }
    function getIdCounter() {
        return _nextId;
    }
    function base(name, type) {
        return {
            id: nextId(),
            name: name,
            type: type,
            visible: true,
            locked: false,
            opacity: 1,
            position: { x: 0, y: 0 },
        };
    }
    function draw(name) {
        return {
            ...base(name || 'Draw Layer', LayerType.Draw),
            type: 'draw',
            blendMode: 'Normal',
        };
    }
    function mask(name) {
        return {
            ...base(name || 'Mask', LayerType.Mask),
            type: 'mask',
            fillColor: 'rgba(239, 68, 68, 0.5)',
            denoiseStrength: 0.75,
            noiseLevel: 0,
        };
    }
    function guidance(name) {
        return {
            ...base(name || 'Guidance', LayerType.Guidance),
            type: 'guidance',
            positivePrompt: '',
            negativePrompt: '',
            referenceImages: [],
            autoNegative: false,
        };
    }
    function control(name) {
        return {
            ...base(name || 'Control', LayerType.Control),
            type: 'control',
            controlModel: '',
            weight: 1,
            beginStep: 0,
            endStep: 1,
            controlMode: 'balanced',
            refImageSrc: '',
            refImageName: '',
        };
    }
    function adjustment(name) {
        return {
            ...base(name || 'Adjustment', LayerType.Adjustment),
            type: 'adjustment',
            brightness: 0,
            contrast: 0,
            saturation: 0,
            temperature: 0,
            tint: 0,
            sharpness: 0,
            hasMask: false,
        };
    }
    function text(name) {
        return {
            ...base(name || 'Text', LayerType.Text),
            type: 'text',
            text: 'Text',
            fontFamily: 'Inter, system-ui, sans-serif',
            fontSize: 32,
            fontWeight: 'normal',
            color: '#ffffff',
            alignment: 'left',
            lineHeight: 1.2,
        };
    }
    function createByType(type, name) {
        switch (type) {
            case 'draw': return draw(name);
            case 'mask': return mask(name);
            case 'guidance': return guidance(name);
            case 'control': return control(name);
            case 'adjustment': return adjustment(name);
            case 'text': return text(name);
            default: return draw(name);
        }
    }
    return {
        nextId: nextId,
        setIdCounter: setIdCounter,
        getIdCounter: getIdCounter,
        draw: draw,
        mask: mask,
        guidance: guidance,
        control: control,
        adjustment: adjustment,
        text: text,
        createByType: createByType,
    };
})();
// ── Validation ──
var LayerValidation = (function () {
    'use strict';
    function clamp(val, min, max) {
        return Math.max(min, Math.min(max, val));
    }
    function validateBase(data) {
        var errors = [];
        if (typeof data.id !== 'number' || data.id < 0)
            errors.push('id must be a non-negative number');
        if (typeof data.name !== 'string' || data.name.length === 0)
            errors.push('name must be non-empty');
        if (typeof data.opacity !== 'number')
            errors.push('opacity must be a number');
        if (typeof data.visible !== 'boolean')
            errors.push('visible must be boolean');
        if (typeof data.locked !== 'boolean')
            errors.push('locked must be boolean');
        return errors;
    }
    /** Safe clamp: coerce null/undefined/NaN to default before clamping */
    function safeNum(val, fallback) {
        var n = typeof val === 'number' && !isNaN(val) ? val : fallback;
        return n;
    }
    function sanitiseBase(data) {
        data.opacity = clamp(safeNum(data.opacity, 1), 0, 1);
        if (!data.name)
            data.name = 'Untitled';
        if (!data.position)
            data.position = { x: 0, y: 0 };
        if (typeof data.visible !== 'boolean')
            data.visible = true;
        if (typeof data.locked !== 'boolean')
            data.locked = false;
    }
    function sanitise(data) {
        sanitiseBase(data);
        switch (data.type) {
            case 'draw':
                if (!data.blendMode)
                    data.blendMode = 'Normal';
                break;
            case 'mask':
                data.denoiseStrength = clamp(safeNum(data.denoiseStrength, 0.75), 0, 1);
                data.noiseLevel = clamp(safeNum(data.noiseLevel, 0), 0, 1);
                break;
            case 'control':
                data.weight = clamp(safeNum(data.weight, 1), 0, 2);
                data.beginStep = clamp(safeNum(data.beginStep, 0), 0, 1);
                data.endStep = clamp(safeNum(data.endStep, 1), 0, 1);
                break;
            case 'adjustment':
                data.brightness = clamp(safeNum(data.brightness, 0), -1, 1);
                data.contrast = clamp(safeNum(data.contrast, 0), -1, 1);
                data.saturation = clamp(safeNum(data.saturation, 0), -1, 1);
                data.temperature = clamp(safeNum(data.temperature, 0), -1, 1);
                data.tint = clamp(safeNum(data.tint, 0), -1, 1);
                data.sharpness = clamp(safeNum(data.sharpness, 0), 0, 1);
                break;
            case 'text':
                data.fontSize = clamp(safeNum(data.fontSize, 32), 1, 1000);
                data.lineHeight = clamp(safeNum(data.lineHeight, 1.2), 0.5, 5);
                break;
        }
        return data;
    }
    function validate(data) {
        var errors = validateBase(data);
        var validTypes = ['draw', 'mask', 'guidance', 'control', 'adjustment', 'text'];
        if (validTypes.indexOf(data.type) === -1) {
            errors.push('invalid layer type: ' + data.type);
        }
        return errors;
    }
    function validateSession(state) {
        var errors = [];
        if (state.version !== 1 && state.version !== 2)
            errors.push('unsupported session version: ' + state.version);
        if (!Array.isArray(state.layers))
            errors.push('layers must be an array');
        if (!state.bbox || typeof state.bbox.width !== 'number')
            errors.push('bbox is invalid');
        if (state.layers) {
            state.layers.forEach(function (l, i) {
                var layerErrors = validate(l.data);
                layerErrors.forEach(function (e) {
                    errors.push('layer[' + i + ']: ' + e);
                });
            });
        }
        return errors;
    }
    return {
        clamp: clamp,
        validate: validate,
        sanitise: sanitise,
        validateSession: validateSession,
    };
})();
// ── Serialization ──
var LayerSerializer = (function () {
    'use strict';
    var SESSION_VERSION = 2;
    var STORAGE_KEY = 'sf-canvas-session';
    function buildSessionState(layers, bbox, activeLayerId, genSettings) {
        // Capture each layer's pixel content as base64 PNG
        var serialised = [];
        var promises = [];
        layers.forEach(function (layer) {
            var entry = {
                data: JSON.parse(JSON.stringify(layer.data)),
                imageData: '',
            };
            serialised.push(entry);
            promises.push(new Promise(function (resolve) {
                try {
                    entry.imageData = layer.konvaLayer.toDataURL();
                }
                catch (e) {
                    entry.imageData = '';
                }
                resolve();
            }));
        });
        return Promise.all(promises).then(function () {
            return {
                version: SESSION_VERSION,
                layers: serialised,
                bbox: bbox,
                activeLayerId: activeLayerId,
                genSettings: genSettings,
            };
        });
    }
    function toJSON(state) {
        return JSON.stringify(state);
    }
    function fromJSON(json) {
        try {
            var parsed = JSON.parse(json);
            var errors = LayerValidation.validateSession(parsed);
            if (errors.length > 0) {
                console.warn('Canvas session validation errors:', errors);
                return null;
            }
            // Sanitise all layers
            parsed.layers.forEach(function (l) {
                LayerValidation.sanitise(l.data);
            });
            return parsed;
        }
        catch (e) {
            console.error('Failed to parse canvas session:', e);
            return null;
        }
    }
    function saveToLocalStorage(state) {
        try {
            localStorage.setItem(STORAGE_KEY, toJSON(state));
        }
        catch (e) {
            console.warn('Failed to save canvas session to localStorage:', e);
        }
    }
    function loadFromLocalStorage() {
        var raw = localStorage.getItem(STORAGE_KEY);
        if (!raw)
            return null;
        return fromJSON(raw);
    }
    function downloadAsFile(state, filename) {
        var json = toJSON(state);
        var blob = new Blob([json], { type: 'application/json' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename || ('canvas_' + Date.now() + '.serenity-canvas');
        a.click();
        URL.revokeObjectURL(url);
    }
    function loadFromFile(file) {
        return new Promise(function (resolve) {
            var reader = new FileReader();
            reader.onload = function (ev) {
                var text = ev.target.result;
                resolve(fromJSON(text));
            };
            reader.onerror = function () { resolve(null); };
            reader.readAsText(file);
        });
    }
    return {
        SESSION_VERSION: SESSION_VERSION,
        STORAGE_KEY: STORAGE_KEY,
        buildSessionState: buildSessionState,
        toJSON: toJSON,
        fromJSON: fromJSON,
        saveToLocalStorage: saveToLocalStorage,
        loadFromLocalStorage: loadFromLocalStorage,
        downloadAsFile: downloadAsFile,
        loadFromFile: loadFromFile,
    };
})();
// ── Blend Mode Helpers ──
var BlendModeUtil = (function () {
    'use strict';
    function toCompositeOp(mode) {
        return BlendMode[mode] || BlendMode.Normal;
    }
    function allKeys() {
        return Object.keys(BlendMode);
    }
    function labelFor(mode) {
        return BLEND_MODE_LABELS[mode] || mode;
    }
    return {
        toCompositeOp: toCompositeOp,
        allKeys: allKeys,
        labelFor: labelFor,
    };
})();
//# sourceMappingURL=layer-types.js.map