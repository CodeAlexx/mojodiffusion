"use strict";
/**
 * Inline widgets for node inputs.
 * Uses Konva shapes with HTML overlay for actual editing.
 */
const WIDGET_HEIGHT = 22;
const WIDGET_PADDING = 4;
class SFWidgetManager {
    constructor() {
        this.activeOverlay = null;
    }
    removeOverlay() {
        if (this.activeOverlay) {
            this.activeOverlay.remove();
            this.activeOverlay = null;
        }
    }
    /**
     * Create a Konva group representing a widget for a node input.
     * Returns { group, getValue, setValue, height }
     */
    createWidget(node, inputName, inputDef, x, y, width) {
        const type = this._detectType(inputDef);
        switch (type) {
            case 'INT':
            case 'FLOAT':
                return this._createNumberWidget(node, inputName, inputDef, x, y, width, type);
            case 'STRING':
                return this._createStringWidget(node, inputName, inputDef, x, y, width);
            case 'BOOLEAN':
                return this._createBooleanWidget(node, inputName, inputDef, x, y, width);
            case 'COMBO':
                return this._createComboWidget(node, inputName, inputDef, x, y, width);
            case 'IMAGE_PICKER':
                return this._createImagePickerWidget(node, inputName, inputDef, x, y, width);
            default:
                return null;
        }
    }
    _detectType(inputDef) {
        if (!inputDef)
            return null;
        // inputDef is usually [type, config] from object_info
        if (Array.isArray(inputDef)) {
            const t = inputDef[0];
            // Combo: array of options — check if all are media filenames
            if (Array.isArray(t)) {
                if (this._isMediaFileList(t))
                    return 'IMAGE_PICKER';
                return 'COMBO';
            }
            if (typeof t === 'string') {
                const upper = t.toUpperCase();
                if (upper === 'INT' || upper === 'FLOAT' || upper === 'STRING' || upper === 'BOOLEAN') {
                    return upper;
                }
            }
        }
        return null;
    }
    _isMediaFileList(options) {
        if (options.length === 0)
            return false;
        // Skip placeholder entries
        const real = options.filter((o) => typeof o === 'string' && !o.startsWith('('));
        if (real.length === 0)
            return false;
        const mediaExts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mp4', '.mov', '.avi', '.mkv'];
        return real.every(name => {
            const dot = name.lastIndexOf('.');
            if (dot < 0)
                return false;
            return mediaExts.includes(name.substring(dot).toLowerCase());
        });
    }
    _getDefault(inputDef) {
        if (Array.isArray(inputDef) && inputDef.length > 1 && inputDef[1]) {
            return inputDef[1].default;
        }
        return undefined;
    }
    _getConfig(inputDef) {
        if (Array.isArray(inputDef) && inputDef.length > 1 && inputDef[1]) {
            return inputDef[1];
        }
        return {};
    }
    _createNumberWidget(node, inputName, inputDef, x, y, width, type) {
        const config = this._getConfig(inputDef);
        const defaultVal = config.default !== undefined ? Number(config.default) : 0;
        const min = config.min !== undefined ? config.min : -Infinity;
        const max = config.max !== undefined ? config.max : Infinity;
        const step = config.step !== undefined ? config.step : (type === 'INT' ? 1 : 0.01);
        let value = defaultVal;
        const group = new Konva.Group({ x: x, y: y });
        // Background bar
        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });
        // Label (left half)
        const labelWidth = Math.floor(width * 0.45);
        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            width: labelWidth,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        // Value display (right half)
        const valueWidth = width - labelWidth - 12;
        const valueText = new Konva.Text({
            x: labelWidth + 6,
            y: 5,
            text: String(value),
            fontSize: 10,
            fill: '#e0e0e0',
            width: valueWidth,
            align: 'right',
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        const updateValueText = () => {
            const formatted = type === 'INT' ? String(Math.round(value)) : value.toFixed(2);
            valueText.text(formatted);
        };
        updateValueText();
        group.add(bg);
        group.add(label);
        group.add(valueText);
        // Drag to scrub
        let dragStartX = 0;
        let dragStartVal = 0;
        let isDragging = false;
        bg.on('mousedown', (e) => {
            e.cancelBubble = true;
            dragStartX = e.evt.clientX;
            dragStartVal = value;
            isDragging = false;
            const onMove = (me) => {
                const dx = me.clientX - dragStartX;
                if (Math.abs(dx) > 3)
                    isDragging = true;
                if (isDragging) {
                    const sensitivity = me.shiftKey ? 0.1 : 1;
                    const delta = dx * step * sensitivity;
                    value = Math.max(min, Math.min(max, dragStartVal + delta));
                    if (type === 'INT')
                        value = Math.round(value);
                    updateValueText();
                    node.setWidgetValue(inputName, value);
                    node.canvas.nodeLayer.batchDraw();
                }
            };
            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
                if (!isDragging) {
                    // Click -- open overlay input
                    this._openNumberOverlay(node, bg, inputName, value, min, max, step, type, (newVal) => {
                        value = newVal;
                        updateValueText();
                        node.setWidgetValue(inputName, value);
                        node.canvas.nodeLayer.batchDraw();
                    });
                }
            };
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        });
        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = type === 'INT' ? Math.round(Number(v)) : Number(v);
                value = Math.max(min, Math.min(max, value));
                updateValueText();
            },
            height: WIDGET_HEIGHT,
        };
    }
    _createStringWidget(node, inputName, inputDef, x, y, width) {
        const config = this._getConfig(inputDef);
        const defaultVal = config.default !== undefined ? String(config.default) : '';
        const multiline = config.multiline || false;
        let value = defaultVal;
        const group = new Konva.Group({ x: x, y: y });
        const h = multiline ? WIDGET_HEIGHT * 3 : WIDGET_HEIGHT;
        const bg = new Konva.Rect({
            width: width,
            height: h,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });
        const strLabelWidth = Math.floor(width * 0.3);
        const label = new Konva.Text({
            x: 6,
            y: multiline ? 3 : 5,
            text: inputName,
            fontSize: multiline ? 9 : 10,
            fill: multiline ? '#606070' : '#808090',
            width: multiline ? width - 12 : strLabelWidth,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        const valueText = new Konva.Text({
            x: multiline ? 6 : (strLabelWidth + 6),
            y: multiline ? 14 : 5,
            width: multiline ? (width - 12) : (width - strLabelWidth - 12),
            height: multiline ? h - 16 : h - 6,
            text: value || '(empty)',
            fontSize: 10,
            fill: value ? '#e0e0e0' : '#505060',
            align: multiline ? 'left' : 'right',
            ellipsis: true,
            wrap: multiline ? 'word' : 'none',
            listening: false,
        });
        group.add(bg);
        group.add(label);
        group.add(valueText);
        bg.on('dblclick', (e) => {
            e.cancelBubble = true;
            this._openStringOverlay(node, bg, inputName, value, multiline, width, h, (newVal) => {
                value = newVal;
                valueText.text(value || '(empty)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
                node.setWidgetValue(inputName, value);
                node.canvas.nodeLayer.batchDraw();
            });
        });
        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = String(v);
                valueText.text(value || '(empty)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
            },
            height: h,
        };
    }
    _createBooleanWidget(node, inputName, inputDef, x, y, width) {
        const config = this._getConfig(inputDef);
        let value = config.default !== undefined ? Boolean(config.default) : false;
        const group = new Konva.Group({ x: x, y: y });
        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });
        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });
        // Checkbox
        const checkBg = new Konva.Rect({
            x: width - 30,
            y: 4,
            width: 14,
            height: 14,
            fill: value ? '#4a9eff' : '#2a2a4a',
            cornerRadius: 2,
            stroke: '#4a4a6a',
            strokeWidth: 1,
            listening: false,
        });
        const checkMark = new Konva.Text({
            x: width - 27,
            y: 4,
            text: value ? '\u2713' : '',
            fontSize: 12,
            fill: '#fff',
            listening: false,
        });
        group.add(bg);
        group.add(label);
        group.add(checkBg);
        group.add(checkMark);
        bg.on('click', (e) => {
            e.cancelBubble = true;
            value = !value;
            checkBg.fill(value ? '#4a9eff' : '#2a2a4a');
            checkMark.text(value ? '\u2713' : '');
            node.setWidgetValue(inputName, value);
            node.canvas.nodeLayer.batchDraw();
        });
        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = Boolean(v);
                checkBg.fill(value ? '#4a9eff' : '#2a2a4a');
                checkMark.text(value ? '\u2713' : '');
            },
            height: WIDGET_HEIGHT,
        };
    }
    _createComboWidget(node, inputName, inputDef, x, y, width) {
        const options = Array.isArray(inputDef[0]) ? inputDef[0] : [];
        const config = this._getConfig(inputDef);
        let value = config.default !== undefined ? String(config.default) : (options[0] || '');
        const group = new Konva.Group({ x: x, y: y });
        const bg = new Konva.Rect({
            width: width,
            height: WIDGET_HEIGHT,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });
        const comboLabelWidth = Math.floor(width * 0.35);
        const label = new Konva.Text({
            x: 6,
            y: 5,
            text: inputName,
            fontSize: 10,
            fill: '#808090',
            width: comboLabelWidth,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        const comboValueWidth = width - comboLabelWidth - 22; // 22 for arrow + padding
        const valueText = new Konva.Text({
            x: comboLabelWidth + 6,
            y: 5,
            text: String(value),
            fontSize: 10,
            fill: '#e0e0e0',
            width: comboValueWidth,
            align: 'right',
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        // Arrow indicator
        const arrow = new Konva.Text({
            x: width - 14,
            y: 5,
            text: '\u25BE',
            fontSize: 10,
            fill: '#808090',
            listening: false,
        });
        group.add(bg);
        group.add(label);
        group.add(valueText);
        group.add(arrow);
        bg.on('click', (e) => {
            e.cancelBubble = true;
            this._openComboOverlay(node, bg, inputName, value, options, width, (newVal) => {
                value = newVal;
                valueText.text(String(value));
                node.setWidgetValue(inputName, value);
                node.canvas.nodeLayer.batchDraw();
            });
        });
        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = String(v);
                valueText.text(value);
            },
            height: WIDGET_HEIGHT,
        };
    }
    // --- HTML Overlay helpers ---
    _getOverlayPosition(node, konvaRect) {
        const absPos = konvaRect.getAbsolutePosition();
        const stage = node.canvas.stage;
        const container = stage.container().getBoundingClientRect();
        return {
            x: container.left + absPos.x * stage.scaleX() + stage.x(),
            y: container.top + absPos.y * stage.scaleY() + stage.y(),
        };
    }
    _openNumberOverlay(node, konvaRect, name, currentVal, min, max, step, type, onDone) {
        this.removeOverlay();
        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = konvaRect.width() * scale;
        const h = konvaRect.height() * scale;
        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';
        const input = document.createElement('input');
        input.type = 'number';
        input.value = String(currentVal);
        input.min = min === -Infinity ? '' : String(min);
        input.max = max === Infinity ? '' : String(max);
        input.step = String(step);
        input.style.width = w + 'px';
        input.style.height = h + 'px';
        div.appendChild(input);
        document.body.appendChild(div);
        this.activeOverlay = div;
        input.focus();
        input.select();
        const finish = () => {
            let v = parseFloat(input.value);
            if (isNaN(v))
                v = currentVal;
            if (type === 'INT')
                v = Math.round(v);
            v = Math.max(min, Math.min(max, v));
            this.removeOverlay();
            onDone(v);
        };
        input.addEventListener('blur', finish);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter')
                finish();
            if (e.key === 'Escape') {
                this.removeOverlay();
            }
        });
    }
    _openStringOverlay(node, konvaRect, name, currentVal, multiline, width, height, onDone) {
        this.removeOverlay();
        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = width * scale;
        const h = height * scale;
        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';
        let input;
        if (multiline) {
            input = document.createElement('textarea');
            input.style.width = w + 'px';
            input.style.height = h + 'px';
        }
        else {
            input = document.createElement('input');
            input.type = 'text';
            input.style.width = w + 'px';
            input.style.height = h + 'px';
        }
        input.value = currentVal;
        div.appendChild(input);
        document.body.appendChild(div);
        this.activeOverlay = div;
        input.focus();
        input.select();
        const finish = () => {
            const v = input.value;
            this.removeOverlay();
            onDone(v);
        };
        input.addEventListener('blur', finish);
        if (!multiline) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter')
                    finish();
                if (e.key === 'Escape') {
                    this.removeOverlay();
                }
            });
        }
    }
    _openComboOverlay(node, konvaRect, name, currentVal, options, width, onDone) {
        this.removeOverlay();
        const pos = this._getOverlayPosition(node, konvaRect);
        const scale = node.canvas.stage.scaleX();
        const w = width * scale;
        const div = document.createElement('div');
        div.className = 'widget-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';
        const select = document.createElement('select');
        select.style.width = w + 'px';
        options.forEach((opt) => {
            const o = document.createElement('option');
            o.value = opt;
            o.text = opt;
            if (opt === currentVal)
                o.selected = true;
            select.appendChild(o);
        });
        div.appendChild(select);
        document.body.appendChild(div);
        this.activeOverlay = div;
        select.focus();
        const finish = () => {
            const v = select.value;
            this.removeOverlay();
            onDone(v);
        };
        select.addEventListener('change', finish);
        select.addEventListener('blur', finish);
    }
    _createImagePickerWidget(node, inputName, inputDef, x, y, width) {
        const options = Array.isArray(inputDef[0]) ? inputDef[0] : [];
        const config = this._getConfig(inputDef);
        let value = config.default !== undefined ? String(config.default) : (options[0] || '');
        const THUMB_H = 48;
        const THUMB_W = 64;
        const widgetH = THUMB_H + 8;
        const group = new Konva.Group({ x: x, y: y });
        const bg = new Konva.Rect({
            width: width,
            height: widgetH,
            fill: '#1a1a35',
            cornerRadius: 3,
            listening: true,
        });
        // Label
        const label = new Konva.Text({
            x: THUMB_W + 12,
            y: 4,
            text: inputName,
            fontSize: 9,
            fill: '#606070',
            width: width - THUMB_W - 18,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        // Filename text (truncated)
        const valueText = new Konva.Text({
            x: THUMB_W + 12,
            y: 16,
            text: value || '(none)',
            fontSize: 10,
            fill: value ? '#e0e0e0' : '#505060',
            width: width - THUMB_W - 18,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        // Thumbnail placeholder rect
        const thumbBg = new Konva.Rect({
            x: 4,
            y: 4,
            width: THUMB_W,
            height: THUMB_H,
            fill: '#14142a',
            cornerRadius: 2,
            listening: false,
        });
        group.add(bg);
        group.add(thumbBg);
        group.add(label);
        group.add(valueText);
        // Konva.Image for the thumbnail
        let thumbImage = null;
        const loadThumbnail = (filename) => {
            if (!filename || filename.startsWith('('))
                return;
            const img = new window.Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                if (thumbImage) {
                    thumbImage.destroy();
                }
                // Fit within THUMB_W x THUMB_H preserving aspect ratio
                const scale = Math.min(THUMB_W / img.width, THUMB_H / img.height);
                const drawW = img.width * scale;
                const drawH = img.height * scale;
                thumbImage = new Konva.Image({
                    x: 4 + (THUMB_W - drawW) / 2,
                    y: 4 + (THUMB_H - drawH) / 2,
                    image: img,
                    width: drawW,
                    height: drawH,
                    listening: false,
                });
                group.add(thumbImage);
                node.canvas.nodeLayer.batchDraw();
            };
            img.src = '/view?filename=' + encodeURIComponent(filename) + '&type=input';
        };
        // Load initial thumbnail
        loadThumbnail(value);
        bg.on('click', (e) => {
            e.cancelBubble = true;
            this._openImagePickerOverlay(node, bg, inputName, value, options, width, (newVal) => {
                value = newVal;
                valueText.text(value || '(none)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
                loadThumbnail(value);
                node.setWidgetValue(inputName, value);
                node.canvas.nodeLayer.batchDraw();
            });
        });
        return {
            group: group,
            getValue: () => value,
            setValue: (v) => {
                value = String(v);
                valueText.text(value || '(none)');
                valueText.fill(value ? '#e0e0e0' : '#505060');
                loadThumbnail(value);
            },
            height: widgetH,
        };
    }
    _openImagePickerOverlay(node, konvaRect, name, currentVal, options, width, onDone) {
        this.removeOverlay();
        const pos = this._getOverlayPosition(node, konvaRect);
        const div = document.createElement('div');
        div.className = 'widget-overlay image-picker-overlay';
        div.style.left = pos.x + 'px';
        div.style.top = pos.y + 'px';
        // Search bar
        const searchBar = document.createElement('input');
        searchBar.type = 'text';
        searchBar.placeholder = 'Filter...';
        searchBar.className = 'image-picker-search';
        // Grid container
        const grid = document.createElement('div');
        grid.className = 'image-picker-grid';
        const mediaExts = ['.mp4', '.mov', '.avi', '.mkv'];
        const buildGrid = (filter) => {
            grid.innerHTML = '';
            const lowerFilter = filter.toLowerCase();
            const filtered = options.filter(o => !o.startsWith('(') && o.toLowerCase().includes(lowerFilter));
            for (const filename of filtered) {
                const item = document.createElement('div');
                item.className = 'image-picker-item';
                if (filename === currentVal)
                    item.classList.add('selected');
                const ext = filename.substring(filename.lastIndexOf('.')).toLowerCase();
                const isVideo = mediaExts.includes(ext);
                if (isVideo) {
                    // Video icon placeholder
                    const icon = document.createElement('div');
                    icon.className = 'image-picker-video-icon';
                    icon.textContent = '\u25B6';
                    item.appendChild(icon);
                }
                else {
                    const img = document.createElement('img');
                    img.src = '/view?filename=' + encodeURIComponent(filename) + '&type=input';
                    img.loading = 'lazy';
                    img.alt = filename;
                    img.onerror = () => { img.style.display = 'none'; };
                    item.appendChild(img);
                }
                const label = document.createElement('div');
                label.className = 'image-picker-label';
                label.textContent = filename;
                label.title = filename;
                item.appendChild(label);
                item.addEventListener('click', () => {
                    this.removeOverlay();
                    onDone(filename);
                });
                grid.appendChild(item);
            }
        };
        searchBar.addEventListener('input', () => buildGrid(searchBar.value));
        // Upload button
        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'image-picker-upload';
        uploadBtn.textContent = 'Upload...';
        uploadBtn.addEventListener('click', () => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*,video/*';
            fileInput.addEventListener('change', () => {
                if (!fileInput.files || fileInput.files.length === 0)
                    return;
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('image', file);
                fetch('/upload/image', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                    if (data.name) {
                        // Add to options and select
                        if (!options.includes(data.name))
                            options.push(data.name);
                        this.removeOverlay();
                        onDone(data.name);
                    }
                })
                    .catch(() => { });
            });
            fileInput.click();
        });
        // Header row with search + upload
        const header = document.createElement('div');
        header.className = 'image-picker-header';
        header.appendChild(searchBar);
        header.appendChild(uploadBtn);
        div.appendChild(header);
        div.appendChild(grid);
        document.body.appendChild(div);
        this.activeOverlay = div;
        buildGrid('');
        searchBar.focus();
        // Close on outside click
        const onClickOutside = (e) => {
            if (!div.contains(e.target)) {
                this.removeOverlay();
                window.removeEventListener('mousedown', onClickOutside);
            }
        };
        // Defer so the current click doesn't immediately close
        setTimeout(() => window.addEventListener('mousedown', onClickOutside), 0);
    }
}
// Global widget manager
const sfWidgets = new SFWidgetManager();
//# sourceMappingURL=widget.js.map