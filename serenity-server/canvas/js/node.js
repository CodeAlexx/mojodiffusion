"use strict";
/**
 * Node rendering: header, ports, widgets, drag, selection highlight.
 */
const NODE_MIN_WIDTH = 220;
const NODE_HEADER_HEIGHT = 24;
const NODE_PORT_SPACING = 24;
const NODE_WIDGET_X_PAD = 14;
const NODE_CONTENT_PAD = 8;
class SFNode {
    constructor(id, nodeType, x, y, info, canvas) {
        this.id = id;
        this.nodeType = nodeType;
        this.x = x;
        this.y = y;
        this.canvas = canvas;
        this.info = info || {};
        // Parse inputs/outputs from object_info
        this.inputs = []; // [{ name, type, config }]
        this.outputs = []; // [{ name, type }]
        this.widgetValues = {};
        this.widgets = {}; // name -> widget object
        this._parseInfo();
        // Calculate dimensions
        this.width = this._calcWidth();
        this.height = 0; // calculated during build
        // Konva group — draggable. Port clicks are filtered in dragstart.
        this.group = new Konva.Group({
            x: x,
            y: y,
            draggable: true,
        });
        this._selected = false;
        this._executionState = null; // 'executing' | 'executed' | 'error' | null
        this._isDragging = false;
        this._collapsed = false;
        this._portGroups = { inputs: [], outputs: [] };
        this._selectionBorder = null;
        this._execBorder = null;
        // Build visual
        this._build();
        // Manual drag: only starts from body/header clicks, not port clicks
        this._setupManualDrag();
    }
    _setupManualDrag() {
        this.group.on('dragstart', () => {
            this.canvas.selectNode(this.id, false);
        });
        this.group.on('dragmove', () => {
            this.x = this.group.x();
            this.y = this.group.y();
            this.canvas.connectionLayer.batchDraw();
        });
        this.group.on('dragend', () => {
            this.x = this.group.x();
            this.y = this.group.y();
            this.canvas.snapNodeToGrid(this);
            this.canvas._updateMinimap();
        });
    }
    _parseInfo() {
        if (!this.info)
            return;
        // Parse inputs
        const inputTypes = this.info.input || {};
        // "required" inputs
        if (inputTypes.required) {
            for (const [name, def] of Object.entries(inputTypes.required)) {
                const type = this._extractType(def);
                this.inputs.push({ name: name, type: type, config: def, required: true });
                // Set default widget value
                const defaultVal = this._extractDefault(def);
                if (defaultVal !== undefined) {
                    this.widgetValues[name] = defaultVal;
                }
            }
        }
        // "optional" inputs
        if (inputTypes.optional) {
            for (const [name, def] of Object.entries(inputTypes.optional)) {
                const type = this._extractType(def);
                this.inputs.push({ name: name, type: type, config: def, required: false });
                const defaultVal = this._extractDefault(def);
                if (defaultVal !== undefined) {
                    this.widgetValues[name] = defaultVal;
                }
            }
        }
        // Parse outputs
        const outputTypes = this.info.output || [];
        const outputNames = this.info.output_name || [];
        for (let i = 0; i < outputTypes.length; i++) {
            this.outputs.push({
                name: outputNames[i] || outputTypes[i],
                type: outputTypes[i],
            });
        }
    }
    _extractType(def) {
        if (Array.isArray(def)) {
            const t = def[0];
            if (Array.isArray(t)) {
                // Detect media file lists → IMAGE_PICKER
                if (sfWidgets._isMediaFileList(t))
                    return 'IMAGE_PICKER';
                return 'COMBO';
            }
            return String(t);
        }
        return '*';
    }
    _extractDefault(def) {
        if (Array.isArray(def) && def.length > 1 && def[1] && def[1].default !== undefined) {
            return def[1].default;
        }
        // For combo, default to first option
        if (Array.isArray(def) && Array.isArray(def[0]) && def[0].length > 0) {
            return def[0][0];
        }
        return undefined;
    }
    _isWidgetType(type) {
        return ['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO', 'IMAGE_PICKER'].includes(type);
    }
    _calcWidth() {
        // Estimate based on name lengths and widget content
        let maxLen = this.nodeType.length * 7 + 30;
        for (const inp of this.inputs) {
            // For widgets: label + value need space side by side
            if (this._isWidgetType(inp.type)) {
                let valueLen = inp.name.length * 6;
                // Add space for combo option text or default value
                if (Array.isArray(inp.config) && Array.isArray(inp.config[0])) {
                    const longest = inp.config[0].reduce((a, b) => a.length > b.length ? a : b, '');
                    valueLen = (inp.name.length + longest.length + 4) * 6;
                }
                else {
                    valueLen = inp.name.length * 6 + 60; // room for number/text value
                }
                maxLen = Math.max(maxLen, valueLen + NODE_WIDGET_X_PAD * 2);
            }
            else {
                maxLen = Math.max(maxLen, inp.name.length * 6 + 40);
            }
        }
        for (const out of this.outputs) {
            maxLen = Math.max(maxLen, out.name.length * 6 + 40);
        }
        return Math.max(NODE_MIN_WIDTH, Math.min(maxLen, 360));
    }
    getInputPortPos(slotIndex) {
        const entry = this._portGroups.inputs[slotIndex];
        if (!entry)
            return { x: this.x, y: this.y };
        return { x: this.x, y: this.y + entry.y };
    }
    getOutputPortPos(slotIndex) {
        const entry = this._portGroups.outputs[slotIndex];
        if (!entry)
            return { x: this.x + this.width, y: this.y };
        return { x: this.x + this.width, y: this.y + entry.y };
    }
    getInputType(slotIndex) {
        return this.inputs[slotIndex] ? this.inputs[slotIndex].type : '*';
    }
    getOutputType(slotIndex) {
        return this.outputs[slotIndex] ? this.outputs[slotIndex].type : '*';
    }
    getInputIndex(name) {
        return this.inputs.findIndex(inp => inp.name === name);
    }
    setWidgetValue(name, value) {
        this.widgetValues[name] = value;
    }
    getWidgetValue(name) {
        return this.widgetValues[name];
    }
    setSelected(selected) {
        this._selected = selected;
        if (this._selectionBorder) {
            this._selectionBorder.visible(selected);
        }
        this.canvas.nodeLayer.batchDraw();
    }
    setExecutionState(state) {
        this._executionState = state;
        if (!this._execBorder)
            return;
        switch (state) {
            case 'executing':
                this._execBorder.stroke('#ffaa00');
                this._execBorder.visible(true);
                break;
            case 'executed':
                this._execBorder.stroke('#4aff4a');
                this._execBorder.visible(true);
                // Auto-hide after 2s
                setTimeout(() => {
                    if (this._executionState === 'executed') {
                        this._execBorder.visible(false);
                        this._executionState = null;
                        this.canvas.nodeLayer.batchDraw();
                    }
                }, 2000);
                break;
            case 'error':
                this._execBorder.stroke('#ff4a4a');
                this._execBorder.visible(true);
                break;
            default:
                this._execBorder.visible(false);
        }
        this.canvas.nodeLayer.batchDraw();
    }
    updateWidgetVisibility() {
        // Hide widgets for connected inputs, show for disconnected
        for (let i = 0; i < this.inputs.length; i++) {
            const inp = this.inputs[i];
            const widget = this.widgets[inp.name];
            if (!widget)
                continue;
            const connected = this.canvas.isInputConnected(this.id, i);
            widget.group.visible(!connected);
        }
        this.canvas.nodeLayer.batchDraw();
    }
    destroy() {
        this.group.destroy();
    }
    // ── Phase 2: Collapse/Expand ──
    toggleCollapse() {
        this._collapsed = !this._collapsed;
        this._build();
        this.canvas.connectionLayer.batchDraw();
    }
    get collapsed() { return this._collapsed; }
    _build() {
        this.group.destroyChildren();
        this._portGroups = { inputs: [], outputs: [] };
        const w = this.width;
        const category = this.info.category || '';
        const headerColor = getCategoryColor(category);
        // --- Header ---
        const header = new Konva.Rect({
            width: w,
            height: NODE_HEADER_HEIGHT,
            fill: headerColor,
            cornerRadius: this._collapsed ? 4 : [4, 4, 0, 0],
            listening: true,
        });
        this.group.add(header);
        // Collapse chevron
        const chevron = new Konva.Text({
            x: w - 18,
            y: 6,
            text: this._collapsed ? '▸' : '▾',
            fontSize: 11,
            fill: '#b0b0c0',
            listening: true,
        });
        chevron.on('click', (e) => {
            e.cancelBubble = true;
            this.toggleCollapse();
        });
        this.group.add(chevron);
        const title = new Konva.Text({
            x: 8,
            y: 6,
            text: this.info.display_name || this.nodeType,
            fontSize: 11,
            fontStyle: 'bold',
            fill: '#e0e0e0',
            width: w - 28,
            ellipsis: true,
            wrap: 'none',
            listening: false,
        });
        this.group.add(title);
        if (this._collapsed) {
            // Collapsed: only header, ports as dots on edges
            this.height = NODE_HEADER_HEIGHT;
            // Input ports spread along left edge
            const inCount = this.inputs.length;
            for (let i = 0; i < inCount; i++) {
                const py = inCount === 1 ? NODE_HEADER_HEIGHT / 2 : 4 + (NODE_HEADER_HEIGHT - 8) * i / (inCount - 1);
                const portGroup = createInputPort(this, i, 0, py, this.inputs[i].type, '');
                this.group.add(portGroup);
                this._portGroups.inputs.push({ y: py, group: portGroup });
            }
            // Output ports spread along right edge
            const outCount = this.outputs.length;
            for (let i = 0; i < outCount; i++) {
                const py = outCount === 1 ? NODE_HEADER_HEIGHT / 2 : 4 + (NODE_HEADER_HEIGHT - 8) * i / (outCount - 1);
                const portGroup = createOutputPort(this, i, w, py, this.outputs[i].type, '');
                this.group.add(portGroup);
                this._portGroups.outputs.push({ y: py, group: portGroup });
            }
        }
        else {
            // Expanded: full build
            this._buildExpanded(w, headerColor);
        }
        // --- Selection border ---
        this._selectionBorder = new Konva.Rect({
            x: -2, y: -2,
            width: w + 4, height: this.height + 4,
            stroke: '#4a9eff', strokeWidth: 2, cornerRadius: 5,
            visible: this._selected, listening: false,
        });
        this.group.add(this._selectionBorder);
        // --- Execution highlight border ---
        this._execBorder = new Konva.Rect({
            x: -2, y: -2,
            width: w + 4, height: this.height + 4,
            stroke: 'transparent', strokeWidth: 2, cornerRadius: 5,
            visible: false, listening: false,
        });
        this.group.add(this._execBorder);
        // Re-apply execution state if active
        if (this._executionState)
            this.setExecutionState(this._executionState);
        this.canvas.nodeLayer.batchDraw();
    }
    _buildExpanded(w, _headerColor) {
        let yPos = NODE_HEADER_HEIGHT;
        const widgetWidth = w - NODE_WIDGET_X_PAD * 2;
        // --- Outputs ---
        for (let i = 0; i < this.outputs.length; i++) {
            const out = this.outputs[i];
            const portY = yPos + NODE_CONTENT_PAD + i * NODE_PORT_SPACING;
            const portGroup = createOutputPort(this, i, w, portY + 6, out.type, out.name);
            this.group.add(portGroup);
            this._portGroups.outputs.push({ y: portY + 6, group: portGroup });
        }
        if (this.outputs.length > 0) {
            yPos += NODE_CONTENT_PAD + this.outputs.length * NODE_PORT_SPACING;
        }
        // --- Inputs + Widgets ---
        for (let i = 0; i < this.inputs.length; i++) {
            const inp = this.inputs[i];
            const isWidget = this._isWidgetType(inp.type);
            if (isWidget) {
                const widgetY = yPos + NODE_CONTENT_PAD;
                const widget = sfWidgets.createWidget(this, inp.name, inp.config, NODE_WIDGET_X_PAD, widgetY, widgetWidth);
                if (widget) {
                    this.group.add(widget.group);
                    this.widgets[inp.name] = widget;
                    widget._yPos = widgetY;
                    const portGroup = createInputPort(this, i, 0, widgetY + widget.height / 2, inp.type, '');
                    this.group.add(portGroup);
                    this._portGroups.inputs.push({ y: widgetY + widget.height / 2, group: portGroup, widgetName: inp.name });
                    yPos = widgetY + widget.height + 2;
                }
                else {
                    const portY = yPos + NODE_CONTENT_PAD + 6;
                    const portGroup = createInputPort(this, i, 0, portY, inp.type, inp.name);
                    this.group.add(portGroup);
                    this._portGroups.inputs.push({ y: portY, group: portGroup });
                    yPos = portY + NODE_PORT_SPACING - 6;
                }
            }
            else {
                const portY = yPos + NODE_CONTENT_PAD + 6;
                const portGroup = createInputPort(this, i, 0, portY, inp.type, inp.name);
                this.group.add(portGroup);
                this._portGroups.inputs.push({ y: portY, group: portGroup });
                yPos = portY + NODE_PORT_SPACING - 6;
            }
        }
        if (this.inputs.length === 0 && this.outputs.length === 0)
            yPos += 20;
        yPos += NODE_CONTENT_PAD;
        this.height = yPos;
        // Body background
        const body = new Konva.Rect({
            y: NODE_HEADER_HEIGHT, width: w,
            height: this.height - NODE_HEADER_HEIGHT,
            fill: '#252545', cornerRadius: [0, 0, 4, 4], listening: true,
        });
        this.group.add(body);
        body.moveToBottom();
    }
    // Serialization helpers
    serialize() {
        return {
            id: this.id,
            type: this.nodeType,
            pos: [this.x, this.y],
            size: [this.width, this.height],
            widgets_values: { ...this.widgetValues },
            collapsed: this._collapsed,
        };
    }
}
//# sourceMappingURL=node.js.map