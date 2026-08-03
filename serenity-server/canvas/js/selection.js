"use strict";
/**
 * Selection: box select, multi-select, copy/paste, delete.
 */
class SFSelection {
    constructor(canvas) {
        this.canvas = canvas;
        this.clipboard = null;
        this.selectionRect = null;
        this._dragStart = null;
        // Initialize keyboard handler properties (assigned in _setupKeyboard)
        this._onKeyDown = () => { };
        this._onKeyUp = () => { };
        this._onBlur = () => { };
        this._setupBoxSelect();
        this._setupKeyboard();
        this._setupGlobalMouseUp();
    }
    _setupBoxSelect() {
        const stage = this.canvas.stage;
        stage.on('mousedown', (e) => {
            // Only start box select on empty space with left button
            if (e.evt.button !== 0)
                return;
            const clickedOnEmpty = e.target === stage || e.target.getLayer() === this.canvas.gridLayer;
            if (!clickedOnEmpty)
                return;
            // Don't start box select if dragging a connection
            if (this.canvas.draggingConnection)
                return;
            if (!e.evt.ctrlKey && !e.evt.metaKey) {
                this.canvas.deselectAll();
            }
            const pointer = stage.getPointerPosition();
            if (!pointer)
                return;
            const worldPos = this.canvas.getWorldPosition(pointer.x, pointer.y);
            this._dragStart = worldPos;
            this.selectionRect = new Konva.Rect({
                x: worldPos.x,
                y: worldPos.y,
                width: 0,
                height: 0,
                fill: 'rgba(74, 158, 255, 0.15)',
                stroke: 'rgba(74, 158, 255, 0.5)',
                strokeWidth: 1 / this.canvas.scale,
                listening: false,
            });
            this.canvas.overlayLayer.add(this.selectionRect);
        });
        stage.on('mousemove', (e) => {
            if (!this._dragStart || !this.selectionRect)
                return;
            if (this.canvas.draggingConnection)
                return;
            const pointer = stage.getPointerPosition();
            if (!pointer)
                return;
            const worldPos = this.canvas.getWorldPosition(pointer.x, pointer.y);
            const x = Math.min(this._dragStart.x, worldPos.x);
            const y = Math.min(this._dragStart.y, worldPos.y);
            const w = Math.abs(worldPos.x - this._dragStart.x);
            const h = Math.abs(worldPos.y - this._dragStart.y);
            this.selectionRect.x(x);
            this.selectionRect.y(y);
            this.selectionRect.width(w);
            this.selectionRect.height(h);
            this.canvas.overlayLayer.batchDraw();
        });
        stage.on('mouseup', (e) => {
            if (!this._dragStart || !this.selectionRect)
                return;
            // Find nodes inside selection rect
            const rx = this.selectionRect.x();
            const ry = this.selectionRect.y();
            const rw = this.selectionRect.width();
            const rh = this.selectionRect.height();
            if (rw > 5 && rh > 5) {
                this.canvas.nodes.forEach((node, id) => {
                    if (node.x + node.width > rx && node.x < rx + rw &&
                        node.y + node.height > ry && node.y < ry + rh) {
                        this.canvas.selectNode(id, true);
                    }
                });
            }
            this.selectionRect.destroy();
            this.selectionRect = null;
            this._dragStart = null;
            this.canvas.overlayLayer.batchDraw();
        });
    }
    _setupGlobalMouseUp() {
        // Connection drag cancellation is now handled by canvas.startConnectionDrag()
        // via its own window mouseup listener (with { once: true })
    }
    _setupKeyboard() {
        this._onKeyDown = (e) => {
            // Input guard: don't intercept when typing in form fields (InvokeAI pattern)
            const target = e.target;
            if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ||
                target.tagName === 'SELECT' || target.isContentEditable) {
                return;
            }
            // Space = temporary pan tool (InvokeAI pattern)
            if (e.key === ' ') {
                e.preventDefault();
                this.canvas.startSpacePan();
                return;
            }
            // Alt = prevent Chrome menu bar activation (InvokeAI pattern)
            if (e.key === 'Alt') {
                e.preventDefault();
                return;
            }
            // Escape = cancel current operation / dismiss context menu
            if (e.key === 'Escape') {
                e.preventDefault();
                // Cancel active connection drag
                if (this.canvas.draggingConnection) {
                    this.canvas.cancelConnectionDrag();
                }
                // Context menu and node search are dismissed by their own Escape handlers
                return;
            }
            // Delete
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.canvas.selectedNodes.size > 0) {
                    e.preventDefault();
                    const action = {
                        type: 'delete',
                        nodes: this._serializeSelected(),
                        connections: this._getSelectedConnections(),
                    };
                    this.canvas.deleteSelected();
                    if (window.sfHistory)
                        sfHistory.push(action);
                }
            }
            // Ctrl+A - select all
            if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
                e.preventDefault();
                this.canvas.nodes.forEach((_, id) => {
                    this.canvas.selectNode(id, true);
                });
            }
            // Ctrl+C - copy
            if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                if (this.canvas.selectedNodes.size > 0) {
                    e.preventDefault();
                    this._copy();
                }
            }
            // Ctrl+V - paste
            if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
                if (this.clipboard) {
                    e.preventDefault();
                    this._paste();
                }
            }
            // Ctrl+D - duplicate
            if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                if (this.canvas.selectedNodes.size > 0) {
                    e.preventDefault();
                    this._copy();
                    this._paste();
                }
            }
            // Ctrl+Z - undo
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                if (window.sfHistory)
                    sfHistory.undo();
            }
            // Ctrl+Y or Ctrl+Shift+Z - redo
            if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                if (window.sfHistory)
                    sfHistory.redo();
            }
            // H - fit view
            if (e.key === 'h' || e.key === 'H') {
                if (!e.ctrlKey && !e.metaKey) {
                    e.preventDefault();
                    this.canvas.fitView(true);
                }
            }
            // Ctrl+L - auto-layout
            if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
                e.preventDefault();
                this.canvas.autoLayout();
            }
            // Shift+G - toggle snap to grid
            if (e.shiftKey && (e.key === 'G' || e.key === 'g')) {
                e.preventDefault();
                this.canvas.snapToGrid = !this.canvas.snapToGrid;
                var snapBtn = document.getElementById('vc-snap');
                if (snapBtn)
                    snapBtn.classList.toggle('active', this.canvas.snapToGrid);
            }
        };
        this._onKeyUp = (e) => {
            // Input guard
            const target = e.target;
            if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ||
                target.tagName === 'SELECT' || target.isContentEditable) {
                return;
            }
            // Space release = revert from pan tool
            if (e.key === ' ') {
                e.preventDefault();
                this.canvas.endSpacePan();
            }
            // Alt release
            if (e.key === 'Alt') {
                e.preventDefault();
            }
        };
        // Window-level blur: release space-pan if window loses focus while space is held
        this._onBlur = () => {
            this.canvas.endSpacePan();
        };
        document.addEventListener('keydown', this._onKeyDown);
        document.addEventListener('keyup', this._onKeyUp);
        window.addEventListener('blur', this._onBlur);
        // Register cleanups
        this.canvas._cleanups.push(() => document.removeEventListener('keydown', this._onKeyDown), () => document.removeEventListener('keyup', this._onKeyUp), () => window.removeEventListener('blur', this._onBlur));
    }
    _copy() {
        const nodes = [];
        const connections = [];
        this.canvas.selectedNodes.forEach((id) => {
            const node = this.canvas.nodes.get(id);
            if (node) {
                nodes.push({
                    id: id,
                    type: node.nodeType,
                    x: node.x,
                    y: node.y,
                    info: node.info,
                    widgetValues: { ...node.widgetValues },
                });
            }
        });
        // Copy connections between selected nodes
        const selectedSet = this.canvas.selectedNodes;
        this.canvas.connections.forEach((c) => {
            if (selectedSet.has(c.sourceNode) && selectedSet.has(c.targetNode)) {
                connections.push({
                    sourceNode: c.sourceNode,
                    sourceSlot: c.sourceSlot,
                    targetNode: c.targetNode,
                    targetSlot: c.targetSlot,
                });
            }
        });
        this.clipboard = { nodes: nodes, connections: connections };
    }
    _paste() {
        if (!this.clipboard)
            return;
        const idMap = {};
        const offset = 30;
        this.canvas.deselectAll();
        // Create nodes with new IDs
        this.clipboard.nodes.forEach((data) => {
            const node = this.canvas.addNode(data.type, data.x + offset, data.y + offset, data.info);
            idMap[data.id] = node.id;
            // Restore widget values
            for (const [name, val] of Object.entries(data.widgetValues)) {
                node.setWidgetValue(name, val);
                if (node.widgets[name]) {
                    node.widgets[name].setValue(val);
                }
            }
            this.canvas.selectNode(node.id, true);
        });
        // Recreate connections
        this.clipboard.connections.forEach((c) => {
            const srcId = idMap[c.sourceNode];
            const tgtId = idMap[c.targetNode];
            if (srcId && tgtId) {
                this.canvas.addConnection(srcId, c.sourceSlot, tgtId, c.targetSlot);
            }
        });
    }
    _serializeSelected() {
        const nodes = [];
        this.canvas.selectedNodes.forEach((id) => {
            const node = this.canvas.nodes.get(id);
            if (node)
                nodes.push(node.serialize());
        });
        return nodes;
    }
    _getSelectedConnections() {
        const selected = this.canvas.selectedNodes;
        return this.canvas.connections
            .filter((c) => selected.has(c.sourceNode) || selected.has(c.targetNode))
            .map((c) => ({
            sourceNode: c.sourceNode,
            sourceSlot: c.sourceSlot,
            targetNode: c.targetNode,
            targetSlot: c.targetSlot,
        }));
    }
}
