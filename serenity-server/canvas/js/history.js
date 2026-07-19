"use strict";
/**
 * Undo/redo stack.
 * Tracks canvas state snapshots (simple approach — serialize full state).
 */
class SFHistory {
    constructor(canvas) {
        this.canvas = canvas;
        this.undoStack = [];
        this.redoStack = [];
        this.maxSize = 50;
        this._paused = false;
    }
    /**
     * Take a snapshot of the current state.
     * Call this before any destructive operation.
     */
    saveState() {
        if (this._paused)
            return;
        const state = this._serialize();
        this.undoStack.push(state);
        if (this.undoStack.length > this.maxSize) {
            this.undoStack.shift();
        }
        // Clear redo stack on new action
        this.redoStack = [];
    }
    undo() {
        if (this.undoStack.length === 0)
            return;
        // Save current state to redo
        const current = this._serialize();
        this.redoStack.push(current);
        const state = this.undoStack.pop();
        this._restore(state);
    }
    redo() {
        if (this.redoStack.length === 0)
            return;
        // Save current state to undo
        const current = this._serialize();
        this.undoStack.push(current);
        const state = this.redoStack.pop();
        this._restore(state);
    }
    // Push a generic action for undo (not full state — lightweight)
    push(action) {
        this.saveState();
    }
    _serialize() {
        const nodes = {};
        this.canvas.nodes.forEach((node, id) => {
            nodes[id] = {
                type: node.nodeType,
                x: node.x,
                y: node.y,
                info: node.info,
                widgetValues: { ...node.widgetValues },
            };
        });
        const connections = this.canvas.connections.map((c) => ({
            sourceNode: c.sourceNode,
            sourceSlot: c.sourceSlot,
            targetNode: c.targetNode,
            targetSlot: c.targetSlot,
        }));
        return { nodes: nodes, connections: connections, nextId: this.canvas.nextNodeId };
    }
    _restore(state) {
        this._paused = true;
        // Clear everything
        const ids = [...this.canvas.nodes.keys()];
        ids.forEach(id => this.canvas.removeNode(id));
        this.canvas.connections.forEach((c) => c.destroy());
        this.canvas.connections = [];
        // Restore nodes
        this.canvas.nextNodeId = state.nextId || 1;
        for (const [id, data] of Object.entries(state.nodes)) {
            const numId = parseInt(id);
            if (numId >= this.canvas.nextNodeId) {
                this.canvas.nextNodeId = numId + 1;
            }
            const node = new SFNode(id, data.type, data.x, data.y, data.info, this.canvas);
            this.canvas.nodes.set(id, node);
            this.canvas.nodeLayer.add(node.group);
            // Restore widget values
            for (const [name, val] of Object.entries(data.widgetValues || {})) {
                node.setWidgetValue(name, val);
                if (node.widgets[name])
                    node.widgets[name].setValue(val);
            }
        }
        // Restore connections
        for (const c of state.connections) {
            if (this.canvas.nodes.has(c.sourceNode) && this.canvas.nodes.has(c.targetNode)) {
                const conn = new SFConnection(c.sourceNode, c.sourceSlot, c.targetNode, c.targetSlot, this.canvas);
                this.canvas.connections.push(conn);
                this.canvas.connectionLayer.add(conn.line);
            }
        }
        this.canvas.nodeLayer.batchDraw();
        this.canvas.connectionLayer.batchDraw();
        this._paused = false;
    }
}
//# sourceMappingURL=history.js.map