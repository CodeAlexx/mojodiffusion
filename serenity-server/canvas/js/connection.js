"use strict";
/**
 * Bezier connections between output and input ports.
 */
class SFConnection {
    constructor(sourceNodeId, sourceSlot, targetNodeId, targetSlot, canvas) {
        this._dashAnim = null;
        this.sourceNode = sourceNodeId;
        this.sourceSlot = sourceSlot;
        this.targetNode = targetNodeId;
        this.targetSlot = targetSlot;
        this.canvas = canvas;
        const srcType = this._getSourceType();
        this._baseColor = getTypeColor(srcType);
        this._executionState = null;
        this.line = new Konva.Shape({
            sceneFunc: (context, shape) => {
                const src = this._getSourcePos();
                const tgt = this._getTargetPos();
                const dx = Math.max(Math.abs(tgt.x - src.x) * 0.5, 30);
                context.beginPath();
                context.moveTo(src.x, src.y);
                context.bezierCurveTo(src.x + dx, src.y, tgt.x - dx, tgt.y, tgt.x, tgt.y);
                context.fillStrokeShape(shape);
            },
            stroke: this._baseColor,
            strokeWidth: 2,
            opacity: 0.78,
            hitStrokeWidth: 10,
            listening: true,
        });
        // Right-click to delete
        this.line.on('contextmenu', (e) => {
            e.evt.preventDefault();
            e.cancelBubble = true;
            // Will be handled by context menu system
        });
        this._watchDrag();
    }
    _getSourcePos() {
        const node = this.canvas.nodes.get(this.sourceNode);
        return node ? node.getOutputPortPos(this.sourceSlot) : { x: 0, y: 0 };
    }
    _getTargetPos() {
        const node = this.canvas.nodes.get(this.targetNode);
        return node ? node.getInputPortPos(this.targetSlot) : { x: 0, y: 0 };
    }
    _getSourceType() {
        const node = this.canvas.nodes.get(this.sourceNode);
        return node ? node.getOutputType(this.sourceSlot) : '*';
    }
    update() {
        const layer = this.line.getLayer();
        if (layer)
            layer.batchDraw();
    }
    /**
     * Toggle animated dash pattern on this connection (used during execution).
     */
    setAnimated(animated) {
        if (animated) {
            this.line.dash([12, 6]);
            if (!this._dashAnim) {
                this._dashAnim = new Konva.Animation((frame) => {
                    var offset = (frame.time / 40) % 12;
                    this.line.dashOffset(-offset);
                }, this.line.getLayer());
            }
            this._dashAnim.start();
        }
        else {
            if (this._dashAnim) {
                this._dashAnim.stop();
                this._dashAnim = null;
            }
            this.line.dash([]);
            this.line.dashOffset(0);
            this.update();
        }
    }
    /** Strong execution coloring so the live path is obvious at a glance. */
    setExecutionState(state) {
        this._executionState = state || null;
        this.setAnimated(state === 'executing');
        switch (state) {
            case 'executing':
                this.line.stroke('#ffd166');
                this.line.strokeWidth(5);
                this.line.opacity(1);
                this.line.shadowColor('#ffb000');
                this.line.shadowBlur(14);
                this.line.shadowOpacity(0.95);
                break;
            case 'executed':
                this.line.stroke('#2ee98f');
                this.line.strokeWidth(3.5);
                this.line.opacity(0.95);
                this.line.shadowColor('#19c97a');
                this.line.shadowBlur(7);
                this.line.shadowOpacity(0.65);
                break;
            case 'error':
                this.line.stroke('#ff496a');
                this.line.strokeWidth(5);
                this.line.opacity(1);
                this.line.shadowColor('#ff234f');
                this.line.shadowBlur(14);
                this.line.shadowOpacity(0.95);
                break;
            default:
                this.line.stroke(this._baseColor);
                this.line.strokeWidth(2);
                this.line.opacity(0.78);
                this.line.shadowBlur(0);
                this.line.shadowOpacity(0);
        }
        this.update();
    }
    _watchDrag() {
        const srcNode = this.canvas.nodes.get(this.sourceNode);
        const tgtNode = this.canvas.nodes.get(this.targetNode);
        if (srcNode)
            srcNode.group.on('dragmove.conn' + this.sourceNode + this.targetNode, () => this.update());
        if (tgtNode)
            tgtNode.group.on('dragmove.conn' + this.sourceNode + this.targetNode, () => this.update());
    }
    destroy() {
        // Clean up drag listeners
        const srcNode = this.canvas.nodes.get(this.sourceNode);
        const tgtNode = this.canvas.nodes.get(this.targetNode);
        const ns = '.conn' + this.sourceNode + this.targetNode;
        if (srcNode)
            srcNode.group.off('dragmove' + ns);
        if (tgtNode)
            tgtNode.group.off('dragmove' + ns);
        if (this._dashAnim) {
            this._dashAnim.stop();
            this._dashAnim = null;
        }
        this.line.destroy();
    }
}
