"use strict";
/**
 * Main canvas using Konva. Handles zoom, pan, background grid.
 */
class SFCanvas {
    constructor(containerId) {
        // Connection drag handlers
        this._connDragMoveHandler = null;
        this._connDragUpHandler = null;
        // Minimap state
        this._mmCanvas = null;
        this._mmCtx = null;
        this._mmSize = 0;
        this._mmMoveHandler = null;
        this._mmUpHandler = null;
        // DOM-level panning. Stage stays draggable:false so it can't
        // conflict with Konva node drag.
        this._isPanning = false;
        this._panStartX = 0;
        this._panStartY = 0;
        this._stageStartX = 0;
        this._stageStartY = 0;
        this.container = document.getElementById(containerId);
        this.stage = new Konva.Stage({
            container: containerId,
            width: this.container.offsetWidth,
            height: this.container.offsetHeight,
            draggable: false,
        });
        // Layers (order matters: grid behind connections behind nodes)
        this.gridLayer = new Konva.Layer({ listening: false });
        this.connectionLayer = new Konva.Layer();
        this.nodeLayer = new Konva.Layer();
        this.overlayLayer = new Konva.Layer(); // Selection box, drag preview
        this.stage.add(this.gridLayer);
        this.stage.add(this.connectionLayer);
        this.stage.add(this.nodeLayer);
        this.stage.add(this.overlayLayer);
        this.scale = 1;
        this.nodes = new Map(); // id -> SFNode
        this.connections = []; // SFConnection[]
        this.nextNodeId = 1;
        this.nodeInfo = {}; // class_type -> object_info entry
        this.selectedNodes = new Set(); // Set of node IDs
        // Drag-connection state
        this.draggingConnection = null; // { sourceNode, sourceSlot, line }
        this.tempLine = null;
        // Settings
        this.snapToGrid = false;
        this.gridSize = 20;
        this.showMinimap = true;
        // Canvas container: suppress touch/selection defaults (InvokeAI pattern)
        this.container.style.touchAction = 'none';
        this.container.style.userSelect = 'none';
        this.container.style.webkitUserSelect = 'none';
        // Space-to-pan state
        this._spaceDown = false;
        this._prePanDraggable = false;
        // Cleanup registry for unmount
        this._cleanups = [];
        this._setupZoom();
        this._setupResize();
        this._setupStageDrag();
        this._setupMiddleMousePan();
        this._drawGrid();
        this._createViewportControls();
        this._createMinimap();
    }
    _setupZoom() {
        this.stage.on('wheel', (e) => {
            e.evt.preventDefault();
            const oldScale = this.stage.scaleX();
            const pointer = this.stage.getPointerPosition();
            if (!pointer)
                return;
            const scaleBy = 1.08;
            const direction = e.evt.deltaY > 0 ? -1 : 1;
            const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;
            if (newScale < 0.1 || newScale > 5)
                return;
            this.scale = newScale;
            this.stage.scale({ x: newScale, y: newScale });
            const mousePointTo = {
                x: (pointer.x - this.stage.x()) / oldScale,
                y: (pointer.y - this.stage.y()) / oldScale,
            };
            const newPos = {
                x: pointer.x - mousePointTo.x * newScale,
                y: pointer.y - mousePointTo.y * newScale,
            };
            this.stage.position(newPos);
            this._drawGrid();
            this._updateMinimap();
        });
    }
    _setupResize() {
        const ro = new ResizeObserver(() => {
            this.stage.width(this.container.offsetWidth);
            this.stage.height(this.container.offsetHeight);
            this._drawGrid();
        });
        ro.observe(this.container);
    }
    _setupStageDrag() {
        const container = this.stage.container();
        container.addEventListener('mousedown', (e) => {
            // Middle mouse always pans
            if (e.button === 1) {
                this._startPan(e);
                e.preventDefault();
                return;
            }
            // Space+left-click pans
            if (e.button === 0 && this._spaceDown) {
                this._startPan(e);
                return;
            }
        });
        // Left-click on empty space pans (Konva event — fires after node handlers)
        this.stage.on('mousedown', (e) => {
            if (e.evt.button !== 0 || this._spaceDown)
                return;
            // Only pan if click hit the stage itself (empty space)
            if (e.target === this.stage) {
                this._startPan(e.evt);
            }
        });
        container.addEventListener('mousemove', (e) => {
            if (!this._isPanning)
                return;
            this.stage.position({
                x: this._stageStartX + (e.clientX - this._panStartX),
                y: this._stageStartY + (e.clientY - this._panStartY),
            });
            this._drawGrid();
            this.stage.batchDraw();
        });
        container.addEventListener('mouseup', () => {
            if (this._isPanning) {
                this._isPanning = false;
                this.stage.container().style.cursor = '';
                this._updateMinimap();
            }
        });
    }
    _startPan(e) {
        this._isPanning = true;
        this._panStartX = e.clientX;
        this._panStartY = e.clientY;
        this._stageStartX = this.stage.x();
        this._stageStartY = this.stage.y();
        this.stage.container().style.cursor = 'grabbing';
    }
    _setupMiddleMousePan() {
        // consolidated into _setupStageDrag
    }
    // Space-to-pan: enable temporary pan mode
    startSpacePan() {
        if (this._spaceDown)
            return;
        this._spaceDown = true;
        this._prePanDraggable = false;
        this.stage.container().style.cursor = 'grab';
    }
    endSpacePan() {
        if (!this._spaceDown)
            return;
        this._spaceDown = false;
        this.stage.container().style.cursor = '';
    }
    _drawGrid() {
        this.gridLayer.destroyChildren();
        const gridSize = 20;
        const majorEvery = 5;
        const scale = this.stage.scaleX();
        const offset = this.stage.position();
        const width = this.stage.width();
        const height = this.stage.height();
        // Don't draw grid when too zoomed out
        if (scale < 0.3) {
            this.gridLayer.batchDraw();
            return;
        }
        const startX = Math.floor(-offset.x / scale / gridSize) * gridSize - gridSize;
        const startY = Math.floor(-offset.y / scale / gridSize) * gridSize - gridSize;
        const endX = startX + width / scale + gridSize * 2;
        const endY = startY + height / scale + gridSize * 2;
        const strokeWidth = 1 / scale;
        for (let x = startX; x < endX; x += gridSize) {
            const isMajor = Math.round(x / gridSize) % majorEvery === 0;
            this.gridLayer.add(new Konva.Line({
                points: [x, startY, x, endY],
                stroke: isMajor ? '#252545' : '#1e1e3e',
                strokeWidth: strokeWidth,
                listening: false,
            }));
        }
        for (let y = startY; y < endY; y += gridSize) {
            const isMajor = Math.round(y / gridSize) % majorEvery === 0;
            this.gridLayer.add(new Konva.Line({
                points: [startX, y, endX, y],
                stroke: isMajor ? '#252545' : '#1e1e3e',
                strokeWidth: strokeWidth,
                listening: false,
            }));
        }
        this.gridLayer.batchDraw();
    }
    addNode(nodeType, x, y, info) {
        const id = String(this.nextNodeId++);
        const node = new SFNode(id, nodeType, x, y, info, this);
        this.nodes.set(id, node);
        this.nodeLayer.add(node.group);
        this.nodeLayer.batchDraw();
        return node;
    }
    removeNode(id) {
        const node = this.nodes.get(id);
        if (!node)
            return;
        // Remove connections to/from this node
        this.connections = this.connections.filter(c => {
            if (c.sourceNode === id || c.targetNode === id) {
                c.destroy();
                return false;
            }
            return true;
        });
        node.destroy();
        this.nodes.delete(id);
        this.selectedNodes.delete(id);
        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
    }
    addConnection(sourceNodeId, sourceSlot, targetNodeId, targetSlot) {
        // Check if target input already has a connection — remove it
        this.connections = this.connections.filter(c => {
            if (c.targetNode === targetNodeId && c.targetSlot === targetSlot) {
                c.destroy();
                return false;
            }
            return true;
        });
        const conn = new SFConnection(sourceNodeId, sourceSlot, targetNodeId, targetSlot, this);
        this.connections.push(conn);
        this.connectionLayer.add(conn.line);
        this.connectionLayer.batchDraw();
        // Hide widget for connected input
        const targetNode = this.nodes.get(targetNodeId);
        if (targetNode)
            targetNode.updateWidgetVisibility();
        return conn;
    }
    removeConnection(conn) {
        const idx = this.connections.indexOf(conn);
        if (idx >= 0) {
            this.connections.splice(idx, 1);
            conn.destroy();
            this.connectionLayer.batchDraw();
            // Show widget for disconnected input
            const targetNode = this.nodes.get(conn.targetNode);
            if (targetNode)
                targetNode.updateWidgetVisibility();
        }
    }
    getConnectionsForNode(nodeId) {
        return this.connections.filter(c => c.sourceNode === nodeId || c.targetNode === nodeId);
    }
    isInputConnected(nodeId, slotIndex) {
        return this.connections.some(c => c.targetNode === nodeId && c.targetSlot === slotIndex);
    }
    getWorldPosition(screenX, screenY) {
        const transform = this.stage.getAbsoluteTransform().copy().invert();
        return transform.point({ x: screenX, y: screenY });
    }
    centerView() {
        if (this.nodes.size === 0)
            return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const sw = this.stage.width();
        const sh = this.stage.height();
        this.stage.position({ x: sw / 2 - cx * this.scale, y: sh / 2 - cy * this.scale });
        this._drawGrid();
    }
    selectNode(id, addToSelection) {
        if (!addToSelection) {
            this.deselectAll();
        }
        this.selectedNodes.add(id);
        const node = this.nodes.get(id);
        if (node)
            node.setSelected(true);
    }
    deselectAll() {
        this.selectedNodes.forEach(id => {
            const node = this.nodes.get(id);
            if (node)
                node.setSelected(false);
        });
        this.selectedNodes.clear();
    }
    deleteSelected() {
        const ids = [...this.selectedNodes];
        ids.forEach(id => this.removeNode(id));
    }
    // Start dragging a new connection from an output port
    startConnectionDrag(nodeId, slotIndex, startPos) {
        this.draggingConnection = {
            sourceNode: nodeId,
            sourceSlot: slotIndex,
            startPos: startPos,
        };
        // Disable stage drag while connecting
        this.stage.draggable(false);
        const srcNode = this.nodes.get(nodeId);
        const srcType = srcNode ? srcNode.getOutputType(slotIndex) : '*';
        const wireColor = getTypeColor(srcType);
        this.tempLine = new Konva.Shape({
            sceneFunc: (context, shape) => {
                if (!this.draggingConnection)
                    return;
                const src = this.draggingConnection.startPos;
                const end = this.draggingConnection.endPos || src;
                const dx = Math.max(Math.abs(end.x - src.x) * 0.5, 30);
                context.beginPath();
                context.moveTo(src.x, src.y);
                context.bezierCurveTo(src.x + dx, src.y, end.x - dx, end.y, end.x, end.y);
                context.fillStrokeShape(shape);
            },
            stroke: wireColor,
            strokeWidth: 2,
            dash: [6, 3],
            listening: false,
        });
        this.overlayLayer.add(this.tempLine);
        // Track mouse movement to update the temp line
        this._connDragMoveHandler = (e) => {
            if (!this.draggingConnection)
                return;
            const pointer = this.stage.getPointerPosition();
            if (!pointer)
                return;
            this.draggingConnection.endPos = this.getWorldPosition(pointer.x, pointer.y);
            this.overlayLayer.batchDraw();
        };
        // End drag on mouseup anywhere on stage
        this._connDragUpHandler = (e) => {
            // If endConnectionDrag was already called by a port mouseup, this is a no-op
            if (this.draggingConnection) {
                this.cancelConnectionDrag();
            }
        };
        this.stage.on('mousemove.conndrag', this._connDragMoveHandler);
        // Use window mouseup with a tiny delay so port mouseup fires first
        window.addEventListener('mouseup', this._connDragUpHandler, { once: true });
    }
    endConnectionDrag(targetNodeId, targetSlot) {
        if (!this.draggingConnection)
            return;
        if (targetNodeId != null && targetSlot !== undefined) {
            const srcNode = this.nodes.get(this.draggingConnection.sourceNode);
            const tgtNode = this.nodes.get(targetNodeId);
            if (srcNode && tgtNode) {
                const srcType = srcNode.getOutputType(this.draggingConnection.sourceSlot);
                const tgtType = tgtNode.getInputType(targetSlot);
                if (typesCompatible(srcType, tgtType)) {
                    if (this.draggingConnection.sourceNode !== targetNodeId) {
                        this.addConnection(this.draggingConnection.sourceNode, this.draggingConnection.sourceSlot, targetNodeId, targetSlot);
                    }
                }
            }
        }
        this._cleanupConnectionDrag();
    }
    cancelConnectionDrag() {
        this._cleanupConnectionDrag();
    }
    _cleanupConnectionDrag() {
        if (this.tempLine) {
            this.tempLine.destroy();
            this.tempLine = null;
        }
        this.draggingConnection = null;
        this.stage.off('mousemove.conndrag');
        // Remove window listener if it hasn't fired yet
        if (this._connDragUpHandler) {
            window.removeEventListener('mouseup', this._connDragUpHandler);
            this._connDragUpHandler = null;
        }
        this._connDragMoveHandler = null;
        this.overlayLayer.batchDraw();
    }
    // ── Phase 1: Fit View ──
    fitView(animate) {
        if (this.nodes.size === 0)
            return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        const pad = 60;
        const bw = maxX - minX + pad * 2;
        const bh = maxY - minY + pad * 2;
        const sw = this.stage.width();
        const sh = this.stage.height();
        const newScale = Math.min(sw / bw, sh / bh, 2);
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const newX = sw / 2 - cx * newScale;
        const newY = sh / 2 - cy * newScale;
        if (animate) {
            const startScale = this.stage.scaleX();
            const startPos = this.stage.position();
            const duration = 300;
            const startTime = performance.now();
            const anim = () => {
                const t = Math.min((performance.now() - startTime) / duration, 1);
                const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // easeInOutQuad
                const s = startScale + (newScale - startScale) * ease;
                const x = startPos.x + (newX - startPos.x) * ease;
                const y = startPos.y + (newY - startPos.y) * ease;
                this.scale = s;
                this.stage.scale({ x: s, y: s });
                this.stage.position({ x, y });
                this._drawGrid();
                this._updateMinimap();
                if (t < 1)
                    requestAnimationFrame(anim);
            };
            requestAnimationFrame(anim);
        }
        else {
            this.scale = newScale;
            this.stage.scale({ x: newScale, y: newScale });
            this.stage.position({ x: newX, y: newY });
            this._drawGrid();
            this._updateMinimap();
        }
    }
    // ── Phase 1: Snap to Grid ──
    snapNodeToGrid(node) {
        if (!this.snapToGrid)
            return;
        const g = this.gridSize;
        node.x = Math.round(node.x / g) * g;
        node.y = Math.round(node.y / g) * g;
        node.group.x(node.x);
        node.group.y(node.y);
        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
    }
    // ── Phase 1: Viewport Controls ──
    _createViewportControls() {
        const html = `
            <div class="viewport-controls" id="viewport-controls">
                <button id="vc-zoom-in" title="Zoom In (+)">+</button>
                <button id="vc-zoom-out" title="Zoom Out (-)">−</button>
                <button id="vc-fit" title="Fit View (H)">⊞</button>
                <button id="vc-minimap" title="Toggle Minimap" class="active">◫</button>
                <button id="vc-snap" title="Snap to Grid (Shift+G)">#</button>
            </div>`;
        this.container.insertAdjacentHTML('beforeend', html);
        const self = this;
        document.getElementById('vc-zoom-in').addEventListener('click', () => {
            self.scale = Math.min(self.scale * 1.25, 5);
            self.stage.scale({ x: self.scale, y: self.scale });
            self._drawGrid();
            self._updateMinimap();
        });
        document.getElementById('vc-zoom-out').addEventListener('click', () => {
            self.scale = Math.max(self.scale / 1.25, 0.1);
            self.stage.scale({ x: self.scale, y: self.scale });
            self._drawGrid();
            self._updateMinimap();
        });
        document.getElementById('vc-fit').addEventListener('click', () => {
            self.fitView(true);
        });
        document.getElementById('vc-minimap').addEventListener('click', function () {
            self.showMinimap = !self.showMinimap;
            this.classList.toggle('active', self.showMinimap);
            const mm = document.getElementById('canvas-minimap');
            if (mm)
                mm.style.display = self.showMinimap ? '' : 'none';
        });
        document.getElementById('vc-snap').addEventListener('click', function () {
            self.snapToGrid = !self.snapToGrid;
            this.classList.toggle('active', self.snapToGrid);
        });
    }
    // ── Phase 1: Minimap ──
    _createMinimap() {
        const mmSize = 160;
        const html = `<canvas id="canvas-minimap" class="canvas-minimap" width="${mmSize}" height="${mmSize}"></canvas>`;
        this.container.insertAdjacentHTML('beforeend', html);
        this._mmCanvas = document.getElementById('canvas-minimap');
        this._mmCtx = this._mmCanvas.getContext('2d');
        this._mmSize = mmSize;
        // Click/drag on minimap to navigate
        const self = this;
        let mmDragging = false;
        const navigateFromMM = (e) => {
            const rect = self._mmCanvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const bounds = self._getNodeBounds();
            if (!bounds)
                return;
            const pad = 60;
            const bw = bounds.maxX - bounds.minX + pad * 2;
            const bh = bounds.maxY - bounds.minY + pad * 2;
            const mmScale = Math.min(mmSize / bw, mmSize / bh);
            const offX = (mmSize - bw * mmScale) / 2;
            const offY = (mmSize - bh * mmScale) / 2;
            const worldX = (mx - offX) / mmScale + bounds.minX - pad;
            const worldY = (my - offY) / mmScale + bounds.minY - pad;
            const sw = self.stage.width();
            const sh = self.stage.height();
            self.stage.position({
                x: sw / 2 - worldX * self.scale,
                y: sh / 2 - worldY * self.scale
            });
            self._drawGrid();
            self._updateMinimap();
        };
        this._mmCanvas.addEventListener('mousedown', (e) => { mmDragging = true; navigateFromMM(e); });
        this._mmMoveHandler = (e) => { if (mmDragging)
            navigateFromMM(e); };
        this._mmUpHandler = () => { mmDragging = false; };
        window.addEventListener('mousemove', this._mmMoveHandler);
        window.addEventListener('mouseup', this._mmUpHandler);
        // Update minimap on zoom/pan
        this.stage.on('dragmove.minimap', () => this._updateMinimap());
    }
    _getNodeBounds() {
        if (this.nodes.size === 0)
            return null;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        return { minX, minY, maxX, maxY };
    }
    _updateMinimap() {
        if (!this.showMinimap || !this._mmCtx)
            return;
        const ctx = this._mmCtx;
        const size = this._mmSize;
        ctx.clearRect(0, 0, size, size);
        const bounds = this._getNodeBounds();
        if (!bounds)
            return;
        const pad = 60;
        const bw = bounds.maxX - bounds.minX + pad * 2;
        const bh = bounds.maxY - bounds.minY + pad * 2;
        const mmScale = Math.min(size / bw, size / bh);
        const offX = (size - bw * mmScale) / 2;
        const offY = (size - bh * mmScale) / 2;
        // Draw nodes
        this.nodes.forEach(node => {
            const nx = (node.x - bounds.minX + pad) * mmScale + offX;
            const ny = (node.y - bounds.minY + pad) * mmScale + offY;
            const nw = node.width * mmScale;
            const nh = node.height * mmScale;
            const selected = this.selectedNodes.has(node.id);
            ctx.fillStyle = selected ? '#5b8def' : '#3a3a6a';
            ctx.fillRect(nx, ny, Math.max(nw, 3), Math.max(nh, 2));
        });
        // Draw viewport rect
        const stagePos = this.stage.position();
        const sw = this.stage.width();
        const sh = this.stage.height();
        const vpLeft = (-stagePos.x / this.scale - bounds.minX + pad) * mmScale + offX;
        const vpTop = (-stagePos.y / this.scale - bounds.minY + pad) * mmScale + offY;
        const vpW = (sw / this.scale) * mmScale;
        const vpH = (sh / this.scale) * mmScale;
        ctx.strokeStyle = 'rgba(91, 141, 239, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(vpLeft, vpTop, vpW, vpH);
        ctx.fillStyle = 'rgba(91, 141, 239, 0.08)';
        ctx.fillRect(vpLeft, vpTop, vpW, vpH);
    }
    // ── Phase 2: Auto-Layout ──
    autoLayout() {
        if (this.nodes.size === 0)
            return;
        // Build adjacency: node → [downstream nodes]
        const downstream = new Map();
        const upstream = new Map();
        this.nodes.forEach((_, id) => { downstream.set(id, new Set()); upstream.set(id, new Set()); });
        this.connections.forEach(c => {
            if (downstream.has(c.sourceNode))
                downstream.get(c.sourceNode).add(c.targetNode);
            if (upstream.has(c.targetNode))
                upstream.get(c.targetNode).add(c.sourceNode);
        });
        // Assign layers (longest path from sources, with cycle protection)
        const layers = new Map();
        const assignLayer = (id, depth, stack) => {
            if (stack.has(id))
                return; // cycle detected — break
            const cur = layers.get(id) || 0;
            if (depth <= cur)
                return; // already visited at equal or greater depth
            layers.set(id, depth);
            stack.add(id);
            downstream.get(id).forEach((child) => assignLayer(child, depth + 1, new Set(stack)));
            stack.delete(id);
        };
        // Start from nodes with no upstream
        this.nodes.forEach((_, id) => {
            if (upstream.get(id).size === 0)
                assignLayer(id, 0, new Set());
        });
        // Handle disconnected nodes
        this.nodes.forEach((_, id) => { if (!layers.has(id))
            layers.set(id, 0); });
        // Group nodes by layer
        const layerGroups = new Map();
        layers.forEach((layer, id) => {
            if (!layerGroups.has(layer))
                layerGroups.set(layer, []);
            layerGroups.get(layer).push(id);
        });
        // Position nodes
        const hGap = 80;
        const vGap = 40;
        let xPos = 0;
        const sortedLayers = [...layerGroups.keys()].sort((a, b) => a - b);
        sortedLayers.forEach(layerIdx => {
            const nodeIds = layerGroups.get(layerIdx);
            let maxW = 0;
            let yPos = 0;
            nodeIds.forEach((id) => {
                const node = this.nodes.get(id);
                if (!node)
                    return;
                node.x = xPos;
                node.y = yPos;
                node.group.x(xPos);
                node.group.y(yPos);
                maxW = Math.max(maxW, node.width);
                yPos += node.height + vGap;
            });
            xPos += maxW + hGap;
        });
        this.nodeLayer.batchDraw();
        this.connectionLayer.batchDraw();
        this.fitView(true);
        this._updateMinimap();
    }
    // ── Phase 1: Edge animation support ──
    setEdgesAnimated(animated) {
        this.connections.forEach(conn => {
            if (conn.setAnimated)
                conn.setAnimated(animated);
        });
    }
    // Disconnect all connections for a given node
    disconnectAllForNode(nodeId) {
        const toRemove = this.connections.filter(c => c.sourceNode === nodeId || c.targetNode === nodeId);
        toRemove.forEach(c => this.removeConnection(c));
    }
    // Cleanup all event listeners (call on unmount)
    destroy() {
        // Run registered cleanups
        this._cleanups.forEach(fn => fn());
        this._cleanups = [];
        // Remove minimap window listeners
        if (this._mmMoveHandler)
            window.removeEventListener('mousemove', this._mmMoveHandler);
        if (this._mmUpHandler)
            window.removeEventListener('mouseup', this._mmUpHandler);
        // Kill connection drag state
        this._cleanupConnectionDrag();
        // Destroy Konva stage
        this.stage.destroy();
    }
}
//# sourceMappingURL=canvas.js.map