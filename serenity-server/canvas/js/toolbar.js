"use strict";
/**
 * Toolbar: queue, interrupt, clear, save, load. Execution status.
 */
class SFToolbar {
    constructor(canvas, api) {
        this.canvas = canvas;
        this.api = api;
        this.queueBtn = document.getElementById('btn-queue');
        this.interruptBtn = document.getElementById('btn-interrupt');
        this.saveBtn = document.getElementById('btn-save');
        this.loadBtn = document.getElementById('btn-load');
        this.statusIndicator = document.getElementById('status-indicator');
        this.queueCount = document.getElementById('queue-count');
        this.workflowStatusLine = document.getElementById('workflow-status-line');
        this.workflowStatusText = document.getElementById('workflow-status-text');
        this.fileInput = document.getElementById('file-input');
        this._badgeContainer = null; // lazy-created div for status badges
        this._badges = new Map(); // nodeId -> badge element
        this._notes = []; // {element, id}
        this._nextNoteId = 1;
        this._isRunning = false;
        this._activePromptId = null;
        this._awaitingPrompt = false;
        this._executionPhase = 'idle';
        this._setupButtons();
        this._setupKeyboard();
        this._setupApiEvents();
        this._patchCanvasEdgeAnimation();
        this._setupNoteButton();
        this._setupBadgeReposition();
    }
    _setupButtons() {
        this.queueBtn.addEventListener('click', () => this._queuePrompt());
        this.interruptBtn.addEventListener('click', () => {
            this._setWorkflowStatus('loading', 'Stopping workflow…');
            this.api.interrupt();
        });
        this.saveBtn.addEventListener('click', () => this._saveWorkflow());
        this.loadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => {
            const target = e.target;
            const file = target.files[0];
            if (file)
                this._loadWorkflow(file);
            this.fileInput.value = '';
        });
        // Templates are now handled by shell.js dropdown
    }
    _setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            if ((localStorage.getItem('sf-active-tab') || 'generate') !== 'workflows') {
                return;
            }
            const target = e.target;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable) {
                return;
            }
            // Ctrl+Enter - queue
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this._queuePrompt();
            }
            // Space - quick queue
            if (e.key === ' ' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
                e.preventDefault();
                this._queuePrompt();
            }
            // Ctrl+S - save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this._saveWorkflow();
            }
            // Ctrl+O - load
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                this.fileInput.click();
            }
        });
    }
    _setupApiEvents() {
        this.api.on('connected', () => {
            this._setStatus('idle');
            if (!this._activePromptId && !this._awaitingPrompt)
                this._setWorkflowStatus('idle', 'Ready');
        });
        this.api.on('disconnected', () => {
            this._setStatus('error');
            this._setWorkflowStatus('error', 'Disconnected from generation server');
        });
        this.api.on('status', (data) => {
            if (data && data.status) {
                const qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
                this.queueCount.textContent = String(qr);
                this._isRunning = qr > 0;
                this._setStatus(qr > 0 ? 'running' : 'idle');
            }
        });
        this.api.on('execution_start', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            this._setRunControls(true);
            this._setStatus('running');
            this._setWorkflowStatus('loading', 'Loading model and conditioning…');
            this._applyWorkflowPhase('loading');
        });
        this.api.on('executing', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            if (data && data.node) {
                const node = this.canvas.nodes.get(data.node);
                if (node) {
                    const phase = this._nodePhase(node);
                    this._applyWorkflowPhase(phase);
                    node.setExecutionState('executing');
                    this._updateBadge(data.node, 'executing');
                    const label = node.info.display_name || node.nodeType;
                    if (phase === 'sampling')
                        this._setWorkflowStatus('running', 'Sampling · ' + label);
                    else if (phase === 'output')
                        this._setWorkflowStatus('output', 'Decoding and saving · ' + label);
                    else
                        this._setWorkflowStatus('loading', 'Loading · ' + label);
                }
            }
            else if (data && data.node === null) {
                this._setWorkflowStatus('output', 'Decoding and saving output…');
                this._applyWorkflowPhase('output');
            }
        });
        this.api.on('executed', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            this._setWorkflowStatus('output', 'Output written · finalizing…');
            this._applyWorkflowPhase('output');
            if (data && data.node) {
                const node = this.canvas.nodes.get(data.node);
                if (node)
                    node.setExecutionState('executed');
                this._updateBadge(data.node, 'completed');
                // Check for image outputs
                if (data.output && data.output.images) {
                    data.output.images.forEach((img) => {
                        if (sfPreview) {
                            sfPreview.showImage(this.api.viewUrl(img.filename, img.subfolder, img.type));
                        }
                    });
                }
                // Check for video outputs (SaveVideo, SaveAnimatedWEBP)
                if (data.output && data.output.videos) {
                    data.output.videos.forEach((vid) => {
                        if (sfPreview) {
                            sfPreview.showVideo(this.api.viewUrl(vid.filename, vid.subfolder, vid.type));
                        }
                    });
                }
            }
        });
        this.api.on('execution_success', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            this._awaitingPrompt = false;
            this._isRunning = false;
            this._setRunControls(false);
            this._setStatus('done');
            this._setWorkflowStatus('complete', 'Complete · output ready');
            this._applyWorkflowPhase('complete');
        });
        this.api.on('execution_error', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            if (data && data.node_id) {
                const node = this.canvas.nodes.get(data.node_id);
                if (node)
                    node.setExecutionState('error');
                this._updateBadge(data.node_id, 'error');
            }
            this._awaitingPrompt = false;
            this._isRunning = false;
            this._setRunControls(false);
            this._setStatus('error');
            this._setWorkflowStatus('error', 'Failed · ' + (data.exception_message || 'unknown error'));
            this._applyWorkflowPhase('error');
            this._setEdgesAnimated(false);
            this._toast('Execution error: ' + (data.exception_message || 'unknown'), 'error');
        });
        this.api.on('progress', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            const value = Number(data && data.value != null ? data.value : 0);
            const max = Number(data && data.max != null ? data.max : 0);
            if (max > 0 && value >= max) {
                this._setWorkflowStatus('output', 'Denoise complete · decoding and saving…');
                this._applyWorkflowPhase('output');
            }
            else {
                this._setWorkflowStatus('running', 'Sampling · Step ' + value + ' / ' + max);
                this._applyWorkflowPhase('sampling');
            }
        });
        this.api.on('execution_interrupted', (data) => {
            if (!this._acceptJobEvent(data))
                return;
            this._awaitingPrompt = false;
            this._isRunning = false;
            this._setRunControls(false);
            this._setStatus('error');
            this._setWorkflowStatus('error', 'Stopped');
            this._applyWorkflowPhase('error');
        });
    }
    async _queuePrompt() {
        const { prompt } = serializeWorkflow(this.canvas);
        if (Object.keys(prompt).length === 0) {
            this._toast('No nodes to execute', 'error');
            return;
        }
        this._activePromptId = null;
        this._awaitingPrompt = true;
        this._clearBadges();
        this._setRunControls(true);
        this._setStatus('running');
        this._setWorkflowStatus('loading', 'Submitting workflow…');
        this._applyWorkflowPhase('loading');
        try {
            const result = await this.api.queuePrompt(prompt);
            if (result.error) {
                throw new Error(result.error);
            }
            if (result.video_result) {
                const video = result.video_result;
                let src = video.mp4_url || '';
                if (!src && video.mp4 && video.video_id) {
                    const name = String(video.mp4).split('/').pop();
                    src = '/out/' + encodeURIComponent(video.video_id) + '/' + encodeURIComponent(name);
                }
                if (!src)
                    throw new Error('video completed without an MP4 URL');
                if (sfPreview)
                    sfPreview.showVideo(src);
                this._awaitingPrompt = false;
                this._setRunControls(false);
                this._setStatus('done');
                this._setWorkflowStatus('complete', 'Complete · video ready');
                this._applyWorkflowPhase('complete');
                this._toast('Video complete', 'success');
            }
            else {
                this._activePromptId = result.prompt_id || this._activePromptId;
                this._awaitingPrompt = false;
                this._setWorkflowStatus('loading', 'Loading model and conditioning…');
            }
        }
        catch (e) {
            this._awaitingPrompt = false;
            this._setRunControls(false);
            this._setStatus('error');
            this._setWorkflowStatus('error', 'Queue failed · ' + (e instanceof Error ? e.message : String(e)));
            this._applyWorkflowPhase('error');
            this._toast('Failed to queue: ' + (e instanceof Error ? e.message : String(e)), 'error');
        }
    }
    _saveWorkflow() {
        const { prompt, nodePositions } = serializeWorkflow(this.canvas);
        const workflow = {
            version: 2,
            prompt: prompt,
            nodes: nodePositions,
            metadata: typeof workflowMeta !== 'undefined' ? { ...workflowMeta } : {},
        };
        const filename = (workflowMeta && workflowMeta.name ? workflowMeta.name.replace(/[^a-z0-9_-]/gi, '_') : 'workflow') + '.json';
        const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        this._toast('Workflow saved', 'success');
    }
    _loadWorkflow(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
                this._toast('Workflow loaded', 'success');
            }
            catch (err) {
                this._toast('Failed to load: ' + (err instanceof Error ? err.message : String(err)), 'error');
            }
        };
        reader.readAsText(file);
    }
    /**
     * Load a workflow from a URL (for default/template workflows).
     */
    async loadWorkflowFromUrl(url) {
        try {
            const r = await fetch(url + '?t=' + Date.now(), { cache: 'no-store' });
            if (!r.ok)
                throw new Error('HTTP ' + r.status);
            const data = await r.json();
            loadWorkflow(this.canvas, data, this.canvas.nodeInfo);
            this.canvas.autoLayout();
            this._toast('Workflow loaded', 'success');
        }
        catch (err) {
            this._toast('Failed to load workflow: ' + (err instanceof Error ? err.message : String(err)), 'error');
        }
    }
    _setStatus(status) {
        const el = this.statusIndicator;
        el.className = 'status-' + status;
    }
    _setWorkflowStatus(kind, text) {
        this._executionPhase = kind || 'idle';
        if (this.workflowStatusLine)
            this.workflowStatusLine.className = 'wf-status-line is-' + this._executionPhase;
        if (this.workflowStatusText)
            this.workflowStatusText.textContent = text || 'Ready';
    }
    _setRunControls(running) {
        const label = this.queueBtn ? this.queueBtn.querySelector('span') : null;
        if (label)
            label.textContent = running ? 'Running...' : 'Generate';
        if (this.queueBtn)
            this.queueBtn.classList.toggle('running', running);
        if (this.interruptBtn)
            this.interruptBtn.disabled = !running;
    }
    _acceptJobEvent(data) {
        if (!this._activePromptId && !this._awaitingPrompt)
            return false;
        const promptId = data && data.prompt_id ? String(data.prompt_id) : '';
        if (this._activePromptId && promptId && promptId !== this._activePromptId)
            return false;
        if (!this._activePromptId && promptId)
            this._activePromptId = promptId;
        return true;
    }
    _nodePhase(node) {
        const type = node ? String(node.nodeType || '').toLowerCase() : '';
        if (type.includes('sampler') || type.includes('flowedit') || type.includes('denoise'))
            return 'sampling';
        if (type.includes('decode') || type.includes('save') || type.includes('preview') ||
            type.includes('upscale') || type.includes('video combine'))
            return 'output';
        return 'loading';
    }
    _applyWorkflowPhase(phase) {
        const states = new Map();
        this.canvas.nodes.forEach((node, id) => {
            const nodePhase = this._nodePhase(node);
            let state = null;
            if (phase === 'loading') {
                state = nodePhase === 'loading' ? 'executing' : null;
            }
            else if (phase === 'sampling') {
                state = nodePhase === 'loading' ? 'executed' :
                    (nodePhase === 'sampling' ? 'executing' : null);
            }
            else if (phase === 'output') {
                state = nodePhase === 'output' ? 'executing' : 'executed';
            }
            else if (phase === 'complete') {
                state = 'executed';
            }
            else if (phase === 'error') {
                state = node._executionState === 'executing' ? 'error' : node._executionState;
            }
            states.set(id, state);
            if (node._executionState !== state)
                node.setExecutionState(state);
        });
        this.canvas.connections.forEach((conn) => {
            const sourceState = states.get(conn.sourceNode);
            const targetState = states.get(conn.targetNode);
            let state = null;
            if (sourceState === 'error' || targetState === 'error')
                state = 'error';
            else if (targetState === 'executing')
                state = 'executing';
            else if (sourceState === 'executed' && targetState === 'executed')
                state = 'executed';
            if (conn.setExecutionState)
                conn.setExecutionState(state);
        });
        this.canvas.nodeLayer.batchDraw();
        this.canvas.connectionLayer.batchDraw();
    }
    _clearBadges() {
        this._badges.forEach(function (badge) { badge.remove(); });
        this._badges.clear();
    }
    _toast(msg, variant) {
        const existing = document.querySelector('.toast');
        if (existing)
            existing.remove();
        const div = document.createElement('div');
        div.className = 'toast' + (variant ? ' toast-' + variant : '');
        div.textContent = msg;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), variant === 'error' ? 5000 : 3000);
    }
    // -- Edge animation during execution --
    _patchCanvasEdgeAnimation() {
        var self = this;
        // Monkey-patch setEdgesAnimated onto sfCanvas once available
        var tryPatch = function () {
            if (typeof sfCanvas !== 'undefined' && sfCanvas && !sfCanvas.setEdgesAnimated) {
                sfCanvas.setEdgesAnimated = function (animated) {
                    this.connections.forEach(function (conn) {
                        if (conn.setAnimated)
                            conn.setAnimated(animated);
                    });
                };
            }
        };
        tryPatch();
        // Also try after a tick in case sfCanvas is set later
        setTimeout(tryPatch, 0);
    }
    _setEdgesAnimated(animated) {
        if (typeof sfCanvas !== 'undefined' && sfCanvas && sfCanvas.setEdgesAnimated) {
            sfCanvas.setEdgesAnimated(animated);
        }
    }
    // -- Node status badges --
    _ensureBadgeContainer() {
        if (this._badgeContainer)
            return this._badgeContainer;
        var container = document.getElementById('canvas-container');
        if (!container)
            return null;
        var div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.top = '0';
        div.style.left = '0';
        div.style.width = '100%';
        div.style.height = '100%';
        div.style.pointerEvents = 'none';
        div.style.overflow = 'hidden';
        div.style.zIndex = '200';
        container.appendChild(div);
        this._badgeContainer = div;
        return div;
    }
    _updateBadge(nodeId, state) {
        var container = this._ensureBadgeContainer();
        if (!container)
            return;
        // Remove existing badge for this node
        var existing = this._badges.get(nodeId);
        if (existing) {
            existing.remove();
            this._badges.delete(nodeId);
        }
        if (!state)
            return;
        var node = this.canvas.nodes.get(nodeId);
        if (!node)
            return;
        var badge = document.createElement('div');
        badge.className = 'node-status-badge ' + state;
        if (state === 'executing') {
            badge.textContent = '\u25CF'; // filled circle
        }
        else if (state === 'completed') {
            badge.textContent = '\u2713'; // check
        }
        else if (state === 'error') {
            badge.textContent = '\u2717'; // X
        }
        container.appendChild(badge);
        this._badges.set(nodeId, badge);
        this._positionBadge(nodeId, badge);
    }
    _positionBadge(nodeId, badge) {
        var node = this.canvas.nodes.get(nodeId);
        if (!node || !node.group)
            return;
        var stage = this.canvas.stage;
        var stageBox = stage.container().getBoundingClientRect();
        var transform = stage.getAbsoluteTransform();
        // Node position in screen coordinates
        var pos = transform.point({ x: node.x, y: node.y });
        var nodeW = (node.width || 200) * this.canvas.scale;
        badge.style.left = (pos.x + nodeW - 6) + 'px';
        badge.style.top = (pos.y - 6) + 'px';
    }
    _repositionAllBadges() {
        var self = this;
        this._badges.forEach(function (badge, nodeId) {
            self._positionBadge(nodeId, badge);
        });
    }
    _setupBadgeReposition() {
        var self = this;
        // Reposition badges on pan, zoom, and node drag
        this.canvas.stage.on('dragmove.badges', function () {
            self._repositionAllBadges();
        });
        this.canvas.stage.on('scaleChange.badges', function () {
            self._repositionAllBadges();
        });
        this.canvas.nodeLayer.on('dragmove.badges', function () {
            self._repositionAllBadges();
        });
    }
    // -- Notes nodes --
    _setupNoteButton() {
        // Insert a "Note" button in the toolbar-center area
        var center = document.querySelector('#toolbar .toolbar-center');
        if (!center)
            return;
        var btn = document.createElement('button');
        btn.id = 'btn-add-note';
        btn.className = 'wf-btn-ghost';
        btn.title = 'Add Note';
        btn.innerHTML = '<span>+ Note</span>';
        center.appendChild(btn);
        var self = this;
        btn.addEventListener('click', function () {
            self._addNote();
        });
    }
    _addNote() {
        var canvasContainer = document.getElementById('canvas-container');
        if (!canvasContainer)
            return;
        var noteId = 'note-' + this._nextNoteId++;
        // Place in center of visible canvas area
        var rect = canvasContainer.getBoundingClientRect();
        var startX = rect.width / 2 - 80;
        var startY = rect.height / 2 - 40;
        var note = document.createElement('div');
        note.className = 'canvas-note';
        note.id = noteId;
        note.style.left = startX + 'px';
        note.style.top = startY + 'px';
        var textarea = document.createElement('textarea');
        textarea.placeholder = 'Type a note...';
        textarea.rows = 3;
        note.appendChild(textarea);
        // Close button
        var closeBtn = document.createElement('button');
        closeBtn.textContent = '\u00d7';
        closeBtn.style.cssText = 'position:absolute;top:2px;right:4px;background:none;border:none;color:#ffd866;cursor:pointer;font-size:14px;line-height:1;padding:2px 4px;opacity:0.6;';
        closeBtn.addEventListener('mouseenter', function () { closeBtn.style.opacity = '1'; });
        closeBtn.addEventListener('mouseleave', function () { closeBtn.style.opacity = '0.6'; });
        closeBtn.addEventListener('click', function () {
            note.remove();
        });
        note.appendChild(closeBtn);
        // Make draggable
        var dragging = false, dragOffX = 0, dragOffY = 0;
        note.addEventListener('mousedown', function (e) {
            if (e.target === textarea)
                return;
            dragging = true;
            dragOffX = e.clientX - note.offsetLeft;
            dragOffY = e.clientY - note.offsetTop;
            e.preventDefault();
        });
        var onMouseMove = function (e) {
            if (!dragging)
                return;
            note.style.left = (e.clientX - dragOffX) + 'px';
            note.style.top = (e.clientY - dragOffY) + 'px';
        };
        var onMouseUp = function () {
            dragging = false;
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        // Clean up listeners when note is removed
        var self = this;
        var origCloseHandler = closeBtn.onclick;
        closeBtn.addEventListener('click', function () {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            self._notes = self._notes.filter(function (n) { return n.id !== noteId; });
        });
        canvasContainer.appendChild(note);
        textarea.focus();
        this._notes.push({ element: note, id: noteId });
    }
}
//# sourceMappingURL=toolbar.js.map
