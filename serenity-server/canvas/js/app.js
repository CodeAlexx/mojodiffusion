"use strict";
/**
 * Entry point. Bootstraps canvas, sidebar, toolbar, etc.
 */
// Global references (classes SFCanvas, SFApi, etc. are defined in their respective .ts files)
let sfCanvas, sfApi, sfSidebar, sfToolbar, sfProperties, sfPreview, sfHistory, sfSelection;
// Track right-click position for node placement
let lastRightClickPos = { x: 0, y: 0 };
document.addEventListener('DOMContentLoaded', async () => {
    // Create API client
    sfApi = new SFApi();
    // Create canvas
    sfCanvas = new SFCanvas('canvas-container');
    // Create subsystems
    sfSidebar = new SFSidebar(sfCanvas, sfApi);
    sfProperties = new SFProperties(sfCanvas);
    sfPreview = new SFPreview(sfApi);
    sfHistory = new SFHistory(sfCanvas);
    sfSelection = new SFSelection(sfCanvas);
    sfToolbar = new SFToolbar(sfCanvas, sfApi);
    // Setup context menu (node/connection right-click + floating search for empty canvas)
    setupContextMenu(sfCanvas, sfSidebar);
    // Setup drag-drop from file drops onto canvas
    setupCanvasDragDrop(sfCanvas);
    // Setup double-click on node to open properties
    sfCanvas.nodeLayer.on('dblclick', (e) => {
        // Find which node was clicked
        let target = e.target;
        while (target && target.getParent() !== sfCanvas.nodeLayer) {
            target = target.getParent();
        }
        if (target) {
            const nodeId = findNodeIdByGroup(sfCanvas, target);
            if (nodeId)
                sfProperties.show(nodeId);
        }
    });
    // Double-click on empty canvas opens node search
    sfCanvas.stage.on('dblclick', (e) => {
        // Only trigger on stage background, not on nodes
        if (e.target !== sfCanvas.stage)
            return;
        // Check it's not the node layer handling it
        const pointer = sfCanvas.stage.getPointerPosition();
        if (!pointer)
            return;
        lastRightClickPos = pointer;
        const stageBox = sfCanvas.stage.container().getBoundingClientRect();
        showNodeSearch(stageBox.left + pointer.x, stageBox.top + pointer.y);
    });
    // Hide canvas hint when first node is added
    const hint = document.getElementById('canvas-hint');
    if (hint) {
        const origAddNode = sfCanvas.addNode.bind(sfCanvas);
        sfCanvas.addNode = function (nodeType, x, y, info) {
            hint.style.display = 'none';
            return origAddNode(nodeType, x, y, info);
        };
    }
    // Auto-save workflow to sessionStorage
    const STORAGE_KEY = 'sf_autosave';
    setInterval(() => {
        if (sfCanvas.nodes.size > 0) {
            try {
                const { prompt, nodePositions } = serializeWorkflow(sfCanvas);
                const workflow = { version: 2, prompt, nodes: nodePositions };
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(workflow));
            }
            catch (_) { }
        }
    }, 5000);
    // Connect to server
    sfApi.connect();
    // Load node types, then restore autosaved workflow
    await sfSidebar.loadNodeTypes();
    // Restore from session storage if available
    try {
        const saved = sessionStorage.getItem(STORAGE_KEY);
        if (saved) {
            const data = JSON.parse(saved);
            loadWorkflow(sfCanvas, data, sfCanvas.nodeInfo);
            if (hint)
                hint.style.display = 'none';
        }
    }
    catch (_) { }
});
function findNodeIdByGroup(canvas, group) {
    for (const [id, node] of canvas.nodes) {
        if (node.group === group)
            return id;
    }
    return null;
}
// --- Floating Node Search ---
function showNodeSearch(x, y) {
    const panel = document.getElementById('node-search-panel');
    if (!panel)
        return;
    // Position — keep on screen
    const pw = 260, ph = 400;
    const left = (x + pw > window.innerWidth) ? x - pw : x;
    const top = (y + ph > window.innerHeight) ? y - ph : y;
    panel.style.left = Math.max(0, left) + 'px';
    panel.style.top = Math.max(0, top) + 'px';
    panel.classList.add('visible');
    // Focus search input
    const input = document.getElementById('node-search-input');
    if (input) {
        input.value = '';
        input.focus();
        renderNodeResults('');
    }
}
function hideNodeSearch() {
    const panel = document.getElementById('node-search-panel');
    if (panel)
        panel.classList.remove('visible');
}
function renderNodeResults(query) {
    const list = document.getElementById('node-search-results');
    if (!list)
        return;
    const nodeInfo = sfCanvas ? sfCanvas.nodeInfo : null;
    if (!nodeInfo) {
        list.innerHTML = '<div class="node-search-empty">Loading nodes...</div>';
        return;
    }
    const q = query.toLowerCase().trim();
    const entries = Object.entries(nodeInfo);
    const filtered = q
        ? entries.filter(function (entry) {
            var type = entry[0];
            var info = entry[1];
            return type.toLowerCase().includes(q) ||
                (info.display_name || '').toLowerCase().includes(q) ||
                (info.category || '').toLowerCase().includes(q);
        })
        : entries;
    // Show max 30 results
    const results = filtered.slice(0, 30);
    if (results.length === 0) {
        list.innerHTML = '<div class="node-search-empty">No nodes found</div>';
        return;
    }
    list.innerHTML = '';
    results.forEach(function (entry) {
        var type = entry[0];
        var info = entry[1];
        var row = document.createElement('div');
        row.className = 'node-search-result';
        row.dataset.type = type;
        var nameSpan = document.createElement('span');
        nameSpan.textContent = info.display_name || type;
        row.appendChild(nameSpan);
        var catSpan = document.createElement('span');
        catSpan.className = 'node-search-category';
        catSpan.textContent = info.category || '';
        row.appendChild(catSpan);
        row.addEventListener('click', function () {
            addNodeFromSearch(type);
            hideNodeSearch();
        });
        list.appendChild(row);
    });
}
function addNodeFromSearch(nodeType) {
    if (!sfCanvas || !sfCanvas.nodeInfo)
        return;
    const info = sfCanvas.nodeInfo[nodeType];
    if (!info)
        return;
    // Convert stage pointer position to world coordinates
    const worldPos = sfCanvas.getWorldPosition(lastRightClickPos.x, lastRightClickPos.y);
    sfCanvas.addNode(nodeType, worldPos.x, worldPos.y, info);
}
// Search input handler
document.addEventListener('DOMContentLoaded', function () {
    var input = document.getElementById('node-search-input');
    if (input) {
        input.addEventListener('input', function () {
            renderNodeResults(input.value);
        });
        // Enter key adds first result
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                var first = document.querySelector('.node-search-result');
                if (first) {
                    addNodeFromSearch(first.dataset.type);
                    hideNodeSearch();
                }
            }
        });
    }
    // Hide on click outside
    document.addEventListener('click', function (e) {
        if (!e.target.closest('#node-search-panel')) {
            hideNodeSearch();
        }
    });
    // Hide on Escape
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            hideNodeSearch();
        }
    });
});
// --- Context Menu ---
function setupContextMenu(canvas, sidebar) {
    const menu = document.getElementById('context-menu');
    const menuItems = document.getElementById('context-menu-items');
    if (!menu || !menuItems)
        return;
    // Hide on click anywhere
    document.addEventListener('click', () => {
        menu.classList.add('hidden');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            menu.classList.add('hidden');
        }
    });
    // Right-click on canvas stage
    canvas.stage.on('contextmenu', (e) => {
        e.evt.preventDefault();
        const target = e.target;
        const pointer = canvas.stage.getPointerPosition();
        if (!pointer)
            return;
        // Store position for node placement
        lastRightClickPos = pointer;
        menuItems.innerHTML = '';
        // Check what was right-clicked
        let clickedNode = null;
        let clickedConnection = null;
        // Walk up to find node group
        let t = target;
        while (t && t.getParent() !== canvas.nodeLayer) {
            t = t.getParent();
        }
        if (t) {
            clickedNode = findNodeIdByGroup(canvas, t);
        }
        // Check connections
        if (!clickedNode) {
            for (const conn of canvas.connections) {
                if (target === conn.line) {
                    clickedConnection = conn;
                    break;
                }
            }
        }
        // Helper: position and show menu, keeping it on-screen
        const showMenu = () => {
            menu.style.left = e.evt.clientX + 'px';
            menu.style.top = e.evt.clientY + 'px';
            menu.classList.remove('hidden');
            const rect = menu.getBoundingClientRect();
            if (rect.right > window.innerWidth) {
                menu.style.left = (window.innerWidth - rect.width - 4) + 'px';
            }
            if (rect.bottom > window.innerHeight) {
                menu.style.top = (window.innerHeight - rect.height - 4) + 'px';
            }
        };
        if (clickedNode) {
            // Node context menu
            addMenuItem(menuItems, 'Properties', () => sfProperties.show(clickedNode));
            addMenuItem(menuItems, 'Duplicate', () => {
                const node = canvas.nodes.get(clickedNode);
                if (node) {
                    const newNode = canvas.addNode(node.nodeType, node.x + 30, node.y + 30, node.info);
                    for (const [name, val] of Object.entries(node.widgetValues)) {
                        newNode.setWidgetValue(name, val);
                        if (newNode.widgets[name])
                            newNode.widgets[name].setValue(val);
                    }
                }
            });
            addSeparator(menuItems);
            const hasConnections = canvas.getConnectionsForNode(clickedNode).length > 0;
            addMenuItem(menuItems, 'Disconnect All', () => {
                sfHistory.saveState();
                canvas.disconnectAllForNode(clickedNode);
            }, !hasConnections);
            addMenuItem(menuItems, 'Delete', () => {
                sfHistory.saveState();
                canvas.removeNode(clickedNode);
            });
            showMenu();
        }
        else if (clickedConnection) {
            // Connection context menu
            addMenuItem(menuItems, 'Delete Connection', () => {
                sfHistory.saveState();
                canvas.removeConnection(clickedConnection);
            });
            showMenu();
        }
        else {
            // Empty canvas context menu
            addMenuItem(menuItems, 'Add Node', () => {
                showNodeSearch(e.evt.clientX, e.evt.clientY);
            });
            addMenuItem(menuItems, 'Reset View', () => {
                canvas.fitView(true);
            });
            const hasPaste = sfSelection && sfSelection.clipboard;
            addMenuItem(menuItems, 'Paste', () => {
                if (sfSelection)
                    sfSelection._paste();
            }, !hasPaste);
            showMenu();
        }
    });
}
function addMenuItem(container, label, onClick, disabled) {
    const item = document.createElement('div');
    item.className = 'context-item';
    item.textContent = label;
    if (disabled) {
        item.style.color = 'var(--text-muted)';
        item.style.cursor = 'default';
    }
    else if (onClick) {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            document.getElementById('context-menu').classList.add('hidden');
            onClick();
        });
    }
    container.appendChild(item);
}
function addSeparator(container) {
    const sep = document.createElement('div');
    sep.className = 'context-separator';
    container.appendChild(sep);
}
// --- Drag and Drop (file drop only, no sidebar drag) ---
function setupCanvasDragDrop(canvas) {
    const container = document.getElementById('canvas-container');
    if (!container)
        return;
    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });
    container.addEventListener('drop', (e) => {
        e.preventDefault();
        // Check for .json file drop (workflow import)
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.name.endsWith('.json')) {
                file.text().then(text => {
                    try {
                        const data = JSON.parse(text);
                        loadWorkflow(canvas, data, canvas.nodeInfo);
                        sfToolbar._toast('Workflow loaded from ' + file.name);
                    }
                    catch (err) {
                        sfToolbar._toast('Failed to load workflow: ' + err.message, 'error');
                    }
                });
            }
        }
    });
}
