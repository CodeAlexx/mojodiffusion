"use strict";
/**
 * Sidebar: node library browser, search, drag-to-add.
 */
class SFSidebar {
    constructor(canvas, api) {
        this.canvas = canvas;
        this.api = api;
        this.nodeTypes = {}; // class_type -> info
        this.categories = {}; // category -> [class_type]
        this.searchInput = document.getElementById('node-search');
        this.nodeList = document.getElementById('node-list');
        this._setupSearch();
    }
    async loadNodeTypes() {
        try {
            this.nodeTypes = await this.api.getObjectInfo();
            this.canvas.nodeInfo = this.nodeTypes;
            this._buildCategories();
            this._render();
        }
        catch (e) {
            console.error('Failed to load node types:', e);
            this.nodeList.innerHTML = '<div style="padding:12px;color:#ff4a4a;">Failed to load nodes</div>';
        }
    }
    _buildCategories() {
        this.categories = {};
        for (const [name, info] of Object.entries(this.nodeTypes)) {
            const cat = info.category || 'uncategorized';
            if (!this.categories[cat])
                this.categories[cat] = [];
            this.categories[cat].push(name);
        }
        // Sort categories and node names within each
        for (const cat of Object.keys(this.categories)) {
            this.categories[cat].sort();
        }
    }
    _render(filter) {
        this.nodeList.innerHTML = '';
        const sortedCats = Object.keys(this.categories).sort();
        const filterLower = filter ? filter.toLowerCase() : '';
        for (const cat of sortedCats) {
            const nodes = this.categories[cat];
            const filtered = filterLower
                ? nodes.filter(n => {
                    const info = this.nodeTypes[n];
                    const displayName = info.display_name || n;
                    return n.toLowerCase().includes(filterLower) ||
                        displayName.toLowerCase().includes(filterLower) ||
                        cat.toLowerCase().includes(filterLower);
                })
                : nodes;
            if (filtered.length === 0)
                continue;
            // Category header
            const catEl = document.createElement('div');
            catEl.className = 'node-category';
            catEl.innerHTML = '<span class="arrow">\u25BE</span> ' + cat;
            catEl.addEventListener('click', () => {
                catEl.classList.toggle('collapsed');
            });
            this.nodeList.appendChild(catEl);
            // Items container
            const itemsEl = document.createElement('div');
            itemsEl.className = 'node-category-items';
            this.nodeList.appendChild(itemsEl);
            for (const nodeName of filtered) {
                const info = this.nodeTypes[nodeName];
                const item = document.createElement('div');
                item.className = 'node-item';
                item.textContent = info.display_name || nodeName;
                item.title = nodeName;
                item.draggable = true;
                // Drag from sidebar
                item.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', nodeName);
                    e.dataTransfer.effectAllowed = 'copy';
                });
                // Double-click to add at center
                item.addEventListener('dblclick', () => {
                    this._addNodeAtCenter(nodeName);
                });
                itemsEl.appendChild(item);
            }
        }
    }
    _setupSearch() {
        this.searchInput.addEventListener('input', () => {
            this._render(this.searchInput.value);
        });
    }
    _addNodeAtCenter(nodeType) {
        const info = this.nodeTypes[nodeType];
        if (!info)
            return;
        const stage = this.canvas.stage;
        const cx = (stage.width() / 2 - stage.x()) / stage.scaleX();
        const cy = (stage.height() / 2 - stage.y()) / stage.scaleY();
        this.canvas.addNode(nodeType, cx - 90, cy - 50, info);
    }
    addNodeAtPosition(nodeType, screenX, screenY) {
        const info = this.nodeTypes[nodeType];
        if (!info)
            return null;
        const worldPos = this.canvas.getWorldPosition(screenX, screenY);
        return this.canvas.addNode(nodeType, worldPos.x, worldPos.y, info);
    }
}
