"use strict";
/**
 * Models Tab — SerenityFlow Phase 6
 * Model browser with search, filter, and architecture detection.
 */
var ModelsTab = (function () {
    'use strict';
    var initialized = false;
    var allModels = [];
    var filteredModels = [];
    var filters = { search: '', type: 'all', archs: {} };
    var selectedForDelete = {}; // name -> true
    var activeDetailModel = null; // currently shown in detail panel
    var ARCH_COLORS = {
        flux: '#6c6af5',
        sdxl: '#3b82f6',
        sd3: '#8b5cf6',
        sd15: '#6b7280',
        ltxv: '#10b981',
        wan: '#f59e0b',
        bernini: '#e879f9',
        klein: '#ec4899',
        any: '#6b7280'
    };
    function estimateSize(filename) {
        var f = filename.toLowerCase();
        if (f.includes('flux') && f.includes('fp8'))
            return '~11 GB';
        if (f.includes('flux'))
            return '~22 GB';
        if (f.includes('sdxl') || f.includes('xl') || f.includes('pony') || f.includes('illustrious'))
            return '~7 GB';
        if (f.includes('sd3'))
            return '~5 GB';
        if (f.includes('bernini'))
            return '~126 GB source · bounded FP8 cache';
        if (f.includes('ltx'))
            return '~19 GB';
        if (f.includes('wan'))
            return '~14 GB';
        if (f.includes('lora'))
            return '~150 MB';
        if (f.includes('controlnet') || f.includes('control'))
            return '~1.5 GB';
        if (f.includes('vae') || f.includes('ae.'))
            return '~350 MB';
        return '~2 GB';
    }
    function buildUI() {
        var panel = document.getElementById('panel-models');
        if (!panel)
            return;
        panel.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'models-layout';
        layout.innerHTML =
            '<div id="models-missing-banner" class="models-missing-banner" style="display:none"></div>' +
                '<div class="models-header">' +
                '<span class="models-header-title">Models</span>' +
                '<div class="models-header-actions">' +
                '<label class="models-checkbox models-select-all" id="models-select-all-label" style="display:none">' +
                '<input type="checkbox" id="models-select-all"> Select All' +
                '</label>' +
                '<button id="models-bulk-delete-btn" class="models-bulk-delete-btn" style="display:none">Delete Selected (0)</button>' +
                '<input type="text" id="models-search" class="models-search" placeholder="Search models...">' +
                '<button id="models-refresh-btn" class="models-refresh-btn" title="Refresh">&#8635;</button>' +
                '</div>' +
                '</div>' +
                '<div class="models-body">' +
                '<div class="models-sidebar">' +
                '<div class="models-filter-group">' +
                '<div class="models-filter-title">Type</div>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="all" checked> All</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="checkpoint"> Checkpoints</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="unet"> UNets</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="lora"> LoRA</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="vae"> VAE</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="controlnet"> ControlNet</label>' +
                '</div>' +
                '<div class="models-filter-group">' +
                '<div class="models-filter-title">Architecture</div>' +
                '<div id="models-arch-filters"></div>' +
                '</div>' +
                '<div class="models-count" id="models-count">0 models</div>' +
                '</div>' +
                '<div class="models-grid" id="models-grid"></div>' +
                '<div class="models-detail-panel" id="models-detail-panel" style="display:none"></div>' +
                '</div>';
        panel.appendChild(layout);
    }
    function bindEvents() {
        var search = document.getElementById('models-search');
        if (search) {
            search.addEventListener('input', function () {
                filters.search = this.value.toLowerCase();
                applyFilters();
            });
        }
        var refreshBtn = document.getElementById('models-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function () {
                ModelUtils.clearCache();
                load();
            });
        }
        // Type filter radios
        document.querySelectorAll('input[name="model-type"]').forEach(function (radio) {
            radio.addEventListener('change', function () {
                filters.type = this.value;
                applyFilters();
            });
        });
        // Select All checkbox
        var selectAll = document.getElementById('models-select-all');
        if (selectAll) {
            selectAll.addEventListener('change', function () {
                var checked = this.checked;
                filteredModels.forEach(function (m) {
                    if (checked) {
                        selectedForDelete[m.name] = true;
                    }
                    else {
                        delete selectedForDelete[m.name];
                    }
                });
                renderGrid(filteredModels);
                updateBulkDeleteUI();
            });
        }
        // Bulk delete button
        var bulkDeleteBtn = document.getElementById('models-bulk-delete-btn');
        if (bulkDeleteBtn) {
            bulkDeleteBtn.addEventListener('click', function () {
                var names = Object.keys(selectedForDelete);
                if (names.length === 0)
                    return;
                showBulkDeleteConfirm(names);
            });
        }
    }
    function load() {
        allModels = [];
        return fetch('/object_info', { cache: 'no-store' })
            .then(function (resp) { return resp.ok ? resp.json() : {}; })
            .then(function (info) {
            var seen = {};
            function addModels(nodeType, inputKey, type, defaultArch) {
                var items = info && info[nodeType] && info[nodeType].input && info[nodeType].input.required && info[nodeType].input.required[inputKey];
                if (items && Array.isArray(items[0])) {
                    items[0].forEach(function (name) {
                        if (seen[name])
                            return;
                        seen[name] = true;
                        allModels.push({
                            name: name,
                            type: type,
                            arch: defaultArch || ModelUtils.detectArchFromFilename(name)
                        });
                    });
                }
            }
            addModels('CheckpointLoaderSimple', 'ckpt_name', 'checkpoint');
            addModels('UNETLoader', 'unet_name', 'unet');
            addModels('LoraLoader', 'lora_name', 'lora');
            addModels('VAELoader', 'vae_name', 'vae', 'any');
            addModels('ControlNetLoader', 'control_net_name', 'controlnet');
            buildArchFilters();
            applyFilters();
            checkMissingModels();
        })
            .catch(function (err) {
            console.error('ModelsTab load failed:', err);
            renderGrid([]);
        });
    }
    /** Track a model name as "used" in localStorage */
    function trackUsedModel(name) {
        var used = getUsedModels();
        if (used.indexOf(name) === -1) {
            used.push(name);
            try {
                localStorage.setItem('sf-used-models', JSON.stringify(used));
            }
            catch (e) { }
        }
    }
    function getUsedModels() {
        try {
            var raw = localStorage.getItem('sf-used-models');
            return raw ? JSON.parse(raw) : [];
        }
        catch (e) {
            return [];
        }
    }
    /** Check if any previously-used models are no longer present */
    function checkMissingModels() {
        var used = getUsedModels();
        if (used.length === 0)
            return;
        var currentNames = {};
        allModels.forEach(function (m) { currentNames[m.name] = true; });
        var missing = used.filter(function (name) { return !currentNames[name]; });
        var banner = document.getElementById('models-missing-banner');
        if (!banner)
            return;
        if (missing.length === 0) {
            banner.style.display = 'none';
            return;
        }
        banner.style.display = 'flex';
        banner.innerHTML =
            '<span class="models-missing-text">\u26A0 ' + missing.length +
                ' previously used model' + (missing.length !== 1 ? 's' : '') + ' not found</span>' +
                '<button class="models-missing-show-btn" id="models-missing-show-btn">Show</button>' +
                '<button class="models-missing-dismiss-btn" id="models-missing-dismiss-btn">\u2715</button>';
        document.getElementById('models-missing-dismiss-btn').addEventListener('click', function () {
            banner.style.display = 'none';
        });
        document.getElementById('models-missing-show-btn').addEventListener('click', function () {
            var existing = banner.querySelector('.models-missing-list');
            if (existing) {
                existing.remove();
                return;
            }
            var list = document.createElement('div');
            list.className = 'models-missing-list';
            missing.forEach(function (name) {
                var item = document.createElement('div');
                item.className = 'models-missing-item';
                item.textContent = name;
                list.appendChild(item);
            });
            banner.appendChild(list);
        });
    }
    function buildArchFilters() {
        var archSet = {};
        allModels.forEach(function (m) { archSet[m.arch] = true; });
        var container = document.getElementById('models-arch-filters');
        if (!container)
            return;
        container.innerHTML = '';
        Object.keys(archSet).sort().forEach(function (arch) {
            var label = document.createElement('label');
            label.className = 'models-checkbox';
            label.innerHTML = '<input type="checkbox" data-arch="' + arch + '" checked> ' + arch.toUpperCase();
            container.appendChild(label);
            filters.archs[arch] = true;
        });
        container.addEventListener('change', function (e) {
            var cb = e.target;
            if (cb.dataset.arch) {
                filters.archs[cb.dataset.arch] = cb.checked;
                applyFilters();
            }
        });
    }
    function applyFilters() {
        filteredModels = allModels.filter(function (m) {
            if (filters.search && m.name.toLowerCase().indexOf(filters.search) === -1)
                return false;
            if (filters.type !== 'all' && m.type !== filters.type)
                return false;
            if (!filters.archs[m.arch])
                return false;
            return true;
        });
        renderGrid(filteredModels);
        var countEl = document.getElementById('models-count');
        if (countEl)
            countEl.textContent = filteredModels.length + ' model' + (filteredModels.length !== 1 ? 's' : '');
    }
    function renderGrid(models) {
        var grid = document.getElementById('models-grid');
        if (!grid)
            return;
        if (models.length === 0) {
            grid.innerHTML = '<div class="models-empty">No models found</div>';
            updateBulkDeleteUI();
            return;
        }
        grid.innerHTML = '';
        models.forEach(function (m) {
            var card = document.createElement('div');
            card.className = 'model-card' + (selectedForDelete[m.name] ? ' model-card-selected' : '');
            var color = ARCH_COLORS[m.arch] || ARCH_COLORS['any'];
            var checked = selectedForDelete[m.name] ? ' checked' : '';
            card.innerHTML =
                '<div class="model-card-top">' +
                    '<input type="checkbox" class="model-card-check" data-name="' + escapeHtml(m.name) + '"' + checked + '>' +
                    '<div class="model-card-badge" style="background:' + color + '">' + m.arch.toUpperCase() + '</div>' +
                    '</div>' +
                    '<div class="model-card-type">' + m.type + '</div>' +
                    '<div class="model-card-name" title="' + escapeHtml(m.name) + '">' + escapeHtml(m.name) + '</div>' +
                    '<div class="model-card-size">' + estimateSize(m.name) + '</div>' +
                    '<button class="model-use-btn" data-name="' + escapeHtml(m.name) + '">Use in Generate</button>';
            grid.appendChild(card);
        });
        grid.onclick = function (e) {
            // Checkbox toggle for bulk delete
            var check = e.target.closest('.model-card-check');
            if (check) {
                if (check.checked) {
                    selectedForDelete[check.dataset.name] = true;
                }
                else {
                    delete selectedForDelete[check.dataset.name];
                }
                var card = check.closest('.model-card');
                if (card)
                    card.classList.toggle('model-card-selected', check.checked);
                updateBulkDeleteUI();
                return;
            }
            // Use in Generate button
            var btn = e.target.closest('.model-use-btn');
            if (btn) {
                useModelInGenerate(btn.dataset.name);
                return;
            }
            // Card click -> show detail panel (but not if clicking checkbox or button)
            var clickedCard = e.target.closest('.model-card');
            if (clickedCard) {
                var nameEl = clickedCard.querySelector('.model-card-name');
                if (nameEl) {
                    var modelName = nameEl.getAttribute('title');
                    var model = findModelByName(modelName);
                    if (model)
                        showDetailPanel(model);
                }
            }
        };
        updateBulkDeleteUI();
    }
    function findModelByName(name) {
        for (var i = 0; i < allModels.length; i++) {
            if (allModels[i].name === name)
                return allModels[i];
        }
        return null;
    }
    /** Update bulk delete button visibility and count */
    function updateBulkDeleteUI() {
        var count = Object.keys(selectedForDelete).length;
        var bulkBtn = document.getElementById('models-bulk-delete-btn');
        var selectAllLabel = document.getElementById('models-select-all-label');
        if (bulkBtn) {
            bulkBtn.style.display = count > 0 ? '' : 'none';
            bulkBtn.textContent = 'Delete Selected (' + count + ')';
        }
        if (selectAllLabel) {
            selectAllLabel.style.display = filteredModels.length > 0 ? '' : 'none';
        }
    }
    /** Show the detail panel for a model (slide-in from right) */
    function showDetailPanel(model) {
        activeDetailModel = model;
        var panel = document.getElementById('models-detail-panel');
        if (!panel)
            return;
        var color = ARCH_COLORS[model.arch] || ARCH_COLORS['any'];
        panel.style.display = 'flex';
        panel.innerHTML =
            '<button class="models-detail-close" id="models-detail-close">\u2715</button>' +
                '<div class="models-detail-badge" style="background:' + color + '">' + model.arch.toUpperCase() + '</div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Name</span><span class="models-detail-value">' + escapeHtml(model.name) + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Type</span><span class="models-detail-value">' + model.type + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Architecture</span><span class="models-detail-value">' + model.arch.toUpperCase() + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Est. Size</span><span class="models-detail-value">' + estimateSize(model.name) + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Path</span><span class="models-detail-value models-detail-path">' + escapeHtml(model.name) + '</span></div>' +
                '<div class="models-detail-actions">' +
                '<button class="models-detail-use-btn" id="models-detail-use-btn">Use in Generate</button>' +
                '<button class="models-detail-delete-btn" id="models-detail-delete-btn">Delete Model</button>' +
                '</div>';
        // Trigger reflow for slide-in animation
        panel.offsetHeight;
        panel.classList.add('models-detail-open');
        document.getElementById('models-detail-close').addEventListener('click', closeDetailPanel);
        document.getElementById('models-detail-use-btn').addEventListener('click', function () {
            useModelInGenerate(model.name);
        });
        document.getElementById('models-detail-delete-btn').addEventListener('click', function () {
            showDeleteConfirm(model);
        });
    }
    function closeDetailPanel() {
        var panel = document.getElementById('models-detail-panel');
        if (panel) {
            panel.classList.remove('models-detail-open');
            panel.style.display = 'none';
        }
        activeDetailModel = null;
    }
    /** Show single-model delete confirmation */
    function showDeleteConfirm(model) {
        showConfirmDialog('Delete ' + model.name + '? This cannot be undone.', function () { deleteModels([model]); });
    }
    /** Show bulk delete confirmation */
    function showBulkDeleteConfirm(names) {
        var models = names.map(function (n) { return findModelByName(n); }).filter(Boolean);
        if (models.length === 0)
            return;
        showConfirmDialog('Delete ' + models.length + ' selected model' + (models.length !== 1 ? 's' : '') + '? This cannot be undone.', function () { deleteModels(models); });
    }
    /** Generic confirmation dialog */
    function showConfirmDialog(message, onConfirm) {
        // Remove any existing dialog
        var existing = document.getElementById('models-confirm-overlay');
        if (existing)
            existing.remove();
        var overlay = document.createElement('div');
        overlay.id = 'models-confirm-overlay';
        overlay.className = 'models-confirm-overlay';
        overlay.innerHTML =
            '<div class="models-confirm-dialog">' +
                '<div class="models-confirm-msg">' + escapeHtml(message) + '</div>' +
                '<div class="models-confirm-actions">' +
                '<button class="models-confirm-cancel" id="models-confirm-cancel">Cancel</button>' +
                '<button class="models-confirm-ok" id="models-confirm-ok">Delete</button>' +
                '</div>' +
                '</div>';
        document.body.appendChild(overlay);
        document.getElementById('models-confirm-cancel').addEventListener('click', function () { overlay.remove(); });
        document.getElementById('models-confirm-ok').addEventListener('click', function () {
            overlay.remove();
            onConfirm();
        });
    }
    /**
     * Delete models -- calls DELETE endpoint then removes from local list.
     * TODO: Backend DELETE /models/{type}/{name} endpoint may not exist yet.
     *       Wire real endpoint when available.
     */
    function deleteModels(models) {
        models.forEach(function (model) {
            // TODO: Call DELETE /models/{type}/{encodeURIComponent(name)} when backend supports it
            // fetch('/models/' + model.type + '/' + encodeURIComponent(model.name), { method: 'DELETE' });
            // Remove from local list
            allModels = allModels.filter(function (m) { return m.name !== model.name; });
            delete selectedForDelete[model.name];
        });
        // Close detail panel if showing a deleted model
        if (activeDetailModel && !findModelByName(activeDetailModel.name)) {
            closeDetailPanel();
        }
        applyFilters();
        updateBulkDeleteUI();
    }
    function useModelInGenerate(modelName) {
        trackUsedModel(modelName);
        if (typeof GenerateTab !== 'undefined' && GenerateTab.state) {
            GenerateTab.state.model = modelName;
            var picker = document.getElementById('gen-model');
            if (picker)
                picker.value = modelName;
        }
        if (typeof switchTab === 'function')
            switchTab('generate');
    }
    function escapeHtml(str) {
        if (!str)
            return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    // Close detail panel on Escape or click outside
    function bindDetailPanelClose() {
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && activeDetailModel) {
                closeDetailPanel();
            }
        });
        document.addEventListener('click', function (e) {
            if (!activeDetailModel)
                return;
            var panel = document.getElementById('models-detail-panel');
            var grid = document.getElementById('models-grid');
            if (panel && !panel.contains(e.target) && grid && !grid.contains(e.target)) {
                closeDetailPanel();
            }
        });
    }
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        bindEvents();
        bindDetailPanelClose();
        load();
    }
    return { init: init, load: load };
})();
//# sourceMappingURL=models.js.map
