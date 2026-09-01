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
    var modelTypeOptions = [];
    var ARCH_COLORS = {
        flux: '#6c6af5',
        sdxl: '#3b82f6',
        sd3: '#8b5cf6',
        sd15: '#6b7280',
        ltxv: '#10b981',
        wan: '#f59e0b',
        bernini: '#e879f9',
        scail2: '#22d3ee',
        klein: '#ec4899',
        lens: '#38bdf8',
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
        if (f.includes('scail'))
            return '~58 GB source + bounded FP8 cache';
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
                '<label class="models-radio"><input type="radio" name="model-type" value="unet"> Diffusion / UNet</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="lora"> LoRA</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="vae"> VAE</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="controlnet"> ControlNet</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="embedding"> Embeddings</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="clip"> CLIP / Text Encoders</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="clip_vision"> CLIP Vision</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="ipadapter"> IP-Adapter</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="feature_adapter"> Feature Adapters</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="runtime_component"> Runtime Components</label>' +
                '<label class="models-radio"><input type="radio" name="model-type" value="upscaler"> Upscalers</label>' +
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
        return Promise.all([
            ModelUtils.loadModelRegistry(),
            fetch('/object_info', { cache: 'no-store' })
                .then(function (resp) { return resp.ok ? resp.json() : {}; })
        ]).then(function (loaded) {
            var registry = loaded[0] || {};
            var info = loaded[1] || {};
            var seen = {};
            modelTypeOptions = Array.isArray(registry.model_type_options)
                ? registry.model_type_options : [];
            (registry.models || []).forEach(function (model) {
                if (!model || !model.name || seen['model:' + model.name])
                    return;
                seen['model:' + model.name] = true;
                allModels.push({
                    name: model.name,
                    displayName: model.card && model.card.title || model.name,
                    path: model.path || model.name,
                    type: model.format === 'full_checkpoint' ? 'checkpoint' : 'unet',
                    arch: model.arch || ModelUtils.archForModel(model.name),
                    registryManaged: true,
                    registryArch: model.arch || 'unknown',
                    detectedArch: model.detected_arch || model.arch || 'unknown',
                    archSource: model.arch_source || 'unknown',
                    archOverride: model.arch_override || '',
                    format: model.format || 'diffusion_model',
                    size: Number(model.size) || 0,
                    runtimeSupported: model.runtime_supported !== false,
                    runtimeReason: model.runtime_reason || '',
                    usesSelectedCheckpoint: model.uses_selected_checkpoint === true,
                    preview: model.preview || ''
                });
            });
            (registry.loras || []).forEach(function (model) {
                if (!model || !model.name || seen['lora:' + model.name])
                    return;
                seen['lora:' + model.name] = true;
                allModels.push({
                    name: model.name,
                    path: model.path || model.name,
                    type: 'lora',
                    arch: model.target_arch || model.arch || ModelUtils.archForModel(model.name),
                    registryManaged: true,
                    registryArch: model.target_arch || model.arch || 'unknown',
                    detectedArch: model.detected_arch || model.target_arch || model.arch || 'unknown',
                    archSource: model.arch_source || 'unknown',
                    archOverride: model.arch_override || '',
                    format: 'lora',
                    size: Number(model.size) || 0,
                    runtimeSupported: true,
                    runtimeReason: '',
                    preview: model.preview || ''
                });
            });
            (registry.artifacts || []).forEach(function (model) {
                if (!model || !model.name || !model.type)
                    return;
                var key = model.type + ':' + model.name;
                if (seen[key])
                    return;
                seen[key] = true;
                allModels.push({
                    name: model.name,
                    path: model.path || model.name,
                    type: model.type,
                    arch: 'any',
                    registryManaged: false,
                    format: model.type,
                    size: Number(model.size) || 0,
                    runtimeSupported: false,
                    runtimeReason: 'Auxiliary model artifact; select it from the matching workflow control.',
                    preview: model.preview || ''
                });
            });
            function addModels(nodeType, inputKey, type, defaultArch) {
                var items = info && info[nodeType] && info[nodeType].input && info[nodeType].input.required && info[nodeType].input.required[inputKey];
                if (items && Array.isArray(items[0])) {
                    items[0].forEach(function (name) {
                        var key = type + ':' + name;
                        if (seen[key])
                            return;
                        seen[key] = true;
                        allModels.push({
                            name: name,
                            path: name,
                            type: type,
                            arch: defaultArch || ModelUtils.archForModel(name),
                            registryManaged: false,
                            format: type,
                            size: 0,
                            runtimeSupported: true,
                            runtimeReason: ''
                        });
                    });
                }
            }
            addModels('VAELoader', 'vae_name', 'vae', 'any');
            addModels('ControlNetLoader', 'control_net_name', 'controlnet');
            addModels('CLIPLoader', 'clip_name', 'clip', 'any');
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
            if (filters.search && m.name.toLowerCase().indexOf(filters.search) === -1 &&
                String(m.displayName || '').toLowerCase().indexOf(filters.search) === -1)
                return false;
            if (filters.type === 'checkpoint') {
                if (m.type !== 'checkpoint' && m.type !== 'unet')
                    return false;
            }
            else if (filters.type !== 'all' && m.type !== filters.type) {
                return false;
            }
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
            var sizeLabel = m.size > 0
                ? (m.size / (1024 * 1024 * 1024)).toFixed(m.size >= 1024 * 1024 * 1024 ? 2 : 3) + ' GB'
                : estimateSize(m.name);
            var useDisabled = m.runtimeSupported === false ? ' disabled' : '';
            var useTitle = m.runtimeSupported === false
                ? ' title="' + escapeHtml(m.runtimeReason || 'No compatible runtime') + '"' : '';
            var isBaseModel = m.type === 'checkpoint' || m.type === 'unet';
            var useButton = isBaseModel
                ? '<button class="model-use-btn" data-name="' + escapeHtml(m.name) + '"' + useDisabled + useTitle + '>Use in Generate</button>'
                : '<div class="model-card-artifact-kind">' + escapeHtml(m.type.replace(/_/g, ' ')) + '</div>';
            card.innerHTML =
                '<div class="model-card-top">' +
                    '<input type="checkbox" class="model-card-check" data-name="' + escapeHtml(m.name) + '"' + checked + '>' +
                    '<div class="model-card-badge" style="background:' + color + '">' + m.arch.toUpperCase() + '</div>' +
                    '</div>' +
                    '<div class="model-card-type">' + m.type + '</div>' +
                    '<div class="model-card-name" data-name="' + escapeHtml(m.name) + '" title="' + escapeHtml(m.displayName || m.name) + '">' + escapeHtml(m.displayName || m.name) + '</div>' +
                    '<div class="model-card-size">' + sizeLabel + '</div>' +
                    useButton;
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
                    var modelName = nameEl.dataset.name || nameEl.getAttribute('title');
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
    function modelTypeEditorHtml(model) {
        if (!model.registryManaged)
            return '';
        var selected = model.archOverride || 'auto';
        var detected = model.detectedArch || 'unknown';
        var options = '<option value="auto"' + (selected === 'auto' ? ' selected' : '') +
            '>Auto-detected (' + escapeHtml(detected.toUpperCase()) + ')</option>';
        modelTypeOptions.forEach(function (option) {
            if (!option || !option.id)
                return;
            var id = String(option.id);
            var label = option.label || id.toUpperCase();
            var support = option.arbitrary_checkpoint_supported
                ? ' — selected file supported' : ' — classification only';
            options += '<option value="' + escapeHtml(id) + '"' +
                (selected === id ? ' selected' : '') + '>' +
                escapeHtml(label + support) + '</option>';
        });
        return '<div class="models-detail-type-editor">' +
            '<label class="models-detail-label" for="models-detail-type-select">Model Type</label>' +
            '<select class="models-detail-type-select" id="models-detail-type-select">' +
            options + '</select>' +
            '<div class="models-detail-type-help">Choose how this ' +
            (model.type === 'lora' ? 'LoRA' : 'checkpoint') +
            ' is classified and routed. ' +
            'An override cannot add a loader for an unsupported architecture.</div>' +
            '<div class="models-detail-type-actions">' +
            '<button class="models-detail-type-save" id="models-detail-type-save">Save Model Type</button>' +
            (model.archOverride
                ? '<button class="models-detail-type-reset" id="models-detail-type-reset">Reset to Auto</button>'
                : '') +
            '</div>' +
            '<div class="models-detail-type-status" id="models-detail-type-status" aria-live="polite"></div>' +
            '</div>';
    }
    function saveModelTypeOverride(model, reset) {
        var select = document.getElementById('models-detail-type-select');
        var status = document.getElementById('models-detail-type-status');
        var saveButton = document.getElementById('models-detail-type-save');
        var resetButton = document.getElementById('models-detail-type-reset');
        var arch = reset ? 'auto' : (select ? select.value : 'auto');
        if (status) {
            status.className = 'models-detail-type-status';
            status.textContent = 'Saving...';
        }
        if (saveButton)
            saveButton.disabled = true;
        if (resetButton)
            resetButton.disabled = true;
        fetch('/v1/models/type', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model.name,
                arch: arch,
                kind: model.type === 'lora' ? 'lora' : 'checkpoint'
            })
        }).then(function (resp) {
            return resp.text().then(function (text) {
                var data = {};
                try {
                    data = text ? JSON.parse(text) : {};
                }
                catch (e) { }
                if (!resp.ok)
                    throw new Error(data.error || ('HTTP ' + resp.status));
                return data;
            });
        }).then(function () {
            ModelUtils.clearCache();
            return load();
        }).then(function () {
            var updated = findModelByName(model.name);
            if (updated)
                showDetailPanel(updated);
        }).catch(function (error) {
            var currentStatus = document.getElementById('models-detail-type-status');
            if (currentStatus) {
                currentStatus.className = 'models-detail-type-status models-detail-type-error';
                currentStatus.textContent = error.message || String(error);
            }
            var currentSave = document.getElementById('models-detail-type-save');
            var currentReset = document.getElementById('models-detail-type-reset');
            if (currentSave)
                currentSave.disabled = false;
            if (currentReset)
                currentReset.disabled = false;
        });
    }
    /** Show the detail panel for a model (slide-in from right) */
    function showDetailPanel(model) {
        activeDetailModel = model;
        var panel = document.getElementById('models-detail-panel');
        if (!panel)
            return;
        var color = ARCH_COLORS[model.arch] || ARCH_COLORS['any'];
        var isBaseModel = model.type === 'checkpoint' || model.type === 'unet';
        panel.style.display = 'flex';
        panel.innerHTML =
            '<button class="models-detail-close" id="models-detail-close">\u2715</button>' +
                '<div class="models-detail-badge" style="background:' + color + '">' + model.arch.toUpperCase() + '</div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Name</span><span class="models-detail-value">' + escapeHtml(model.name) + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Type</span><span class="models-detail-value">' + model.type + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Architecture</span><span class="models-detail-value">' + model.arch.toUpperCase() + '</span></div>' +
                (model.registryManaged ? '<div class="models-detail-row"><span class="models-detail-label">Architecture Source</span><span class="models-detail-value">' + escapeHtml((model.archSource || 'unknown').replace(/_/g, ' ')) + '</span></div>' : '') +
                modelTypeEditorHtml(model) +
                '<div class="models-detail-row"><span class="models-detail-label">Format</span><span class="models-detail-value">' + escapeHtml(model.format || '') + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Loads selected file</span><span class="models-detail-value">' + (model.usesSelectedCheckpoint ? 'Yes' : 'No') + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Est. Size</span><span class="models-detail-value">' + estimateSize(model.name) + '</span></div>' +
                '<div class="models-detail-row"><span class="models-detail-label">Path</span><span class="models-detail-value models-detail-path">' + escapeHtml(model.path || model.name) + '</span></div>' +
                (model.runtimeReason ? '<div class="models-detail-row"><span class="models-detail-label">Runtime</span><span class="models-detail-value">' + escapeHtml(model.runtimeReason) + '</span></div>' : '') +
                '<div class="models-detail-actions">' +
                (isBaseModel ? '<button class="models-detail-use-btn" id="models-detail-use-btn"' + (model.runtimeSupported === false ? ' disabled' : '') + '>Use in Generate</button>' : '') +
                '<button class="models-detail-delete-btn" id="models-detail-delete-btn">Delete Model</button>' +
                '</div>';
        // Trigger reflow for slide-in animation
        panel.offsetHeight;
        panel.classList.add('models-detail-open');
        document.getElementById('models-detail-close').addEventListener('click', closeDetailPanel);
        var detailUse = document.getElementById('models-detail-use-btn');
        if (detailUse) {
            detailUse.addEventListener('click', function () {
                useModelInGenerate(model.name);
            });
        }
        var typeSave = document.getElementById('models-detail-type-save');
        if (typeSave) {
            typeSave.addEventListener('click', function () {
                saveModelTypeOverride(model, false);
            });
        }
        var typeReset = document.getElementById('models-detail-type-reset');
        if (typeReset) {
            typeReset.addEventListener('click', function () {
                saveModelTypeOverride(model, true);
            });
        }
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
        if (typeof switchTab === 'function')
            switchTab('generate');
        if (typeof GenerateTab !== 'undefined' &&
            typeof GenerateTab.selectModel === 'function') {
            GenerateTab.selectModel(modelName);
        }
        else {
            console.error('Generate model selector is unavailable');
        }
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
