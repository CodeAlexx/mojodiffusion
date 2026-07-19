"use strict";
/**
 * Shell — Global navigation shell for SerenityFlow.
 * Handles icon rail, tab switching, mode toggle (Simple/Advanced),
 * Lucide icons, Konva resize, templates dropdown, and workflow
 * execution state via SerenityWS.
 */
// Apply persisted settings immediately (Settings module loaded before shell.js)
if (typeof Settings !== 'undefined') {
    Settings.applyTheme(Settings.get('theme'));
    Settings.applyAccentColor(Settings.get('accentColor'));
}
var currentMode = 'advanced'; // single-mode UI (Simple mode removed)
function setMode(mode) {
    mode = 'advanced'; // single-mode UI (Simple mode removed)
    currentMode = mode;
    localStorage.setItem('sf-mode', mode);
    var isAdvanced = true;
    // Icon rail is always visible (provides navigation in both modes)
    document.getElementById('icon-rail').style.display = 'flex';
    var _smc = document.getElementById('simple-mode-container'); if (_smc) _smc.style.display = 'none';
    document.getElementById('content-area').style.display = isAdvanced ? 'flex' : 'none';
    // Update toggle button states
    var _msb = document.getElementById('mode-simple-btn'); if (_msb) _msb.classList.toggle('active', false);
    var _mab = document.getElementById('mode-advanced-btn'); if (_mab) _mab.classList.toggle('active', true);
    // Init Simple mode on first switch
    if (!isAdvanced && typeof SimpleMode !== 'undefined') {
        SimpleMode.init();
    }
    // If switching to advanced, restore the saved tab
    if (isAdvanced) {
        var savedTab = localStorage.getItem('sf-active-tab') || 'generate';
        switchTab(savedTab);
    }
}
function switchTab(tabId) {
    var previousTab = localStorage.getItem('sf-active-tab') || 'generate';
    if (previousTab === 'workflows' && tabId !== 'workflows' &&
        typeof WorkflowSync !== 'undefined') {
        WorkflowSync.syncGenerateFromCanvas();
    }
    if (tabId === 'workflows' && previousTab !== 'workflows' &&
        typeof GenerateTab !== 'undefined' && typeof WorkflowSync !== 'undefined') {
        GenerateTab.init();
        if (GenerateTab.getParams)
            WorkflowSync.syncWorkflowFromGenerate(GenerateTab.getParams());
    }
    // Update icon rail active state
    document.querySelectorAll('.nav-btn').forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    // Show/hide panels
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
        panel.style.display = panel.id === 'panel-' + tabId ? 'flex' : 'none';
    });
    // Init Generate tab on first switch
    if (tabId === 'generate' && typeof GenerateTab !== 'undefined') {
        GenerateTab.init();
        // Check for pending image from Queue tab
        var pendingView = localStorage.getItem('sf-view-image');
        if (pendingView) {
            localStorage.removeItem('sf-view-image');
            try {
                var viewData = JSON.parse(pendingView);
                if (viewData.src && GenerateTab.displayResult) {
                    GenerateTab.displayResult(viewData.src, viewData.isVideo);
                }
            }
            catch (e) { }
        }
    }
    // Init Canvas tab on first switch
    if (tabId === 'canvas' && typeof CanvasTab !== 'undefined') {
        CanvasTab.init();
        requestAnimationFrame(function () { CanvasTab.resize(); });
    }
    // Init Queue tab on first switch
    if (tabId === 'queue' && typeof QueueTab !== 'undefined') {
        QueueTab.init();
    }
    // Init Models tab on first switch
    if (tabId === 'models' && typeof ModelsTab !== 'undefined') {
        ModelsTab.init();
    }
    // Init Settings tab on first switch
    if (tabId === 'settings' && typeof SettingsTab !== 'undefined') {
        SettingsTab.init();
    }
    // Init Video Edit tab on first switch (deferred — Konva needs visible container)
    if (tabId === 'video-edit' && typeof VideoEditTab !== 'undefined') {
        if (!VideoEditTab._initialized) {
            VideoEditTab.init();
        } else {
            requestAnimationFrame(function () { VideoEditTab.resize(); });
        }
    }
    // Resize Konva stage when workflows tab becomes visible
    if (tabId === 'workflows') {
        requestAnimationFrame(resizeWorkflowStage);
    }
    // Persist
    localStorage.setItem('sf-active-tab', tabId);
}
/**
 * Resize the Konva stage to fill available space within the workflows panel.
 * Accounts for sidebar, properties panel, and toolbar heights.
 */
function resizeWorkflowStage() {
    if (typeof sfCanvas === 'undefined' || !sfCanvas || !sfCanvas.stage)
        return;
    var container = document.getElementById('canvas-container');
    if (!container)
        return;
    // canvas-container has flex:1, fills full width (no sidebar).
    sfCanvas.stage.width(container.offsetWidth);
    sfCanvas.stage.height(container.offsetHeight);
    sfCanvas.stage.batchDraw();
}
window.addEventListener('resize', function () {
    var activeTab = localStorage.getItem('sf-active-tab') || 'generate';
    if (activeTab === 'workflows') {
        resizeWorkflowStage();
    }
    if (activeTab === 'canvas' && typeof CanvasTab !== 'undefined') {
        CanvasTab.resize();
    }
    if (activeTab === 'video-edit' && typeof VideoEditTab !== 'undefined' && VideoEditTab._initialized) {
        VideoEditTab.resize();
    }
});
/**
 * Update topbar connection indicator from SerenityWS.
 */
function setupTopbarWS() {
    if (typeof SerenityWS === 'undefined')
        return;
    var dot = document.querySelector('.queue-dot');
    var label = document.querySelector('.queue-label');
    if (!dot || !label)
        return;
    SerenityWS.on('connected', function () {
        dot.className = 'queue-dot idle';
        label.textContent = 'Idle';
    });
    SerenityWS.on('disconnected', function () {
        dot.className = 'queue-dot';
        dot.style.background = 'var(--shell-error)';
        label.textContent = 'Disconnected';
    });
    SerenityWS.on('status', function (data) {
        if (!data || !data.status)
            return;
        var qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
        if (qr > 0) {
            dot.className = 'queue-dot running';
            label.textContent = 'Running (' + qr + ')';
        }
        else {
            dot.className = 'queue-dot idle';
            label.textContent = 'Idle';
        }
    });
}
/**
 * Setup the workflow execution state — enable/disable Stop button,
 * toggle Queue button running state.
 */
function setupWorkflowExecution() {
    if (typeof SerenityWS === 'undefined')
        return;
    var queueBtn = document.getElementById('btn-queue');
    var stopBtn = document.getElementById('btn-interrupt');
    if (!queueBtn || !stopBtn)
        return;
    function setWorkflowRunning(running) {
        var queueLabel = queueBtn.querySelector('span');
        if (queueLabel) {
            queueLabel.textContent = running ? 'Running...' : 'Generate';
        }
        queueBtn.classList.toggle('running', running);
        stopBtn.disabled = !running;
    }
    SerenityWS.on('execution_start', function () {
        setWorkflowRunning(true);
    });
    SerenityWS.on('execution_success', function () {
        setWorkflowRunning(false);
        // Clear node highlights after 3s
        setTimeout(function () {
            if (typeof sfCanvas !== 'undefined' && sfCanvas) {
                sfCanvas.nodes.forEach(function (n) {
                    n.setExecutionState(null);
                });
            }
        }, 3000);
    });
    SerenityWS.on('execution_error', function () {
        setWorkflowRunning(false);
    });
    // Status-based fallback (when execution finishes without explicit success event)
    SerenityWS.on('status', function (data) {
        if (!data || !data.status)
            return;
        var qr = data.status.exec_info ? data.status.exec_info.queue_remaining : 0;
        if (qr === 0) {
            setWorkflowRunning(false);
        }
    });
}
/**
 * Setup Templates dropdown.
 */
function setupTemplatesDropdown() {
    var btn = document.getElementById('btn-templates');
    var dropdown = document.getElementById('templates-dropdown');
    var list = document.getElementById('templates-list');
    if (!btn || !dropdown || !list)
        return;
    // Built-in production presets use the same WorkflowBuilder as Generate, so
    // model contracts cannot drift between the two screens. User templates from
    // /templates still take precedence when present.
    const fallbackTemplates = [
        { name: 'FLUX.1 Dev · Text to Image', preset: { model: 'flux1-dev', prompt: 'a lighthouse on a rocky coast at sunset, dramatic sky', steps: 20, cfg: 1, guidance: 4, scheduler: 'euler' } },
        { name: 'SDXL · Text to Image', preset: { model: 'sdxl_unet_bf16', prompt: 'a red vintage bicycle against a blue garden wall, morning sunlight', steps: 20, cfg: 7, scheduler: 'euler' } },
        { name: 'Klein 9B Base · Text to Image', preset: { model: 'flux-2-klein-base-9b', prompt: 'a green ceramic teapot beside yellow lemons on linen', steps: 50, cfg: 4, scheduler: 'euler' } },
        { name: 'Krea 2 Turbo · Text to Image', preset: { model: 'krea2-turbo', prompt: 'a white owl on a mossy branch in a moonlit forest', steps: 8, cfg: 0, scheduler: 'euler' } },
        { name: 'Chroma HD · Text to Image', preset: { model: 'chroma1_hd_bf16', prompt: 'a lighthouse on a rocky coast at sunset, dramatic sky', steps: 30, cfg: 4, scheduler: 'euler' } },
        { name: 'Qwen Image · Text to Image', preset: { model: 'qwen_image_fp8_e4m3fn', prompt: 'a wooden rowboat on a misty lake at dawn', steps: 20, cfg: 4, scheduler: 'euler' } },
        { name: 'Z-Image · Text to Image', preset: { model: 'zimage_base', prompt: 'a glass terrarium filled with tiny ferns on a wooden table', width: 512, height: 512, steps: 16, cfg: 5, scheduler: 'euler' } },
        { name: 'Anima · Text to Image', preset: { model: 'anima', prompt: 'a red fox in a field of lavender, detailed illustration', steps: 20, cfg: 4.5, scheduler: 'euler' } },
        { name: 'Ideogram 4 · Text to Image', preset: { model: 'ideogram-4-fp8', prompt: 'a bakery storefront sign reading SERENITY, warm evening light', steps: 20, cfg: 7, scheduler: 'euler' } },
        { name: 'SenseNova · Text to Image', preset: { model: 'sensenova_u1', prompt: 'a silver robot watering orange flowers in a greenhouse', steps: 30, cfg: 4, scheduler: 'euler' } },
        { name: 'Wan 2.2 5B · Text to Video', preset: { model: 'Wan2.2-TI2V-5B-Mojo', prompt: 'two anthropomorphic cats in comfortable boxing gear and bright gloves fight intensely on a spotlighted stage, cinematic lighting, dynamic camera movement', width: 832, height: 480, steps: 50, cfg: 5, scheduler: 'uni_pc', frames: 121, fps: 24 } },
        { name: 'Bernini-R · Text to Video', preset: { model: 'Bernini-R-Diffusers', prompt: 'a woman in a flowing red coat walks through a rain-soaked neon city at night, cinematic tracking shot, natural motion, detailed reflections, atmospheric depth', width: 848, height: 480, steps: 40, cfg: 4, scheduler: 'uni_pc', frames: 81, fps: 16 } },
        { name: 'LTX 2.3 Dev · Text to Video', preset: { model: 'ltx-2.3-22b-dev-fp8', prompt: 'a red balloon drifting over a wildflower meadow, gentle camera movement', width: 1920, height: 1088, steps: 15, cfg: 3, scheduler: 'euler', frames: 121, fps: 24 } },
    ];
    function admittedFallbackTemplates() {
        // A preset is one-click production UI, so it must never advertise a
        // model absent from the server's gated model inventory. Bernini enters
        // this list automatically only after its product evidence gate passes.
        var available = new Set();
        if (typeof GenerateTab !== 'undefined' && GenerateTab.state && Array.isArray(GenerateTab.state.allModels)) {
            GenerateTab.state.allModels.forEach(function (model) { available.add(model.name); });
        }
        return fallbackTemplates.filter(function (template) {
            return !template.preset || available.has(template.preset.model);
        });
    }
    // The linked SerenityFlow pack predates the current product contracts
    // (unsupported edits/LoRAs/14B video names and image-queue video routing).
    // Preserve those files on disk, but do not advertise them as one-click
    // production templates. Unknown names remain visible as user templates.
    const legacyTemplateNames = new Set([
        'flux1_dev_edit', 'flux1_dev_edit_lora', 'flux1_dev_t2i', 'flux1_dev_t2i_lora',
        'flux2_dev_edit', 'flux2_dev_edit_lora', 'flux2_dev_t2i', 'flux2_dev_t2i_lora',
        'klein4b_edit', 'klein4b_edit_lora', 'klein4b_t2i', 'klein4b_t2i_lora',
        'klein9b_edit', 'klein9b_edit_lora', 'klein9b_t2i', 'klein9b_t2i_lora',
        'ltx23_a2v', 'ltx23_i2v', 'ltx23_ia2v', 'ltx23_serenityfp8_a2v',
        'ltx23_serenityfp8_i2v', 'ltx23_serenityfp8_ia2v', 'ltx23_serenityfp8_t2v',
        'ltx23_t2v', 'qwen_edit', 'qwen_edit_lora', 'qwen_image_t2i',
        'qwen_image_t2i_lora', 'sd35_large_t2i', 'sdxl_t2i', 'wan22_i2v',
        'wan22_i2v_lora', 'wan22_t2v', 'wan22_t2v_lora', 'wan23_i2v',
        'wan23_i2v_lora', 'wan23_t2v', 'wan23_t2v_lora', 'zimage_t2i',
        'zimage_t2i_lora'
    ]);
    function renderTemplates(templates) {
        list.innerHTML = '';
        if (!templates || templates.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'wf-template-empty';
            empty.textContent = 'No templates available';
            list.appendChild(empty);
            return;
        }
        templates.forEach(function (t) {
            var item = document.createElement('div');
            item.className = 'wf-template-item';
            item.textContent = t.name || (t.file ? t.file.replace(/\.json$/i, '').replace(/_/g, ' ') : 'Workflow');
            item.addEventListener('click', function (e) {
                e.stopPropagation();
                dropdown.classList.add('hidden');
                loadTemplate(t);
            });
            list.appendChild(item);
        });
    }
    function loadTemplate(template) {
        if (!template) {
            console.error('No template provided');
            return;
        }
        if (template.preset) {
            var p = Object.assign({
                negPrompt: '', width: 1024, height: 1024, seed: -1,
                guidance: 4, frames: 121, fps: 24, loras: []
            }, template.preset);
            var graph = WorkflowBuilder.build(p);
            var presetNameInput = document.getElementById('workflow-name');
            if (presetNameInput)
                presetNameInput.value = template.name || 'Untitled Workflow';
            if (typeof loadWorkflow !== 'undefined' && typeof sfCanvas !== 'undefined') {
                loadWorkflow(sfCanvas, graph, sfCanvas.nodeInfo);
                if (typeof GenerateTab !== 'undefined' && GenerateTab.applyParams)
                    GenerateTab.applyParams(p, { source: 'template' });
                if (typeof WorkflowSync !== 'undefined')
                    WorkflowSync.markSynced(typeof GenerateTab !== 'undefined' && GenerateTab.getParams
                        ? GenerateTab.getParams() : p);
                if (typeof sfToolbar !== 'undefined' && sfToolbar)
                    sfToolbar._toast('Workflow loaded', 'success');
            }
            return;
        }
        var baseUrl = template.url;
        if (!baseUrl && template.file) {
            baseUrl = 'workflows/' + template.file;
        }
        if (!baseUrl) {
            console.error('Template URL is missing', template);
            return;
        }
        var url = baseUrl + '?t=' + Date.now();
        var nameInput = document.getElementById('workflow-name');
        if (nameInput) {
            var displayName = template.name || (template.file ? template.file.replace(/\.json$/i, '').replace(/_/g, ' ') : '');
            if (displayName) {
                displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);
                nameInput.value = displayName;
            }
        }
        if (typeof sfToolbar !== 'undefined' && sfToolbar) {
            sfToolbar.loadWorkflowFromUrl(url);
        }
        else {
            fetch(url, { cache: 'no-store' })
                .then(function (r) {
                if (!r.ok)
                    throw new Error('HTTP ' + r.status);
                return r.json();
            })
                .then(function (data) {
                if (typeof loadWorkflow !== 'undefined' && typeof sfCanvas !== 'undefined') {
                    loadWorkflow(sfCanvas, data, sfCanvas.nodeInfo);
                }
            })
                .catch(function (err) {
                console.error('Failed to load template:', err);
            });
        }
    }
    // Toggle dropdown
    btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var isOpen = !dropdown.classList.contains('hidden');
        if (isOpen) {
            dropdown.classList.add('hidden');
            return;
        }
        // Try to fetch template list from server, fall back to hardcoded
        fetch('/templates')
            .then(function (r) {
            if (!r.ok)
                throw new Error('no endpoint');
            return r.json();
        })
            .then(function (templates) {
            var custom = (templates || []).filter(function (t) {
                return !legacyTemplateNames.has(t.name || '');
            });
            var merged = new Map();
            admittedFallbackTemplates().forEach(function (t) { merged.set(t.name, t); });
            custom.forEach(function (t) { merged.set(t.name || t.file || t.url, t); });
            renderTemplates(Array.from(merged.values()));
        })
            .catch(function () {
            renderTemplates(admittedFallbackTemplates());
        });
        dropdown.classList.remove('hidden');
    });
    // Close dropdown on outside click
    document.addEventListener('click', function () {
        dropdown.classList.add('hidden');
    });
}
document.addEventListener('DOMContentLoaded', function () {
    // Init Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    // Attach click handlers to rail buttons
    document.querySelectorAll('.nav-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            // If in simple mode, switch to advanced first
            if (currentMode !== 'advanced') {
                setMode('advanced');
            }
            switchTab(btn.dataset.tab);
        });
    });
    // Setup shared WS integrations
    setupTopbarWS();
    setupWorkflowExecution();
    setupTemplatesDropdown();
    // Mode toggle buttons
    var simpleModeBtn = document.getElementById('mode-simple-btn');
    var advancedModeBtn = document.getElementById('mode-advanced-btn');
    if (simpleModeBtn) {
        simpleModeBtn.addEventListener('click', function () { setMode('simple'); });
    }
    if (advancedModeBtn) {
        advancedModeBtn.addEventListener('click', function () { setMode('advanced'); });
    }
    // Restore mode (default: advanced).
    // setMode('advanced') internally calls switchTab to restore the saved tab.
    // Always use Advanced mode — Simple mode removed.
    setMode('advanced');
    // Restore saved extra model directories, then warm object_info cache
    var savedDirs = [];
    try {
        savedDirs = JSON.parse(localStorage.getItem('sf-extra-model-dirs') || '[]');
    }
    catch (e) { }
    var dirPromises = savedDirs.map(function (dir) {
        return fetch('/folder_paths/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dir })
        }).catch(function () { });
    });
    Promise.all(dirPromises).then(function () {
        if (typeof ModelUtils !== 'undefined' && ModelUtils.loadObjectInfo) {
            ModelUtils.loadObjectInfo().catch(function () { });
        }
    });
    // ── Keyboard Shortcuts ──
    var tabKeys = { '1': 'generate', '2': 'queue', '3': 'canvas', '4': 'video-edit', '5': 'models', '6': 'workflows', '7': 'settings' };
    document.addEventListener('keydown', function (e) {
        var target = e.target;
        var tag = target.tagName;
        var isTyping = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable);
        var activeTab = localStorage.getItem('sf-active-tab') || 'generate';
        // Ctrl+Enter belongs to the visible screen. The Workflow toolbar owns
        // the same shortcut on its tab, preventing two different submissions.
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && activeTab === 'generate') {
            e.preventDefault();
            if (typeof GenerateTab !== 'undefined' && GenerateTab.generate) {
                GenerateTab.generate();
            }
            return;
        }
        // --- Global: Escape → Cancel generation ---
        if (e.key === 'Escape' && !e.ctrlKey && !e.altKey && !e.shiftKey) {
            if (typeof SerenityAPI !== 'undefined' && SerenityAPI.interrupt) {
                SerenityAPI.interrupt();
            }
            return;
        }
        // --- Global: Ctrl+A on generate tab → Select all gallery images ---
        if ((e.ctrlKey || e.metaKey) && e.key === 'a' && activeTab === 'generate' && !isTyping) {
            if (typeof GenerateTab !== 'undefined' && GenerateTab.state && GenerateTab.state.gallery) {
                e.preventDefault();
                var st = GenerateTab.state;
                st.selectedImages = [];
                for (var i = 0; i < st.gallery.length; i++) {
                    st.selectedImages.push(i);
                }
                // Update selection UI via DOM
                var thumbs = document.querySelectorAll('#gen-right-panel .gen-thumb-wrap');
                thumbs.forEach(function (wrap) {
                    var idx = parseInt(wrap.dataset.galleryIndex);
                    wrap.classList.toggle('gen-selected', st.selectedImages.indexOf(idx) >= 0);
                });
                var badge = document.getElementById('gen-selection-badge');
                if (badge) {
                    if (st.selectedImages.length > 1) {
                        badge.textContent = st.selectedImages.length + ' selected';
                        badge.classList.add('visible');
                    }
                    else {
                        badge.classList.remove('visible');
                    }
                }
                var bulkBar = document.getElementById('gen-bulk-bar');
                if (bulkBar) {
                    bulkBar.classList.toggle('visible', st.selectedImages.length > 1);
                }
            }
            return;
        }
        // Everything below requires NOT typing in an input
        if (isTyping)
            return;
        // --- Global: 1-6 → Switch tabs (only in advanced mode) ---
        if (!e.ctrlKey && !e.altKey && !e.metaKey && tabKeys[e.key]) {
            if (currentMode === 'advanced') {
                e.preventDefault();
                switchTab(tabKeys[e.key]);
            }
            return;
        }
        // --- Generate tab only shortcuts ---
        if (activeTab !== 'generate' || typeof GenerateTab === 'undefined')
            return;
        var leftPanel = document.getElementById('gen-left-panel');
        var rightPanel = document.getElementById('gen-right-panel');
        var st = GenerateTab.state;
        // T or O → Toggle left panel
        if (e.key === 't' || e.key === 'T' || e.key === 'o' || e.key === 'O') {
            if (leftPanel && st) {
                st.leftPanelVisible = !st.leftPanelVisible;
                leftPanel.classList.toggle('gen-panel-hidden', !st.leftPanelVisible);
                var floatBtn = document.getElementById('gen-float-toggle-left');
                if (floatBtn)
                    floatBtn.classList.toggle('active', st.leftPanelVisible);
            }
            return;
        }
        // G → Toggle right panel (gallery)
        if (e.key === 'g' || e.key === 'G') {
            if (rightPanel && st) {
                st.rightPanelVisible = !st.rightPanelVisible;
                rightPanel.classList.toggle('gen-panel-hidden', !st.rightPanelVisible);
                var galleryToggle = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle)
                    galleryToggle.classList.toggle('active', st.rightPanelVisible);
            }
            return;
        }
        // F → Toggle both panels (full screen preview)
        if (e.key === 'f' || e.key === 'F') {
            if (leftPanel && rightPanel && st) {
                // If either panel is visible, hide both; if both hidden, show both
                var hideAll = st.leftPanelVisible || st.rightPanelVisible;
                st.leftPanelVisible = !hideAll;
                st.rightPanelVisible = !hideAll;
                leftPanel.classList.toggle('gen-panel-hidden', hideAll);
                rightPanel.classList.toggle('gen-panel-hidden', hideAll);
                var floatBtn2 = document.getElementById('gen-float-toggle-left');
                if (floatBtn2)
                    floatBtn2.classList.toggle('active', !hideAll);
                var galleryToggle2 = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle2)
                    galleryToggle2.classList.toggle('active', !hideAll);
            }
            return;
        }
        // Delete / Backspace → Delete selected gallery images
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (st && st.selectedImages && st.selectedImages.length > 0) {
                e.preventDefault();
                var bulkDelete = document.getElementById('gen-bulk-delete');
                if (bulkDelete) {
                    bulkDelete.click();
                }
            }
            return;
        }
        // . (period) → Toggle star on selected images
        if (e.key === '.') {
            if (st && st.selectedImages && st.selectedImages.length > 0 && st.gallery) {
                st.selectedImages.forEach(function (idx) {
                    if (st.gallery[idx]) {
                        st.gallery[idx].starred = !st.gallery[idx].starred;
                    }
                });
                // Update star icons in DOM
                var thumbs = document.querySelectorAll('#gen-right-panel .gen-thumb-wrap');
                thumbs.forEach(function (wrap) {
                    var idx = parseInt(wrap.dataset.galleryIndex);
                    if (st.gallery[idx]) {
                        var starEl = wrap.querySelector('.gen-thumb-star');
                        if (starEl) {
                            var isStarred = st.gallery[idx].starred;
                            starEl.textContent = isStarred ? '\u2605' : '\u2606';
                            starEl.classList.toggle('starred', isStarred);
                        }
                    }
                });
                // Persist
                try {
                    localStorage.setItem('sf-gallery', JSON.stringify(st.gallery));
                }
                catch (ex) { }
            }
            return;
        }
    });
    // ── Server Console ──
    var sfConsole = document.getElementById('sf-console');
    var sfConsoleLog = document.getElementById('sf-console-log');
    var sfConsoleVisible = false;
    var MAX_CONSOLE_LINES = 500;

    function toggleConsole() {
        sfConsoleVisible = !sfConsoleVisible;
        if (sfConsole) sfConsole.style.display = sfConsoleVisible ? 'flex' : 'none';
    }

    function logToConsole(type, data) {
        if (!sfConsoleLog) return;
        var line = document.createElement('div');
        line.className = 'log-line';
        var now = new Date();
        var ts = String(now.getHours()).padStart(2, '0') + ':' +
                 String(now.getMinutes()).padStart(2, '0') + ':' +
                 String(now.getSeconds()).padStart(2, '0');

        var colorClass = 'log-type-' + (type || 'default');
        var detail = '';
        if (data) {
            if (type === 'executing' && data.node) {
                detail = 'node ' + data.node + (data.display_node ? ' (' + data.display_node + ')' : '');
            } else if (type === 'progress' && data.value !== undefined) {
                detail = data.value + '/' + data.max;
            } else if (type === 'execution_error' && data.exception_message) {
                detail = data.exception_message;
            } else if (type === 'status' && data.status && data.status.exec_info) {
                detail = 'queue: ' + data.status.exec_info.queue_remaining;
            } else if (typeof data === 'object') {
                try { detail = JSON.stringify(data).slice(0, 200); } catch(e) {}
            } else {
                detail = String(data).slice(0, 200);
            }
        }

        line.innerHTML = '<span class="log-ts">' + ts + '</span>' +
                         '<span class="log-type ' + colorClass + '">' + type + '</span>' +
                         '<span>' + detail + '</span>';
        sfConsoleLog.appendChild(line);

        // Trim old lines
        while (sfConsoleLog.children.length > MAX_CONSOLE_LINES) {
            sfConsoleLog.removeChild(sfConsoleLog.firstChild);
        }
        sfConsoleLog.scrollTop = sfConsoleLog.scrollHeight;
    }

    // Hook into all WS events
    if (typeof SerenityWS !== 'undefined') {
        var _origEmit = SerenityWS._emit;
        // Tap into the ws module's emit — check if it exposes onAny
        // Fallback: listen to specific known events
        var logEvents = [
            'connected', 'disconnected', 'status',
            'execution_start', 'executing', 'progress', 'executed',
            'execution_success', 'execution_error', 'execution_cached',
            'export_progress', 'export_complete', 'export_error',
        ];
        logEvents.forEach(function (evType) {
            SerenityWS.on(evType, function (data) {
                logToConsole(evType, data);
            });
        });
    }

    // Console toggle: backtick key
    document.addEventListener('keydown', function (e) {
        if (e.key === '`' && !e.ctrlKey && !e.altKey && !e.metaKey) {
            var tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            e.preventDefault();
            toggleConsole();
        }
    });

    // Console buttons
    var clearBtn = document.getElementById('sf-console-clear');
    if (clearBtn) clearBtn.addEventListener('click', function () {
        if (sfConsoleLog) sfConsoleLog.innerHTML = '';
    });
    var closeBtn = document.getElementById('sf-console-close');
    if (closeBtn) closeBtn.addEventListener('click', toggleConsole);

    logToConsole('connected', 'Console ready. Press ` (backtick) to toggle.');
});
//# sourceMappingURL=shell.js.map
