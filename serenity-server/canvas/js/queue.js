"use strict";
/**
 * Queue Tab — SerenityFlow Phase 6
 * Real-time job tracking with pending queue, current job, and history.
 */
var QueueTab = (function () {
    'use strict';
    var initialized = false;
    var MAX_HISTORY = 100;
    var state = {
        current: null,
        pending: [],
        history: [],
        paused: false
    };
    var els = {};
    var elapsedTimer = null;
    function buildUI() {
        var panel = document.getElementById('panel-queue');
        if (!panel)
            return;
        panel.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'queue-layout';
        layout.innerHTML =
            '<div class="queue-header">' +
                '<span class="queue-header-title">Queue</span>' +
                '<div class="queue-header-actions">' +
                '<button id="queue-pause-btn" class="queue-pause-btn">\u23f8 Pause</button>' +
                '<button id="queue-clear-btn" class="queue-clear-btn">Clear Finished</button>' +
                '</div>' +
                '</div>' +
                '<div id="queue-stats" class="queue-stats"></div>' +
                '<div class="queue-sections">' +
                '<div class="queue-section">' +
                '<div class="queue-section-label">Current</div>' +
                '<div id="queue-current-card" class="queue-current-card">' +
                '<div class="queue-empty-state">No job running</div>' +
                '</div>' +
                '</div>' +
                '<div class="queue-section" id="queue-pending-section" style="display:none">' +
                '<div class="queue-section-label">Pending <span id="queue-pending-count"></span></div>' +
                '<div id="queue-pending-list"></div>' +
                '</div>' +
                '<div class="queue-section">' +
                '<div class="queue-section-label">History</div>' +
                '<div id="queue-history-list"></div>' +
                '</div>' +
                '</div>';
        panel.appendChild(layout);
        els.currentCard = document.getElementById('queue-current-card');
        els.pendingSection = document.getElementById('queue-pending-section');
        els.pendingCount = document.getElementById('queue-pending-count');
        els.pendingList = document.getElementById('queue-pending-list');
        els.historyList = document.getElementById('queue-history-list');
        els.clearBtn = document.getElementById('queue-clear-btn');
        els.pauseBtn = document.getElementById('queue-pause-btn');
        els.statsBar = document.getElementById('queue-stats');
    }
    function bindEvents() {
        els.clearBtn.addEventListener('click', function () {
            state.history = [];
            renderHistory();
            renderStats();
            saveHistory();
        });
        els.pauseBtn.addEventListener('click', function () {
            state.paused = !state.paused;
            renderPauseBtn();
        });
    }
    function renderPauseBtn() {
        if (!els.pauseBtn)
            return;
        if (state.paused) {
            els.pauseBtn.textContent = '\u25b6 Resume';
            els.pauseBtn.classList.add('paused');
        }
        else {
            els.pauseBtn.textContent = '\u23f8 Pause';
            els.pauseBtn.classList.remove('paused');
        }
    }
    function connectWS() {
        SerenityWS.on('execution_start', function (data) {
            var id = data && data.prompt_id ? data.prompt_id : null;
            // Find matching pending entry
            var found = null;
            state.pending = state.pending.filter(function (p) {
                if (p.promptId === id) {
                    found = p;
                    return false;
                }
                return true;
            });
            var f = found;
            state.current = {
                promptId: id,
                prompt: f ? f.prompt : '',
                model: f ? f.model : '',
                batchLabel: f ? f.batchLabel : '',
                promptData: f ? f.promptData : null,
                startedAt: Date.now(),
                step: 0, maxStep: 0, node: null, src: null, filename: null
            };
            startElapsedTimer();
            render();
        });
        SerenityWS.on('executing', function (data) {
            if (!state.current || !data)
                return;
            if (data.node)
                state.current.node = data.node;
            renderCurrentJob();
        });
        SerenityWS.on('progress', function (data) {
            if (!state.current || !data)
                return;
            state.current.step = data.value;
            state.current.maxStep = data.max;
            renderCurrentJob();
        });
        SerenityWS.on('executed', function (data) {
            if (!state.current || !data || !data.output)
                return;
            var out = data.output.ui || data.output;
            var items = out.images || out.videos;
            if (items && items.length > 0) {
                var file = items[0];
                state.current.src = SerenityAPI.viewUrl(file.filename, file.subfolder, file.type);
                state.current.filename = file.filename;
            }
        });
        SerenityWS.on('execution_success', function () {
            if (!state.current)
                return;
            stopElapsedTimer();
            state.history.unshift({
                promptId: state.current.promptId,
                prompt: state.current.prompt,
                model: state.current.model,
                batchLabel: state.current.batchLabel,
                promptData: state.current.promptData || null,
                status: 'success',
                completedAt: Date.now(),
                filename: state.current.filename,
                src: state.current.src
            });
            if (state.history.length > MAX_HISTORY)
                state.history.pop();
            state.current = null;
            saveHistory();
            render();
        });
        SerenityWS.on('execution_error', function (data) {
            if (!state.current)
                return;
            stopElapsedTimer();
            state.history.unshift({
                promptId: state.current.promptId,
                prompt: state.current.prompt,
                model: state.current.model,
                batchLabel: state.current.batchLabel,
                promptData: state.current.promptData || null,
                status: 'error',
                error: (data && data.exception_message) || 'Unknown error',
                completedAt: Date.now()
            });
            if (state.history.length > MAX_HISTORY)
                state.history.pop();
            state.current = null;
            saveHistory();
            render();
        });
    }
    function startElapsedTimer() {
        stopElapsedTimer();
        elapsedTimer = setInterval(function () {
            if (state.current)
                renderCurrentJob();
        }, 1000);
    }
    function stopElapsedTimer() {
        if (elapsedTimer) {
            clearInterval(elapsedTimer);
            elapsedTimer = null;
        }
    }
    // -- Rendering --
    function render() {
        renderStats();
        renderPauseBtn();
        renderCurrentJob();
        renderPending();
        renderHistory();
        updateQueueBadge();
    }
    function renderStats() {
        if (!els.statsBar)
            return;
        var pendingCount = state.pending.length;
        var runningCount = state.current ? 1 : 0;
        var completedCount = 0;
        var failedCount = 0;
        state.history.forEach(function (h) {
            if (h.status === 'success')
                completedCount++;
            else if (h.status === 'error')
                failedCount++;
        });
        els.statsBar.innerHTML =
            '<span class="queue-stat pending">\u23f3 ' + pendingCount + ' Pending</span>' +
                '<span class="queue-stat running">\ud83d\udd04 ' + runningCount + ' Running</span>' +
                '<span class="queue-stat completed">\u2713 ' + completedCount + ' Completed</span>' +
                '<span class="queue-stat failed">\u2717 ' + failedCount + ' Failed</span>';
    }
    function renderCurrentJob() {
        if (!els.currentCard)
            return;
        if (!state.current) {
            els.currentCard.innerHTML = '<div class="queue-empty-state">No job running</div>';
            return;
        }
        var c = state.current;
        var pct = c.maxStep > 0 ? Math.round(c.step / c.maxStep * 100) : 0;
        var elapsed = Math.round((Date.now() - c.startedAt) / 1000);
        var label = c.batchLabel ? c.batchLabel + ' · ' : '';
        els.currentCard.innerHTML =
            '<div class="queue-job-card current">' +
                (c.src ? '<img class="queue-thumb" src="' + c.src + '">' : '<div class="queue-thumb-placeholder"></div>') +
                '<div class="queue-job-info">' +
                '<div class="queue-job-model">' + label + escapeHtml(c.model || 'Unknown') + '</div>' +
                '<div class="queue-job-prompt">' + escapeHtml(truncate(c.prompt, 60)) + '</div>' +
                '<div class="queue-progress-wrap">' +
                '<div class="queue-progress-bar"><div class="queue-progress-fill" style="width:' + pct + '%"></div></div>' +
                '</div>' +
                '<div class="queue-progress-text">Step ' + c.step + ' / ' + c.maxStep + ' · ' + pct + '% · ' + elapsed + 's' + (c.node ? ' · ' + c.node : '') + '</div>' +
                '</div>' +
                '<button class="queue-cancel-btn" id="queue-cancel-current">Cancel</button>' +
                '</div>';
        var cancelBtn = document.getElementById('queue-cancel-current');
        if (cancelBtn) {
            cancelBtn.onclick = function () { SerenityAPI.interrupt(); };
        }
    }
    function renderPending() {
        if (!els.pendingSection || !els.pendingList)
            return;
        if (state.pending.length === 0) {
            els.pendingSection.style.display = 'none';
            return;
        }
        els.pendingSection.style.display = 'block';
        els.pendingCount.textContent = '(' + state.pending.length + ')';
        els.pendingList.innerHTML = '';
        state.pending.forEach(function (p, idx) {
            var row = document.createElement('div');
            row.className = 'queue-pending-row';
            row.innerHTML =
                '<span class="queue-pending-num">#' + (idx + 1) + '</span>' +
                    '<span class="queue-pending-prompt">' + escapeHtml(truncate(p.prompt, 50)) + '</span>' +
                    (p.batchLabel ? '<span class="queue-pending-batch">' + p.batchLabel + '</span>' : '') +
                    '<button class="queue-pending-remove" data-id="' + p.promptId + '">Remove</button>';
            els.pendingList.appendChild(row);
        });
        els.pendingList.onclick = function (e) {
            var btn = e.target.closest('.queue-pending-remove');
            if (!btn)
                return;
            var id = btn.dataset.id;
            state.pending = state.pending.filter(function (p) { return p.promptId !== id; });
            renderPending();
            renderStats();
            updateQueueBadge();
        };
    }
    function renderHistory() {
        if (!els.historyList)
            return;
        if (state.history.length === 0) {
            els.historyList.innerHTML = '<div class="queue-empty-state">No history</div>';
            return;
        }
        els.historyList.innerHTML = '';
        state.history.forEach(function (entry, idx) {
            var row = document.createElement('div');
            row.className = 'queue-history-entry ' + entry.status;
            var isSuccess = entry.status === 'success';
            var isFailed = entry.status === 'error';
            var timeAgo = formatTimeAgo(entry.completedAt);
            row.innerHTML =
                '<span class="queue-status-icon">' + (isSuccess ? '\u2713' : '\u2717') + '</span>' +
                    (isSuccess && entry.src ? '<img class="queue-thumb-sm" src="' + entry.src + '">' : '') +
                    '<div class="queue-history-info">' +
                    '<div class="queue-history-filename">' + escapeHtml(entry.filename || entry.model || 'Unknown') + '</div>' +
                    (entry.batchLabel ? '<span class="queue-history-batch">' + entry.batchLabel + '</span>' : '') +
                    (!isSuccess && entry.error ? '<div class="queue-history-error">' + escapeHtml(entry.error) + '</div>' : '') +
                    '</div>' +
                    '<span class="queue-history-time">' + timeAgo + '</span>' +
                    '<div class="queue-history-actions">' +
                    (isFailed && entry.promptData ? '<button class="queue-action-btn queue-retry-btn" data-idx="' + idx + '">Retry</button>' : '') +
                    (isSuccess && entry.src ? '<button class="queue-action-btn queue-view-btn" data-src="' + entry.src + '" data-video="' + (/\.(webp|mp4|gif)$/i.test(entry.filename || '') ? '1' : '0') + '">View</button>' : '') +
                    '<button class="queue-action-btn queue-dismiss-btn" data-id="' + entry.promptId + '">\u00d7</button>' +
                    '</div>';
            // Expandable details panel (hidden by default)
            var details = document.createElement('div');
            details.className = 'queue-history-details';
            details.style.display = 'none';
            var detailParts = [];
            if (entry.prompt)
                detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Prompt</span><span class="queue-detail-value">' + escapeHtml(entry.prompt) + '</span></div>');
            if (entry.model)
                detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Model</span><span class="queue-detail-value">' + escapeHtml(entry.model) + '</span></div>');
            if (entry.promptData) {
                var pd = entry.promptData;
                if (pd.width && pd.height)
                    detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Dimensions</span><span class="queue-detail-value">' + pd.width + ' \u00d7 ' + pd.height + '</span></div>');
                if (pd.seed != null)
                    detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Seed</span><span class="queue-detail-value">' + pd.seed + '</span></div>');
                if (pd.scheduler)
                    detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Scheduler</span><span class="queue-detail-value">' + escapeHtml(String(pd.scheduler)) + '</span></div>');
                if (pd.steps)
                    detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">Steps</span><span class="queue-detail-value">' + pd.steps + '</span></div>');
                if (pd.cfg)
                    detailParts.push('<div class="queue-detail-row"><span class="queue-detail-label">CFG</span><span class="queue-detail-value">' + pd.cfg + '</span></div>');
            }
            if (detailParts.length > 0) {
                details.innerHTML = detailParts.join('');
            }
            var wrapper = document.createElement('div');
            wrapper.className = 'queue-history-item';
            wrapper.appendChild(row);
            if (detailParts.length > 0)
                wrapper.appendChild(details);
            els.historyList.appendChild(wrapper);
        });
        els.historyList.onclick = function (e) {
            var retryBtn = e.target.closest('.queue-retry-btn');
            if (retryBtn) {
                var idx = parseInt(retryBtn.dataset.idx, 10);
                var entry = state.history[idx];
                if (entry && entry.promptData) {
                    retryJob(entry);
                }
                return;
            }
            var viewBtn = e.target.closest('.queue-view-btn');
            if (viewBtn) {
                localStorage.setItem('sf-view-image', JSON.stringify({
                    src: viewBtn.dataset.src,
                    isVideo: viewBtn.dataset.video === '1'
                }));
                if (typeof switchTab === 'function')
                    switchTab('generate');
                return;
            }
            var dismissBtn = e.target.closest('.queue-dismiss-btn');
            if (dismissBtn) {
                var id = dismissBtn.dataset.id;
                state.history = state.history.filter(function (h) { return h.promptId !== id; });
                renderHistory();
                renderStats();
                saveHistory();
                return;
            }
            // Toggle expanded details on row click
            var item = e.target.closest('.queue-history-item');
            if (item) {
                var detailsEl = item.querySelector('.queue-history-details');
                if (detailsEl) {
                    var isOpen = detailsEl.style.display !== 'none';
                    detailsEl.style.display = isOpen ? 'none' : 'block';
                    item.classList.toggle('expanded', !isOpen);
                }
            }
        };
    }
    function retryJob(entry) {
        if (!entry.promptData)
            return;
        var pd = entry.promptData;
        fetch('/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: pd.workflow,
                client_id: typeof SerenityWS !== 'undefined' ? SerenityWS.getClientId() : ''
            })
        })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (data) {
            registerPending({
                promptId: data.prompt_id,
                prompt: entry.prompt || '',
                model: entry.model || '',
                queuedAt: Date.now(),
                batchLabel: entry.batchLabel || '',
                promptData: pd
            });
        })
            .catch(function (err) {
            console.error('[QueueTab] Retry failed:', err);
        });
    }
    // -- Queue Badge on Icon Rail --
    function updateQueueBadge() {
        var badge = document.getElementById('queue-rail-badge');
        if (!badge)
            return;
        var running = state.current !== null;
        var pending = state.pending.length;
        if (running || pending > 0) {
            badge.style.display = 'flex';
            badge.textContent = '' + (running ? (pending + 1) : pending);
        }
        else {
            badge.style.display = 'none';
        }
    }
    // -- Helpers --
    function truncate(str, max) {
        if (!str)
            return '';
        return str.length > max ? str.substring(0, max) + '...' : str;
    }
    function escapeHtml(str) {
        if (!str)
            return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    function formatTimeAgo(timestamp) {
        if (!timestamp)
            return '';
        var diff = Math.floor((Date.now() - timestamp) / 1000);
        if (diff < 60)
            return 'just now';
        if (diff < 3600)
            return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400)
            return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }
    function saveHistory() {
        try {
            localStorage.setItem('sf-queue-history', JSON.stringify(state.history.slice(0, MAX_HISTORY)));
        }
        catch (e) { }
    }
    function loadHistory() {
        try {
            var raw = localStorage.getItem('sf-queue-history');
            var saved = raw ? JSON.parse(raw) : null;
            if (saved && Array.isArray(saved))
                state.history = saved.slice(0, MAX_HISTORY);
        }
        catch (e) {
            state.history = [];
        }
    }
    // -- Public --
    function registerPending(entry) {
        // If paused, hold items in a paused queue -- they won't be sent to backend
        if (state.paused) {
            state.pending.push(entry);
            render();
            return;
        }
        state.pending.push(entry);
        render();
    }
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        bindEvents();
        loadHistory();
        connectWS();
        renderPauseBtn();
        render();
    }
    return {
        init: init,
        registerPending: registerPending,
        state: state
    };
})();
//# sourceMappingURL=queue.js.map