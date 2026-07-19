/**
 * app.js -- Main application controller for SerenityBoard.
 *
 * Manages state, fetches data from REST API, coordinates charts and
 * live WebSocket updates.
 */

/* global echarts, getOrCreateChart, disposeChart, getChartInstance, resizeAllCharts,
          LiveConnection, createChart, updateChart, appendPoints,
          destroyChart, reSmoothChart, getChartIds, getChartInfo,
          removeRunFromChart, getRunColor, setCustomRunColor, RUN_COLORS,
          setGlobalScalarStepMax, resetGlobalSyncState,
          exportChartCSV,
          createHistogramChart, updateHistogramChart, destroyHistogramChart,
          updateDistributionChart,
          getHistogramChartIds, setHistogramMode, getHistogramMode,
          createHParamsChart, destroyHParamsChart,
          createTraceTimeline, createEvalGrid, createArtifactGallery,
          createPRCurveChart,
          createAudioGallery,
          hashString */

(function() {
    'use strict';

    // ─── Application state ────────────────────────────────────────────

    var state = {
        runs: [],                   // [{name, status, hparams, start_time}]
        selectedRuns: new Set(),     // Set of run names
        selectedTags: new Set(),     // Set of tag strings (scalars)
        selectedHistogramTags: new Set(), // Set of tensor tag strings (histograms)
        allTags: new Set(),          // Union of scalar tags from all selected runs
        allTensorTags: new Set(),    // Union of tensor tags from all selected runs
        tagsByRun: {},               // runName -> {scalars: [...], tensors: [...], ...}
        smoothingWeight: 0.95,
        xAxisMode: 'step',
        runStartTimes: {},           // runName -> wall_time of first point
        runSessions: {},             // runName -> latest session_id from live stream
        scalarFetchSeq: {},          // "run\x00tag" -> monotonically increasing request id
        scalarFetchInflight: {},     // "run\x00tag" -> boolean (one request at a time)
        scalarFetchPending: {},      // "run\x00tag" -> boolean (run one more after inflight)
        selectedPRCurveTags: new Set(), // Set of PR curve tag strings
        allPRCurveTags: new Set(),     // Union of PR curve tags from all selected runs
        selectedAudioTags: new Set(), // Set of audio tag strings
        allAudioTags: new Set(),      // Union of audio tags from all selected runs
        activeTab: 'scalars',        // 'scalars' | 'histograms' | 'hparams' | 'traces' | 'eval' | 'artifacts' | 'audio' | 'pr-curves' | 'projector'
        hparamsData: null,           // cached hparams comparison data
        collapsedGroups: {},         // groupName -> boolean (true = collapsed)
        tagMetrics: {},              // tag -> {count, last_step, type}
        customScalarsLayout: null,   // cached layout config
        customScalarsMode: false,    // true = custom layout view, false = standard
    };

    var liveConn = new LiveConnection();

    // ─── DOM references ───────────────────────────────────────────────

    var runListEl = document.getElementById('run-list');
    var tagListEl = document.getElementById('tag-list');
    var tagFilterEl = document.getElementById('tag-filter');
    var chartGridEl = document.getElementById('chart-grid');
    var smoothingSliderEl = document.getElementById('smoothing-slider');
    var smoothingValueEl = document.getElementById('smoothing-value');
    var xAxisModeEl = document.getElementById('xaxis-mode');
    var resetAxesEl = document.getElementById('reset-axes');
    var connectionStatusEl = document.getElementById('connection-status');
    var dataStatusEl = document.getElementById('data-status');
    var sidebarToggleEl = document.getElementById('sidebar-toggle');
    var sidebarEl = document.getElementById('sidebar');
    var tabBarEl = document.getElementById('tab-bar');
    var panelScalarsEl = document.getElementById('panel-scalars');
    var panelHistogramsEl = document.getElementById('panel-histograms');
    var panelHparamsEl = document.getElementById('panel-hparams');
    var histogramGridEl = document.getElementById('histogram-grid');
    var hparamsContainerEl = document.getElementById('hparams-container');
    var panelTracesEl = document.getElementById('panel-traces');
    var panelEvalEl = document.getElementById('panel-eval');
    var panelArtifactsEl = document.getElementById('panel-artifacts');
    var panelPRCurvesEl = document.getElementById('panel-pr-curves');
    var panelAudioEl = document.getElementById('panel-audio');
    var tracesContainerEl = document.getElementById('traces-container');
    var evalContainerEl = document.getElementById('eval-container');
    var artifactsContainerEl = document.getElementById('artifacts-container');
    var prCurvesContainerEl = document.getElementById('pr-curves-container');
    var audioContainerEl = document.getElementById('audio-container');
    var panelProjectorEl = document.getElementById('panel-projector');
    var panelLoraEl = document.getElementById('panel-lora');
    var exportScalarsEl = document.getElementById('export-scalars');
    var runNotesSectionEl = document.getElementById('run-notes-section');
    var runNotesTextareaEl = document.getElementById('run-notes-textarea');
    var runNotesSavedEl = document.getElementById('run-notes-saved');
    var runNotesBodyEl = document.getElementById('run-notes-body');

    // ─── Initialization ───────────────────────────────────────────────

    function init() {
        restoreFromUrlHash();
        fetchRuns();
        setupEventListeners();
        setupLiveConnection();

        // Refresh runs list periodically
        setInterval(fetchRuns, 10000);
        // Fallback scalar resync to recover from missed live updates.
        setInterval(pollScalarFallback, 5000);
    }

    // ─── API calls ────────────────────────────────────────────────────

    function fetchJsonNoCache(url) {
        return fetch(url, { cache: 'no-store' }).then(function(r) { return r.json(); });
    }

    function fetchMetrics(runName) {
        var url = '/api/runs/' + encodeURIComponent(runName) + '/metrics';
        return fetchJsonNoCache(url);
    }

    function fetchRuns() {
        fetchJsonNoCache('/api/runs')
            .then(function(runs) {
                state.runs = runs;
                renderRunList();
                // Keep selected run tag catalogs fresh so new tabs (hist/artifacts/eval)
                // populate during an active run without requiring page reload.
                refreshSelectedRunTags();
                // Apply any pending state from URL hash (runs/tags/tab)
                if (state._pendingRuns || state._pendingTags || state._pendingTab) {
                    applyPendingHashState();
                }
            })
            .catch(function(err) {
                console.error('Failed to fetch runs:', err);
            });
    }

    function fetchTags(runName) {
        return fetchJsonNoCache('/api/runs/' + encodeURIComponent(runName) + '/tags')
            .then(function(tags) {
                state.tagsByRun[runName] = tags;
                return tags;
            });
    }

    function fetchScalars(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/scalars?tag=' + encodeURIComponent(tag) + '&downsample=0';
        return fetchJsonNoCache(url);
    }

    function fetchScalarsLast(runName, tags) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/scalars/last?tags=' + tags.map(encodeURIComponent).join(',');
        return fetchJsonNoCache(url);
    }

    function fetchHistograms(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/histograms?tag=' + encodeURIComponent(tag) + '&downsample=100';
        return fetchJsonNoCache(url);
    }

    function fetchDistributions(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/distributions?tag=' + encodeURIComponent(tag) + '&downsample=100';
        return fetchJsonNoCache(url);
    }

    function fetchHParams(runs) {
        var url = '/api/compare/hparams?runs=' + runs.map(encodeURIComponent).join(',');
        return fetchJsonNoCache(url);
    }

    function fetchTraces(runName, stepFrom, stepTo) {
        var url = '/api/runs/' + encodeURIComponent(runName) + '/traces';
        var params = [];
        if (stepFrom != null) params.push('step_from=' + stepFrom);
        if (stepTo != null) params.push('step_to=' + stepTo);
        if (params.length) url += '?' + params.join('&');
        return fetchJsonNoCache(url);
    }

    function fetchEval(runName, suite) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/eval?suite=' + encodeURIComponent(suite);
        return fetchJsonNoCache(url);
    }

    function fetchArtifacts(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/artifacts?tag=' + encodeURIComponent(tag);
        return fetchJsonNoCache(url);
    }

    function fetchPRCurves(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) + '/pr-curves?tag=' + encodeURIComponent(tag) + '&downsample=50';
        return fetchJsonNoCache(url);
    }

    function fetchAudio(runName, tag) {
        var url = '/api/runs/' + encodeURIComponent(runName) + '/audio?tag=' + encodeURIComponent(tag) + '&downsample=50';
        return fetchJsonNoCache(url).then(function(items) {
            return (items || []).map(function(item) {
                if (!item.tag) item.tag = tag;
                return item;
            });
        });
    }

    function fetchCustomScalarsLayout(runName) {
        var url = '/api/runs/' + encodeURIComponent(runName) + '/custom-scalars/layout';
        return fetchJsonNoCache(url);
    }

    function fetchCustomScalarsData(runName, tagRegexes, downsample) {
        var url = '/api/runs/' + encodeURIComponent(runName) +
                  '/custom-scalars/data?tags=' + tagRegexes.map(encodeURIComponent).join(',') +
                  '&downsample=' + (downsample || 5000);
        return fetchJsonNoCache(url);
    }

    function refreshSelectedRunTags() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) return Promise.resolve();

        return Promise.all(runs.map(function(runName) {
            return fetchTags(runName);
        })).then(function() {
            rebuildTagList();
        }).catch(function(err) {
            console.error('Failed to refresh selected run tags:', err);
        });
    }

    // ─── Run list rendering ───────────────────────────────────────────

    function renderRunList() {
        if (!state.runs.length) {
            runListEl.innerHTML = '<div class="placeholder">No runs found</div>';
            return;
        }

        var html = '';
        state.runs.forEach(function(run) {
            var checked = state.selectedRuns.has(run.name) ? ' checked' : '';
            var color = getRunColor(run.name);
            var statusClass = 'run-status-' + (run.status || 'unknown');
            var tooltip = '';
            if (run.last_activity) {
                var ago = _formatTimeAgo(run.last_activity);
                tooltip = ' title="last activity: ' + ago + '"';
            }
            // Show leaf name in the label, full path on hover
            var displayName = run.name.indexOf('__') !== -1
                ? run.name.split('__').pop()
                : run.name;

            html += '<label class="checkbox-item">' +
                '<input type="checkbox" value="' + escapeHtml(run.name) + '"' + checked + '>' +
                '<span class="color-indicator color-indicator-clickable" ' +
                    'style="background-color: ' + color + '" ' +
                    'data-run="' + escapeHtml(run.name) + '"></span>' +
                '<span class="checkbox-label" title="' + escapeHtml(run.name) + '">' + escapeHtml(displayName) + '</span>' +
                '<span class="' + statusClass + ' run-status-badge"' + tooltip + '>' +
                    escapeHtml(run.status || '') + '</span>' +
                '<span class="run-delete-btn" data-run="' + escapeHtml(run.name) + '" title="Delete run">&times;</span>' +
                '</label>';
        });

        runListEl.innerHTML = html;

        // Attach change handlers
        var checkboxes = runListEl.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(function(cb) {
            cb.addEventListener('change', function() {
                onRunToggle(cb.value, cb.checked);
            });
        });

        // Attach color indicator click handlers
        var colorDots = runListEl.querySelectorAll('.color-indicator-clickable');
        colorDots.forEach(function(dot) {
            dot.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                showColorPicker(dot, dot.getAttribute('data-run'));
            });
        });

        // Attach delete button handlers
        var deleteBtns = runListEl.querySelectorAll('.run-delete-btn');
        deleteBtns.forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var runName = btn.getAttribute('data-run');
                if (confirm('Delete run "' + runName + '"? This removes all data from disk.')) {
                    deleteRun(runName);
                }
            });
        });
    }

    function showColorPicker(anchorEl, runName) {
        // Remove any existing color picker
        var existing = document.querySelector('.color-picker-popup');
        if (existing) existing.remove();

        var popup = document.createElement('div');
        popup.className = 'color-picker-popup';

        RUN_COLORS.forEach(function(color) {
            var swatch = document.createElement('span');
            swatch.className = 'color-swatch';
            swatch.style.backgroundColor = color;
            if (getRunColor(runName) === color) {
                swatch.classList.add('color-swatch-active');
            }
            swatch.addEventListener('click', function(e) {
                e.stopPropagation();
                setCustomRunColor(runName, color);
                popup.remove();
                // Re-render run list to update indicators
                renderRunList();
                // Re-render all scalar charts
                reSmoothAllCharts();
            });
            popup.appendChild(swatch);
        });

        // Position popup near the anchor
        var rect = anchorEl.getBoundingClientRect();
        popup.style.top = (rect.bottom + 4) + 'px';
        popup.style.left = rect.left + 'px';
        document.body.appendChild(popup);

        // Close on outside click
        function closePopup(e) {
            if (!popup.contains(e.target) && e.target !== anchorEl) {
                popup.remove();
                document.removeEventListener('click', closePopup);
            }
        }
        // Defer to avoid immediate close from the click that opened it
        setTimeout(function() {
            document.addEventListener('click', closePopup);
        }, 0);
    }

    // ─── Tag list rendering ───────────────────────────────────────────

    function rebuildTagList() {
        var allTags = new Set();
        var allTensorTags = new Set();
        var allPRCurveTags = new Set();
        var allAudioTags = new Set();
        state.selectedRuns.forEach(function(runName) {
            var tags = state.tagsByRun[runName];
            if (tags && tags.scalars) {
                tags.scalars.forEach(function(t) { allTags.add(t); });
            }
            if (tags && tags.tensors) {
                tags.tensors.forEach(function(t) { allTensorTags.add(t); });
            }
            if (tags && tags.pr_curves) {
                tags.pr_curves.forEach(function(t) { allPRCurveTags.add(t); });
            }
            if (tags && tags.audio) {
                tags.audio.forEach(function(t) { allAudioTags.add(t); });
            }
        });
        state.allTags = allTags;
        state.allTensorTags = allTensorTags;
        state.allPRCurveTags = allPRCurveTags;
        state.allAudioTags = allAudioTags;
        renderTagList();

        // Also fetch metrics for type badges and count display
        state.selectedRuns.forEach(function(runName) {
            fetchMetrics(runName).then(function(metrics) {
                if (!metrics) return;
                ['scalars', 'tensors', 'artifacts', 'text_events'].forEach(function(type) {
                    if (metrics[type]) {
                        metrics[type].forEach(function(m) {
                            var existing = state.tagMetrics[m.tag];
                            if (!existing || m.count > existing.count) {
                                state.tagMetrics[m.tag] = {
                                    count: m.count,
                                    last_step: m.last_step,
                                    type: type
                                };
                            }
                        });
                    }
                });
                renderTagList();
            }).catch(function(err) {
                console.warn('Failed to fetch metrics for run ' + runName + ':', err);
            });
        });
    }

    function renderTagList() {
        var rawFilter = tagFilterEl.value;
        var filter = rawFilter.toLowerCase();
        var isRegexMode = false;
        var filterRegex = null;
        var regexError = false;

        // Detect regex mode: input starts with /
        if (rawFilter.indexOf('/') === 0 && rawFilter.length > 1) {
            isRegexMode = true;
            // Extract pattern: /pattern/ or /pattern (closing slash optional)
            var regexStr = rawFilter.substring(1);
            if (regexStr.charAt(regexStr.length - 1) === '/' && regexStr.length > 1) {
                regexStr = regexStr.substring(0, regexStr.length - 1);
            }
            try {
                filterRegex = new RegExp(regexStr, 'i');
            } catch (e) {
                regexError = true;
                filterRegex = null;
            }
        }

        // Update visual indicators on the filter input
        tagFilterEl.classList.toggle('tag-filter-regex', isRegexMode && !regexError);
        tagFilterEl.classList.toggle('tag-filter-regex-error', regexError);

        // Update regex badge
        var existingBadge = tagFilterEl.parentNode.querySelector('.tag-filter-regex-badge');
        if (isRegexMode && !regexError) {
            if (!existingBadge) {
                var badge = document.createElement('span');
                badge.className = 'tag-filter-regex-badge';
                badge.textContent = '.*';
                tagFilterEl.parentNode.appendChild(badge);
            }
        } else {
            if (existingBadge) existingBadge.remove();
        }

        // Choose which tags and selected set to show based on active tab
        var activeTags, selectedSet, emptyMsg, toggleFn;
        if (state.activeTab === 'histograms') {
            activeTags = Array.from(state.allTensorTags).sort();
            selectedSet = state.selectedHistogramTags;
            emptyMsg = 'No tensor tags available';
            toggleFn = onHistogramTagToggle;
        } else if (state.activeTab === 'pr-curves') {
            activeTags = Array.from(state.allPRCurveTags).sort();
            selectedSet = state.selectedPRCurveTags;
            emptyMsg = 'No PR curve tags available';
            toggleFn = onPRCurveTagToggle;
        } else if (state.activeTab === 'audio') {
            activeTags = Array.from(state.allAudioTags || []).sort();
            selectedSet = state.selectedAudioTags;
            emptyMsg = 'No audio tags available';
            toggleFn = onAudioTagToggle;
        } else {
            activeTags = Array.from(state.allTags).sort();
            selectedSet = state.selectedTags;
            emptyMsg = 'No scalar tags available';
            toggleFn = onTagToggle;
        }

        // HParams tab: hide tag list
        if (state.activeTab === 'hparams') {
            tagListEl.innerHTML = '<div class="placeholder">Tags not used for HParams view</div>';
            return;
        }

        if (!activeTags.length) {
            tagListEl.innerHTML = '<div class="placeholder">' + emptyMsg + '</div>';
            return;
        }

        // Tag filter function: regex or substring
        function matchesFilter(tag) {
            if (!rawFilter) return true;
            if (isRegexMode) {
                // On regex error, keep previous results (show all)
                if (!filterRegex) return true;
                return filterRegex.test(tag);
            }
            return tag.toLowerCase().indexOf(filter) !== -1;
        }

        // Group tags by prefix (e.g., "loss/train" -> group "loss")
        var groups = {};
        activeTags.forEach(function(tag) {
            var parts = tag.split('/');
            var group = parts.length > 1 ? parts[0] : '';
            if (!groups[group]) groups[group] = [];
            groups[group].push(tag);
        });

        var html = '';
        Object.keys(groups).sort().forEach(function(group) {
            var groupTags = groups[group].filter(matchesFilter);
            if (!groupTags.length) return;

            if (group) {
                var isCollapsed = !!state.collapsedGroups[group];
                var chevronClass = isCollapsed ? 'tag-chevron-collapsed' : 'tag-chevron-expanded';
                var allSelected = groupTags.every(function(t) { return selectedSet.has(t); });
                var someSelected = !allSelected && groupTags.some(function(t) { return selectedSet.has(t); });
                var selectAllChecked = allSelected ? ' checked' : '';

                html += '<div class="tag-group-header tag-group-collapsible" data-group="' + escapeHtml(group) + '">';
                html += '<span class="tag-chevron ' + chevronClass + '"></span>';
                html += '<input type="checkbox" class="tag-group-select-all" data-group="' + escapeHtml(group) + '"' +
                        selectAllChecked + (someSelected ? ' data-indeterminate="true"' : '') + '>';
                html += '<span class="tag-group-label">' + escapeHtml(group) + '</span>';
                html += '</div>';

                if (isCollapsed) return; // skip rendering child tags if collapsed
            }

            groupTags.forEach(function(tag) {
                var checked = selectedSet.has(tag) ? ' checked' : '';
                var countBadge = '';
                if (state.tagMetrics && state.tagMetrics[tag]) {
                    var m = state.tagMetrics[tag];
                    countBadge = '<span class="tag-count-badge">' + m.count + ' pts</span>';
                }
                html += '<label class="checkbox-item tag-item">' +
                    '<input type="checkbox" value="' + escapeHtml(tag) + '"' + checked + '>' +
                    '<span class="checkbox-label">' + escapeHtml(tag) + countBadge + '</span>' +
                    '</label>';
            });
        });

        if (!html) {
            tagListEl.innerHTML = '<div class="placeholder">No tags match filter</div>';
            return;
        }

        tagListEl.innerHTML = html;

        // Set indeterminate state for select-all checkboxes (can't be set via HTML attribute)
        var selectAllCheckboxes = tagListEl.querySelectorAll('.tag-group-select-all[data-indeterminate="true"]');
        selectAllCheckboxes.forEach(function(cb) {
            cb.indeterminate = true;
        });

        // Attach change handlers for individual tag checkboxes
        var checkboxes = tagListEl.querySelectorAll('.tag-item input[type="checkbox"]');
        checkboxes.forEach(function(cb) {
            cb.addEventListener('change', function() {
                toggleFn(cb.value, cb.checked);
            });
        });

        // Attach click handlers for collapsible group headers
        var groupHeaders = tagListEl.querySelectorAll('.tag-group-collapsible');
        groupHeaders.forEach(function(header) {
            // Click on the header text or chevron toggles collapse
            header.addEventListener('click', function(e) {
                // Don't toggle collapse when clicking the select-all checkbox
                if (e.target.classList.contains('tag-group-select-all')) return;
                var group = header.getAttribute('data-group');
                state.collapsedGroups[group] = !state.collapsedGroups[group];
                renderTagList();
            });
        });

        // Attach change handlers for select-all group checkboxes
        var groupSelectAlls = tagListEl.querySelectorAll('.tag-group-select-all');
        groupSelectAlls.forEach(function(cb) {
            cb.addEventListener('change', function(e) {
                e.stopPropagation(); // prevent header click from firing
                var group = cb.getAttribute('data-group');
                var groupTags = (groups[group] || []).filter(matchesFilter);
                groupTags.forEach(function(tag) {
                    if (cb.checked && !selectedSet.has(tag)) {
                        toggleFn(tag, true);
                    } else if (!cb.checked && selectedSet.has(tag)) {
                        toggleFn(tag, false);
                    }
                });
                renderTagList();
            });
        });
    }

    // ─── Event handlers ───────────────────────────────────────────────

    function deleteRun(runName) {
        fetch('/api/runs/' + encodeURIComponent(runName), { method: 'DELETE' })
            .then(function(resp) {
                if (!resp.ok) throw new Error('Delete failed');
                state.selectedRuns.delete(runName);
                state.runs = state.runs.filter(function(r) { return r.name !== runName; });
                renderRunList();
                rebuildTagList();
                renderAllCharts();
            })
            .catch(function(err) {
                alert('Failed to delete run: ' + err.message);
            });
    }

    function onRunToggle(runName, selected) {
        if (selected) {
            state.selectedRuns.add(runName);
            // Fetch tags for newly selected run
            fetchTags(runName).then(function() {
                rebuildTagList();
                // Fetch data for existing selected tags (scalars)
                state.selectedTags.forEach(function(tag) {
                    loadChartData(runName, tag);
                });
                // Fetch data for existing selected histogram tags
                state.selectedHistogramTags.forEach(function(tag) {
                    loadHistogramData(runName, tag);
                });
                updateLiveSubscription();
                // Refresh hparams if on that tab
                if (state.activeTab === 'hparams') {
                    refreshHParams();
                }
                // Refresh custom scalars if on scalars tab
                if (state.activeTab === 'scalars') {
                    refreshCustomScalars();
                }
                // Refresh PR curves / audio if on those tabs
                if (state.activeTab === 'pr-curves') {
                    refreshPRCurves();
                }
                if (state.activeTab === 'audio') {
                    refreshAudio();
                }
            });
        } else {
            state.selectedRuns.delete(runName);
            rebuildTagList();
            // Remove run traces from all scalar charts
            getChartIds().forEach(function(chartId) {
                removeRunFromChart(chartId, runName);
            });
            // Clean up empty charts
            cleanupEmptyCharts();
            updateLiveSubscription();
            // Refresh hparams if on that tab
            if (state.activeTab === 'hparams') {
                refreshHParams();
            }
            // Refresh custom scalars if on scalars tab
            if (state.activeTab === 'scalars') {
                refreshCustomScalars();
            }
            // Refresh PR curves / audio if on those tabs
            if (state.activeTab === 'pr-curves') {
                refreshPRCurves();
            }
            if (state.activeTab === 'audio') {
                refreshAudio();
            }
        }
        updateRunNotes();
        updateUrlHash();
    }

    function onTagToggle(tag, selected) {
        if (selected) {
            state.selectedTags.add(tag);
            ensureChartExists(tag);
            // Fetch data for all selected runs
            state.selectedRuns.forEach(function(runName) {
                loadChartData(runName, tag);
            });
            updateLiveSubscription();
        } else {
            state.selectedTags.delete(tag);
            var chartId = tagToChartId(tag);
            destroyChart(chartId);
            var chartEl = document.getElementById(chartId);
            if (chartEl) {
                // Remove the wrapper (.chart-panel-wrapper) if it exists, otherwise just the panel
                var wrapper = chartEl.parentNode;
                if (wrapper && wrapper.classList.contains('chart-panel-wrapper')) {
                    wrapper.parentNode.removeChild(wrapper);
                } else if (chartEl.parentNode) {
                    chartEl.parentNode.removeChild(chartEl);
                }
            }
            updateLiveSubscription();
        }
        updateEmptyState();
        updateUrlHash();
    }

    function onHistogramTagToggle(tag, selected) {
        if (selected) {
            state.selectedHistogramTags.add(tag);
            ensureHistogramChartExists(tag);
            // Fetch histogram data for all selected runs
            state.selectedRuns.forEach(function(runName) {
                loadHistogramData(runName, tag);
            });
        } else {
            state.selectedHistogramTags.delete(tag);
            var chartId = histogramTagToChartId(tag);
            destroyHistogramChart(chartId);
            var chartEl = document.getElementById(chartId);
            if (chartEl) {
                var wrapper = chartEl.parentNode;
                if (wrapper && wrapper.classList.contains('histogram-panel-wrapper')) {
                    wrapper.parentNode.removeChild(wrapper);
                } else if (chartEl.parentNode) {
                    chartEl.parentNode.removeChild(chartEl);
                }
            }
        }
        updateHistogramEmptyState();
    }

    function onPRCurveTagToggle(tag, selected) {
        if (selected) {
            state.selectedPRCurveTags.add(tag);
        } else {
            state.selectedPRCurveTags.delete(tag);
        }
        refreshPRCurves();
    }

    function onAudioTagToggle(tag, selected) {
        if (selected) {
            state.selectedAudioTags.add(tag);
        } else {
            state.selectedAudioTags.delete(tag);
        }
        if (state.activeTab === 'audio') {
            refreshAudio();
        }
    }

    // ─── Tab switching ──────────────────────────────────────────────

    function switchTab(tabName) {
        if (state.activeTab === tabName) return;
        state.activeTab = tabName;

        // Update tab button styles
        var buttons = tabBarEl.querySelectorAll('.tab-btn');
        buttons.forEach(function(btn) {
            if (btn.getAttribute('data-tab') === tabName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Show/hide panels
        var panels = [
            [panelScalarsEl, 'scalars'],
            [panelHistogramsEl, 'histograms'],
            [panelHparamsEl, 'hparams'],
            [panelTracesEl, 'traces'],
            [panelEvalEl, 'eval'],
            [panelArtifactsEl, 'artifacts'],
            [panelPRCurvesEl, 'pr-curves'],
            [panelAudioEl, 'audio'],
            [panelProjectorEl, 'projector'],
            [panelLoraEl, 'lora'],
        ];
        panels.forEach(function(pair) {
            if (pair[0]) {
                pair[0].style.display = tabName === pair[1] ? '' : 'none';
                pair[0].classList.toggle('active', tabName === pair[1]);
            }
        });

        // Re-render tag list for the active tab context
        renderTagList();

        // Tab-specific actions (wrapped to prevent one broken tab from
        // crashing the entire UI)
        try {
        if (tabName === 'hparams') {
            refreshHParams();
        } else if (tabName === 'traces') {
            refreshTraces();
        } else if (tabName === 'eval') {
            refreshEval();
        } else if (tabName === 'artifacts') {
            refreshArtifacts();
        } else if (tabName === 'pr-curves') {
            refreshPRCurves();
        } else if (tabName === 'audio') {
            refreshAudio();
        } else if (tabName === 'projector') {
            refreshProjector();
        } else if (tabName === 'scalars') {
            // Recreate chart elements if they were lost
            state.selectedTags.forEach(function(tag) {
                ensureChartExists(tag);
            });
            updateEmptyState();
            // Reset sync state so arriving data triggers full x-axis equalization
            resetGlobalSyncState();
            refreshCustomScalars();
        }
        } catch (err) {
            console.error('Tab render error (' + tabName + '):', err);
        }

        // Resize visible charts
        if (tabName === 'scalars') {
            // Deferred re-render: browser needs to reflow after display change
            // so chart containers have proper dimensions for Plotly/WebGL.
            setTimeout(function() {
                reSmoothAllCharts();
                resyncAllScalarData();
                syncScalarXAxisRange();
            }, 50);
        } else if (tabName === 'histograms') {
            getHistogramChartIds().forEach(function(chartId) {
                var inst = getChartInstance(chartId);
                if (inst) inst.resize();
            });
        } else if (tabName === 'projector') {
            // Resize projector plot after becoming visible (hidden tabs have zero dimensions)
            var projInst = getChartInstance('projector-plot');
            if (projInst) {
                setTimeout(function() { projInst.resize(); }, 50);
            }
        } else if (tabName === 'lora') {
            if (typeof window.resizeLoraCharts === 'function') {
                setTimeout(window.resizeLoraCharts, 50);
            }
        }

        updateUrlHash();
    }

    // ─── HParams refresh ────────────────────────────────────────────

    function refreshHParams() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) {
            hparamsContainerEl.innerHTML =
                '<div class="empty-state"><p>Select runs to compare hyperparameters</p></div>';
            return;
        }

        fetchHParams(runs).then(function(data) {
            state.hparamsData = data;

            if (!data || !data.length) {
                hparamsContainerEl.innerHTML =
                    '<div class="empty-state"><p>No hyperparameter data available for selected runs</p></div>';
                return;
            }

            // Ensure chart container exists
            var chartId = 'hparams-parcoords';
            var chartEl = document.getElementById(chartId);
            if (!chartEl) {
                hparamsContainerEl.innerHTML = '';
                chartEl = document.createElement('div');
                chartEl.id = chartId;
                chartEl.className = 'hparams-chart';
                hparamsContainerEl.appendChild(chartEl);
            }

            createHParamsChart(chartId, data);
        }).catch(function(err) {
            console.error('Failed to fetch hparams:', err);
            hparamsContainerEl.innerHTML =
                '<div class="empty-state"><p>Failed to load hyperparameter data</p></div>';
        });
    }

    // ─── Traces refresh ─────────────────────────────────────────────

    function refreshTraces() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) {
            tracesContainerEl.innerHTML =
                '<div class="empty-state"><p>Select runs to view trace timeline</p></div>';
            return;
        }

        tracesContainerEl.innerHTML = '<div class="traces-loading">Loading traces...</div>';

        var promises = runs.map(function(runName) {
            return fetchTraces(runName).then(function(events) {
                return { run: runName, events: events };
            });
        });

        Promise.all(promises).then(function(results) {
            var hasData = results.some(function(r) { return r.events && r.events.length; });
            if (!hasData) {
                tracesContainerEl.innerHTML =
                    '<div class="empty-state"><p>No trace events available for selected runs</p></div>';
                return;
            }

            var chartId = 'traces-timeline';
            tracesContainerEl.innerHTML = '';
            var chartEl = document.createElement('div');
            chartEl.id = chartId;
            chartEl.className = 'traces-chart';
            tracesContainerEl.appendChild(chartEl);

            createTraceTimeline(chartId, results);
        }).catch(function(err) {
            console.error('Failed to fetch traces:', err);
            tracesContainerEl.innerHTML =
                '<div class="empty-state"><p>Failed to load trace data</p></div>';
        });
    }

    // ─── Eval refresh ───────────────────────────────────────────────

    function refreshEval() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) {
            evalContainerEl.innerHTML =
                '<div class="empty-state"><p>Select runs to view evaluation results</p></div>';
            return;
        }

        evalContainerEl.innerHTML = '<div class="eval-loading">Loading eval results...</div>';
        refreshSelectedRunTags().then(function() {
            // Get tags for first selected run to discover eval suites
            var firstRun = runs[0];
            var tags = state.tagsByRun[firstRun];
            var suites = (tags && tags.eval_suites) ? tags.eval_suites : [];

            if (!suites.length) {
                evalContainerEl.innerHTML =
                    '<div class="empty-state"><p>No evaluation suites found for selected runs</p></div>';
                return;
            }

            // Fetch eval data for all runs and all suites
            var promises = [];
            runs.forEach(function(runName) {
                suites.forEach(function(suite) {
                    promises.push(
                        fetchEval(runName, suite).then(function(results) {
                            return { run: runName, suite: suite, results: results };
                        })
                    );
                });
            });

            Promise.all(promises).then(function(allResults) {
                var hasData = allResults.some(function(r) { return r.results && r.results.length; });
                if (!hasData) {
                    evalContainerEl.innerHTML =
                        '<div class="empty-state"><p>No evaluation results available</p></div>';
                    return;
                }

                var gridId = 'eval-grid';
                evalContainerEl.innerHTML = '';
                var gridEl = document.createElement('div');
                gridEl.id = gridId;
                gridEl.className = 'eval-grid';
                evalContainerEl.appendChild(gridEl);

                createEvalGrid(gridId, allResults);
            }).catch(function(err) {
                console.error('Failed to fetch eval results:', err);
                evalContainerEl.innerHTML =
                    '<div class="empty-state"><p>Failed to load evaluation data</p></div>';
            });
        });
    }

    // ─── Artifacts refresh ──────────────────────────────────────────

    function refreshArtifacts() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) {
            artifactsContainerEl.innerHTML =
                '<div class="empty-state"><p>Select runs to view artifacts</p></div>';
            return;
        }

        artifactsContainerEl.innerHTML = '<div class="artifacts-loading">Loading artifacts...</div>';
        refreshSelectedRunTags().then(function() {
            // Get artifact tags from first selected run
            var firstRun = runs[0];
            var tags = state.tagsByRun[firstRun];
            var artifactTags = (tags && tags.artifacts) ? tags.artifacts : [];

            if (!artifactTags.length) {
                artifactsContainerEl.innerHTML =
                    '<div class="empty-state"><p>No artifacts found for selected runs</p></div>';
                return;
            }

            // Fetch artifacts for all runs and all tags
            var promises = [];
            runs.forEach(function(runName) {
                artifactTags.forEach(function(tag) {
                    promises.push(
                        fetchArtifacts(runName, tag).then(function(artifacts) {
                            return { run: runName, tag: tag, artifacts: artifacts };
                        })
                    );
                });
            });

            Promise.all(promises).then(function(allResults) {
                var hasData = allResults.some(function(r) { return r.artifacts && r.artifacts.length; });
                if (!hasData) {
                    artifactsContainerEl.innerHTML =
                        '<div class="empty-state"><p>No artifacts available</p></div>';
                    return;
                }

                var galleryId = 'artifact-gallery';
                artifactsContainerEl.innerHTML = '';
                var galleryEl = document.createElement('div');
                galleryEl.id = galleryId;
                galleryEl.className = 'artifact-gallery';
                artifactsContainerEl.appendChild(galleryEl);

                createArtifactGallery(galleryId, allResults);
            }).catch(function(err) {
                console.error('Failed to fetch artifacts:', err);
                artifactsContainerEl.innerHTML =
                    '<div class="empty-state"><p>Failed to load artifacts</p></div>';
            });
        });
    }

    // ─── PR Curves refresh ───────────────────────────────────────────

    function refreshPRCurves() {
        if (!prCurvesContainerEl) return;
        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedPRCurveTags);
        if (runs.length === 0 || tags.length === 0) {
            prCurvesContainerEl.innerHTML = '<div class="empty-state"><p>Select runs and PR curve tags to view</p></div>';
            return;
        }
        prCurvesContainerEl.innerHTML = '<div class="empty-state"><p>Loading PR curves...</p></div>';

        // For each tag, create a separate chart section
        var allPromises = [];
        var tagResults = {};

        tags.forEach(function(tag) {
            tagResults[tag] = {};
            runs.forEach(function(run) {
                allPromises.push(
                    fetchPRCurves(run, tag).then(function(data) {
                        tagResults[tag][run] = data;
                    })
                );
            });
        });

        Promise.all(allPromises).then(function() {
            prCurvesContainerEl.innerHTML = '';
            tags.forEach(function(tag) {
                var section = document.createElement('div');
                section.className = 'pr-curve-section';

                var heading = document.createElement('h3');
                heading.className = 'pr-curve-tag-heading';
                heading.textContent = tag;
                section.appendChild(heading);

                var chartContainer = document.createElement('div');
                var chartContainerId = 'pr-curve-' + tag.replace(/[^a-zA-Z0-9]/g, '_');
                chartContainer.id = chartContainerId;
                section.appendChild(chartContainer);

                prCurvesContainerEl.appendChild(section);
                createPRCurveChart(chartContainerId, tagResults[tag]);
            });
        }).catch(function(err) {
            console.error('Failed to fetch PR curves:', err);
            prCurvesContainerEl.innerHTML = '<div class="empty-state"><p>Error loading PR curves</p></div>';
        });
    }

    // ─── Audio refresh ───────────────────────────────────────────────

    function refreshAudio() {
        var container = document.getElementById('audio-container');
        if (!container) return;
        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedAudioTags);
        if (runs.length === 0 || tags.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>Select runs and audio tags to listen</p></div>';
            return;
        }
        container.innerHTML = '<div class="empty-state"><p>Loading audio...</p></div>';

        var promises = [];
        var resultMap = {};
        runs.forEach(function(run) {
            tags.forEach(function(tag) {
                promises.push(
                    fetchAudio(run, tag).then(function(data) {
                        if (!resultMap[run]) resultMap[run] = [];
                        resultMap[run] = resultMap[run].concat(data);
                    })
                );
            });
        });

        Promise.all(promises).then(function() {
            createAudioGallery('audio-container', resultMap);
        }).catch(function(err) {
            console.error('Failed to fetch audio:', err);
            container.innerHTML = '<div class="empty-state"><p>Error loading audio</p></div>';
        });
    }

    // ─── Projector refresh ──────────────────────────────────────────

    function refreshProjector() {
        if (typeof window.projectorSetRuns === 'function') {
            window.projectorSetRuns(state.runs);
        }
    }

    // ─── Custom Scalars ─────────────────────────────────────────────

    function refreshCustomScalars() {
        var runs = Array.from(state.selectedRuns);
        if (!runs.length) return;

        // Fetch layout from first selected run
        fetchCustomScalarsLayout(runs[0]).then(function(layout) {
            state.customScalarsLayout = layout;
            if (!layout || !layout.categories || !layout.categories.length) {
                // No custom layout - hide toggle if present
                var toggle = document.getElementById('custom-scalars-toggle');
                if (toggle) toggle.style.display = 'none';
                return;
            }
            // Show toggle
            ensureCustomScalarsToggle();
            if (state.customScalarsMode) {
                renderCustomScalarsView(runs, layout);
            }
        }).catch(function(err) {
            console.error('Failed to fetch custom scalars layout:', err);
        });
    }

    function ensureCustomScalarsToggle() {
        var existing = document.getElementById('custom-scalars-toggle');
        if (existing) {
            existing.style.display = '';
            return;
        }

        var toggle = document.createElement('div');
        toggle.id = 'custom-scalars-toggle';
        toggle.className = 'custom-scalars-toggle';

        var stdBtn = document.createElement('button');
        stdBtn.className = 'custom-scalars-mode-btn custom-scalars-mode-active';
        stdBtn.textContent = 'Standard';
        stdBtn.setAttribute('data-mode', 'standard');

        var customBtn = document.createElement('button');
        customBtn.className = 'custom-scalars-mode-btn';
        customBtn.textContent = 'Custom Layout';
        customBtn.setAttribute('data-mode', 'custom');

        toggle.appendChild(stdBtn);
        toggle.appendChild(customBtn);

        // Insert before chart grid
        panelScalarsEl.insertBefore(toggle, chartGridEl);

        [stdBtn, customBtn].forEach(function(btn) {
            btn.addEventListener('click', function() {
                var mode = btn.getAttribute('data-mode');
                state.customScalarsMode = (mode === 'custom');
                stdBtn.classList.toggle('custom-scalars-mode-active', mode === 'standard');
                customBtn.classList.toggle('custom-scalars-mode-active', mode === 'custom');

                if (state.customScalarsMode) {
                    chartGridEl.style.display = 'none';
                    ensureCustomScalarsContainer();
                    renderCustomScalarsView(Array.from(state.selectedRuns), state.customScalarsLayout);
                } else {
                    chartGridEl.style.display = '';
                    var container = document.getElementById('custom-scalars-container');
                    if (container) container.style.display = 'none';
                }
            });
        });
    }

    function ensureCustomScalarsContainer() {
        var existing = document.getElementById('custom-scalars-container');
        if (existing) {
            existing.style.display = '';
            return;
        }
        var container = document.createElement('div');
        container.id = 'custom-scalars-container';
        container.className = 'custom-scalars-container';
        panelScalarsEl.appendChild(container);
    }

    function renderCustomScalarsView(runs, layout) {
        var container = document.getElementById('custom-scalars-container');
        if (!container || !layout || !layout.categories) return;

        container.innerHTML = '';

        layout.categories.forEach(function(category) {
            var section = document.createElement('div');
            section.className = 'custom-scalars-category';

            var heading = document.createElement('h3');
            heading.className = 'custom-scalars-category-title';
            heading.textContent = category.title;
            section.appendChild(heading);

            var chartsGrid = document.createElement('div');
            chartsGrid.className = 'custom-scalars-charts-grid';

            (category.charts || []).forEach(function(chartDef) {
                var chartWrapper = document.createElement('div');
                chartWrapper.className = 'chart-panel-wrapper';

                var chartEl = document.createElement('div');
                chartEl.className = 'chart-panel custom-scalar-chart';
                var chartId = 'custom-chart-' + hashString(category.title + chartDef.title);
                chartEl.id = chartId;
                chartWrapper.appendChild(chartEl);
                chartsGrid.appendChild(chartWrapper);

                // Collect all tag patterns from this chart definition
                var tagPatterns = chartDef.tags || [];

                // Fetch data for ALL runs, then render once with combined traces
                var runPromises = runs.map(function(runName) {
                    return fetchCustomScalarsData(runName, tagPatterns, 5000).then(function(data) {
                        return { run: runName, data: data };
                    }).catch(function(err) {
                        console.error('Failed to fetch custom scalars data:', err);
                        return { run: runName, data: null };
                    });
                });

                Promise.all(runPromises).then(function(results) {
                    var series = [];
                    results.forEach(function(result) {
                        if (!result.data) return;
                        var tags = Object.keys(result.data).sort();
                        tags.forEach(function(tag, idx) {
                            var rows = result.data[tag];
                            if (!rows || !rows.length) return;

                            var color = RUN_COLORS[(idx + hashString(result.run)) % RUN_COLORS.length];
                            series.push({
                                name: (runs.length > 1 ? result.run + ' / ' : '') + tag,
                                type: 'line',
                                large: true,
                                largeThreshold: 2000,
                                sampling: 'lttb',
                                showSymbol: false,
                                lineStyle: { width: 2, color: color },
                                itemStyle: { color: color },
                                data: rows.map(function(r) { return [r[0], r[2]]; }),
                            });
                        });
                    });

                    var instance = getOrCreateChart(chartId);
                    if (instance) {
                        instance.setOption({
                            title: { text: chartDef.title },
                            tooltip: { trigger: 'axis' },
                            xAxis: { type: 'value', name: 'Step' },
                            yAxis: { type: 'value', name: 'Value', scale: true },
                            series: series,
                            animation: false,
                        }, { replaceMerge: ['series'] });
                    }
                });
            });

            section.appendChild(chartsGrid);
            container.appendChild(section);
        });
    }

    // ─── Run Notes ─────────────────────────────────────────────────────

    var _notesSaveTimer = null;
    var _notesCollapsed = false;
    var _notesCurrentRun = null;

    function updateRunNotes() {
        var selected = Array.from(state.selectedRuns);
        if (selected.length === 1) {
            var runName = selected[0];
            _notesCurrentRun = runName;
            runNotesSectionEl.style.display = '';
            // Fetch current note
            fetchJsonNoCache('/api/runs/' + encodeURIComponent(runName) + '/notes')
                .then(function(data) {
                    if (_notesCurrentRun !== runName) return; // stale response
                    runNotesTextareaEl.value = data.note || '';
                })
                .catch(function(err) {
                    console.warn('Failed to fetch notes:', err);
                    runNotesTextareaEl.value = '';
                });
        } else {
            _notesCurrentRun = null;
            runNotesSectionEl.style.display = 'none';
            runNotesTextareaEl.value = '';
        }
    }

    function saveRunNote() {
        var runName = _notesCurrentRun;
        if (!runName) return;
        var text = runNotesTextareaEl.value;
        fetch('/api/runs/' + encodeURIComponent(runName) + '/notes', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ note: text }),
        }).then(function(resp) {
            if (!resp.ok) throw new Error('Save failed');
            showNotesSaved();
        }).catch(function(err) {
            console.warn('Failed to save note:', err);
        });
    }

    function showNotesSaved() {
        runNotesSavedEl.textContent = 'Saved';
        runNotesSavedEl.classList.add('visible');
        setTimeout(function() {
            runNotesSavedEl.classList.remove('visible');
        }, 1500);
    }

    function scheduleNoteSave() {
        if (_notesSaveTimer) clearTimeout(_notesSaveTimer);
        _notesSaveTimer = setTimeout(function() {
            _notesSaveTimer = null;
            saveRunNote();
        }, 1000);
    }

    function setupEventListeners() {
        // Tab bar
        tabBarEl.addEventListener('click', function(e) {
            var btn = e.target.closest('.tab-btn');
            if (btn) {
                switchTab(btn.getAttribute('data-tab'));
            }
        });

        // Smoothing slider
        smoothingSliderEl.addEventListener('input', function() {
            state.smoothingWeight = parseFloat(smoothingSliderEl.value);
            smoothingValueEl.textContent = state.smoothingWeight.toFixed(3);
            reSmoothAllCharts();
            updateUrlHash();
        });

        // X-axis mode
        xAxisModeEl.addEventListener('change', function() {
            state.xAxisMode = xAxisModeEl.value;
            reSmoothAllCharts();
            updateUrlHash();
        });

        // Reset all chart axes to auto range.
        if (resetAxesEl) {
            resetAxesEl.addEventListener('click', function() {
                resetAllAxes();
                resyncAllScalarData();
            });
        }

        // Export scalars button
        if (exportScalarsEl) {
            exportScalarsEl.addEventListener('click', function() {
                exportAllVisibleScalars();
            });
        }

        // Tag filter
        tagFilterEl.addEventListener('input', function() {
            renderTagList();
        });

        // Sidebar toggle
        sidebarToggleEl.addEventListener('click', function() {
            sidebarEl.classList.toggle('collapsed');
        });

        // Window resize -> resize all ECharts instances
        window.addEventListener('resize', function() {
            resizeAllCharts();
        });

        // Run notes: debounced auto-save on input, immediate save on blur
        runNotesTextareaEl.addEventListener('input', function() {
            scheduleNoteSave();
        });
        runNotesTextareaEl.addEventListener('blur', function() {
            if (_notesSaveTimer) {
                clearTimeout(_notesSaveTimer);
                _notesSaveTimer = null;
            }
            saveRunNote();
        });

        // Collapsible notes heading
        var notesHeading = document.querySelector('.run-notes-heading');
        if (notesHeading) {
            notesHeading.addEventListener('click', function() {
                _notesCollapsed = !_notesCollapsed;
                runNotesBodyEl.style.display = _notesCollapsed ? 'none' : '';
            });
        }
    }

    // ─── Chart management ─────────────────────────────────────────────

    function tagToChartId(tag) {
        // Use a simple hash to avoid collisions between tags like "loss/train" and "loss.train"
        var hash = 0;
        for (var i = 0; i < tag.length; i++) {
            hash = ((hash << 5) - hash + tag.charCodeAt(i)) | 0;
        }
        return 'chart-' + tag.replace(/[^a-zA-Z0-9]/g, '_') + '_' + (hash >>> 0).toString(16);
    }

    function ensureChartExists(tag) {
        var chartId = tagToChartId(tag);
        if (document.getElementById(chartId)) return chartId;

        // Remove empty state message
        var emptyState = chartGridEl.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        // Create chart panel wrapper with header
        var wrapper = document.createElement('div');
        wrapper.className = 'chart-panel-wrapper';

        // CSV download button
        var csvBtn = document.createElement('button');
        csvBtn.className = 'chart-csv-btn';
        csvBtn.title = 'Download CSV';
        csvBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">' +
            '<path d="M7 0v8m0 0L4 5.5M7 8l3-2.5"/>' +
            '<path d="M1 10v2.5h12V10" fill="none" stroke="currentColor" stroke-width="1.5"/>' +
            '</svg>';
        csvBtn.addEventListener('click', function() {
            exportSingleChartCSV(chartId);
        });
        wrapper.appendChild(csvBtn);

        // Create chart panel
        var panel = document.createElement('div');
        panel.className = 'chart-panel';
        panel.id = chartId;
        wrapper.appendChild(panel);

        chartGridEl.appendChild(wrapper);

        createChart(chartId, tag);
        return chartId;
    }

    function loadChartData(runName, tag) {
        var chartId = tagToChartId(tag);
        var requestKey = runName + '\x00' + tag;
        if (state.scalarFetchInflight[requestKey]) {
            state.scalarFetchPending[requestKey] = true;
            return;
        }
        state.scalarFetchInflight[requestKey] = true;
        state.scalarFetchPending[requestKey] = false;
        var requestSeq = (state.scalarFetchSeq[requestKey] || 0) + 1;
        state.scalarFetchSeq[requestKey] = requestSeq;

        fetchScalars(runName, tag)
            .then(function(data) {
                // Drop out-of-order responses to avoid older payloads overwriting newer chart state.
                if (state.scalarFetchSeq[requestKey] !== requestSeq) return;
                // Ignore responses for runs/tags that were deselected while the request was in-flight.
                if (!state.selectedRuns.has(runName) || !state.selectedTags.has(tag)) return;
                if (!data || !data.length) {
                    // Important: empty payload must clear stale in-memory trace for this run/tag.
                    // Otherwise old points can survive forever and distort per-chart x-ranges.
                    removeRunFromChart(chartId, runName);
                    updateDataStatus();
                    return;
                }

                // Record run start time (wall_time of first data point)
                if (!state.runStartTimes[runName] && data.length > 0) {
                    state.runStartTimes[runName] = data[0][1];
                }

                // Guard rail: never let an apparently truncated full-fetch payload
                // shrink an already-longer local series for the same run/tag.
                var incomingMax = data[data.length - 1][0];
                var localMax = getLocalSeriesMaxStep(tag, runName);
                if (localMax !== null && Number.isFinite(incomingMax) && incomingMax < localMax) {
                    console.warn(
                        'Ignoring truncated scalar payload',
                        runName, tag,
                        'incoming max:', incomingMax,
                        'local max:', localMax
                    );
                    state.scalarFetchPending[requestKey] = true;
                    return;
                }

                updateChart(
                    chartId, runName, data,
                    state.smoothingWeight, state.xAxisMode,
                    state.runStartTimes[runName]
                );
                updateDataStatus();
            })
            .catch(function(err) {
                console.error('Failed to load scalars for', runName, tag, err);
            })
            .finally(function() {
                state.scalarFetchInflight[requestKey] = false;
                if (state.scalarFetchPending[requestKey]) {
                    state.scalarFetchPending[requestKey] = false;
                    if (state.selectedRuns.has(runName) && state.selectedTags.has(tag)) {
                        loadChartData(runName, tag);
                    }
                }
            });
    }

    function reSmoothAllCharts() {
        getChartIds().forEach(function(chartId) {
            reSmoothChart(chartId, state.smoothingWeight, state.xAxisMode, state.runStartTimes);
        });
    }

    function resetAllAxes() {
        // Reset all chart zoom to full range, then re-render
        getChartIds().forEach(function(chartId) {
            var inst = getChartInstance(chartId);
            if (!inst) return;
            inst.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
        });
        reSmoothAllCharts();
    }

    function resyncAllScalarData() {
        if (!state.selectedRuns.size || !state.selectedTags.size) return;

        state.selectedRuns.forEach(function(runName) {
            state.selectedTags.forEach(function(tag) {
                loadChartData(runName, tag);
            });
        });
    }

    function resyncRunScalarData(runName) {
        if (!state.selectedRuns.has(runName) || !state.selectedTags.size) return;
        state.selectedTags.forEach(function(tag) {
            loadChartData(runName, tag);
        });
    }

    function syncScalarXAxisRange() {
        if (state.xAxisMode !== 'step') return;
        if (!state.selectedRuns.size || !state.selectedTags.size) return;

        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedTags);

        Promise.all(runs.map(function(runName) {
            return fetchScalarsLast(runName, tags).then(function(lastByTag) {
                return { run: runName, lastByTag: lastByTag || {} };
            }).catch(function() {
                return { run: runName, lastByTag: {} };
            });
        })).then(function(results) {
            var maxStep = null;
            results.forEach(function(entry) {
                tags.forEach(function(tag) {
                    var point = entry.lastByTag[tag];
                    if (!point || typeof point.step !== 'number') return;
                    if (maxStep === null || point.step > maxStep) {
                        maxStep = point.step;
                    }
                });
            });
            if (maxStep === null) {
                // Keep previous global max on transient fetch errors; this prevents
                // temporary range collapse across charts.
                return;
            }
            setGlobalScalarStepMax(maxStep);

            // Re-render all charts so _buildScalarLayout picks up the updated
            // global max step and deterministic tick marks.  This replaces the
            // previous Plotly.relayout() calls that raced with Plotly.react()
            // in _renderScalarChart, causing inconsistent axis display.
            reSmoothAllCharts();
        });
    }

    function pollScalarFallback() {
        if (state.activeTab !== 'scalars') return;
        if (!state.selectedRuns.size || !state.selectedTags.size) return;
        resyncAllScalarData();
        reconcileScalarSeriesCompleteness();
        syncScalarXAxisRange();
    }

    function getLocalSeriesMaxStep(tag, runName) {
        var chartId = tagToChartId(tag);
        var chart = getChartInfo(chartId);
        if (!chart || !chart.traces || !chart.traces[runName]) return null;
        var rows = chart.traces[runName].rawPoints || [];
        if (!rows.length) return null;
        var last = rows[rows.length - 1];
        return Array.isArray(last) ? last[0] : null;
    }

    function reconcileScalarSeriesCompleteness() {
        if (!state.selectedRuns.size || !state.selectedTags.size) return;

        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedTags);

        Promise.all(runs.map(function(runName) {
            return fetchScalarsLast(runName, tags).then(function(lastByTag) {
                return { run: runName, lastByTag: lastByTag || {} };
            }).catch(function() {
                return { run: runName, lastByTag: {} };
            });
        })).then(function(results) {
            results.forEach(function(entry) {
                var runName = entry.run;
                tags.forEach(function(tag) {
                    var serverPoint = entry.lastByTag[tag];
                    if (!serverPoint || typeof serverPoint.step !== 'number') return;
                    var localMax = getLocalSeriesMaxStep(tag, runName);
                    if (localMax === null || localMax < serverPoint.step) {
                        loadChartData(runName, tag);
                    }
                });
            });
        });
    }

    function cleanupEmptyCharts() {
        getChartIds().forEach(function(chartId) {
            var info = getChartInfo(chartId);
            if (!info) return;
            var hasTraces = Object.keys(info.traces).length > 0;
            if (!hasTraces) {
                destroyChart(chartId);
                var el = document.getElementById(chartId);
                if (el) {
                    var wrapper = el.parentNode;
                    if (wrapper && wrapper.classList.contains('chart-panel-wrapper')) {
                        wrapper.parentNode.removeChild(wrapper);
                    } else if (el.parentNode) {
                        el.parentNode.removeChild(el);
                    }
                }
            }
        });
        updateEmptyState();
    }

    function updateEmptyState() {
        var hasCharts = getChartIds().length > 0;
        var emptyState = chartGridEl.querySelector('.empty-state');
        if (!hasCharts && !emptyState) {
            var div = document.createElement('div');
            div.className = 'empty-state';
            div.innerHTML = '<p>Select runs and tags to view charts</p>';
            chartGridEl.appendChild(div);
        } else if (hasCharts && emptyState) {
            emptyState.remove();
        }
    }

    // ─── Histogram chart management ─────────────────────────────────

    function histogramTagToChartId(tag) {
        var hash = 0;
        for (var i = 0; i < tag.length; i++) {
            hash = ((hash << 5) - hash + tag.charCodeAt(i)) | 0;
        }
        return 'hist-' + tag.replace(/[^a-zA-Z0-9]/g, '_') + '_' + (hash >>> 0).toString(16);
    }

    function ensureHistogramChartExists(tag) {
        var chartId = histogramTagToChartId(tag);
        if (document.getElementById(chartId)) return chartId;

        // Remove empty state message
        var emptyState = histogramGridEl.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        // Create wrapper with mode toggle
        var wrapper = document.createElement('div');
        wrapper.className = 'histogram-panel-wrapper';

        var modeToggle = document.createElement('div');
        modeToggle.className = 'histogram-mode-toggle';
        var heatmapBtn = document.createElement('button');
        heatmapBtn.className = 'histogram-mode-btn histogram-mode-active';
        heatmapBtn.textContent = 'Heatmap';
        heatmapBtn.setAttribute('data-mode', 'heatmap');
        var ridgelineBtn = document.createElement('button');
        ridgelineBtn.className = 'histogram-mode-btn';
        ridgelineBtn.textContent = 'Ridgeline';
        ridgelineBtn.setAttribute('data-mode', 'ridgeline');
        var distributionBtn = document.createElement('button');
        distributionBtn.className = 'histogram-mode-btn';
        distributionBtn.textContent = 'Distribution';
        distributionBtn.setAttribute('data-mode', 'distribution');
        modeToggle.appendChild(heatmapBtn);
        modeToggle.appendChild(ridgelineBtn);
        modeToggle.appendChild(distributionBtn);
        wrapper.appendChild(modeToggle);

        // Create histogram panel
        var panel = document.createElement('div');
        panel.className = 'histogram-panel';
        panel.id = chartId;
        wrapper.appendChild(panel);

        histogramGridEl.appendChild(wrapper);

        // Attach mode toggle handlers
        var capturedChartId = chartId;
        var capturedTag = tag;
        [heatmapBtn, ridgelineBtn, distributionBtn].forEach(function(btn) {
            btn.addEventListener('click', function() {
                var mode = btn.getAttribute('data-mode');
                setHistogramMode(capturedChartId, mode);
                // Update all button active states
                heatmapBtn.classList.toggle('histogram-mode-active', mode === 'heatmap');
                ridgelineBtn.classList.toggle('histogram-mode-active', mode === 'ridgeline');
                distributionBtn.classList.toggle('histogram-mode-active', mode === 'distribution');
                // Fetch distribution data if switching to distribution mode
                if (mode === 'distribution') {
                    loadDistributionData(capturedChartId, capturedTag);
                }
            });
        });

        createHistogramChart(chartId, tag);
        return chartId;
    }

    function loadHistogramData(runName, tag) {
        var chartId = histogramTagToChartId(tag);

        fetchHistograms(runName, tag).then(function(data) {
            if (!data || !data.length) return;

            updateHistogramChart(chartId, runName, data);
            updateDataStatus();
        }).catch(function(err) {
            console.error('Failed to load histograms for', runName, tag, err);
        });
    }

    function loadDistributionData(chartId, tag) {
        var runs = Array.from(state.selectedRuns);
        runs.forEach(function(runName) {
            fetchDistributions(runName, tag).then(function(data) {
                if (!data || !data.length) return;
                updateDistributionChart(chartId, runName, data);
            }).catch(function(err) {
                console.error('Failed to load distributions for', runName, tag, err);
            });
        });
    }

    function updateHistogramEmptyState() {
        var hasCharts = getHistogramChartIds().length > 0;
        var emptyState = histogramGridEl.querySelector('.empty-state');
        if (!hasCharts && !emptyState) {
            var div = document.createElement('div');
            div.className = 'empty-state';
            div.innerHTML = '<p>Select runs and tensor tags to view histograms</p>';
            histogramGridEl.appendChild(div);
        } else if (hasCharts && emptyState) {
            emptyState.remove();
        }
    }

    // ─── Live connection ──────────────────────────────────────────────

    function setupLiveConnection() {
        liveConn.onStatus(function(status) {
            updateConnectionStatus(status);
        });

        liveConn.onMessage(function(msg) {
            handleLiveMessage(msg);
        });

        liveConn.connect();
    }

    function updateLiveSubscription() {
        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedTags);
        var kinds = ['scalar', 'trace', 'eval'];

        if (runs.length && tags.length) {
            liveConn.subscribe(runs, tags, kinds);
        } else if (runs.length) {
            // Subscribe for trace/eval even without tags selected
            liveConn.subscribe(runs, ['*'], kinds);
        } else {
            liveConn.unsubscribe();
        }
    }

    function handleLiveMessage(msg) {
        if (msg.type === 'scalar') {
            var tag = msg.tag;
            var run = msg.run;

            if (!state.selectedTags.has(tag) || !state.selectedRuns.has(run)) return;

            var chartId = tagToChartId(tag);
            if (!document.getElementById(chartId)) return;

            // Track run start time from live data
            if (!state.runStartTimes[run] && msg.points && msg.points.length) {
                state.runStartTimes[run] = msg.points[0].wall_time;
            }

            // Guard against session rollover mixing old/new streams on the same chart.
            if (msg.session_id) {
                var prevSession = state.runSessions[run];
                if (prevSession && prevSession !== msg.session_id) {
                    state.runSessions[run] = msg.session_id;
                    state.runStartTimes[run] = undefined;
                    state.selectedTags.forEach(function(tag) {
                        removeRunFromChart(tagToChartId(tag), run);
                    });
                    resyncRunScalarData(run);
                    updateDataStatus();
                    return;
                }
                state.runSessions[run] = msg.session_id;
            }

            // Use canonical DB reload on live updates to avoid client-side drift.
            loadChartData(run, tag);

            updateDataStatus();
        } else if (msg.type === 'trace') {
            // Refresh traces panel if active
            if (state.activeTab === 'traces') {
                refreshTraces();
            }
        } else if (msg.type === 'eval') {
            // Refresh eval panel if active
            if (state.activeTab === 'eval') {
                refreshEval();
            }
        } else if (msg.type === 'session_changed') {
            // Session changed — refresh all data
            console.log('Session changed for run:', msg.run, msg.old_session_id, '->', msg.new_session_id);
            state.runSessions[msg.run] = msg.new_session_id;
            state.runStartTimes[msg.run] = undefined;
            state.selectedTags.forEach(function(tag) {
                removeRunFromChart(tagToChartId(tag), msg.run);
            });
            resyncRunScalarData(msg.run);
            fetchRuns();
            if (state.activeTab === 'traces') refreshTraces();
            if (state.activeTab === 'eval') refreshEval();
            if (state.activeTab === 'artifacts') refreshArtifacts();
            if (state.activeTab === 'audio') refreshAudio();
        }
    }

    // ─── Status bar ───────────────────────────────────────────────────

    function updateConnectionStatus(status) {
        var dotClass = 'status-disconnected';
        var text = 'Disconnected';

        if (status === 'connected') {
            dotClass = 'status-connected';
            text = 'Connected';
        } else if (status === 'connecting') {
            dotClass = 'status-connecting';
            text = 'Connecting...';
        }

        connectionStatusEl.className = 'status-indicator ' + dotClass;
        connectionStatusEl.querySelector('.status-text').textContent = text;
    }

    function updateDataStatus() {
        var runCount = state.selectedRuns.size;
        var tagCount = state.selectedTags.size;
        var chartCount = getChartIds().length;

        var parts = [];
        if (runCount) parts.push(runCount + ' run' + (runCount !== 1 ? 's' : ''));
        if (tagCount) parts.push(tagCount + ' tag' + (tagCount !== 1 ? 's' : ''));
        if (chartCount) parts.push(chartCount + ' chart' + (chartCount !== 1 ? 's' : ''));

        dataStatusEl.textContent = parts.length ? parts.join(' | ') : '';
    }

    // ─── URL hash persistence ──────────────────────────────────────────

    var _hashUpdateTimer = null;

    function updateUrlHash() {
        // Debounce to avoid excessive history entries
        if (_hashUpdateTimer) clearTimeout(_hashUpdateTimer);
        _hashUpdateTimer = setTimeout(function() {
            var params = [];
            params.push('smoothing=' + state.smoothingWeight);
            params.push('xaxis=' + state.xAxisMode);
            params.push('tab=' + state.activeTab);
            if (state.selectedRuns.size) {
                params.push('runs=' + Array.from(state.selectedRuns).map(encodeURIComponent).join(','));
            }
            if (state.selectedTags.size) {
                params.push('tags=' + Array.from(state.selectedTags).map(encodeURIComponent).join(','));
            }
            var hash = '#' + params.join('&');
            // Use replaceState to avoid polluting browser history
            if (window.history && window.history.replaceState) {
                window.history.replaceState(null, '', hash);
            } else {
                window.location.hash = hash;
            }
        }, 300);
    }

    function restoreFromUrlHash() {
        var hash = window.location.hash;
        if (!hash || hash.length < 2) return;

        var params = {};
        hash.substring(1).split('&').forEach(function(pair) {
            var parts = pair.split('=');
            if (parts.length === 2) {
                params[parts[0]] = parts[1];
            }
        });

        // Restore smoothing
        if (params.smoothing !== undefined) {
            var smoothVal = parseFloat(params.smoothing);
            if (Number.isFinite(smoothVal) && smoothVal >= 0 && smoothVal <= 0.999) {
                state.smoothingWeight = smoothVal;
                smoothingSliderEl.value = smoothVal;
                smoothingValueEl.textContent = smoothVal.toFixed(3);
            }
        }

        // Restore x-axis mode
        if (params.xaxis && ['step', 'wall_time', 'relative'].indexOf(params.xaxis) !== -1) {
            state.xAxisMode = params.xaxis;
            xAxisModeEl.value = params.xaxis;
        }

        // Restore active tab
        if (params.tab) {
            var validTabs = ['scalars', 'histograms', 'hparams', 'traces', 'eval', 'artifacts', 'audio', 'pr-curves', 'projector', 'lora'];
            if (validTabs.indexOf(params.tab) !== -1) {
                // Defer tab switch until after runs/tags are loaded
                state._pendingTab = params.tab;
            }
        }

        // Restore selected runs (will be applied after runs are fetched)
        if (params.runs) {
            state._pendingRuns = params.runs.split(',').map(decodeURIComponent);
        }

        // Restore selected tags (will be applied after tags are fetched)
        if (params.tags) {
            state._pendingTags = params.tags.split(',').map(decodeURIComponent);
        }
    }

    function applyPendingHashState() {
        var needsTagLoad = false;

        // Apply pending runs
        if (state._pendingRuns && state._pendingRuns.length) {
            var runNames = state.runs.map(function(r) { return r.name; });
            state._pendingRuns.forEach(function(runName) {
                if (runNames.indexOf(runName) !== -1 && !state.selectedRuns.has(runName)) {
                    state.selectedRuns.add(runName);
                    needsTagLoad = true;
                }
            });
            delete state._pendingRuns;
        }

        if (needsTagLoad) {
            renderRunList();
            updateRunNotes();
            refreshSelectedRunTags().then(function() {
                rebuildTagList();

                // Apply pending tags
                if (state._pendingTags && state._pendingTags.length) {
                    state._pendingTags.forEach(function(tag) {
                        if (state.allTags.has(tag)) {
                            state.selectedTags.add(tag);
                            ensureChartExists(tag);
                            state.selectedRuns.forEach(function(runName) {
                                loadChartData(runName, tag);
                            });
                        }
                    });
                    delete state._pendingTags;
                    renderTagList();
                    updateLiveSubscription();
                }

                // Apply pending tab
                if (state._pendingTab) {
                    switchTab(state._pendingTab);
                    delete state._pendingTab;
                }
            });
        } else {
            // Apply pending tab even without runs
            if (state._pendingTab) {
                switchTab(state._pendingTab);
                delete state._pendingTab;
            }
        }
    }

    // ─── Utilities ────────────────────────────────────────────────────

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function _formatTimeAgo(wallTime) {
        var secs = Math.floor(Date.now() / 1000 - wallTime);
        if (secs < 60) return secs + 's ago';
        var mins = Math.floor(secs / 60);
        if (mins < 60) return mins + 'm ago';
        var hrs = Math.floor(mins / 60);
        if (hrs < 24) return hrs + 'h ago';
        return Math.floor(hrs / 24) + 'd ago';
    }

    // ─── Export helpers ──────────────────────────────────────────────────

    function exportAllVisibleScalars() {
        var runs = Array.from(state.selectedRuns);
        var tags = Array.from(state.selectedTags);

        if (!runs.length || !tags.length) {
            alert('Select at least one run and one tag to export.');
            return;
        }

        // For each selected run, trigger a CSV download via the server export endpoint
        runs.forEach(function(runName) {
            var url = '/api/runs/' + encodeURIComponent(runName) +
                      '/export?format=csv' +
                      '&tags=' + tags.map(encodeURIComponent).join(',') +
                      '&x_axis=' + encodeURIComponent(state.xAxisMode);
            var link = document.createElement('a');
            link.href = url;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    function exportSingleChartCSV(chartId) {
        var chart = getChartInfo(chartId);
        if (!chart || !chart.tag) return;

        var runs = Object.keys(chart.traces || {});
        if (!runs.length) return;

        var tag = chart.tag;
        runs.forEach(function(runName) {
            var url = '/api/runs/' + encodeURIComponent(runName) +
                      '/export?format=csv' +
                      '&tags=' + encodeURIComponent(tag) +
                      '&x_axis=' + encodeURIComponent(state.xAxisMode);
            var link = document.createElement('a');
            link.href = url;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // ─── Start ────────────────────────────────────────────────────────

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
