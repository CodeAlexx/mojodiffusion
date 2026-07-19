/**
 * LoRA Weight Inspector — SerenityBoard frontend module.
 *
 * Provides single-file analysis and two-file comparison of LoRA
 * safetensors weights. Renders per-layer metrics tables and ECharts
 * bar charts for spectral norms.
 */
(function () {
    'use strict';

    // DOM refs (resolved after DOMContentLoaded)
    var container, tableContainer, summaryEl, diagnosticsEl, chartsEl;
    var path1Input, path2Input, file1Input, file2Input;
    var analyzeBtn, exportBtn;
    var chartAEl, chartBEl;
    var chartA = null, chartB = null;
    var lastResult = null;

    function init() {
        container = document.getElementById('lora-container');
        tableContainer = document.getElementById('lora-table-container');
        summaryEl = document.getElementById('lora-summary');
        diagnosticsEl = document.getElementById('lora-diagnostics');
        chartsEl = document.getElementById('lora-charts');
        path1Input = document.getElementById('lora-path-1');
        path2Input = document.getElementById('lora-path-2');
        file1Input = document.getElementById('lora-file-1');
        file2Input = document.getElementById('lora-file-2');
        analyzeBtn = document.getElementById('lora-analyze-btn');
        exportBtn = document.getElementById('lora-export-csv-btn');
        chartAEl = document.getElementById('lora-chart-a-spectral');
        chartBEl = document.getElementById('lora-chart-b-spectral');

        if (!analyzeBtn) return;

        analyzeBtn.addEventListener('click', onAnalyze);
        exportBtn.addEventListener('click', onExportCSV);

        // File picker updates path input
        file1Input.addEventListener('change', function () {
            if (file1Input.files.length) path1Input.value = file1Input.files[0].name;
        });
        file2Input.addEventListener('change', function () {
            if (file2Input.files.length) path2Input.value = file2Input.files[0].name;
        });
    }

    // ── Analysis trigger ──────────────────────────────────────────

    function onAnalyze() {
        var hasFile1 = file1Input.files && file1Input.files.length > 0;
        var hasFile2 = file2Input.files && file2Input.files.length > 0;
        var hasPath1 = path1Input.value.trim();
        var hasPath2 = path2Input.value.trim();

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';

        var promise;
        if (hasFile1 && hasFile2) {
            promise = uploadCompare(file1Input.files[0], file2Input.files[0]);
        } else if (hasFile1 && !hasFile2) {
            promise = uploadAnalyze(file1Input.files[0]);
        } else if (hasPath1 && hasPath2) {
            promise = pathCompare(hasPath1, hasPath2);
        } else if (hasPath1) {
            promise = pathAnalyze(hasPath1);
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
            return;
        }

        promise
            .then(function (data) {
                lastResult = data;
                renderResult(data);
            })
            .catch(function (err) {
                tableContainer.innerHTML = '<div class="empty-state"><p>Error: ' +
                    escHtml(err.message || String(err)) + '</p></div>';
            })
            .finally(function () {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze';
            });
    }

    // ── API calls ─────────────────────────────────────────────────

    function pathAnalyze(path) {
        return fetch('/api/lora/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path }),
        }).then(checkResp);
    }

    function pathCompare(pathA, pathB) {
        return fetch('/api/lora/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path_a: pathA, path_b: pathB }),
        }).then(checkResp);
    }

    function uploadAnalyze(file) {
        var fd = new FormData();
        fd.append('file', file);
        return fetch('/api/lora/analyze-upload', { method: 'POST', body: fd }).then(checkResp);
    }

    function uploadCompare(fileA, fileB) {
        var fd = new FormData();
        fd.append('file_a', fileA);
        fd.append('file_b', fileB);
        return fetch('/api/lora/compare-upload', { method: 'POST', body: fd }).then(checkResp);
    }

    function checkResp(r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || d.error?.message || r.statusText); });
        return r.json();
    }

    // ── Render ─────────────────────────────────────────────────────

    function renderResult(data) {
        var isCompare = !!data.file_b || !!data.summary;
        var layers = data.layers || {};
        var names = Object.keys(layers);

        // Summary bar
        if (isCompare && data.summary) {
            var s = data.summary;
            summaryEl.style.display = 'block';
            summaryEl.className = 'lora-summary-bar';
            summaryEl.innerHTML =
                '<strong>Global Diff</strong> &mdash; ' +
                'A-Spectral: <span class="' + diffClass(s.mean_diff_a_spectral_pct) + '">' +
                fmtPct(s.mean_diff_a_spectral_pct) + '</span> | ' +
                'B-Spectral: <span class="' + diffClass(s.mean_diff_b_spectral_pct) + '">' +
                fmtPct(s.mean_diff_b_spectral_pct) + '</span>';
        } else {
            summaryEl.style.display = 'none';
        }

        // Diagnostics
        if (data.diagnostics && data.diagnostics.length) {
            diagnosticsEl.style.display = 'block';
            diagnosticsEl.innerHTML = '<div class="lora-diag-title">Diagnostics</div>' +
                data.diagnostics.map(function (w) {
                    return '<div class="lora-diag-item">' + escHtml(w) + '</div>';
                }).join('');
        } else {
            diagnosticsEl.style.display = 'none';
        }

        // Table
        if (!names.length) {
            tableContainer.innerHTML = '<div class="empty-state"><p>No LoRA layers found in file</p></div>';
            chartsEl.style.display = 'none';
            exportBtn.style.display = 'none';
            return;
        }

        var html = '<table class="lora-table"><thead><tr>';
        html += '<th>Layer</th>';
        if (isCompare) {
            html += '<th>L1: A-Spec</th><th>L1: B-Spec</th>';
            html += '<th>L2: A-Spec</th><th>L2: B-Spec</th>';
            html += '<th>Diff A%</th><th>Diff B%</th>';
        } else {
            html += '<th>A-L2</th><th>A-Spec</th><th>B-L2</th><th>B-Spec</th>';
            html += '<th>Eff Rank</th><th>A/B</th><th>Cond</th>';
        }
        html += '</tr></thead><tbody>';

        for (var i = 0; i < names.length; i++) {
            var name = names[i];
            var entry = layers[name];
            var short = name.length > 45 ? name.slice(-45) : name;
            html += '<tr>';
            html += '<td class="lora-layer-name" title="' + escHtml(name) + '">' + escHtml(short) + '</td>';

            if (isCompare) {
                var m1 = entry.lora1 || {};
                var m2 = entry.lora2 || {};
                var dA = entry.diff_a_spectral_norm_pct || 0;
                var dB = entry.diff_b_spectral_norm_pct || 0;
                html += '<td>' + fmtNum(m1.a_spectral_norm) + '</td>';
                html += '<td>' + fmtNum(m1.b_spectral_norm) + '</td>';
                html += '<td>' + fmtNum(m2.a_spectral_norm) + '</td>';
                html += '<td>' + fmtNum(m2.b_spectral_norm) + '</td>';
                html += '<td class="' + diffClass(dA) + '">' + fmtPct(dA) + '</td>';
                html += '<td class="' + diffClass(dB) + '">' + fmtPct(dB) + '</td>';
            } else {
                var m = entry;
                html += '<td>' + fmtNum(m.a_l2_norm) + '</td>';
                html += '<td>' + fmtNum(m.a_spectral_norm) + '</td>';
                html += '<td>' + fmtNum(m.b_l2_norm) + '</td>';
                html += '<td>' + fmtNum(m.b_spectral_norm) + '</td>';
                html += '<td>' + fmtNum(m.effective_rank) + '</td>';
                html += '<td>' + fmtNum(m.ab_ratio) + '</td>';
                html += '<td>' + fmtNum(m.condition_number) + '</td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        tableContainer.innerHTML = html;
        exportBtn.style.display = '';

        // Charts
        renderCharts(names, layers, isCompare);
    }

    function renderCharts(names, layers, isCompare) {
        if (typeof echarts === 'undefined') {
            chartsEl.style.display = 'none';
            return;
        }
        chartsEl.style.display = 'block';

        var shortNames = names.map(function (n) {
            var parts = n.split('.');
            return parts.length > 2 ? parts.slice(-2).join('.') : n;
        });

        // A-Spectral chart
        if (!chartA) chartA = echarts.init(chartAEl, 'serenityDark');
        var aOpt = {
            title: { text: 'A-Spectral Norm by Layer', left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: shortNames, axisLabel: { rotate: 45, fontSize: 9 } },
            yAxis: { type: 'value', name: 'Spectral Norm' },
            grid: { bottom: 100, left: 60, right: 20 },
            series: [],
        };
        if (isCompare) {
            aOpt.legend = { data: ['LoRA 1', 'LoRA 2'], top: 25 };
            aOpt.series.push({
                name: 'LoRA 1', type: 'bar',
                data: names.map(function (n) { return (layers[n].lora1 || {}).a_spectral_norm || 0; }),
            });
            aOpt.series.push({
                name: 'LoRA 2', type: 'bar',
                data: names.map(function (n) { return (layers[n].lora2 || {}).a_spectral_norm || 0; }),
            });
        } else {
            aOpt.series.push({
                type: 'bar',
                data: names.map(function (n) { return layers[n].a_spectral_norm || 0; }),
            });
        }
        chartA.setOption(aOpt, true);

        // B-Spectral chart
        if (!chartB) chartB = echarts.init(chartBEl, 'serenityDark');
        var bOpt = {
            title: { text: 'B-Spectral Norm by Layer', left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: shortNames, axisLabel: { rotate: 45, fontSize: 9 } },
            yAxis: { type: 'value', name: 'Spectral Norm' },
            grid: { bottom: 100, left: 60, right: 20 },
            series: [],
        };
        if (isCompare) {
            bOpt.legend = { data: ['LoRA 1', 'LoRA 2'], top: 25 };
            bOpt.series.push({
                name: 'LoRA 1', type: 'bar',
                data: names.map(function (n) { return (layers[n].lora1 || {}).b_spectral_norm || 0; }),
            });
            bOpt.series.push({
                name: 'LoRA 2', type: 'bar',
                data: names.map(function (n) { return (layers[n].lora2 || {}).b_spectral_norm || 0; }),
            });
        } else {
            bOpt.series.push({
                type: 'bar',
                data: names.map(function (n) { return layers[n].b_spectral_norm || 0; }),
            });
        }
        chartB.setOption(bOpt, true);

        // Handle resize
        setTimeout(function () { chartA.resize(); chartB.resize(); }, 100);
    }

    // ── Export CSV ─────────────────────────────────────────────────

    function onExportCSV() {
        if (!lastResult || !lastResult.layers) return;
        var layers = lastResult.layers;
        var names = Object.keys(layers);
        var isCompare = !!lastResult.file_b;

        var rows = [];
        if (isCompare) {
            rows.push(['Layer', 'L1_A_Spectral', 'L1_B_Spectral', 'L2_A_Spectral', 'L2_B_Spectral', 'Diff_A_Pct', 'Diff_B_Pct']);
            for (var i = 0; i < names.length; i++) {
                var e = layers[names[i]];
                var m1 = e.lora1 || {}, m2 = e.lora2 || {};
                rows.push([names[i], m1.a_spectral_norm, m1.b_spectral_norm,
                    m2.a_spectral_norm, m2.b_spectral_norm,
                    e.diff_a_spectral_norm_pct, e.diff_b_spectral_norm_pct]);
            }
        } else {
            rows.push(['Layer', 'A_L2', 'A_Spectral', 'B_L2', 'B_Spectral', 'Eff_Rank', 'AB_Ratio', 'Condition']);
            for (var i = 0; i < names.length; i++) {
                var m = layers[names[i]];
                rows.push([names[i], m.a_l2_norm, m.a_spectral_norm, m.b_l2_norm,
                    m.b_spectral_norm, m.effective_rank, m.ab_ratio, m.condition_number]);
            }
        }

        var csv = rows.map(function (r) { return r.join(','); }).join('\n');
        var blob = new Blob([csv], { type: 'text/csv' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'lora_analysis.csv';
        a.click();
    }

    // ── Helpers ────────────────────────────────────────────────────

    function fmtNum(v) {
        if (v == null || v === undefined) return '-';
        if (!isFinite(v)) return 'Inf';
        return v < 100 ? v.toFixed(4) : v.toFixed(1);
    }

    function fmtPct(v) {
        if (v == null) return '-';
        if (!isFinite(v)) return 'Inf';
        return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
    }

    function diffClass(v) {
        if (v == null || !isFinite(v)) return '';
        return v > 5 ? 'diff-up' : v < -5 ? 'diff-down' : '';
    }

    function escHtml(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Resize handler for LoRA tab ───────────────────────────────

    window.resizeLoraCharts = function () {
        if (chartA) chartA.resize();
        if (chartB) chartB.resize();
    };

    // ── Init on DOM ready ─────────────────────────────────────────

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
