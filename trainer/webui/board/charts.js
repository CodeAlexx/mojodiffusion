/**
 * charts.js -- ECharts wrapper + EMA smoothing for SerenityBoard.
 *
 * Manages chart creation, update, and live-append using ECharts
 * with large dataset optimizations.
 */

/* global echarts, getOrCreateChart, disposeChart, getChartInstance */

// ─── Run color generation ─────────────────────────────────────────────
// Deterministic color from run name hash. Produces consistent, visually
// distinct colors across runs.

var RUN_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
];

function hashString(s) {
    var hash = 0;
    for (var i = 0; i < s.length; i++) {
        hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
}

var _customRunColors = {}; // runName -> color string

function getRunColor(runName) {
    if (_customRunColors[runName]) return _customRunColors[runName];
    return RUN_COLORS[hashString(runName) % RUN_COLORS.length];
}

function setCustomRunColor(runName, color) {
    _customRunColors[runName] = color;
}

function getCustomRunColors() {
    return _customRunColors;
}

// ─── EMA smoothing with debiasing correction ──────────────────────────
// Matches TensorBoard's EMA smoothing behavior.

function smoothEMA(points, weight) {
    if (!points.length || weight <= 0) return points;
    weight = Math.min(weight, 1);

    var last = 0;
    var numAccum = 0;
    var isConstant = points.every(function(p) { return p.y === points[0].y; });
    if (isConstant) return points;

    return points.map(function(p) {
        if (!Number.isFinite(p.y)) return { x: p.x, y: p.y };
        last = last * weight + (1 - weight) * p.y;
        numAccum++;
        var debias = weight === 1 ? 1 : 1 - Math.pow(weight, numAccum);
        return { x: p.x, y: last / debias };
    });
}

// ─── Chart registry ───────────────────────────────────────────────────
// Tracks which runs/data are displayed per chart container.

var _charts = {};  // containerId -> { tag, traces: { runName: { rawPoints, traceIndex } } }
var _globalScalarStepMax = null; // shared scalar step ceiling set by app.js
var _lastKnownGlobalMax = -1;  // cached global max for change detection

// ─── X-axis value extraction ──────────────────────────────────────────

function extractXValues(rawData, xAxisMode, runStartTime) {
    return rawData.map(function(row) {
        var step = row[0], wallTime = row[1];
        if (xAxisMode === 'wall_time') {
            return new Date(wallTime * 1000);
        } else if (xAxisMode === 'relative') {
            return wallTime - (runStartTime || wallTime);
        }
        return step;
    });
}

function getXAxisTitle(xAxisMode) {
    if (xAxisMode === 'wall_time') return 'Time';
    if (xAxisMode === 'relative') return 'Seconds';
    return 'Step';
}

function getXAxisType(xAxisMode) {
    if (xAxisMode === 'wall_time') return 'time';
    return 'value';
}

// ─── Raw data to {x, y} points ───────────────────────────────────────

function rawToPoints(rawData, xAxisMode, runStartTime) {
    var xs = extractXValues(rawData, xAxisMode, runStartTime);
    return rawData.map(function(row, i) {
        return { x: xs[i], y: row[2] };
    });
}

// ─── Nice tick formatting ────────────────────────────────────────────

function _niceTickInterval(range, targetTicks) {
    if (range <= 0) return 1;
    var rough = range / targetTicks;
    var mag = Math.pow(10, Math.floor(Math.log10(rough)));
    var residual = rough / mag;
    var nice;
    if (residual <= 1.5) nice = 1;
    else if (residual <= 3) nice = 2;
    else if (residual <= 7) nice = 5;
    else nice = 10;
    return nice * mag;
}

function _stepAxisFormatter(value) {
    if (value >= 1000) return (value / 1000) + 'k';
    return '' + value;
}

// ─── Public API ───────────────────────────────────────────────────────

/**
 * Create a new ECharts chart in the given container.
 * @param {string} containerId - DOM id of the chart container div.
 * @param {string} tag - The scalar tag this chart displays.
 */
function createChart(containerId, tag) {
    var chart = getOrCreateChart(containerId);
    if (!chart) return;

    // TensorBoard-matched chart initialization
    chart.setOption({
        title: { text: tag },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'line', lineStyle: { color: 'rgba(255,255,255,0.3)', width: 1 } },
            formatter: function(params) {
                if (!params || !params.length) return '';
                var lines = [];
                for (var i = 0; i < params.length; i++) {
                    var p = params[i];
                    if (p.seriesName && p.seriesName.indexOf('(raw)') !== -1) continue;
                    var val = typeof p.value === 'number' ? p.value :
                              (Array.isArray(p.value) ? p.value[1] : p.value);
                    if (val != null && Number.isFinite(val)) {
                        lines.push(
                            '<span style="display:inline-block;width:10px;height:10px;' +
                            'border-radius:50%;background:' + p.color + ';margin-right:5px;"></span>' +
                            p.seriesName + ': <b>' + val.toPrecision(4) + '</b>'
                        );
                    }
                }
                var xLabel = params[0].axisValueLabel || params[0].name || '';
                return xLabel + '<br>' + lines.join('<br>');
            },
        },
        xAxis: {
            type: 'value',
            axisLabel: { formatter: _stepAxisFormatter, fontSize: 11, color: '#999' },
            axisLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            axisTick: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        },
        yAxis: {
            type: 'value',
            scale: true,
            axisLabel: { fontSize: 11, color: '#999' },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        },
        toolbox: {
            show: false,
        },
        dataZoom: [
            { type: 'inside', xAxisIndex: 0 },
            { type: 'inside', yAxisIndex: 0 },
        ],
        series: [],
        animation: false,
    });

    _charts[containerId] = {
        tag: tag,
        traces: {},
        xAxisMode: 'step',
        lastSmoothingWeight: 0.6,
    };
}

// ─── Global max step computation (single owner of X-axis range) ──────

function _computeGlobalMaxStep() {
    var globalMax = -1;
    var chartIds = Object.keys(_charts);
    for (var i = 0; i < chartIds.length; i++) {
        var chart = _charts[chartIds[i]];
        if (!chart || !chart.traces) continue;
        var runs = Object.keys(chart.traces);
        for (var j = 0; j < runs.length; j++) {
            var entry = chart.traces[runs[j]];
            if (!entry || !entry.rawPoints || !entry.rawPoints.length) continue;
            var lastRow = entry.rawPoints[entry.rawPoints.length - 1];
            if (lastRow && Number.isFinite(lastRow[0])) {
                globalMax = Math.max(globalMax, lastRow[0]);
            }
        }
    }
    if (Number.isFinite(_globalScalarStepMax) && _globalScalarStepMax > globalMax) {
        globalMax = _globalScalarStepMax;
    }
    return globalMax;
}

function _syncAllChartsIfNeeded(excludeId) {
    var newMax = _computeGlobalMaxStep();
    if (newMax > 0 && newMax !== _lastKnownGlobalMax) {
        _lastKnownGlobalMax = newMax;
        var chartIds = Object.keys(_charts);
        for (var i = 0; i < chartIds.length; i++) {
            if (chartIds[i] === excludeId) continue;
            var chart = _charts[chartIds[i]];
            if (chart) {
                _renderScalarChart(chartIds[i], chart.lastSmoothingWeight, chart.xAxisMode);
            }
        }
    }
}

function setGlobalScalarStepMax(maxStep) {
    if (typeof maxStep === 'number' && Number.isFinite(maxStep) && maxStep >= 0) {
        _globalScalarStepMax = maxStep;
    } else {
        _globalScalarStepMax = null;
    }
}

function resetGlobalSyncState() {
    _lastKnownGlobalMax = -1;
}

function _mergeRowsByStep(existingRows, incomingRows) {
    var merged = {};
    var i;
    for (i = 0; i < existingRows.length; i++) {
        merged[existingRows[i][0]] = existingRows[i];
    }
    for (i = 0; i < incomingRows.length; i++) {
        merged[incomingRows[i][0]] = incomingRows[i];
    }
    return Object.keys(merged)
        .map(function(step) { return merged[step]; })
        .sort(function(a, b) { return a[0] - b[0]; });
}

function _renderScalarChart(containerId, smoothingWeight, xAxisMode) {
    var chart = _charts[containerId];
    if (!chart) return;

    var el = document.getElementById(containerId);
    if (!el) return;

    // Skip rendering when element is in a hidden ancestor (e.g. tab not active)
    if (el.offsetParent === null) return;

    var instance = getChartInstance(containerId);
    if (!instance) {
        instance = getOrCreateChart(containerId);
        if (!instance) return;
    }

    var series = [];
    var runNames = Object.keys(chart.traces).sort();
    var maxStep = _computeGlobalMaxStep();
    var yMin = Infinity, yMax = -Infinity;

    runNames.forEach(function(run) {
        var entry = chart.traces[run];
        if (!entry || !entry.rawPoints || !entry.rawPoints.length) return;

        var points = rawToPoints(entry.rawPoints, xAxisMode, entry.runStartTime);
        var color = getRunColor(run);
        var useLarge = points.length > 5000;

        // Track Y bounds for explicit axis range
        for (var pi = 0; pi < points.length; pi++) {
            var v = points[pi].y;
            if (Number.isFinite(v)) {
                if (v < yMin) yMin = v;
                if (v > yMax) yMax = v;
            }
        }

        if (smoothingWeight <= 0) {
            // No smoothing — single bold line
            series.push({
                name: run,
                type: 'line',
                large: useLarge,
                largeThreshold: 5000,
                sampling: useLarge ? 'lttb' : undefined,
                showSymbol: false,
                lineStyle: { width: 2, color: color },
                itemStyle: { color: color },
                data: points.map(function(p) { return [p.x, p.y]; }),
            });
        } else {
            var smoothed = smoothEMA(points, smoothingWeight);
            // Raw trace — visible thin line (TensorBoard style)
            series.push({
                name: run + ' (raw)',
                type: 'line',
                large: useLarge,
                largeThreshold: 5000,
                sampling: useLarge ? 'lttb' : undefined,
                showSymbol: false,
                lineStyle: { width: 1, color: color, opacity: 0.4 },
                itemStyle: { color: color, opacity: 0 },
                data: points.map(function(p) { return [p.x, p.y]; }),
                silent: true,
                z: 1,
            });
            // Smoothed EMA curve — thick prominent line
            series.push({
                name: run,
                type: 'line',
                large: useLarge,
                largeThreshold: 5000,
                sampling: useLarge ? 'lttb' : undefined,
                showSymbol: false,
                lineStyle: { width: 2.5, color: color },
                itemStyle: { color: color },
                data: smoothed.map(function(p) { return [p.x, p.y]; }),
                z: 2,
            });
        }
    });

    // TensorBoard-matched axis config with visible grid
    var xAxisOpt = {
        type: getXAxisType(xAxisMode),
        axisLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
        axisTick: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
        splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        axisLabel: { fontSize: 11, color: '#999' },
    };

    if (xAxisMode === 'step' && Number.isFinite(maxStep) && maxStep > 0) {
        xAxisOpt.min = 0;
        xAxisOpt.max = maxStep;
        xAxisOpt.axisLabel.formatter = _stepAxisFormatter;
    } else {
        xAxisOpt.min = null;
        xAxisOpt.max = null;
        if (xAxisMode === 'step') {
            xAxisOpt.axisLabel.formatter = _stepAxisFormatter;
        }
    }

    // Build legend data — only show non-raw entries
    var legendData = [];
    series.forEach(function(s) {
        if (s.name.indexOf('(raw)') === -1) {
            legendData.push(s.name);
        }
    });

    // Compute explicit Y-axis bounds to avoid flatline rendering
    var yAxisOpt = {
        type: 'value', scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        axisLabel: { fontSize: 11, color: '#999' },
    };
    if (Number.isFinite(yMin) && Number.isFinite(yMax)) {
        var yRange = yMax - yMin;
        if (yRange > 0) {
            var pad = yRange * 0.05;
            yAxisOpt.min = yMin - pad;
            yAxisOpt.max = yMax + pad;
        } else {
            var cpad = Math.abs(yMin) * 0.1 || 0.001;
            yAxisOpt.min = yMin - cpad;
            yAxisOpt.max = yMax + cpad;
        }
    }

    instance.setOption({
        title: { text: chart.tag },
        legend: {
            data: legendData,
            type: 'scroll',
            bottom: 0,
            textStyle: { fontSize: 11 },
        },
        xAxis: xAxisOpt,
        yAxis: yAxisOpt,
        series: series,
    }, { replaceMerge: ['series'] });

    chart.xAxisMode = xAxisMode;
    chart.lastSmoothingWeight = smoothingWeight;
}

/**
 * Update or add a trace for a run on an existing chart.
 */
function updateChart(containerId, run, rawData, smoothingWeight, xAxisMode, runStartTime) {
    var chart = _charts[containerId];
    if (!chart) return;

    chart.traces[run] = chart.traces[run] || { rawPoints: [] };
    var existing = chart.traces[run].rawPoints || [];
    chart.traces[run].rawPoints = _mergeRowsByStep(existing, rawData || []);
    chart.traces[run].runStartTime = runStartTime;

    _renderScalarChart(containerId, smoothingWeight, xAxisMode);
    _syncAllChartsIfNeeded(containerId);
}

/**
 * Append new points to a chart for live updates.
 */
function appendPoints(containerId, run, newPoints, smoothingWeight, xAxisMode, runStartTime) {
    var chart = _charts[containerId];
    if (!chart || !chart.traces[run]) return;

    var newRaw = newPoints.map(function(p) {
        return [p.step, p.wall_time, p.value];
    });
    if (!newRaw.length) return;

    var existing = chart.traces[run].rawPoints || [];
    chart.traces[run].rawPoints = _mergeRowsByStep(existing, newRaw);
    chart.traces[run].runStartTime = runStartTime;
    _renderScalarChart(containerId, smoothingWeight, xAxisMode);
    _syncAllChartsIfNeeded(containerId);
}

/**
 * Remove a run's traces from a chart.
 */
function removeRunFromChart(containerId, run) {
    var chart = _charts[containerId];
    if (!chart || !chart.traces[run]) return;
    delete chart.traces[run];
    _renderScalarChart(
        containerId,
        chart.lastSmoothingWeight || 0.6,
        chart.xAxisMode || 'step'
    );
    _syncAllChartsIfNeeded(containerId);
}

/**
 * Destroy a chart and remove it from the registry.
 */
function destroyChart(containerId) {
    disposeChart(containerId);
    delete _charts[containerId];
}

/**
 * Re-smooth all traces on a chart with a new weight.
 */
function reSmoothChart(containerId, smoothingWeight, xAxisMode, runStartTimes) {
    var chart = _charts[containerId];
    if (!chart) return;

    for (var run in chart.traces) {
        if (chart.traces.hasOwnProperty(run)) {
            chart.traces[run].runStartTime = runStartTimes ? runStartTimes[run] : undefined;
        }
    }
    _renderScalarChart(containerId, smoothingWeight, xAxisMode);
}

/**
 * Get all registered chart container IDs.
 * @returns {string[]}
 */
function getChartIds() {
    return Object.keys(_charts);
}

/**
 * Get chart info (tag and run names).
 * @param {string} containerId
 * @returns {Object|null}
 */
function getChartInfo(containerId) {
    return _charts[containerId] || null;
}

/**
 * Export chart data as CSV and trigger browser download.
 */
function exportChartCSV(containerId) {
    var chart = _charts[containerId];
    if (!chart) return;

    var rows = ['step,wall_time,value,run'];
    var runNames = Object.keys(chart.traces).sort();

    runNames.forEach(function(run) {
        var entry = chart.traces[run];
        if (!entry || !entry.rawPoints) return;
        entry.rawPoints.forEach(function(row) {
            rows.push(row[0] + ',' + row[1] + ',' + row[2] + ',' + run);
        });
    });

    var csvContent = rows.join('\n');
    var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', chart.tag.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.csv');
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// ─── Histogram charts ─────────────────────────────────────────────────

var _histogramCharts = {};  // containerId -> { tag, runs: Set, mode, runData, distributionData }

/**
 * Create a new ECharts heatmap chart for histogram data.
 */
function createHistogramChart(containerId, tag) {
    var chart = getOrCreateChart(containerId);
    if (!chart) return;

    chart.setOption({
        title: { text: tag },
        tooltip: {},
        xAxis: { type: 'value', name: 'Step' },
        yAxis: { type: 'value', name: 'Value' },
        series: [],
        animation: false,
    });

    _histogramCharts[containerId] = {
        tag: tag,
        runs: new Set(),
        mode: 'heatmap',
        runData: {},
        distributionData: {},
    };
}

/**
 * Update a histogram chart with data from a run.
 */
function updateHistogramChart(containerId, run, histogramData) {
    var chart = _histogramCharts[containerId];
    if (!chart) return;

    if (!histogramData || !histogramData.length) return;

    chart.runData[run] = histogramData;
    chart.runs.add(run);

    _renderHistogramChart(containerId);
}

function _renderHistogramChart(containerId) {
    var chart = _histogramCharts[containerId];
    if (!chart) return;

    var el = document.getElementById(containerId);
    if (!el) return;

    var run = null;
    var histogramData = null;
    chart.runs.forEach(function(r) {
        if (!run) {
            run = r;
            histogramData = chart.runData[r];
        }
    });

    if (!run || !histogramData || !histogramData.length) return;

    if (chart.mode === 'distribution') {
        _renderDistributionChart(containerId, chart, run);
    } else if (chart.mode === 'ridgeline') {
        _renderRidgelineHistogram(containerId, chart, run, histogramData);
    } else {
        _renderHeatmapHistogram(containerId, chart, run, histogramData);
    }
}

function _renderHeatmapHistogram(containerId, chart, run, histogramData) {
    var instance = getOrCreateChart(containerId);
    if (!instance) return;

    var globalCenters = null;
    for (var i = 0; i < histogramData.length; i++) {
        var bins = histogramData[i].bins;
        if (bins && bins.length > 0) {
            var centers = bins.map(function(b) { return (b[0] + b[1]) / 2; });
            if (!globalCenters || centers.length > globalCenters.length) {
                globalCenters = centers;
            }
        }
    }

    if (!globalCenters || globalCenters.length === 0) return;

    var numBins = globalCenters.length;
    var data = [];  // [stepIndex, binIndex, value]

    for (var s = 0; s < histogramData.length; s++) {
        var bins = histogramData[s].bins;
        if (bins && bins.length > 0) {
            var stepCenters = bins.map(function(bin) { return (bin[0] + bin[1]) / 2; });
            var stepCounts = bins.map(function(bin) { return bin[2]; });

            for (var b = 0; b < numBins; b++) {
                var targetCenter = globalCenters[b];
                var closestIdx = 0;
                var closestDist = Math.abs(stepCenters[0] - targetCenter);
                for (var k = 1; k < stepCenters.length; k++) {
                    var dist = Math.abs(stepCenters[k] - targetCenter);
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestIdx = k;
                    }
                }
                var val = Math.log10(stepCounts[closestIdx] + 1);
                if (val > 0) {
                    data.push([histogramData[s].step, globalCenters[b], val]);
                }
            }
        }
    }

    instance.setOption({
        title: { text: chart.tag + ' \u2014 ' + run },
        tooltip: {
            formatter: function(params) {
                if (!params.value) return '';
                return 'Step: ' + params.value[0] +
                    '<br>Value: ' + params.value[1].toPrecision(4) +
                    '<br>log10(count+1): ' + params.value[2].toFixed(2);
            },
        },
        grid: { left: 70, right: 20, top: 56, bottom: 50 },
        xAxis: { type: 'value', name: 'Step', axisLabel: { formatter: _stepAxisFormatter } },
        yAxis: { type: 'value', name: 'Value' },
        visualMap: {
            min: 0,
            max: Math.max.apply(null, data.map(function(d) { return d[2]; })) || 1,
            calculable: true,
            orient: 'vertical',
            right: 0,
            top: 'center',
            dimension: 2,
            inRange: {
                color: ['#16213e', '#1a3a5c', '#2a6090', '#4fc3f7', '#81d4fa', '#e1f5fe'],
            },
            textStyle: { color: '#a0a0b0', fontSize: 11 },
            text: ['High', 'Low'],
        },
        series: [{
            type: 'heatmap',
            data: data,
            emphasis: {
                itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' },
            },
        }],
    }, { replaceMerge: ['series', 'visualMap'] });
}

function _renderRidgelineHistogram(containerId, chart, run, histogramData) {
    var instance = getOrCreateChart(containerId);
    if (!instance) return;

    var maxTraces = 30;
    var data = histogramData;
    if (data.length > maxTraces) {
        var sampled = [];
        for (var i = 0; i < maxTraces; i++) {
            var idx = Math.round(i * (data.length - 1) / (maxTraces - 1));
            sampled.push(data[idx]);
        }
        data = sampled;
    }

    var color = getRunColor(run);

    // Find global max count for normalization
    var globalMaxCount = 0;
    data.forEach(function(entry) {
        if (entry.bins) {
            entry.bins.forEach(function(bin) {
                if (bin[2] > globalMaxCount) globalMaxCount = bin[2];
            });
        }
    });
    if (globalMaxCount === 0) globalMaxCount = 1;

    // Parse hex to RGB
    var r = parseInt(color.slice(1, 3), 16);
    var g = parseInt(color.slice(3, 5), 16);
    var b = parseInt(color.slice(5, 7), 16);

    var series = [];
    data.forEach(function(entry, i) {
        if (!entry.bins || !entry.bins.length) return;

        var centers = entry.bins.map(function(bin) { return (bin[0] + bin[1]) / 2; });
        var counts = entry.bins.map(function(bin) { return bin[2] / globalMaxCount; });
        var opacity = 0.15 + 0.6 * (i / (data.length - 1 || 1));

        series.push({
            name: 'Step ' + entry.step,
            type: 'line',
            showSymbol: false,
            lineStyle: { width: 1, color: color },
            areaStyle: {
                color: 'rgba(' + r + ',' + g + ',' + b + ',' + opacity + ')',
            },
            itemStyle: { color: color },
            data: centers.map(function(c, idx) { return [c, counts[idx]]; }),
            z: i,
        });
    });

    instance.setOption({
        title: { text: chart.tag + ' \u2014 ' + run + ' (Ridgeline)' },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                if (!params || !params.length) return '';
                var p = params[0];
                return 'Value: ' + (p.value ? p.value[0].toPrecision(4) : '') +
                    '<br>Density: ' + (p.value ? p.value[1].toFixed(3) : '');
            },
        },
        grid: { left: 70, right: 20, top: 56, bottom: 50 },
        xAxis: { type: 'value', name: 'Value' },
        yAxis: { type: 'value', name: 'Density (normalized)' },
        legend: { show: false },
        series: series,
    }, { replaceMerge: ['series', 'visualMap'] });
}

function _renderDistributionChart(containerId, chart, run) {
    var instance = getOrCreateChart(containerId);
    if (!instance) return;

    var distData = chart.distributionData[run];
    if (!distData || !distData.length) {
        instance.setOption({
            title: { text: chart.tag + ' \u2014 ' + run + ' (Distribution)' },
            series: [],
        }, { replaceMerge: ['series', 'visualMap'] });
        return;
    }

    var color = getRunColor(run);
    var r = parseInt(color.slice(1, 3), 16);
    var g = parseInt(color.slice(3, 5), 16);
    var b = parseInt(color.slice(5, 7), 16);

    function rgba(alpha) {
        return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    var steps = distData.map(function(d) { return d.step; });

    function bpValues(bpIndex) {
        return distData.map(function(d) {
            return d.percentiles[bpIndex] ? d.percentiles[bpIndex].value : null;
        });
    }

    var bp0 = bpValues(0);   // min
    var bp1 = bpValues(1);   // p6.68
    var bp2 = bpValues(2);   // p15.87
    var bp3 = bpValues(3);   // p30.85
    var bp4 = bpValues(4);   // median
    var bp5 = bpValues(5);   // p69.15
    var bp6 = bpValues(6);   // p84.13
    var bp7 = bpValues(7);   // p93.32
    var bp8 = bpValues(8);   // max

    // ECharts band approach: lower line invisible, upper fills "tozeroy" style
    // We use stacked area pairs to create bands.
    var series = [];

    // Helper: create a band between lower and upper
    function makeBand(lower, upper, fillColor, name, zLevel) {
        // Lower boundary (invisible line, serves as base)
        series.push({
            name: name + '_lower',
            type: 'line',
            data: steps.map(function(s, i) { return [s, lower[i]]; }),
            lineStyle: { opacity: 0 },
            showSymbol: false,
            stack: name,
            silent: true,
            z: zLevel,
        });
        // Upper boundary: the band height is (upper - lower)
        series.push({
            name: name,
            type: 'line',
            data: steps.map(function(s, i) {
                return [s, upper[i] != null && lower[i] != null ? upper[i] - lower[i] : null];
            }),
            lineStyle: { opacity: 0 },
            showSymbol: false,
            stack: name,
            areaStyle: { color: fillColor },
            silent: true,
            z: zLevel,
        });
    }

    makeBand(bp0, bp8, rgba(0.08), 'min\u2013max', 1);
    makeBand(bp1, bp7, rgba(0.14), '\u00b11.5\u03c3', 2);
    makeBand(bp2, bp6, rgba(0.22), '\u00b11\u03c3', 3);
    makeBand(bp3, bp5, rgba(0.35), '\u00b10.5\u03c3', 4);

    // Median line
    series.push({
        name: run + ' median',
        type: 'line',
        data: steps.map(function(s, i) { return [s, bp4[i]]; }),
        lineStyle: { width: 2, color: color },
        itemStyle: { color: color },
        showSymbol: false,
        z: 5,
    });

    instance.setOption({
        title: { text: chart.tag + ' \u2014 ' + run + ' (Distribution)' },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                if (!params || !params.length) return '';
                var medianParam = params.find(function(p) { return p.seriesName.indexOf('median') !== -1; });
                if (medianParam && medianParam.value) {
                    return 'Step ' + medianParam.value[0] + '<br>Median: ' + medianParam.value[1].toPrecision(4);
                }
                return '';
            },
        },
        grid: { left: 70, right: 20, top: 56, bottom: 50 },
        xAxis: { type: 'value', name: 'Step', axisLabel: { formatter: _stepAxisFormatter } },
        yAxis: { type: 'value', name: 'Value', scale: true },
        legend: { show: false },
        series: series,
    }, { replaceMerge: ['series', 'visualMap'] });
}

/**
 * Store distribution data for a run and re-render if in distribution mode.
 */
function updateDistributionChart(containerId, run, distributionData) {
    var chart = _histogramCharts[containerId];
    if (!chart) return;

    chart.distributionData[run] = distributionData;

    if (chart.mode === 'distribution') {
        _renderDistributionChart(containerId, chart, run);
    }
}

/**
 * Set histogram display mode and re-render.
 */
function setHistogramMode(containerId, mode) {
    var chart = _histogramCharts[containerId];
    if (!chart) return;
    chart.mode = mode;
    _renderHistogramChart(containerId);
}

/**
 * Get current histogram mode.
 */
function getHistogramMode(containerId) {
    var chart = _histogramCharts[containerId];
    return chart ? chart.mode : 'heatmap';
}

/**
 * Destroy a histogram chart and remove from registry.
 */
function destroyHistogramChart(containerId) {
    disposeChart(containerId);
    delete _histogramCharts[containerId];
}

/**
 * Get all registered histogram chart container IDs.
 * @returns {string[]}
 */
function getHistogramChartIds() {
    return Object.keys(_histogramCharts);
}

// ─── HParams parallel coordinates chart ──────────────────────────────

var _hparamsChart = null;

/**
 * Create or update a parallel coordinates chart for comparing hyperparameters.
 */
function createHParamsChart(containerId, data) {
    var el = document.getElementById(containerId);
    if (!el) return;

    if (!data || !data.length) {
        disposeChart(containerId);
        _hparamsChart = null;
        return;
    }

    var instance = getOrCreateChart(containerId);
    if (!instance) return;

    // Collect all numeric hparam keys and metric keys across all runs
    var hparamKeys = {};
    var metricKeys = {};

    data.forEach(function(entry) {
        if (entry.hparams) {
            Object.keys(entry.hparams).forEach(function(k) {
                var v = entry.hparams[k];
                if (typeof v === 'number' && Number.isFinite(v)) {
                    hparamKeys[k] = true;
                }
            });
        }
        if (entry.metrics) {
            Object.keys(entry.metrics).forEach(function(k) {
                var v = entry.metrics[k];
                if (typeof v === 'number' && Number.isFinite(v)) {
                    metricKeys[k] = true;
                }
            });
        }
    });

    var allHparamKeys = Object.keys(hparamKeys).sort();
    var allMetricKeys = Object.keys(metricKeys).sort();

    if (allHparamKeys.length === 0 && allMetricKeys.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No numeric hyperparameters or metrics found</p></div>';
        return;
    }

    // Pick the first metric as color dimension
    var colorMetric = allMetricKeys.length > 0 ? allMetricKeys[0] : null;

    // Build parallelAxis array
    var parallelAxis = [];
    var dimIndex = 0;

    allHparamKeys.forEach(function(key) {
        var values = data.map(function(entry) {
            if (entry.hparams && typeof entry.hparams[key] === 'number') {
                return entry.hparams[key];
            }
            return 0;
        });
        var hasValues = values.some(function(v) { return v !== 0; });
        if (!hasValues && data.every(function(e) { return !e.hparams || e.hparams[key] === undefined; })) return;

        parallelAxis.push({
            dim: dimIndex,
            name: key,
            min: Math.min.apply(null, values),
            max: Math.max.apply(null, values),
        });
        dimIndex++;
    });

    allMetricKeys.forEach(function(key) {
        var values = data.map(function(entry) {
            if (entry.metrics && typeof entry.metrics[key] === 'number') {
                return entry.metrics[key];
            }
            return 0;
        });

        parallelAxis.push({
            dim: dimIndex,
            name: key + ' *',
            min: Math.min.apply(null, values),
            max: Math.max.apply(null, values),
        });
        dimIndex++;
    });

    if (parallelAxis.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No numeric hyperparameters or metrics found</p></div>';
        return;
    }

    // Build data rows: each row is [val0, val1, ..., valN]
    var seriesData = data.map(function(entry) {
        var row = [];
        allHparamKeys.forEach(function(key) {
            if (entry.hparams && typeof entry.hparams[key] === 'number') {
                row.push(entry.hparams[key]);
            } else {
                row.push(0);
            }
        });
        allMetricKeys.forEach(function(key) {
            if (entry.metrics && typeof entry.metrics[key] === 'number') {
                row.push(entry.metrics[key]);
            } else {
                row.push(0);
            }
        });
        return row;
    });

    // Color values from first metric
    var colorValues = data.map(function(entry, idx) {
        if (colorMetric && entry.metrics && typeof entry.metrics[colorMetric] === 'number') {
            return entry.metrics[colorMetric];
        }
        return idx;
    });
    var minColor = Math.min.apply(null, colorValues);
    var maxColor = Math.max.apply(null, colorValues);

    instance.setOption({
        title: { text: 'Hyperparameter Comparison (' + data.length + ' runs)' },
        parallelAxis: parallelAxis,
        parallel: {
            left: 80,
            right: 80,
            top: 60,
            bottom: 40,
            parallelAxisDefault: {
                nameTextStyle: { color: '#e0e0e0', fontSize: 12 },
                axisLine: { lineStyle: { color: '#3a3a5a' } },
                axisTick: { lineStyle: { color: '#3a3a5a' } },
                axisLabel: { color: '#a0a0b0', fontSize: 10 },
            },
        },
        visualMap: {
            min: minColor,
            max: maxColor,
            dimension: parallelAxis.length > 0 ? allHparamKeys.length : 0,  // color by first metric dim
            inRange: {
                color: ['#2980b9', '#4fc3f7', '#4caf50', '#ffc107', '#f44336'],
            },
            text: [colorMetric || 'High', ''],
            textStyle: { color: '#a0a0b0' },
            right: 0,
        },
        series: {
            type: 'parallel',
            lineStyle: { width: 2, opacity: 0.7 },
            data: seriesData,
            smooth: false,
        },
    }, { replaceMerge: ['series', 'parallelAxis', 'visualMap'] });

    _hparamsChart = containerId;
}

/**
 * Destroy the hparams chart.
 */
function destroyHParamsChart(containerId) {
    disposeChart(containerId);
    if (_hparamsChart === containerId) {
        _hparamsChart = null;
    }
}

// ─── Trace timeline chart ────────────────────────────────────────────

/**
 * Create a horizontal bar chart for trace events.
 */
function createTraceTimeline(containerId, runData) {
    var instance = getOrCreateChart(containerId);
    if (!instance) return;

    var series = [];
    var allYLabels = [];

    runData.forEach(function(rd) {
        if (!rd.events || !rd.events.length) return;

        var color = getRunColor(rd.run);

        // Group by phase
        var phaseMap = {};
        rd.events.forEach(function(evt) {
            if (!phaseMap[evt.phase]) phaseMap[evt.phase] = [];
            phaseMap[evt.phase].push(evt);
            var label = 'Step ' + evt.step;
            if (allYLabels.indexOf(label) === -1) allYLabels.push(label);
        });

        Object.keys(phaseMap).sort().forEach(function(phase) {
            var events = phaseMap[phase];
            var phaseData = allYLabels.map(function() { return 0; });

            events.forEach(function(e) {
                var label = 'Step ' + e.step;
                var idx = allYLabels.indexOf(label);
                if (idx !== -1) phaseData[idx] = e.duration_ms;
            });

            series.push({
                name: rd.run + ' / ' + phase,
                type: 'bar',
                data: phaseData,
                itemStyle: { color: color, opacity: 0.8 },
                barGap: '10%',
            });
        });
    });

    // Reverse y-axis labels for top-down
    allYLabels.reverse();

    instance.setOption({
        title: { text: 'Trace Timeline' },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
        },
        legend: {
            type: 'scroll',
            bottom: 0,
            textStyle: { fontSize: 11 },
        },
        grid: { left: 80, right: 20, top: 56, bottom: 50 },
        xAxis: { type: 'value', name: 'Duration (ms)' },
        yAxis: {
            type: 'category',
            data: allYLabels,
            inverse: false,
        },
        series: series,
    }, { replaceMerge: ['series'] });
}

// ─── Eval score grid ─────────────────────────────────────────────────
// (Pure HTML table - no chart library needed)

function createEvalGrid(containerId, allResults) {
    var el = document.getElementById(containerId);
    if (!el) return;

    var html = '';

    var suiteMap = {};
    allResults.forEach(function(r) {
        if (!suiteMap[r.suite]) suiteMap[r.suite] = [];
        suiteMap[r.suite].push(r);
    });

    Object.keys(suiteMap).sort().forEach(function(suite) {
        var suiteResults = suiteMap[suite];
        var runs = [];
        var scoreMap = {};

        suiteResults.forEach(function(sr) {
            if (runs.indexOf(sr.run) === -1) runs.push(sr.run);
            if (sr.results) {
                sr.results.forEach(function(res) {
                    var key = res.score_name;
                    if (!scoreMap[key]) scoreMap[key] = {};
                    var existing = scoreMap[key][sr.run];
                    if (!existing || res.step >= existing.step) {
                        scoreMap[key][sr.run] = res;
                    }
                });
            }
        });

        var scoreNames = Object.keys(scoreMap).sort();
        if (!scoreNames.length) return;

        html += '<div class="eval-suite-section">';
        html += '<h3 class="eval-suite-heading">' + escapeHtmlCharts(suite) + '</h3>';
        html += '<table class="eval-table"><thead><tr><th>Score</th>';
        runs.forEach(function(run) {
            var color = getRunColor(run);
            html += '<th><span class="eval-run-header" style="border-bottom: 2px solid ' + color + '">' +
                    escapeHtmlCharts(run) + '</span></th>';
        });
        html += '</tr></thead><tbody>';

        scoreNames.forEach(function(scoreName) {
            html += '<tr><td class="eval-score-name">' + escapeHtmlCharts(scoreName) + '</td>';
            var values = runs.map(function(run) {
                var entry = scoreMap[scoreName][run];
                return entry ? entry.score_value : null;
            });
            var maxVal = Math.max.apply(null, values.filter(function(v) { return v !== null; }));

            runs.forEach(function(run) {
                var entry = scoreMap[scoreName][run];
                if (entry) {
                    var isBest = Math.abs(entry.score_value - maxVal) < 1e-9;
                    var cls = isBest ? 'eval-value eval-value-best' : 'eval-value';
                    html += '<td class="' + cls + '">' + entry.score_value.toFixed(4) + '</td>';
                } else {
                    html += '<td class="eval-value eval-value-na">\u2014</td>';
                }
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
    });

    el.innerHTML = html;
}

function escapeHtmlCharts(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ─── Artifact gallery ────────────────────────────────────────────────
// (Pure HTML - unchanged from Plotly version, no chart library needed)

var _artifactSliderState = {};

// Workspace sample images are served by the supervisor's /files scope (img_url).
// Fall back to the DB blob route for board-stored artifacts that lack a path.
function artifactSrc(run, artifact) {
    if (artifact.img_url) return artifact.img_url;
    return '/api/runs/' + encodeURIComponent(run) + '/blob/' +
           encodeURIComponent(artifact.blob_key);
}

function createArtifactGallery(containerId, allResults) {
    var el = document.getElementById(containerId);
    if (!el) return;

    var html = '';

    allResults.forEach(function(result) {
        if (!result.artifacts || !result.artifacts.length) return;

        var sorted = result.artifacts.slice().sort(function(a, b) { return a.step - b.step; });
        var sectionKey = result.run + '|||' + result.tag;

        if (!_artifactSliderState[sectionKey]) {
            _artifactSliderState[sectionKey] = { currentIndex: sorted.length - 1 };
        }
        var curIdx = _artifactSliderState[sectionKey].currentIndex;
        if (curIdx >= sorted.length) curIdx = sorted.length - 1;
        if (curIdx < 0) curIdx = 0;
        _artifactSliderState[sectionKey].currentIndex = curIdx;

        var current = sorted[curIdx];
        var minStep = sorted[0].step;
        var maxStep = sorted[sorted.length - 1].step;
        var isImage = current.mime_type && current.mime_type.indexOf('image/') === 0;
        var escapedKey = escapeHtmlCharts(sectionKey);

        html += '<div class="artifact-section artifact-slider-section" data-section-key="' + escapedKey + '">';
        html += '<h3 class="artifact-heading">' + escapeHtmlCharts(result.run) + ' / ' +
                escapeHtmlCharts(result.tag) + '</h3>';

        html += '<div class="artifact-slider-controls">';
        html += '<span class="artifact-step-label">Step: <strong class="artifact-step-display">' +
                current.step + '</strong></span>';
        html += '<input type="range" class="artifact-step-slider" ' +
                'min="0" max="' + (sorted.length - 1) + '" value="' + curIdx + '" ' +
                'data-section-key="' + escapedKey + '">';
        html += '<span class="artifact-step-range">' + minStep + ' \u2014 ' + maxStep +
                ' (' + sorted.length + ' steps)</span>';
        html += '</div>';

        html += '<div class="artifact-preview-wrap">';
        if (isImage && (current.img_url || current.blob_key)) {
            html += '<img class="artifact-preview-img" src="' +
                    artifactSrc(result.run, current) + '" ' +
                    'alt="Step ' + current.step + '">';
        } else {
            html += '<div class="artifact-preview-placeholder">' +
                    (current.mime_type || 'unknown') + '</div>';
        }
        if (current.width && current.height) {
            html += '<div class="artifact-preview-dims">' + current.width + 'x' + current.height + '</div>';
        }
        html += '</div>';

        html += '<div class="artifact-filmstrip">';
        sorted.forEach(function(artifact, idx) {
            var activeClass = idx === curIdx ? ' artifact-filmstrip-active' : '';
            var thumbIsImage = artifact.mime_type && artifact.mime_type.indexOf('image/') === 0;

            html += '<div class="artifact-filmstrip-item' + activeClass + '" ' +
                    'data-index="' + idx + '" data-section-key="' + escapedKey + '" title="Step ' + artifact.step + '">';
            if (thumbIsImage && (artifact.img_url || artifact.blob_key)) {
                html += '<img class="artifact-filmstrip-thumb" src="' +
                        artifactSrc(result.run, artifact) + '" ' +
                        'alt="Step ' + artifact.step + '" loading="lazy">';
            } else {
                html += '<div class="artifact-filmstrip-placeholder">' +
                        (artifact.mime_type || 'unknown') + '</div>';
            }
            html += '<span class="artifact-filmstrip-step">' + artifact.step + '</span>';
            html += '</div>';
        });
        html += '</div>';

        html += '</div>';
    });

    el.innerHTML = html;
    _attachArtifactSliderHandlers(el, allResults);
}

function _attachArtifactSliderHandlers(containerEl, allResults) {
    var sectionLookup = {};
    allResults.forEach(function(result) {
        if (!result.artifacts || !result.artifacts.length) return;
        var sorted = result.artifacts.slice().sort(function(a, b) { return a.step - b.step; });
        var sectionKey = result.run + '|||' + result.tag;
        sectionLookup[sectionKey] = { run: result.run, tag: result.tag, artifacts: sorted };
    });

    var sliders = containerEl.querySelectorAll('.artifact-step-slider');
    sliders.forEach(function(slider) {
        slider.addEventListener('input', function() {
            var sectionKey = slider.getAttribute('data-section-key');
            var idx = parseInt(slider.value, 10);
            _updateArtifactSection(containerEl, sectionKey, idx, sectionLookup[sectionKey]);
        });
    });

    var filmItems = containerEl.querySelectorAll('.artifact-filmstrip-item');
    filmItems.forEach(function(item) {
        item.addEventListener('click', function() {
            var sectionKey = item.getAttribute('data-section-key');
            var idx = parseInt(item.getAttribute('data-index'), 10);
            _updateArtifactSection(containerEl, sectionKey, idx, sectionLookup[sectionKey]);
        });
    });
}

function _updateArtifactSection(containerEl, sectionKey, idx, sectionData) {
    if (!sectionData) return;
    var artifacts = sectionData.artifacts;
    if (idx < 0 || idx >= artifacts.length) return;

    _artifactSliderState[sectionKey] = { currentIndex: idx };
    var current = artifacts[idx];
    var isImage = current.mime_type && current.mime_type.indexOf('image/') === 0;

    var sections = containerEl.querySelectorAll('.artifact-slider-section');
    var sectionEl = null;
    for (var i = 0; i < sections.length; i++) {
        if (sections[i].getAttribute('data-section-key') === sectionKey) {
            sectionEl = sections[i];
            break;
        }
    }
    if (!sectionEl) return;

    var stepDisplay = sectionEl.querySelector('.artifact-step-display');
    if (stepDisplay) stepDisplay.textContent = current.step;

    var slider = sectionEl.querySelector('.artifact-step-slider');
    if (slider) slider.value = idx;

    var previewWrap = sectionEl.querySelector('.artifact-preview-wrap');
    if (previewWrap) {
        var previewHtml = '';
        if (isImage && current.blob_key) {
            previewHtml = '<img class="artifact-preview-img" src="/api/runs/' +
                    encodeURIComponent(sectionData.run) + '/blob/' +
                    encodeURIComponent(current.blob_key) + '" ' +
                    'alt="Step ' + current.step + '">';
        } else {
            previewHtml = '<div class="artifact-preview-placeholder">' +
                    (current.mime_type || 'unknown') + '</div>';
        }
        if (current.width && current.height) {
            previewHtml += '<div class="artifact-preview-dims">' + current.width + 'x' + current.height + '</div>';
        }
        previewWrap.innerHTML = previewHtml;
    }

    var filmItems = sectionEl.querySelectorAll('.artifact-filmstrip-item');
    filmItems.forEach(function(item) {
        var itemIdx = parseInt(item.getAttribute('data-index'), 10);
        if (itemIdx === idx) {
            item.classList.add('artifact-filmstrip-active');
        } else {
            item.classList.remove('artifact-filmstrip-active');
        }
    });

    if (filmItems[idx]) {
        filmItems[idx].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
}

// ── PR Curve Chart ─────────────────────────────────────────────────
function createPRCurveChart(containerId, allResults, selectedStep) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    var runs = Object.keys(allResults);
    if (runs.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No PR curve data available</p></div>';
        return;
    }

    var allSteps = new Set();
    runs.forEach(function(run) {
        (allResults[run] || []).forEach(function(item) {
            allSteps.add(item.step);
        });
    });
    var sortedSteps = Array.from(allSteps).sort(function(a, b) { return a - b; });

    if (sortedSteps.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No PR curve data available</p></div>';
        return;
    }

    // Step selector
    var controls = document.createElement('div');
    controls.className = 'pr-curve-controls';

    var stepLabel = document.createElement('label');
    stepLabel.textContent = 'Step: ';
    stepLabel.className = 'pr-curve-step-label';

    var stepSelect = document.createElement('select');
    stepSelect.className = 'pr-curve-step-select';
    sortedSteps.forEach(function(s) {
        var opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        if (selectedStep !== undefined && s === selectedStep) opt.selected = true;
        stepSelect.appendChild(opt);
    });

    if (selectedStep === undefined) {
        stepSelect.value = sortedSteps[sortedSteps.length - 1];
    }

    stepLabel.appendChild(stepSelect);
    controls.appendChild(stepLabel);
    container.appendChild(controls);

    var chartEl = document.createElement('div');
    chartEl.className = 'chart-panel pr-curve-chart';
    chartEl.id = containerId + '-plot';
    container.appendChild(chartEl);

    function renderForStep(step) {
        var instance = getOrCreateChart(chartEl.id);
        if (!instance) return;

        var series = [];
        var legendData = [];

        runs.forEach(function(run) {
            var items = (allResults[run] || []).filter(function(item) {
                return item.step === step;
            });
            items.forEach(function(item) {
                var runColor = getRunColor(run);
                var name = run + (item.class_index > 0 ? ' (class ' + item.class_index + ')' : '');

                // Compute AUC
                var auc = 0;
                for (var i = 1; i < item.recall.length; i++) {
                    auc += Math.abs(item.recall[i] - item.recall[i-1]) * (item.precision[i] + item.precision[i-1]) / 2;
                }

                var legendName = name + ' (AUC=' + auc.toFixed(3) + ')';
                legendData.push(legendName);

                series.push({
                    name: legendName,
                    type: 'line',
                    showSymbol: false,
                    lineStyle: { width: 2, color: runColor },
                    itemStyle: { color: runColor },
                    data: item.recall.map(function(r, i) {
                        return [r, item.precision[i]];
                    }),
                });
            });
        });

        instance.setOption({
            title: { text: 'Precision-Recall Curve (Step ' + step + ')' },
            tooltip: {
                trigger: 'axis',
                formatter: function(params) {
                    if (!params || !params.length) return '';
                    var lines = [];
                    params.forEach(function(p) {
                        if (p.value) {
                            lines.push(p.seriesName + '<br>Recall: ' + p.value[0].toFixed(3) +
                                ', Precision: ' + p.value[1].toFixed(3));
                        }
                    });
                    return lines.join('<br>');
                },
            },
            legend: {
                data: legendData,
                bottom: 0,
                textStyle: { fontSize: 11 },
            },
            grid: { left: 70, right: 20, top: 56, bottom: 50 },
            xAxis: { type: 'value', name: 'Recall', min: 0, max: 1.05 },
            yAxis: { type: 'value', name: 'Precision', min: 0, max: 1.05 },
            series: series,
        }, { replaceMerge: ['series'] });
    }

    renderForStep(parseInt(stepSelect.value));

    stepSelect.addEventListener('change', function() {
        renderForStep(parseInt(stepSelect.value));
    });
}

// ── Audio Gallery ──────────────────────────────────────────────────
function formatAudioDuration(durationMs) {
    if (!durationMs && durationMs !== 0) return '';
    return (durationMs / 1000).toFixed(1) + 's';
}

function formatAudioDb(value) {
    if (value === null || value === undefined || !isFinite(value)) return '';
    return value.toFixed(1) + ' dBFS';
}

function pauseOtherAudioPlayers(activeAudio) {
    document.querySelectorAll('.audio-card audio').forEach(function(player) {
        if (player !== activeAudio) player.pause();
    });
}

function createAudioBlobUrl(run, blobKey) {
    return '/api/runs/' + encodeURIComponent(run) + '/blob/' + encodeURIComponent(blobKey);
}

function drawAudioWaveform(canvas, peaks, progress) {
    if (!canvas || !peaks || !peaks.length) return;

    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    var width = canvas.width;
    var height = canvas.height;
    var mid = height / 2;
    var amp = (height * 0.5) - 6;
    var styles = getComputedStyle(document.documentElement);
    var bg = styles.getPropertyValue('--bg-tertiary').trim() || '#111827';
    var border = styles.getPropertyValue('--border-color').trim() || '#334155';
    var muted = styles.getPropertyValue('--text-secondary').trim() || '#94a3b8';
    var accent = styles.getPropertyValue('--accent-color').trim() || '#4f8cff';
    var progressX = width * Math.max(0, Math.min(1, progress || 0));

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = border;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    function drawPeaks(color, maxX) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var i = 0; i < peaks.length; i++) {
            var pair = peaks[i];
            if (!pair || pair.length < 2) continue;
            var x = peaks.length === 1 ? width / 2 : (i / (peaks.length - 1)) * width;
            if (maxX !== null && x > maxX) continue;
            var yMin = mid - (pair[1] * amp);
            var yMax = mid - (pair[0] * amp);
            ctx.moveTo(x, yMin);
            ctx.lineTo(x, yMax);
        }
        ctx.stroke();
    }

    drawPeaks(muted, null);
    if (progressX > 0) drawPeaks(accent, progressX);
}

function drawAudioSpectrogram(canvas, columns, progress) {
    if (!canvas || !columns || !columns.length) return;

    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    var width = canvas.width;
    var height = canvas.height;
    var styles = getComputedStyle(document.documentElement);
    var bg = styles.getPropertyValue('--bg-tertiary').trim() || '#111827';
    var border = styles.getPropertyValue('--border-color').trim() || '#334155';
    var progressColor = styles.getPropertyValue('--accent-color').trim() || '#4f8cff';
    var progressX = width * Math.max(0, Math.min(1, progress || 0));
    var cols = columns.length;
    var rows = (columns[0] || []).length;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    if (!rows) return;

    var cellWidth = width / cols;
    var cellHeight = height / rows;
    for (var x = 0; x < cols; x++) {
        var column = columns[x] || [];
        for (var y = 0; y < rows; y++) {
            var intensity = Math.max(0, Math.min(1, column[y] || 0));
            var hue = 220 - (intensity * 190);
            var lightness = 10 + (intensity * 55);
            ctx.fillStyle = 'hsl(' + hue.toFixed(0) + ' 88% ' + lightness.toFixed(0) + '%)';
            ctx.fillRect(
                x * cellWidth,
                height - ((y + 1) * cellHeight),
                Math.ceil(cellWidth + 0.5),
                Math.ceil(cellHeight + 0.5)
            );
        }
    }

    ctx.strokeStyle = border;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    if (progressX > 0) {
        ctx.strokeStyle = progressColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(progressX, 0);
        ctx.lineTo(progressX, height);
        ctx.stroke();
    }
}

function attachAudioTimelineSeek(canvas, audio) {
    if (!canvas || !audio) return;
    canvas.addEventListener('click', function(evt) {
        if (!audio.duration || !isFinite(audio.duration)) return;
        var rect = canvas.getBoundingClientRect();
        if (!rect.width) return;
        var ratio = (evt.clientX - rect.left) / rect.width;
        audio.currentTime = Math.max(0, Math.min(1, ratio)) * audio.duration;
    });
}

function attachAudioPreview(audio, renderers) {
    if (!audio || !renderers || !renderers.length) return;

    function render(progressOverride) {
        var progress = progressOverride;
        if (progress === undefined) {
            progress = audio.duration ? (audio.currentTime / audio.duration) : 0;
        }
        renderers.forEach(function(renderer) { renderer(progress); });
    }

    render(0);

    audio.addEventListener('loadedmetadata', function() { render(); });
    audio.addEventListener('timeupdate', function() { render(); });
    audio.addEventListener('seeked', function() { render(); });
    audio.addEventListener('ended', function() {
        render(0);
    });
    audio.addEventListener('play', function() {
        pauseOtherAudioPlayers(audio);
        render();
    });
    audio.addEventListener('pause', function() { render(); });
}

// (Pure HTML - no chart library needed)
function createAudioGallery(containerId, allResults) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    var runs = Object.keys(allResults);
    if (runs.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No audio data available</p></div>';
        return;
    }

    runs.forEach(function(run) {
        var items = allResults[run];
        if (!items || !items.length) return;

        var itemsByTag = {};
        items.forEach(function(item) {
            var tag = item.tag || 'audio';
            if (!itemsByTag[tag]) itemsByTag[tag] = [];
            itemsByTag[tag].push(item);
        });

        var section = document.createElement('div');
        section.className = 'audio-run-section';

        var heading = document.createElement('h3');
        heading.className = 'audio-run-heading';
        heading.textContent = run;
        section.appendChild(heading);

        Object.keys(itemsByTag).sort().forEach(function(tag) {
            var tagSection = document.createElement('div');
            tagSection.className = 'audio-tag-section';

            var tagHeading = document.createElement('div');
            tagHeading.className = 'audio-tag-heading';
            tagHeading.textContent = tag;
            tagSection.appendChild(tagHeading);

            var grid = document.createElement('div');
            grid.className = 'audio-grid';

            itemsByTag[tag].sort(function(a, b) { return b.step - a.step; }).forEach(function(item) {
                var card = document.createElement('div');
                card.className = 'audio-card';

                var info = document.createElement('div');
                info.className = 'audio-info';
                info.innerHTML = '<span class="audio-step">Step ' + item.step + '</span>' +
                    '<span class="audio-meta">' + item.sample_rate + ' Hz' +
                    (item.duration_ms ? ' \u00b7 ' + formatAudioDuration(item.duration_ms) : '') +
                    (item.num_channels > 1 ? ' \u00b7 ' + item.num_channels + 'ch' : '') +
                    '</span>';
                card.appendChild(info);

                var stats = document.createElement('div');
                stats.className = 'audio-stats';
                stats.innerHTML =
                    '<span class="audio-stat">Peak ' + formatAudioDb(item.peak_db) + '</span>' +
                    '<span class="audio-stat">RMS ' + formatAudioDb(item.rms_db) + '</span>';
                card.appendChild(stats);

                var blobUrl = createAudioBlobUrl(run, item.blob_key);
                var waveform = document.createElement('canvas');
                waveform.className = 'audio-waveform';
                waveform.width = 640;
                waveform.height = 88;
                if (item.waveform && item.waveform.length) {
                    drawAudioWaveform(waveform, item.waveform, 0);
                } else {
                    waveform.classList.add('is-empty');
                    var waveEmpty = document.createElement('div');
                    waveEmpty.className = 'audio-waveform-empty';
                    waveEmpty.textContent = 'Waveform preview unavailable';
                    card.appendChild(waveform);
                    card.appendChild(waveEmpty);
                }

                if (item.waveform && item.waveform.length) {
                    card.appendChild(waveform);
                }

                var spectrogram = document.createElement('canvas');
                spectrogram.className = 'audio-spectrogram';
                spectrogram.width = 640;
                spectrogram.height = 120;
                if (item.spectrogram && item.spectrogram.length) {
                    drawAudioSpectrogram(spectrogram, item.spectrogram, 0);
                    card.appendChild(spectrogram);
                } else {
                    spectrogram.classList.add('is-empty');
                    var specEmpty = document.createElement('div');
                    specEmpty.className = 'audio-waveform-empty';
                    specEmpty.textContent = 'Spectrogram preview unavailable';
                    card.appendChild(spectrogram);
                    card.appendChild(specEmpty);
                }

                var controls = document.createElement('div');
                controls.className = 'audio-controls-row';

                var audio = document.createElement('audio');
                audio.className = 'audio-player';
                audio.controls = true;
                audio.preload = 'none';
                audio.src = blobUrl;
                controls.appendChild(audio);

                var actions = document.createElement('div');
                actions.className = 'audio-actions';

                var download = document.createElement('a');
                download.className = 'audio-action-link';
                download.href = blobUrl;
                download.download = item.blob_key;
                download.textContent = 'Download';
                actions.appendChild(download);

                var open = document.createElement('a');
                open.className = 'audio-action-link';
                open.href = blobUrl;
                open.target = '_blank';
                open.rel = 'noopener noreferrer';
                open.textContent = 'Open';
                actions.appendChild(open);

                controls.appendChild(actions);
                card.appendChild(controls);

                var renderers = [];
                if (item.waveform && item.waveform.length) {
                    renderers.push(function(progress) { drawAudioWaveform(waveform, item.waveform, progress); });
                    attachAudioTimelineSeek(waveform, audio);
                }
                if (item.spectrogram && item.spectrogram.length) {
                    renderers.push(function(progress) { drawAudioSpectrogram(spectrogram, item.spectrogram, progress); });
                    attachAudioTimelineSeek(spectrogram, audio);
                }
                attachAudioPreview(audio, renderers);

                if (item.label) {
                    var label = document.createElement('div');
                    label.className = 'audio-label';
                    label.textContent = item.label;
                    card.appendChild(label);
                }

                grid.appendChild(card);
            });

            tagSection.appendChild(grid);
            section.appendChild(tagSection);
        });

        container.appendChild(section);
    });
}
