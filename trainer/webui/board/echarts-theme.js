/**
 * echarts-theme.js -- Dark theme registration + instance registry for SerenityBoard.
 *
 * Replaces Plotly's DOM-attached state with an explicit instance map.
 * Must be loaded AFTER echarts.min.js and BEFORE charts.js.
 */

/* global echarts */

(function() {
    'use strict';

    // ─── Dark theme ──────────────────────────────────────────────────────

    // TensorBoard-inspired dark theme: clean, minimal, professional
    echarts.registerTheme('serenity-dark', {
        backgroundColor: 'transparent',
        textStyle: {
            color: '#9e9e9e',
            fontFamily: 'Roboto, system-ui, -apple-system, sans-serif',
            fontSize: 11,
        },
        title: {
            textStyle: { color: '#e0e0e0', fontSize: 13, fontWeight: 500 },
            left: 12,
            top: 10,
        },
        grid: {
            backgroundColor: 'rgba(255,255,255,0.02)',
            show: true,
            borderColor: 'rgba(255,255,255,0.08)',
            borderWidth: 1,
            left: 55,
            right: 16,
            top: 48,
            bottom: 36,
            containLabel: false,
        },
        categoryAxis: {
            axisLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            axisTick: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            axisLabel: { color: '#999', fontSize: 11 },
            splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        },
        valueAxis: {
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: '#999', fontSize: 11 },
            splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        },
        timeAxis: {
            axisLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            axisTick: { show: true, lineStyle: { color: 'rgba(255,255,255,0.15)' } },
            axisLabel: { color: '#999', fontSize: 11 },
            splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.08)', width: 1 } },
        },
        legend: {
            backgroundColor: 'transparent',
            borderColor: 'transparent',
            borderWidth: 0,
            textStyle: { color: '#9e9e9e', fontSize: 11 },
            pageTextStyle: { color: '#757575' },
        },
        tooltip: {
            backgroundColor: 'rgba(30,30,30,0.95)',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            textStyle: { color: '#e0e0e0', fontSize: 12 },
            extraCssText: 'border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,0.3);',
        },
        toolbox: {
            iconStyle: { borderColor: '#616161' },
            emphasis: { iconStyle: { borderColor: '#bdbdbd' } },
        },
        dataZoom: {
            backgroundColor: 'rgba(255,255,255,0.03)',
            fillerColor: 'rgba(255,152,0,0.08)',
            handleColor: '#ff9800',
            borderColor: 'rgba(255,255,255,0.06)',
            textStyle: { color: '#757575' },
        },
        visualMap: {
            textStyle: { color: '#757575' },
        },
        // TensorBoard's default palette (d3.schemeCategory10)
        color: [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ],
    });

    // ─── Instance registry ───────────────────────────────────────────────

    var _instances = {};  // containerId -> ECharts instance

    /**
     * Get or create an ECharts instance for a container element.
     * @param {string} id - DOM element id.
     * @returns {Object|null} ECharts instance or null if element not found.
     */
    window.getOrCreateChart = function(id) {
        if (_instances[id]) {
            // Verify the DOM element still matches
            var existing = _instances[id];
            if (!existing.isDisposed()) return existing;
            delete _instances[id];
        }
        var el = document.getElementById(id);
        if (!el) return null;
        var instance = echarts.init(el, 'serenity-dark');
        _instances[id] = instance;
        return instance;
    };

    /**
     * Dispose an ECharts instance and remove from registry.
     * @param {string} id - DOM element id.
     */
    window.disposeChart = function(id) {
        var instance = _instances[id];
        if (instance && !instance.isDisposed()) {
            instance.dispose();
        }
        delete _instances[id];
    };

    /**
     * Get existing instance (no creation).
     * @param {string} id - DOM element id.
     * @returns {Object|null}
     */
    window.getChartInstance = function(id) {
        var instance = _instances[id];
        if (instance && !instance.isDisposed()) return instance;
        if (instance) delete _instances[id];
        return null;
    };

    /**
     * Resize all registered chart instances.
     */
    window.resizeAllCharts = function() {
        var ids = Object.keys(_instances);
        for (var i = 0; i < ids.length; i++) {
            var instance = _instances[ids[i]];
            if (instance && !instance.isDisposed()) {
                instance.resize();
            } else {
                delete _instances[ids[i]];
            }
        }
    };

    /**
     * Get all registered instance IDs.
     * @returns {string[]}
     */
    window.getRegisteredChartIds = function() {
        return Object.keys(_instances);
    };

})();
