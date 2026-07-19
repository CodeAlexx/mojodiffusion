/**
 * projector.js -- Embedding Projector for SerenityBoard.
 * Fetches embedding blobs, runs PCA, renders 3D scatter via ECharts-GL.
 */
(function() {
    'use strict';

    // DOM refs
    var runSelect = document.getElementById('projector-run-select');
    var tagSelect = document.getElementById('projector-tag-select');
    var stepSelect = document.getElementById('projector-step-select');
    var methodSelect = document.getElementById('projector-method');
    var colorBySelect = document.getElementById('projector-color-by');
    var loadBtn = document.getElementById('projector-load-btn');
    var plotDiv = document.getElementById('projector-plot');
    var statusEl = document.getElementById('projector-status');

    // Cache the last loaded data so color-by changes don't re-fetch
    var lastLoadedData = null;

    // Guard against concurrent Load clicks
    var loadInProgress = false;

    // --- API helpers ---

    function fetchJSON(url) {
        return fetch(url, { cache: 'no-store' }).then(function(r) {
            if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + r.statusText);
            return r.json();
        });
    }

    function fetchBlob(url) {
        return fetch(url, { cache: 'no-store' }).then(function(r) {
            if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + r.statusText);
            return r.arrayBuffer();
        });
    }

    // --- Populate dropdowns ---

    // Called by app.js when projector tab activates
    window.projectorSetRuns = function(runs) {
        runSelect.innerHTML = '<option value="">Select run...</option>';
        runs.forEach(function(r) {
            var opt = document.createElement('option');
            opt.value = r.name || r;
            opt.textContent = r.name || r;
            runSelect.appendChild(opt);
        });
    };

    runSelect.addEventListener('change', function() {
        var run = runSelect.value;
        if (!run) return;
        tagSelect.innerHTML = '<option value="">Loading...</option>';
        stepSelect.innerHTML = '<option value="">Select step...</option>';
        colorBySelect.innerHTML = '<option value="">Color by...</option>';
        lastLoadedData = null;
        fetchJSON('/api/runs/' + encodeURIComponent(run) + '/embeddings')
            .then(function(data) {
                var tags = [];
                var seen = {};
                (data || []).forEach(function(d) {
                    if (!seen[d.tag]) {
                        seen[d.tag] = true;
                        tags.push(d.tag);
                    }
                });
                tagSelect.innerHTML = '<option value="">Select tag...</option>';
                if (tags.length === 0) {
                    tagSelect.innerHTML = '<option value="">No embeddings found</option>';
                    return;
                }
                tags.forEach(function(t) {
                    var opt = document.createElement('option');
                    opt.value = t;
                    opt.textContent = t;
                    tagSelect.appendChild(opt);
                });
            })
            .catch(function(err) {
                console.error('Failed to fetch embedding tags:', err);
                tagSelect.innerHTML = '<option value="">Error loading tags</option>';
            });
    });

    tagSelect.addEventListener('change', function() {
        var run = runSelect.value;
        var tag = tagSelect.value;
        if (!run || !tag) return;
        stepSelect.innerHTML = '<option value="">Loading...</option>';
        colorBySelect.innerHTML = '<option value="">Color by...</option>';
        lastLoadedData = null;
        fetchJSON('/api/runs/' + encodeURIComponent(run) + '/embeddings?tag=' + encodeURIComponent(tag))
            .then(function(data) {
                stepSelect.innerHTML = '<option value="">Select step...</option>';
                if (!data || data.length === 0) {
                    stepSelect.innerHTML = '<option value="">No steps found</option>';
                    return;
                }
                data.forEach(function(d) {
                    var opt = document.createElement('option');
                    opt.value = d.step;
                    opt.textContent = 'Step ' + d.step;
                    stepSelect.appendChild(opt);
                });
            })
            .catch(function(err) {
                console.error('Failed to fetch embedding steps:', err);
                stepSelect.innerHTML = '<option value="">Error loading steps</option>';
            });
    });

    // --- Load & Render ---

    loadBtn.addEventListener('click', function() {
        var run = runSelect.value;
        var tag = tagSelect.value;
        var step = stepSelect.value;
        if (!run || !tag || step === '') {
            statusEl.textContent = 'Select run, tag, and step first.';
            return;
        }
        if (loadInProgress) return;
        loadInProgress = true;
        loadBtn.disabled = true;
        statusEl.textContent = 'Loading embedding...';

        fetchJSON('/api/runs/' + encodeURIComponent(run) + '/embeddings?tag=' + encodeURIComponent(tag) + '&step=' + step)
            .then(function(emb) {
                // emb is a single object (tag+step returns one result)
                var blobUrl = '/api/runs/' + encodeURIComponent(run) + '/blob/' + emb.tensor_blob_key;
                var n = emb.num_points;
                var d = emb.dimensions;
                var metadata = emb.metadata || null;
                var headers = emb.metadata_header || [];

                // Populate color-by dropdown
                colorBySelect.innerHTML = '<option value="">None</option>';
                if (headers.length > 0) {
                    headers.forEach(function(h, i) {
                        var opt = document.createElement('option');
                        opt.value = i;
                        opt.textContent = h;
                        colorBySelect.appendChild(opt);
                    });
                } else if (metadata && metadata.length > 0) {
                    var opt = document.createElement('option');
                    opt.value = 0;
                    opt.textContent = 'Label';
                    colorBySelect.appendChild(opt);
                }

                statusEl.textContent = 'Fetching ' + n + 'x' + d + ' matrix...';
                return fetchBlob(blobUrl).then(function(buf) {
                    return { buf: buf, n: n, d: d, metadata: metadata, headers: headers };
                });
            })
            .then(function(data) {
                var mat = new Float32Array(data.buf);
                statusEl.textContent = 'Running PCA on ' + data.n + ' points...';

                // Run PCA
                var projected = pcaProject(mat, data.n, data.d, 3);

                // Cache for color-by changes
                lastLoadedData = {
                    projected: projected,
                    n: data.n,
                    metadata: data.metadata,
                    headers: data.headers,
                };

                renderWithCurrentColorBy();
                statusEl.textContent = data.n + ' points projected via PCA.';
            })
            .catch(function(err) {
                statusEl.textContent = 'Error: ' + err.message;
                console.error('Projector error:', err);
            })
            .finally(function() {
                loadInProgress = false;
                loadBtn.disabled = false;
            });
    });

    // Re-render when color-by changes (no re-fetch needed)
    colorBySelect.addEventListener('change', function() {
        if (lastLoadedData) {
            renderWithCurrentColorBy();
        }
    });

    function renderWithCurrentColorBy() {
        if (!lastLoadedData) return;
        var data = lastLoadedData;

        // Build labels/colors
        var labels = null;
        var colors = null;
        if (data.metadata && data.metadata.length === data.n) {
            labels = data.metadata.map(function(m) {
                return Array.isArray(m) ? m.join(', ') : String(m);
            });

            var colIdx = colorBySelect.value;
            if (colIdx !== '') {
                colIdx = parseInt(colIdx);
                colors = data.metadata.map(function(m) {
                    return Array.isArray(m) ? m[colIdx] : m;
                });
            }
        }

        renderScatter3D(data.projected, data.n, labels, colors);
    }

    // --- PCA Implementation ---

    function pcaProject(mat, n, d, nComponents) {
        // Center the data (compute mean per dimension)
        var mean = new Float64Array(d);
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < d; j++) {
                mean[j] += mat[i * d + j];
            }
        }
        for (var j = 0; j < d; j++) mean[j] /= n;

        // Centered matrix
        var centered = new Float64Array(n * d);
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < d; j++) {
                centered[i * d + j] = mat[i * d + j] - mean[j];
            }
        }

        // For large D, use power iteration to find top-k eigenvectors
        // of the covariance matrix (d x d) or the Gram matrix (n x n)
        var result = new Float64Array(n * nComponents);

        if (d <= n) {
            // Covariance matrix approach (d x d)
            var cov = new Float64Array(d * d);
            for (var i = 0; i < n; i++) {
                for (var j = 0; j < d; j++) {
                    for (var k = j; k < d; k++) {
                        var val = centered[i * d + j] * centered[i * d + k];
                        cov[j * d + k] += val;
                        if (j !== k) cov[k * d + j] += val;
                    }
                }
            }
            for (var j = 0; j < d * d; j++) cov[j] /= (n - 1 || 1);

            // Power iteration for top nComponents eigenvectors
            var eigenvecs = powerIteration(cov, d, nComponents, 100);

            // Project: result[i][c] = dot(centered[i], eigenvec[c])
            for (var c = 0; c < nComponents; c++) {
                for (var i = 0; i < n; i++) {
                    var dot = 0;
                    for (var j = 0; j < d; j++) {
                        dot += centered[i * d + j] * eigenvecs[c * d + j];
                    }
                    result[i * nComponents + c] = dot;
                }
            }
        } else {
            // Gram matrix approach (n x n) when d > n
            // Eigenvectors u of K = X*X^T/(n-1) relate to PCA projections
            // via: projection = sqrt(lambda) * u
            // Power iteration gives us unit eigenvectors; we compute the
            // eigenvalue (Rayleigh quotient) to recover the correct scale.
            var gram = new Float64Array(n * n);
            for (var i = 0; i < n; i++) {
                for (var j = i; j < n; j++) {
                    var dot = 0;
                    for (var k = 0; k < d; k++) {
                        dot += centered[i * d + k] * centered[j * d + k];
                    }
                    gram[i * n + j] = dot / (n - 1 || 1);
                    gram[j * n + i] = gram[i * n + j];
                }
            }

            var eigenvecs = powerIteration(gram, n, nComponents, 100);

            // For each component, compute eigenvalue via Rayleigh quotient
            // and scale the projection by sqrt(lambda)
            for (var c = 0; c < nComponents; c++) {
                // Compute lambda = u^T * K * u (Rayleigh quotient)
                var lambda = 0;
                for (var i = 0; i < n; i++) {
                    var Ku_i = 0;
                    for (var j = 0; j < n; j++) {
                        Ku_i += gram[i * n + j] * eigenvecs[c * n + j];
                    }
                    lambda += eigenvecs[c * n + i] * Ku_i;
                }
                var scale = Math.sqrt(Math.abs(lambda));
                if (scale < 1e-10) scale = 1; // degenerate case: near-zero eigenvalue
                for (var i = 0; i < n; i++) {
                    result[i * nComponents + c] = eigenvecs[c * n + i] * scale;
                }
            }
        }

        return result;
    }

    function powerIteration(matrix, size, nVecs, maxIter) {
        var vecs = new Float64Array(nVecs * size);

        for (var v = 0; v < nVecs; v++) {
            // Random initial vector
            var vec = new Float64Array(size);
            for (var i = 0; i < size; i++) vec[i] = Math.random() - 0.5;
            normalize(vec);

            for (var iter = 0; iter < maxIter; iter++) {
                // Multiply: new_vec = matrix * vec
                var newVec = new Float64Array(size);
                for (var i = 0; i < size; i++) {
                    var sum = 0;
                    for (var j = 0; j < size; j++) {
                        sum += matrix[i * size + j] * vec[j];
                    }
                    newVec[i] = sum;
                }

                // Deflate: remove projections onto previous eigenvectors
                for (var p = 0; p < v; p++) {
                    var dot = 0;
                    for (var i = 0; i < size; i++) {
                        dot += newVec[i] * vecs[p * size + i];
                    }
                    for (var i = 0; i < size; i++) {
                        newVec[i] -= dot * vecs[p * size + i];
                    }
                }

                normalize(newVec);
                vec = newVec;
            }

            for (var i = 0; i < size; i++) vecs[v * size + i] = vec[i];
        }

        return vecs;
    }

    function normalize(vec) {
        var norm = 0;
        for (var i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
        norm = Math.sqrt(norm) || 1;
        for (var i = 0; i < vec.length; i++) vec[i] /= norm;
    }

    // --- ECharts-GL 3D Scatter ---

    /* global echarts, getOrCreateChart, disposeChart */

    function renderScatter3D(projected, n, labels, colors) {
        var data = [];
        for (var i = 0; i < n; i++) {
            var point = [projected[i * 3], projected[i * 3 + 1], projected[i * 3 + 2]];
            if (labels && labels[i]) {
                point.push(labels[i]);
            }
            data.push(point);
        }

        var instance = getOrCreateChart('projector-plot');
        if (!instance) return;

        var markerOpt = {
            symbolSize: 4,
            opacity: 0.8,
        };

        // Color handling
        var visualMapOpt = null;
        if (colors) {
            var uniqueVals = [];
            var seen = {};
            colors.forEach(function(c) {
                var key = String(c);
                if (!seen[key]) {
                    seen[key] = true;
                    uniqueVals.push(c);
                }
            });

            // Add color index to each data point
            if (uniqueVals.length <= 50) {
                // Categorical
                for (var i = 0; i < n; i++) {
                    data[i].push(uniqueVals.indexOf(colors[i]));
                }
                visualMapOpt = {
                    type: 'piecewise',
                    categories: uniqueVals.map(String),
                    dimension: labels ? 4 : 3,
                    textStyle: { color: '#a0a0b0' },
                    right: 10,
                    top: 'center',
                };
            } else {
                // Numeric
                var numColors = colors.map(Number);
                for (var i = 0; i < n; i++) {
                    data[i].push(numColors[i]);
                }
                visualMapOpt = {
                    min: Math.min.apply(null, numColors),
                    max: Math.max.apply(null, numColors),
                    dimension: labels ? 4 : 3,
                    inRange: { color: ['#440154', '#31688e', '#35b779', '#fde725'] },
                    textStyle: { color: '#a0a0b0' },
                    right: 10,
                    top: 'center',
                };
            }
        }

        var option = {
            backgroundColor: '#1a1a2e',
            tooltip: {
                formatter: function(params) {
                    if (!params.value) return '';
                    if (labels && params.value[3]) {
                        return params.value[3];
                    }
                    return 'PC1: ' + params.value[0].toFixed(3) +
                        '<br>PC2: ' + params.value[1].toFixed(3) +
                        '<br>PC3: ' + params.value[2].toFixed(3);
                },
            },
            xAxis3D: { name: 'PC1', type: 'value', axisLine: { lineStyle: { color: '#3a3a5a' } }, axisLabel: { color: '#a0a0b0' }, splitLine: { lineStyle: { color: '#2a2a4a' } } },
            yAxis3D: { name: 'PC2', type: 'value', axisLine: { lineStyle: { color: '#3a3a5a' } }, axisLabel: { color: '#a0a0b0' }, splitLine: { lineStyle: { color: '#2a2a4a' } } },
            zAxis3D: { name: 'PC3', type: 'value', axisLine: { lineStyle: { color: '#3a3a5a' } }, axisLabel: { color: '#a0a0b0' }, splitLine: { lineStyle: { color: '#2a2a4a' } } },
            grid3D: {
                viewControl: {
                    autoRotate: false,
                    projection: 'perspective',
                },
                environment: '#1a1a2e',
                light: {
                    main: { intensity: 1.2 },
                    ambient: { intensity: 0.3 },
                },
            },
            series: [{
                type: 'scatter3D',
                data: data,
                symbolSize: markerOpt.symbolSize,
                itemStyle: {
                    opacity: markerOpt.opacity,
                    color: colors ? undefined : '#4fc3f7',
                },
            }],
            animation: false,
        };

        if (visualMapOpt) {
            option.visualMap = visualMapOpt;
        }

        instance.setOption(option, { replaceMerge: ['series', 'visualMap'] });
    }

})();
