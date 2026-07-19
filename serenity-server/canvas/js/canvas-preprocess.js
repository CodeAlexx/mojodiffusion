"use strict";
/**
 * Preprocessors — SerenityFlow Canvas v2
 *
 * Runs image preprocessing on the backend for ControlNet layers.
 * Canny, depth, lineart, pose, soft_edge, tile, normal, color, scribble.
 */
// ── Preprocessor Module ──
var CanvasPreprocess = (function () {
    'use strict';
    var _processing = false;
    var METHODS = {
        canny: 'Canny Edge',
        depth: 'Depth',
        lineart: 'Lineart',
        pose: 'Pose',
        soft_edge: 'Soft Edge',
        tile: 'Tile',
        normal: 'Normal Map',
        color: 'Color Map',
        scribble: 'Scribble',
    };
    function getApiBase() {
        return window.location.protocol + '//' + window.location.host;
    }
    function getAllMethods() {
        return METHODS;
    }
    function isProcessing() {
        return _processing;
    }
    /**
     * Run a preprocessor on an image.
     * @param method - Preprocessor key (canny, depth, etc.)
     * @param imageDataUrl - Source image as data URL
     * @param params - Optional extra params (e.g. {low: 100, high: 200} for canny)
     * @returns Promise<string> - Processed image as data URL
     */
    function process(method, imageDataUrl, params) {
        if (_processing)
            return Promise.reject(new Error('Already processing'));
        if (!METHODS[method])
            return Promise.reject(new Error('Unknown method: ' + method));
        _processing = true;
        // Convert data URL to Blob
        var parts = imageDataUrl.split(',');
        var byteString = atob(parts[1] || '');
        var mimeString = parts[0].split(':')[1].split(';')[0];
        var ab = new ArrayBuffer(byteString.length);
        var ia = new Uint8Array(ab);
        for (var i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        var blob = new Blob([ab], { type: mimeString });
        var form = new FormData();
        form.append('image', blob, 'source.png');
        if (params) {
            form.append('params', JSON.stringify(params));
        }
        return fetch(getApiBase() + '/canvas/preprocess/' + method, {
            method: 'POST',
            body: form,
        })
            .then(function (res) {
            _processing = false;
            if (!res.ok) {
                return res.json().then(function (err) {
                    throw new Error(err.error || 'Preprocessing failed');
                });
            }
            return res.blob();
        })
            .then(function (resultBlob) {
            return new Promise(function (resolve) {
                var reader = new FileReader();
                reader.onload = function () { resolve(reader.result); };
                reader.readAsDataURL(resultBlob);
            });
        })
            .catch(function (err) {
            _processing = false;
            throw err;
        });
    }
    /**
     * Process the active control layer's reference image and update it.
     */
    function processActiveControlLayer(method, ctx, params) {
        var al = ctx.getActiveLayer();
        if (!al || al.data.type !== 'control')
            return;
        var cd = al.data;
        if (!cd.refImageSrc)
            return;
        var processBtn = document.getElementById('cv-preprocess-btn');
        if (processBtn) {
            processBtn.textContent = 'Processing...';
            processBtn.disabled = true;
        }
        process(method, cd.refImageSrc, params)
            .then(function (resultDataUrl) {
            // Update the control layer with processed image
            cd.refImageSrc = resultDataUrl;
            // Update the canvas display
            var img = new window.Image();
            img.onload = function () {
                al.konvaLayer.destroyChildren();
                var bb = ctx.boundingBox;
                var kImg = new Konva.Image({
                    image: img,
                    x: bb.x(), y: bb.y(),
                    width: bb.width(), height: bb.height(),
                    opacity: 0.5,
                    listening: false,
                });
                al.konvaLayer.add(kImg);
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            };
            img.src = resultDataUrl;
            // Update the image well preview
            var well = document.getElementById('cv-control-well');
            if (well) {
                well.innerHTML = '<img src="' + resultDataUrl + '">';
            }
        })
            .catch(function (err) {
            console.error('[Preprocess]', err.message || err);
        })
            .finally(function () {
            if (processBtn) {
                processBtn.textContent = 'Process';
                processBtn.disabled = false;
            }
        });
    }
    return {
        getAllMethods: getAllMethods,
        isProcessing: isProcessing,
        process: process,
        processActiveControlLayer: processActiveControlLayer,
    };
})();
//# sourceMappingURL=canvas-preprocess.js.map