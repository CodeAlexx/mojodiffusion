"use strict";
/**
 * Video Timeline & Masking — SerenityFlow Canvas v2
 *
 * Minimal video timeline: load video, extract frames, scrub, display
 * current frame on canvas. Integrates with SAM for per-frame mask tracking.
 */
// ── Video Timeline Module ──
var CanvasVideo = (function () {
    'use strict';
    var _state = {
        loaded: false,
        src: '',
        frames: [],
        currentFrame: 0,
        fps: 24,
        totalFrames: 0,
        width: 0,
        height: 0,
        masks: new Map(),
        tracking: false,
    };
    var _timelineEl = null;
    var _frameImage = null;
    var _frameLayer = null;
    var _maskOverlay = null;
    var _videoEl = null;
    var _extractCanvas = null;
    var _extractCtx = null;
    // ── Frame Extraction ──
    function loadVideo(file, ctx) {
        return new Promise(function (resolve, reject) {
            var url = URL.createObjectURL(file);
            _state.src = url;
            var video = document.createElement('video');
            video.muted = true;
            video.preload = 'auto';
            _videoEl = video;
            video.onloadedmetadata = function () {
                _state.width = video.videoWidth;
                _state.height = video.videoHeight;
                _state.fps = 24; // Default; could detect from metadata
                _state.totalFrames = Math.round(video.duration * _state.fps);
                _state.loaded = true;
                _state.currentFrame = 0;
                _state.frames = [];
                _state.masks = new Map();
                // Extract frames
                extractFrames(video, ctx).then(function () {
                    showTimeline(ctx);
                    goToFrame(0, ctx);
                    resolve();
                }).catch(reject);
            };
            video.onerror = function () {
                reject(new Error('Failed to load video'));
            };
            video.src = url;
        });
    }
    function extractFrames(video, ctx) {
        return new Promise(function (resolve) {
            if (!_extractCanvas) {
                _extractCanvas = document.createElement('canvas');
            }
            _extractCanvas.width = video.videoWidth;
            _extractCanvas.height = video.videoHeight;
            _extractCtx = _extractCanvas.getContext('2d');
            var frameInterval = 1 / _state.fps;
            var currentTime = 0;
            var frameIndex = 0;
            var maxFrames = Math.min(_state.totalFrames, 300); // Cap at 300 frames
            function extractNext() {
                if (frameIndex >= maxFrames || currentTime > video.duration) {
                    _state.totalFrames = _state.frames.length;
                    resolve();
                    return;
                }
                video.currentTime = currentTime;
            }
            video.onseeked = function () {
                _extractCtx.drawImage(video, 0, 0);
                var dataUrl = _extractCanvas.toDataURL('image/jpeg', 0.8); // JPEG for size
                _state.frames.push({
                    index: frameIndex,
                    dataUrl: dataUrl,
                    maskDataUrl: '',
                    width: video.videoWidth,
                    height: video.videoHeight,
                });
                frameIndex++;
                currentTime += frameInterval;
                extractNext();
            };
            extractNext();
        });
    }
    // ── Timeline UI ──
    function showTimeline(ctx) {
        if (_timelineEl)
            _timelineEl.remove();
        _timelineEl = document.createElement('div');
        _timelineEl.id = 'video-timeline';
        _timelineEl.className = 'video-timeline';
        var html = '<div class="vt-header">' +
            '<span class="vt-title">Video Timeline</span>' +
            '<span class="vt-info" id="vt-frame-info">Frame 0 / ' + _state.totalFrames + '</span>' +
            '<button class="vt-close" id="vt-close" title="Close video">&times;</button>' +
            '</div>' +
            '<div class="vt-strip" id="vt-strip"></div>' +
            '<div class="vt-controls">' +
            '<input type="range" class="vt-scrubber" id="vt-scrubber" min="0" max="' + (_state.totalFrames - 1) + '" value="0">' +
            '<div class="vt-buttons">' +
            '<button class="vt-btn" id="vt-prev" title="Previous frame">&larr;</button>' +
            '<button class="vt-btn" id="vt-play" title="Play/Pause">&#9654;</button>' +
            '<button class="vt-btn" id="vt-next" title="Next frame">&rarr;</button>' +
            '<span class="vt-sep">|</span>' +
            '<button class="vt-btn vt-track-btn" id="vt-track" title="Track object across frames">Track</button>' +
            '<button class="vt-btn" id="vt-clear-masks" title="Clear all masks">Clear Masks</button>' +
            '</div>' +
            '</div>';
        _timelineEl.innerHTML = html;
        var stageContainer = document.getElementById('canvas-stage-container');
        if (stageContainer && stageContainer.parentElement) {
            stageContainer.parentElement.appendChild(_timelineEl);
        }
        // Build frame strip (thumbnails)
        buildFrameStrip();
        bindTimelineEvents(ctx);
    }
    function buildFrameStrip() {
        var strip = document.getElementById('vt-strip');
        if (!strip)
            return;
        strip.innerHTML = '';
        // Show every Nth frame as thumbnail
        var step = Math.max(1, Math.floor(_state.totalFrames / 60));
        for (var i = 0; i < _state.totalFrames; i += step) {
            var frame = _state.frames[i];
            if (!frame)
                continue;
            var thumb = document.createElement('div');
            thumb.className = 'vt-thumb' + (i === _state.currentFrame ? ' active' : '');
            thumb.dataset.frame = String(i);
            thumb.style.backgroundImage = 'url(' + frame.dataUrl + ')';
            // Mask indicator
            if (_state.masks.has(i)) {
                thumb.classList.add('has-mask');
            }
            strip.appendChild(thumb);
        }
    }
    function bindTimelineEvents(ctx) {
        var scrubber = document.getElementById('vt-scrubber');
        if (scrubber) {
            scrubber.addEventListener('input', function () {
                goToFrame(parseInt(scrubber.value), ctx);
            });
        }
        var prevBtn = document.getElementById('vt-prev');
        var nextBtn = document.getElementById('vt-next');
        var playBtn = document.getElementById('vt-play');
        var trackBtn = document.getElementById('vt-track');
        var clearBtn = document.getElementById('vt-clear-masks');
        var closeBtn = document.getElementById('vt-close');
        if (prevBtn)
            prevBtn.addEventListener('click', function () {
                goToFrame(Math.max(0, _state.currentFrame - 1), ctx);
            });
        if (nextBtn)
            nextBtn.addEventListener('click', function () {
                goToFrame(Math.min(_state.totalFrames - 1, _state.currentFrame + 1), ctx);
            });
        var _playing = false;
        var _playTimer = null;
        if (playBtn)
            playBtn.addEventListener('click', function () {
                _playing = !_playing;
                playBtn.textContent = _playing ? '\u23F8' : '\u25B6';
                if (_playing) {
                    _playTimer = setInterval(function () {
                        var next = _state.currentFrame + 1;
                        if (next >= _state.totalFrames) {
                            next = 0;
                        }
                        goToFrame(next, ctx);
                    }, 1000 / _state.fps);
                }
                else {
                    if (_playTimer) {
                        clearInterval(_playTimer);
                        _playTimer = null;
                    }
                }
            });
        if (trackBtn)
            trackBtn.addEventListener('click', function () {
                trackObject(ctx);
            });
        if (clearBtn)
            clearBtn.addEventListener('click', function () {
                _state.masks.clear();
                removeMaskOverlay(ctx);
                buildFrameStrip();
            });
        if (closeBtn)
            closeBtn.addEventListener('click', function () {
                unloadVideo(ctx);
            });
        // Click on thumbnail
        var strip = document.getElementById('vt-strip');
        if (strip) {
            strip.addEventListener('click', function (e) {
                var thumb = e.target.closest('.vt-thumb');
                if (thumb && thumb.dataset.frame) {
                    goToFrame(parseInt(thumb.dataset.frame), ctx);
                }
            });
        }
    }
    // ── Frame Display ──
    function goToFrame(index, ctx) {
        if (index < 0 || index >= _state.frames.length)
            return;
        _state.currentFrame = index;
        var frame = _state.frames[index];
        if (!frame)
            return;
        // Update frame image on canvas
        var bb = ctx.boundingBox;
        var img = new window.Image();
        img.onload = function () {
            if (_frameImage)
                _frameImage.destroy();
            if (!_frameLayer) {
                _frameLayer = new Konva.Layer({ listening: false });
                ctx.stage.add(_frameLayer);
                // Keep below UI layer
                _frameLayer.moveToBottom();
                // But above background
            }
            _frameImage = new Konva.Image({
                image: img,
                x: bb.x(), y: bb.y(),
                width: bb.width(), height: bb.height(),
                listening: false,
            });
            _frameLayer.destroyChildren();
            _frameLayer.add(_frameImage);
            _frameLayer.batchDraw();
            // Show mask overlay if exists
            showMaskForFrame(index, ctx);
        };
        img.src = frame.dataUrl;
        // Update UI
        updateFrameInfo(index);
        var scrubber = document.getElementById('vt-scrubber');
        if (scrubber)
            scrubber.value = String(index);
        // Highlight active thumbnail
        var strip = document.getElementById('vt-strip');
        if (strip) {
            strip.querySelectorAll('.vt-thumb').forEach(function (t) {
                t.classList.toggle('active', t.dataset.frame === String(index));
            });
        }
    }
    function showMaskForFrame(index, ctx) {
        removeMaskOverlay(ctx);
        var maskUrl = _state.masks.get(index);
        if (!maskUrl)
            return;
        var bb = ctx.boundingBox;
        var img = new window.Image();
        img.onload = function () {
            _maskOverlay = new Konva.Image({
                image: img,
                x: bb.x(), y: bb.y(),
                width: bb.width(), height: bb.height(),
                opacity: 0.4,
                listening: false,
                name: 'video-mask-overlay',
            });
            ctx.uiLayer.add(_maskOverlay);
            ctx.uiLayer.batchDraw();
        };
        img.src = maskUrl;
    }
    function removeMaskOverlay(ctx) {
        if (_maskOverlay) {
            _maskOverlay.destroy();
            _maskOverlay = null;
            ctx.uiLayer.batchDraw();
        }
    }
    function updateFrameInfo(index) {
        var info = document.getElementById('vt-frame-info');
        if (info) {
            var timeStr = (index / _state.fps).toFixed(2) + 's';
            info.textContent = 'Frame ' + index + ' / ' + _state.totalFrames + '  (' + timeStr + ')';
        }
    }
    // ── SAM Video Tracking ──
    function trackObject(ctx) {
        if (_state.tracking || !_state.loaded)
            return;
        if (_state.frames.length === 0)
            return;
        // Get current SAM instances (masks from current frame)
        var samInstances = typeof CanvasSAM !== 'undefined' ? CanvasSAM.getInstances() : [];
        var clickPoints = [];
        // Also check if user has click points from SAM tool
        // We'll send the first frame + any segmentation hints to the backend
        _state.tracking = true;
        var trackBtn = document.getElementById('vt-track');
        if (trackBtn) {
            trackBtn.textContent = 'Tracking...';
            trackBtn.classList.add('tracking');
        }
        // Collect frame data — send first frame as image + masks/points
        var firstFrame = _state.frames[0];
        if (!firstFrame) {
            _state.tracking = false;
            return;
        }
        var apiBase = window.location.protocol + '//' + window.location.host;
        // Convert first frame dataUrl to blob
        var byteString = atob(firstFrame.dataUrl.split(',')[1]);
        var mimeString = firstFrame.dataUrl.split(',')[0].split(':')[1].split(';')[0];
        var ab = new ArrayBuffer(byteString.length);
        var ia = new Uint8Array(ab);
        for (var i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        var blob = new Blob([ab], { type: mimeString });
        var form = new FormData();
        form.append('video', blob, 'frame0.jpg');
        // If we have SAM instances, send their mask data as the prompt
        if (samInstances.length > 0) {
            form.append('prompt', samInstances[0].label || 'object');
        }
        // If user clicked points in SAM click mode, send those
        form.append('points', JSON.stringify(clickPoints));
        form.append('propagate', 'true');
        fetch(apiBase + '/canvas/sam3/video', {
            method: 'POST',
            body: form,
        })
            .then(function (res) { return res.json(); })
            .then(function (data) {
            _state.tracking = false;
            if (trackBtn) {
                trackBtn.textContent = 'Track';
                trackBtn.classList.remove('tracking');
            }
            if (data.error) {
                // Video tracking stub — populate with frame 0 mask for now
                console.warn('[VideoTrack]', data.error);
                // If we have a mask on frame 0 from SAM, propagate it as a static mask
                // to all frames (placeholder until real SAM 3 video tracking)
                if (samInstances.length > 0 && samInstances[0].mask_png) {
                    var maskUrl = 'data:image/png;base64,' + samInstances[0].mask_png;
                    for (var fi = 0; fi < _state.totalFrames; fi++) {
                        _state.masks.set(fi, maskUrl);
                    }
                    showMaskForFrame(_state.currentFrame, ctx);
                    buildFrameStrip();
                }
                return;
            }
            // Apply per-frame masks
            if (data.frames) {
                data.frames.forEach(function (f) {
                    if (f.mask_png) {
                        _state.masks.set(f.index, 'data:image/png;base64,' + f.mask_png);
                    }
                });
            }
            showMaskForFrame(_state.currentFrame, ctx);
            buildFrameStrip();
        })
            .catch(function (err) {
            _state.tracking = false;
            if (trackBtn) {
                trackBtn.textContent = 'Track';
                trackBtn.classList.remove('tracking');
            }
            console.error('[VideoTrack] Error:', err);
        });
    }
    // ── Set mask for current frame (from SAM results) ──
    function setMaskForCurrentFrame(maskDataUrl, ctx) {
        _state.masks.set(_state.currentFrame, maskDataUrl);
        showMaskForFrame(_state.currentFrame, ctx);
        buildFrameStrip();
    }
    // ── Get masks for compositor/inpainting ──
    function getAllMasks() {
        return _state.masks;
    }
    function getMasksAsArray() {
        var result = [];
        _state.masks.forEach(function (url, frame) {
            result.push({ frame: frame, maskDataUrl: url });
        });
        return result;
    }
    // ── Cleanup ──
    function unloadVideo(ctx) {
        if (_frameImage) {
            _frameImage.destroy();
            _frameImage = null;
        }
        if (_frameLayer) {
            _frameLayer.destroy();
            _frameLayer = null;
        }
        removeMaskOverlay(ctx);
        if (_timelineEl) {
            _timelineEl.remove();
            _timelineEl = null;
        }
        if (_videoEl) {
            _videoEl.src = '';
            _videoEl = null;
        }
        if (_state.src) {
            URL.revokeObjectURL(_state.src);
        }
        _state = {
            loaded: false, src: '', frames: [], currentFrame: 0,
            fps: 24, totalFrames: 0, width: 0, height: 0,
            masks: new Map(), tracking: false,
        };
    }
    // ── Public API ──
    function isLoaded() { return _state.loaded; }
    function getCurrentFrame() { return _state.currentFrame; }
    function getTotalFrames() { return _state.totalFrames; }
    function getFps() { return _state.fps; }
    function isTracking() { return _state.tracking; }
    function getFrameData(index) {
        return _state.frames[index] || null;
    }
    return {
        loadVideo: loadVideo,
        goToFrame: goToFrame,
        trackObject: trackObject,
        unloadVideo: unloadVideo,
        setMaskForCurrentFrame: setMaskForCurrentFrame,
        getAllMasks: getAllMasks,
        getMasksAsArray: getMasksAsArray,
        isLoaded: isLoaded,
        getCurrentFrame: getCurrentFrame,
        getTotalFrames: getTotalFrames,
        getFps: getFps,
        isTracking: isTracking,
        getFrameData: getFrameData,
    };
})();
//# sourceMappingURL=canvas-video.js.map