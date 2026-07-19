"use strict";
/**
 * Canvas Tools — SerenityFlow Canvas v2
 *
 * Each tool is a standalone object with activate/deactivate/handlers.
 * Tools operate on the active CanvasLayer via the CanvasToolContext.
 * All types are global (no modules) to match SF conventions.
 */
// ── Tool Definitions ──
var CanvasTools = (function () {
    'use strict';
    // Shared drawing state
    var _drawing = false;
    var _currentLine = null;
    var _drawScheduled = false;
    // Rect tool state
    var _rectStart = null;
    var _rectPreview = null;
    // Gradient tool state
    var _gradStart = null;
    var _gradPreview = null;
    // Move tool state
    var _moveTarget = null;
    var _moveStart = null;
    var _moveNodeStart = null;
    // Lasso tool state
    var _lassoPoints = [];
    var _lassoLine = null;
    var _lassoSelection = null;
    // Clone stamp state
    var _cloneSource = null;
    var _cloneOffset = null;
    var _cloneCrosshair = null;
    // Fill tool state
    var _fillThreshold = 32;
    // Brush opacity (separate from hardness)
    var _brushOpacity = 1;
    // Last stroke endpoint for shift+click straight line mode
    var _lastStrokeEnd = null;
    // Clipboard for copy/paste
    var _clipboard = null;
    function isDrawTool(name) {
        return name === 'brush' || name === 'eraser' || name === 'mask';
    }
    function getStrokeColor(toolName, brushColor) {
        if (toolName === 'mask')
            return 'rgba(239, 68, 68, 0.8)';
        if (toolName === 'eraser')
            return '#000';
        return brushColor;
    }
    function getCompositeOp(toolName) {
        return toolName === 'eraser' ? 'destination-out' : 'source-over';
    }
    function scheduleLayerDraw(layer) {
        if (!_drawScheduled) {
            _drawScheduled = true;
            requestAnimationFrame(function () {
                layer.batchDraw();
                _drawScheduled = false;
            });
        }
    }
    // ── Brush Tool ──
    var BrushTool = {
        name: 'brush',
        cursor: 'none',
        showsBrushCursor: true,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked)
                return;
            var t = al.data.type;
            if (t !== 'draw' && t !== 'mask' && t !== 'guidance')
                return;
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0)
                return;
            var pos = ctx.getRelativePointerPosition();
            // Shift+click: straight line from last stroke endpoint
            if (e.evt && e.evt.shiftKey && _lastStrokeEnd) {
                _drawing = true;
                _currentLine = new Konva.Line({
                    stroke: ctx.getBrushColor(),
                    strokeWidth: ctx.getBrushSize(),
                    globalCompositeOperation: 'source-over',
                    lineCap: 'round',
                    lineJoin: 'round',
                    tension: 0.3,
                    opacity: ctx.getBrushHardness() * _brushOpacity,
                    points: [_lastStrokeEnd.x, _lastStrokeEnd.y, pos.x, pos.y],
                    listening: false
                });
                var layer = ctx.getActiveKonvaLayer();
                if (layer) layer.add(_currentLine);
                _lastStrokeEnd = { x: pos.x, y: pos.y };
                ctx.pushHistoryGrouped();
                _drawing = false;
                _currentLine = null;
                return;
            }
            _drawing = true;
            _currentLine = new Konva.Line({
                stroke: ctx.getBrushColor(),
                strokeWidth: ctx.getBrushSize(),
                globalCompositeOperation: 'source-over',
                lineCap: 'round',
                lineJoin: 'round',
                tension: 0.3,
                opacity: ctx.getBrushHardness() * _brushOpacity,
                points: [pos.x, pos.y, pos.x, pos.y],
                listening: false
            });
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                layer.add(_currentLine);
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_currentLine)
                return;
            _currentLine.points(_currentLine.points().concat([pos.x, pos.y]));
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                scheduleLayerDraw(layer);
        },
        onMouseUp: function (ctx) {
            if (_drawing && _currentLine) {
                var pts = _currentLine.points();
                if (pts.length >= 2) {
                    _lastStrokeEnd = { x: pts[pts.length - 2], y: pts[pts.length - 1] };
                }
                ctx.pushHistoryGrouped();
            }
            _drawing = false;
            _currentLine = null;
        }
    };
    // ── Eraser Tool ──
    var EraserTool = {
        name: 'eraser',
        cursor: 'none',
        showsBrushCursor: true,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked)
                return;
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0)
                return;
            var pos = ctx.getRelativePointerPosition();
            // Shift+click: straight line from last stroke endpoint
            if (e.evt && e.evt.shiftKey && _lastStrokeEnd) {
                _drawing = true;
                _currentLine = new Konva.Line({
                    stroke: '#000',
                    strokeWidth: ctx.getBrushSize(),
                    globalCompositeOperation: 'destination-out',
                    lineCap: 'round',
                    lineJoin: 'round',
                    tension: 0.3,
                    opacity: 1,
                    points: [_lastStrokeEnd.x, _lastStrokeEnd.y, pos.x, pos.y],
                    listening: false
                });
                var layer = ctx.getActiveKonvaLayer();
                if (layer) layer.add(_currentLine);
                _lastStrokeEnd = { x: pos.x, y: pos.y };
                ctx.pushHistoryGrouped();
                _drawing = false;
                _currentLine = null;
                return;
            }
            _drawing = true;
            _currentLine = new Konva.Line({
                stroke: '#000',
                strokeWidth: ctx.getBrushSize(),
                globalCompositeOperation: 'destination-out',
                lineCap: 'round',
                lineJoin: 'round',
                tension: 0.3,
                opacity: 1,
                points: [pos.x, pos.y, pos.x, pos.y],
                listening: false
            });
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                layer.add(_currentLine);
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_currentLine)
                return;
            _currentLine.points(_currentLine.points().concat([pos.x, pos.y]));
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                scheduleLayerDraw(layer);
        },
        onMouseUp: function (ctx) {
            if (_drawing && _currentLine) {
                var pts = _currentLine.points();
                if (pts.length >= 2) {
                    _lastStrokeEnd = { x: pts[pts.length - 2], y: pts[pts.length - 1] };
                }
                ctx.pushHistoryGrouped();
            }
            _drawing = false;
            _currentLine = null;
        }
    };
    // ── Mask Tool (brush variant for mask layers) ──
    var MaskTool = {
        name: 'mask',
        cursor: 'none',
        showsBrushCursor: true,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked)
                return;
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0)
                return;
            _drawing = true;
            var pos = ctx.getRelativePointerPosition();
            _currentLine = new Konva.Line({
                stroke: 'rgba(239, 68, 68, 0.8)',
                strokeWidth: ctx.getBrushSize(),
                globalCompositeOperation: 'source-over',
                lineCap: 'round',
                lineJoin: 'round',
                tension: 0.3,
                opacity: ctx.getBrushHardness(),
                points: [pos.x, pos.y, pos.x, pos.y],
                listening: false
            });
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                layer.add(_currentLine);
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_currentLine)
                return;
            _currentLine.points(_currentLine.points().concat([pos.x, pos.y]));
            var layer = ctx.getActiveKonvaLayer();
            if (layer)
                scheduleLayerDraw(layer);
        },
        onMouseUp: function (ctx) {
            if (_drawing && _currentLine)
                ctx.pushHistoryGrouped();
            _drawing = false;
            _currentLine = null;
        }
    };
    // ── Rect Tool ──
    var RectTool = {
        name: 'rect',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked)
                return;
            if (al.data.type !== 'draw' && al.data.type !== 'mask')
                return;
            _rectStart = ctx.getRelativePointerPosition();
            _rectPreview = new Konva.Rect({
                x: _rectStart.x, y: _rectStart.y,
                width: 0, height: 0,
                fill: al.data.type === 'mask' ? 'rgba(239, 68, 68, 0.5)' : ctx.getBrushColor(),
                opacity: _brushOpacity,
                stroke: 'rgba(255,255,255,0.3)',
                strokeWidth: 1,
                dash: [4, 4],
                listening: false
            });
            ctx.uiLayer.add(_rectPreview);
        },
        onMouseMove: function (ctx, pos) {
            if (!_rectStart || !_rectPreview)
                return;
            var x = Math.min(_rectStart.x, pos.x);
            var y = Math.min(_rectStart.y, pos.y);
            var w = Math.abs(pos.x - _rectStart.x);
            var h = Math.abs(pos.y - _rectStart.y);
            _rectPreview.x(x);
            _rectPreview.y(y);
            _rectPreview.width(w);
            _rectPreview.height(h);
            ctx.uiLayer.batchDraw();
        },
        onMouseUp: function (ctx) {
            if (!_rectStart || !_rectPreview)
                return;
            var al = ctx.getActiveLayer();
            if (al && _rectPreview.width() > 2 && _rectPreview.height() > 2) {
                var committed = new Konva.Rect({
                    x: _rectPreview.x(), y: _rectPreview.y(),
                    width: _rectPreview.width(), height: _rectPreview.height(),
                    fill: _rectPreview.fill(),
                    opacity: _rectPreview.opacity(),
                    listening: false
                });
                al.konvaLayer.add(committed);
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            }
            _rectPreview.destroy();
            _rectPreview = null;
            _rectStart = null;
            ctx.uiLayer.batchDraw();
        }
    };
    // ── Gradient Tool ──
    var GradientTool = {
        name: 'gradient',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked || al.data.type !== 'draw')
                return;
            _gradStart = ctx.getRelativePointerPosition();
            _gradPreview = new Konva.Line({
                points: [_gradStart.x, _gradStart.y, _gradStart.x, _gradStart.y],
                stroke: '#fff',
                strokeWidth: 2,
                dash: [6, 3],
                listening: false
            });
            ctx.uiLayer.add(_gradPreview);
        },
        onMouseMove: function (ctx, pos) {
            if (!_gradStart || !_gradPreview)
                return;
            _gradPreview.points([_gradStart.x, _gradStart.y, pos.x, pos.y]);
            ctx.uiLayer.batchDraw();
        },
        onMouseUp: function (ctx) {
            if (!_gradStart || !_gradPreview)
                return;
            var al = ctx.getActiveLayer();
            var pts = _gradPreview.points();
            _gradPreview.destroy();
            _gradPreview = null;
            ctx.uiLayer.batchDraw();
            if (!al) {
                _gradStart = null;
                return;
            }
            var x0 = pts[0], y0 = pts[1], x1 = pts[2], y1 = pts[3];
            var dist = Math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            if (dist < 5) {
                _gradStart = null;
                return;
            }
            // Draw gradient onto active layer using an offscreen canvas
            var bb = ctx.boundingBox;
            var bx = bb.x(), by = bb.y(), bw = bb.width(), bh = bb.height();
            var tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = bw;
            tmpCanvas.height = bh;
            var gc = tmpCanvas.getContext('2d');
            var grad = gc.createLinearGradient(x0 - bx, y0 - by, x1 - bx, y1 - by);
            grad.addColorStop(0, ctx.getBrushColor());
            grad.addColorStop(1, 'transparent');
            gc.fillStyle = grad;
            gc.fillRect(0, 0, bw, bh);
            var img = new Image();
            img.onload = function () {
                var kImg = new Konva.Image({ image: img, x: bx, y: by, width: bw, height: bh, opacity: _brushOpacity, listening: false });
                al.konvaLayer.add(kImg);
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            };
            img.src = tmpCanvas.toDataURL();
            _gradStart = null;
        }
    };
    // ── Move Tool ──
    var MoveTool = {
        name: 'move',
        cursor: 'default',
        showsBrushCursor: false,
        onActivate: function (ctx) {
            // Make images draggable
            // This is handled in setTool now
        },
        onMouseDown: function (ctx, e) {
            var target = e.target;
            if (!target || target === ctx.stage)
                return;
            if (target.name() === 'bounding-box' || target.name().indexOf('handle-') === 0)
                return;
            if (target instanceof Konva.Image || target instanceof Konva.Text) {
                _moveTarget = target;
                var pos = ctx.getRelativePointerPosition();
                _moveStart = pos;
                _moveNodeStart = { x: target.x(), y: target.y() };
            }
        },
        onMouseMove: function (ctx, pos) {
            if (!_moveTarget || !_moveStart || !_moveNodeStart)
                return;
            var dx = pos.x - _moveStart.x;
            var dy = pos.y - _moveStart.y;
            _moveTarget.x(_moveNodeStart.x + dx);
            _moveTarget.y(_moveNodeStart.y + dy);
            var parent = _moveTarget.getLayer();
            if (parent)
                parent.batchDraw();
        },
        onMouseUp: function (ctx) {
            if (_moveTarget)
                ctx.pushHistory();
            _moveTarget = null;
            _moveStart = null;
            _moveNodeStart = null;
        },
        onKeyDown: function (ctx, e) {
            // Arrow keys nudge
            var al = ctx.getActiveLayer();
            if (!al)
                return;
            var step = e.shiftKey ? 10 : 1;
            var moved = false;
            al.konvaLayer.getChildren().forEach(function (child) {
                switch (e.code) {
                    case 'ArrowUp':
                        child.y(child.y() - step);
                        moved = true;
                        break;
                    case 'ArrowDown':
                        child.y(child.y() + step);
                        moved = true;
                        break;
                    case 'ArrowLeft':
                        child.x(child.x() - step);
                        moved = true;
                        break;
                    case 'ArrowRight':
                        child.x(child.x() + step);
                        moved = true;
                        break;
                }
            });
            if (moved) {
                e.preventDefault();
                al.konvaLayer.batchDraw();
                ctx.pushHistory();
            }
            if (e.code === 'Delete' || e.code === 'Backspace') {
                ctx.deleteActiveLayer();
            }
        }
    };
    // ── Bbox Tool (select/manipulate bounding box) ──
    var BboxTool = {
        name: 'select',
        cursor: 'default',
        showsBrushCursor: false,
        // Bbox manipulation is handled by the existing handle system in canvas-tab.
        // This tool just enables the handles and default cursor.
        onActivate: function (ctx) {
            // Make images draggable in select mode
        }
    };
    // ── Color Picker Tool ──
    var ColorPickerTool = {
        name: 'colorpicker',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, _e) {
            var pos = ctx.getRelativePointerPosition();
            // Sample from the composite canvas
            var dataUrl = ctx.stage.toDataURL({
                x: pos.x, y: pos.y,
                width: 1, height: 1,
                pixelRatio: 1
            });
            var tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = 1;
            tmpCanvas.height = 1;
            var tctx = tmpCanvas.getContext('2d');
            var img = new Image();
            img.onload = function () {
                tctx.drawImage(img, 0, 0);
                var pixel = tctx.getImageData(0, 0, 1, 1).data;
                var hex = '#' + ((1 << 24) + (pixel[0] << 16) + (pixel[1] << 8) + pixel[2]).toString(16).slice(1);
                ctx.setBrushColor(hex);
                // Switch back to brush after picking
                ctx.setActiveTool('brush');
            };
            img.src = dataUrl;
        }
    };
    // ── Text Tool ──
    var TextTool = {
        name: 'text',
        cursor: 'text',
        showsBrushCursor: false,
        onMouseDown: function (ctx, e) {
            // Only block handle clicks, allow clicks on bounding box for text placement
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0)
                return;
            var pos = ctx.getRelativePointerPosition();
            var al = ctx.getActiveLayer();
            // If active layer is a text layer, update position of text
            if (al && al.data.type === 'text') {
                var kText = al.konvaLayer.findOne('Text');
                if (kText) {
                    kText.x(pos.x);
                    kText.y(pos.y);
                    al.konvaLayer.batchDraw();
                    ctx.pushHistory();
                }
                return;
            }
            // Otherwise create new text layer
            var newLayer = ctx.addLayer('Text ' + Date.now().toString(36), 'text');
            var td = newLayer.data;
            td.position = { x: pos.x, y: pos.y };
            var kTextNew = newLayer.konvaLayer.findOne('Text');
            if (kTextNew) {
                kTextNew.x(pos.x);
                kTextNew.y(pos.y);
                newLayer.konvaLayer.batchDraw();
                // Auto-trigger inline edit so user can type immediately
                kTextNew.fire('dblclick');
            }
        }
    };
    // ── Fill Tool ──
    var FillTool = {
        name: 'fill',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, _e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked)
                return;
            if (al.data.type !== 'draw' && al.data.type !== 'mask')
                return;
            var pos = ctx.getRelativePointerPosition();
            var bb = ctx.boundingBox;
            var bx = bb.x(), by = bb.y(), bw = Math.round(bb.width()), bh = Math.round(bb.height());
            // Get layer content as ImageData
            var layerUrl = al.konvaLayer.toDataURL({ x: bx, y: by, width: bw, height: bh, pixelRatio: 1 });
            var tmpImg = new Image();
            tmpImg.onload = function () {
                var tmpCanvas = document.createElement('canvas');
                tmpCanvas.width = bw;
                tmpCanvas.height = bh;
                var fctx = tmpCanvas.getContext('2d');
                fctx.drawImage(tmpImg, 0, 0);
                var imageData = fctx.getImageData(0, 0, bw, bh);
                var px = Math.round(pos.x - bx);
                var py = Math.round(pos.y - by);
                if (px < 0 || px >= bw || py < 0 || py >= bh)
                    return;
                var fillColor = al.data.type === 'mask' ? [239, 68, 68, 128] : hexToRgba(ctx.getBrushColor(), Math.round(_brushOpacity * 255));
                floodFill(imageData, px, py, fillColor, _fillThreshold);
                fctx.putImageData(imageData, 0, 0);
                var resultImg = new Image();
                resultImg.onload = function () {
                    al.konvaLayer.destroyChildren();
                    al.konvaLayer.add(new Konva.Image({ image: resultImg, x: bx, y: by, width: bw, height: bh, listening: false }));
                    al.konvaLayer.batchDraw();
                    ctx.pushHistory();
                };
                resultImg.src = tmpCanvas.toDataURL();
            };
            tmpImg.src = layerUrl;
        }
    };
    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16) || 0;
        var g = parseInt(hex.slice(3, 5), 16) || 0;
        var b = parseInt(hex.slice(5, 7), 16) || 0;
        return [r, g, b, alpha];
    }
    function floodFill(imageData, startX, startY, fillColor, threshold) {
        var data = imageData.data;
        var w = imageData.width;
        var h = imageData.height;
        var idx = (startY * w + startX) * 4;
        var targetR = data[idx], targetG = data[idx + 1], targetB = data[idx + 2], targetA = data[idx + 3];
        // Don't fill if target is already the fill color
        if (targetR === fillColor[0] && targetG === fillColor[1] && targetB === fillColor[2] && Math.abs(targetA - fillColor[3]) < 5)
            return;
        var stack = [startX, startY];
        var visited = new Uint8Array(w * h);
        while (stack.length > 0) {
            var y = stack.pop();
            var x = stack.pop();
            var i = y * w + x;
            if (x < 0 || x >= w || y < 0 || y >= h || visited[i])
                continue;
            var pi = i * 4;
            var dr = Math.abs(data[pi] - targetR);
            var dg = Math.abs(data[pi + 1] - targetG);
            var db = Math.abs(data[pi + 2] - targetB);
            var da = Math.abs(data[pi + 3] - targetA);
            if (dr + dg + db + da > threshold * 4)
                continue;
            visited[i] = 1;
            data[pi] = fillColor[0];
            data[pi + 1] = fillColor[1];
            data[pi + 2] = fillColor[2];
            data[pi + 3] = fillColor[3];
            stack.push(x + 1, y, x - 1, y, x, y + 1, x, y - 1);
        }
    }
    // ── Lasso Tool ──
    var LassoTool = {
        name: 'lasso',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, _e) {
            _lassoPoints = [];
            var pos = ctx.getRelativePointerPosition();
            _lassoPoints.push(pos.x, pos.y);
            if (_lassoLine) {
                _lassoLine.destroy();
                _lassoLine = null;
            }
            _lassoLine = new Konva.Line({
                points: _lassoPoints,
                stroke: '#6c6af5',
                strokeWidth: 2 / ctx.stage.scaleX(),
                dash: [6, 3],
                closed: false,
                listening: false
            });
            ctx.uiLayer.add(_lassoLine);
            _drawing = true;
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_lassoLine)
                return;
            _lassoPoints.push(pos.x, pos.y);
            _lassoLine.points(_lassoPoints);
            ctx.uiLayer.batchDraw();
        },
        onMouseUp: function (ctx) {
            _drawing = false;
            if (_lassoLine && _lassoPoints.length >= 6) {
                _lassoLine.closed(true);
                _lassoLine.fill('rgba(108, 106, 245, 0.15)');
                ctx.uiLayer.batchDraw();
                _lassoSelection = _lassoPoints.slice();
            }
            else {
                if (_lassoLine) {
                    _lassoLine.destroy();
                    _lassoLine = null;
                }
                _lassoSelection = null;
                ctx.uiLayer.batchDraw();
            }
        },
        onDeactivate: function (ctx) {
            if (_lassoLine) {
                _lassoLine.destroy();
                _lassoLine = null;
            }
            _lassoSelection = null;
            _lassoPoints = [];
            ctx.uiLayer.batchDraw();
        }
    };
    // ── Clone Stamp Tool ──
    var CloneStampTool = {
        name: 'clonestamp',
        cursor: 'none',
        showsBrushCursor: true,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked || al.data.type !== 'draw')
                return;
            var pos = ctx.getRelativePointerPosition();
            // Alt+click sets source
            if (e.evt && e.evt.altKey) {
                _cloneSource = { x: pos.x, y: pos.y };
                _cloneOffset = null;
                // Show crosshair at source
                if (_cloneCrosshair)
                    _cloneCrosshair.destroy();
                _cloneCrosshair = new Konva.Group({ listening: false });
                _cloneCrosshair.add(new Konva.Line({
                    points: [pos.x - 10, pos.y, pos.x + 10, pos.y],
                    stroke: '#ff0', strokeWidth: 1.5 / ctx.stage.scaleX(), listening: false
                }));
                _cloneCrosshair.add(new Konva.Line({
                    points: [pos.x, pos.y - 10, pos.x, pos.y + 10],
                    stroke: '#ff0', strokeWidth: 1.5 / ctx.stage.scaleX(), listening: false
                }));
                ctx.uiLayer.add(_cloneCrosshair);
                ctx.uiLayer.batchDraw();
                return;
            }
            if (!_cloneSource) {
                // Show hint
                if (!document.getElementById('cv-clone-hint')) {
                    var hint = document.createElement('div');
                    hint.id = 'cv-clone-hint';
                    hint.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.8);color:#fff;padding:10px 18px;border-radius:8px;font-size:13px;z-index:999;pointer-events:none;';
                    hint.textContent = 'Alt + Click to set clone source point';
                    document.body.appendChild(hint);
                    setTimeout(function () { hint.remove(); }, 2000);
                }
                return;
            }
            if (!_cloneOffset) {
                _cloneOffset = { x: pos.x - _cloneSource.x, y: pos.y - _cloneSource.y };
            }
            _drawing = true;
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_cloneSource || !_cloneOffset)
                return;
            // Sample source region and paint at current position
            var al = ctx.getActiveLayer();
            if (!al)
                return;
            var srcX = pos.x - _cloneOffset.x;
            var srcY = pos.y - _cloneOffset.y;
            var size = ctx.getBrushSize();
            var half = size / 2;
            // Get source pixel data from layer
            try {
                var srcUrl = al.konvaLayer.toDataURL({
                    x: srcX - half, y: srcY - half,
                    width: size, height: size,
                    pixelRatio: 1
                });
                var stampImg = new Image();
                stampImg.onload = function () {
                    var stamp = new Konva.Image({
                        image: stampImg,
                        x: pos.x - half, y: pos.y - half,
                        width: size, height: size,
                        listening: false
                    });
                    al.konvaLayer.add(stamp);
                    scheduleLayerDraw(al.konvaLayer);
                };
                stampImg.src = srcUrl;
            }
            catch (_) { /* ignore cross-origin */ }
        },
        onMouseUp: function (ctx) {
            if (_drawing)
                ctx.pushHistoryGrouped();
            _drawing = false;
        },
        onDeactivate: function (ctx) {
            if (_cloneCrosshair) {
                _cloneCrosshair.destroy();
                _cloneCrosshair = null;
            }
            _cloneSource = null;
            _cloneOffset = null;
            ctx.uiLayer.batchDraw();
        }
    };
    // ── Pencil Tool (hard-edged, 1px, no smoothing) ──
    var PencilTool = {
        name: 'pencil',
        cursor: 'none',
        showsBrushCursor: true,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked) return;
            var t = al.data.type;
            if (t !== 'draw' && t !== 'mask' && t !== 'guidance') return;
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0) return;
            var pos = ctx.getRelativePointerPosition();
            // Use mask color on mask layers, brush color otherwise
            var colorKey = t === 'mask' ? 'mask' : 'pencil';
            // Shift+click: straight line
            if (e.evt && e.evt.shiftKey && _lastStrokeEnd) {
                _currentLine = new Konva.Line({
                    stroke: getStrokeColor(colorKey, ctx.getBrushColor()),
                    strokeWidth: 1,
                    globalCompositeOperation: 'source-over',
                    lineCap: 'butt', lineJoin: 'miter',
                    tension: 0,
                    opacity: 1,
                    points: [_lastStrokeEnd.x, _lastStrokeEnd.y, pos.x, pos.y],
                    listening: false
                });
                var layer = ctx.getActiveKonvaLayer();
                if (layer) layer.add(_currentLine);
                _lastStrokeEnd = { x: pos.x, y: pos.y };
                ctx.pushHistoryGrouped();
                _currentLine = null;
                return;
            }
            _drawing = true;
            _currentLine = new Konva.Line({
                stroke: getStrokeColor(colorKey, ctx.getBrushColor()),
                strokeWidth: 1,
                globalCompositeOperation: 'source-over',
                lineCap: 'butt', lineJoin: 'miter',
                tension: 0,
                opacity: 1,
                points: [pos.x, pos.y, pos.x, pos.y],
                listening: false
            });
            var layer = ctx.getActiveKonvaLayer();
            if (layer) layer.add(_currentLine);
        },
        onMouseMove: function (ctx, pos) {
            if (!_drawing || !_currentLine) return;
            _currentLine.points(_currentLine.points().concat([pos.x, pos.y]));
            var layer = ctx.getActiveKonvaLayer();
            if (layer) scheduleLayerDraw(layer);
        },
        onMouseUp: function (ctx) {
            if (_drawing && _currentLine) {
                var pts = _currentLine.points();
                if (pts.length >= 2) {
                    _lastStrokeEnd = { x: pts[pts.length - 2], y: pts[pts.length - 1] };
                }
                ctx.pushHistoryGrouped();
            }
            _drawing = false;
            _currentLine = null;
        }
    };
    // ── Speech Bubble Tool ──
    // Renders as flat raster image to survive undo/redo (history uses toDataURL snapshots).
    // Text entered via prompt() before placement.
    var SpeechBubbleTool = {
        name: 'speechbubble',
        cursor: 'crosshair',
        showsBrushCursor: false,
        onMouseDown: function (ctx, e) {
            var al = ctx.getActiveLayer();
            if (!al || al.data.locked) return;
            var t = al.data.type;
            if (t !== 'draw' && t !== 'text') return;
            if (e.target && e.target.name() && e.target.name().indexOf('handle-') === 0) return;

            var text = prompt('Speech bubble text:', 'Hello!');
            if (!text) return;

            var pos = ctx.getRelativePointerPosition();
            var bubbleW = 180;
            var bubbleH = 80;
            var tailH = 20;
            var radius = 12;

            // Render to offscreen canvas then place as Konva.Image
            var offCanvas = document.createElement('canvas');
            offCanvas.width = bubbleW;
            offCanvas.height = bubbleH + tailH;
            var c = offCanvas.getContext('2d');

            // Draw bubble path
            c.fillStyle = '#fff';
            c.strokeStyle = '#333';
            c.lineWidth = 2;
            c.beginPath();
            c.moveTo(radius, 0);
            c.lineTo(bubbleW - radius, 0);
            c.quadraticCurveTo(bubbleW, 0, bubbleW, radius);
            c.lineTo(bubbleW, bubbleH - radius);
            c.quadraticCurveTo(bubbleW, bubbleH, bubbleW - radius, bubbleH);
            c.lineTo(50, bubbleH);
            c.lineTo(30, bubbleH + tailH);
            c.lineTo(35, bubbleH);
            c.lineTo(radius, bubbleH);
            c.quadraticCurveTo(0, bubbleH, 0, bubbleH - radius);
            c.lineTo(0, radius);
            c.quadraticCurveTo(0, 0, radius, 0);
            c.closePath();
            c.fill();
            c.stroke();

            // Draw text
            c.fillStyle = '#333';
            c.font = '14px sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            // Word-wrap text
            var words = text.split(' ');
            var lines = [];
            var line = '';
            var maxW = bubbleW - 24;
            for (var wi = 0; wi < words.length; wi++) {
                var test = line + (line ? ' ' : '') + words[wi];
                if (c.measureText(test).width > maxW && line) {
                    lines.push(line);
                    line = words[wi];
                } else {
                    line = test;
                }
            }
            if (line) lines.push(line);
            var lineH = 18;
            var startY = bubbleH / 2 - (lines.length - 1) * lineH / 2;
            for (var li = 0; li < lines.length; li++) {
                c.fillText(lines[li], bubbleW / 2, startY + li * lineH);
            }

            // Place as Konva.Image (survives undo/redo)
            var img = new Image();
            img.onload = function () {
                var kImg = new Konva.Image({
                    image: img,
                    x: pos.x - bubbleW / 2,
                    y: pos.y - bubbleH / 2,
                    width: bubbleW,
                    height: bubbleH + tailH,
                    listening: false
                });
                var layer = ctx.getActiveKonvaLayer();
                if (layer) {
                    layer.add(kImg);
                    layer.batchDraw();
                    ctx.pushHistory();
                }
            };
            img.src = offCanvas.toDataURL();
        }
    };
    // ── Pan Tool ──
    var PanTool = {
        name: 'pan',
        cursor: 'grab',
        showsBrushCursor: false,
        // Pan is handled by the existing pan logic in canvas-tab
    };
    // ── Tool Registry ──
    // SAM tool registered dynamically after canvas-sam.ts loads
    var _tools = {
        select: BboxTool,
        brush: BrushTool,
        eraser: EraserTool,
        mask: MaskTool,
        rect: RectTool,
        gradient: GradientTool,
        move: MoveTool,
        colorpicker: ColorPickerTool,
        text: TextTool,
        fill: FillTool,
        lasso: LassoTool,
        clonestamp: CloneStampTool,
        pencil: PencilTool,
        speechbubble: SpeechBubbleTool,
        pan: PanTool,
    };
    function registerTool(name, tool) {
        _tools[name] = tool;
    }
    function get(name) {
        return _tools[name] || null;
    }
    function all() {
        return _tools;
    }
    function isDrawingActive() {
        return _drawing;
    }
    function getLassoSelection() {
        return _lassoSelection;
    }
    function clearLassoSelection(ctx) {
        if (_lassoLine) {
            _lassoLine.destroy();
            _lassoLine = null;
        }
        _lassoSelection = null;
        _lassoPoints = [];
        ctx.uiLayer.batchDraw();
    }
    function convertLassoToMask(ctx) {
        if (!_lassoSelection || _lassoSelection.length < 6)
            return;
        var maskLayer = ctx.addLayer('Mask from Lasso', 'mask');
        var poly = new Konva.Line({
            points: _lassoSelection,
            fill: 'rgba(239, 68, 68, 0.5)',
            closed: true,
            listening: false
        });
        maskLayer.konvaLayer.add(poly);
        maskLayer.konvaLayer.batchDraw();
        clearLassoSelection(ctx);
        ctx.pushHistory();
    }
    function getFillThreshold() { return _fillThreshold; }
    function setFillThreshold(t) { _fillThreshold = Math.max(0, Math.min(255, t)); }
    function getBrushOpacity() { return _brushOpacity; }
    function setBrushOpacity(o) { _brushOpacity = Math.max(0, Math.min(1, o)); }
    function getClipboard() { return _clipboard; }
    function setClipboard(data) { _clipboard = data; }
    return {
        get: get,
        all: all,
        registerTool: registerTool,
        isDrawingActive: isDrawingActive,
        getLassoSelection: getLassoSelection,
        clearLassoSelection: clearLassoSelection,
        convertLassoToMask: convertLassoToMask,
        getFillThreshold: getFillThreshold,
        setFillThreshold: setFillThreshold,
        getBrushOpacity: getBrushOpacity,
        setBrushOpacity: setBrushOpacity,
        getClipboard: getClipboard,
        setClipboard: setClipboard,
    };
})();
//# sourceMappingURL=canvas-tools.js.map