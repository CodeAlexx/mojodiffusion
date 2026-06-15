/* tools_brush.js — module 'toolsBrush' (Design B).
   Owns: the vertical tool dock (#tooldock) + brush/eraser controls in #canvas-toolbar,
   and freehand painting onto the ACTIVE layer's Konva.Group via Konva.Line.
   Talks to other modules ONLY through state/get/set + bus + the shared Konva stage.

   Decoupling notes (30-agent contract):
   - canvasCore owns the Stage (state.canvas.stage / bus 'canvas:stage'). We wait for it.
   - layers owns the layer model (state.layers / state.activeLayerId). We DON'T own it; we
     resolve the active layer's drawing Group cooperatively:
       1. bus call 'layer:getGroup' (request/response via a one-shot reply) — if layers
          provides it, we use that exact Group (lines land in the right layer);
       2. else a Group we stash on the stage keyed by layer id (so re-selecting a layer
          re-uses its strokes);
     and if there is NO active layer at all we paint onto our own fallback paint layer so
     the brush is always functional even before the layers module ships.
   - Mask layers paint red semi-transparent; eraser uses globalCompositeOperation
     'destination-out'. We infer "mask vs raster" from the active layer's type in state. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") { return; }

  S.register("toolsBrush", { init: init });

  // ---- tool catalog (icon = inline SVG path data; title; cursor) -------------
  var TOOLS = [
    { id: "move",     title: "Move (V)",       key: "v", cursor: "default" },
    { id: "brush",    title: "Brush (B)",      key: "b", cursor: "crosshair" },
    { id: "eraser",   title: "Eraser (E)",     key: "e", cursor: "crosshair" },
    { id: "rect",     title: "Rectangle (R)",  key: "r", cursor: "crosshair" },
    { id: "bbox",     title: "Gen region (G)", key: "g", cursor: "crosshair" },
    { id: "colorpick",title: "Color pick (I)", key: "i", cursor: "crosshair" },
    { id: "pan",      title: "Pan (H)",        key: "h", cursor: "grab" },
  ];

  // minimal monochrome 24x24 icons (stroke=currentColor)
  var ICONS = {
    move:     '<path d="M12 2v20M2 12h20M12 2l-3 3M12 2l3 3M12 22l-3-3M12 22l3-3M2 12l3-3M2 12l3 3M22 12l-3-3M22 12l-3 3"/>',
    brush:    '<path d="M3 21c2 0 3-1 3-3 0-1.5 1.5-3 3-3l8-8a2.8 2.8 0 0 0-4-4l-8 8c0 1.5-1.5 3-3 3-2 0-3 1-3 3z"/>',
    eraser:   '<path d="M4 16l8-8 6 6-5 5H7zM3 21h18"/>',
    rect:     '<rect x="4" y="6" width="16" height="12" rx="1"/>',
    bbox:     '<rect x="4" y="6" width="16" height="12" rx="1" stroke-dasharray="3 2"/><path d="M4 6h3M17 6h3M4 18h3M17 18h3"/>',
    colorpick:'<path d="M19 3a2 2 0 0 1 2 2l-9 9-3 1 1-3z"/><path d="M11 11l-7 7v2h2l7-7"/>',
    pan:      '<path d="M9 11V5a1.5 1.5 0 0 1 3 0v6m0-1V4a1.5 1.5 0 0 1 3 0v7m0-1V6a1.5 1.5 0 0 1 3 0v8a6 6 0 0 1-6 6h-2a5 5 0 0 1-4-2l-3-4a1.7 1.7 0 0 1 2.5-2.2L9 14"/>',
  };

  function svgIcon(name) {
    return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" ' +
           'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" ' +
           'stroke-linejoin="round">' + (ICONS[name] || "") + "</svg>";
  }

  function init(ctx) {
    var state = ctx.state, get = ctx.get, set = ctx.set, bus = ctx.bus, dom = ctx.dom;
    var Konva = ctx.Konva || window.Konva;

    injectCss();

    // ----- module-local brush settings (mirrored into a state namespace for others) ---
    var brush = {
      size: 32,
      color: "#ffffff",
      maskColor: "#ff3b3b",
      opacity: 1.0,
    };
    // expose for other modules / restore (best-effort; not in the core schema)
    try { state.tools = state.tools || {}; state.tools.brush = brush; } catch (_) {}

    // ----- runtime drawing state ----
    var stage = null;
    var paintLayer = null;        // our fallback Konva.Layer (only used if no layer group)
    var groupCache = {};          // layerId -> Konva.Group (our own stash)
    var drawing = false;
    var curLine = null;

    // -------------------------------------------------------------------------
    // 1) TOOL DOCK  (#tooldock)
    // -------------------------------------------------------------------------
    buildToolDock();
    function buildToolDock() {
      var host = dom.tooldock;
      if (!host) return;
      var wrap = document.createElement("div");
      wrap.id = "tb-dock";
      TOOLS.forEach(function (t) {
        var b = document.createElement("button");
        b.className = "tb-tool";
        b.dataset.tool = t.id;
        b.title = t.title;
        b.setAttribute("aria-label", t.title);
        b.innerHTML = svgIcon(t.id);
        b.addEventListener("click", function () { selectTool(t.id); });
        wrap.appendChild(b);
      });
      host.appendChild(wrap);
      reflectActiveTool();
    }

    // -------------------------------------------------------------------------
    // 2) CONTEXTUAL TOOLBAR  (#canvas-toolbar) — brush size + color
    // -------------------------------------------------------------------------
    buildToolbar();
    function buildToolbar() {
      var host = dom.canvasToolbar;
      if (!host) return;
      var bar = document.createElement("div");
      bar.id = "tb-toolbar";
      bar.innerHTML =
        '<span class="tb-tb-label" id="tb-tool-name">Move</span>' +
        '<span class="tb-sep"></span>' +
        '<label class="tb-ctl tb-brush-only">size' +
          '<input id="tb-size" type="range" min="1" max="200" step="1" value="' + brush.size + '">' +
          '<span id="tb-size-val">' + brush.size + '</span>' +
        '</label>' +
        '<label class="tb-ctl tb-brush-only">color' +
          '<input id="tb-color" type="color" value="' + brush.color + '">' +
        '</label>' +
        '<label class="tb-ctl tb-brush-only">flow' +
          '<input id="tb-opacity" type="range" min="5" max="100" step="5" value="' + Math.round(brush.opacity * 100) + '">' +
        '</label>';
      host.appendChild(bar);

      var size = bar.querySelector("#tb-size");
      var sizeVal = bar.querySelector("#tb-size-val");
      var color = bar.querySelector("#tb-color");
      var opacity = bar.querySelector("#tb-opacity");

      size.addEventListener("input", function () {
        brush.size = parseInt(size.value, 10) || 1;
        sizeVal.textContent = brush.size;
        try { bus.emit("brush:size", brush.size); } catch (_) {}
      });
      color.addEventListener("input", function () {
        brush.color = color.value;
        try { bus.emit("brush:color", brush.color); } catch (_) {}
      });
      opacity.addEventListener("input", function () {
        brush.opacity = (parseInt(opacity.value, 10) || 100) / 100;
        try { bus.emit("brush:opacity", brush.opacity); } catch (_) {}
      });

      reflectToolbar();
    }

    // -------------------------------------------------------------------------
    // 3) TOOL SELECTION  (writes state.ui.activeTool; everyone reads it)
    // -------------------------------------------------------------------------
    function selectTool(id) {
      set("ui.activeTool", id);   // emits change:ui.activeTool — bbox/pan/etc react too
    }
    function activeTool() {
      var t = get("ui.activeTool");
      return t || "move";
    }
    function reflectActiveTool() {
      var host = dom.tooldock;
      if (!host) return;
      var cur = activeTool();
      var btns = host.querySelectorAll(".tb-tool");
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle("active", btns[i].dataset.tool === cur);
      }
      applyCursor(cur);
      reflectToolbar();
    }
    function reflectToolbar() {
      var bar = document.getElementById("tb-toolbar");
      if (!bar) return;
      var cur = activeTool();
      var isPaint = (cur === "brush" || cur === "eraser");
      var name = (TOOLS.filter(function (t) { return t.id === cur; })[0] || {}).title || cur;
      var nameEl = bar.querySelector("#tb-tool-name");
      if (nameEl) nameEl.textContent = name.replace(/\s*\(.\)$/, "");
      var onlyBrush = bar.querySelectorAll(".tb-brush-only");
      for (var i = 0; i < onlyBrush.length; i++) {
        onlyBrush[i].style.display = isPaint ? "" : "none";
      }
    }
    function applyCursor(toolId) {
      var def = (TOOLS.filter(function (t) { return t.id === toolId; })[0] || {}).cursor || "default";
      var host = dom.canvasHost || (dom.konvaStage && dom.konvaStage.parentElement);
      if (host) host.style.cursor = def;
      if (stage && stage.container) {
        try { stage.container().style.cursor = def; } catch (_) {}
      }
    }

    // react to external tool changes (keyboard shortcuts, other modules)
    bus.on("change:ui.activeTool", function () { reflectActiveTool(); });

    // keyboard shortcuts (ignore while typing in an input/textarea)
    document.addEventListener("keydown", function (e) {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var tgt = e.target;
      if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable)) return;
      var k = (e.key || "").toLowerCase();
      var hit = TOOLS.filter(function (t) { return t.key === k; })[0];
      if (hit) { selectTool(hit.id); e.preventDefault(); }
      // bracket keys resize brush
      if (k === "[" || k === "]") {
        var d = (k === "]") ? 4 : -4;
        var ns = Math.max(1, Math.min(200, brush.size + d));
        brush.size = ns;
        var si = document.getElementById("tb-size"); if (si) si.value = ns;
        var sv = document.getElementById("tb-size-val"); if (sv) sv.textContent = ns;
      }
    });

    // -------------------------------------------------------------------------
    // 4) STAGE WIRING  (wait for canvasCore's stage; graceful if it never comes)
    // -------------------------------------------------------------------------
    if (state.canvas && state.canvas.stage) { attachStage(state.canvas.stage); }
    bus.on("canvas:stage", function (st) { if (st) attachStage(st); });

    function attachStage(st) {
      if (!st || st === stage) return;
      stage = st;
      // Konva pointer events fire on the stage; we filter by active tool.
      stage.on("mousedown.tb touchstart.tb", onDown);
      stage.on("mousemove.tb touchmove.tb", onMove);
      stage.on("mouseup.tb touchend.tb mouseleave.tb", onUp);
      applyCursor(activeTool());
    }

    // resolve the Konva.Group we should draw into for the current active layer.
    function resolveDrawGroup() {
      if (!stage) return null;
      var layerId = get("activeLayerId");

      // (a) ask the layers module (if present) for the authoritative group.
      if (layerId != null) {
        var reply = { group: null };
        try { bus.emit("layer:getGroup", { id: layerId, reply: reply }); } catch (_) {}
        if (reply.group && typeof reply.group.add === "function") {
          return reply.group;
        }
      }

      // (b) our own per-layer stash on a dedicated layer (so strokes persist per layer).
      if (layerId != null) {
        ensurePaintLayer();
        if (!groupCache[layerId]) {
          var g = new Konva.Group({ name: "tb-layer-" + layerId, id: "tb-grp-" + layerId });
          groupCache[layerId] = g;
          paintLayer.add(g);
        }
        // keep only the active layer's group visible to avoid cross-layer bleed in fallback
        for (var k in groupCache) {
          if (groupCache.hasOwnProperty(k)) groupCache[k].visible(String(k) === String(layerId));
        }
        return groupCache[layerId];
      }

      // (c) no active layer at all — single shared fallback group.
      ensurePaintLayer();
      if (!groupCache.__default) {
        groupCache.__default = new Konva.Group({ name: "tb-default" });
        paintLayer.add(groupCache.__default);
      }
      return groupCache.__default;
    }

    function ensurePaintLayer() {
      if (paintLayer || !stage) return;
      paintLayer = new Konva.Layer({ name: "tb-paint" });
      stage.add(paintLayer);
    }

    // is the active layer a mask layer? (mask = red semi-transparent paint)
    function activeLayerIsMask() {
      var layerId = get("activeLayerId");
      var layers = get("layers") || [];
      for (var i = 0; i < layers.length; i++) {
        if (layers[i] && layers[i].id === layerId) {
          var t = layers[i].type;
          return t === "mask" || t === "regional";
        }
      }
      return false;
    }

    // -------------------------------------------------------------------------
    // 5) PAINTING
    // -------------------------------------------------------------------------
    function localPoint(group) {
      var pos = stage.getPointerPosition();
      if (!pos) return null;
      // map stage (screen) coords into the group's local coordinate space so
      // strokes stay aligned when the layer is panned/zoomed.
      try {
        var tr = group.getAbsoluteTransform().copy();
        tr.invert();
        return tr.point(pos);
      } catch (_) {
        return pos;
      }
    }

    function onDown(e) {
      var tool = activeTool();
      if (tool !== "brush" && tool !== "eraser") return;
      if (e && e.evt && e.evt.preventDefault) e.evt.preventDefault();
      var group = resolveDrawGroup();
      if (!group) return;
      var p = localPoint(group);
      if (!p) return;

      drawing = true;
      var eraser = (tool === "eraser");
      var isMask = activeLayerIsMask();
      var stroke = eraser ? "#000" : (isMask ? brush.maskColor : brush.color);
      var opacity = eraser ? 1 : (isMask ? Math.min(0.55, brush.opacity) : brush.opacity);

      curLine = new Konva.Line({
        points: [p.x, p.y, p.x, p.y],
        stroke: stroke,
        strokeWidth: brush.size,
        opacity: opacity,
        lineCap: "round",
        lineJoin: "round",
        tension: 0,
        listening: false,
        globalCompositeOperation: eraser ? "destination-out" : "source-over",
        name: eraser ? "tb-erase" : (isMask ? "tb-mask" : "tb-stroke"),
      });
      group.add(curLine);
      var ly = group.getLayer();
      if (ly) ly.batchDraw();
    }

    function onMove() {
      if (!drawing || !curLine) return;
      var group = curLine.getParent();
      var p = localPoint(group || curLine);
      if (!p) return;
      var pts = curLine.points();
      pts.push(p.x, p.y);
      curLine.points(pts);
      var ly = curLine.getLayer();
      if (ly) ly.batchDraw();
    }

    function onUp() {
      if (!drawing) return;
      drawing = false;
      if (curLine) {
        var grp = curLine.getParent();
        try { bus.emit("paint:stroke", { layerId: get("activeLayerId"), node: curLine }); } catch (_) {}
        if (grp) { var ly = grp.getLayer(); if (ly) ly.batchDraw(); }
      }
      curLine = null;
    }

    // -------------------------------------------------------------------------
    // 6) SCOPED CSS
    // -------------------------------------------------------------------------
    function injectCss() {
      if (document.getElementById("style-toolsBrush")) return;
      var css =
        "#tooldock #tb-dock{display:flex;flex-direction:column;gap:6px;width:100%;align-items:center}" +
        "#tooldock .tb-tool{width:36px;height:36px;display:flex;align-items:center;justify-content:center;" +
          "background:transparent;border:1px solid transparent;border-radius:8px;color:var(--muted);cursor:pointer;padding:0}" +
        "#tooldock .tb-tool:hover{color:var(--text);border-color:var(--line);background:var(--panel2)}" +
        "#tooldock .tb-tool.active{color:#fff;background:var(--accent2);border-color:var(--accent)}" +
        "#tooldock .tb-tool svg{display:block}" +
        "#canvas-toolbar #tb-toolbar{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--text)}" +
        "#canvas-toolbar #tb-tool-name{font-weight:600;color:var(--muted);min-width:46px}" +
        "#canvas-toolbar .tb-sep{width:1px;height:18px;background:var(--line)}" +
        "#canvas-toolbar .tb-ctl{display:flex;align-items:center;gap:6px;color:var(--muted)}" +
        "#canvas-toolbar .tb-ctl input[type=range]{width:96px}" +
        "#canvas-toolbar .tb-ctl input[type=color]{width:28px;height:22px;padding:0;border-radius:5px;cursor:pointer}" +
        "#canvas-toolbar #tb-size-val{min-width:22px;text-align:right;color:var(--text)}";
      var el = document.createElement("style");
      el.id = "style-toolsBrush";
      el.textContent = css;
      document.head.appendChild(el);
    }
  }
})();
