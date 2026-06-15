/* sam.js — SAM-assisted masking (module 'sam').
   OWNS: a [SAM] tool button in #canvas-toolbar + click-point / drag-box collection on the
   shared Konva stage. Sends coords to api.sam('points', {...}); receives a mask image and
   loads it into an inpaint-mask layer.

   Coordination contract (NEVER edits layers.js / canvas_core.js / state.js / api.js):
   - Stage: read from state.canvas.stage, or wait for bus 'canvas:stage'.
   - Mask -> layer: emit bus 'sam:mask' AND a request 'layers:ensureMask' / 'layers:setMaskImage'
     so the layers module (if present) owns the layer model. As a graceful fallback (layers
     module not built yet) we draw the mask into our OWN Konva group on the stage so the user
     still sees output.
   - All cross-module talk via state/bus/api only.

   Degrades gracefully: api.sam() may 404 until the backend ships -> show a status message,
   keep markers, never throw fatally. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  S.register("sam", {
    init: function (ctx) {
      var bus = ctx.bus, api = ctx.api, get = ctx.get, set = ctx.set, Konva = ctx.Konva;
      var dom = ctx.dom || {};

      // ---- module-local state ----
      var stage = null;            // shared Konva.Stage
      var samLayer = null;         // our own Konva.Layer for markers + fallback mask
      var markerGroup = null;      // Konva.Group holding click markers + box
      var maskGroup = null;        // Konva.Group holding the fallback mask image
      var active = false;          // SAM tool selected?
      var mode = "points";         // 'points' | 'box'
      var points = [];             // [{x,y,label}] label 1=foreground 0=background (stage coords)
      var markerNodes = [];        // Konva nodes for points
      var boxRect = null;          // Konva.Rect for the drag box
      var boxStart = null;         // {x,y} drag origin (stage coords)
      var dragging = false;
      var busy = false;            // request in flight
      var rootId = "sam-ui";

      // ===================== CSS (scoped, injected from init) =====================
      injectCSS();

      function injectCSS() {
        if (document.getElementById("style-sam")) return;
        var st = document.createElement("style");
        st.id = "style-sam";
        st.textContent = [
          "#" + rootId + "{display:flex;align-items:center;gap:6px}",
          "#" + rootId + " .sam-btn{background:var(--panel2);border:1px solid var(--line);color:var(--text);" +
            "border-radius:6px;padding:4px 9px;cursor:pointer;font-size:12px;display:flex;align-items:center;gap:5px}",
          "#" + rootId + " .sam-btn:hover{border-color:var(--accent)}",
          "#" + rootId + " .sam-btn.on{background:var(--accent2);border-color:var(--accent);color:#fff;font-weight:600}",
          "#" + rootId + " .sam-sub{display:none;align-items:center;gap:4px;border-left:1px solid var(--line);" +
            "margin-left:2px;padding-left:6px}",
          "#" + rootId + ".active .sam-sub{display:flex}",
          "#" + rootId + " .sam-chip{background:var(--panel);border:1px solid var(--line);color:var(--muted);" +
            "border-radius:5px;padding:3px 7px;cursor:pointer;font-size:11px}",
          "#" + rootId + " .sam-chip.on{color:#fff;border-color:var(--accent);background:var(--accent2)}",
          "#" + rootId + " .sam-chip[disabled]{opacity:.45;cursor:default}",
          "#sam-status{position:absolute;bottom:30px;left:8px;z-index:6;color:var(--muted);font-size:11px;" +
            "background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:3px 8px;display:none}",
          "#sam-status.show{display:block}",
          "#sam-status.err{color:var(--danger);border-color:var(--danger)}",
          "#sam-status.ok{color:var(--ok);border-color:var(--ok)}",
        ].join("\n");
        document.head.appendChild(st);
      }

      // ===================== toolbar UI =====================
      var root = document.createElement("div");
      root.id = rootId;
      root.innerHTML =
        '<button class="sam-btn" id="sam-toggle" title="SAM-assisted masking: click subjects (Shift=exclude), or switch to Box and drag">' +
          '<span>◉</span><span>SAM</span></button>' +
        '<span class="sam-sub">' +
          '<span class="sam-chip on" id="sam-mode-points" title="Click points (Shift-click to exclude)">Points</span>' +
          '<span class="sam-chip" id="sam-mode-box" title="Drag a box">Box</span>' +
          '<span class="sam-chip" id="sam-run" title="Send to SAM &rarr; mask">Run</span>' +
          '<span class="sam-chip" id="sam-clear" title="Clear selection">Clear</span>' +
        '</span>';
      if (dom.canvasToolbar) dom.canvasToolbar.appendChild(root);

      var btnToggle = root.querySelector("#sam-toggle");
      var chipPoints = root.querySelector("#sam-mode-points");
      var chipBox = root.querySelector("#sam-mode-box");
      var chipRun = root.querySelector("#sam-run");
      var chipClear = root.querySelector("#sam-clear");

      // status line in the canvas host
      var statusEl = document.createElement("div");
      statusEl.id = "sam-status";
      if (dom.canvasHost) dom.canvasHost.appendChild(statusEl);
      else if (dom.canvasToolbar && dom.canvasToolbar.parentNode) dom.canvasToolbar.parentNode.appendChild(statusEl);

      function status(msg, kind) {
        if (!statusEl) return;
        statusEl.textContent = msg || "";
        statusEl.className = (msg ? "show " : "") + (kind || "");
      }

      // ===================== events: toolbar =====================
      btnToggle.addEventListener("click", function () { active ? deactivate() : activate(); });
      chipPoints.addEventListener("click", function () { setMode("points"); });
      chipBox.addEventListener("click", function () { setMode("box"); });
      chipRun.addEventListener("click", function () { runSam(); });
      chipClear.addEventListener("click", function () { clearSelection(); status("Cleared."); });

      function setMode(m) {
        mode = m;
        chipPoints.classList.toggle("on", m === "points");
        chipBox.classList.toggle("on", m === "box");
        clearSelection();
        status(m === "points" ? "Click subjects (Shift-click to exclude)." : "Drag a box around the subject.");
      }

      // ===================== activate / deactivate the tool =====================
      function activate() {
        active = true;
        btnToggle.classList.add("on");
        root.classList.add("active");
        // tell the rest of the app a tool change happened (toolsBrush owns ui.activeTool, but
        // we also set it so other tools can deselect; this is plain state per contract).
        try { set("ui.activeTool", "sam"); } catch (_) {}
        bindStage();
        status(mode === "points" ? "Click subjects (Shift-click to exclude)." : "Drag a box around the subject.");
      }

      function deactivate() {
        active = false;
        btnToggle.classList.remove("on");
        root.classList.remove("active");
        unbindStage();
        status("");
        if (get && get("ui.activeTool") === "sam") { try { set("ui.activeTool", "move"); } catch (_) {} }
      }

      // If another tool becomes active, drop out of SAM mode (without recursing).
      bus.on("change:ui.activeTool", function (tool) {
        if (tool !== "sam" && active) {
          active = false;
          btnToggle.classList.remove("on");
          root.classList.remove("active");
          unbindStage();
          status("");
        }
      });

      // ===================== stage acquisition =====================
      function tryGetStage() {
        var s = (ctx.state && ctx.state.canvas && ctx.state.canvas.stage) || null;
        if (s) { setStage(s); return true; }
        return false;
      }
      function setStage(s) {
        if (stage === s || !s) return;
        stage = s;
        ensureOwnLayer();
      }
      // canvasCore exposes the stage via this bus event + state.canvas.stage
      bus.on("canvas:stage", function (s) { setStage(s); });
      // also poll briefly in case canvasCore inited before us and won't re-emit
      if (!tryGetStage()) {
        var tries = 0;
        var iv = setInterval(function () {
          if (tryGetStage() || ++tries > 40) clearInterval(iv);
        }, 100);
      }

      function ensureOwnLayer() {
        if (!stage || !Konva) return;
        if (samLayer && samLayer.getStage() === stage) return;
        samLayer = new Konva.Layer({ name: "sam-overlay", listening: false });
        maskGroup = new Konva.Group({ name: "sam-mask-fallback" });
        markerGroup = new Konva.Group({ name: "sam-markers" });
        samLayer.add(maskGroup);
        samLayer.add(markerGroup);
        stage.add(samLayer);
        samLayer.moveToTop();
      }

      // ===================== pointer interaction on the stage =====================
      function bindStage() {
        if (!stage) { ensureOwnLayer(); }
        if (!stage) { status("Canvas not ready yet.", "err"); return; }
        ensureOwnLayer();
        // make the overlay layer listen while SAM is active so it can catch clicks
        samLayer.listening(true);
        stage.on("mousedown.sam touchstart.sam", onDown);
        stage.on("mousemove.sam touchmove.sam", onMove);
        stage.on("mouseup.sam touchend.sam", onUp);
        var c = stage.container && stage.container();
        if (c) c.style.cursor = "crosshair";
      }
      function unbindStage() {
        if (!stage) return;
        stage.off(".sam");
        if (samLayer) samLayer.listening(false);
        dragging = false;
        var c = stage.container && stage.container();
        if (c) c.style.cursor = "";
      }

      // Convert pointer position into stage (image/world) coordinates, accounting for the
      // stage transform that canvas_core applies for pan/zoom.
      function stagePoint() {
        if (!stage) return null;
        var pos = stage.getPointerPosition();
        if (!pos) return null;
        var tr = stage.getAbsoluteTransform().copy();
        tr.invert();
        return tr.point(pos);
      }

      function onDown(e) {
        if (!active) return;
        var p = stagePoint();
        if (!p) return;
        if (mode === "points") {
          var shift = (e.evt && (e.evt.shiftKey || e.evt.button === 2)) ? true : false;
          addPoint(p.x, p.y, shift ? 0 : 1);
        } else {
          dragging = true;
          boxStart = { x: p.x, y: p.y };
          clearBox();
          boxRect = new Konva.Rect({
            x: p.x, y: p.y, width: 0, height: 0,
            stroke: "#6c8cff", strokeWidth: 2, dash: [6, 4],
            fill: "rgba(108,140,255,0.12)", listening: false,
          });
          markerGroup.add(boxRect);
          samLayer.batchDraw();
        }
      }

      function onMove() {
        if (!active || mode !== "box" || !dragging || !boxRect || !boxStart) return;
        var p = stagePoint();
        if (!p) return;
        var x = Math.min(boxStart.x, p.x), y = Math.min(boxStart.y, p.y);
        var w = Math.abs(p.x - boxStart.x), h = Math.abs(p.y - boxStart.y);
        boxRect.setAttrs({ x: x, y: y, width: w, height: h });
        samLayer.batchDraw();
      }

      function onUp() {
        if (!active) return;
        if (mode === "box" && dragging) {
          dragging = false;
          if (boxRect && (boxRect.width() < 3 || boxRect.height() < 3)) { clearBox(); status("Box too small."); return; }
          status("Box set. Click Run to generate the mask.");
        }
      }

      function addPoint(x, y, label) {
        points.push({ x: x, y: y, label: label });
        var dot = new Konva.Circle({
          x: x, y: y, radius: 6,
          fill: label ? "#46c46a" : "#e0556b",
          stroke: "#0b0d12", strokeWidth: 2, listening: false,
        });
        var ring = new Konva.Circle({
          x: x, y: y, radius: 9, stroke: label ? "#46c46a" : "#e0556b",
          strokeWidth: 1.5, opacity: 0.5, listening: false,
        });
        markerGroup.add(ring);
        markerGroup.add(dot);
        markerNodes.push(dot, ring);
        samLayer.batchDraw();
        status(points.length + " point" + (points.length === 1 ? "" : "s") + " — click Run to generate the mask.");
      }

      function clearBox() {
        if (boxRect) { boxRect.destroy(); boxRect = null; }
        boxStart = null;
      }

      function clearSelection() {
        points = [];
        markerNodes.forEach(function (n) { try { n.destroy(); } catch (_) {} });
        markerNodes = [];
        clearBox();
        if (samLayer) samLayer.batchDraw();
      }

      // ===================== build the SAM request payload =====================
      // Reference image: prefer a generated raster, else the bbox region. We pass what we can
      // (filename via gallery, plus the bbox geometry) and let the backend resolve it. The
      // backend may also use the last-rendered output server-side; we degrade either way.
      function currentImageRef() {
        // try last gallery item filename
        var g = (get && get("gallery")) || [];
        if (g && g.length) {
          var last = g[g.length - 1];
          if (last && last.filename) return { filename: last.filename, type: last.type || "output", subfolder: last.subfolder || "" };
        }
        // try an active raster layer reference if layers module recorded one
        var layers = (get && get("layers")) || [];
        for (var i = layers.length - 1; i >= 0; i--) {
          var L = layers[i];
          if (L && (L.type === "raster" || L.type === "reference") && (L.filename || L.image)) {
            return { filename: L.filename || null, image: L.image || null, type: L.type, layerId: L.id };
          }
        }
        return null;
      }

      function buildPayload() {
        var bbox = (get && get("canvas.bbox")) || { x: 0, y: 0, width: get("params.width") || 1024, height: get("params.height") || 1024 };
        var payload = {
          width: bbox.width, height: bbox.height,
          bbox: bbox,
          image: currentImageRef(),
        };
        if (mode === "points") {
          payload.points = points.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; });
          payload.point_labels = points.map(function (p) { return p.label; });
          // also a structured form for backends that expect it
          payload.points_struct = points.map(function (p) { return { x: Math.round(p.x), y: Math.round(p.y), label: p.label }; });
        } else if (boxRect) {
          payload.box = [
            Math.round(boxRect.x()), Math.round(boxRect.y()),
            Math.round(boxRect.x() + boxRect.width()), Math.round(boxRect.y() + boxRect.height()),
          ];
        }
        return payload;
      }

      // ===================== run SAM =====================
      function runSam() {
        if (busy) return;
        if (mode === "points" && !points.length) { status("Click at least one point first.", "err"); return; }
        if (mode === "box" && !boxRect) { status("Drag a box first.", "err"); return; }
        if (!api || typeof api.sam !== "function") { status("SAM API unavailable.", "err"); return; }

        busy = true;
        chipRun.setAttribute("disabled", "");
        status("Running SAM…");
        var payload = buildPayload();

        var p;
        try { p = api.sam("points", payload); } // 'points' kind covers point+box per api.js contract
        catch (e) { busy = false; chipRun.removeAttribute("disabled"); status("SAM call failed.", "err"); console.warn("[sam] call threw", e); return; }

        if (!p || typeof p.then !== "function") {
          busy = false; chipRun.removeAttribute("disabled");
          status("SAM returned no promise.", "err");
          return;
        }

        p.then(function (res) {
          busy = false; chipRun.removeAttribute("disabled");
          onMaskResult(res);
        }).catch(function (err) {
          busy = false; chipRun.removeAttribute("disabled");
          // 404 = backend not shipped yet: degrade gracefully, keep markers
          var msg = String(err && err.message || err || "");
          if (/404/.test(msg)) status("SAM backend not available yet (404). Markers kept.", "err");
          else status("SAM failed: " + msg, "err");
          console.warn("[sam] request failed", err);
        });
      }

      // ===================== handle the mask result =====================
      // Accept several shapes the backend might use: a view-able filename, a data URL, or
      // raw base64 PNG. Normalize to a loadable image source.
      function maskSrcFromResult(res) {
        if (!res) return null;
        if (typeof res === "string") return normalizeImgSrc(res);
        // common ComfyUI-ish shapes
        if (res.mask) return normalizeImgSrc(res.mask);
        if (res.image) return normalizeImgSrc(res.image);
        if (res.mask_b64 || res.b64) return "data:image/png;base64," + (res.mask_b64 || res.b64);
        if (res.filename) {
          if (api && typeof api.viewUrl === "function")
            return api.viewUrl(res.filename, res.type || "output", res.subfolder || "");
          return res.filename;
        }
        // array of masks -> take the first
        if (Array.isArray(res) && res.length) return maskSrcFromResult(res[0]);
        if (res.masks && res.masks.length) return maskSrcFromResult(res.masks[0]);
        return null;
      }
      function normalizeImgSrc(s) {
        if (!s) return null;
        if (/^data:|^https?:|^blob:/.test(s)) return s;
        if (/^[A-Za-z0-9+/=\s]+$/.test(s) && s.length > 64) return "data:image/png;base64," + s.replace(/\s+/g, "");
        // treat as a filename to view
        if (api && typeof api.viewUrl === "function") return api.viewUrl(s, "output", "");
        return s;
      }

      function onMaskResult(res) {
        var src = maskSrcFromResult(res);
        if (!src) { status("SAM returned no usable mask.", "err"); console.warn("[sam] no mask in result", res); return; }

        var img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = function () { deliverMask(img, src); };
        img.onerror = function () {
          // still hand the source off via bus so a layers module can try its own loader
          status("Mask loaded reference (image decode failed locally).", "err");
          emitMask(null, src);
        };
        img.src = src;
      }

      // Hand the mask to the layers module (preferred) AND draw a fallback locally.
      function deliverMask(img, src) {
        emitMask(img, src);
        drawFallbackMask(img);
        status("Mask ready.", "ok");
      }

      function emitMask(img, src) {
        var bbox = (get && get("canvas.bbox")) || { x: 0, y: 0 };
        var payload = {
          source: "sam",
          image: img || null,          // HTMLImageElement when decoded
          src: src,                    // url / data-url for re-loading elsewhere
          x: bbox.x || 0, y: bbox.y || 0,
          width: (img && img.naturalWidth) || bbox.width,
          height: (img && img.naturalHeight) || bbox.height,
          mode: "inpaint-mask",
        };
        // 1) generic event any module may consume
        bus.emit("sam:mask", payload);
        // 2) explicit requests to the layers module (it owns state.layers / activeLayerId).
        //    layers.js can listen for either; if it doesn't exist, nothing happens (graceful).
        bus.emit("layers:ensureMask", { name: "SAM Mask", type: "mask" });
        bus.emit("layers:setMaskImage", payload);
        // 3) leave a breadcrumb in state so generate_ws can find the latest SAM mask without
        //    racing the layers module. This is plain state (not layers' owned arrays).
        try { set("canvas.samMask", { src: src, x: payload.x, y: payload.y, width: payload.width, height: payload.height, ts: Date.now() }); } catch (_) {}
      }

      // Fallback render: tint the mask over the image region so the user sees a result even
      // before layers.js exists. Lives in our own group; layers.js owns the "real" mask layer.
      function drawFallbackMask(img) {
        if (!stage || !Konva || !maskGroup) return;
        // only draw fallback if no layers module appears to have claimed the mask
        var layers = (get && get("layers")) || [];
        var hasMaskLayer = layers.some(function (L) { return L && L.type === "mask"; });
        // clear any prior fallback
        maskGroup.destroyChildren();
        if (hasMaskLayer) { samLayer.batchDraw(); return; } // layers owns it; don't double-draw
        var bbox = (get && get("canvas.bbox")) || { x: 0, y: 0 };
        var kImg = new Konva.Image({
          x: bbox.x || 0, y: bbox.y || 0,
          image: img,
          width: img.naturalWidth || bbox.width,
          height: img.naturalHeight || bbox.height,
          opacity: 0.5,
          listening: false,
          name: "sam-mask-preview",
        });
        // tint magenta so it reads as a mask overlay
        try {
          kImg.cache();
          kImg.filters([Konva.Filters.RGBA]);
          kImg.red(224); kImg.green(85); kImg.blue(160); kImg.alpha(0.5);
        } catch (_) { /* filters may not be configured; plain overlay is fine */ }
        maskGroup.add(kImg);
        samLayer.batchDraw();
      }

      // ===================== external triggers =====================
      // Let other modules invoke SAM programmatically (e.g. a tooldock button in toolsBrush).
      bus.on("sam:activate", function () { if (!active) activate(); });
      bus.on("sam:run", function () { runSam(); });
      bus.on("sam:clear", function () { clearSelection(); });

      // boot breadcrumb
      console.info("[sam] ready");
    },
  });
})();
