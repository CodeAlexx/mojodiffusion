/* mask_paint.js — module 'maskPaint'. In-browser mask-painting + outpaint editor
   for the inpaint path (SwarmUI "Edit Image" parity, GAP §2).

   What it does
   ------------
   - Adds a Mask tool to the contextual canvas toolbar (#canvas-toolbar). When the
     tool is active you paint a soft red mask over the current image directly on the
     Konva stage. The mask marks the region to REGENERATE (white in the exported
     mask PNG); everything else is preserved.
   - Brush controls: size, hardness (feather), clear, invert. Bracket keys [ ] resize.
   - Outpaint: 4 edge handles (top/right/bottom/left) extend the generation canvas
     outward by a margin; the extended area is auto-marked as mask so it is filled.
   - Export: rasterizes the painted mask to a PNG (white=regen on black) at the
     generation dims and wires it into the EXISTING inpaint path:
       * attaches blob + dataURL to the inpaint *mask layer* in state.layers, so the
         already-built generate_ws.assembleGraph -> layerToBlob -> api.uploadMask path
         consumes it (LoadImage -> SetLatentNoiseMask), and
       * writes state.params.mask_data (base64 PNG), state.params.mask_channel and the
         outpaint margins, which the orchestrator forwards onto /v1/generate so the
         server/worker inpaint path (init_image + mask_image, lanpaint_mask_channel)
         can pick them up once an upload/data-URI sink exists server-side.

   Ownership / contract
   --------------------
   - One file. Registers window.Serenity.modules.maskPaint = { init(ctx) }.
   - Talks to the rest of the app ONLY through ctx.{state,get,set,bus,Konva,dom}.
   - The Konva stage is owned by canvasCore. We NEVER wipe a container: we add our
     OWN child Konva.Layer ("mask-paint") to the shared stage (like bbox.js), and our
     OWN toolbar group inside #canvas-toolbar. Degrades gracefully if the stage / a
     base image isn't there yet (we still let you paint on the bbox region).
   - Cooperates with the 'layers' module: if an inpaint "mask" layer exists in
     state.layers we target it; otherwise we operate standalone and still export.
*/
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  S.register("maskPaint", {
    init: function (ctx) {
      var state = ctx.state, get = ctx.get, set = ctx.set, bus = ctx.bus;
      var Konva = ctx.Konva || window.Konva;
      var dom = ctx.dom || {};
      if (!Konva) { console.warn("[maskPaint] Konva unavailable; disabled"); return; }

      injectCss();

      // ---- module-local config -------------------------------------------------
      var cfg = {
        size: 48,          // brush diameter (stage px)
        hardness: 0.7,     // 0..1 -> feather; 1 = hard edge
        maskColor: "#ff3b5c",
        maskAlpha: 0.5,    // on-screen mask opacity (export is always full white)
        outpaintMargin: 256, // px the edge handles add per click
      };

      // ---- runtime ------------------------------------------------------------
      var stage = null;
      var maskLayer = null;   // our dedicated Konva.Layer (own child of the stage)
      var strokeGroup = null; // holds all mask strokes/rects
      var cursorLayer = null; // brush-size cursor ring
      var cursorRing = null;
      var painting = false, erasing = false, curLine = null;
      var active = false;     // is the Mask tool selected?
      var toolbar = null;     // our toolbar element inside #canvas-toolbar

      // -------------------------------------------------------------------------
      // 1) STAGE ACQUISITION (canvasCore owns it; may arrive after us)
      // -------------------------------------------------------------------------
      function getStage() { return (state.canvas && state.canvas.stage) || stage; }

      function attach(st) {
        if (!st || st === stage) return;
        stage = st;
        ensureLayers();
        wireStage();
        console.info("[maskPaint] attached to stage");
      }
      bus.on("canvas:stage", function (st) { attach(st || getStage()); });
      if (getStage()) attach(getStage());
      else {
        // poll briefly for a late stage (bounded, self-stopping)
        var tries = 0;
        var poll = setInterval(function () {
          tries++;
          var s = getStage();
          if (s) { attach(s); clearInterval(poll); }
          else if (tries > 40) clearInterval(poll);
        }, 250);
      }

      function ensureLayers() {
        if (!stage) return;
        if (!maskLayer || maskLayer.getStage() !== stage) {
          maskLayer = new Konva.Layer({ name: "mask-paint" });
          strokeGroup = new Konva.Group({ name: "mask-strokes" });
          maskLayer.add(strokeGroup);
          stage.add(maskLayer);
          maskLayer.opacity(cfg.maskAlpha);
        }
        if (!cursorLayer || cursorLayer.getStage() !== stage) {
          cursorLayer = new Konva.Layer({ name: "mask-cursor", listening: false });
          cursorRing = new Konva.Circle({
            stroke: "#ffffff", strokeWidth: 1, dash: [3, 3],
            radius: cfg.size / 2, visible: false, listening: false,
          });
          cursorLayer.add(cursorRing);
          stage.add(cursorLayer);
        }
        // keep mask + cursor above the content/result layers
        try { maskLayer.moveToTop(); cursorLayer.moveToTop(); } catch (_) {}
        reflectVisible();
      }

      // -------------------------------------------------------------------------
      // 2) GENERATION FRAME — where the mask maps to in image space.
      //    Prefer the active raster/result image bounds; else the bbox rect; else
      //    a centered rect sized to params.width/height.
      // -------------------------------------------------------------------------
      function genFrame() {
        // Try a result/raster Konva.Image on the stage (resultView draws one).
        // NOTE: Konva's selector engine does NOT support descendant combinators
        // (e.g. ".result-view Image"), so we scan for the first real Image node,
        // preferring one inside the resultView layer.
        if (stage) {
          var imgNode = resultImageNode(stage) || firstImageNode(stage);
          if (imgNode) {
            var r = imgNode.getClientRect({ relativeTo: stage });
            if (r && r.width > 4 && r.height > 4) {
              return { x: r.x, y: r.y, w: r.width, h: r.height,
                       iw: imgNode.width ? imgNode.width() : r.width,
                       ih: imgNode.height ? imgNode.height() : r.height };
            }
          }
        }
        // Fall back to the bbox (the generation region) in stage coords.
        var b = get("canvas.bbox") || {};
        var pw = num(get("params.width"), 1024), ph = num(get("params.height"), 1024);
        var bw = num(b.width, pw), bh = num(b.height, ph);
        var bx = num(b.x, 0), by = num(b.y, 0);
        // map bbox (world coords) through the stage transform to screen coords
        if (stage) {
          var sc = stage.scaleX() || 1;
          return { x: bx * sc + stage.x(), y: by * sc + stage.y(),
                   w: bw * sc, h: bh * sc, iw: bw, ih: bh };
        }
        return { x: 0, y: 0, w: bw, h: bh, iw: bw, ih: bh };
      }

      function firstImageNode(st) {
        var found = null;
        try {
          st.find("Image").forEach(function (n) {
            if (found || !(n.image && n.image())) return;
            // never treat our own mask/cursor nodes (e.g. the Invert image) as
            // the source image — that would map the mask to itself.
            var ly = n.getLayer && n.getLayer();
            var lname = ly && ly.name && ly.name();
            if (lname === "mask-paint" || lname === "mask-cursor") return;
            found = n;
          });
        } catch (_) {}
        return found;
      }

      // Prefer an Image inside the resultView layer (named "result-view"); never
      // pick our own mask/cursor nodes.
      function resultImageNode(st) {
        var found = null;
        try {
          var layers = st.getLayers ? st.getLayers() : [];
          for (var i = 0; i < layers.length && !found; i++) {
            var lname = layers[i].name && layers[i].name();
            if (lname === "mask-paint" || lname === "mask-cursor") continue;
            layers[i].find("Image").forEach(function (n) {
              if (!found && n.image && n.image()) found = n;
            });
          }
        } catch (_) {}
        return found;
      }

      // -------------------------------------------------------------------------
      // 3) TOOLBAR (in #canvas-toolbar) — appears when the Mask tool is active.
      // -------------------------------------------------------------------------
      buildToolButton();
      buildToolbar();
      // toolsBrush may build #tb-dock AFTER us; once the app is ready, make sure
      // our button lives inside that flex column (relocate if needed).
      bus.on("app:ready", buildToolButton);

      function buildToolButton() {
        // Add a Mask tool to the vertical tool dock if it exists (tools_brush owns
        // it but appends; we just add our own button that toggles via state.ui).
        var dock = dom.tooldock;
        if (!dock) return;
        var holder = dock.querySelector("#tb-dock") || dock;
        var existing = dock.querySelector('[data-tool="mask"]');
        if (existing) {
          // relocate into #tb-dock if it appeared after our first placement
          if (holder !== existing.parentNode) holder.appendChild(existing);
          return;
        }
        var b = document.createElement("button");
        b.className = "tb-tool mp-tool-btn";
        b.dataset.tool = "mask";
        b.title = "Mask paint (M)";
        b.setAttribute("aria-label", "Mask paint");
        b.innerHTML =
          '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" ' +
          'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M3 17c2 0 3-1 3-3 0-1.5 1.5-3 3-3l7-7a2.6 2.6 0 0 1 4 4l-7 7c0 1.5-1.5 3-3 3-2 0-3 1-3 3z"/>' +
          '<circle cx="17.5" cy="6.5" r="1.2" fill="currentColor"/></svg>';
        b.addEventListener("click", function () { set("ui.activeTool", "mask"); });
        holder.appendChild(b);
      }

      function buildToolbar() {
        var host = dom.canvasToolbar;
        if (!host) return;
        toolbar = document.createElement("div");
        toolbar.id = "mp-toolbar";
        toolbar.style.display = "none";
        toolbar.innerHTML =
          '<span class="mp-title">Mask</span>' +
          '<span class="mp-sep"></span>' +
          '<label class="mp-ctl">size<input id="mp-size" type="range" min="2" max="300" step="1" value="' + cfg.size + '"><span id="mp-size-val">' + cfg.size + '</span></label>' +
          '<label class="mp-ctl">hardness<input id="mp-hard" type="range" min="0" max="100" step="5" value="' + Math.round(cfg.hardness * 100) + '"></label>' +
          '<span class="mp-sep"></span>' +
          '<button class="btn mp-b" id="mp-erase" title="Erase mask (hold Alt)">Erase</button>' +
          '<button class="btn mp-b" id="mp-invert" title="Invert mask">Invert</button>' +
          '<button class="btn mp-b" id="mp-clear" title="Clear mask">Clear</button>' +
          '<span class="mp-sep"></span>' +
          '<span class="mp-title">Outpaint</span>' +
          '<button class="btn mp-b mp-out" data-edge="top" title="Extend top">▲</button>' +
          '<button class="btn mp-b mp-out" data-edge="bottom" title="Extend bottom">▼</button>' +
          '<button class="btn mp-b mp-out" data-edge="left" title="Extend left">◀</button>' +
          '<button class="btn mp-b mp-out" data-edge="right" title="Extend right">▶</button>' +
          '<label class="mp-ctl">px<input id="mp-margin" type="number" min="32" max="2048" step="32" value="' + cfg.outpaintMargin + '"></label>' +
          '<span class="mp-sep"></span>' +
          '<button class="btn btn-primary mp-b" id="mp-apply" title="Export mask -> inpaint">Apply mask</button>';
        host.appendChild(toolbar);

        var sz = toolbar.querySelector("#mp-size");
        var szVal = toolbar.querySelector("#mp-size-val");
        var hard = toolbar.querySelector("#mp-hard");
        sz.addEventListener("input", function () {
          cfg.size = parseInt(sz.value, 10) || 2; szVal.textContent = cfg.size; updateCursorRadius();
        });
        hard.addEventListener("input", function () { cfg.hardness = (parseInt(hard.value, 10) || 0) / 100; });
        toolbar.querySelector("#mp-erase").addEventListener("click", function () {
          erasing = !erasing; this.classList.toggle("btn-primary", erasing);
        });
        toolbar.querySelector("#mp-invert").addEventListener("click", invertMask);
        toolbar.querySelector("#mp-clear").addEventListener("click", clearMask);
        toolbar.querySelector("#mp-margin").addEventListener("change", function () {
          cfg.outpaintMargin = Math.max(32, Math.min(2048, parseInt(this.value, 10) || 256));
        });
        var outs = toolbar.querySelectorAll(".mp-out");
        for (var i = 0; i < outs.length; i++) {
          outs[i].addEventListener("click", (function (edge) {
            return function () { outpaintExtend(edge); };
          })(outs[i].dataset.edge));
        }
        toolbar.querySelector("#mp-apply").addEventListener("click", applyMask);
      }

      function updateCursorRadius() { if (cursorRing) { cursorRing.radius(cfg.size / 2); cursorLayer && cursorLayer.batchDraw(); } }

      // -------------------------------------------------------------------------
      // 4) TOOL ACTIVATION — driven by state.ui.activeTool (tools_brush writes it).
      // -------------------------------------------------------------------------
      bus.on("change:ui.activeTool", function () { reflectVisible(); });
      reflectVisible();

      function reflectVisible() {
        active = (get("ui.activeTool") === "mask");
        if (toolbar) toolbar.style.display = active ? "flex" : "none";
        if (cursorRing) cursorRing.visible(false);
        // reflect dock active state
        var dock = dom.tooldock;
        if (dock) {
          var btns = dock.querySelectorAll(".mp-tool-btn");
          for (var i = 0; i < btns.length; i++) btns[i].classList.toggle("active", active);
        }
        if (maskLayer) maskLayer.listening(active);
        if (cursorLayer) cursorLayer.batchDraw && cursorLayer.batchDraw();
      }

      // keyboard: M selects mask tool; [ ] resize while active
      document.addEventListener("keydown", function (e) {
        if (e.ctrlKey || e.metaKey) return;
        var t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
        var k = (e.key || "").toLowerCase();
        if (k === "m") { set("ui.activeTool", "mask"); e.preventDefault(); return; }
        if (active && (k === "[" || k === "]")) {
          var d = (k === "]") ? 6 : -6;
          cfg.size = Math.max(2, Math.min(300, cfg.size + d));
          var sz = document.getElementById("mp-size"); if (sz) sz.value = cfg.size;
          var sv = document.getElementById("mp-size-val"); if (sv) sv.textContent = cfg.size;
          updateCursorRadius();
        }
      });

      // -------------------------------------------------------------------------
      // 5) PAINTING — Konva.Line strokes on our own strokeGroup. Mask = white in
      //    export; eraser uses destination-out so it carves the mask back out.
      // -------------------------------------------------------------------------
      function wireStage() {
        if (!stage) return;
        stage.on("mousedown.mp touchstart.mp", onDown);
        stage.on("mousemove.mp touchmove.mp", onMove);
        stage.on("mouseup.mp touchend.mp mouseleave.mp", onUp);
        stage.on("mouseover.mp", function () { if (active && cursorRing) cursorRing.visible(true); });
        stage.on("mouseout.mp", function () { if (cursorRing) { cursorRing.visible(false); cursorLayer.batchDraw(); } });
      }

      function pointer() { return stage ? stage.getPointerPosition() : null; }
      // map a screen pointer into the strokeGroup's local space (so strokes stay
      // aligned under pan/zoom).
      function localPoint(p) {
        if (!strokeGroup || !p) return p;
        try { var tr = strokeGroup.getAbsoluteTransform().copy(); tr.invert(); return tr.point(p); }
        catch (_) { return p; }
      }

      function onDown(e) {
        if (!active) return;
        // let space-pan (canvasCore) win when the stage is draggable
        if (stage.draggable && stage.draggable()) return;
        if (e && e.evt && e.evt.preventDefault) e.evt.preventDefault();
        var alt = !!(e && e.evt && e.evt.altKey);
        var carve = erasing || alt;
        var sp = pointer(); if (!sp) return;
        var p = localPoint(sp);
        painting = true;
        // feather: lower hardness -> larger soft shadow blur on the line
        var feather = (1 - cfg.hardness) * cfg.size * 0.6;
        curLine = new Konva.Line({
          points: [p.x, p.y, p.x, p.y],
          stroke: carve ? "#000" : "#ffffff",   // export reads luminance; white=mask
          strokeWidth: cfg.size,
          lineCap: "round", lineJoin: "round", tension: 0,
          shadowColor: carve ? "#000" : "#ffffff",
          shadowBlur: feather, shadowOpacity: feather > 0 ? 0.9 : 0,
          listening: false,
          globalCompositeOperation: carve ? "destination-out" : "source-over",
          name: carve ? "mp-erase" : "mp-paint",
        });
        // Display the painted region as semi-transparent red (readable over any
        // image). Export does NOT key on color — rasterizeMask binarizes by ALPHA
        // (any painted/non-transparent pixel -> white mask; erased -> transparent ->
        // black), so the red display tint is purely cosmetic and the eraser's
        // destination-out genuinely carves the mask back out.
        if (!carve) curLine.stroke(cfg.maskColor);
        strokeGroup.add(curLine);
        maskLayer.batchDraw();
        moveCursor(sp);
      }

      function onMove() {
        var sp = pointer();
        if (active && cursorRing && sp) moveCursor(sp);
        if (!painting || !curLine) return;
        if (!sp) return;
        var p = localPoint(sp);
        var pts = curLine.points(); pts.push(p.x, p.y); curLine.points(pts);
        maskLayer.batchDraw();
      }

      function onUp() {
        if (!painting) return;
        painting = false; curLine = null;
        maskLayer.batchDraw();
      }

      function moveCursor(sp) {
        if (!cursorRing) return;
        cursorRing.position({ x: sp.x, y: sp.y });
        cursorRing.visible(active);
        cursorLayer.batchDraw();
      }

      function clearMask() {
        if (!strokeGroup) return;
        strokeGroup.destroyChildren();
        maskLayer.batchDraw();
      }

      // Invert = paint a full-frame white rect with the existing strokes carved out.
      // We flip by adding a frame rect at the bottom set to XOR-like: easiest correct
      // approach is to rasterize current mask, invert pixels, and lay it back as one image.
      function invertMask() {
        var f = genFrame();
        if (!f || f.w <= 0) return;
        var data = rasterizeMask(f, false); // current mask, white-on-black
        if (!data) return;
        var cv = document.createElement("canvas");
        cv.width = data.width; cv.height = data.height;
        var g = cv.getContext("2d");
        g.putImageData(data, 0, 0);
        // Invert by ALPHA (export keys on alpha, NOT luminance). The rasterized
        // `data` is white-on-black RGB with alpha 255 everywhere; after invert the
        // PREVIOUSLY-masked area must become transparent and the previously-clear
        // area must become opaque red. Keying on the original luminance, set:
        //   was white (masked)   -> transparent (unmasked)
        //   was black (unmasked) -> opaque mask red (masked)
        var rgb = hexToRgb(cfg.maskColor);
        var id = g.getImageData(0, 0, cv.width, cv.height);
        var px = id.data;
        for (var i = 0; i < px.length; i += 4) {
          var wasMasked = px[i] > 127; // luminance high == was white == masked
          if (wasMasked) { px[i + 3] = 0; }            // -> transparent
          else { px[i] = rgb.r; px[i + 1] = rgb.g; px[i + 2] = rgb.b; px[i + 3] = 255; }
        }
        g.putImageData(id, 0, 0);
        // replace strokes with one image positioned over the frame (in local space)
        clearMask();
        var imgEl = new Image();
        imgEl.onload = function () {
          var local0 = localPoint({ x: f.x, y: f.y });
          var local1 = localPoint({ x: f.x + f.w, y: f.y + f.h });
          var kimg = new Konva.Image({
            image: imgEl,
            x: Math.min(local0.x, local1.x), y: Math.min(local0.y, local1.y),
            width: Math.abs(local1.x - local0.x), height: Math.abs(local1.y - local0.y),
            listening: false, name: "mp-paint",
          });
          strokeGroup.add(kimg);
          maskLayer.batchDraw();
        };
        imgEl.src = cv.toDataURL("image/png");
      }

      // -------------------------------------------------------------------------
      // 6) OUTPAINT — extend the generation canvas by margin on one edge. We grow
      //    params.width/height + canvas.bbox, shift existing content visually, and
      //    auto-mask the newly exposed strip so the worker fills it.
      // -------------------------------------------------------------------------
      function outpaintExtend(edge) {
        var m = cfg.outpaintMargin;
        var w = num(get("params.width"), 1024), h = num(get("params.height"), 1024);
        var b = get("canvas.bbox") || { x: 0, y: 0, width: w, height: h };
        var nb = { x: num(b.x, 0), y: num(b.y, 0), width: num(b.width, w), height: num(b.height, h) };
        var op = get("params.outpaint") || { top: 0, right: 0, bottom: 0, left: 0 };
        op = { top: op.top || 0, right: op.right || 0, bottom: op.bottom || 0, left: op.left || 0 };

        if (edge === "top")    { h += m; nb.height += m; nb.y -= m; op.top += m; }
        if (edge === "bottom") { h += m; nb.height += m; op.bottom += m; }
        if (edge === "left")   { w += m; nb.width += m; nb.x -= m; op.left += m; }
        if (edge === "right")  { w += m; nb.width += m; op.right += m; }

        // snap to /8 for diffusion
        w = Math.round(w / 8) * 8; h = Math.round(h / 8) * 8;
        set("params.width", w);
        set("params.height", h);
        set("canvas.bbox", nb);
        set("params.outpaint", op);
        // outpaint requires an init image (the original) to extend FROM
        set("params.outpaint_enabled", true);

        // auto-mask the new strip: a white rect over the just-added region (in local
        // coords of the new frame). Recompute frame after the bbox change.
        setTimeout(function () {
          var f = genFrame();
          if (!f) return;
          // strip rect in screen space for the extended edge
          var sc = stage ? (stage.scaleX() || 1) : 1;
          var stripScreen = null;
          if (edge === "top")    stripScreen = { x: f.x, y: f.y, w: f.w, h: m * sc };
          if (edge === "bottom") stripScreen = { x: f.x, y: f.y + f.h - m * sc, w: f.w, h: m * sc };
          if (edge === "left")   stripScreen = { x: f.x, y: f.y, w: m * sc, h: f.h };
          if (edge === "right")  stripScreen = { x: f.x + f.w - m * sc, y: f.y, w: m * sc, h: f.h };
          if (!stripScreen) return;
          var l0 = localPoint({ x: stripScreen.x, y: stripScreen.y });
          var l1 = localPoint({ x: stripScreen.x + stripScreen.w, y: stripScreen.y + stripScreen.h });
          var r = new Konva.Rect({
            x: Math.min(l0.x, l1.x), y: Math.min(l0.y, l1.y),
            width: Math.abs(l1.x - l0.x), height: Math.abs(l1.y - l0.y),
            fill: cfg.maskColor, listening: false, name: "mp-paint",
          });
          strokeGroup.add(r);
          maskLayer.batchDraw();
        }, 30);
        bus.emit("maskPaint:outpaint", { edge: edge, margin: m, width: w, height: h });
      }

      // -------------------------------------------------------------------------
      // 7) EXPORT — rasterize the painted mask to a black/white PNG at the
      //    generation dims, then wire into the existing inpaint path.
      // -------------------------------------------------------------------------
      function rasterizeMask(frame, recolorWhite) {
        if (!strokeGroup || !stage) return null;
        var iw = Math.max(1, Math.round(frame.iw || frame.w));
        var ih = Math.max(1, Math.round(frame.ih || frame.h));
        // Clone the stroke group into an offscreen stage sized to the frame so we
        // capture ONLY the mask region at full image resolution, regardless of zoom.
        var off = document.createElement("canvas");
        off.width = iw; off.height = ih;
        var g = off.getContext("2d");
        g.fillStyle = "#000"; g.fillRect(0, 0, iw, ih); // black = preserve

        // Render the live mask layer to a canvas, then blit the frame sub-rect into
        // the output at image scale. maskLayer.toCanvas gives us screen-space pixels.
        // Neutralize the on-screen display opacity (0.5) for the capture so painted
        // strokes read at full strength, then restore it.
        var src;
        var savedOpacity = maskLayer.opacity();
        try {
          maskLayer.opacity(1);
          src = maskLayer.toCanvas({
            x: frame.x, y: frame.y, width: frame.w, height: frame.h, pixelRatio: 1,
          });
        } catch (e) { console.warn("[maskPaint] toCanvas failed", e); return null; }
        finally { maskLayer.opacity(savedOpacity); maskLayer.batchDraw(); }
        // scale the captured frame region to the image dims
        g.drawImage(src, 0, 0, src.width, src.height, 0, 0, iw, ih);

        // Binarize to white-on-black by LUMINANCE. We composited the mask over an
        // opaque black fill, so alpha is uniformly 255 here — keying on alpha would
        // mark the whole frame. Painted strokes (white/red) have high luminance;
        // unpainted area is black (luminance 0). Threshold at a low luminance so
        // soft feathered edges still count as masked.
        var id = g.getImageData(0, 0, iw, ih);
        var px = id.data;
        for (var i = 0; i < px.length; i += 4) {
          // Rec. 601 luma; red maskColor (~0.3*255≈76) and white (255) both clear 24.
          var lum = px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114;
          var v = lum > 24 ? 255 : 0;
          px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
        }
        g.putImageData(id, 0, 0);
        if (recolorWhite === false) return id; // caller wants the raw ImageData
        return off; // canvas
      }

      function applyMask() {
        var f = genFrame();
        if (!f || f.w <= 2) { console.warn("[maskPaint] no generation frame to mask"); return; }
        var off = rasterizeMask(f, true);
        if (!off) return;
        var dataURL = off.toDataURL("image/png");
        var b64 = dataURL.split(",")[1] || "";

        off.toBlob(function (blob) {
          // (a) feed the existing inpaint LAYER so generate_ws picks it up via
          //     layerToBlob -> api.uploadMask -> SetLatentNoiseMask. If the user
          //     never created a mask layer we create a minimal one ourselves so
          //     the inpaint graph path fires (matches the layers.js layer shape).
          var maskLy = ensureMaskLayer();
          if (maskLy) {
            maskLy.blob = blob;
            maskLy.dataURL = dataURL;
            maskLy.uploadedName = null; // force re-upload of the fresh mask
            maskLy.visible = true;
            bus.emit("layers:changed", (get("layers") || []).slice());
          }

          // (b) write params for the orchestrator-forwarded backend fields. The
          //     worker's inpaint path is path-based (init_image+mask_image); a
          //     data-URI/upload sink is needed server-side to land this as a path.
          set("params.mask_data", b64);            // base64 PNG (no data: prefix)
          set("params.mask_channel", "luminance"); // white=regen
          set("params.mask_mime", "image/png");
          set("params.denoise", num(get("params.denoise"), 1.0)); // full regen in masked area
          bus.emit("maskPaint:applied", { dataURL: dataURL, hasLayer: !!maskLy });
          flash("Mask applied" + (maskLy ? "" : " (exported only — no layer model)"));
        }, "image/png");
      }

      function findMaskLayer() {
        var layers = get("layers") || [];
        for (var i = 0; i < layers.length; i++) {
          if (layers[i] && layers[i].type === "mask") return layers[i];
        }
        return null;
      }

      // Return the existing mask layer, or create a minimal one in state.layers so
      // generate_ws's SetLatentNoiseMask path consumes our exported PNG. We mutate
      // state.layers directly (the layers module exposes no create-bus hook) and
      // let layers.js re-render off the 'layers:changed' broadcast.
      function ensureMaskLayer() {
        var existing = findMaskLayer();
        if (existing) return existing;
        if (!state.layers || !state.layers.push) {
          try { state.layers = state.layers || []; } catch (_) { return null; }
        }
        var ly = {
          id: "mask-paint-" + Date.now().toString(36),
          type: "mask",
          name: "Inpaint Mask",
          visible: true,
          opacity: 1,
          locked: false,
        };
        // newest on top (matches layers.js unshift ordering)
        state.layers.unshift(ly);
        return ly;
      }

      // -------------------------------------------------------------------------
      // small UI helpers
      // -------------------------------------------------------------------------
      function flash(msg) {
        var host = dom.canvasStatusbar;
        if (!host) { console.info("[maskPaint]", msg); return; }
        var el = host.querySelector("#mp-flash");
        if (!el) { el = document.createElement("span"); el.id = "mp-flash"; host.appendChild(el); }
        el.textContent = msg;
        el.style.opacity = "1";
        clearTimeout(el._t);
        el._t = setTimeout(function () { el.style.opacity = "0"; }, 2200);
      }

      function num(v, d) { var n = Number(v); return Number.isFinite(n) ? n : d; }

      function hexToRgb(hex) {
        var h = String(hex || "").replace("#", "");
        if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        var n = parseInt(h, 16);
        if (!Number.isFinite(n)) return { r: 255, g: 59, b: 92 };
        return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
      }

      function injectCss() {
        if (document.getElementById("style-maskPaint")) return;
        var css = [
          "#canvas-toolbar #mp-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;",
            "font-size:12px;color:var(--text)}",
          "#mp-toolbar .mp-title{font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-size:10px}",
          "#mp-toolbar .mp-sep{width:1px;height:18px;background:var(--line)}",
          "#mp-toolbar .mp-ctl{display:flex;align-items:center;gap:5px;color:var(--muted)}",
          "#mp-toolbar .mp-ctl input[type=range]{width:90px;accent-color:var(--accent)}",
          "#mp-toolbar .mp-ctl input[type=number]{width:58px;padding:3px}",
          "#mp-toolbar #mp-size-val{min-width:22px;text-align:right;color:var(--text)}",
          "#mp-toolbar .mp-b{padding:3px 9px;font-size:12px}",
          "#mp-toolbar .mp-out{padding:3px 7px;min-width:26px;font-weight:700}",
          "#tooldock .mp-tool-btn{width:36px;height:36px;display:flex;align-items:center;justify-content:center;",
            "background:transparent;border:1px solid transparent;border-radius:8px;color:var(--muted);cursor:pointer;padding:0}",
          "#tooldock .mp-tool-btn:hover{color:var(--text);border-color:var(--line);background:var(--panel2)}",
          "#tooldock .mp-tool-btn.active{color:#fff;background:var(--accent2);border-color:var(--accent)}",
          "#canvas-statusbar #mp-flash{margin-left:10px;color:var(--ok);font-size:11px;transition:opacity .4s;opacity:0}",
        ].join("\n");
        var st = document.createElement("style");
        st.id = "style-maskPaint"; st.textContent = css;
        document.head.appendChild(st);
      }

      console.info("[maskPaint] ready");
    },
  });
})();
