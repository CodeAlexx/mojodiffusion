/* bbox.js — module "bbox" (Design B).
   Bounding box = the generation region. On bus "canvas:stage" we drop a
   Konva.Rect + Konva.Transformer onto an overlay layer of the shared stage.
   Two-way binding:
     - dragging / resizing the rect  -> state.canvas.bbox AND state.params.width/height
     - changing params.width/height  -> moves/resizes the rect (listen change:params.width/.height)
   Live dims are mirrored into #canvas-statusbar.

   Ownership rules (CONTRACT): we own ONLY this file. We talk to the rest of the
   app exclusively through ctx.state/get/set + ctx.bus. We never edit the shared
   stage's other layers; we keep our rect+transformer inside our own Group on an
   overlay layer. CSS (status pill) is injected scoped from init().

   Graceful degradation: canvas_core may not have shipped / may init after us, so
   we both (a) check state.canvas.stage right away and (b) subscribe to the
   'canvas:stage' bus event. We never throw if the stage is missing. */
(function () {
  "use strict";
  const S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  S.register("bbox", {
    init(ctx) {
      const { state, get, set, bus, Konva, dom } = ctx;

      // ---- guard: Konva must be present (vendor script). Degrade if not. ----
      if (!Konva) {
        console.warn("[bbox] Konva not available; bbox disabled.");
        return;
      }

      // ---- scoped CSS for the status pill (own id, scoped under host) ----
      injectStyle();

      // ---- internal handles (filled once a stage arrives) ----
      let stage = null;
      let layer = null;        // overlay layer we draw on (shared if one named 'overlay' exists)
      let group = null;        // our private group inside that layer
      let rect = null;         // the bbox rectangle
      let tr = null;           // the transformer
      let statusEl = null;     // status pill element in #canvas-statusbar

      // Re-entrancy guards so the two-way binding doesn't loop forever.
      let applyingFromState = false;   // true while we mutate the rect from state
      let applyingFromRect = false;    // true while we mutate state from the rect

      // Minimum sensible generation size (px) so the box can't collapse.
      const MIN = 64;

      // -------- helpers --------

      // Read current bbox geometry from state with sane fallbacks.
      // params.width/height are the AUTHORITATIVE generation dims and win over a
      // stale canvas.bbox; canvas.bbox supplies position (x/y) and a dims fallback
      // only when params are absent.
      function bboxFromState() {
        const b = (get("canvas.bbox") || {});
        const pw = get("params.width");
        const ph = get("params.height");
        const width = num(pw, num(b.width, 1024));
        const height = num(ph, num(b.height, 1024));
        return {
          x: num(b.x, 0),
          y: num(b.y, 0),
          width: clampMin(width),
          height: clampMin(height),
        };
      }

      function num(v, d) { const n = Number(v); return Number.isFinite(n) ? n : d; }
      function clampMin(v) { return Math.max(MIN, Math.round(v)); }

      // Round to the multiple of 8 that diffusion pipelines expect, but never
      // below MIN. Keeps params.width/height clean when the user drags.
      function snap8(v) { return Math.max(MIN, Math.round(v / 8) * 8); }

      // -------- status bar pill --------
      function ensureStatus() {
        const host = dom && dom.canvasStatusbar;
        if (!host) return null;
        let el = host.querySelector("#bbox-status");
        if (!el) {
          el = document.createElement("span");
          el.id = "bbox-status";
          host.appendChild(el);
        }
        return el;
      }

      function paintStatus(w, h) {
        if (!statusEl) statusEl = ensureStatus();
        if (statusEl) statusEl.textContent = "▭ " + Math.round(w) + " × " + Math.round(h) + " px";
      }

      // -------- stage / layer acquisition --------

      // Find an overlay layer named 'overlay' if canvas_core made one; else make
      // our own thin layer. Keeping our own group means we never disturb others.
      function resolveLayer() {
        if (!stage) return null;
        let lay = null;
        try {
          const named = stage.findOne(".overlay") || stage.findOne("#overlay");
          if (named && named.getClassName && named.getClassName() === "Layer") lay = named;
          if (!lay) {
            // any layer explicitly flagged as the overlay by name attribute
            const layers = stage.getLayers ? stage.getLayers() : [];
            for (let i = 0; i < layers.length; i++) {
              if (layers[i].name && layers[i].name() === "overlay") { lay = layers[i]; break; }
            }
          }
        } catch (_) { /* ignore find errors */ }
        if (!lay) {
          lay = new Konva.Layer({ name: "overlay-bbox", listening: true });
          stage.add(lay);
        }
        return lay;
      }

      function buildNodes() {
        const g = bboxFromState();

        group = new Konva.Group({ name: "bbox-group", draggable: false });

        rect = new Konva.Rect({
          name: "bbox-rect",
          x: g.x, y: g.y, width: g.width, height: g.height,
          stroke: "#6c8cff",
          strokeWidth: 1.5,
          strokeScaleEnabled: false,   // crisp 1.5px regardless of stage zoom
          dash: [6, 4],
          fill: "rgba(108,140,255,0.06)",
          draggable: true,
          cornerRadius: 2,
        });

        tr = new Konva.Transformer({
          name: "bbox-transformer",
          rotateEnabled: false,
          keepRatio: false,
          ignoreStroke: true,
          borderStroke: "#6c8cff",
          borderStrokeWidth: 1,
          anchorStroke: "#6c8cff",
          anchorFill: "#1c1f26",
          anchorSize: 9,
          anchorCornerRadius: 2,
          enabledAnchors: [
            "top-left", "top-center", "top-right",
            "middle-left", "middle-right",
            "bottom-left", "bottom-center", "bottom-right",
          ],
          // Block sub-minimum boxes during live resize.
          boundBoxFunc: function (oldBox, newBox) {
            if (newBox.width < MIN || newBox.height < MIN) return oldBox;
            return newBox;
          },
        });

        group.add(rect);
        layer.add(group);
        layer.add(tr);
        tr.nodes([rect]);

        wireRectEvents();
        layer.draw();
        paintStatus(g.width, g.height);
      }

      // -------- rect -> state --------
      function commitFromRect() {
        if (!rect) return;
        applyingFromRect = true;
        try {
          // Transformer scales the node; bake scale into width/height so stored
          // dims stay in clean pixels and the rect's scale resets to 1.
          const sX = rect.scaleX() || 1;
          const sY = rect.scaleY() || 1;
          let w = rect.width() * sX;
          let h = rect.height() * sY;
          w = snap8(w);
          h = snap8(h);
          rect.scaleX(1);
          rect.scaleY(1);
          rect.width(w);
          rect.height(h);

          const x = Math.round(rect.x());
          const y = Math.round(rect.y());

          // Single source of truth: canvas.bbox object + params dims.
          set("canvas.bbox", { x: x, y: y, width: w, height: h });
          // Only emit a width/height change if it actually changed (avoids churn).
          if (num(get("params.width"), null) !== w) set("params.width", w);
          if (num(get("params.height"), null) !== h) set("params.height", h);

          paintStatus(w, h);
          layer.batchDraw();
        } finally {
          applyingFromRect = false;
        }
      }

      function wireRectEvents() {
        // Live feedback while dragging/resizing.
        rect.on("dragmove", function () {
          if (!rect) return;
          // keep bbox.x/y in sync live, but don't fight a state-driven update
          if (applyingFromState) return;
          paintStatus(rect.width() * (rect.scaleX() || 1), rect.height() * (rect.scaleY() || 1));
        });
        rect.on("transform", function () {
          if (!rect || applyingFromState) return;
          paintStatus(rect.width() * (rect.scaleX() || 1), rect.height() * (rect.scaleY() || 1));
        });
        // Commit to state on release.
        rect.on("dragend", function () { if (!applyingFromState) commitFromRect(); });
        rect.on("transformend", function () { if (!applyingFromState) commitFromRect(); });

        // Pointer cursor affordance.
        rect.on("mouseenter", function () { setCursor("move"); });
        rect.on("mouseleave", function () { setCursor("default"); });
      }

      function setCursor(c) {
        try {
          const container = stage && stage.container && stage.container();
          if (container) container.style.cursor = c;
        } catch (_) { /* ignore */ }
      }

      // -------- state -> rect --------
      function applyFromState() {
        if (!rect) return;
        applyingFromState = true;
        try {
          const g = bboxFromState();
          rect.scaleX(1);
          rect.scaleY(1);
          rect.position({ x: g.x, y: g.y });
          rect.size({ width: g.width, height: g.height });
          if (tr) tr.forceUpdate();
          paintStatus(g.width, g.height);
          layer.batchDraw();
          // Keep canvas.bbox consistent with the (authoritative) params dims so
          // other modules reading state.canvas.bbox see the truth. Guarded by
          // applyingFromState so the change:canvas.bbox listener no-ops.
          const cur = get("canvas.bbox") || {};
          if (cur.x !== g.x || cur.y !== g.y || cur.width !== g.width || cur.height !== g.height) {
            set("canvas.bbox", { x: g.x, y: g.y, width: g.width, height: g.height });
          }
        } finally {
          applyingFromState = false;
        }
      }

      // -------- bring everything online once a stage exists --------
      function attach(stg) {
        if (!stg || stage === stg) return;   // already attached to this stage
        // If re-attaching to a new stage, drop old nodes cleanly.
        teardownNodes();
        stage = stg;
        layer = resolveLayer();
        if (!layer) { console.warn("[bbox] no overlay layer; bbox disabled."); return; }
        statusEl = ensureStatus();
        buildNodes();
        console.info("[bbox] attached to stage");
      }

      function teardownNodes() {
        try {
          if (tr) tr.destroy();
          if (rect) rect.destroy();
          if (group) group.destroy();
          // Only destroy the layer if WE created it (our private overlay).
          if (layer && layer.name && layer.name() === "overlay-bbox") layer.destroy();
        } catch (_) { /* ignore */ }
        tr = rect = group = null;
        // keep `layer` only if it's a shared one we didn't make
        if (layer && layer.name && layer.name() === "overlay-bbox") layer = null;
      }

      // -------- wire external events (state + bus) --------

      // params.width / params.height -> rect
      bus.on("change:params.width", function () { if (!applyingFromRect) applyFromState(); });
      bus.on("change:params.height", function () { if (!applyingFromRect) applyFromState(); });
      // an external module may set canvas.bbox wholesale (e.g. fit-to-image).
      // Ignore our own writes (guarded by both re-entrancy flags).
      bus.on("change:canvas.bbox", function () {
        if (!applyingFromRect && !applyingFromState) applyFromState();
      });

      // stage availability (canvas_core emits this; may fire before OR after us)
      bus.on("canvas:stage", function (stg) { attach(stg || get("canvas.stage")); });

      // If canvas_core already created+stashed the stage before we registered,
      // pick it up immediately.
      const existing = get("canvas.stage");
      if (existing) attach(existing);
      else paintStatus(num(get("params.width"), 1024), num(get("params.height"), 1024));

      // ----- scoped style injection -----
      function injectStyle() {
        if (document.getElementById("style-bbox")) return;
        const css = [
          "#canvas-statusbar #bbox-status{",
          "  display:inline-block;padding:2px 8px;border-radius:6px;",
          "  background:rgba(28,31,38,0.72);border:1px solid var(--line,#2c303a);",
          "  color:var(--text,#e6e8ee);font-size:11px;font-variant-numeric:tabular-nums;",
          "  letter-spacing:.02em;backdrop-filter:blur(2px);",
          "}",
        ].join("\n");
        const st = document.createElement("style");
        st.id = "style-bbox";
        st.textContent = css;
        document.head.appendChild(st);
      }
    },
  });
})();
