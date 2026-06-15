/* canvas_core.js — module "canvasCore". Owns the ONE Konva.Stage.
   Design B: the Konva canvas is ALWAYS present. This module creates the single
   shared stage (3 layers: checkerboard bg / content / overlay), wires pan
   (space-drag) + zoom (wheel toward cursor), tracks the host element size, and
   publishes the stage so every other module can attach Groups to it.

   Contract:
     - register once as 'canvasCore'
     - mounts into ctx.dom.konvaStage (#konva-stage, fills #canvas-host)
     - sets ctx.state.canvas.stage = stage AND bus.emit('canvas:stage', stage)
     - FEW Layers, MANY Groups: exposes content/overlay/bg layers via state+bus
     - degrades gracefully (no API calls here; never throws to a fatal point)
*/
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") {
    console.error("[canvasCore] Serenity registry missing; cannot register");
    return;
  }

  S.register("canvasCore", {
    init: function (ctx) {
      try {
        boot(ctx);
      } catch (e) {
        // Never let a render error kill the whole app boot.
        console.error("[canvasCore] init failed", e);
      }
    },
  });

  // ---- tunables ----
  var CHECKER = 16;          // checkerboard cell size in screen px
  var ZOOM_MIN = 0.05;
  var ZOOM_MAX = 32;
  var ZOOM_STEP = 1.06;      // wheel scale factor per notch

  function boot(ctx) {
    var state = ctx.state;
    var bus = ctx.bus;
    var set = ctx.set;
    var Konva = ctx.Konva;
    var mount = ctx.dom && ctx.dom.konvaStage;

    if (!Konva) {
      console.error("[canvasCore] Konva library not loaded");
      return;
    }
    if (!mount) {
      console.error("[canvasCore] mount element #konva-stage not found");
      return;
    }

    injectCss(mount.id || "konva-stage");

    // Size from the mount element. #konva-stage is position:absolute;inset:0 so
    // it fills #canvas-host; clientWidth/Height give us the live pixel size.
    var size = measure(mount);

    var stage = new Konva.Stage({
      container: mount,
      width: size.w,
      height: size.h,
      draggable: false, // pan is gated behind space-drag (see below)
    });

    // ---- layer architecture: FEW layers, MANY groups ----
    // 1) checkerboard background — non-interactive, drawn in SCREEN space so it
    //    always tiles the whole viewport regardless of pan/zoom.
    var bgLayer = new Konva.Layer({ listening: false, name: "bg" });
    var bgRect = new Konva.Rect({ x: 0, y: 0, width: size.w, height: size.h });
    bgLayer.add(bgRect);

    // 2) content — where every other module adds its Groups (raster/control/etc).
    var contentLayer = new Konva.Layer({ name: "content" });

    // 3) overlay — transformers, bbox handles, brush cursors, selection chrome.
    var overlayLayer = new Konva.Layer({ name: "overlay" });

    stage.add(bgLayer);
    stage.add(contentLayer);
    stage.add(overlayLayer);

    paintChecker();

    // ---- publish the stage + layers (state is the single source of truth) ----
    state.canvas = state.canvas || {};
    state.canvas.stage = stage;
    state.canvas.layers = {
      bg: bgLayer,
      content: contentLayer,
      overlay: overlayLayer,
    };
    // keep zoom mirror in state in sync
    state.canvas.zoom = stage.scaleX();

    // Late subscribers (modules that init after us, or re-init) can ask for it.
    bus.on("canvas:stage?", function () {
      bus.emit("canvas:stage", stage);
      bus.emit("canvas:layers", state.canvas.layers);
    });
    bus.emit("canvas:stage", stage);
    bus.emit("canvas:layers", state.canvas.layers);

    // =========================================================================
    // Checkerboard: a tiled radial/grid pattern rendered to an offscreen canvas
    // once, then applied as a repeating fillPattern on a full-viewport rect.
    // Drawn in screen space (rect not transformed) so panning the content does
    // not move the backdrop — feels like an infinite design surface.
    // =========================================================================
    var patternCanvas = buildCheckerPattern();
    bgRect.fillPatternImage(patternCanvas);
    bgRect.fillPatternRepeat("repeat");

    function paintChecker() {
      bgRect.width(stage.width());
      bgRect.height(stage.height());
      bgLayer.batchDraw();
    }

    // =========================================================================
    // Pan: hold SPACE then drag (Figma/Konva convention). We toggle the stage's
    // own draggable while space is held, and swap the cursor for feedback.
    // =========================================================================
    var spaceDown = false;
    var panning = false;

    function setCursor(c) {
      var el = stage.container();
      if (el) el.style.cursor = c;
    }

    window.addEventListener("keydown", function (e) {
      if (e.code === "Space" && !spaceDown) {
        // Don't hijack space while typing into an input/textarea.
        var t = e.target;
        var tag = t && t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || (t && t.isContentEditable)) return;
        spaceDown = true;
        stage.draggable(true);
        setCursor("grab");
        e.preventDefault();
      }
    });
    window.addEventListener("keyup", function (e) {
      if (e.code === "Space") {
        spaceDown = false;
        if (!panning) {
          stage.draggable(false);
          setCursor("default");
        }
      }
    });

    stage.on("dragstart", function () {
      if (spaceDown) {
        panning = true;
        setCursor("grabbing");
      } else {
        // A drag started without space (e.g. a child started it); leave stage
        // pan off so tools/transformers keep working.
        stage.draggable(false);
      }
    });
    stage.on("dragmove", function () {
      // backdrop stays put (screen space); just publish camera changes.
      emitCamera();
    });
    stage.on("dragend", function () {
      panning = false;
      stage.draggable(spaceDown);
      setCursor(spaceDown ? "grab" : "default");
      emitCamera();
    });

    // =========================================================================
    // Zoom: wheel zooms toward the pointer, clamped. Holds the world point under
    // the cursor fixed so zooming feels anchored.
    // =========================================================================
    stage.on("wheel", function (e) {
      e.evt.preventDefault();
      var oldScale = stage.scaleX();
      var pointer = stage.getPointerPosition();
      if (!pointer) return;

      var worldPoint = {
        x: (pointer.x - stage.x()) / oldScale,
        y: (pointer.y - stage.y()) / oldScale,
      };

      var dir = e.evt.deltaY > 0 ? -1 : 1; // wheel up = zoom in
      var newScale = dir > 0 ? oldScale * ZOOM_STEP : oldScale / ZOOM_STEP;
      newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newScale));
      if (newScale === oldScale) return;

      stage.scale({ x: newScale, y: newScale });
      stage.position({
        x: pointer.x - worldPoint.x * newScale,
        y: pointer.y - worldPoint.y * newScale,
      });
      emitCamera();
    });

    function emitCamera() {
      // Mirror camera into state (zoom) and notify pan/zoom subscribers.
      state.canvas.zoom = stage.scaleX();
      var cam = { x: stage.x(), y: stage.y(), scale: stage.scaleX() };
      // write zoom through the reactive setter so change:* fires for listeners
      try { set("canvas.zoom", cam.scale); } catch (_) {}
      bus.emit("canvas:camera", cam);
      contentLayer.batchDraw();
      overlayLayer.batchDraw();
    }

    // Programmatic camera control for other modules (fit/reset/center).
    bus.on("canvas:reset", function () {
      stage.scale({ x: 1, y: 1 });
      stage.position({ x: 0, y: 0 });
      emitCamera();
    });
    bus.on("canvas:zoomTo", function (payload) {
      var s = payload && typeof payload.scale === "number" ? payload.scale : 1;
      s = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, s));
      stage.scale({ x: s, y: s });
      emitCamera();
    });

    // =========================================================================
    // Resize: track the host element. ResizeObserver where available, window
    // resize as a universal fallback.
    // =========================================================================
    function resize() {
      var m = measure(mount);
      if (m.w <= 0 || m.h <= 0) return;
      if (m.w === stage.width() && m.h === stage.height()) return;
      stage.width(m.w);
      stage.height(m.h);
      paintChecker();
      bus.emit("canvas:resize", { width: m.w, height: m.h });
      contentLayer.batchDraw();
      overlayLayer.batchDraw();
    }

    var ro = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(function () { resize(); });
      // Observe the host (parent) since #konva-stage is inset:0 inside it.
      ro.observe(mount.parentElement || mount);
      ro.observe(mount);
    }
    window.addEventListener("resize", resize);

    // One delayed resize in case the grid lays out after our init (fonts/layout).
    setTimeout(resize, 0);
    setTimeout(resize, 120);

    // ---- helpers ----------------------------------------------------------

    function buildCheckerPattern() {
      var c = document.createElement("canvas");
      var s = CHECKER * 2;
      c.width = s;
      c.height = s;
      var g = c.getContext("2d");
      // Two subtle greys that read on the dark --bg.
      var a = "#202329";
      var b = "#1a1d23";
      g.fillStyle = a;
      g.fillRect(0, 0, s, s);
      g.fillStyle = b;
      g.fillRect(0, 0, CHECKER, CHECKER);
      g.fillRect(CHECKER, CHECKER, CHECKER, CHECKER);
      return c;
    }

    console.info("[canvasCore] stage ready", stage.width() + "x" + stage.height());
  }

  function measure(el) {
    // Prefer the parent host's box (the stage container is inset:0 within it).
    var host = el.parentElement || el;
    var w = el.clientWidth || host.clientWidth || 0;
    var h = el.clientHeight || host.clientHeight || 0;
    if (!w || !h) {
      var r = host.getBoundingClientRect();
      w = w || Math.round(r.width);
      h = h || Math.round(r.height);
    }
    return { w: w, h: h };
  }

  function injectCss(rootId) {
    var id = "style-canvasCore";
    if (document.getElementById(id)) return;
    var st = document.createElement("style");
    st.id = id;
    // Scoped under our mount id. Make sure the stage container fills the host
    // and shows a sensible default cursor; Konva injects its own .konvajs-content.
    st.textContent =
      "#" + rootId + "{width:100%;height:100%;}" +
      "#" + rootId + " .konvajs-content{position:absolute;inset:0;}" +
      "#" + rootId + " canvas{display:block;}";
    document.head.appendChild(st);
  }
})();
