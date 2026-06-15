/* result_view.js — module 'resultView'. Shows the latest generated image BIG in the
   center canvas (SwarmUI/InvokeAI behavior), not just as a gallery thumbnail.
   Listens bus 'result:ready' {url,...}; draws a fit-to-view Konva.Image on a
   dedicated top layer of the shared stage. Own file; talks via bus/state only. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  S.register("resultView", {
    init: function (ctx) {
      var bus = ctx.bus, Konva = ctx.Konva, state = ctx.state;
      var stage = null, layer = null, node = null;

      bus.on("canvas:stage", function (s) { stage = s; });
      function getStage() { return (state.canvas && state.canvas.stage) || stage; }

      function show(url) {
        var st = getStage();
        if (!st || !Konva) { console.warn("[resultView] no stage yet"); return; }
        var img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = function () {
          try {
            if (!layer) { layer = new Konva.Layer({ name: "result-view" }); st.add(layer); }
            if (node) { node.destroy(); node = null; }
            var sw = st.width() || 1, sh = st.height() || 1;
            var scale = Math.min(sw / img.width, sh / img.height) * 0.96;
            if (!isFinite(scale) || scale <= 0) scale = 1;
            var w = img.width * scale, h = img.height * scale;
            node = new Konva.Image({ image: img, x: (sw - w) / 2, y: (sh - h) / 2, width: w, height: h });
            layer.add(node); layer.moveToTop(); layer.draw();
            console.info("[resultView] displayed", url);
          } catch (e) { console.warn("[resultView] draw failed", e); }
        };
        img.onerror = function () { console.warn("[resultView] image load failed", url); };
        img.src = url;
      }

      bus.on("result:ready", function (p) { if (p && p.url) show(p.url); });
      console.info("[resultView] ready");
    },
  });
})();
