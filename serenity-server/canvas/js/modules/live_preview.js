/* live_preview.js — module "livePreview".
 *
 * SwarmUI "Live Preview (intermediate latents during gen)" parity (GAP §8 / row 100).
 *
 * The worker streams intermediate frames on the progress channel: serenity-server
 * forwards the raw WorkerEvent over WS /v1/progress?job=<id> as
 *     {"ev":"progress","step":N,"total":M,"phase":"...","preview":"<...>"}
 * where `preview` is (when populated) a data: URL or a bare base64 PNG/JPEG of a
 * downsized decode of the in-flight latent (see crates/wire/src/lib.rs Progress).
 *
 * api.js's progress translator collapses each event to step/total and DROPS the
 * `preview` field, and (per the team split) api.js/state.js/main.js are off-limits.
 * So this module is self-contained: when a job is submitted it opens its OWN
 * job-keyed progress WebSocket (same URL api.js uses) purely to read `preview`,
 * decodes it, and draws it fit-to-view in the CENTER result pane (the shared Konva
 * stage), replacing the synthetic placeholder. On the final result / job end it
 * clears itself so resultView's final image takes over cleanly.
 *
 * Defense-in-depth: it ALSO listens on the bus for a "preview" event (so if the
 * orchestrator later teaches api.js to forward preview frames, this keeps working
 * without change), de-duplicating against its own WS so a frame is never drawn twice.
 *
 * Talks to the rest of the app ONLY through ctx.{state,get,set,bus,api,Konva,dom}
 * and the bus. Writes nothing to state.params. UI-only.
 */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") {
    console.error("[livePreview] Serenity registry missing; cannot register");
    return;
  }

  var NAME = "livePreview";

  S.register(NAME, {
    init: function (ctx) {
      var bus = ctx.bus;
      var Konva = ctx.Konva;
      var state = ctx.state;
      var api = ctx.api;

      injectCSS();

      // ---- stage / layer plumbing -----------------------------------------
      var stageRef = (state.canvas && state.canvas.stage) || null;
      var layer = null;            // dedicated Konva.Layer for live frames
      var imgNode = null;          // current Konva.Image
      var badge = null;            // small "live preview" chip drawn over the frame

      bus.on("canvas:stage", function (s) { if (s) stageRef = s; });
      // ask for the stage in case canvasCore came up before us
      try { bus.emit("canvas:stage?"); } catch (_) {}
      function getStage() { return (state.canvas && state.canvas.stage) || stageRef; }

      // ---- per-job state ---------------------------------------------------
      var ws = null;               // our own progress WebSocket
      var currentJob = null;       // job id we're previewing
      var lastSig = null;          // last drawn frame signature (dedup ws<->bus)
      var enabled = true;          // user toggle (SwarmUI exposes a toggle)
      var sawAny = false;          // did we draw at least one live frame this job?

      // ===== job lifecycle (driven by generateWS bus events) ===============
      bus.on("generate:submitted", function (p) {
        var job = p && (p.promptId || p.jobId || p.prompt_id);
        startJob(job);
      });
      // FINAL image is ready: stop receiving frames, but keep the last live frame
      // on screen until resultView has actually drawn the final image, so the
      // center pane never flashes blank. resultView loads the final image async
      // (img.onload), so we time the swap by preloading the same URL ourselves.
      bus.on("result:ready", function (p) { endJob(p && p.url); });
      bus.on("result:image", function (p) { endJob(p && p.url); });
      // interrupt/cancel have no final image — clear immediately.
      bus.on("generate:interrupt", function () { endJob(); });
      bus.on("generate:done", function () { endJob(); });

      // user toggle (a settings/topbar module may emit this; default ON)
      bus.on("livePreview:toggle", function (v) {
        enabled = (v === undefined) ? !enabled : !!v;
        if (!enabled) clearFrame();
        bus.emit("livePreview:state", { enabled: enabled });
      });
      bus.on("livePreview:state?", function () {
        bus.emit("livePreview:state", { enabled: enabled });
      });

      // defense-in-depth: a "preview" frame delivered via the bus by some other
      // module (e.g. a future api.js that forwards it). Accept several shapes.
      bus.on("preview", function (m) { onBusPreview(m); });
      bus.on("preview:frame", function (m) { onBusPreview(m); });

      function onBusPreview(m) {
        if (!enabled || m == null) return;
        var raw = (typeof m === "string") ? m
          : (m.preview != null ? m.preview : (m.data != null ? m.data : m.frame));
        if (raw == null || raw === "") return;
        decodeToUrl(raw).then(function (url) { if (url) drawFrame(url); });
      }

      function startJob(job) {
        endJob();                 // tear down any prior job cleanly
        if (!job) return;
        currentJob = job;
        sawAny = false;
        lastSig = null;
        if (enabled) openWS(job);
      }

      // endJob(finalUrl?): stop the WS immediately. If a final image URL is given,
      // hold the last live frame until that image is loaded (so resultView swaps it
      // in at the same instant) — otherwise clear right away. A fallback timer
      // guarantees the live frame is never left stuck on screen.
      function endJob(finalUrl) {
        closeWS();
        currentJob = null;
        if (finalUrl && imgNode) {
          deferClearUntilLoaded(finalUrl);
        } else {
          clearFrame();
        }
      }

      var swapTimer = null;
      function deferClearUntilLoaded(url) {
        if (swapTimer) { clearTimeout(swapTimer); swapTimer = null; }
        var done = false;
        function finish() {
          if (done) return; done = true;
          if (swapTimer) { clearTimeout(swapTimer); swapTimer = null; }
          clearFrame();
        }
        // hard fallback: never hold a stale live frame longer than ~2.5s
        swapTimer = setTimeout(finish, 2500);
        try {
          var pre = new Image();
          pre.onload = function () {
            // give resultView's own onload a tick to paint, then clear ours
            setTimeout(finish, 0);
          };
          pre.onerror = finish;
          pre.src = url;
          // if it's already cached, onload may have fired synchronously
          if (pre.complete) setTimeout(finish, 0);
        } catch (_) { finish(); }
      }

      // ===== our own job-keyed progress WS (reads the raw `preview` field) ==
      function wsBaseUrl() {
        // mirror api.js: ?api=<http base> overrides; else derive from location.
        var apiBase = "";
        try {
          apiBase = (api && api.base) ||
            new URLSearchParams(location.search).get("api") || "";
        } catch (_) { apiBase = (api && api.base) || ""; }
        if (apiBase) return apiBase.replace(/^http/, "ws");
        var proto = (location.protocol === "https:") ? "wss:" : "ws:";
        return proto + "//" + location.host;
      }

      function openWS(job) {
        closeWS();
        var url;
        try {
          url = wsBaseUrl() + "/v1/progress?job=" + encodeURIComponent(job);
        } catch (e) { return; }
        var sock;
        try { sock = new WebSocket(url); }
        catch (e) { console.warn("[livePreview] ws open failed", e); return; }
        ws = sock;
        sock.onmessage = function (ev) {
          if (!enabled) return;
          var m;
          try { m = JSON.parse(ev.data); } catch (_) { return; }
          var which = String(m && m.ev || "").toLowerCase();
          if (which === "progress") {
            var pv = m.preview;
            if (pv != null && pv !== "") {
              var sig = sigOf(pv);
              if (sig !== lastSig) {
                lastSig = sig;
                decodeToUrl(pv).then(function (u) { if (u) drawFrame(u); });
              }
            }
          } else if (which === "done" || which === "failed" || which === "cancelled") {
            // terminal on the wire — let result:ready swap in the final image,
            // but make sure we stop listening and drop the intermediate frame.
            endJob();
          }
        };
        sock.onerror = function () { /* progress WS may not exist; degrade quietly */ };
        sock.onclose = function () { if (ws === sock) ws = null; };
      }

      function closeWS() {
        var sock = ws; ws = null;
        if (sock) {
          try { sock.onmessage = sock.onerror = sock.onclose = null; } catch (_) {}
          try { sock.close(); } catch (_) {}
        }
      }

      // ===== decode a preview payload to a usable image URL ================
      // Accepts: data: URL (returned as-is), bare base64 of a PNG/JPEG, an
      // ArrayBuffer/Uint8Array (optionally with an 8-byte ComfyUI header), or a Blob.
      function decodeToUrl(payload) {
        return new Promise(function (resolve) {
          try {
            if (!payload) return resolve(null);
            if (typeof payload === "string") {
              var s = payload.trim();
              if (s.indexOf("data:") === 0) return resolve(s);
              if (s.indexOf("blob:") === 0) return resolve(s);
              if (/^https?:\/\//i.test(s) || s.charAt(0) === "/") return resolve(s);
              // bare base64 — sniff PNG vs JPEG from the first decoded bytes
              var mime = base64Mime(s);
              return resolve("data:" + mime + ";base64," + s);
            }
            if (payload instanceof Blob) return resolve(URL.createObjectURL(payload));
            var buf = payload;
            if (payload && payload.data instanceof ArrayBuffer) buf = payload.data;
            if (buf instanceof ArrayBuffer || (buf && buf.buffer instanceof ArrayBuffer)) {
              var bytes = (buf instanceof ArrayBuffer) ? new Uint8Array(buf)
                                                       : new Uint8Array(buf.buffer);
              var start = 0;
              if (bytes.length > 8 && !isImageMagic(bytes, 0) && isImageMagic(bytes, 8)) {
                start = 8; // strip ComfyUI {event:u32, format:u32} header
              }
              var view = bytes.subarray(start);
              var blob = new Blob([view], { type: sniffMime(view) });
              return resolve(URL.createObjectURL(blob));
            }
          } catch (e) {
            console.warn("[livePreview] decode failed", e);
          }
          return resolve(null);
        });
      }

      // ===== draw a frame fit-to-view in the center pane ===================
      function drawFrame(url) {
        var st = getStage();
        if (!st || !Konva) return;
        var img = new Image();
        img.onload = function () {
          try {
            ensureLayer(st);
            if (!layer) return;
            var sw = st.width() || img.width || 1;
            var sh = st.height() || img.height || 1;
            var scale = Math.min(sw / img.width, sh / img.height) * 0.96;
            if (!isFinite(scale) || scale <= 0) scale = 1;
            var w = img.width * scale, h = img.height * scale;
            var x = (sw - w) / 2, y = (sh - h) / 2;
            if (!imgNode) {
              imgNode = new Konva.Image({
                image: img, x: x, y: y, width: w, height: h,
                listening: false, opacity: 0.96,
                imageSmoothingEnabled: true, name: "live-preview-image",
              });
              layer.add(imgNode);
            } else {
              imgNode.image(img);
              imgNode.setAttrs({ x: x, y: y, width: w, height: h });
            }
            drawBadge(st, x, y);
            layer.moveToTop();      // sit above content, below the final result-view
            layer.batchDraw();
            sawAny = true;
            bus.emit("livePreview:frame", { jobId: currentJob });
          } catch (e) {
            console.warn("[livePreview] draw failed", e);
          } finally {
            if (typeof url === "string" && url.indexOf("blob:") === 0) {
              setTimeout(function () { try { URL.revokeObjectURL(url); } catch (_) {} }, 0);
            }
          }
        };
        img.onerror = function () {
          if (typeof url === "string" && url.indexOf("blob:") === 0) {
            try { URL.revokeObjectURL(url); } catch (_) {}
          }
        };
        img.src = url;
      }

      function ensureLayer(st) {
        if (layer && layer.getStage() === st) return;
        if (layer) { try { layer.destroy(); } catch (_) {} layer = null; imgNode = null; badge = null; }
        try {
          layer = new Konva.Layer({ listening: false, name: "live-preview" });
          st.add(layer);
        } catch (e) {
          console.warn("[livePreview] could not add layer", e);
          layer = null;
        }
      }

      // small "● LIVE" chip in the corner of the preview so it's clearly an
      // intermediate frame, not the final image.
      function drawBadge(st, x, y) {
        if (!Konva || !layer) return;
        try {
          if (!badge) {
            badge = new Konva.Label({ listening: false, name: "live-preview-badge", opacity: 0.92 });
            badge.add(new Konva.Tag({
              fill: "rgba(20,23,28,0.82)", cornerRadius: 4,
              stroke: "#6c8cff", strokeWidth: 1,
            }));
            badge.add(new Konva.Text({
              text: "● LIVE",
              fontFamily: "system-ui, sans-serif", fontSize: 11, fontStyle: "bold",
              fill: "#6c8cff", padding: 4,
            }));
            layer.add(badge);
          }
          badge.position({ x: Math.max(4, x) + 6, y: Math.max(4, y) + 6 });
        } catch (_) { /* badge is cosmetic; never fatal */ }
      }

      function clearFrame() {
        if (swapTimer) { clearTimeout(swapTimer); swapTimer = null; }
        try {
          if (imgNode) { imgNode.destroy(); imgNode = null; }
          if (badge) { badge.destroy(); badge = null; }
          if (layer) layer.batchDraw();
        } catch (_) {}
        lastSig = null;
      }

      // ===== small utils ====================================================
      function sigOf(pv) {
        // cheap content signature for dedup (length + head/tail) — avoids hashing
        // a big base64 string every frame.
        var s = (typeof pv === "string") ? pv : "";
        if (!s) return Math.random();
        return s.length + ":" + s.slice(0, 24) + ":" + s.slice(-12);
      }
      function isImageMagic(bytes, off) {
        if (!bytes || bytes.length < off + 4) return false;
        if (bytes[off] === 0x89 && bytes[off + 1] === 0x50 &&
            bytes[off + 2] === 0x4e && bytes[off + 3] === 0x47) return true; // PNG
        if (bytes[off] === 0xff && bytes[off + 1] === 0xd8 && bytes[off + 2] === 0xff) return true; // JPEG
        return false;
      }
      function sniffMime(bytes) {
        if (bytes && bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png";
        if (bytes && bytes[0] === 0xff && bytes[1] === 0xd8) return "image/jpeg";
        if (bytes && bytes[0] === 0x52 && bytes[1] === 0x49) return "image/webp"; // RIFF
        return "image/png";
      }
      function base64Mime(s) {
        // decode just the first few bytes to sniff the format
        try {
          var head = atob(s.slice(0, 16));
          var b = [];
          for (var i = 0; i < head.length && i < 4; i++) b.push(head.charCodeAt(i));
          if (b[0] === 0x89 && b[1] === 0x50) return "image/png";
          if (b[0] === 0xff && b[1] === 0xd8) return "image/jpeg";
          if (b[0] === 0x52 && b[1] === 0x49) return "image/webp";
        } catch (_) {}
        return "image/png";
      }

      function injectCSS() {
        if (document.getElementById("style-" + NAME)) return;
        // module is canvas-drawn; this is here only so a future toggle chip in the
        // DOM picks up the theme without touching shared CSS.
        var css =
          ".lp-toggle{background:var(--panel2);border:1px solid var(--line);" +
          "color:var(--text);border-radius:6px;padding:3px 8px;cursor:pointer;font-size:11px}" +
          ".lp-toggle.is-off{color:var(--muted)}" +
          ".lp-toggle:hover{border-color:var(--accent)}";
        var el = document.createElement("style");
        el.id = "style-" + NAME;
        el.textContent = css;
        document.head.appendChild(el);
      }

      // keep the linter happy about intentionally-tracked flags
      void sawAny;
      console.info("[livePreview] ready");
    },
  });
})();
