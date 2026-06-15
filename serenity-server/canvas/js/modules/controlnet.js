/* controlnet.js — module "controlnet" (Design B).
 *
 * Responsibility (per CONTRACT.md):
 *   - For control-type layers, surface a preprocessor picker (depth/canny/openpose/...)
 *     in #canvas-toolbar whenever a control layer is the active layer.
 *   - On "Apply", grab the layer's current image as a dataURL and call
 *     api.preprocess(method, {image:dataURL}); set the returned processed map as the
 *     control layer's image.
 *   - Store the controlnet model + weight + begin/end percent on the layer.
 *   - List ControlNet models from api.objectInfo() when available.
 *   - Degrade gracefully when the backend 404s (the picker still works as a no-op
 *     stub; nothing crashes the app).
 *
 * Ownership rules: this file ONLY. We talk to the rest of the app through
 * Serenity.state / get / set / bus / api. The `layers` module owns state.layers and
 * state.activeLayerId; we read those and write back the *same* layer objects (mutating
 * controlnet config + image fields) then re-emit via set('layers', ...) so renderers
 * and the layers panel observe the change. We never edit another module's file; any
 * DOM we attach to a layer row is appended (augmentation), never a file edit.
 */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  S.register("controlnet", { init: init });

  // ---- preprocessor catalog (label, method id sent to api.preprocess) ----------
  // method ids match common ComfyUI / controlnet-aux preprocessors; the backend may
  // expose more via object_info — we merge those in when available.
  var PREPROCESSORS = [
    { id: "none",        label: "None (passthrough)" },
    { id: "canny",       label: "Canny (edges)" },
    { id: "depth",       label: "Depth (MiDaS)" },
    { id: "depth_zoe",   label: "Depth (Zoe)" },
    { id: "openpose",    label: "OpenPose" },
    { id: "dwpose",      label: "DWPose" },
    { id: "lineart",     label: "Lineart" },
    { id: "lineart_anime", label: "Lineart (anime)" },
    { id: "softedge",    label: "SoftEdge (HED)" },
    { id: "scribble",    label: "Scribble" },
    { id: "mlsd",        label: "MLSD (straight lines)" },
    { id: "normalmap",   label: "Normal map" },
    { id: "seg",         label: "Segmentation" },
    { id: "tile",        label: "Tile" },
  ];

  function init(ctx) {
    var dom = ctx.dom || {};
    var bus = ctx.bus, api = ctx.api, get = ctx.get, set = ctx.set;
    var toolbar = dom.canvasToolbar; // #canvas-toolbar (we own a slice of it)
    if (!toolbar) { console.warn("[controlnet] no canvas-toolbar mount; idle"); return; }

    injectCSS();

    // ---- our root inside the shared toolbar (scoped, removable) -----------------
    var root = document.createElement("div");
    root.id = "cn-toolbar";
    root.className = "cn-toolbar";
    root.style.display = "none"; // hidden until a control layer is active
    toolbar.appendChild(root);

    // discovered CN model list (filled async from object_info)
    var cnModels = [];
    var cnModelsLoaded = false;

    // build the picker UI once; we re-bind it to whatever control layer is active.
    var ui = buildPicker(root);

    // ----- model discovery (graceful) ------------------------------------------
    loadControlNetModels();

    function loadControlNetModels() {
      // ComfyUI exposes ControlNet checkpoints via object_info for the
      // ControlNetLoader node: input.required.control_net_name = [[names...]].
      var done = function (names) {
        cnModels = Array.isArray(names) ? names.filter(Boolean) : [];
        cnModelsLoaded = true;
        refreshModelOptions();
      };
      if (!api || typeof api.objectInfo !== "function") { done([]); return; }
      // Try the specific node first (smaller payload), fall back to the full dump.
      api.objectInfo("ControlNetLoader")
        .then(function (info) { done(extractCNNames(info, "ControlNetLoader")); })
        .catch(function () {
          api.objectInfo()
            .then(function (all) { done(extractCNNames(all, null)); })
            .catch(function () { done([]); }); // 404 / backend down → empty, no crash
        });
    }

    function extractCNNames(info, cls) {
      try {
        var node = cls ? info[cls] : (info && (info.ControlNetLoader || info.ControlNetLoaderAdvanced));
        if (!node && info && cls && info[cls] === undefined) node = info; // /object_info/CLS returns the node directly in some builds
        var req = node && node.input && node.input.required;
        if (req && req.control_net_name && Array.isArray(req.control_net_name)) {
          var first = req.control_net_name[0];
          if (Array.isArray(first)) return first;
        }
      } catch (_) {}
      return [];
    }

    function refreshModelOptions() {
      var sel = ui.model;
      var prev = sel.value;
      sel.innerHTML = "";
      var optAuto = document.createElement("option");
      optAuto.value = "";
      optAuto.textContent = cnModels.length ? "(select model)" : (cnModelsLoaded ? "(no models found)" : "(loading…)");
      sel.appendChild(optAuto);
      cnModels.forEach(function (m) {
        var o = document.createElement("option");
        o.value = m; o.textContent = m;
        sel.appendChild(o);
      });
      // restore previous selection if still present, else from active layer cfg
      var lyr = activeControlLayer();
      var want = (lyr && lyr.controlnet && lyr.controlnet.model) || prev || "";
      if (want && cnModels.indexOf(want) === -1 && want !== "") {
        // keep an out-of-list model visible so the user's stored choice survives
        var o = document.createElement("option");
        o.value = want; o.textContent = want + " (saved)";
        sel.appendChild(o);
      }
      sel.value = want;
    }

    // ----- active-layer helpers -------------------------------------------------
    function layersArr() { var l = get("layers"); return Array.isArray(l) ? l : []; }
    function activeId() { return get("activeLayerId"); }
    function activeLayer() {
      var id = activeId(); if (id == null) return null;
      var arr = layersArr();
      for (var i = 0; i < arr.length; i++) if (arr[i] && arr[i].id === id) return arr[i];
      return null;
    }
    function isControl(layer) {
      return !!layer && (layer.type === "control" || layer.type === "controlnet");
    }
    function activeControlLayer() { var l = activeLayer(); return isControl(l) ? l : null; }

    // ----- show/hide on selection change ---------------------------------------
    function syncVisibility() {
      var lyr = activeControlLayer();
      if (lyr) {
        root.style.display = "";
        bindToLayer(lyr);
      } else {
        root.style.display = "none";
        setStatus("");
      }
      // also (re)decorate layer rows if the layers panel rendered them
      decorateLayerRows();
    }

    function bindToLayer(lyr) {
      var cn = ensureCN(lyr);
      // method
      ui.method.value = cn.preprocessor || "none";
      // model
      refreshModelOptions();
      // weight / begin / end
      ui.weight.value = String(cn.weight);
      ui.weightOut.textContent = fmt(cn.weight);
      ui.begin.value = String(cn.begin);
      ui.beginOut.textContent = pct(cn.begin);
      ui.end.value = String(cn.end);
      ui.endOut.textContent = pct(cn.end);
      // status
      setStatus(cn.processed ? ("processed: " + (cn.preprocessor || "?")) : "");
      ui.title.textContent = "CN: " + (lyr.name || lyr.id || "control");
    }

    function ensureCN(lyr) {
      if (!lyr.controlnet || typeof lyr.controlnet !== "object") {
        lyr.controlnet = {
          model: "", weight: 1.0, begin: 0.0, end: 1.0,
          preprocessor: "none", processed: false,
        };
      }
      var cn = lyr.controlnet;
      if (typeof cn.weight !== "number") cn.weight = 1.0;
      if (typeof cn.begin !== "number") cn.begin = 0.0;
      if (typeof cn.end !== "number") cn.end = 1.0;
      if (typeof cn.preprocessor !== "string") cn.preprocessor = "none";
      if (typeof cn.model !== "string") cn.model = "";
      return cn;
    }

    // persist mutations to the active control layer and notify the app
    function commitLayer(lyr) {
      // re-emit the (same-ref) layers array so observers (renderer, layers panel,
      // generate_ws) pick up the change via change:layers / change.
      set("layers", layersArr());
      bus.emit("controlnet:changed", { layerId: lyr.id, controlnet: lyr.controlnet });
    }

    // ----- get the layer's current image as a dataURL ---------------------------
    // Preference order: explicit dataURL field → HTMLImageElement → Konva node on
    // the stage whose id/name matches the layer → null (degrade).
    function layerImageDataURL(lyr) {
      // 1) common dataURL-ish fields a sibling module might set
      var keys = ["src", "dataURL", "image", "imageSrc", "url"];
      for (var i = 0; i < keys.length; i++) {
        var v = lyr[keys[i]];
        if (typeof v === "string" && v.indexOf("data:") === 0) return v;
      }
      // 2) an HTMLImageElement / canvas stored on the layer
      var imgCand = lyr.imageEl || lyr.img || (lyr.image && lyr.image.nodeType ? lyr.image : null);
      var canvasUrl = elementToDataURL(imgCand);
      if (canvasUrl) return canvasUrl;
      // 3) pull from the Konva stage
      var stageUrl = stageNodeToDataURL(lyr.id);
      if (stageUrl) return stageUrl;
      return null;
    }

    function elementToDataURL(el) {
      try {
        if (!el) return null;
        if (el.tagName === "CANVAS") return el.toDataURL("image/png");
        if (el.tagName === "IMG" && el.complete && el.naturalWidth) {
          var c = document.createElement("canvas");
          c.width = el.naturalWidth; c.height = el.naturalHeight;
          c.getContext("2d").drawImage(el, 0, 0);
          return c.toDataURL("image/png");
        }
      } catch (_) {}
      return null;
    }

    function getStage() {
      var st = get("canvas.stage");
      if (st) return st;
      return ctx.state && ctx.state.canvas ? ctx.state.canvas.stage : null;
    }

    function stageNodeToDataURL(layerId) {
      try {
        var stage = getStage();
        if (!stage || typeof stage.findOne !== "function") return null;
        // layers module is expected to tag its Konva node with the layer id
        var node = stage.findOne("#" + cssEscape(String(layerId)))
                || stage.findOne("." + cssEscape("layer-" + layerId))
                || null;
        if (node && typeof node.toDataURL === "function") {
          return node.toDataURL({ pixelRatio: 1, mimeType: "image/png" });
        }
      } catch (_) {}
      return null;
    }

    function cssEscape(s) {
      if (window.CSS && CSS.escape) return CSS.escape(s);
      return String(s).replace(/[^a-zA-Z0-9_\-]/g, "\\$&");
    }

    // ----- apply: preprocess + set processed map on the layer -------------------
    var busy = false;
    function applyPreprocess() {
      var lyr = activeControlLayer();
      if (!lyr || busy) return;
      var cn = ensureCN(lyr);
      cn.preprocessor = ui.method.value || "none";

      if (cn.preprocessor === "none") {
        cn.processed = false;
        setStatus("passthrough (no preprocessing)");
        bus.emit("controlnet:processed", { layerId: lyr.id, src: layerImageDataURL(lyr), method: "none" });
        commitLayer(lyr);
        return;
      }

      var src = layerImageDataURL(lyr);
      if (!src) {
        setStatus("no source image on this layer", true);
        return;
      }

      busy = true;
      setBusy(true);
      setStatus("processing " + cn.preprocessor + "…");

      var payload = { image: src };
      // include any tuning the backend might honor
      if (typeof cn.resolution === "number") payload.resolution = cn.resolution;

      var p = (api && typeof api.preprocess === "function")
        ? api.preprocess(cn.preprocessor, payload)
        : Promise.reject(new Error("no preprocess api"));

      p.then(function (res) {
        var out = extractProcessedImage(res);
        if (!out) throw new Error("no image in response");
        setProcessedImage(lyr, out, cn.preprocessor);
        setStatus("done: " + cn.preprocessor);
      }).catch(function (err) {
        // graceful: backend not up yet → keep original, inform the user, don't crash
        console.warn("[controlnet] preprocess failed:", err && err.message);
        setStatus("preprocess unavailable (" + describeErr(err) + ")", true);
      }).then(function () {
        busy = false; setBusy(false);
      });
    }

    function extractProcessedImage(res) {
      if (!res) return null;
      if (typeof res === "string") {
        if (res.indexOf("data:") === 0) return res;
        // bare base64 → wrap as png
        if (/^[A-Za-z0-9+/=\s]+$/.test(res) && res.length > 64) return "data:image/png;base64," + res.replace(/\s+/g, "");
        return null;
      }
      // common shapes
      var cand = res.image || res.images || res.processed || res.result || res.data || res.output;
      if (Array.isArray(cand)) cand = cand[0];
      if (cand && typeof cand === "object") {
        // ComfyUI-style {filename,type,subfolder} → build a /view url
        if (cand.filename && api && typeof api.viewUrl === "function") {
          return api.viewUrl(cand.filename, cand.type || "output", cand.subfolder || "");
        }
        cand = cand.image || cand.url || cand.b64 || cand.base64 || null;
      }
      if (typeof cand === "string") {
        if (cand.indexOf("data:") === 0 || cand.indexOf("http") === 0 || cand.charAt(0) === "/") return cand;
        if (/^[A-Za-z0-9+/=\s]+$/.test(cand) && cand.length > 64) return "data:image/png;base64," + cand.replace(/\s+/g, "");
      }
      // top-level filename
      if (res.filename && api && typeof api.viewUrl === "function") {
        return api.viewUrl(res.filename, res.type || "output", res.subfolder || "");
      }
      return null;
    }

    // set the processed map as the control layer's image (multiple field aliases so
    // whichever convention the renderer/layers module uses, it finds the image), and
    // broadcast so the canvas renderer can swap the Konva image.
    function setProcessedImage(lyr, srcUrl, method) {
      lyr.src = srcUrl;
      lyr.imageSrc = srcUrl;
      lyr.controlImage = srcUrl;
      var cn = ensureCN(lyr);
      cn.processed = true;
      cn.preprocessor = method;
      cn.processedSrc = srcUrl;
      commitLayer(lyr);
      bus.emit("controlnet:processed", { layerId: lyr.id, src: srcUrl, method: method });
      bus.emit("layer:imageChanged", { layerId: lyr.id, src: srcUrl }); // generic hook for canvas renderer
    }

    // ----- picker UI ------------------------------------------------------------
    function buildPicker(rootEl) {
      var u = {};

      var title = el("span", "cn-title");
      title.textContent = "ControlNet";
      rootEl.appendChild(title);
      u.title = title;

      // preprocessor method
      u.method = sel(rootEl, "cn-method", "Preprocessor");
      fillMethods(u.method);

      // gear that toggles the advanced model/weight/range popover
      var gear = btn(rootEl, "⚙", "cn-gear", "ControlNet model & weights");
      var pop = el("div", "cn-pop");
      pop.style.display = "none";
      rootEl.appendChild(pop);
      gear.addEventListener("click", function () {
        pop.style.display = pop.style.display === "none" ? "" : "none";
      });
      u.pop = pop;

      // model select
      u.model = sel(pop, "cn-model", "Model");
      var mAuto = document.createElement("option");
      mAuto.value = ""; mAuto.textContent = "(loading…)";
      u.model.appendChild(mAuto);

      // weight
      var wr = labeledRange(pop, "Weight", 0, 2, 0.05, 1.0);
      u.weight = wr.input; u.weightOut = wr.out;
      u.weight.addEventListener("input", function () {
        u.weightOut.textContent = fmt(parseFloat(u.weight.value));
      });

      // begin %
      var br = labeledRange(pop, "Begin", 0, 1, 0.01, 0.0);
      u.begin = br.input; u.beginOut = br.out;
      u.begin.addEventListener("input", function () {
        u.beginOut.textContent = pct(parseFloat(u.begin.value));
      });

      // end %
      var er = labeledRange(pop, "End", 0, 1, 0.01, 1.0);
      u.end = er.input; u.endOut = er.out;
      u.end.addEventListener("input", function () {
        u.endOut.textContent = pct(parseFloat(u.end.value));
      });

      // commit model/weight/range on change → store on layer
      function commitCfg() {
        var lyr = activeControlLayer(); if (!lyr) return;
        var cn = ensureCN(lyr);
        cn.model = u.model.value || "";
        cn.weight = clamp(parseFloat(u.weight.value), 0, 2, 1.0);
        cn.begin = clamp(parseFloat(u.begin.value), 0, 1, 0.0);
        cn.end = clamp(parseFloat(u.end.value), 0, 1, 1.0);
        if (cn.end < cn.begin) cn.end = cn.begin;
        commitLayer(lyr);
      }
      u.model.addEventListener("change", commitCfg);
      u.weight.addEventListener("change", commitCfg);
      u.begin.addEventListener("change", commitCfg);
      u.end.addEventListener("change", commitCfg);
      u.method.addEventListener("change", function () {
        var lyr = activeControlLayer(); if (!lyr) return;
        ensureCN(lyr).preprocessor = u.method.value || "none";
        commitLayer(lyr);
      });

      // apply button
      u.apply = btn(rootEl, "Apply", "cn-apply btn-primary", "Run preprocessor on this layer");
      u.apply.addEventListener("click", applyPreprocess);

      // status
      u.status = el("span", "cn-status");
      rootEl.appendChild(u.status);

      return u;
    }

    function fillMethods(selEl) {
      selEl.innerHTML = "";
      PREPROCESSORS.forEach(function (p) {
        var o = document.createElement("option");
        o.value = p.id; o.textContent = p.label;
        selEl.appendChild(o);
      });
    }

    // ----- layer-row decoration (augment, don't own) ----------------------------
    // The layers panel is owned by the `layers` module. If it renders rows that
    // expose the layer id (data-layer-id) we append a tiny CN badge/gear so a control
    // layer is recognizable from the panel. This is additive DOM, never a file edit,
    // and it self-heals on every layers change.
    function decorateLayerRows() {
      var panel = dom.layersPanel;
      if (!panel) return;
      // The layers module tags its rows with data-id (some builds use data-layer-id);
      // match either so the CN badge appears regardless of the convention it landed on.
      var rows = panel.querySelectorAll("[data-layer-id],[data-id]");
      if (!rows.length) return;
      var byId = {};
      layersArr().forEach(function (l) { if (l) byId[l.id] = l; });
      rows.forEach(function (row) {
        var id = row.getAttribute("data-layer-id") || row.getAttribute("data-id");
        var lyr = byId[id];
        var existing = row.querySelector(".cn-row-badge");
        if (lyr && isControl(lyr)) {
          if (!existing) {
            var badge = el("span", "cn-row-badge");
            badge.textContent = "CN";
            badge.title = "ControlNet layer — select to configure";
            row.appendChild(badge);
          }
        } else if (existing) {
          existing.remove();
        }
      });
    }

    // ----- small DOM helpers ----------------------------------------------------
    function el(tag, cls) { var e = document.createElement(tag); if (cls) e.className = cls; return e; }
    function btn(parent, label, cls, title) {
      var b = el("button", "cn-btn " + (cls || ""));
      b.type = "button"; b.textContent = label; if (title) b.title = title;
      parent.appendChild(b); return b;
    }
    function sel(parent, cls, title) {
      var s = el("select", "cn-select " + (cls || ""));
      if (title) s.title = title;
      parent.appendChild(s); return s;
    }
    function labeledRange(parent, label, min, max, step, val) {
      var row = el("div", "cn-range-row");
      var lab = el("label", "cn-range-label"); lab.textContent = label;
      var inp = el("input", "cn-range");
      inp.type = "range"; inp.min = String(min); inp.max = String(max);
      inp.step = String(step); inp.value = String(val);
      var out = el("span", "cn-range-out");
      out.textContent = (max <= 1 ? pct(val) : fmt(val));
      row.appendChild(lab); row.appendChild(inp); row.appendChild(out);
      parent.appendChild(row);
      return { input: inp, out: out };
    }

    function setStatus(msg, isErr) {
      ui.status.textContent = msg || "";
      ui.status.classList.toggle("cn-err", !!isErr);
    }
    function setBusy(b) {
      ui.apply.disabled = b;
      ui.apply.textContent = b ? "…" : "Apply";
      root.classList.toggle("cn-busy", b);
    }

    function fmt(n) { return (isFinite(n) ? n : 0).toFixed(2); }
    function pct(n) { return Math.round((isFinite(n) ? n : 0) * 100) + "%"; }
    function clamp(n, lo, hi, dflt) { n = parseFloat(n); if (!isFinite(n)) return dflt; return Math.min(hi, Math.max(lo, n)); }
    function describeErr(e) {
      var m = (e && e.message) || "error";
      if (/404/.test(m)) return "backend not ready";
      return m.length > 40 ? m.slice(0, 40) + "…" : m;
    }

    // ----- wiring ---------------------------------------------------------------
    bus.on("change:activeLayerId", syncVisibility);
    bus.on("change:layers", function () {
      // a layer may have been added/removed/typed; re-sync + re-decorate
      syncVisibility();
    });
    // re-decorate after the layers panel (re)renders, if it announces it
    bus.on("layers:rendered", decorateLayerRows);
    bus.on("app:ready", function () { syncVisibility(); decorateLayerRows(); });

    // initial sync (in case a control layer is already active at boot)
    syncVisibility();

    console.info("[controlnet] ready");
  }

  // ---- scoped CSS injected once ------------------------------------------------
  function injectCSS() {
    if (document.getElementById("style-controlnet")) return;
    var css = [
      "#canvas-toolbar .cn-toolbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap}",
      "#canvas-toolbar .cn-title{font-weight:600;color:var(--accent);font-size:12px}",
      "#canvas-toolbar .cn-select{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:6px;padding:3px 6px;font-size:12px;max-width:170px}",
      "#canvas-toolbar .cn-btn{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:6px;padding:3px 8px;cursor:pointer;font-size:12px}",
      "#canvas-toolbar .cn-btn:hover{border-color:var(--accent)}",
      "#canvas-toolbar .cn-btn.btn-primary{background:var(--accent2);border-color:var(--accent);color:#fff;font-weight:600}",
      "#canvas-toolbar .cn-btn:disabled{opacity:.5;cursor:default}",
      "#canvas-toolbar .cn-gear{padding:3px 6px}",
      "#canvas-toolbar .cn-pop{position:absolute;top:100%;left:0;margin-top:6px;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:8px;min-width:220px;box-shadow:0 6px 20px rgba(0,0,0,.4);z-index:30}",
      "#canvas-toolbar .cn-pop .cn-select{max-width:none;width:100%;margin-bottom:6px}",
      "#canvas-toolbar .cn-range-row{display:flex;align-items:center;gap:6px;margin:4px 0}",
      "#canvas-toolbar .cn-range-label{color:var(--muted);min-width:48px;font-size:11px}",
      "#canvas-toolbar .cn-range{flex:1;accent-color:var(--accent)}",
      "#canvas-toolbar .cn-range-out{color:var(--text);min-width:40px;text-align:right;font-size:11px;font-variant-numeric:tabular-nums}",
      "#canvas-toolbar .cn-status{color:var(--muted);font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
      "#canvas-toolbar .cn-status.cn-err{color:var(--danger)}",
      "#canvas-toolbar .cn-toolbar.cn-busy{opacity:.85}",
      "#layers-panel .cn-row-badge{display:inline-block;margin-left:6px;padding:0 4px;border:1px solid var(--accent);color:var(--accent);border-radius:4px;font-size:9px;font-weight:700;line-height:1.4;vertical-align:middle}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-controlnet";
    st.textContent = css;
    document.head.appendChild(st);
  }
})();
