/* bbox_builder.js — module 'bboxBuilder'. Visual Ideogram-4 caption builder in the
   Workflows tab (ported from KJNodes Ideogram4PromptBuilderKJ). Draw regions on a
   Konva canvas; set each region's type/desc/color; set high_level_description +
   background + style; it assembles the structured JSON caption WITH bboxes
   ([ymin,xmin,ymax,xmax] on a 0-1000 grid) — the format Ideogram-4 was trained on —
   and renders it through the ideogram4 worker. A toggle in the workflow toolbar
   switches between the node graph and this builder. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  function injectCSS() {
    if (document.getElementById("style-bboxBuilder")) return;
    var css = [
      "#bbox-panel{position:absolute;inset:0;display:grid;grid-template-columns:280px 1fr 260px;grid-template-rows:1fr auto;background:var(--bg);z-index:1}",
      "#bbox-left{grid-row:1/3;border-right:1px solid var(--line);padding:10px;overflow-y:auto;background:var(--panel)}",
      "#bbox-stagewrap{position:relative;overflow:hidden;background:#0e0f15}",
      "#bbox-stage{position:absolute;inset:0}",
      "#bbox-right{grid-row:1/3;border-left:1px solid var(--line);padding:10px;overflow-y:auto;background:var(--panel)}",
      "#bbox-bottom{grid-column:2/3;display:flex;align-items:center;gap:10px;padding:8px;border-top:1px solid var(--line);background:var(--panel)}",
      "#bbox-panel label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 3px}",
      "#bbox-panel textarea{width:100%;min-height:44px}",
      "#bbox-panel input,#bbox-panel select{width:100%}",
      "#bbox-panel .bb-row{display:flex;gap:6px;align-items:center;margin:4px 0;padding:5px;border:1px solid var(--line);border-radius:6px;cursor:pointer}",
      "#bbox-panel .bb-row.sel{border-color:var(--accent)}",
      "#bbox-panel .bb-row input[type=text]{flex:1;min-width:0}",
      "#bbox-panel .bb-sw{width:18px;height:18px;border-radius:4px;border:1px solid var(--line);flex:none}",
      "#bbox-panel .bb-del{padding:3px 7px}",
      "#bbox-panel .bb-hint{color:var(--muted);font-size:11px;margin:6px 0}",
      "#bbox-gen{min-width:170px;font-size:14px;font-weight:700}",
      "#bbox-result img{max-height:84px;border-radius:6px;border:1px solid var(--line)}",
    ].join("\n");
    var st = document.createElement("style"); st.id = "style-bboxBuilder"; st.textContent = css;
    document.head.appendChild(st);
  }
  function el(tag, cls, txt) { var n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }
  function c1000(v) { return Math.max(0, Math.min(1000, Math.round(v * 1000))); }

  var PALETTE = ["#6c8cff", "#46c46a", "#e0b341", "#e0556b", "#b36cff", "#3ad0d0"];

  S.register("bboxBuilder", {
    init: function (ctx) {
      injectCSS();
      var Konva = ctx.Konva, bus = ctx.bus, api = ctx.api, set = ctx.set, get = ctx.get;
      var host = document.getElementById("view-ideogram");
      if (!host) { console.warn("[bboxBuilder] no #view-ideogram"); return; }

      // ---- panel scaffold ----
      var panel = el("div"); panel.id = "bbox-panel";
      var left = el("div"); left.id = "bbox-left";
      var stagewrap = el("div"); stagewrap.id = "bbox-stagewrap";
      var stageHost = el("div"); stageHost.id = "bbox-stage"; stagewrap.appendChild(stageHost);
      var right = el("div"); right.id = "bbox-right";
      var bottom = el("div"); bottom.id = "bbox-bottom";
      panel.appendChild(left); panel.appendChild(stagewrap); panel.appendChild(right); panel.appendChild(bottom);
      host.appendChild(panel);

      // ---- left: caption fields ----
      function field(lbl, ph, multiline) {
        left.appendChild(el("label", null, lbl));
        var i = multiline ? el("textarea") : el("input");
        i.setAttribute("placeholder", ph || ""); left.appendChild(i); return i;
      }
      var hld = field("High level description", "one-sentence overview, starts with the subject", true);
      var bg = field("Background (required)", "scene background", true);
      var aes = field("Aesthetics", "e.g. cinematic, moody");
      var lig = field("Lighting", "e.g. soft natural light");
      left.appendChild(el("label", null, "Medium"));
      var medium = el("select");
      ["photograph", "illustration", "3d_render", "digital_art", "painting"].forEach(function (m) {
        var o = el("option", null, m); o.value = m; medium.appendChild(o);
      });
      left.appendChild(medium);
      left.appendChild(el("label", null, "Photo / camera (optional)"));
      var photo = el("input"); photo.setAttribute("placeholder", "35mm, f/1.8 (blank = use medium)"); left.appendChild(photo);
      var wIn = el("input"), hIn = el("input");
      wIn.type = "number"; hIn.type = "number"; wIn.value = get("params.width") || 1024; hIn.value = get("params.height") || 1024;
      var dimRow = el("div", "pr-grid2"); dimRow.style.display = "grid"; dimRow.style.gridTemplateColumns = "1fr 1fr"; dimRow.style.gap = "8px";
      var wc = el("div"); wc.appendChild(el("label", null, "Width")); wc.appendChild(wIn);
      var hc = el("div"); hc.appendChild(el("label", null, "Height")); hc.appendChild(hIn);
      dimRow.appendChild(wc); dimRow.appendChild(hc); left.appendChild(dimRow);
      left.appendChild(el("div", "bb-hint", "Drag on the canvas to draw a region. Click a region to select; Del removes it."));

      // ---- right: element list ----
      right.appendChild(el("label", null, "Elements (bbox regions)"));
      var listEl = el("div"); right.appendChild(listEl);

      // ---- Konva stage ----
      var stage = null, layer = null, tr = null, built = false;
      var elems = [];          // {id, rect, type, desc, text, color, row}
      var selId = null, nextId = 1, drawing = null;

      function frame() { // the image frame inside the stage (fit aspect)
        var W = stageHost.clientWidth || 800, H = stageHost.clientHeight || 600;
        var aw = parseInt(wIn.value, 10) || 1024, ah = parseInt(hIn.value, 10) || 1024;
        var s = Math.min((W - 24) / aw, (H - 24) / ah);
        var fw = aw * s, fh = ah * s;
        return { x: (W - fw) / 2, y: (H - fh) / 2, w: fw, h: fh };
      }
      var frameRect = null;
      function drawFrame() {
        var f = frame();
        if (!frameRect) { frameRect = new Konva.Rect({ stroke: "#444a5e", strokeWidth: 1, dash: [6, 6], listening: false }); layer.add(frameRect); frameRect.moveToBottom(); }
        frameRect.setAttrs({ x: f.x, y: f.y, width: f.w, height: f.h });
        layer.batchDraw();
      }
      function build() {
        if (built) return;
        var W = stageHost.clientWidth || 800, H = stageHost.clientHeight || 600;
        stage = new Konva.Stage({ container: stageHost, width: W, height: H });
        layer = new Konva.Layer(); stage.add(layer);
        tr = new Konva.Transformer({ rotateEnabled: false, borderStroke: "#6c8cff" }); layer.add(tr);
        drawFrame();
        // draw-create on empty
        stage.on("mousedown", function (e) {
          if (e.target !== stage && e.target !== frameRect) return;
          var p = stage.getPointerPosition();
          var r = new Konva.Rect({ x: p.x, y: p.y, width: 1, height: 1, stroke: PALETTE[(nextId - 1) % PALETTE.length], strokeWidth: 2, fill: PALETTE[(nextId - 1) % PALETTE.length] + "22", draggable: true });
          layer.add(r); drawing = { rect: r, x0: p.x, y0: p.y };
          tr.nodes([]); layer.batchDraw();
        });
        stage.on("mousemove", function () {
          if (!drawing) return;
          var p = stage.getPointerPosition();
          drawing.rect.setAttrs({ x: Math.min(p.x, drawing.x0), y: Math.min(p.y, drawing.y0), width: Math.abs(p.x - drawing.x0), height: Math.abs(p.y - drawing.y0) });
          layer.batchDraw();
        });
        stage.on("mouseup", function () {
          if (!drawing) return;
          var r = drawing.rect; drawing = null;
          if (r.width() < 6 || r.height() < 6) { r.destroy(); layer.batchDraw(); return; }
          addElem(r);
        });
        window.addEventListener("keydown", function (e) {
          if (host.classList.contains("show") && (e.key === "Delete" || e.key === "Backspace") && selId != null
              && document.activeElement && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
            removeElem(selId);
          }
        });
        built = true;
      }

      function addElem(rect) {
        var id = nextId++;
        var color = PALETTE[(id - 1) % PALETTE.length];
        var item = { id: id, rect: rect, type: "obj", desc: "", text: "", color: color };
        rect.on("click", function (e) { e.cancelBubble = true; select(id); });
        rect.on("dragstart transformstart", function () { select(id); });
        elems.push(item); rebuildList(); select(id);
      }
      function removeElem(id) {
        var i = elems.findIndex(function (x) { return x.id === id; });
        if (i < 0) return;
        elems[i].rect.destroy(); elems.splice(i, 1);
        if (selId === id) { selId = null; tr.nodes([]); }
        rebuildList(); layer.batchDraw();
      }
      function select(id) {
        selId = id;
        var it = elems.find(function (x) { return x.id === id; });
        if (it) { tr.nodes([it.rect]); layer.batchDraw(); }
        rebuildList();
      }
      function rebuildList() {
        listEl.innerHTML = "";
        elems.forEach(function (it) {
          var row = el("div", "bb-row" + (it.id === selId ? " sel" : ""));
          var sw = el("span", "bb-sw"); sw.style.background = it.color;
          var ty = el("select"); ["obj", "text"].forEach(function (t) { var o = el("option", null, t); o.value = t; if (t === it.type) o.selected = true; ty.appendChild(o); }); ty.style.width = "62px";
          var ds = el("input"); ds.type = "text"; ds.value = it.desc; ds.setAttribute("placeholder", "description");
          var del = el("button", "btn bb-del", "✕");
          ty.addEventListener("change", function (e) { e.stopPropagation(); it.type = ty.value; });
          ds.addEventListener("input", function (e) { e.stopPropagation(); it.desc = ds.value; });
          ds.addEventListener("click", function (e) { e.stopPropagation(); });
          del.addEventListener("click", function (e) { e.stopPropagation(); removeElem(it.id); });
          row.addEventListener("click", function () { select(it.id); });
          row.appendChild(sw); row.appendChild(ty); row.appendChild(ds); row.appendChild(del);
          listEl.appendChild(row);
        });
        if (!elems.length) listEl.appendChild(el("div", "bb-hint", "No regions yet — drag on the canvas."));
      }

      // bbox of a rect -> [ymin,xmin,ymax,xmax] on 0-1000 relative to the frame
      function bboxOf(rect) {
        var f = frame();
        var x = (rect.x() - f.x) / f.w, y = (rect.y() - f.y) / f.h;
        var w = (rect.width() * rect.scaleX()) / f.w, h = (rect.height() * rect.scaleY()) / f.h;
        return [c1000(y), c1000(x), c1000(y + h), c1000(x + w)];
      }

      function buildCaptionObject() {
        var cap = {};
        if (hld.value.trim()) cap.high_level_description = hld.value.trim();
        var sd = { aesthetics: aes.value, lighting: lig.value };
        if (photo.value.trim()) { sd.photo = photo.value.trim(); sd.medium = "photograph"; }
        else { sd.medium = medium.value; }
        cap.style_description = sd;
        var elements = elems.map(function (it) {
          var e = { type: it.type === "text" ? "text" : "obj", bbox: bboxOf(it.rect) };
          if (it.type === "text" && it.text) e.text = it.text;
          e.desc = it.desc || "";
          e.color_palette = [it.color.toUpperCase()];
          return e;
        });
        cap.compositional_deconstruction = { background: bg.value.trim(), elements: elements };
        return cap;
      }

      function buildCaption() {
        return JSON.stringify(buildCaptionObject());  // minified, key order preserved
      }

      // ---- bottom: generate ----
      var genBtn = el("button", "btn btn-primary"); genBtn.id = "bbox-gen"; genBtn.textContent = "▶ Generate (Ideogram4)";
      var resWrap = el("div"); resWrap.id = "bbox-result";
      var capPreview = el("div", "bb-hint"); capPreview.style.flex = "1"; capPreview.style.overflow = "hidden"; capPreview.style.whiteSpace = "nowrap"; capPreview.style.textOverflow = "ellipsis";
      bottom.appendChild(genBtn); bottom.appendChild(capPreview); bottom.appendChild(resWrap);

      function showResult(url) { resWrap.innerHTML = ""; var im = new Image(); im.src = url; resWrap.appendChild(im); }
      bus.on("result:ready", function (p) { if (p && p.url) showResult(p.url); });

      genBtn.addEventListener("click", function () {
        var promptJson = buildCaptionObject();
        var caption = JSON.stringify(promptJson);
        capPreview.textContent = caption;
        set("params.prompt", caption);
        set("params.prompt_raw", caption);
        set("params.prompt_json", promptJson);
        set("params.negative", "");
        set("params.model", "ideogram4");
        // VERIFIED Ideogram-4 path (the one that renders, PSNR 29.7dB vs torch in
        // pipeline/ideogram4_generate.mojo): logit-normal schedule + dual cond/uncond
        // transformers + constant cfg 7, euler, 20 steps. The simple_flowmatch/AuraFlow
        // + cfg_override "KJ recipe" collapses our fp8 latents to a gray block — measured
        // job-0007/0008 gray vs job-0009 fox on this path. cfg_override is OFF (the worker
        // only accepts it on the simple scheduler, which is the broken path here).
        set("params.scheduler", "ideogram_logitnormal");
        set("params.sampler", "euler");
        set("params.cfg", 7.0);
        set("params.cfg_override", -1);
        set("params.steps", 20);
        set("params.width", parseInt(wIn.value, 10) || 1024);
        set("params.height", parseInt(hIn.value, 10) || 1024);
        var params = {
          model: "ideogram4",
          prompt: caption,
          prompt_raw: caption,
          prompt_json: promptJson,
          negative: "",
          scheduler: "ideogram_logitnormal",
          sampler: "euler",
          cfg: 7.0,
          cfg_override: -1,
          steps: 20,
          width: parseInt(wIn.value, 10) || 1024,
          height: parseInt(hIn.value, 10) || 1024,
        };
        genBtn.disabled = true; genBtn.textContent = "▶ …";
        api.connectWS(ctx.clientId, function (m) {
          if (!m) return;
          if (m.type === "progress") { var d = m.data || {}; genBtn.textContent = "▶ " + (d.value || 0) + "/" + (d.max || 0); }
          else if (m.type === "executed") { var imgs = (m.data && m.data.output && m.data.output.images) || []; if (imgs[0]) { var u = api.viewUrl(imgs[0].filename); showResult(u); bus.emit("result:ready", { url: u, filename: imgs[0].filename, params: {} }); } }
          else if (m.type === "executing" && m.data && m.data.node === null) { genBtn.disabled = false; genBtn.textContent = "▶ Generate (Ideogram4)"; }
          else if (m.type === "execution_error") { genBtn.disabled = false; genBtn.textContent = "▶ Generate (error)"; console.error("[bboxBuilder]", m.data); }
        });
        api.submitPrompt({ params: params }, ctx.clientId).then(function (r) { console.info("[bboxBuilder] queued", r && r.job_id); })
          .catch(function (e) { genBtn.disabled = false; genBtn.textContent = "▶ Generate (failed)"; console.warn(e); });
      });

      // ---- the Ideogram tab IS this builder (no toggle; the panel fills the view) ----
      function activate() {
        build();
        setTimeout(function () {
          if (stage) { stage.size({ width: stageHost.clientWidth, height: stageHost.clientHeight }); drawFrame(); }
          rebuildList();
        }, 30);
      }
      bus.on("nav:tab", function (t) { if (t === "ideogram") activate(); });
      if (host.classList.contains("show")) activate();   // already on this tab at init
      wIn.addEventListener("change", function () { if (built) drawFrame(); });
      hIn.addEventListener("change", function () { if (built) drawFrame(); });
      console.info("[bboxBuilder] ready");
    },
  });
})();
