/* ideogram4_bbox_node.js — STAGE NODES-A. The Ideogram4PromptBuilderKJ NODE.

   This is a NODE in the Konva node-graph editor (workflows.js), NOT a panel/tab/overlay.
   It registers a node type "Ideogram4PromptBuilderKJ" against the Stage-Infra contract
   (window.Serenity.workflows.register). When you "+ Add Node" -> "📐 Ideogram4 Bbox" it
   drops a draggable node onto the wf-stage whose BODY is an interactive image frame:
   drag inside the frame to draw a bbox, click a box to select (Konva.Transformer resize +
   drag to move within the frame). These in-body interactions never move/pan the node
   (the infra gives body-nodes a title-bar-only drag handle; we cancelBubble in the body).

   def.props renders the caption fields + per-box element list into the docked #wf-props
   panel. Everything lives in node.data; node.data.caption is the strict-schema JSON
   (REUSING bbox_builder.js's c1000 / bboxOf / buildCaption logic, coords RELATIVE to the
   frame -> [ymin,xmin,ymax,xmax] on a 0-1000 grid). The node has a "CAPTION" output port.

   RENDER RECIPE (stored in node.data so the lowering stage can read it, NOT changed here):
   model "ideogram4", scheduler "ideogram_logitnormal", cfg 7, cfg_override -1 (OFF),
   sampler "euler", steps 20. The simple_flowmatch/cfg_override path is FORBIDDEN.

   Self-bootstrapping: registers against the contract via the "workflows:ready" bus event
   (and immediately if the contract is already live), so it needs no main.js wiring — it
   is purely a consumer of Serenity.workflows. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S) return;

  // ------------------------------------------------------------------
  // reused bbox_builder.js logic (palette + 0-1000 clamp)
  // ------------------------------------------------------------------
  function c1000(v) { return Math.max(0, Math.min(1000, Math.round(v * 1000))); }
  var PALETTE = ["#6c8cff", "#46c46a", "#e0b341", "#e0556b", "#b36cff", "#3ad0d0"];

  // body geometry: node is wider/taller than a port-only node so the image frame fits
  var BODY = { w: 300, h: 296, pad: 12, top: 34, frameH: 200 };

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }

  // ------------------------------------------------------------------
  // the node's default per-node data (deep-cloned by the infra per node)
  // ------------------------------------------------------------------
  function defaultData() {
    return {
      // caption fields
      high_level_description: "",
      background: "",
      aesthetics: "",
      lighting: "",
      medium: "photograph",
      photo: "",
      width: 1024,
      height: 1024,
      // bbox elements: {id,type:'obj'|'text',desc,text,color, x,y,w,h} where x/y/w/h are
      // NORMALIZED to the image frame (0..1) so they survive node resize / frame re-fit.
      elements: [],
      nextId: 1,
      selId: null,
      // computed strict-schema JSON caption (kept fresh on every edit)
      caption: "",
      // RENDER RECIPE the lowering stage must honor (do NOT change these values)
      recipe: {
        model: "ideogram4",
        scheduler: "ideogram_logitnormal",
        cfg: 7.0,
        cfg_override: -1,
        sampler: "euler",
        steps: 20,
      },
    };
  }

  // ------------------------------------------------------------------
  // strict-schema caption (REUSE of bbox_builder.js buildCaption — key order preserved)
  // coordinates are stored normalized; bbox = [ymin,xmin,ymax,xmax] on 0-1000.
  // ------------------------------------------------------------------
  function bboxOfElem(e) {
    return [c1000(e.y), c1000(e.x), c1000(e.y + e.h), c1000(e.x + e.w)];
  }
  function buildCaption(d) {
    var cap = {};
    if (d.high_level_description && d.high_level_description.trim())
      cap.high_level_description = d.high_level_description.trim();
    var sd = { aesthetics: d.aesthetics || "", lighting: d.lighting || "" };
    if (d.photo && d.photo.trim()) { sd.photo = d.photo.trim(); sd.medium = "photograph"; }
    else { sd.medium = d.medium || "photograph"; }
    cap.style_description = sd;
    var elements = (d.elements || []).map(function (it) {
      var e = { type: it.type === "text" ? "text" : "obj", bbox: bboxOfElem(it) };
      if (it.type === "text" && it.text) e.text = it.text;
      e.desc = it.desc || "";
      e.color_palette = [String(it.color || "#6c8cff").toUpperCase()];
      return e;
    });
    cap.compositional_deconstruction = { background: (d.background || "").trim(), elements: elements };
    return JSON.stringify(cap); // minified, key order preserved
  }

  // recompute node.data.caption and stash a render-recipe-flat patch the lowering stage
  // can pick up from node.data (model/scheduler/cfg/... + prompt=caption + width/height).
  function refreshCaption(node) {
    var d = node.getAttr("data") || defaultData();
    d.caption = buildCaption(d);
    node.setAttr("data", d);
    return d;
  }

  // ==================================================================
  // def.body — paint the interactive image frame + bboxes INTO the node group.
  //   Coordinates of child rects live in STAGE/group space; we convert to/from the
  //   normalized element store relative to the frame on every edit & redraw.
  // ==================================================================
  function makeBody(Konva) {
    return function body(group, node, ctx) {
      var d = node.getAttr("data") || defaultData();
      node.setAttr("data", d);

      // frame rect: aspect-fit width/height into the body's frame area
      var bodyW = BODY.w, fx = BODY.pad, fy = BODY.top + 4;
      function frameGeom() {
        var aw = parseInt(d.width, 10) || 1024, ah = parseInt(d.height, 10) || 1024;
        var availW = bodyW - 2 * BODY.pad, availH = BODY.frameH;
        var s = Math.min(availW / aw, availH / ah);
        var fw = aw * s, fh = ah * s;
        return { x: fx + (availW - fw) / 2, y: fy + (availH - fh) / 2, w: fw, h: fh };
      }

      var bodyLayer = new Konva.Group({ listening: true });
      group.add(bodyLayer);

      // visible image-frame outline
      var frameRect = new Konva.Rect({
        stroke: "#444a5e", strokeWidth: 1, dash: [6, 6],
        fill: "#0e0f15", listening: false,
      });
      bodyLayer.add(frameRect);

      // transparent draw surface ON TOP of the frame: captures drag-to-draw.
      var surface = new Konva.Rect({ fill: "rgba(0,0,0,0.001)" });
      bodyLayer.add(surface);

      var tr = new Konva.Transformer({ rotateEnabled: false, borderStroke: "#6c8cff", ignoreStroke: true });
      bodyLayer.add(tr);

      var rectsById = {};   // id -> Konva.Rect (box shapes)
      var drawing = null;   // { rect, x0, y0 }

      function f() { return frameGeom(); }
      function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

      // select the NODE (so the infra mounts def.props) by firing the infra frame's
      // mousedown handler; the infra frame is the first Rect child of the group.
      function selectNodeViaInfra() {
        var infraFrame = group.find("Rect")[0];
        if (infraFrame) infraFrame.fire("mousedown", { cancelBubble: true }, false);
      }

      function setBoxFromElem(rect, it) {
        var g = f();
        rect.setAttrs({
          x: g.x + it.x * g.w, y: g.y + it.y * g.h,
          width: it.w * g.w, height: it.h * g.h,
          scaleX: 1, scaleY: 1,
        });
      }
      // write a box's current geometry back into its normalized element record
      function elemFromBox(it, rect) {
        var g = f();
        var x = (rect.x() - g.x) / g.w, y = (rect.y() - g.y) / g.h;
        var w = (rect.width() * rect.scaleX()) / g.w, h = (rect.height() * rect.scaleY()) / g.h;
        // clamp into the frame
        x = clamp(x, 0, 1); y = clamp(y, 0, 1);
        w = clamp(w, 0, 1 - x); h = clamp(h, 0, 1 - y);
        it.x = x; it.y = y; it.w = w; it.h = h;
      }

      function attachBox(it) {
        var rect = new Konva.Rect({
          stroke: it.color, strokeWidth: 2, fill: it.color + "22", draggable: true,
        });
        setBoxFromElem(rect, it);
        rect.on("mousedown click", function (e) { e.cancelBubble = true; selectBox(it.id); });
        rect.on("dragstart transformstart", function (e) { e.cancelBubble = true; selectBox(it.id); });
        rect.on("dragmove transform", function () {
          // keep the box inside the frame while dragging/resizing
          var g = f();
          var nx = clamp(rect.x(), g.x, g.x + g.w - Math.max(2, rect.width() * rect.scaleX()));
          var ny = clamp(rect.y(), g.y, g.y + g.h - Math.max(2, rect.height() * rect.scaleY()));
          rect.position({ x: nx, y: ny });
        });
        rect.on("dragend transformend", function () {
          elemFromBox(it, rect);
          refreshCaption(node);
          renderPropsIfActive(node, ctx);
          bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw();
        });
        rectsById[it.id] = rect;
        bodyLayer.add(rect);
        tr.moveToTop();
      }

      function selectBox(id) {
        d.selId = id;
        var rect = rectsById[id];
        tr.nodes(rect ? [rect] : []);
        node.setAttr("data", d);
        selectNodeViaInfra();             // make sure the node's props panel is mounted
        renderPropsIfActive(node, ctx);   // reflect the new selection in the list
        bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw();
      }

      // expose so def.props can drive selection / rebuild from the docked panel
      node.setAttr("_bboxApi", {
        rebuild: function () { rebuildAll(); },
        select: function (id) { selectBox(id); },
        getFrame: f,
      });

      function rebuildAll() {
        // re-fit the frame + every box (e.g. after width/height edits in props)
        var g = f();
        frameRect.setAttrs({ x: g.x, y: g.y, width: g.w, height: g.h });
        surface.setAttrs({ x: g.x, y: g.y, width: g.w, height: g.h });
        // drop boxes whose element no longer exists
        Object.keys(rectsById).forEach(function (id) {
          var still = (d.elements || []).some(function (e) { return String(e.id) === String(id); });
          if (!still) { rectsById[id].destroy(); delete rectsById[id]; }
        });
        (d.elements || []).forEach(function (it) {
          if (rectsById[it.id]) setBoxFromElem(rectsById[it.id], it);
          else attachBox(it);
        });
        var sel = rectsById[d.selId];
        tr.nodes(sel ? [sel] : []);
        tr.moveToTop();
        bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw();
      }

      // ---- drag-to-draw a NEW bbox on the surface ----
      surface.on("mousedown", function (e) {
        e.cancelBubble = true;            // never pan the stage / drag the node
        selectNodeViaInfra();             // selecting the node mounts the props editor
        // pointer in group-local space
        var grp = group.getRelativePointerPosition();
        var g = f();
        var x0 = clamp(grp.x, g.x, g.x + g.w), y0 = clamp(grp.y, g.y, g.y + g.h);
        var color = PALETTE[(d.nextId - 1) % PALETTE.length];
        var r = new Konva.Rect({ x: x0, y: y0, width: 1, height: 1, stroke: color, strokeWidth: 2, fill: color + "22" });
        bodyLayer.add(r);
        tr.nodes([]); tr.moveToTop();
        drawing = { rect: r, x0: x0, y0: y0, color: color };
        bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw();
      });
      surface.on("mousemove", function (e) {
        if (!drawing) return;
        e.cancelBubble = true;
        var grp = group.getRelativePointerPosition();
        var g = f();
        var px = clamp(grp.x, g.x, g.x + g.w), py = clamp(grp.y, g.y, g.y + g.h);
        drawing.rect.setAttrs({
          x: Math.min(px, drawing.x0), y: Math.min(py, drawing.y0),
          width: Math.abs(px - drawing.x0), height: Math.abs(py - drawing.y0),
        });
        bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw();
      });
      function finishDraw(e) {
        if (!drawing) return;
        if (e) e.cancelBubble = true;
        var r = drawing.rect, color = drawing.color; drawing = null;
        if (r.width() < 6 || r.height() < 6) { r.destroy(); bodyLayer.getLayer() && bodyLayer.getLayer().batchDraw(); return; }
        r.destroy(); // re-create through the normalized element store for consistency
        var id = d.nextId++;
        var it = { id: id, type: "obj", desc: "", text: "", color: color, x: 0, y: 0, w: 0, h: 0 };
        // seed normalized geometry from the drawn rect
        var g = f();
        it.x = clamp((r.x() - g.x) / g.w, 0, 1);
        it.y = clamp((r.y() - g.y) / g.h, 0, 1);
        it.w = clamp((r.width()) / g.w, 0, 1 - it.x);
        it.h = clamp((r.height()) / g.h, 0, 1 - it.y);
        d.elements.push(it);
        node.setAttr("data", d);
        attachBox(it);
        refreshCaption(node);
        selectBox(id);
        renderPropsIfActive(node, ctx);
      }
      surface.on("mouseup", finishDraw);

      // initial paint
      rebuildAll();
      refreshCaption(node);
    };
  }

  // ==================================================================
  // def.props — render caption fields + element list into the docked panel.
  // ==================================================================
  // remember which node's props are currently mounted, so body edits can refresh it
  var activePropsNode = null, activePanelEl = null, activeCtx = null;

  function renderPropsIfActive(node, ctx) {
    if (activePropsNode === node && activePanelEl) {
      props(node, activePanelEl, ctx || activeCtx);
    }
  }

  function props(node, panelEl, ctx) {
    activePropsNode = node; activePanelEl = panelEl; activeCtx = ctx;
    var d = node.getAttr("data") || defaultData();
    node.setAttr("data", d);
    panelEl.innerHTML = "";

    function commit() { refreshCaption(node); var p = capPreview(); if (p) p.textContent = node.getAttr("data").caption; }

    function fieldInput(label, key, ph, multiline) {
      panelEl.appendChild(el("label", null, label));
      var i = multiline ? el("textarea") : el("input");
      i.setAttribute("placeholder", ph || "");
      i.value = d[key] != null ? d[key] : "";
      i.addEventListener("input", function () { d[key] = i.value; node.setAttr("data", d); commit(); });
      panelEl.appendChild(i);
      return i;
    }

    fieldInput("High level description", "high_level_description", "one-sentence overview, starts with the subject", true);
    fieldInput("Background (required)", "background", "scene background", true);
    fieldInput("Aesthetics", "aesthetics", "e.g. cinematic, moody");
    fieldInput("Lighting", "lighting", "e.g. soft natural light");

    panelEl.appendChild(el("label", null, "Medium"));
    var medium = el("select");
    ["photograph", "illustration", "3d_render", "digital_art", "painting"].forEach(function (m) {
      var o = el("option", null, m); o.value = m; if (m === d.medium) o.selected = true; medium.appendChild(o);
    });
    medium.addEventListener("change", function () { d.medium = medium.value; node.setAttr("data", d); commit(); });
    panelEl.appendChild(medium);

    fieldInput("Photo / camera (optional)", "photo", "35mm, f/1.8 (blank = use medium)");

    // width / height (re-fit the in-node frame on change)
    panelEl.appendChild(el("label", null, "Width / Height"));
    var dimRow = el("div"); dimRow.style.display = "grid"; dimRow.style.gridTemplateColumns = "1fr 1fr"; dimRow.style.gap = "8px";
    var wIn = el("input"); wIn.type = "number"; wIn.value = d.width;
    var hIn = el("input"); hIn.type = "number"; hIn.value = d.height;
    function onDim() {
      d.width = parseInt(wIn.value, 10) || 1024;
      d.height = parseInt(hIn.value, 10) || 1024;
      node.setAttr("data", d);
      var apiref = node.getAttr("_bboxApi"); if (apiref) apiref.rebuild();
      commit();
    }
    wIn.addEventListener("change", onDim); hIn.addEventListener("change", onDim);
    dimRow.appendChild(wIn); dimRow.appendChild(hIn);
    panelEl.appendChild(dimRow);

    panelEl.appendChild(el("div", "wf-props-empty", "Drag on the node's image frame to draw a region. Click a box to select; resize with the handles."));

    // ---- element list ----
    panelEl.appendChild(el("label", null, "Elements (bbox regions)"));
    var listEl = el("div"); panelEl.appendChild(listEl);

    (d.elements || []).forEach(function (it) {
      var row = el("div", "bb-row" + (it.id === d.selId ? " sel" : ""));
      row.style.display = "flex"; row.style.gap = "6px"; row.style.alignItems = "center";
      row.style.margin = "4px 0"; row.style.padding = "5px";
      row.style.border = "1px solid var(--line)"; row.style.borderRadius = "6px"; row.style.cursor = "pointer";
      if (it.id === d.selId) row.style.borderColor = "var(--accent)";

      var sw = el("span"); sw.style.width = "18px"; sw.style.height = "18px"; sw.style.borderRadius = "4px";
      sw.style.border = "1px solid var(--line)"; sw.style.flex = "none"; sw.style.background = it.color;

      var ty = el("select"); ty.style.width = "62px"; ty.style.flex = "none";
      ["obj", "text"].forEach(function (t) { var o = el("option", null, t); o.value = t; if (t === it.type) o.selected = true; ty.appendChild(o); });

      var ds = el("input"); ds.type = "text"; ds.value = it.desc; ds.setAttribute("placeholder", "description"); ds.style.flex = "1"; ds.style.minWidth = "0";

      var del = el("button", "btn", "✕"); del.style.flex = "none"; del.style.padding = "3px 7px";

      ty.addEventListener("change", function (e) { e.stopPropagation(); it.type = ty.value; node.setAttr("data", d); commit(); props(node, panelEl, ctx); });
      ds.addEventListener("input", function (e) { e.stopPropagation(); it.desc = ds.value; node.setAttr("data", d); commit(); });
      ds.addEventListener("click", function (e) { e.stopPropagation(); });
      del.addEventListener("click", function (e) {
        e.stopPropagation();
        d.elements = d.elements.filter(function (x) { return x.id !== it.id; });
        if (d.selId === it.id) d.selId = null;
        node.setAttr("data", d);
        var apiref = node.getAttr("_bboxApi"); if (apiref) apiref.rebuild();
        commit(); props(node, panelEl, ctx);
      });
      row.addEventListener("click", function () {
        var apiref = node.getAttr("_bboxApi"); if (apiref) apiref.select(it.id);
        else { d.selId = it.id; node.setAttr("data", d); props(node, panelEl, ctx); }
      });

      // optional text content for "text" elements
      row.appendChild(sw); row.appendChild(ty); row.appendChild(ds); row.appendChild(del);
      listEl.appendChild(row);
      if (it.type === "text") {
        var txt = el("input"); txt.type = "text"; txt.value = it.text || ""; txt.setAttribute("placeholder", "rendered text"); txt.style.margin = "0 0 4px 0";
        txt.addEventListener("input", function (e) { e.stopPropagation(); it.text = txt.value; node.setAttr("data", d); commit(); });
        listEl.appendChild(txt);
      }
    });
    if (!(d.elements || []).length) listEl.appendChild(el("div", "wf-props-empty", "No regions yet — drag on the node's frame."));

    // ---- caption preview ----
    panelEl.appendChild(el("label", null, "Caption (Ideogram-4 schema)"));
    var prev = el("textarea", "i4-cap-preview"); prev.readOnly = true; prev.style.minHeight = "64px"; prev.style.fontFamily = "monospace"; prev.style.fontSize = "11px";
    prev.value = d.caption || buildCaption(d);
    panelEl.appendChild(prev);
    function capPreview() { return panelEl.querySelector(".i4-cap-preview"); }

    refreshCaption(node);
    prev.value = node.getAttr("data").caption;
  }

  // ==================================================================
  // registration against the Stage-Infra contract
  // ==================================================================
  var registeredOnce = false;
  function registerNode(workflows) {
    if (registeredOnce || !workflows || typeof workflows.register !== "function") return;
    var Konva = (S.ctx && S.ctx.Konva) || window.Konva;
    if (!Konva) { console.warn("[ideogram4BboxNode] Konva not available yet"); return; }
    workflows.register("Ideogram4PromptBuilderKJ", {
      label: "📐 Ideogram4 Bbox",
      color: "#7a3a8a",
      w: BODY.w,
      h: BODY.h,
      ins: [],
      outs: ["CAPTION"],
      category: "Ideogram-4",
      data: defaultData(),
      body: makeBody(Konva),
      props: props,
    });
    registeredOnce = true;
    console.info("[ideogram4BboxNode] registered Ideogram4PromptBuilderKJ node");
  }

  // bus-driven: register when the infra announces the contract is live...
  if (S.bus && typeof S.bus.on === "function") {
    S.bus.on("workflows:ready", function (wf) { registerNode(wf || S.workflows); });
  }
  // ...or immediately if the contract already exists (infra inited before us).
  if (S.workflows) registerNode(S.workflows);

  // also expose a no-op module init so main.js EXPECTED (if it lists us) doesn't warn,
  // and a late init still finds the contract.
  if (typeof S.register === "function") {
    S.register("ideogram4BboxNode", { init: function () { if (S.workflows) registerNode(S.workflows); } });
  }
})();
