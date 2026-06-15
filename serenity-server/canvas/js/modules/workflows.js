/* workflows.js — module 'workflows'. A Konva node-graph editor in the Workflows tab
   (#view-workflows, created by nav_shell). Toolbar (name/Add Node/Templates/Load/Save/
   Queue) + a grid canvas with draggable nodes, input/output ports, wire dragging,
   pan (drag empty space) and zoom (wheel).

   STAGE INFRA owns THE CONTRACT here. Nodes are no longer ports-only: a node type may
   register a custom interactive Konva body (def.body) + an HTML properties editor
   (def.props) + per-node data. Other stages register node types against:

     window.Serenity.workflows = {
       register(typeId, def),          // add a node type to the registry + Add-Node menu
       graph(),                        // { nodes:[{type,data,group}], wires:[{from,fromI,to,toI}] }
       addNodeByType(typeId, x, y),    // spawn a node of typeId at (x,y); returns the Konva.Group
     }

   def = {
     label,            // title-bar text (string)
     color,            // accent color (hex string)
     w, h,             // body size; for body-nodes h should include the painted area
     ins:  [labels],   // input port labels (top→bottom)
     outs: [labels],   // output port labels (top→bottom)
     body(group,node,ctx),       // OPTIONAL: paint an interactive Konva body INTO `group`.
                                 //   When present the node is bigger and ONLY its title bar
                                 //   drags the node (group.draggable === false) so in-body
                                 //   pointer events (drawing, transformers) never pan the node.
     props(node,panelEl,ctx),    // OPTIONAL: render the HTML editor for the selected node
                                 //   into the docked #wf-props panel.
     data,             // OPTIONAL: initial per-node data object (deep-ish-cloned per node).
   }

   Queue: if typeof Serenity.wfLower === 'function', Queue calls Serenity.wfLower(graph())
   and merges the returned params-patch object into state.params before submitting via the
   PROVEN /v1/generate path (api.submitPrompt reads Serenity.state.params). If wfLower is
   absent / returns falsy, Queue keeps the existing model+prompt behavior. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  function injectCSS() {
    if (document.getElementById("style-workflows")) return;
    var css = [
      "#view-workflows{display:none;flex-direction:column;height:100%}",
      "#view-workflows.show{display:flex}",
      "#wf-bar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line)}",
      "#wf-bar .wf-name{width:220px}",
      "#wf-bar .wf-sp{flex:1}",
      "#wf-bar .wf-b{font-size:12px;padding:6px 10px}",
      "#wf-canvas{flex:1;position:relative;overflow:hidden;",
      "  background-color:#11131a;background-image:linear-gradient(#1c2030 1px,transparent 1px),linear-gradient(90deg,#1c2030 1px,transparent 1px);background-size:24px 24px}",
      "#wf-stage{position:absolute;inset:0}",
      "#wf-menu{position:absolute;z-index:5;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:4px;display:none;min-width:200px;max-height:70%;overflow-y:auto}",
      "#wf-menu.show{display:block}",
      "#wf-menu .wf-menu-sep{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:7px 10px 3px}",
      "#wf-menu button{display:block;width:100%;text-align:left;background:transparent;border:0;color:var(--text);padding:7px 10px;border-radius:5px;cursor:pointer;font-size:12px}",
      "#wf-menu button:hover{background:var(--accent2);color:#fff}",
      "#wf-bar .wf-prompt{flex:1;min-width:120px}",
      "#wf-bar .wf-model{cursor:pointer}",
      "#wf-result{position:absolute;top:12px;right:12px;z-index:6;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:6px}",
      "#wf-result .wf-result-img{display:block;max-width:280px;max-height:280px;border-radius:6px}",
      "#wf-status{position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:8;background:var(--accent2);color:#fff;",
      "  border-radius:20px;padding:8px 18px;font-size:13px;font-weight:600;box-shadow:0 4px 14px rgba(0,0,0,.45);pointer-events:none}",
      // docked right-side properties panel (inside #wf-canvas)
      "#wf-props{position:absolute;top:0;right:0;bottom:0;width:300px;z-index:7;background:var(--panel);border-left:1px solid var(--line);",
      "  padding:12px;overflow-y:auto;display:none;box-shadow:-6px 0 16px rgba(0,0,0,.35)}",
      "#wf-props.show{display:block}",
      "#wf-props .wf-props-head{display:flex;align-items:center;gap:8px;margin-bottom:10px}",
      "#wf-props .wf-props-title{flex:1;font-size:13px;font-weight:700;color:var(--text)}",
      "#wf-props .wf-props-close{background:transparent;border:0;color:var(--muted);cursor:pointer;font-size:16px;padding:2px 6px}",
      "#wf-props .wf-props-close:hover{color:var(--text)}",
      "#wf-props .wf-props-empty{color:var(--muted);font-size:12px}",
      "#wf-props label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 3px}",
      "#wf-props input,#wf-props select,#wf-props textarea{width:100%}",
      "#wf-props textarea{min-height:44px}",
    ].join("\n");
    var st = document.createElement("style"); st.id = "style-workflows"; st.textContent = css;
    document.head.appendChild(st);
  }
  function el(tag, cls, txt) { var n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }

  // shallow clone of a def.data seed so every node gets its own copy
  function cloneData(d) {
    if (d == null) return {};
    try { return JSON.parse(JSON.stringify(d)); } catch (_) { return {}; }
  }

  S.register("workflows", {
    init: function (ctx) {
      injectCSS();
      var Konva = ctx.Konva, bus = ctx.bus, api = ctx.api, set = ctx.set, get = ctx.get;
      var host = document.getElementById("view-workflows");
      if (!host) { console.warn("[workflows] #view-workflows not found"); return; }

      // ============================================================
      // NODE-TYPE REGISTRY (THE CONTRACT backing store)
      // ============================================================
      // typeId -> normalized def. typeId defaults to def.label when register() is
      // called with the label-only legacy shape.
      var registry = {};       // { typeId: def }
      var registryOrder = [];   // typeIds in registration order (for the menu)

      function normalizeDef(typeId, def) {
        var ins = def.ins || [];
        var outs = def.outs || [];
        var rows = Math.max(ins.length, outs.length, 1);
        return {
          typeId: typeId,
          label: def.label != null ? def.label : typeId,
          color: def.color || "#555",
          // body nodes set their own w/h; port-only nodes get the classic sizing.
          w: def.w || 168,
          h: def.h || (34 + rows * 16 + 8),
          ins: ins,
          outs: outs,
          body: typeof def.body === "function" ? def.body : null,
          props: typeof def.props === "function" ? def.props : null,
          data: def.data || null,
          menu: def.menu !== false,          // show in Add-Node menu unless explicitly hidden
          category: def.category || "Nodes", // menu grouping
          resizable: !!def.resizable,        // opt-in bottom-right resize grip
          minW: def.minW || 220,
          minH: def.minH || 180,
          onResize: typeof def.onResize === "function" ? def.onResize : null,
        };
      }

      function register(typeId, def) {
        if (!typeId || !def) { console.warn("[workflows] register(typeId,def) needs both"); return; }
        var existed = !!registry[typeId];
        registry[typeId] = normalizeDef(typeId, def);
        if (!existed) registryOrder.push(typeId);
        if (built) rebuildMenu();   // late registrations still appear in the menu
        return registry[typeId];
      }

      // ============================================================
      // toolbar
      // ============================================================
      var bar = el("div"); bar.id = "wf-bar";
      var name = el("input", "wf-name"); name.value = "Untitled Workflow";
      var modelSel = el("select", "wf-b wf-model");
      [["Z-Image (base)", "z-image"], ["Ideogram4", "ideogram4"]].forEach(function (o) {
        var op = el("option", null, o[0]); op.value = o[1]; modelSel.appendChild(op);
      });
      var add = el("button", "btn wf-b", "+ Add Node");
      var promptIn = el("input", "wf-prompt"); promptIn.placeholder = "Prompt for this workflow…";
      var tpl = el("button", "btn wf-b", "Templates");
      var load = el("button", "btn wf-b", "Load");
      var save = el("button", "btn wf-b", "Save");
      var queue = el("button", "btn btn-primary wf-b", "▶ Generate");
      [name, modelSel, add, promptIn, tpl, load, save, queue].forEach(function (n) { bar.appendChild(n); });

      var cv = el("div"); cv.id = "wf-canvas";
      var stageHost = el("div"); stageHost.id = "wf-stage";  // Konva owns (and wipes) THIS
      var menu = el("div"); menu.id = "wf-menu";
      cv.appendChild(stageHost);   // keep menu/result/props as cv siblings, not inside Konva's container
      cv.appendChild(menu);
      host.appendChild(bar); host.appendChild(cv);

      // ============================================================
      // docked right-side properties panel (inside #wf-canvas)
      // ============================================================
      var propsPanel = el("div"); propsPanel.id = "wf-props";
      var propsHead = el("div", "wf-props-head");
      var propsTitle = el("div", "wf-props-title", "Properties");
      var propsClose = el("button", "wf-props-close", "✕");
      propsHead.appendChild(propsTitle); propsHead.appendChild(propsClose);
      var propsBody = el("div"); propsBody.className = "wf-props-body";
      propsPanel.appendChild(propsHead); propsPanel.appendChild(propsBody);
      cv.appendChild(propsPanel);
      propsClose.addEventListener("click", function () { selectNode(null); });

      function renderProps(node) {
        propsBody.innerHTML = "";
        if (!node) {
          propsPanel.classList.remove("show");
          return;
        }
        var def = registry[node.getAttr("wfType")];
        propsTitle.textContent = (def && def.label) || node.getAttr("wfType") || "Properties";
        propsPanel.classList.add("show");
        if (def && def.props) {
          try { def.props(node, propsBody, ctx); }
          catch (e) { console.error("[workflows] props() failed", e); propsBody.appendChild(el("div", "wf-props-empty", "props error: " + e.message)); }
        } else {
          propsBody.appendChild(el("div", "wf-props-empty", "No editable properties for this node."));
        }
      }

      // ============================================================
      // Add-Node menu (built FROM the registry)
      // ============================================================
      var lastMenuX = 80, lastMenuY = 80;
      function rebuildMenu() {
        menu.innerHTML = "";
        var lastCat = null;
        registryOrder.forEach(function (typeId) {
          var def = registry[typeId];
          if (!def || !def.menu) return;
          if (def.category !== lastCat) {
            menu.appendChild(el("div", "wf-menu-sep", def.category));
            lastCat = def.category;
          }
          var b = el("button", null, def.label);
          b.addEventListener("click", function () { addNodeByType(typeId, lastMenuX, lastMenuY); menu.classList.remove("show"); });
          menu.appendChild(b);
        });
      }
      add.addEventListener("click", function () {
        rebuildMenu();
        menu.style.left = "16px"; menu.style.top = "8px"; lastMenuX = 60; lastMenuY = 60; menu.classList.toggle("show");
      });

      // ============================================================
      // Konva stage (built lazily when the tab is first shown & sized)
      // ============================================================
      var stage = null, wireLayer = null, nodeLayer = null, built = false;
      var nodes = [], wires = [], drag = null, selectedNode = null;

      function build() {
        if (built) return;
        var w = cv.clientWidth || 1200, h = cv.clientHeight || 700;
        stage = new Konva.Stage({ container: stageHost, width: w, height: h, draggable: true });
        wireLayer = new Konva.Layer(); nodeLayer = new Konva.Layer();
        stage.add(wireLayer); stage.add(nodeLayer);
        // zoom on wheel (around pointer)
        stage.on("wheel", function (e) {
          e.evt.preventDefault();
          var old = stage.scaleX(); var ptr = stage.getPointerPosition();
          var by = e.evt.deltaY > 0 ? 0.9 : 1.1; var ns = Math.max(0.2, Math.min(2.5, old * by));
          var mp = { x: (ptr.x - stage.x()) / old, y: (ptr.y - stage.y()) / old };
          stage.scale({ x: ns, y: ns });
          stage.position({ x: ptr.x - mp.x * ns, y: ptr.y - mp.y * ns });
          stage.batchDraw();
        });
        // finalize/cancel wire on empty mouseup; click empty stage clears selection
        stage.on("mouseup", function () { if (drag) { cancelDrag(); } });
        stage.on("mousemove", function () { if (drag) { var p = stage.getRelativePointerPosition(); drag.line.points([drag.x0, drag.y0, p.x, p.y]); wireLayer.batchDraw(); } });
        stage.on("mousedown", function (e) { if (e.target === stage) selectNode(null); });
        // safety net: a mouse release OUTSIDE the stage container (Konva stage events only
        // fire inside it) must still tear down any in-flight title-bar node drag or wire
        // drag, so a node never gets "stuck" to the cursor.
        window.addEventListener("mouseup", function () {
          stage.off("mousemove.wfdrag"); stage.off("mouseup.wfdrag");
          stage.container().style.cursor = "default";
          if (drag) cancelDrag();
        });
        built = true;
        rebuildMenu();
        seed();
      }

      function portPos(node, kind, i) {
        var d = node.getAttr("nd");
        var x = node.x() + (kind === "out" ? d.w : 0);
        var y = node.y() + 34 + i * 16;
        return { x: x, y: y };
      }

      // ============================================================
      // addNode — registry-driven; supports custom bodies + title-bar drag
      // ============================================================
      function addNodeByType(typeId, x, y) {
        var def = registry[typeId];
        if (!def) { console.warn("[workflows] addNodeByType: unknown type", typeId); return null; }
        return addNode(def, x, y);
      }

      function addNode(def, x, y) {
        if (!built) build();
        var w = def.w, h = def.h, hasBody = !!def.body;
        // Body nodes: the GROUP is not draggable; only the title bar drags it (so in-body
        // pointer events — drawing, transformers — never move the node). Port-only nodes
        // keep the classic whole-group drag.
        var g = new Konva.Group({ x: x, y: y, draggable: !hasBody });
        g.setAttr("nd", { w: w, h: h, type: def.label });
        g.setAttr("wfType", def.typeId);
        g.setAttr("data", cloneData(def.data));

        // frame + title bar
        var frameRect = new Konva.Rect({ width: w, height: h, fill: "#1c1f26", stroke: def.color, strokeWidth: 2, cornerRadius: 7, shadowColor: "#000", shadowBlur: 8, shadowOpacity: 0.4 });
        g.add(frameRect);
        var titleBar = new Konva.Rect({ width: w, height: 24, fill: def.color, cornerRadius: [7, 7, 0, 0] });
        g.add(titleBar);
        g.add(new Konva.Text({ text: def.label, x: 8, y: 6, fontSize: 12, fontStyle: "bold", fill: "#fff", listening: false }));

        // title bar is the drag handle for body nodes (group itself isn't draggable)
        if (hasBody) {
          // The body frame swallows any mousedown that lands on empty body area so it
          // never reaches the (draggable) stage and starts a pan — it just selects the
          // node. Body shapes painted by def.body sit ON TOP of this frame and receive
          // their own events first, so this only guards the gaps. The node moves ONLY
          // by the title bar; in-body pointer events never move/pan the node, even if a
          // body author forgets cancelBubble.
          frameRect.on("mousedown", function (e) { e.cancelBubble = true; selectNode(g); });
          titleBar.on("mousedown", function (e) {
            e.cancelBubble = true;                 // don't start a stage pan
            selectNode(g);
            var start = stage.getRelativePointerPosition();
            var ox = start.x - g.x(), oy = start.y - g.y();
            function onMove() {
              var p = stage.getRelativePointerPosition();
              g.position({ x: p.x - ox, y: p.y - oy });
              redrawWires(); nodeLayer.batchDraw();
            }
            function onUp() { stage.off("mousemove.wfdrag"); stage.off("mouseup.wfdrag"); stage.container().style.cursor = "default"; }
            stage.on("mousemove.wfdrag", onMove);
            stage.on("mouseup.wfdrag", onUp);
          });
          // make title text not eat the drag
          titleBar.on("mouseenter", function () { stage.container().style.cursor = "move"; });
          titleBar.on("mouseleave", function () { stage.container().style.cursor = "default"; });
        } else {
          // whole-group drag selects on grab
          g.on("mousedown", function (e) { e.cancelBubble = true; selectNode(g); });
        }

        // ports
        def.ins.forEach(function (label, i) {
          g.add(new Konva.Text({ text: label, x: 12, y: 28 + i * 16, fontSize: 10, fill: "#aab", listening: false }));
          var c = new Konva.Circle({ x: 0, y: 34 + i * 16, radius: 5, fill: "#6c8cff", stroke: "#fff", strokeWidth: 1 });
          c.setAttr("port", { node: g, kind: "in", i: i });
          c.on("mouseup", function (e) { e.cancelBubble = true; finishDrag(g, "in", i); });
          g.add(c);
        });
        var outPortEls = [];
        def.outs.forEach(function (label, i) {
          var lbl = new Konva.Text({ text: label, x: w - 12 - label.length * 6, y: 28 + i * 16, fontSize: 10, fill: "#aab", listening: false });
          g.add(lbl);
          var c = new Konva.Circle({ x: w, y: 34 + i * 16, radius: 5, fill: "#ffcc6c", stroke: "#fff", strokeWidth: 1 });
          c.setAttr("port", { node: g, kind: "out", i: i });
          c.on("mousedown", function (e) { e.cancelBubble = true; startDrag(g, "out", i); });
          g.add(c);
          outPortEls.push({ circle: c, label: lbl });
        });

        // custom interactive body — painted AFTER ports so it sits on top of the frame
        if (hasBody) {
          try { def.body(g, g, ctx); }
          catch (e) { console.error("[workflows] body() failed for", def.typeId, e); }
        }

        // optional resize grip (bottom-right) for nodes that opt in via def.resizable.
        // Painted LAST so it sits above the body and stays clickable; drags on the STAGE
        // (so leaving the grip mid-resize doesn't break it) and calls def.onResize so the
        // body reflows. Boxes inside the bbox node are normalized, so they re-fit.
        if (def.resizable) {
          var grip = new Konva.Rect({ x: w - 16, y: h - 16, width: 13, height: 13, fill: def.color, opacity: 0.85, cornerRadius: 2, stroke: "#fff", strokeWidth: 1 });
          grip.on("mouseenter", function () { stage.container().style.cursor = "nwse-resize"; });
          grip.on("mouseleave", function () { stage.container().style.cursor = "default"; });
          grip.on("mousedown", function (e) {
            e.cancelBubble = true;
            function onMove() {
              var p = stage.getRelativePointerPosition();
              var nw = Math.max(def.minW || 220, Math.round(p.x - g.x()));
              var nh = Math.max(def.minH || 180, Math.round(p.y - g.y()));
              var nd = g.getAttr("nd"); nd.w = nw; nd.h = nh; g.setAttr("nd", nd);
              frameRect.setAttrs({ width: nw, height: nh });
              titleBar.setAttrs({ width: nw });
              grip.position({ x: nw - 16, y: nh - 16 });
              outPortEls.forEach(function (pe) { pe.circle.x(nw); pe.label.x(nw - 12 - pe.label.text().length * 6); });
              if (typeof def.onResize === "function") { try { def.onResize(g, nw, nh); } catch (_) {} }
              redrawWires(); nodeLayer.batchDraw();
            }
            function onUp() { stage.off("mousemove.wfresize"); stage.off("mouseup.wfresize"); stage.container().style.cursor = "default"; }
            stage.on("mousemove.wfresize", onMove); stage.on("mouseup.wfresize", onUp);
          });
          g.add(grip);
        }

        g.on("dragmove", redrawWires);   // port-only nodes (group draggable) redraw wires
        nodeLayer.add(g); nodes.push(g); nodeLayer.batchDraw();
        return g;
      }

      // ============================================================
      // selection
      // ============================================================
      function selectNode(g) {
        // Re-selecting the already-selected node is a no-op: the props panel is already
        // mounted and we must not re-run def.props (it would tear down/rebuild the editor
        // mid-interaction). Selection changes mount the new editor / unmount on null.
        if (selectedNode === g) return;
        selectedNode = g;
        renderProps(g);   // g==null → clears + hides the docked props panel (clean unmount)
      }

      // ============================================================
      // wires
      // ============================================================
      function startDrag(node, kind, i) {
        var p = portPos(node, "out", i);
        var line = new Konva.Line({ points: [p.x, p.y, p.x, p.y], stroke: "#ffcc6c", strokeWidth: 2, bezier: true, tension: 0 });
        wireLayer.add(line); wireLayer.batchDraw();
        drag = { from: node, fromI: i, line: line, x0: p.x, y0: p.y };
      }
      function finishDrag(node, kind, i) {
        if (!drag || kind !== "in") return;
        wires.push({ from: drag.from, fromI: drag.fromI, to: node, toI: i, line: drag.line });
        drag.line.stroke("#8fa0c8"); drag = null; redrawWires();
      }
      function cancelDrag() { if (drag) { drag.line.destroy(); drag = null; wireLayer.batchDraw(); } }

      function redrawWires() {
        wires.forEach(function (wd) {
          var a = portPos(wd.from, "out", wd.fromI), b = portPos(wd.to, "in", wd.toI);
          var dx = Math.abs(b.x - a.x) * 0.5;
          wd.line.points([a.x, a.y, a.x + dx, a.y, b.x - dx, b.y, b.x, b.y]);
        });
        wireLayer.batchDraw();
      }

      // ============================================================
      // graph() — { nodes:[{type,data,group}], wires:[{from,fromI,to,toI}] }
      // ============================================================
      function graph() {
        var nodeList = nodes.map(function (g) {
          return { type: g.getAttr("wfType"), data: g.getAttr("data"), group: g };
        });
        var wireList = wires.map(function (wd) {
          return {
            from: wd.from.getAttr("wfType"), fromI: wd.fromI,
            to: wd.to.getAttr("wfType"), toI: wd.toI,
            fromGroup: wd.from, toGroup: wd.to,
          };
        });
        return { nodes: nodeList, wires: wireList };
      }

      // ============================================================
      // register the existing 6 built-in node types (no regression)
      // ============================================================
      [
        { typeId: "Load Checkpoint", ins: [], outs: ["MODEL", "CLIP", "VAE"], color: "#3a4b8a", category: "Built-in" },
        { typeId: "CLIP Text Encode", ins: ["CLIP"], outs: ["COND"], color: "#3a7a4b", category: "Built-in" },
        { typeId: "Empty Latent", ins: [], outs: ["LATENT"], color: "#7a6a3a", category: "Built-in" },
        { typeId: "KSampler", ins: ["MODEL", "+COND", "-COND", "LATENT"], outs: ["LATENT"], color: "#8a3a6a", category: "Built-in" },
        { typeId: "VAE Decode", ins: ["LATENT", "VAE"], outs: ["IMAGE"], color: "#3a6a8a", category: "Built-in" },
        { typeId: "Save Image", ins: ["IMAGE"], outs: [], color: "#555", category: "Built-in" },
      ].forEach(function (b) {
        register(b.typeId, { label: b.typeId, ins: b.ins, outs: b.outs, color: b.color, category: b.category });
      });

      function seed() {
        var n1 = addNodeByType("Load Checkpoint", 40, 60);
        var n2 = addNodeByType("CLIP Text Encode", 280, 50);
        var n3 = addNodeByType("Empty Latent", 280, 200);
        var n4 = addNodeByType("KSampler", 520, 90);
        var n5 = addNodeByType("VAE Decode", 760, 110);
        var n6 = addNodeByType("Save Image", 980, 120);
        function wire(a, ai, b, bi) {
          var pa = portPos(a, "out", ai), pb = portPos(b, "in", bi);
          var line = new Konva.Line({ stroke: "#8fa0c8", strokeWidth: 2 }); wireLayer.add(line);
          wires.push({ from: a, fromI: ai, to: b, toI: bi, line: line });
        }
        wire(n1, 1, n2, 0); wire(n1, 0, n4, 0); wire(n2, 0, n4, 1); wire(n3, 0, n4, 3);
        wire(n4, 0, n5, 0); wire(n1, 2, n5, 1); wire(n5, 0, n6, 0);
        redrawWires(); nodeLayer.batchDraw();
      }

      // ============================================================
      // PUBLIC CONTRACT
      // ============================================================
      S.workflows = {
        register: register,
        graph: function () { if (!built) build(); return graph(); },
        addNodeByType: function (typeId, x, y) { if (!built) build(); return addNodeByType(typeId, x, y); },
      };

      // result preview panel (top-right of the canvas)
      var resPanel = el("div"); resPanel.id = "wf-result"; resPanel.style.display = "none"; cv.appendChild(resPanel);
      function showResult(url) {
        resPanel.innerHTML = "";
        var img = new Image(); img.className = "wf-result-img"; img.src = url;
        resPanel.appendChild(img); resPanel.style.display = "block";
      }
      // robust display: any finished result (this workflow OR elsewhere) shows here
      bus.on("result:ready", function (p) { if (p && p.url) showResult(p.url); });

      // ============================================================
      // Queue -> run via the PROVEN /v1/generate path.
      //   - If Serenity.wfLower is defined, call it with graph(); it returns a params
      //     patch (object) which we merge into state.params before submit. A node (e.g.
      //     the Ideogram-4 bbox node) thus drives the exact render recipe + prompt.
      //   - Otherwise keep the existing model+prompt toolbar behavior.
      // ============================================================
      function applyParamsPatch(patch) {
        if (!patch || typeof patch !== "object") return;
        Object.keys(patch).forEach(function (k) { set("params." + k, patch[k]); });
      }
      // prominent generation-status overlay (phase + progress) over the canvas
      var wfStatus = el("div"); wfStatus.id = "wf-status"; wfStatus.style.display = "none"; cv.appendChild(wfStatus);
      var PHASE_LABEL = { encoding: "Encoding prompt…", loading: "Loading model…", decoding: "Decoding image…", sampling: "Sampling…", prepare: "Preparing…" };
      function setStatus(txt) { if (!txt) { wfStatus.style.display = "none"; return; } wfStatus.textContent = txt; wfStatus.style.display = "block"; }
      function endRun(label) { queue.disabled = false; queue.textContent = "▶ Generate" + (label || ""); }

      queue.addEventListener("click", function () {
        var lowered = null;
        if (typeof S.wfLower === "function") {
          try { lowered = S.wfLower(graph()); }
          catch (e) { console.error("[workflows] wfLower() failed", e); }
        }
        if (lowered && typeof lowered === "object") {
          applyParamsPatch(lowered);
        } else {
          // legacy behavior: toolbar model + prompt
          set("params.model", modelSel.value);
          if (promptIn.value.trim()) set("params.prompt", promptIn.value);
        }
        queue.disabled = true; queue.textContent = "▶ …";
        setStatus("⏳ Starting…");
        var shown = false, poll = null;
        function stopPoll() { if (poll) { clearInterval(poll); poll = null; } }
        // idempotent result display — fired by the WS 'done' OR the fallback poll,
        // whichever lands first, so a dropped/missed WS event never loses the image.
        function show(url, fname) {
          if (shown) return; shown = true; stopPoll();
          showResult(url);
          bus.emit("result:ready", { url: url, filename: fname || url, params: {} });
          endRun(""); setStatus("✓ Done"); setTimeout(function () { setStatus(""); }, 1500);
        }
        api.connectWS(ctx.clientId, function (msg) {
          if (!msg) return;
          if (msg.type === "progress") {
            var d = msg.data || {}, ph = (d.phase || "").toLowerCase();
            var txt = PHASE_LABEL[ph] || (d.max ? ("Step " + (d.value || 0) + " / " + d.max) : "Working…");
            setStatus("⏳ " + txt);
            queue.textContent = "▶ " + (ph || ((d.value || 0) + "/" + (d.max || 0)));
          } else if (msg.type === "executed") {
            var imgs = (msg.data && msg.data.output && msg.data.output.images) || [];
            if (imgs[0]) show(api.viewUrl(imgs[0].filename), imgs[0].filename);
          } else if (msg.type === "executing" && msg.data && msg.data.node === null) {
            if (!shown) endRun("");
          } else if (msg.type === "execution_error") {
            stopPoll(); endRun(" (error)"); setStatus("⚠ Failed: " + ((msg.data && msg.data.error) || "see console"));
            console.error("[workflows] exec error", msg.data);
          }
        });
        api.submitPrompt(null, ctx.clientId)
          .then(function (res) {
            var jid = res && (res.job_id || res.prompt_id);
            console.info("[workflows] generate", jid);
            // FALLBACK: the worker writes out_dir/<job_id>.png -> /out/<job_id>.png. Poll it
            // so the image shows even if the progress WS missed the 'done' event.
            if (jid) {
              var url = api.viewUrl(jid + ".png"), tries = 0;
              poll = setInterval(function () {
                if (shown || tries++ > 400) { stopPoll(); return; }
                fetch(url, { method: "HEAD", cache: "no-store" })
                  .then(function (r) { if (r.ok) show(url, jid + ".png"); })
                  .catch(function () {});
              }, 2500);
            }
          })
          .catch(function (e) { stopPoll(); endRun(" (failed)"); setStatus("⚠ Submit failed"); console.warn("[workflows] submit failed", e); });
      });
      [tpl, load, save].forEach(function (b) { b.addEventListener("click", function () { console.info("[workflows]", b.textContent, "(TODO)"); }); });

      // build/size the stage when the Workflows tab is shown
      bus.on("nav:tab", function (t) {
        if (t !== "workflows") return;
        setTimeout(function () {
          build();
          if (stage) { stage.size({ width: cv.clientWidth || 1200, height: cv.clientHeight || 700 }); stage.batchDraw(); }
        }, 30);
      });
      // let other stages know the contract is live (so registrations done after init still wire up)
      bus.emit("workflows:ready", S.workflows);
      console.info("[workflows] ready (contract: Serenity.workflows.register/graph/addNodeByType)");
    },
  });
})();
