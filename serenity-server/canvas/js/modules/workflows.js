/* workflows.js — module 'workflows'. A Konva node-graph editor in the Workflows tab
   (#view-workflows, created by nav_shell). Toolbar (name/Add Node/Templates/Load/Save/
   Queue) + a grid canvas with draggable nodes, input/output ports, wire dragging,
   pan (drag empty space) and zoom (wheel). v1 interactive core; Queue->/prompt later. */
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
      "#wf-menu{position:absolute;z-index:5;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:4px;display:none;min-width:170px}",
      "#wf-menu.show{display:block}",
      "#wf-menu button{display:block;width:100%;text-align:left;background:transparent;border:0;color:var(--text);padding:7px 10px;border-radius:5px;cursor:pointer;font-size:12px}",
      "#wf-menu button:hover{background:var(--accent2);color:#fff}",
      "#wf-bar .wf-prompt{flex:1;min-width:120px}",
      "#wf-bar .wf-model{cursor:pointer}",
      "#wf-result{position:absolute;top:12px;right:12px;z-index:6;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:6px}",
      "#wf-result .wf-result-img{display:block;max-width:280px;max-height:280px;border-radius:6px}",
    ].join("\n");
    var st = document.createElement("style"); st.id = "style-workflows"; st.textContent = css;
    document.head.appendChild(st);
  }
  function el(tag, cls, txt) { var n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }

  var NODE_TYPES = [
    { t: "Load Checkpoint", ins: [], outs: ["MODEL", "CLIP", "VAE"], c: "#3a4b8a" },
    { t: "CLIP Text Encode", ins: ["CLIP"], outs: ["COND"], c: "#3a7a4b" },
    { t: "Empty Latent", ins: [], outs: ["LATENT"], c: "#7a6a3a" },
    { t: "KSampler", ins: ["MODEL", "+COND", "-COND", "LATENT"], outs: ["LATENT"], c: "#8a3a6a" },
    { t: "VAE Decode", ins: ["LATENT", "VAE"], outs: ["IMAGE"], c: "#3a6a8a" },
    { t: "Save Image", ins: ["IMAGE"], outs: [], c: "#555" },
  ];

  S.register("workflows", {
    init: function (ctx) {
      injectCSS();
      var Konva = ctx.Konva, bus = ctx.bus, api = ctx.api, set = ctx.set, get = ctx.get;
      var host = document.getElementById("view-workflows");
      if (!host) { console.warn("[workflows] #view-workflows not found"); return; }

      // ---- toolbar ----
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
      var queue = el("button", "btn btn-primary wf-b", "▶ Queue");
      [name, modelSel, add, promptIn, tpl, load, save, queue].forEach(function (n) { bar.appendChild(n); });

      var cv = el("div"); cv.id = "wf-canvas";
      var stageHost = el("div"); stageHost.id = "wf-stage";  // Konva owns (and wipes) THIS
      var menu = el("div"); menu.id = "wf-menu";
      cv.appendChild(stageHost);   // keep menu/result panel as cv siblings, not inside Konva's container
      cv.appendChild(menu);
      host.appendChild(bar); host.appendChild(cv);

      // add-node menu
      NODE_TYPES.forEach(function (nt) {
        var b = el("button", null, nt.t);
        b.addEventListener("click", function () { addNode(nt, lastMenuX, lastMenuY); menu.classList.remove("show"); });
        menu.appendChild(b);
      });
      var lastMenuX = 80, lastMenuY = 80;
      add.addEventListener("click", function () {
        menu.style.left = "16px"; menu.style.top = "8px"; lastMenuX = 60; lastMenuY = 60; menu.classList.toggle("show");
      });

      // ---- Konva stage (built lazily when the tab is first shown & sized) ----
      var stage = null, gridDummy = null, wireLayer = null, nodeLayer = null, built = false;
      var nodes = [], wires = [], drag = null;

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
        // finalize/cancel wire on empty mouseup
        stage.on("mouseup", function () { if (drag) { cancelDrag(); } });
        stage.on("mousemove", function () { if (drag) { var p = stage.getRelativePointerPosition(); drag.line.points([drag.x0, drag.y0, p.x, p.y]); wireLayer.batchDraw(); } });
        built = true;
        seed();
      }

      function portPos(node, kind, i) {
        var d = node.getAttr("nd");
        var x = node.x() + (kind === "out" ? d.w : 0);
        var y = node.y() + 34 + i * 16;
        return { x: x, y: y };
      }

      function addNode(nt, x, y) {
        if (!built) build();
        var w = 168, rows = Math.max(nt.ins.length, nt.outs.length, 1), h = 34 + rows * 16 + 8;
        var g = new Konva.Group({ x: x, y: y, draggable: true });
        g.setAttr("nd", { w: w, h: h, type: nt.t });
        g.add(new Konva.Rect({ width: w, height: h, fill: "#1c1f26", stroke: nt.c, strokeWidth: 2, cornerRadius: 7, shadowColor: "#000", shadowBlur: 8, shadowOpacity: 0.4 }));
        g.add(new Konva.Rect({ width: w, height: 24, fill: nt.c, cornerRadius: [7, 7, 0, 0] }));
        g.add(new Konva.Text({ text: nt.t, x: 8, y: 6, fontSize: 12, fontStyle: "bold", fill: "#fff" }));
        nt.ins.forEach(function (label, i) {
          g.add(new Konva.Text({ text: label, x: 12, y: 28 + i * 16, fontSize: 10, fill: "#aab" }));
          var c = new Konva.Circle({ x: 0, y: 34 + i * 16, radius: 5, fill: "#6c8cff", stroke: "#fff", strokeWidth: 1 });
          c.setAttr("port", { node: g, kind: "in", i: i });
          c.on("mouseup", function (e) { e.cancelBubble = true; finishDrag(g, "in", i); });
          g.add(c);
        });
        nt.outs.forEach(function (label, i) {
          g.add(new Konva.Text({ text: label, x: w - 12 - label.length * 6, y: 28 + i * 16, fontSize: 10, fill: "#aab" }));
          var c = new Konva.Circle({ x: w, y: 34 + i * 16, radius: 5, fill: "#ffcc6c", stroke: "#fff", strokeWidth: 1 });
          c.setAttr("port", { node: g, kind: "out", i: i });
          c.on("mousedown", function (e) { e.cancelBubble = true; startDrag(g, "out", i); });
          g.add(c);
        });
        g.on("dragmove", redrawWires);
        nodeLayer.add(g); nodes.push(g); nodeLayer.batchDraw();
        return g;
      }

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

      function seed() {
        var n1 = addNode(NODE_TYPES[0], 40, 60);
        var n2 = addNode(NODE_TYPES[1], 280, 50);
        var n3 = addNode(NODE_TYPES[2], 280, 200);
        var n4 = addNode(NODE_TYPES[3], 520, 90);
        var n5 = addNode(NODE_TYPES[4], 760, 110);
        var n6 = addNode(NODE_TYPES[5], 980, 120);
        function wire(a, ai, b, bi) {
          var pa = portPos(a, "out", ai), pb = portPos(b, "in", bi);
          var line = new Konva.Line({ stroke: "#8fa0c8", strokeWidth: 2 }); wireLayer.add(line);
          wires.push({ from: a, fromI: ai, to: b, toI: bi, line: line });
        }
        wire(n1, 1, n2, 0); wire(n1, 0, n4, 0); wire(n2, 0, n4, 1); wire(n3, 0, n4, 3);
        wire(n4, 0, n5, 0); wire(n1, 2, n5, 1); wire(n5, 0, n6, 0);
        redrawWires(); nodeLayer.batchDraw();
      }

      // result preview panel (top-right of the canvas)
      var resPanel = el("div"); resPanel.id = "wf-result"; resPanel.style.display = "none"; cv.appendChild(resPanel);
      function showResult(url) {
        resPanel.innerHTML = "";
        var img = new Image(); img.className = "wf-result-img"; img.src = url;
        resPanel.appendChild(img); resPanel.style.display = "block";
      }
      // robust display: any finished result (this workflow OR elsewhere) shows here
      bus.on("result:ready", function (p) { if (p && p.url) showResult(p.url); });
      // Queue -> run the workflow via the PROVEN /v1/generate path. The server swaps
      // the worker binary by model (z-image <-> ideogram4), so picking Ideogram4 here
      // routes to serenity_worker_ideogram4.
      queue.addEventListener("click", function () {
        set("params.model", modelSel.value);
        if (promptIn.value.trim()) set("params.prompt", promptIn.value);
        queue.disabled = true; queue.textContent = "▶ …";
        api.connectWS(ctx.clientId, function (msg) {
          if (!msg) return;
          if (msg.type === "progress") {
            var d = msg.data || {}; queue.textContent = "▶ " + (d.value || 0) + "/" + (d.max || 0);
          } else if (msg.type === "executed") {
            var imgs = (msg.data && msg.data.output && msg.data.output.images) || [];
            if (imgs[0]) {
              var url = api.viewUrl(imgs[0].filename);
              showResult(url);
              bus.emit("result:ready", { url: url, filename: imgs[0].filename, params: {} });
            }
          } else if (msg.type === "executing" && msg.data && msg.data.node === null) {
            queue.disabled = false; queue.textContent = "▶ Queue";
          } else if (msg.type === "execution_error") {
            queue.disabled = false; queue.textContent = "▶ Queue (error)";
            console.error("[workflows] exec error", msg.data);
          }
        });
        api.submitPrompt(null, ctx.clientId)
          .then(function (res) { console.info("[workflows] queued", res && res.job_id, "model=", modelSel.value); })
          .catch(function (e) { queue.disabled = false; queue.textContent = "▶ Queue (failed)"; console.warn("[workflows] submit failed", e); });
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
      console.info("[workflows] ready");
    },
  });
})();
