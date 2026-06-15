/* ideogram4_nodes.js — module 'ideogram4Nodes' (STAGE NODES-B).

   The KJNodes Ideogram-4 node SET + the render LOWERING, registered against the
   Stage-Infra contract (window.Serenity.workflows). Everything here is a NODE in the
   Konva node-graph editor (workflows.js) — NOT a panel, NOT a tab, NOT an overlay.

   This file owns three things:

   (1) The supporting KJ Ideogram-4 node TYPES — plain port-nodes (no def.body) that
       carry the fixed render recipe in node.data. They wire visually around the
       interactive bbox node (Ideogram4PromptBuilderKJ, registered by STAGE NODES-A,
       which paints the drawable bbox body and stores its caption in node.data.caption).

   (2) window.Serenity.wfLower(graph): finds the Ideogram4PromptBuilderKJ node, reads
       node.data.caption + width/height, and RETURNS the /v1/generate params patch for
       the VERIFIED Ideogram-4 render recipe — model 'ideogram4', scheduler
       'ideogram_logitnormal', cfg 7, cfg_override -1 (OFF), sampler 'euler', steps 20
       (overridable from a BasicScheduler/KSampler node). The simple_flowmatch +
       cfg_override "KJ recipe" GRAY-BLOCKS our fp8 latents and is FORBIDDEN here.

   (3) An "Ideogram4 (KJ)" template: adds + wires the whole KJ node chain (incl. the
       bbox node) onto the graph via Serenity.workflows.addNodeByType. Hooked to the
       Templates toolbar button.

   Wiring is ordering-agnostic: we bind to the contract on the 'workflows:ready' bus
   event, and also immediately if Serenity.workflows already exists. Registration is
   idempotent (guarded), so being inited both as a module AND via the bus is safe. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  // The typeId of the interactive bbox node painted by STAGE NODES-A.
  var BBOX_TYPE = "Ideogram4PromptBuilderKJ";

  // ============================================================
  // KJ Ideogram-4 supporting node types (port-only; recipe in node.data)
  // ============================================================
  // These mirror the KJNodes Ideogram-4 graph. They are visual/structural: the actual
  // render recipe is fixed and emitted by wfLower(). Each node carries its knobs in
  // node.data so a future lowering can read them; today wfLower reads steps/dims and
  // pins everything else to the VERIFIED path.
  var NODE_DEFS = [
    {
      typeId: "UNETLoader (ideogram4 cond)",
      label: "UNETLoader (ideogram4 cond)",
      color: "#3a4b8a",
      ins: [], outs: ["MODEL"],
      data: { unet_name: "ideogram4", weight_dtype: "fp8", role: "cond" },
    },
    {
      typeId: "UNETLoader (uncond)",
      label: "UNETLoader (uncond)",
      color: "#3a4b8a",
      ins: [], outs: ["MODEL"],
      data: { unet_name: "ideogram4", weight_dtype: "fp8", role: "uncond" },
    },
    {
      typeId: "ModelSamplingAuraFlow",
      label: "ModelSamplingAuraFlow",
      color: "#5a4b8a",
      ins: ["MODEL"], outs: ["MODEL"],
      data: { shift: 7 },
    },
    {
      typeId: "DualModelGuider",
      label: "DualModelGuider",
      color: "#8a3a6a",
      ins: ["MODEL (cond)", "MODEL (uncond)", "+COND", "-COND"], outs: ["GUIDER"],
      data: { cfg: 7 },
    },
    {
      // structural only — wfLower keeps cfg_override OFF (-1) on this VERIFIED path.
      typeId: "CFGOverride",
      label: "CFGOverride",
      color: "#8a6a3a",
      ins: ["GUIDER"], outs: ["GUIDER"],
      data: { cfg_override_start: 3, cfg_override_value: 0.7, cfg_override_end: 1 },
    },
    {
      typeId: "BasicScheduler",
      label: "BasicScheduler",
      color: "#7a6a3a",
      ins: ["MODEL"], outs: ["SIGMAS"],
      data: { scheduler: "ideogram_logitnormal", steps: 20, denoise: 1.0 },
    },
    {
      typeId: "KSamplerSelect",
      label: "KSamplerSelect",
      color: "#8a3a6a",
      ins: [], outs: ["SAMPLER"],
      data: { sampler_name: "euler" },
    },
    {
      typeId: "ConditioningZeroOut",
      label: "ConditioningZeroOut",
      color: "#3a7a4b",
      ins: ["COND"], outs: ["COND"],
      data: {},
    },
    {
      typeId: "VAEDecode",
      label: "VAEDecode",
      color: "#3a6a8a",
      ins: ["LATENT", "VAE"], outs: ["IMAGE"],
      data: {},
    },
    {
      typeId: "SaveImage",
      label: "SaveImage",
      color: "#555",
      ins: ["IMAGE"], outs: [],
      data: { filename_prefix: "ideogram4" },
    },
  ];

  // ============================================================
  // lowering — graph -> /v1/generate params patch (VERIFIED recipe)
  // ============================================================
  function findByType(graph, typeId) {
    var n = graph && graph.nodes;
    if (!n) return null;
    for (var i = 0; i < n.length; i++) if (n[i].type === typeId) return n[i];
    return null;
  }
  // The bbox node may be registered under BBOX_TYPE; be tolerant and also accept any
  // node whose data carries a non-empty caption (so a renamed/aliased builder node
  // still drives the render).
  function findBboxNode(graph) {
    var exact = findByType(graph, BBOX_TYPE);
    if (exact) return exact;
    var n = graph && graph.nodes;
    if (!n) return null;
    for (var i = 0; i < n.length; i++) {
      var d = n[i].data;
      if (d && typeof d.caption === "string" && d.caption.trim()) return n[i];
    }
    return null;
  }
  function intOr(v, dflt) {
    var x = parseInt(v, 10);
    return isFinite(x) && x > 0 ? x : dflt;
  }

  // wfLower: called by workflows.js Queue with graph(). Returns the params patch merged
  // into Serenity.state.params before the PROVEN /v1/generate submit, or null to fall
  // back to the toolbar model+prompt behavior.
  function wfLower(graph) {
    var bbox = findBboxNode(graph);
    if (!bbox) return null;                          // no Ideogram-4 builder node → legacy path
    var d = bbox.data || {};
    var caption = (d.caption != null ? String(d.caption) : "").trim();
    if (!caption) return null;                       // nothing drawn yet → legacy path

    // steps: prefer a BasicScheduler/KSampler node's value, else the bbox node's, else 20.
    var steps = 20;
    var sch = findByType(graph, "BasicScheduler");
    if (sch && sch.data && sch.data.steps != null) steps = intOr(sch.data.steps, steps);
    else if (d.steps != null) steps = intOr(d.steps, steps);

    var width = intOr(d.width, 1024);
    var height = intOr(d.height, 1024);

    // VERIFIED Ideogram-4 render path (pipeline/ideogram4_generate.mojo, PSNR 29.7dB vs
    // torch): logit-normal schedule + dual cond/uncond + constant cfg 7, euler, 20 steps.
    // cfg_override is OFF (-1): the worker only honors it on the simple scheduler, which
    // GRAY-BLOCKS fp8 latents here. Do NOT change these knobs.
    return {
      model: "ideogram4",
      prompt: caption,
      scheduler: "ideogram_logitnormal",
      cfg: 7.0,
      cfg_override: -1,
      sampler: "euler",
      steps: steps,
      width: width,
      height: height,
    };
  }

  // ============================================================
  // "Ideogram4 (KJ)" template — adds + wires the node chain (incl. the bbox node)
  // ============================================================
  function buildTemplate(wf) {
    var add = wf.addNodeByType;

    // Place the KJ Ideogram-4 chain left→right in signal-flow order. The contract's
    // public API is addNodeByType (returns the Konva.Group); wire creation is not part
    // of the frozen Stage-Infra contract, so nodes are laid out in flow order for the
    // user to connect. The render is driven end-to-end by wfLower reading the bbox
    // node's caption — it needs NO wires to produce the verified Ideogram-4 image.
    //
    // bbox builder node (STAGE NODES-A) sits at the head. If it isn't registered yet
    // addNodeByType returns null; the supporting chain is still placed and wfLower falls
    // back to the legacy path until a builder node exists.
    var placed = [];
    function add0(type, x, y) { var g = add(type, x, y); if (g) placed.push(g); return g; }

    var bbox = add0(BBOX_TYPE, 40, 60);

    // row of loaders/guider/decode (signal flow)
    add0("UNETLoader (ideogram4 cond)", 40, 380);
    add0("UNETLoader (uncond)", 40, 470);
    add0("ConditioningZeroOut", 40, 560);
    add0("ModelSamplingAuraFlow", 300, 380);
    add0("BasicScheduler", 300, 500);
    add0("KSamplerSelect", 300, 590);
    add0("DualModelGuider", 560, 400);
    add0("CFGOverride", 820, 400);
    add0("VAEDecode", 1060, 400);
    add0("SaveImage", 1300, 400);

    console.info("[ideogram4Nodes] template 'Ideogram4 (KJ)' placed " + placed.length + " nodes",
      bbox ? "(with bbox node)" : "(bbox node not registered yet — render still works once one is added)");
    return bbox;
  }

  // hook the Templates toolbar button (best-effort; non-fatal if absent)
  function hookTemplatesButton(wf) {
    var bar = document.getElementById("wf-bar");
    if (!bar) return;
    var btns = bar.querySelectorAll("button");
    for (var i = 0; i < btns.length; i++) {
      var b = btns[i];
      if ((b.textContent || "").trim() === "Templates") {
        if (b.getAttribute("data-i4-hooked")) return;
        b.setAttribute("data-i4-hooked", "1");
        b.addEventListener("click", function (e) {
          e.stopPropagation();
          buildTemplate(wf);
        });
        return;
      }
    }
  }

  // ============================================================
  // contract binding (ordering-agnostic, idempotent)
  // ============================================================
  var bound = false;
  function bind(wf) {
    if (bound || !wf || typeof wf.register !== "function") return;
    bound = true;
    NODE_DEFS.forEach(function (def) {
      wf.register(def.typeId, {
        label: def.label,
        color: def.color,
        ins: def.ins,
        outs: def.outs,
        data: def.data,
        category: "Ideogram-4 (KJ)",
      });
    });
    // install the render lowering used by Queue
    window.Serenity.wfLower = wfLower;
    // wire the Templates button (the wf-bar exists once workflows init has run)
    hookTemplatesButton(wf);
    console.info("[ideogram4Nodes] bound: " + NODE_DEFS.length + " KJ node types + wfLower + Ideogram4 (KJ) template");
  }

  S.register("ideogram4Nodes", {
    init: function (ctx) {
      // bind now if the contract is already up...
      if (window.Serenity.workflows) bind(window.Serenity.workflows);
      // ...and/or when workflows announces itself (covers any init ordering).
      if (ctx && ctx.bus && typeof ctx.bus.on === "function") {
        ctx.bus.on("workflows:ready", function (wf) { bind(wf || window.Serenity.workflows); });
      } else if (S.bus && typeof S.bus.on === "function") {
        S.bus.on("workflows:ready", function (wf) { bind(wf || window.Serenity.workflows); });
      }
      // expose the template + lowering for tests / external callers
      S.ideogram4Nodes = { wfLower: wfLower, buildTemplate: buildTemplate, BBOX_TYPE: BBOX_TYPE, NODE_DEFS: NODE_DEFS };
    },
  });

  // also self-bind off the bus even if this module is never inited by main.js
  // (e.g. not yet in EXPECTED): keeps the stage self-contained.
  if (S.bus && typeof S.bus.on === "function") {
    S.bus.on("workflows:ready", function (wf) { bind(wf || window.Serenity.workflows); });
  }
  if (window.Serenity.workflows) bind(window.Serenity.workflows);
})();
