/* advanced_sampling.js — module 'advancedSampling'. A collapsible
   "Advanced Sampling" panel injected into the LEFT param rail (#param-rail),
   mirroring SwarmUI's Sampling group. Every control writes state.params.* via
   set(); the orchestrator forwards the new fields to the backend (api.js
   generateBody). This module owns ONLY this file; it talks to the rest of the
   app exclusively through ctx (state/get/set/bus/api) and never imports other
   modules.

   Fields written:
     params.cfg                (float, shared w/ param_rail — kept in sync via bus)
     params.sigma_shift        (float, rectified-flow / DiT freq balance)
     params.clip_skip          (int,   CLIP stop-at-layer)
     params.eta                (float, -1 = model default)
     params.sigma_min          (float, -1 = model default)
     params.sigma_max          (float, -1 = model default)
     params.restart_sampling   (bool)
     params.batch_size         (int,   per-GPU batch — distinct from images count)
     params.end_steps_early_pct(float, 0..1; 0 = run all steps)
     params.no_seed_increment  (bool,  SwarmUI "No Seed Increment")

   Two-way binding: each control listens to bus 'change:<path>' so that if
   another module (e.g. param_rail, or reuse-params from the gallery) mutates
   the same state path, this panel stays in sync without fighting the user
   (we skip updating a control while it has focus).
*/
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  var ROOT_ID = "param-rail";

  // ----------------------------------------------------------------- CSS
  function injectCSS() {
    if (document.getElementById("style-advancedSampling")) return;
    var P = "#" + ROOT_ID + " ";
    var css = [
      P + ".as-section{margin-bottom:10px}",
      P + ".as-body{padding:0 8px 8px}",
      P + ".as-lbl{display:block;color:var(--muted);font-size:11px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.04em}",
      P + ".as-lbl .as-val{float:right;color:var(--text);text-transform:none;letter-spacing:0;font-weight:600}",
      P + ".as-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}",
      P + ".as-field{flex:1;min-width:0}",
      P + ".as-field input,#" + ROOT_ID + " .as-field select{width:100%}",
      P + ".as-num{width:100%;-moz-appearance:textfield}",
      P + ".as-check{display:flex;align-items:center;gap:8px;margin:8px 0;cursor:pointer}",
      P + ".as-check input{flex:0 0 auto}",
      P + ".as-check span{color:var(--text)}",
      P + ".as-hint{color:var(--muted);font-size:11px;margin:4px 0 2px}",
      P + ".group.as-section[open]>summary{border-bottom:1px solid var(--line)}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-advancedSampling";
    st.textContent = css;
    document.head.appendChild(st);
  }

  // ----------------------------------------------------------------- DOM helper
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k.slice(0, 2) === "on" && typeof attrs[k] === "function") n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    }
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  // ----------------------------------------------------------------- numeric helpers
  function decimals(step) { return (String(step).split(".")[1] || "").length; }
  function parseStep(v, step) {
    var n = parseFloat(v);
    if (isNaN(n)) return NaN;
    if (step != null && step < 1) return +n.toFixed(decimals(step));
    return Math.round(n);
  }
  function fmt(v, o) {
    if (v == null || v === "") return "";
    if (o && o.dec != null) return (+v).toFixed(o.dec);
    if (o && o.step != null && o.step < 1) return (+v).toFixed(decimals(o.step));
    return String(v);
  }

  // ----------------------------------------------------------------- module
  S.register("advancedSampling", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus;
      var root = (ctx.dom && ctx.dom.paramRail) || document.getElementById(ROOT_ID);
      if (!root) { console.warn("[advancedSampling] mount not found:", ROOT_ID); return; }

      // idempotent: if this panel was already injected, bail (boot is single-shot
      // but this keeps a re-init from duplicating the group).
      if (root.querySelector("#as-panel")) return;

      // Seed any new state.params.* defaults that aren't in the shared store yet,
      // WITHOUT clobbering values another module/state already set.
      seedDefault("params.sigma_shift", 3.0);
      seedDefault("params.batch_size", 1);
      seedDefault("params.end_steps_early_pct", 0.0);
      seedDefault("params.no_seed_increment", false);
      function seedDefault(path, def) { if (get(path) == null) set(path, def); }

      // ---- factories (close over get/set/bus) ----------------------------

      // labelled slider with a live value readout + number echo not needed here;
      // SwarmUI uses sliders for cfg/shift/early-cut and number boxes for sigmas.
      function slider(opts) {
        // opts: { path, label, min, max, step, def, dec, fmt(v) }
        var cur = get(opts.path); if (cur == null) cur = opts.def;
        var valSpan = el("span", { class: "as-val", text: render(cur) });
        var lbl = el("label", { class: "as-lbl" }, [opts.label, valSpan]);
        var rng = el("input", { type: "range", min: opts.min, max: opts.max, step: opts.step });
        rng.value = cur;
        rng.addEventListener("input", function () {
          var v = parseStep(rng.value, opts.step);
          set(opts.path, v); valSpan.textContent = render(v);
        });
        bus.on("change:" + opts.path, function (v) {
          if (v == null) return;
          if (document.activeElement !== rng) rng.value = v;
          valSpan.textContent = render(v);
        });
        function render(v) { return opts.fmt ? opts.fmt(v) : fmt(v, opts); }
        return el("div", { class: "as-section" }, [lbl, rng]);
      }

      // number input bound to a state path
      function numField(path, o) {
        var cur = get(path); if (cur == null) cur = o.def;
        var inp = el("input", { type: "number", class: "as-num", min: o.min, max: o.max, step: o.step });
        inp.value = cur;
        inp.addEventListener("change", function () {
          var v = parseStep(inp.value, o.step);
          if (isNaN(v)) v = o.def;
          if (o.min != null && v < o.min) v = o.min;
          if (o.max != null && v > o.max) v = o.max;
          set(path, v); inp.value = v;
        });
        bus.on("change:" + path, function (v) {
          if (document.activeElement !== inp && v != null) inp.value = v;
        });
        return inp;
      }

      // checkbox bound to a boolean state path
      function checkField(path, def, label) {
        var cur = get(path); if (cur == null) cur = def;
        var inp = el("input", { type: "checkbox" });
        inp.checked = !!cur;
        inp.addEventListener("change", function () { set(path, inp.checked); });
        bus.on("change:" + path, function (v) { if (document.activeElement !== inp) inp.checked = !!v; });
        return el("label", { class: "as-check" }, [inp, el("span", { text: label })]);
      }

      // ---- build controls -------------------------------------------------

      // CFG (shared with param_rail's CFG slider — kept in sync via bus)
      var cfgSlider = slider({
        path: "params.cfg", label: "CFG Scale", min: 0, max: 30, step: 0.1, def: 1.5, dec: 1,
      });

      // Sigma Shift (rectified-flow / DiT timestep shift). 0 disables.
      var shiftSlider = slider({
        path: "params.sigma_shift", label: "Sigma Shift", min: 0, max: 12, step: 0.05, def: 3.0,
        fmt: function (v) { return (+v).toFixed(2); },
      });

      // End Steps Early — shown as a % but stored as a 0..1 fraction.
      var earlyCur = get("params.end_steps_early_pct");
      if (earlyCur == null) earlyCur = 0.0;
      var earlyVal = el("span", { class: "as-val", text: Math.round(earlyCur * 100) + "%" });
      var earlyRng = el("input", { type: "range", min: 0, max: 90, step: 1 });
      earlyRng.value = Math.round(earlyCur * 100);
      earlyRng.addEventListener("input", function () {
        var pct = parseInt(earlyRng.value, 10) || 0;
        earlyVal.textContent = pct + "%";
        set("params.end_steps_early_pct", +(pct / 100).toFixed(4));
      });
      bus.on("change:params.end_steps_early_pct", function (v) {
        if (v == null) return;
        var pct = Math.round(v * 100);
        if (document.activeElement !== earlyRng) earlyRng.value = pct;
        earlyVal.textContent = pct + "%";
      });
      var earlyBlock = el("div", { class: "as-section" }, [
        el("label", { class: "as-lbl" }, ["End Steps Early", earlyVal]),
        earlyRng,
      ]);

      // Sigma min/max + eta + clip skip (number boxes; -1 = model default)
      var clipSkip = numField("params.clip_skip", { min: 0, max: 12, step: 1, def: 0 });
      var etaIn   = numField("params.eta",        { min: -1, max: 1,   step: 0.01, def: -1 });
      var sigMin  = numField("params.sigma_min",  { min: -1, max: 10,  step: 0.01, def: -1 });
      var sigMax  = numField("params.sigma_max",  { min: -1, max: 100, step: 0.1,  def: -1 });

      // Batch Size (per-GPU batch; distinct from the "Images" count in param_rail)
      var batchSize = numField("params.batch_size", { min: 1, max: 16, step: 1, def: 1 });

      // boolean toggles
      var restartChk = checkField("params.restart_sampling", false, "Restart sampling");
      var noSeedIncChk = checkField("params.no_seed_increment", false, "No seed increment (batch reuses seed)");

      // ---- assemble the collapsible group ---------------------------------
      var panel = el("details", { id: "as-panel", class: "group as-section", open: "" }, [
        el("summary", { text: "Advanced Sampling" }),
        el("div", { class: "as-body" }, [
          cfgSlider,
          shiftSlider,
          earlyBlock,
          el("div", { class: "as-hint", text: "Sigma / eta / CLIP skip — −1 leaves the model default." }),
          el("div", { class: "as-grid2" }, [
            el("div", { class: "as-field" }, [el("label", { class: "as-lbl", text: "CLIP skip" }), clipSkip]),
            el("div", { class: "as-field" }, [el("label", { class: "as-lbl", text: "Eta" }), etaIn]),
          ]),
          el("div", { class: "as-grid2" }, [
            el("div", { class: "as-field" }, [el("label", { class: "as-lbl", text: "Sigma min" }), sigMin]),
            el("div", { class: "as-field" }, [el("label", { class: "as-lbl", text: "Sigma max" }), sigMax]),
          ]),
          el("div", { class: "as-grid2", style: "margin-top:4px" }, [
            el("div", { class: "as-field" }, [el("label", { class: "as-lbl", text: "Batch size" }), batchSize]),
            el("div", { class: "as-field" }, []),
          ]),
          restartChk,
          noSeedIncChk,
        ]),
      ]);

      root.appendChild(panel);
      console.info("[advancedSampling] ready");
    },
  });
})();
