/* refiner_upscale.js — module 'refinerUpscale'. SwarmUI "Refine / Upscale" group.
   This panel authors workflow intent. Generate turns active settings into a
   SerenityRefinerUpscaleIntent graph node; Rust workflow lowering/preflight
   decides whether that graph is production-admitted. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  var ROOT_ID = "param-rail";

  // refiner control method retained for workflow admission.
  var METHODS = [
    { v: "postapply", label: "PostApply (refine after base)" },
    { v: "stepswap", label: "StepSwap (swap model mid-sample)" },
  ];

  // upscale factor presets (SwarmUI lets you pick a multiplier; backend hires_scale
  // is a float >1.0 enabling the 2nd pass; 1.0 = off).
  var UPSCALE_FACTORS = [
    { v: 1.0, label: "Off (1×)" },
    { v: 1.5, label: "1.5×" },
    { v: 2.0, label: "2×" },
    { v: 3.0, label: "3×" },
    { v: 4.0, label: "4×" },
  ];

  function injectCSS() {
    if (document.getElementById("style-refinerUpscale")) return;
    var p = "#" + ROOT_ID + " ";
    var css = [
      p + ".ru-body{padding:0 8px 8px}",
      p + ".ru-lbl{display:block;color:var(--muted);font-size:11px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.04em}",
      p + ".ru-lbl .ru-val{float:right;color:var(--text);text-transform:none;letter-spacing:0;font-weight:600;font-variant-numeric:tabular-nums}",
      p + ".ru-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}",
      p + ".ru-field{min-width:0}",
      p + ".ru-field input,#" + ROOT_ID + " .ru-field select{width:100%}",
      p + ".ru-row{display:flex;align-items:center;gap:8px;margin:8px 0}",
      p + ".ru-row label{min-width:0;color:var(--text);cursor:pointer}",
      p + ".ru-sub{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}",
      p + ".ru-sub-h{font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;margin:0 0 4px}",
      p + ".ru-hint{color:var(--muted);font-size:11px;margin:4px 0 0}",
      p + ".ru-badge{display:inline-block;margin-left:6px;padding:0 5px;border-radius:4px;font-size:9px;font-weight:700;line-height:1.5;vertical-align:middle;letter-spacing:.04em}",
      p + ".ru-badge.wired{border:1px solid var(--ok);color:var(--ok)}",
      p + ".ru-badge.disp{border:1px solid var(--warn);color:var(--warn)}",
      p + ".ru-disabled{opacity:.5;pointer-events:none}",
      p + "input[type=range].ru-range{flex:1;accent-color:var(--accent)}",
      p + "input[type=number].ru-num{-moz-appearance:textfield}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-refinerUpscale";
    st.textContent = css;
    document.head.appendChild(st);
  }

  // tiny DOM helper (same idiom as param_rail.el)
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

  function fillSelect(sel, opts, current) {
    sel.innerHTML = "";
    (opts || []).forEach(function (o) {
      var val = typeof o === "string" ? o : (o.v != null ? o.v : o.value);
      var lab = typeof o === "string" ? o : (o.label != null ? o.label : val);
      sel.appendChild(el("option", { value: val, text: lab }));
    });
    if (current != null) {
      var has = (opts || []).some(function (o) {
        var v = typeof o === "string" ? o : (o.v != null ? o.v : o.value);
        return String(v) === String(current);
      });
      if (has) sel.value = current;
    }
  }

  // ComfyUI /models style list -> flat string names (tolerant of shapes)
  function normalizeList(list) {
    if (!list) return [];
    if (Array.isArray(list)) {
      return list.map(function (x) { return typeof x === "string" ? x : (x && (x.name || x.filename || x.title)); })
        .filter(Boolean);
    }
    if (typeof list === "object") {
      var arr = list.loras || list.models || list.items || list.data;
      if (Array.isArray(arr)) return normalizeList(arr);
    }
    return [];
  }

  function badge(kind, text) {
    return el("span", { class: "ru-badge " + (kind === "wired" ? "wired" : "disp"), text: text });
  }

  S.register("refinerUpscale", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api;
      var root = (ctx.dom && ctx.dom.paramRail) || document.getElementById(ROOT_ID);
      if (!root) { console.warn("[refinerUpscale] mount not found:", ROOT_ID); return; }

      // ---- read any pre-existing structured state so the disabled panel renders
      //      consistently before clearing stale state. ----
      var ref0 = get("params.refiner") || {};
      var up0 = get("params.upscaler") || {};

      // master enable for the refiner sub-section
      var refEnabled = ref0.enabled != null ? !!ref0.enabled
        : (get("params.hires_scale") != null && get("params.hires_scale") > 1.0);

      // ===================== REFINER SUB-SECTION =====================
      var refEnableChk = el("input", { type: "checkbox" });
      refEnableChk.checked = refEnabled;

      // refiner model (display-only; no separate-checkpoint worker support yet)
      var refModelSel = el("select");
      fillSelect(refModelSel, ["(same as base)"], ref0.model || null);

      // control % (WIRED -> hires_denoise). SwarmUI "Refiner Control Percentage":
      // higher = the refiner has more say (more denoise). We map 0..1 directly to
      // hires_denoise so the existing 2-pass path honors it.
      var ctlInit = ref0.control != null ? ref0.control
        : (get("params.hires_denoise") != null ? get("params.hires_denoise") : 0.4);
      var ctlRange = el("input", { type: "range", class: "ru-range", min: 0, max: 1, step: 0.01 });
      ctlRange.value = ctlInit;
      var ctlVal = el("span", { class: "ru-val", text: (+ctlInit).toFixed(2) });

      // method (display-only)
      var methodSel = el("select");
      fillSelect(methodSel, METHODS, ref0.method || "postapply");

      // refiner steps (display-only)
      var refStepsIn = el("input", { type: "number", class: "ru-num", min: 1, max: 100, step: 1 });
      refStepsIn.value = ref0.steps != null ? ref0.steps : (get("params.steps") || 8);

      // refiner CFG (display-only)
      var refCfgIn = el("input", { type: "number", class: "ru-num", min: 0, max: 30, step: 0.1 });
      refCfgIn.value = ref0.cfg != null ? ref0.cfg : (get("params.cfg") != null ? get("params.cfg") : 1.5);

      // refiner tiling (display-only)
      var tilingChk = el("input", { type: "checkbox" });
      tilingChk.checked = !!ref0.tiling;

      // ===================== UPSCALE SUB-SECTION =====================
      // upscaler model (display-only — backend uses Lanczos3 on the hires path)
      var upModelSel = el("select");
      fillSelect(upModelSel, ["Lanczos3 (built-in)"], up0.model || null);

      // upscale factor (WIRED -> hires_scale). >1.0 enables the 2nd refine pass.
      var factorInit = up0.factor != null ? up0.factor
        : (get("params.hires_scale") != null ? get("params.hires_scale") : (refEnabled ? 2.0 : 1.0));
      var factorSel = el("select");
      // include a custom slot if the saved factor isn't a preset
      var factorOpts = UPSCALE_FACTORS.slice();
      if (!factorOpts.some(function (o) { return Math.abs(o.v - factorInit) < 1e-6; })) {
        factorOpts.push({ v: factorInit, label: factorInit + "× (custom)" });
      }
      fillSelect(factorSel, factorOpts, factorInit);

      // ---- the wiring: a single writer that pushes BOTH flat fields (for api.js
      //      forwarding) AND the structured objects, then derives the real backend
      //      hires_scale / hires_denoise. -----------------------------------------
      function num(v, dflt) { var n = parseFloat(v); return isNaN(n) ? dflt : n; }
      function intNum(v, dflt) { var n = parseInt(v, 10); return isNaN(n) ? dflt : n; }

      // re-entrancy guard: when WE write hires_scale/hires_denoise via commit(), the
      // store emits change:* which our own listeners catch. Without this flag the
      // "off" path (hires_scale->1.0) would reset the user's factor dropdown, and the
      // denoise echo could fight the slider. The flag lets the listeners ignore our
      // own writes and react ONLY to external writes (presets / the basic param_rail
      // Refiner group).
      var selfWrite = false;

      function commit() {
        selfWrite = true;
        try { commitInner(); } finally { selfWrite = false; }
      }

      function commitInner() {
        var enabled = refEnableChk.checked;
        var control = num(ctlRange.value, 0.4);
        var factor = num(factorSel.value, 1.0);
        var method = methodSel.value || "postapply";
        var refModel = (refModelSel.value && refModelSel.value !== "(same as base)") ? refModelSel.value : "";
        var upModel = (upModelSel.value && upModelSel.value.indexOf("Lanczos3") !== 0) ? upModelSel.value : "";
        var steps = intNum(refStepsIn.value, 8);
        var cfg = num(refCfgIn.value, 1.5);
        var tiling = !!tilingChk.checked;

        // --- structured objects (mirror; supersets the basic param_rail Refiner) ---
        set("params.refiner", enabled ? {
          enabled: true, model: refModel, control: control, method: method,
          steps: steps, cfg: cfg, tiling: tiling,
        } : null);
        set("params.upscaler", (enabled && factor > 1.0) ? {
          model: upModel, factor: factor,
        } : null);

        // --- flat mirrors used by workflow graph assembly and preset restore ---
        set("params.refiner_model", enabled ? refModel : "");
        set("params.refiner_method", enabled ? method : "");
        set("params.refiner_steps", enabled ? steps : 0);
        set("params.refiner_cfg", enabled ? cfg : -1);
        set("params.refiner_control", enabled ? control : -1);
        set("params.refiner_tiling", enabled ? tiling : false);
        set("params.upscaler_model", (enabled && factor > 1.0) ? upModel : "");
        set("params.upscale_by", (enabled && factor > 1.0) ? factor : 1.0);

        // --- workflow intent fields ---
        // Generate reads these into SerenityRefinerUpscaleIntent. The Rust
        // capability gate rejects them until a real two-pass executor is admitted.
        if (enabled && factor > 1.0) {
          set("params.hires_scale", factor);
          set("params.hires_denoise", control);
        } else {
          set("params.hires_scale", 1.0);
          set("params.hires_denoise", control);
        }

        syncEnabled();
      }

      // gray out the inner controls when the refiner is disabled
      var refInner, upInner;
      function syncEnabled() {
        var on = refEnableChk.checked;
        if (refInner) refInner.classList.toggle("ru-disabled", !on);
        if (upInner) upInner.classList.toggle("ru-disabled", !on);
      }

      // live value readout for the control slider
      ctlRange.addEventListener("input", function () { ctlVal.textContent = (+ctlRange.value).toFixed(2); });

      // every input commits the whole panel (simple + robust)
      [refEnableChk, refModelSel, ctlRange, methodSel, refStepsIn, refCfgIn, tilingChk,
        upModelSel, factorSel].forEach(function (n) {
        n.addEventListener("change", commit);
      });

      // ===================== BUILD THE PANEL =====================
      refInner = el("div", {}, [
        // refiner model + control %
        el("label", { class: "ru-lbl" }, ["Refiner model", badge("wired", "workflow")]),
        refModelSel,
        el("label", { class: "ru-lbl" }, ["Control %", ctlVal, badge("wired", "workflow")]),
        ctlRange,
        el("div", { class: "ru-hint", text: "Workflow preflight decides whether this graph is admitted." }),
        // method
        el("label", { class: "ru-lbl" }, ["Method", badge("disp", "display-only")]),
        methodSel,
        // steps / cfg
        el("div", { class: "ru-grid2", style: "margin-top:8px" }, [
          el("div", { class: "ru-field" }, [
            el("label", { class: "ru-lbl" }, ["Refiner steps", badge("disp", "n/a")]), refStepsIn,
          ]),
          el("div", { class: "ru-field" }, [
            el("label", { class: "ru-lbl" }, ["Refiner CFG", badge("disp", "n/a")]), refCfgIn,
          ]),
        ]),
        // tiling
        el("div", { class: "ru-row" }, [
          tilingChk, el("label", {}, ["Refiner do-tiling", badge("disp", "display-only")]),
        ]),
      ]);

      upInner = el("div", { class: "ru-sub" }, [
        el("p", { class: "ru-sub-h", text: "Upscale" }),
        el("label", { class: "ru-lbl" }, ["Upscaler model", badge("wired", "workflow")]),
        upModelSel,
        el("label", { class: "ru-lbl" }, ["Upscale factor", badge("wired", "workflow")]),
        factorSel,
        el("div", { class: "ru-hint", text: "Unsupported two-pass graphs fail at Rust workflow preflight before enqueue." }),
      ]);

      var body = el("div", { class: "ru-body" }, [
        el("div", { class: "ru-row" }, [
          refEnableChk, el("label", { text: "Enable Refine / Upscale" }),
        ]),
        refInner,
        upInner,
      ]);

      var panel = el("details", { class: "group pr-section" }, [
        el("summary", { text: "Refine / Upscale" }),
        body,
      ]);
      if (refEnabled) panel.setAttribute("open", "");
      root.appendChild(panel);

      // ---- async-load model lists for the two dropdowns (graceful on 404) ----
      if (api && typeof api.models === "function") {
        Promise.resolve().then(function () { return api.models(); }).then(function (list) {
          var names = normalizeList(list);
          if (names && names.length) {
            var refCur = (get("params.refiner") && get("params.refiner").model) || ref0.model || null;
            fillSelect(refModelSel, ["(same as base)"].concat(names), refCur);
            var upCur = (get("params.upscaler") && get("params.upscaler").model) || up0.model || null;
            // upscaler list: keep built-in first, then any model names (server has no
            // dedicated upscaler endpoint yet, so we offer the same catalog as a stub).
            fillSelect(upModelSel, ["Lanczos3 (built-in)"].concat(names), upCur);
          }
        }).catch(function () { /* keep fallbacks */ });
      }

      // ---- react if another module (e.g. the param_rail basic Refiner group, or a
      //      preset reuse) writes hires_scale / hires_denoise out from under us ----
      bus.on("change:params.hires_scale", function (v) {
        if (selfWrite || v == null) return;          // ignore our own commit() writes
        if (Math.abs(num(factorSel.value, 1.0) - v) < 1e-6) return;
        var opts = UPSCALE_FACTORS.slice();
        if (!opts.some(function (o) { return Math.abs(o.v - v) < 1e-6; })) opts.push({ v: v, label: v + "× (custom)" });
        fillSelect(factorSel, opts, v);
        if (v > 1.0 && !refEnableChk.checked) { refEnableChk.checked = true; syncEnabled(); panel.setAttribute("open", ""); }
      });
      bus.on("change:params.hires_denoise", function (v) {
        if (selfWrite || v == null || document.activeElement === ctlRange) return;
        if (Math.abs(num(ctlRange.value, 0.4) - v) < 1e-6) return;
        ctlRange.value = v; ctlVal.textContent = (+v).toFixed(2);
      });

      // initial state push so the backend sees a coherent hires_scale/denoise even
      // if the user never touches the panel (off => hires_scale 1.0 => no 2nd pass).
      commit();

      console.info("[refinerUpscale] ready");
    },
  });
})();
