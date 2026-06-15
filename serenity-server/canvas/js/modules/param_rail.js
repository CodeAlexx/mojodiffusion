/* param_rail.js — module 'paramRail'. SwarmUI-style LEFT param rail (Design B).
   ONE screen; this rail is ALWAYS on the left. "simple/advanced" is NOT a separate
   screen — it is purely which controls are shown: an [adv] toggle reveals deeper
   controls IN PLACE (clip_skip / sigma_min / sigma_max / eta / restart_sampling).
   Every control is bound to state.params.* via set(). Degrades gracefully when the
   ComfyUI API (objectInfo / loras) 404s — falls back to sane built-in lists.
   Owns ONLY this file. Talks to the rest of the app via state/get/set/bus/api. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  var ROOT_ID = "param-rail"; // dom.paramRail element id (see index.html / main.js)

  // ---- aspect presets (label -> [w,h] at a 1MP-ish budget, multiples of 64) ----
  var ASPECTS = [
    { k: "1:1", w: 1024, h: 1024 },
    { k: "3:2", w: 1216, h: 832 },
    { k: "2:3", w: 832, h: 1216 },
    { k: "4:3", w: 1152, h: 896 },
    { k: "3:4", w: 896, h: 1152 },
    { k: "16:9", w: 1344, h: 768 },
    { k: "9:16", w: 768, h: 1344 },
    { k: "21:9", w: 1536, h: 640 },
    { k: "custom", w: 0, h: 0 },
  ];

  // built-in fallbacks so the rail is fully usable before the backend ships
  var FALLBACK_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "uni_pc", "lcm",
  ];
  var FALLBACK_SCHEDULERS = [
    "simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform", "beta",
  ];

  // -------------------------------------------------------------------------
  function injectCSS() {
    if (document.getElementById("style-paramRail")) return;
    var css = [
      "#" + ROOT_ID + " .pr-section{margin-bottom:10px}",
      "#" + ROOT_ID + " .pr-lbl{display:block;color:var(--muted);font-size:11px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.04em}",
      "#" + ROOT_ID + " .pr-lbl .pr-val{float:right;color:var(--text);text-transform:none;letter-spacing:0;font-weight:600}",
      "#" + ROOT_ID + " textarea.pr-prompt{min-height:74px;font-size:13px;line-height:1.45}",
      "#" + ROOT_ID + " textarea.pr-neg{min-height:48px}",
      "#" + ROOT_ID + " .pr-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}",
      "#" + ROOT_ID + " .pr-grid2 .row{margin:0}",
      "#" + ROOT_ID + " .pr-field input,#" + ROOT_ID + " .pr-field select{width:100%}",
      "#" + ROOT_ID + " .pr-field{flex:1;min-width:0}",
      "#" + ROOT_ID + " .group>div.pr-body{padding:0 8px 8px}",
      "#" + ROOT_ID + " .pr-adv{display:none}",
      "#" + ROOT_ID + " .pr-adv-on .pr-adv{display:block}",
      "#" + ROOT_ID + " .pr-adv-on .pr-adv.row{display:flex}",
      "#" + ROOT_ID + " .pr-advbtn{margin-left:auto;font-size:10px;padding:2px 7px;border-radius:5px;" +
        "border:1px solid var(--line);background:var(--panel2);color:var(--muted);cursor:pointer;text-transform:uppercase;letter-spacing:.05em}",
      "#" + ROOT_ID + " .pr-advbtn.on{background:var(--accent2);border-color:var(--accent);color:#fff}",
      "#" + ROOT_ID + " .pr-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}",
      "#" + ROOT_ID + " .pr-head h2{font-size:13px;margin:0;font-weight:700;letter-spacing:.02em}",
      "#" + ROOT_ID + " .pr-icon{padding:5px 9px;min-width:0;line-height:1}",
      "#" + ROOT_ID + " .pr-seedwrap{display:flex;gap:6px;align-items:center}",
      "#" + ROOT_ID + " .pr-seedwrap input{flex:1;min-width:0}",
      "#" + ROOT_ID + " .pr-num{width:100%}",
      "#" + ROOT_ID + " .pr-loras{display:flex;flex-direction:column;gap:6px}",
      "#" + ROOT_ID + " .pr-lora{display:flex;gap:6px;align-items:center}",
      "#" + ROOT_ID + " .pr-lora select{flex:1;min-width:0}",
      "#" + ROOT_ID + " .pr-lora input{width:58px}",
      "#" + ROOT_ID + " .pr-lora .pr-rm{padding:4px 8px}",
      "#" + ROOT_ID + " .pr-add{font-size:11px;padding:4px 8px;align-self:flex-start}",
      "#" + ROOT_ID + " .pr-hint{color:var(--muted);font-size:11px;margin:2px 0}",
      "#" + ROOT_ID + " .pr-gen{margin-top:12px}",
      "#" + ROOT_ID + " #btn-generate{width:100%;padding:11px;font-size:14px}",
      "#" + ROOT_ID + " .group[open]>summary{border-bottom:1px solid var(--line)}",
      "#" + ROOT_ID + " input[type=number]{-moz-appearance:textfield}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-paramRail";
    st.textContent = css;
    document.head.appendChild(st);
  }

  // small DOM helper
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
    var list = opts && opts.length ? opts : [];
    if (!list.length) {
      sel.appendChild(el("option", { value: "", text: "(none)" }));
      return;
    }
    list.forEach(function (o) {
      var val = typeof o === "string" ? o : (o.value != null ? o.value : o.k);
      var lab = typeof o === "string" ? o : (o.label != null ? o.label : val);
      sel.appendChild(el("option", { value: val, text: lab }));
    });
    if (current != null && list.some(function (o) {
      var v = typeof o === "string" ? o : (o.value != null ? o.value : o.k);
      return String(v) === String(current);
    })) sel.value = current;
  }

  // -------------------------------------------------------------------------
  S.register("paramRail", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api;
      var root = (ctx.dom && ctx.dom.paramRail) || document.getElementById(ROOT_ID);
      if (!root) { console.warn("[paramRail] mount not found:", ROOT_ID); return; }
      root.innerHTML = "";
      if (get("ui.advanced")) root.classList.add("pr-adv-on");

      // ---- header + [adv] toggle ----
      var advBtn = el("button", {
        class: "pr-advbtn" + (get("ui.advanced") ? " on" : ""),
        title: "Reveal deeper controls in place (not a separate screen)",
        type: "button",
        text: "adv",
      });
      advBtn.addEventListener("click", function () {
        var next = !get("ui.advanced");
        set("ui.advanced", next);
      });
      root.appendChild(el("div", { class: "pr-head" }, [
        el("h2", { text: "Generate" }), advBtn,
      ]));

      // keep [adv] state in sync if another module toggles it
      bus.on("change:ui.advanced", function (v) {
        root.classList.toggle("pr-adv-on", !!v);
        advBtn.classList.toggle("on", !!v);
      });

      // ===================== PROMPTS =====================
      var prompt = el("textarea", {
        class: "pr-prompt", placeholder: "Describe what to generate…", rows: 3,
      });
      prompt.value = get("params.prompt") || "";
      prompt.addEventListener("input", function () { set("params.prompt", prompt.value); });

      var neg = el("textarea", {
        class: "pr-neg", placeholder: "What to avoid…", rows: 2,
      });
      neg.value = get("params.negative") || "";
      neg.addEventListener("input", function () { set("params.negative", neg.value); });

      var negGroup = el("details", { class: "group pr-section" }, [
        el("summary", { text: "Negative prompt" }),
        el("div", { class: "pr-body" }, [neg]),
      ]);

      root.appendChild(el("div", { class: "pr-section" }, [
        el("label", { class: "pr-lbl", text: "Prompt" }), prompt,
      ]));
      root.appendChild(negGroup);

      // ===================== ASPECT + RESOLUTION =====================
      var aspectSel = el("select");
      fillSelect(aspectSel, ASPECTS, get("params.aspect") || "1:1");

      var wIn = el("input", { type: "number", class: "pr-num", min: 64, max: 8192, step: 8 });
      var hIn = el("input", { type: "number", class: "pr-num", min: 64, max: 8192, step: 8 });
      wIn.value = get("params.width") || 1024;
      hIn.value = get("params.height") || 1024;

      function applyAspect(k) {
        var a = ASPECTS.filter(function (x) { return x.k === k; })[0];
        if (a && a.w > 0) {
          set("params.width", a.w); set("params.height", a.h);
          wIn.value = a.w; hIn.value = a.h;
        }
      }
      aspectSel.addEventListener("change", function () {
        set("params.aspect", aspectSel.value);
        applyAspect(aspectSel.value);
      });
      function dimsChanged() {
        // manual W/H edit => mark aspect custom (don't fight the user)
        set("params.aspect", "custom");
        if (aspectSel.value !== "custom") aspectSel.value = "custom";
      }
      wIn.addEventListener("change", function () {
        var v = clampInt(wIn.value, 64, 8192, 1024); wIn.value = v; set("params.width", v); dimsChanged();
      });
      hIn.addEventListener("change", function () {
        var v = clampInt(hIn.value, 64, 8192, 1024); hIn.value = v; set("params.height", v); dimsChanged();
      });

      root.appendChild(el("div", { class: "pr-section" }, [
        el("label", { class: "pr-lbl", text: "Aspect" }), aspectSel,
        el("div", { class: "pr-grid2", style: "margin-top:8px" }, [
          el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Width" }), wIn]),
          el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Height" }), hIn]),
        ]),
      ]));

      // keep W/H + aspect reactive to two-way bbox binding (bbox module writes width/height)
      bus.on("change:params.width", function (v) { if (document.activeElement !== wIn && v != null) wIn.value = v; });
      bus.on("change:params.height", function (v) { if (document.activeElement !== hIn && v != null) hIn.value = v; });
      bus.on("change:params.aspect", function (v) { if (v != null && aspectSel.value !== v) aspectSel.value = v; });

      // ===================== STEPS / CFG (sliders + number) =====================
      function slider(opts) {
        // opts: { path, label, min, max, step, def, fmt }
        var cur = get(opts.path); if (cur == null) cur = opts.def;
        var valSpan = el("span", { class: "pr-val", text: fmtNum(cur, opts) });
        var lbl = el("label", { class: "pr-lbl" }, [opts.label, valSpan]);
        var rng = el("input", { type: "range", min: opts.min, max: opts.max, step: opts.step });
        rng.value = cur;
        rng.addEventListener("input", function () {
          var v = numParse(rng.value, opts.step);
          set(opts.path, v); valSpan.textContent = fmtNum(v, opts);
        });
        bus.on("change:" + opts.path, function (v) {
          if (v == null) return;
          if (document.activeElement !== rng) rng.value = v;
          valSpan.textContent = fmtNum(v, opts);
        });
        return el("div", { class: "pr-section" }, [lbl, rng]);
      }

      root.appendChild(slider({ path: "params.steps", label: "Steps", min: 1, max: 100, step: 1, def: 8 }));
      root.appendChild(slider({ path: "params.cfg", label: "CFG", min: 0, max: 30, step: 0.1, def: 1.5, dec: 1 }));

      // ===================== SEED (+ random / lock) =====================
      var seedLocked = false;
      var seedIn = el("input", { type: "number", step: 1, min: -1 });
      seedIn.value = get("params.seed");
      var randBtn = el("button", { class: "btn pr-icon", type: "button", title: "Randomize seed", html: "🎲" });
      var lockBtn = el("button", { class: "btn pr-icon", type: "button", title: "Lock seed (-1 randomizes per run)", html: "🔓" });
      seedIn.addEventListener("change", function () {
        var v = parseInt(seedIn.value, 10); if (isNaN(v)) v = -1;
        set("params.seed", v);
      });
      randBtn.addEventListener("click", function () {
        if (seedLocked) return;
        var v = Math.floor(Math.random() * 4294967295);
        seedIn.value = v; set("params.seed", v);
      });
      lockBtn.addEventListener("click", function () {
        seedLocked = !seedLocked;
        lockBtn.innerHTML = seedLocked ? "🔒" : "🔓";
        lockBtn.classList.toggle("btn-primary", seedLocked);
        seedIn.disabled = seedLocked;
        randBtn.disabled = seedLocked;
      });
      bus.on("change:params.seed", function (v) { if (document.activeElement !== seedIn && v != null) seedIn.value = v; });

      root.appendChild(el("div", { class: "pr-section" }, [
        el("label", { class: "pr-lbl", text: "Seed (-1 = random)" }),
        el("div", { class: "pr-seedwrap" }, [seedIn, randBtn, lockBtn]),
      ]));

      // ===================== IMAGES (batch count) =====================
      root.appendChild(slider({ path: "params.images", label: "Images", min: 1, max: 16, step: 1, def: 1 }));

      // ===================== GROUP: Sampling =====================
      var samplerSel = el("select");
      var schedSel = el("select");
      fillSelect(samplerSel, FALLBACK_SAMPLERS, get("params.sampler") || "euler");
      fillSelect(schedSel, FALLBACK_SCHEDULERS, get("params.scheduler") || "simple");
      samplerSel.addEventListener("change", function () { set("params.sampler", samplerSel.value); });
      schedSel.addEventListener("change", function () { set("params.scheduler", schedSel.value); });
      bus.on("change:params.sampler", function (v) { if (v != null && samplerSel.value !== v) samplerSel.value = v; });
      bus.on("change:params.scheduler", function (v) { if (v != null && schedSel.value !== v) schedSel.value = v; });

      // advanced-in-place fields revealed by [adv]: clip_skip / sigma_min / sigma_max / eta / restart
      var clipSkip = numField("params.clip_skip", { min: 0, max: 12, step: 1, def: 0 });
      var sigMin = numField("params.sigma_min", { min: -1, max: 10, step: 0.01, def: -1 });
      var sigMax = numField("params.sigma_max", { min: -1, max: 100, step: 0.1, def: -1 });
      var etaIn = numField("params.eta", { min: -1, max: 1, step: 0.01, def: -1 });
      var restart = checkField("params.restart_sampling", false);

      var samplingGroup = el("details", { class: "group pr-section", open: "" }, [
        el("summary", { text: "Sampling" }),
        el("div", { class: "pr-body" }, [
          el("div", { class: "pr-grid2" }, [
            el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Sampler" }), samplerSel]),
            el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Scheduler" }), schedSel]),
          ]),
          // --- advanced (revealed in place) ---
          el("div", { class: "pr-adv" }, [
            el("div", { class: "pr-grid2" }, [
              el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "CLIP skip" }), clipSkip]),
              el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Eta" }), etaIn]),
            ]),
            el("div", { class: "pr-grid2" }, [
              el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Sigma min" }), sigMin]),
              el("div", { class: "pr-field" }, [el("label", { class: "pr-lbl", text: "Sigma max" }), sigMax]),
            ]),
            el("div", { class: "row" }, [restart, el("label", { text: "Restart sampling", style: "min-width:0;color:var(--text)" })]),
            el("div", { class: "pr-hint", text: "−1 = model default" }),
          ]),
        ]),
      ]);
      root.appendChild(samplingGroup);

      // ===================== GROUP: Init image / Denoise =====================
      var denoise = slider({ path: "params.denoise", label: "Denoise", min: 0, max: 1, step: 0.01, def: 1.0, dec: 2 });
      var initGroup = el("details", { class: "group pr-section" }, [
        el("summary", { text: "Init / Denoise" }),
        el("div", { class: "pr-body" }, [
          el("div", { class: "pr-hint", text: "Use a canvas layer as init image; denoise controls how much it changes." }),
          denoise,
        ]),
      ]);
      root.appendChild(initGroup);

      // ===================== GROUP: ControlNet =====================
      var cnEnable = checkField("__cn_enable", !!get("params.controlnet"));
      var cnStrength = el("input", { type: "range", min: 0, max: 2, step: 0.05 });
      var cnStr0 = (get("params.controlnet") && get("params.controlnet").strength) || 1.0;
      cnStrength.value = cnStr0;
      var cnStrVal = el("span", { class: "pr-val", text: (+cnStr0).toFixed(2) });
      function writeCN() {
        if (cnEnable.checked) {
          set("params.controlnet", { enabled: true, strength: numParse(cnStrength.value, 0.05) });
        } else {
          set("params.controlnet", null);
        }
      }
      cnEnable.addEventListener("change", writeCN);
      cnStrength.addEventListener("input", function () { cnStrVal.textContent = (+cnStrength.value).toFixed(2); writeCN(); });
      var cnGroup = el("details", { class: "group pr-section" }, [
        el("summary", { text: "ControlNet" }),
        el("div", { class: "pr-body" }, [
          el("div", { class: "row" }, [cnEnable, el("label", { text: "Enable ControlNet", style: "min-width:0;color:var(--text)" })]),
          el("label", { class: "pr-lbl" }, ["Strength", cnStrVal]),
          cnStrength,
          el("div", { class: "pr-hint", text: "Add a control layer (right panel) and pick a preprocessor (⚙)." }),
        ]),
      ]);
      root.appendChild(cnGroup);

      // ===================== GROUP: LoRAs =====================
      var loraNames = ["(none)"]; // populated from api.loras()
      var loraListEl = el("div", { class: "pr-loras" });
      var addLoraBtn = el("button", { class: "btn pr-add", type: "button", text: "+ Add LoRA" });

      function readLoras() {
        var arr = [];
        loraListEl.querySelectorAll(".pr-lora").forEach(function (rowEl) {
          var name = rowEl.querySelector("select").value;
          var w = parseFloat(rowEl.querySelector("input").value);
          if (name && name !== "(none)") arr.push({ name: name, weight: isNaN(w) ? 1.0 : w });
        });
        set("params.loras", arr);
      }
      function addLoraRow(preset) {
        var sel = el("select");
        fillSelect(sel, loraNames, preset && preset.name);
        var w = el("input", { type: "number", step: 0.05, min: -2, max: 2, value: (preset && preset.weight != null) ? preset.weight : 1.0 });
        var rm = el("button", { class: "btn pr-rm", type: "button", html: "✕", title: "Remove" });
        var rowEl = el("div", { class: "pr-lora" }, [sel, w, rm]);
        sel.addEventListener("change", readLoras);
        w.addEventListener("change", readLoras);
        rm.addEventListener("click", function () { rowEl.remove(); readLoras(); });
        loraListEl.appendChild(rowEl);
        return rowEl;
      }
      addLoraBtn.addEventListener("click", function () { addLoraRow(); });

      // seed rows from any pre-existing state.params.loras
      var existingLoras = get("params.loras") || [];
      if (existingLoras.length) existingLoras.forEach(addLoraRow);

      var loraGroup = el("details", { class: "group pr-section" }, [
        el("summary", { text: "LoRAs" }),
        el("div", { class: "pr-body" }, [
          loraListEl, addLoraBtn,
          el("div", { class: "pr-hint pr-loras-empty", text: "No LoRAs loaded — backend list unavailable." }),
        ]),
      ]);
      root.appendChild(loraGroup);

      // ===================== GROUP: Refiner =====================
      var refEnable = checkField("__ref_enable", !!get("params.refiner"));
      var refModel = el("select");
      fillSelect(refModel, ["(none)"], null);
      var refSwitch = el("input", { type: "range", min: 0, max: 1, step: 0.05, value: (get("params.refiner") && get("params.refiner").switch_at) || 0.8 });
      var refSwVal = el("span", { class: "pr-val", text: (+refSwitch.value).toFixed(2) });
      function writeRefiner() {
        if (refEnable.checked) {
          set("params.refiner", {
            enabled: true,
            model: refModel.value && refModel.value !== "(none)" ? refModel.value : "",
            switch_at: numParse(refSwitch.value, 0.05),
          });
        } else set("params.refiner", null);
      }
      refEnable.addEventListener("change", writeRefiner);
      refModel.addEventListener("change", writeRefiner);
      refSwitch.addEventListener("input", function () { refSwVal.textContent = (+refSwitch.value).toFixed(2); writeRefiner(); });
      var refGroup = el("details", { class: "group pr-section" }, [
        el("summary", { text: "Refiner" }),
        el("div", { class: "pr-body" }, [
          el("div", { class: "row" }, [refEnable, el("label", { text: "Enable refiner", style: "min-width:0;color:var(--text)" })]),
          el("label", { class: "pr-lbl", text: "Refiner model" }), refModel,
          el("label", { class: "pr-lbl", style: "margin-top:8px" }, ["Switch at", refSwVal]),
          refSwitch,
        ]),
      ]);
      root.appendChild(refGroup);

      // ===================== GENERATE button (rendered here; generateWS binds behavior) =====================
      var genBtn = el("button", { id: "btn-generate", class: "btn btn-primary", type: "button", text: "Generate" });
      root.appendChild(el("div", { class: "pr-section pr-gen" }, [genBtn]));

      // -----------------------------------------------------------------
      // Populate sampler/scheduler/lora/refiner-model lists from the API.
      // 404 / network failure => keep fallbacks; never throw past here.
      // -----------------------------------------------------------------
      loadSamplerInfo();
      loadLoras();
      loadRefinerModels();

      function loadSamplerInfo() {
        if (!api || typeof api.objectInfo !== "function") return;
        Promise.resolve()
          .then(function () { return api.objectInfo("KSampler"); })
          .then(function (info) {
            var samplers = extractCombo(info, "KSampler", "sampler_name");
            var scheds = extractCombo(info, "KSampler", "scheduler");
            if (samplers && samplers.length) fillSelect(samplerSel, samplers, get("params.sampler") || samplers[0]);
            if (scheds && scheds.length) fillSelect(schedSel, scheds, get("params.scheduler") || scheds[0]);
          })
          .catch(function () { /* backend not up — keep fallbacks */ });
      }

      function loadLoras() {
        if (!api || typeof api.loras !== "function") return;
        Promise.resolve().then(function () { return api.loras(); })
          .then(function (list) {
            var names = normalizeList(list);
            if (names && names.length) {
              loraNames = ["(none)"].concat(names);
              // refresh option lists in existing rows (preserve current selection)
              loraListEl.querySelectorAll(".pr-lora select").forEach(function (sel) {
                var cur = sel.value; fillSelect(sel, loraNames, cur);
              });
              var emptyHint = loraGroup.querySelector(".pr-loras-empty");
              if (emptyHint) emptyHint.textContent = names.length + " LoRA(s) available.";
            }
          })
          .catch(function () { /* keep (none) */ });
      }

      function loadRefinerModels() {
        if (!api || typeof api.models !== "function") return;
        Promise.resolve().then(function () { return api.models(); })
          .then(function (list) {
            var names = normalizeList(list);
            if (names && names.length) {
              var cur = (get("params.refiner") && get("params.refiner").model) || null;
              fillSelect(refModel, ["(none)"].concat(names), cur);
            }
          })
          .catch(function () { /* keep (none) */ });
      }

      console.info("[paramRail] ready");

      // ============ tiny field factories (use closures over get/set/bus) ============
      function numField(path, o) {
        var cur = get(path); if (cur == null) cur = o.def;
        var inp = el("input", { type: "number", class: "pr-num", min: o.min, max: o.max, step: o.step });
        inp.value = cur;
        inp.addEventListener("change", function () {
          var v = numParse(inp.value, o.step);
          if (isNaN(v)) v = o.def;
          set(path, v); inp.value = v;
        });
        bus.on("change:" + path, function (v) { if (document.activeElement !== inp && v != null) inp.value = v; });
        return inp;
      }
      function checkField(path, def) {
        var isStateBacked = path.indexOf("__") !== 0;
        var cur = isStateBacked ? get(path) : def;
        if (cur == null) cur = def;
        var inp = el("input", { type: "checkbox" });
        inp.checked = !!cur;
        if (isStateBacked) {
          inp.addEventListener("change", function () { set(path, inp.checked); });
          bus.on("change:" + path, function (v) { inp.checked = !!v; });
        }
        return inp;
      }
    },
  });

  // ---- stateless numeric helpers (module scope) ----
  function clampInt(v, lo, hi, def) {
    var n = parseInt(v, 10);
    if (isNaN(n)) n = def;
    n = Math.max(lo, Math.min(hi, n));
    return Math.round(n / 8) * 8 || lo;
  }
  function numParse(v, step) {
    var n = parseFloat(v);
    if (isNaN(n)) return 0;
    if (step != null && step < 1) {
      var dec = (String(step).split(".")[1] || "").length;
      return +n.toFixed(dec);
    }
    return Math.round(n);
  }
  function fmtNum(v, opts) {
    if (v == null) return "";
    if (opts && opts.dec != null) return (+v).toFixed(opts.dec);
    if (opts && opts.step != null && opts.step < 1) {
      var dec = (String(opts.step).split(".")[1] || "").length;
      return (+v).toFixed(dec);
    }
    return String(v);
  }

  // ---- ComfyUI objectInfo extraction (defensive against shape) ----
  // info can be the whole map { KSampler:{input:{required:{sampler_name:[[...]]}}} }
  // or just the KSampler entry. Combo lists are [[a,b,c], {...}] in ComfyUI.
  function extractCombo(info, cls, field) {
    if (!info || typeof info !== "object") return null;
    var node = info[cls] || info; // either the full map or the single-class object
    var req = node && node.input && node.input.required;
    if (!req || !req[field]) return null;
    var spec = req[field];
    // spec is typically [ ["euler", ...], {default:...} ]
    if (Array.isArray(spec) && Array.isArray(spec[0])) return spec[0];
    if (Array.isArray(spec)) return spec.filter(function (x) { return typeof x === "string"; });
    return null;
  }

  // ComfyUI /models/loras returns a flat array of strings; tolerate {loras:[]} too.
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
})();
