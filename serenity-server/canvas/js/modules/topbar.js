/* topbar.js — module "topbar". Owns the top bar:
   - model picker (api.models)            -> writes state.params.model
   - VAE override picker                   -> writes state.params.vae
   - live VRAM / util meter (poll systemStats ~3s)
   - light/dark theme toggle (data-theme on <html>)
   - settings affordance (opens a small popover)
   Design B: ONE screen, this is the full-width topbar. Talks ONLY via state/bus/api.
   Degrades gracefully when the API 404s (the ComfyUI-compat backend ships separately). */
(function () {
  "use strict";
  if (!window.Serenity || typeof window.Serenity.register !== "function") return;

  var ROOT_ID = "topbar";
  var STYLE_ID = "style-topbar";
  var POLL_MS = 3000;
  var THEME_KEY = "serenity.theme";

  // ---- scoped CSS (injected once; all rules under #topbar) ------------------
  function injectCSS() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = [
      "#topbar .tb-brand{display:flex;align-items:center;gap:8px;font-weight:700;letter-spacing:.3px;white-space:nowrap;user-select:none}",
      "#topbar .tb-dot{width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 0 6px var(--accent)}",
      "#topbar .tb-sep{width:1px;height:24px;background:var(--line);margin:0 2px}",
      "#topbar .tb-field{display:flex;align-items:center;gap:6px}",
      "#topbar .tb-field>label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px}",
      "#topbar select.tb-select{max-width:220px;min-width:120px;text-overflow:ellipsis;cursor:pointer}",
      "#topbar select.tb-select:disabled{opacity:.6;cursor:default}",
      "#topbar .tb-spacer{flex:1 1 auto}",
      "#topbar .tb-meter{display:flex;align-items:center;gap:10px;min-width:0}",
      "#topbar .tb-gauge{display:flex;flex-direction:column;gap:3px;min-width:120px}",
      "#topbar .tb-gauge .tb-glabel{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;line-height:1}",
      "#topbar .tb-gauge .tb-glabel b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}",
      "#topbar .tb-bar{height:6px;border-radius:4px;background:var(--panel2);border:1px solid var(--line);overflow:hidden}",
      "#topbar .tb-bar>i{display:block;height:100%;width:0;border-radius:4px;background:var(--ok);transition:width .4s ease,background .4s ease}",
      "#topbar .tb-bar.warn>i{background:var(--warn)}",
      "#topbar .tb-bar.crit>i{background:var(--danger)}",
      "#topbar .tb-gpu{color:var(--muted);font-size:10px;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
      "#topbar .tb-iconbtn{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;font-size:15px;line-height:1;padding:0}",
      "#topbar .tb-offline{color:var(--warn);font-size:11px;display:flex;align-items:center;gap:5px;white-space:nowrap}",
      "#topbar .tb-offline .tb-odot{width:8px;height:8px;border-radius:50%;background:var(--warn)}",
      "#topbar .tb-pop{position:absolute;top:46px;right:8px;z-index:50;background:var(--panel2);border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 8px 24px rgba(0,0,0,.4);padding:12px;min-width:230px}",
      "#topbar .tb-pop h4{margin:0 0 8px;font-size:12px;color:var(--text)}",
      "#topbar .tb-pop .row{margin:6px 0}",
      "#topbar .tb-pop label{min-width:auto;flex:1}",
      "#topbar .tb-pop small{display:block;color:var(--muted);margin-top:8px;line-height:1.3}"
    ].join("\n");
    document.head.appendChild(s);
  }

  // ---- helpers -------------------------------------------------------------
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  // Pull a list of names out of whatever shape the API hands back.
  // Accepts: ["a","b"], {checkpoints:[...]}, {models:[...]}, [{name},{title}], etc.
  function asNameList(data) {
    if (!data) return [];
    if (Array.isArray(data)) {
      return data.map(function (it) {
        if (typeof it === "string") return it;
        if (it && typeof it === "object") return it.name || it.title || it.filename || it.id || "";
        return "";
      }).filter(Boolean);
    }
    if (typeof data === "object") {
      // common containers
      var keys = ["checkpoints", "models", "ckpt", "items", "data"];
      for (var i = 0; i < keys.length; i++) {
        if (Array.isArray(data[keys[i]])) return asNameList(data[keys[i]]);
      }
      // map of name -> meta
      var vals = Object.keys(data);
      if (vals.length && vals.every(function (k) { return typeof data[k] !== "function"; })) return vals;
    }
    return [];
  }

  function fmtGiB(bytes) {
    if (bytes == null || isNaN(bytes)) return "?";
    return (bytes / (1024 * 1024 * 1024)).toFixed(1);
  }

  // ComfyUI /system_stats -> derive a VRAM gauge + util from the first device.
  function parseStats(stats) {
    var out = { vramPct: null, vramText: "VRAM —", utilPct: null, gpuName: "" };
    if (!stats || typeof stats !== "object") return out;
    var dev = null;
    if (Array.isArray(stats.devices) && stats.devices.length) {
      // prefer a cuda/gpu device over cpu
      dev = stats.devices.find(function (d) { return d && d.type && /cuda|gpu/i.test(d.type); }) || stats.devices[0];
    }
    if (dev) {
      out.gpuName = dev.name || dev.type || "";
      var total = dev.vram_total != null ? dev.vram_total : dev.total;
      var free = dev.vram_free != null ? dev.vram_free : dev.free;
      if (total != null && free != null && total > 0) {
        var used = total - free;
        out.vramPct = Math.max(0, Math.min(100, (used / total) * 100));
        out.vramText = fmtGiB(used) + " / " + fmtGiB(total) + " GiB";
      }
      // util may arrive as a direct percent on some compat backends
      var u = dev.utilization != null ? dev.utilization : (dev.gpu_util != null ? dev.gpu_util : null);
      if (u != null && !isNaN(u)) out.utilPct = Math.max(0, Math.min(100, u <= 1 ? u * 100 : u));
    }
    // torch_vram fallback at system level (no devices array)
    if (out.vramPct == null && stats.system) {
      var st = stats.system;
      if (st.vram_total && st.vram_free != null) {
        var u2 = st.vram_total - st.vram_free;
        out.vramPct = Math.max(0, Math.min(100, (u2 / st.vram_total) * 100));
        out.vramText = fmtGiB(u2) + " / " + fmtGiB(st.vram_total) + " GiB";
      }
    }
    return out;
  }

  function setBar(barEl, fillEl, pct) {
    barEl.classList.remove("warn", "crit");
    if (pct == null) { fillEl.style.width = "0"; return; }
    fillEl.style.width = pct.toFixed(0) + "%";
    if (pct >= 90) barEl.classList.add("crit");
    else if (pct >= 75) barEl.classList.add("warn");
  }

  // ---- module --------------------------------------------------------------
  Serenity.register("topbar", {
    init: function (ctx) {
      var root = (ctx.dom && ctx.dom.topbar) || document.getElementById(ROOT_ID);
      if (!root) { console.warn("[topbar] no mount"); return; }
      injectCSS();
      root.innerHTML = "";
      root.style.position = root.style.position || "relative"; // anchor popovers/meter

      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api;

      // brand
      var brand = el("div", "tb-brand");
      brand.appendChild(el("span", "tb-dot"));
      brand.appendChild(el("span", null, "Serenity Studio"));
      root.appendChild(brand);
      root.appendChild(el("div", "tb-sep"));

      // model picker
      var modelField = el("div", "tb-field");
      modelField.appendChild(el("label", null, "Model"));
      var modelSel = el("select", "tb-select");
      modelSel.title = "Checkpoint / base model";
      var mPlaceholder = el("option", null, "— loading —");
      mPlaceholder.value = "";
      modelSel.appendChild(mPlaceholder);
      modelSel.disabled = true;
      modelField.appendChild(modelSel);
      root.appendChild(modelField);

      // VAE override picker
      var vaeField = el("div", "tb-field");
      vaeField.appendChild(el("label", null, "VAE"));
      var vaeSel = el("select", "tb-select");
      vaeSel.title = "VAE override (Automatic = baked-in)";
      var vAuto = el("option", null, "Automatic");
      vAuto.value = "";
      vaeSel.appendChild(vAuto);
      vaeField.appendChild(vaeSel);
      root.appendChild(vaeField);

      root.appendChild(el("div", "tb-spacer"));

      // VRAM / util meter
      var meter = el("div", "tb-meter");
      var offline = el("div", "tb-offline");
      offline.appendChild(el("span", "tb-odot"));
      offline.appendChild(el("span", null, "API offline"));
      offline.style.display = "none";

      // vram gauge
      var vramGauge = el("div", "tb-gauge");
      var vramLabel = el("div", "tb-glabel");
      vramLabel.appendChild(el("span", null, "VRAM"));
      var vramVal = el("b", null, "—");
      vramLabel.appendChild(vramVal);
      vramGauge.appendChild(vramLabel);
      var vramBar = el("div", "tb-bar");
      var vramFill = el("i");
      vramBar.appendChild(vramFill);
      vramGauge.appendChild(vramBar);

      // util gauge
      var utilGauge = el("div", "tb-gauge");
      var utilLabel = el("div", "tb-glabel");
      utilLabel.appendChild(el("span", null, "GPU"));
      var utilVal = el("b", null, "—");
      utilLabel.appendChild(utilVal);
      utilGauge.appendChild(utilLabel);
      var utilBar = el("div", "tb-bar");
      var utilFill = el("i");
      utilBar.appendChild(utilFill);
      utilGauge.appendChild(utilBar);

      var gpuName = el("div", "tb-gpu");

      meter.appendChild(offline);
      meter.appendChild(vramGauge);
      meter.appendChild(utilGauge);
      meter.appendChild(gpuName);
      root.appendChild(meter);

      root.appendChild(el("div", "tb-sep"));

      // theme toggle
      var themeBtn = el("button", "btn tb-iconbtn");
      themeBtn.title = "Toggle light / dark theme";
      // settings
      var gearBtn = el("button", "btn tb-iconbtn", "⚙"); // ⚙
      gearBtn.title = "Settings";
      root.appendChild(themeBtn);
      root.appendChild(gearBtn);

      // ---- theme --------------------------------------------------------
      function currentTheme() {
        return document.documentElement.getAttribute("data-theme") || "dark";
      }
      function applyTheme(t) {
        document.documentElement.setAttribute("data-theme", t);
        themeBtn.textContent = t === "light" ? "☾" : "☀"; // moon when light, sun when dark
        try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
        bus.emit("theme:changed", t);
      }
      // restore persisted preference (if any) without forcing a flip
      try {
        var saved = localStorage.getItem(THEME_KEY);
        if (saved === "light" || saved === "dark") applyTheme(saved);
        else applyTheme(currentTheme());
      } catch (e) { applyTheme(currentTheme()); }
      themeBtn.addEventListener("click", function () {
        applyTheme(currentTheme() === "light" ? "dark" : "light");
      });

      // ---- settings popover ---------------------------------------------
      var pop = null;
      function closePop() {
        if (pop) { pop.remove(); pop = null; document.removeEventListener("mousedown", onDocDown, true); }
      }
      function onDocDown(ev) {
        if (pop && !pop.contains(ev.target) && ev.target !== gearBtn) closePop();
      }
      function openPop() {
        if (pop) { closePop(); return; }
        pop = el("div", "tb-pop");
        pop.appendChild(el("h4", null, "Settings"));

        // theme row
        var rTheme = el("div", "row");
        rTheme.appendChild(el("label", null, "Theme"));
        var tSel = el("select");
        ["dark", "light"].forEach(function (t) {
          var o = el("option", null, t.charAt(0).toUpperCase() + t.slice(1));
          o.value = t; if (currentTheme() === t) o.selected = true; tSel.appendChild(o);
        });
        tSel.addEventListener("change", function () { applyTheme(tSel.value); });
        rTheme.appendChild(tSel);
        pop.appendChild(rTheme);

        // api base (read-only, informational)
        var rApi = el("div", "row");
        rApi.appendChild(el("label", null, "API"));
        var apiIn = el("input");
        apiIn.type = "text";
        apiIn.readOnly = true;
        apiIn.value = (api && api.base) ? api.base : "(same origin)";
        apiIn.style.flex = "1";
        rApi.appendChild(apiIn);
        pop.appendChild(rApi);

        // refresh models
        var rRef = el("div", "row");
        var refBtn = el("button", "btn", "Refresh models");
        refBtn.style.flex = "1";
        refBtn.addEventListener("click", function () { loadModels(); loadVaes(); });
        rRef.appendChild(refBtn);
        pop.appendChild(rRef);

        pop.appendChild(el("small", null, "Client: " + (ctx.clientId || "?") + ". Stats poll every " + (POLL_MS / 1000) + "s."));

        root.appendChild(pop);
        // defer doc listener so the opening click doesn't immediately close it
        setTimeout(function () { document.addEventListener("mousedown", onDocDown, true); }, 0);
      }
      gearBtn.addEventListener("click", openPop);

      // ---- model / vae state wiring -------------------------------------
      modelSel.addEventListener("change", function () { set("params.model", modelSel.value); });
      vaeSel.addEventListener("change", function () { set("params.vae", vaeSel.value); });

      function fillSelect(sel, names, keepFirstOptions) {
        var prev = sel.value;
        // remove all but the kept leading options (placeholder/automatic)
        while (sel.children.length > (keepFirstOptions || 0)) sel.removeChild(sel.lastChild);
        names.forEach(function (n) {
          var o = el("option", null, n);
          o.value = n;
          sel.appendChild(o);
        });
        // restore selection: prefer state, then previous, then leave at first
        var want = sel === modelSel ? (get("params.model") || prev) : (get("params.vae") || prev);
        if (want && names.indexOf(want) >= 0) sel.value = want;
        else if (keepFirstOptions) sel.selectedIndex = 0;
      }

      function loadModels() {
        if (!api || typeof api.models !== "function") return;
        modelSel.disabled = true;
        // The serenity_worker_zimage backend ONLY runs Z-Image base today, so surface
        // that as the active model (selecting other list entries does not switch the
        // worker yet — multi-model routing is future work).
        var ACTIVE = "Z-Image (base)";
        api.models()
          .then(function (data) {
            var names = asNameList(data).filter(function (n) { return !/z\-?image/i.test(n); });
            names = [ACTIVE].concat(names);
            if (modelSel.firstChild) modelSel.firstChild.textContent = "— select model —";
            fillSelect(modelSel, names, 1);
            modelSel.disabled = false;
            modelSel.value = ACTIVE;
            set("params.model", ACTIVE);
          })
          .catch(function () {
            // backend model list unavailable — still show the real active model
            if (modelSel.firstChild) modelSel.firstChild.textContent = "— select model —";
            fillSelect(modelSel, [ACTIVE], 1);
            modelSel.disabled = false;
            modelSel.value = ACTIVE;
            set("params.model", ACTIVE);
          });
      }

      function loadVaes() {
        // VAE list isn't a first-class API endpoint in the contract; try the generic
        // object_info hint, else fall back gracefully to "Automatic" only.
        if (!api || typeof api.objectInfo !== "function") return;
        api.objectInfo("VAELoader")
          .then(function (info) {
            var names = [];
            try {
              // ComfyUI: {VAELoader:{input:{required:{vae_name:[[...names...]]}}}}
              var node = info && (info.VAELoader || info);
              var req = node && node.input && node.input.required;
              if (req && req.vae_name && Array.isArray(req.vae_name[0])) names = req.vae_name[0];
            } catch (e) {}
            fillSelect(vaeSel, asNameList(names), 1); // keep "Automatic"
          })
          .catch(function () { /* keep Automatic-only; not fatal */ });
      }

      // keep selects in sync if another module changes model/vae
      bus.on("change:params.model", function (v) {
        if (modelSel.value !== (v || "")) {
          var has = Array.prototype.some.call(modelSel.options, function (o) { return o.value === v; });
          if (has) modelSel.value = v || "";
        }
      });
      bus.on("change:params.vae", function (v) {
        if (vaeSel.value !== (v || "")) {
          var has = Array.prototype.some.call(vaeSel.options, function (o) { return o.value === v; });
          if (has) vaeSel.value = v || "";
        }
      });

      // ---- stats polling -------------------------------------------------
      var pollTimer = null, alive = true;
      function setOffline(isOff) {
        offline.style.display = isOff ? "flex" : "none";
        vramGauge.style.display = isOff ? "none" : "flex";
        utilGauge.style.display = isOff ? "none" : "flex";
        gpuName.style.display = isOff ? "none" : "block";
      }
      function poll() {
        if (!alive || !api || typeof api.systemStats !== "function") { setOffline(true); return; }
        api.systemStats()
          .then(function (stats) {
            var p = parseStats(stats);
            setOffline(false);
            vramVal.textContent = p.vramText.replace("VRAM ", "");
            setBar(vramBar, vramFill, p.vramPct);
            if (p.utilPct != null) {
              utilVal.textContent = p.utilPct.toFixed(0) + "%";
              setBar(utilBar, utilFill, p.utilPct);
              utilGauge.style.display = "flex";
            } else {
              // no util data from backend: show vram-only, hide GPU% gauge
              utilGauge.style.display = "none";
            }
            gpuName.textContent = p.gpuName || "";
            gpuName.title = p.gpuName || "";
          })
          .catch(function () { setOffline(true); });
      }

      // initial loads + polling loop
      loadModels();
      loadVaes();
      poll();
      pollTimer = setInterval(poll, POLL_MS);

      // pause polling while tab hidden (saves a request storm); resume on focus
      document.addEventListener("visibilitychange", function () {
        if (document.hidden) { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
        else if (!pollTimer) { poll(); pollTimer = setInterval(poll, POLL_MS); }
      });

      // graceful teardown if the app ever tears down
      bus.on("app:teardown", function () {
        alive = false;
        if (pollTimer) clearInterval(pollTimer);
        closePop();
      });

      console.info("[topbar] ready");
    }
  });
})();
