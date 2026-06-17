/* settings_tabs.js — module 'settingsTabs'. Fills the Settings overlay view
   (#view-settings, created by nav_shell) with three SwarmUI-style sub-tabs:

     Settings   — theme, prompt-autocomplete source, generation defaults
     Server     — backend / GPU status (reuses Serenity.api.systemStats) + logs
     Utilities  — power-user actions (export/import settings, params JSON,
                  clear gallery stars, model-tool placeholders that report
                  honestly when the backend can't do them yet)

   Mirrors workflows.js: we DO NOT edit nav_shell.js. nav_shell already builds an
   empty placeholder #view-settings; on 'app:ready' we replace that placeholder's
   contents with our sub-tab UI. All wiring is through ctx.state/get/set/bus/api,
   so we never import another module directly. Covers GAP section 11 rows:
   Utilities tab, Server tab, User/settings tab. Degrades gracefully when the
   backend 404s. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") { console.warn("[settingsTabs] Serenity not ready"); return; }

  // ---- persistence keys (namespaced, never collide with other modules) -------
  var THEME_KEY = "serenity.theme";                  // shared with topbar
  var AUTOCOMPLETE_KEY = "serenity.autocomplete.source";
  var DEFAULTS_KEY = "serenity.gen.defaults";
  var STAR_KEY = "serenity.gallery.stars";           // owned by gallery.js; we only clear it

  // generation-default fields we manage (subset of state.params worth persisting)
  var DEFAULT_FIELDS = ["steps", "cfg", "sampler", "scheduler", "width", "height", "images"];
  var SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "lcm"];
  var SCHEDULERS = ["simple", "normal", "karras", "exponential", "beta", "sgm_uniform"];
  var AUTOCOMPLETE_SOURCES = [
    ["none", "Off"],
    ["danbooru", "Danbooru tags"],
    ["a1111", "A1111 / e621 style"],
  ];

  function el(tag, cls, txt) { var n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; }
  function readJSON(key, fallback) { try { var v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; } }
  function writeJSON(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {} }

  // ---- scoped CSS (injected once; all rules under #settings-root) ------------
  function injectCSS() {
    if (document.getElementById("style-settingsTabs")) return;
    var css = [
      "#settings-root{display:flex;flex-direction:column;height:100%;color:var(--text)}",
      "#settings-root .st-subtabs{display:flex;gap:4px;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line)}",
      "#settings-root .st-subtabs .st-tab{background:transparent;border:1px solid transparent;color:var(--muted);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600}",
      "#settings-root .st-subtabs .st-tab:hover{color:var(--text)}",
      "#settings-root .st-subtabs .st-tab.active{background:var(--accent2);border-color:var(--accent);color:#fff}",
      "#settings-root .st-body{flex:1;overflow:auto;padding:18px 22px}",
      "#settings-root .st-pane{display:none;max-width:760px}",
      "#settings-root .st-pane.show{display:block}",
      "#settings-root .st-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;margin-bottom:14px}",
      "#settings-root .st-card>h3{margin:0 0 4px;font-size:13px;color:var(--text);font-weight:700}",
      "#settings-root .st-card>p.st-hint{margin:0 0 12px;color:var(--muted);font-size:11px;line-height:1.4}",
      "#settings-root .st-row{display:flex;align-items:center;gap:12px;margin:8px 0}",
      "#settings-root .st-row>label{color:var(--muted);min-width:150px;font-size:12px}",
      "#settings-root .st-row>label.wide{min-width:200px}",
      "#settings-root .st-row select,#settings-root .st-row input[type=text],#settings-root .st-row input[type=number]{flex:1;max-width:280px}",
      "#settings-root .st-row .st-num{max-width:120px}",
      "#settings-root .st-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}",
      "#settings-root .st-stat{display:grid;grid-template-columns:160px 1fr;gap:6px 14px;font-size:12px}",
      "#settings-root .st-stat .k{color:var(--muted)}",
      "#settings-root .st-stat .v{color:var(--text);font-variant-numeric:tabular-nums;word-break:break-word}",
      "#settings-root .st-bar{height:8px;border-radius:5px;background:var(--panel2);border:1px solid var(--line);overflow:hidden;margin-top:4px}",
      "#settings-root .st-bar>i{display:block;height:100%;width:0;background:var(--ok);transition:width .4s ease,background .4s ease}",
      "#settings-root .st-bar.warn>i{background:var(--warn)}",
      "#settings-root .st-bar.crit>i{background:var(--danger)}",
      "#settings-root .st-pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line)}",
      "#settings-root .st-pill .dot{width:8px;height:8px;border-radius:50%}",
      "#settings-root .st-pill.ok{color:var(--ok);border-color:var(--ok)} #settings-root .st-pill.ok .dot{background:var(--ok)}",
      "#settings-root .st-pill.off{color:var(--warn);border-color:var(--warn)} #settings-root .st-pill.off .dot{background:var(--warn)}",
      "#settings-root .st-logs{background:#0d0f14;border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;line-height:1.5;color:#c8cdda;white-space:pre-wrap;max-height:320px;overflow:auto;margin-top:8px}",
      "#settings-root .st-toast{position:fixed;bottom:148px;left:50%;transform:translateX(-50%);z-index:120;background:var(--panel2);border:1px solid var(--accent);color:var(--text);padding:8px 16px;border-radius:8px;font-size:12px;box-shadow:0 8px 24px rgba(0,0,0,.45);opacity:0;transition:opacity .25s ease;pointer-events:none}",
      "#settings-root .st-toast.show{opacity:1}",
      "#settings-root .st-saved{color:var(--ok);font-size:11px;margin-left:8px;opacity:0;transition:opacity .25s ease}",
      "#settings-root .st-saved.show{opacity:1}",
    ].join("\n");
    var st = document.createElement("style"); st.id = "style-settingsTabs"; st.textContent = css;
    document.head.appendChild(st);
  }

  S.register("settingsTabs", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api, state = ctx.state;

      // nav_shell builds #view-settings AFTER all module init() runs synchronously
      // (it appends to the overlay in navShell.init). Defer mounting until app:ready
      // so the placeholder exists, then take it over. Guard against double-mount.
      var mounted = false;
      function mountWhenReady() {
        if (mounted) return;
        var host = document.getElementById("view-settings");
        if (!host) return false;
        mounted = true;
        build(host);
        return true;
      }

      bus.on("app:ready", mountWhenReady);
      // also try immediately + once on next tick, in case ordering already satisfied
      if (!mountWhenReady()) { setTimeout(mountWhenReady, 0); }

      // -------------------------------------------------------------------
      function build(host) {
        // wipe nav_shell's placeholder (.view-pad) and install our own root
        host.innerHTML = "";
        var root = el("div"); root.id = "settings-root";
        host.appendChild(root);

        // toast (transient confirmation)
        var toast = el("div", "st-toast"); root.appendChild(toast);
        var toastTimer = null;
        function showToast(msg) {
          toast.textContent = msg; toast.classList.add("show");
          if (toastTimer) clearTimeout(toastTimer);
          toastTimer = setTimeout(function () { toast.classList.remove("show"); }, 1800);
        }

        // ---- sub-tab bar ----
        var subbar = el("div", "st-subtabs");
        var body = el("div", "st-body");
        var panes = {};
        var SUBTABS = [["settings", "Settings"], ["server", "Server"], ["utilities", "Utilities"]];
        var tabBtns = {};
        SUBTABS.forEach(function (t) {
          var b = el("button", "st-tab", t[1]); b.dataset.sub = t[0];
          b.addEventListener("click", function () { showSub(t[0]); });
          subbar.appendChild(b); tabBtns[t[0]] = b;
          var pane = el("div", "st-pane"); pane.dataset.sub = t[0];
          panes[t[0]] = pane; body.appendChild(pane);
        });
        root.appendChild(subbar); root.appendChild(body);

        var activeSub = "settings";
        function showSub(id) {
          activeSub = id;
          SUBTABS.forEach(function (t) {
            tabBtns[t[0]].classList.toggle("active", t[0] === id);
            panes[t[0]].classList.toggle("show", t[0] === id);
          });
          if (id === "server") startServerPoll(); else stopServerPoll();
        }

        buildSettingsPane(panes.settings, showToast);
        var serverApi = buildServerPane(panes.server);
        buildUtilitiesPane(panes.utilities, showToast);

        var startServerPoll = serverApi.start, stopServerPoll = serverApi.stop;

        showSub("settings");

        // only poll the server pane while the Settings overlay tab is actually visible
        bus.on("nav:tab", function (t) {
          if (t === "settings" && activeSub === "server") startServerPoll();
          else stopServerPoll();
        });
        bus.on("app:teardown", function () { stopServerPoll(); });
      }

      // =====================================================================
      // SETTINGS pane
      // =====================================================================
      function buildSettingsPane(pane, showToast) {
        // ---- Appearance / theme ----
        var cTheme = el("div", "st-card");
        cTheme.appendChild(el("h3", null, "Appearance"));
        cTheme.appendChild(el("p", "st-hint", "Theme is shared with the top bar toggle and remembered between sessions."));
        var rTheme = el("div", "st-row");
        rTheme.appendChild(el("label", null, "Theme"));
        var themeSel = el("select");
        [["dark", "Dark"], ["light", "Light"]].forEach(function (o) {
          var op = el("option", null, o[1]); op.value = o[0]; themeSel.appendChild(op);
        });
        themeSel.value = currentTheme();
        themeSel.addEventListener("change", function () { applyTheme(themeSel.value); showToast("Theme: " + themeSel.value); });
        rTheme.appendChild(themeSel);
        cTheme.appendChild(rTheme);
        pane.appendChild(cTheme);
        // keep dropdown in sync if topbar toggles theme
        bus.on("theme:changed", function (t) { if (themeSel.value !== t) themeSel.value = t; });

        // ---- Prompt autocomplete source ----
        var cAuto = el("div", "st-card");
        cAuto.appendChild(el("h3", null, "Prompt autocomplete"));
        cAuto.appendChild(el("p", "st-hint", "Which tag source the prompt bar should use for autocomplete. Saved locally; prompt modules read 'serenity.autocomplete.source' / listen for 'settings:autocomplete'."));
        var rAuto = el("div", "st-row");
        rAuto.appendChild(el("label", null, "Tag source"));
        var autoSel = el("select");
        AUTOCOMPLETE_SOURCES.forEach(function (o) { var op = el("option", null, o[1]); op.value = o[0]; autoSel.appendChild(op); });
        autoSel.value = readJSON(AUTOCOMPLETE_KEY, "none");
        autoSel.addEventListener("change", function () {
          writeJSON(AUTOCOMPLETE_KEY, autoSel.value);
          bus.emit("settings:autocomplete", autoSel.value);
          showToast("Autocomplete: " + autoSel.value);
        });
        rAuto.appendChild(autoSel);
        cAuto.appendChild(rAuto);
        pane.appendChild(cAuto);
        // broadcast the persisted value once so prompt modules can pick it up on boot
        bus.emit("settings:autocomplete", autoSel.value);

        // ---- Prompt generator (LLM) ----
        // Which local GGUF LLM expands a short idea into the Ideogram-4 structured
        // caption (the "Generate from idea" button on the Ideogram4 bbox node).
        // Persisted to 'serenity.promptgen.llm' (the gguf PATH); the node reads it.
        var cLLM = el("div", "st-card");
        cLLM.appendChild(el("h3", null, "Prompt generator (LLM)"));
        cLLM.appendChild(el("p", "st-hint", "Local GGUF model that turns a short idea into the Ideogram-4 bbox caption. Runs on demand via an ephemeral llama-server (GPU, freed after). Abliterated models won't refuse."));
        var rLLM = el("div", "st-row");
        rLLM.appendChild(el("label", null, "Model"));
        var llmSel = el("select");
        llmSel.appendChild(function () { var o = el("option", null, "(loading…)"); o.value = ""; return o; }());
        rLLM.appendChild(llmSel);
        cLLM.appendChild(rLLM);
        pane.appendChild(cLLM);
        var LLM_KEY = "serenity.promptgen.llm";
        fetch("/v1/llms").then(function (r) { return r.json(); }).then(function (j) {
          var list = (j && j.llms) || [];
          llmSel.innerHTML = "";
          if (!list.length) { var o = el("option", null, "(no GGUF models found)"); o.value = ""; llmSel.appendChild(o); return; }
          var saved = readJSON(LLM_KEY, "");
          // default to a Qwen3 if present (abliterated prompt-gen), else first
          var def = list.find(function (m) { return /qwen3/i.test(m.name); }) || list[0];
          list.forEach(function (m) {
            var o = el("option", null, m.name); o.value = m.path; llmSel.appendChild(o);
          });
          llmSel.value = (saved && list.some(function (m) { return m.path === saved; })) ? saved : def.path;
          writeJSON(LLM_KEY, llmSel.value);
        }).catch(function () {
          llmSel.innerHTML = ""; var o = el("option", null, "(server unavailable)"); o.value = ""; llmSel.appendChild(o);
        });
        llmSel.addEventListener("change", function () {
          writeJSON(LLM_KEY, llmSel.value);
          bus.emit("settings:promptgen", llmSel.value);
          showToast("Prompt LLM: " + (llmSel.options[llmSel.selectedIndex] || {}).text);
        });

        // ---- Generation defaults ----
        var cDef = el("div", "st-card");
        cDef.appendChild(el("h3", null, "Generation defaults"));
        cDef.appendChild(el("p", "st-hint", "Default values applied to new sessions. “Apply now” writes them into the current params immediately."));
        var saved = el("span", "st-saved", "saved");
        cDef.querySelector("h3").appendChild(saved);

        var defaults = readJSON(DEFAULTS_KEY, {});
        var inputs = {};

        function numRow(field, label, opts) {
          opts = opts || {};
          var r = el("div", "st-row");
          r.appendChild(el("label", null, label));
          var i = el("input", "st-num"); i.type = "number";
          if (opts.step != null) i.step = opts.step;
          if (opts.min != null) i.min = opts.min;
          var cur = defaults[field] != null ? defaults[field] : get("params." + field);
          if (cur != null) i.value = cur;
          inputs[field] = i; r.appendChild(i);
          cDef.appendChild(r);
        }
        function selRow(field, label, options) {
          var r = el("div", "st-row");
          r.appendChild(el("label", null, label));
          var s = el("select");
          options.forEach(function (o) { var op = el("option", null, o); op.value = o; s.appendChild(op); });
          var cur = defaults[field] != null ? defaults[field] : get("params." + field);
          if (cur != null) s.value = cur;
          inputs[field] = s; r.appendChild(s);
          cDef.appendChild(r);
        }

        numRow("steps", "Steps", { step: 1, min: 1 });
        numRow("cfg", "CFG scale", { step: 0.1, min: 0 });
        selRow("sampler", "Sampler", SAMPLERS);
        selRow("scheduler", "Scheduler", SCHEDULERS);
        numRow("width", "Width", { step: 8, min: 64 });
        numRow("height", "Height", { step: 8, min: 64 });
        numRow("images", "Batch size", { step: 1, min: 1 });

        function collect() {
          var out = {};
          DEFAULT_FIELDS.forEach(function (f) {
            var node = inputs[f]; if (!node) return;
            if (node.tagName === "SELECT") out[f] = node.value;
            else { var n = parseFloat(node.value); out[f] = isNaN(n) ? get("params." + f) : n; }
          });
          return out;
        }
        function flashSaved() { saved.classList.add("show"); setTimeout(function () { saved.classList.remove("show"); }, 1200); }

        var actions = el("div", "st-actions");
        var saveBtn = el("button", "btn", "Save as defaults");
        var applyBtn = el("button", "btn btn-primary", "Apply now");
        var resetBtn = el("button", "btn", "Reset to current params");
        saveBtn.addEventListener("click", function () { writeJSON(DEFAULTS_KEY, collect()); flashSaved(); showToast("Defaults saved"); });
        applyBtn.addEventListener("click", function () {
          var d = collect(); writeJSON(DEFAULTS_KEY, d);
          DEFAULT_FIELDS.forEach(function (f) { if (d[f] != null) set("params." + f, d[f]); });
          flashSaved(); showToast("Applied to current params");
        });
        resetBtn.addEventListener("click", function () {
          DEFAULT_FIELDS.forEach(function (f) {
            var cur = get("params." + f); if (inputs[f] && cur != null) inputs[f].value = cur;
          });
          showToast("Loaded current params");
        });
        actions.appendChild(saveBtn); actions.appendChild(applyBtn); actions.appendChild(resetBtn);
        cDef.appendChild(actions);
        pane.appendChild(cDef);
      }

      // =====================================================================
      // SERVER pane
      // =====================================================================
      function buildServerPane(pane) {
        // status pill + backend/GPU stat block
        var cStat = el("div", "st-card");
        var head = el("div", "st-row");
        head.style.justifyContent = "space-between";
        head.appendChild(el("h3", null, "Backend status"));
        var pill = el("span", "st-pill off");
        pill.appendChild(el("span", "dot")); pill.appendChild(el("span", null, "checking…"));
        head.appendChild(pill);
        cStat.appendChild(head);

        var grid = el("div", "st-stat");
        function statRow(key) {
          var k = el("div", "k", key); var v = el("div", "v", "—");
          grid.appendChild(k); grid.appendChild(v); return v;
        }
        var vApi = statRow("API base");
        var vGpu = statRow("GPU");
        var vVram = statRow("VRAM");
        var vUtil = statRow("GPU utilization");
        var vBackend = statRow("Backend");
        cStat.appendChild(grid);

        // vram visual bar
        var vbar = el("div", "st-bar"); var vfill = el("i"); vbar.appendChild(vfill); cStat.appendChild(vbar);

        var refresh = el("button", "btn", "Refresh");
        refresh.style.marginTop = "12px";
        refresh.addEventListener("click", function () { poll(true); });
        cStat.appendChild(refresh);
        pane.appendChild(cStat);

        vApi.textContent = (api && api.base) ? api.base : "(same origin)";

        // ---- logs ----
        var cLogs = el("div", "st-card");
        var lhead = el("div", "st-row"); lhead.style.justifyContent = "space-between";
        lhead.appendChild(el("h3", null, "Server logs"));
        var logBtn = el("button", "btn", "Refresh logs");
        lhead.appendChild(logBtn);
        cLogs.appendChild(lhead);
        cLogs.appendChild(el("p", "st-hint", "Tails the backend log if the server exposes /v1/logs. The serenity worker streams progress over WS; full log tailing depends on the server build."));
        var logBox = el("div", "st-logs", "(no logs loaded)");
        cLogs.appendChild(logBox);
        pane.appendChild(cLogs);

        logBtn.addEventListener("click", loadLogs);
        function loadLogs() {
          logBox.textContent = "loading…";
          var base = (api && api.base) || "";
          fetch(base + "/v1/logs").then(function (r) {
            if (!r.ok) throw new Error(r.status);
            return r.text();
          }).then(function (txt) {
            logBox.textContent = txt && txt.trim() ? txt : "(empty)";
            logBox.scrollTop = logBox.scrollHeight;
          }).catch(function () {
            logBox.textContent = "Log endpoint unavailable (/v1/logs not served by this backend).\n" +
              "Generation progress is available live in the gallery/queue strip via the WS stream.";
          });
        }

        // ---- polling control (driven by sub-tab visibility) ----
        var timer = null;
        function setBar(pct) {
          vbar.classList.remove("warn", "crit");
          if (pct == null) { vfill.style.width = "0"; return; }
          vfill.style.width = pct.toFixed(0) + "%";
          if (pct >= 90) vbar.classList.add("crit"); else if (pct >= 75) vbar.classList.add("warn");
        }
        function gib(b) { return (b == null || isNaN(b)) ? null : (b / (1024 * 1024 * 1024)).toFixed(1); }
        function poll(announce) {
          if (!api || typeof api.systemStats !== "function") { setOffline(); return; }
          api.systemStats().then(function (stats) {
            setOnline();
            var dev = null;
            if (stats && Array.isArray(stats.devices) && stats.devices.length) {
              dev = stats.devices.find(function (d) { return d && d.type && /cuda|gpu/i.test(d.type); }) || stats.devices[0];
            }
            if (dev) {
              vGpu.textContent = dev.name || dev.type || "—";
              var total = dev.vram_total != null ? dev.vram_total : dev.total;
              var free = dev.vram_free != null ? dev.vram_free : dev.free;
              if (total != null && free != null && total > 0) {
                var used = total - free, pct = Math.max(0, Math.min(100, used / total * 100));
                vVram.textContent = gib(used) + " / " + gib(total) + " GiB (" + pct.toFixed(0) + "%)";
                setBar(pct);
              } else { vVram.textContent = "—"; setBar(null); }
              var u = dev.utilization != null ? dev.utilization : (dev.gpu_util != null ? dev.gpu_util : null);
              vUtil.textContent = (u != null && !isNaN(u)) ? (u <= 1 ? (u * 100).toFixed(0) : Number(u).toFixed(0)) + "%" : "n/a";
            } else {
              vGpu.textContent = "—"; vVram.textContent = "—"; vUtil.textContent = "—"; setBar(null);
            }
            // backend identity if the health payload reports it
            var bk = (stats && (stats.backend || stats.worker || stats.model)) ||
              (stats && stats.system && (stats.system.backend || stats.system.os)) || "online";
            vBackend.textContent = String(bk);
            if (announce) { /* manual refresh: no-op beyond updating fields */ }
          }).catch(function () { setOffline(); });
        }
        function setOnline() { pill.className = "st-pill ok"; pill.lastChild.textContent = "online"; }
        function setOffline() {
          pill.className = "st-pill off"; pill.lastChild.textContent = "offline";
          vGpu.textContent = "—"; vVram.textContent = "—"; vUtil.textContent = "—"; vBackend.textContent = "—"; setBar(null);
        }
        function start() { if (timer) return; poll(); timer = setInterval(poll, 3000); }
        function stop() { if (timer) { clearInterval(timer); timer = null; } }
        return { start: start, stop: stop };
      }

      // =====================================================================
      // UTILITIES pane
      // =====================================================================
      function buildUtilitiesPane(pane, showToast) {
        // ---- settings backup ----
        var cBak = el("div", "st-card");
        cBak.appendChild(el("h3", null, "Settings backup"));
        cBak.appendChild(el("p", "st-hint", "Export every Serenity localStorage setting (theme, defaults, autocomplete, gallery stars) to a JSON file, or restore from one."));
        var bakActions = el("div", "st-actions");
        var expBtn = el("button", "btn", "Export settings");
        var impBtn = el("button", "btn", "Import settings");
        var impFile = el("input"); impFile.type = "file"; impFile.accept = "application/json"; impFile.style.display = "none";
        expBtn.addEventListener("click", function () { exportSettings(showToast); });
        impBtn.addEventListener("click", function () { impFile.click(); });
        impFile.addEventListener("change", function () {
          var f = impFile.files && impFile.files[0]; if (!f) return;
          var rdr = new FileReader();
          rdr.onload = function () {
            try {
              var obj = JSON.parse(rdr.result);
              Object.keys(obj).forEach(function (k) {
                if (k.indexOf("serenity.") === 0) localStorage.setItem(k, typeof obj[k] === "string" ? obj[k] : JSON.stringify(obj[k]));
              });
              showToast("Settings imported — reload to apply");
            } catch (e) { showToast("Import failed: not valid JSON"); }
            impFile.value = "";
          };
          rdr.readAsText(f);
        });
        bakActions.appendChild(expBtn); bakActions.appendChild(impBtn); bakActions.appendChild(impFile);
        cBak.appendChild(bakActions);
        pane.appendChild(cBak);

        // ---- current params tools ----
        var cParams = el("div", "st-card");
        cParams.appendChild(el("h3", null, "Current parameters"));
        cParams.appendChild(el("p", "st-hint", "Copy the live generation params (everything that would be submitted) to the clipboard as JSON — handy for sharing or debugging."));
        var pActions = el("div", "st-actions");
        var copyBtn = el("button", "btn", "Copy params JSON");
        var dlBtn = el("button", "btn", "Download params JSON");
        copyBtn.addEventListener("click", function () {
          var txt = paramsJSON();
          copyText(txt).then(function (ok) { showToast(ok ? "Params copied" : "Copy blocked — select & copy manually"); });
        });
        dlBtn.addEventListener("click", function () { downloadText("serenity-params.json", paramsJSON()); showToast("Downloaded params"); });
        pActions.appendChild(copyBtn); pActions.appendChild(dlBtn);
        cParams.appendChild(pActions);
        pane.appendChild(cParams);

        // ---- gallery maintenance ----
        var cGal = el("div", "st-card");
        cGal.appendChild(el("h3", null, "Gallery maintenance"));
        cGal.appendChild(el("p", "st-hint", "Clear locally-stored starred-image markers. Does not delete any files on disk."));
        var gActions = el("div", "st-actions");
        var clrStars = el("button", "btn", "Clear gallery stars");
        clrStars.addEventListener("click", function () {
          try { localStorage.removeItem(STAR_KEY); } catch (e) {}
          bus.emit("gallery:stars-cleared");
          showToast("Gallery stars cleared (reload to refresh strip)");
        });
        gActions.appendChild(clrStars);
        cGal.appendChild(gActions);
        pane.appendChild(cGal);

        // ---- model tools (honest placeholders) ----
        var cTools = el("div", "st-card");
        cTools.appendChild(el("h3", null, "Model tools"));
        cTools.appendChild(el("p", "st-hint", "Checks use the same prequeue gate as generation and report the local Mojodiffusion block profile."));
        var tActions = el("div", "st-actions");
        var pfOut = el("div", "st-logs", "(not checked)");
        var pfBtn = el("button", "btn", "Check current model");
        pfBtn.addEventListener("click", function () {
          if (!api || typeof api.preflight !== "function") { showToast("Preflight endpoint unavailable"); return; }
          pfOut.textContent = "checking...";
          api.preflight().then(function (r) {
            var bp = (r && r.block_profile) || {};
            var blocks = bp.block_count != null ? String(bp.block_count) : "n/a";
            var backend = (r && (r.backend || bp.family)) || "unknown";
            var verdict = r && r.admitted ? "admitted" : "blocked";
            var msg = verdict + " | " + backend + " | blocks " + blocks;
            if (r && r.error) msg += "\n" + r.error;
            if (bp.source) msg += "\nsource: " + bp.source;
            pfOut.textContent = msg;
            showToast(verdict + ": " + backend);
            console.info("[settingsTabs] preflight", r);
          }).catch(function (e) {
            var msg = e && e.message ? e.message : String(e || "preflight failed");
            pfOut.textContent = msg;
            showToast("Preflight failed: " + msg);
          });
        });
        tActions.appendChild(pfBtn);
        [
          ["Pickle → safetensors", "Conversion needs a /v1/convert endpoint (not served yet)."],
          ["Download model", "Model downloader needs a /v1/download endpoint (not served yet)."],
          ["Rebuild metadata", "Metadata tools need /v1/models/refresh (not served yet)."],
        ].forEach(function (t) {
          var b = el("button", "btn", t[0]);
          b.title = t[1];
          b.addEventListener("click", function () { showToast(t[1]); console.info("[settingsTabs] utility:", t[0], "—", t[1]); });
          tActions.appendChild(b);
        });
        cTools.appendChild(tActions);
        cTools.appendChild(pfOut);
        pane.appendChild(cTools);
      }

      // ---- helpers shared by panes ----------------------------------------
      function currentTheme() {
        try { var s = localStorage.getItem(THEME_KEY); if (s === "light" || s === "dark") return s; } catch (e) {}
        return document.documentElement.getAttribute("data-theme") || "dark";
      }
      function applyTheme(t) {
        document.documentElement.setAttribute("data-theme", t);
        try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
        bus.emit("theme:changed", t);
      }

      function paramsJSON() {
        var p = (state && state.params) || {};
        try { return JSON.stringify(p, null, 2); } catch (e) { return "{}"; }
      }
      function exportSettings(showToast) {
        var out = {};
        try {
          for (var i = 0; i < localStorage.length; i++) {
            var k = localStorage.key(i);
            if (k && k.indexOf("serenity.") === 0) out[k] = localStorage.getItem(k);
          }
        } catch (e) {}
        downloadText("serenity-settings.json", JSON.stringify(out, null, 2));
        if (showToast) showToast("Exported " + Object.keys(out).length + " keys");
      }
      function downloadText(name, text) {
        var blob = new Blob([text], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a"); a.href = url; a.download = name;
        document.body.appendChild(a); a.click();
        setTimeout(function () { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
      }
      function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          return navigator.clipboard.writeText(text).then(function () { return true; }).catch(function () { return fallbackCopy(text); });
        }
        return Promise.resolve(fallbackCopy(text));
      }
      function fallbackCopy(text) {
        try {
          var ta = document.createElement("textarea"); ta.value = text;
          ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta);
          ta.select(); var ok = document.execCommand("copy"); document.body.removeChild(ta); return ok;
        } catch (e) { return false; }
      }

      console.info("[settingsTabs] ready");
    },
  });
})();
