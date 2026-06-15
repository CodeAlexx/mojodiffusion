/* presets_browser.js — module 'presetsBrowser'.
   A SwarmUI-style presets browser for the Generate tab:
     - folders (encoded as "folder/name" in the preset name; tree view)
     - multi-preset toggle-STACKING (toggle several presets on -> their params
       merge, in order, onto state.params)
     - save-current-as-preset (captures the SwarmUI param signature)
     - drag-drop import of .json preset files (single preset or {presets:[...]})
     - export a preset back to a .json file
     - <preset:name> injection in the prompt: a capture-phase hook on
       #btn-generate resolves <preset:foo> tokens in state.params.prompt just
       before submit, splicing in the referenced preset's prompt + merging its
       params, then restores the editor text after submit fires.

   Backing store: the PROVEN daemon API at /v1/presets (probed at init via
   api.base). On 404 / network failure it transparently falls back to
   localStorage so the browser is always usable. The daemon preset shape is
   {name, params}; folders ride on the name ("portraits/cinematic"). We always
   POST the ROOT /v1/presets endpoint ({name,params}) because the daemon's
   /v1/presets/:name path-form rejects '/' in the name.

   Self-contained: registers window.Serenity.modules.presetsBrowser and talks to
   the rest of the app ONLY through ctx.{state,get,set,bus,api}. Owns a floating
   panel toggled from a #tooldock button (and the bus event 'presets:open').
   Writes the active stack to state.presets.* (a NEW state subtree, never edits
   state.js). Param restore mirrors gallery.js's PARAM_KEYS exactly. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  // SwarmUI param signature — same surface gallery.js restores (state.js params).
  var PARAM_KEYS = [
    "model", "vae", "prompt", "negative", "width", "height", "aspect",
    "steps", "cfg", "seed", "images", "sampler", "scheduler",
    "clip_skip", "sigma_min", "sigma_max", "eta", "restart_sampling",
    "loras", "controlnet", "refiner", "denoise",
  ];
  var LS_KEY = "serenity.presets.v1";       // localStorage fallback doc
  var MAX_PRESET_DEPTH = 8;                 // <preset:> recursion guard

  // ---------------------------------------------------------------- DOM helper
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k === "style") n.setAttribute("style", attrs[k]);
      else if (k.slice(0, 2) === "on" && typeof attrs[k] === "function") n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    }
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  // folder/name parsing — last "/" splits folder from leaf
  function splitName(name) {
    var s = String(name || "");
    var i = s.lastIndexOf("/");
    if (i < 0) return { folder: "", leaf: s };
    return { folder: s.slice(0, i), leaf: s.slice(i + 1) };
  }

  function injectCSS() {
    if (document.getElementById("style-presetsBrowser")) return;
    var css = [
      // tooldock launcher button
      "#tooldock .pbz-launch{width:34px;height:34px;display:flex;align-items:center;justify-content:center;font-size:16px;line-height:1;padding:0}",
      "#tooldock .pbz-launch.on{background:var(--accent2);border-color:var(--accent);color:#fff}",
      // floating panel
      "#pbz-panel{position:fixed;top:96px;left:64px;z-index:70;width:340px;max-height:72vh;display:none;",
      "  flex-direction:column;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);",
      "  box-shadow:0 12px 36px rgba(0,0,0,.5);font:13px/1.4 system-ui,sans-serif;color:var(--text);overflow:hidden}",
      "#pbz-panel.show{display:flex}",
      "#pbz-panel.drag{outline:2px dashed var(--accent);outline-offset:-4px}",
      "#pbz-head{display:flex;align-items:center;gap:8px;padding:9px 10px;background:var(--panel2);",
      "  border-bottom:1px solid var(--line);cursor:move;user-select:none}",
      "#pbz-head h3{margin:0;font-size:13px;font-weight:700;flex:1}",
      "#pbz-head .pbz-x{width:24px;height:24px;padding:0;font-size:14px;line-height:1}",
      "#pbz-tools{display:flex;gap:6px;padding:8px 10px;border-bottom:1px solid var(--line);flex-wrap:wrap}",
      "#pbz-tools .pbz-filter{flex:1;min-width:120px}",
      "#pbz-tools .pbz-b{font-size:11px;padding:5px 9px}",
      "#pbz-list{overflow:auto;padding:6px 8px 10px;flex:1}",
      "#pbz-list .pbz-folder{margin:4px 0}",
      "#pbz-list .pbz-foldhead{display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--muted);",
      "  font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:3px 2px;user-select:none}",
      "#pbz-list .pbz-foldhead:hover{color:var(--text)}",
      "#pbz-list .pbz-foldhead .pbz-caret{display:inline-block;width:10px;transition:transform .12s}",
      "#pbz-list .pbz-folder.collapsed .pbz-items{display:none}",
      "#pbz-list .pbz-folder.collapsed .pbz-caret{transform:rotate(-90deg)}",
      "#pbz-list .pbz-items{display:flex;flex-direction:column;gap:4px;padding-left:8px}",
      "#pbz-list .pbz-row{display:flex;align-items:center;gap:7px;padding:5px 7px;border:1px solid var(--line);",
      "  border-radius:6px;background:var(--panel2);cursor:pointer}",
      "#pbz-list .pbz-row:hover{border-color:var(--accent)}",
      "#pbz-list .pbz-row.active{border-color:var(--accent);background:var(--accent2);color:#fff}",
      "#pbz-list .pbz-row.active .pbz-meta{color:#dfe4ff}",
      "#pbz-list .pbz-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}",
      "#pbz-list .pbz-meta{color:var(--muted);font-size:10px;white-space:nowrap}",
      "#pbz-list .pbz-mini{width:22px;height:22px;padding:0;font-size:12px;line-height:1;background:transparent;border:0;color:var(--muted);cursor:pointer;border-radius:5px}",
      "#pbz-list .pbz-mini:hover{background:var(--line);color:var(--text)}",
      "#pbz-list .pbz-stackbadge{font-size:9px;background:var(--accent);color:#fff;border-radius:8px;padding:1px 6px;font-weight:700}",
      "#pbz-empty{color:var(--muted);padding:14px 8px;text-align:center;font-size:12px}",
      "#pbz-stackbar{padding:7px 10px;border-top:1px solid var(--line);background:var(--panel2);font-size:11px;color:var(--muted);display:none}",
      "#pbz-stackbar.show{display:block}",
      "#pbz-stackbar b{color:var(--text)}",
      "#pbz-stackbar .pbz-clear{float:right;font-size:10px;padding:2px 8px}",
      "#pbz-toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:90;background:var(--panel2);",
      "  border:1px solid var(--line);border-radius:8px;padding:8px 14px;color:var(--text);font-size:12px;",
      "  box-shadow:0 8px 24px rgba(0,0,0,.4);opacity:0;transition:opacity .2s;pointer-events:none}",
      "#pbz-toast.show{opacity:1}",
    ].join("\n");
    var st = el("style", { id: "style-presetsBrowser" });
    st.textContent = css;
    document.head.appendChild(st);
  }

  S.register("presetsBrowser", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api;
      var apiBase = (api && api.base) || "";

      // ---- new state subtree (never touches state.js's defaults) -------------
      if (!ctx.state.presets) ctx.state.presets = { active: [], useBackend: true };

      var presets = [];          // [{name, params}] (mirror of the store)
      var collapsed = {};        // folder -> bool
      var useBackend = true;     // flips to false on first failed backend call

      // ================= store layer (backend with localStorage fallback) =====
      function lsLoad() {
        try {
          var raw = localStorage.getItem(LS_KEY);
          var doc = raw ? JSON.parse(raw) : null;
          if (doc && Array.isArray(doc.presets)) return doc.presets;
        } catch (_) {}
        return [];
      }
      function lsSave() {
        try { localStorage.setItem(LS_KEY, JSON.stringify({ schema: "serenity.presets.v1", presets: presets })); }
        catch (_) {}
      }

      function fetchJSON(path, opts) {
        return fetch(apiBase + path, opts).then(function (r) {
          if (!r.ok) throw new Error(r.status + " " + path);
          return r.json();
        });
      }

      // Load the full preset list. Tries the backend first; on ANY failure falls
      // back to localStorage and stays in fallback mode for the session.
      function reload() {
        return fetchJSON("/v1/presets")
          .then(function (doc) {
            useBackend = true;
            ctx.state.presets.useBackend = true;
            presets = (doc && Array.isArray(doc.presets)) ? doc.presets.slice() : [];
            render();
          })
          .catch(function () {
            useBackend = false;
            ctx.state.presets.useBackend = false;
            presets = lsLoad();
            render();
          });
      }

      // Upsert one preset {name, params}. ALWAYS the root POST so folder names
      // (which contain '/') are accepted. Mirrors into local array + persists.
      function upsert(name, params) {
        var entry = { name: name, params: params };
        var i = indexOfName(name);
        if (i >= 0) presets[i] = entry; else presets.push(entry);
        if (useBackend) {
          return fetchJSON("/v1/presets", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, params: params }),
          }).then(function () { render(); }).catch(function () {
            useBackend = false; ctx.state.presets.useBackend = false; lsSave(); render();
          });
        }
        lsSave(); render();
        return Promise.resolve();
      }

      function remove(name) {
        var i = indexOfName(name);
        if (i >= 0) presets.splice(i, 1);
        // drop from the active stack too
        var act = (get("presets.active") || []).filter(function (n) { return n !== name; });
        set("presets.active", act);
        if (useBackend) {
          return fetchJSON("/v1/presets/" + encodeURIComponent(name), { method: "DELETE" })
            .then(function () { render(); })
            .catch(function () { useBackend = false; ctx.state.presets.useBackend = false; lsSave(); render(); });
        }
        lsSave(); render();
        return Promise.resolve();
      }

      function indexOfName(name) {
        for (var i = 0; i < presets.length; i++) if (presets[i].name === name) return i;
        return -1;
      }
      function byName(name) { var i = indexOfName(name); return i >= 0 ? presets[i] : null; }

      // ================= param capture / apply ================================
      function captureCurrentParams() {
        var out = {};
        PARAM_KEYS.forEach(function (k) {
          var v = get("params." + k);
          if (v !== undefined) out[k] = v;
        });
        return out;
      }

      // Merge a preset's params onto state.params. Only the known SwarmUI keys
      // are forwarded (api.js forwards exactly these), mirroring gallery restore.
      function applyParams(params, opts) {
        if (!params) return;
        opts = opts || {};
        PARAM_KEYS.forEach(function (k) {
          if (params[k] === undefined || params[k] === null) return;
          if (opts.skipPrompt && (k === "prompt" || k === "negative")) return;
          set("params." + k, params[k]);
        });
        bus.emit("params:restored", params);
      }

      // ================= active stack (toggle-stacking) =======================
      function isActive(name) { return (get("presets.active") || []).indexOf(name) >= 0; }

      function toggleStack(name) {
        var act = (get("presets.active") || []).slice();
        var i = act.indexOf(name);
        if (i >= 0) act.splice(i, 1); else act.push(name);
        // set() emits change:presets.active -> the bus listener renders once (badges).
        set("presets.active", act);
        // then merge the stacked params/prompt onto state.params.
        applyStack(act);
      }

      // Re-apply the whole stack from a clean-ish base: each preset's params
      // merge in order, later presets winning. prompt/negative are concatenated
      // so multiple style presets compose instead of clobbering each other.
      function applyStack(act) {
        if (!act || !act.length) { renderStackbar(); return; }
        var promptParts = [], negParts = [];
        act.forEach(function (name) {
          var p = byName(name);
          if (!p || !p.params) return;
          applyParams(p.params, { skipPrompt: true });
          if (typeof p.params.prompt === "string" && p.params.prompt.trim()) promptParts.push(p.params.prompt.trim());
          if (typeof p.params.negative === "string" && p.params.negative.trim()) negParts.push(p.params.negative.trim());
        });
        // Compose prompts: keep the user's current text first, then stacked styles.
        if (promptParts.length) {
          var base = (get("params.prompt") || "").trim();
          var seen = {};
          var merged = (base ? [base] : []).concat(promptParts).filter(function (s) {
            if (seen[s]) return false; seen[s] = 1; return true;
          });
          set("params.prompt", merged.join(", "));
        }
        if (negParts.length) {
          var nbase = (get("params.negative") || "").trim();
          var nseen = {};
          var nmerged = (nbase ? [nbase] : []).concat(negParts).filter(function (s) {
            if (nseen[s]) return false; nseen[s] = 1; return true;
          });
          set("params.negative", nmerged.join(", "));
        }
        renderStackbar();
      }

      // ================= <preset:name> prompt resolver ========================
      // Replace <preset:foo> tokens with foo's params.prompt (recursive, guarded)
      // and collect the referenced presets so their non-prompt params can merge.
      function resolvePromptDirectives(text, depth, collected) {
        if (typeof text !== "string" || text.indexOf("<preset:") < 0) return text;
        if (depth > MAX_PRESET_DEPTH) return text;
        return text.replace(/<preset:([^>]+)>/g, function (_m, raw) {
          var name = String(raw).trim();
          var p = byName(name);
          if (!p) { console.warn("[presetsBrowser] <preset:" + name + "> not found"); return ""; }
          collected.push(p);
          var sub = (p.params && typeof p.params.prompt === "string") ? p.params.prompt : "";
          return resolvePromptDirectives(sub, depth + 1, collected);
        });
      }

      // Capture-phase hook on the Generate button: rewrite the prompt in place,
      // merge referenced presets' params, let submit read state.params, then
      // restore the editor text so the user keeps their <preset:> tokens.
      var restoreTimer = null;
      function onGenerateCapture() {
        var promptText = get("params.prompt");
        if (typeof promptText !== "string" || promptText.indexOf("<preset:") < 0) return;
        var collected = [];
        var resolved = resolvePromptDirectives(promptText, 0, collected);
        // merge non-prompt params from referenced presets (first reference wins
        // only where state.params doesn't already differ — keep it non-destructive
        // by applying then letting the explicit user fields stand via order).
        collected.forEach(function (p) { applyParams(p.params, { skipPrompt: true }); });
        set("params.prompt", resolved.replace(/\s{2,}/g, " ").trim());
        // restore the authored text shortly after submit has read the params
        if (restoreTimer) clearTimeout(restoreTimer);
        restoreTimer = setTimeout(function () { set("params.prompt", promptText); }, 0);
      }

      // bind in CAPTURE phase so we run BEFORE generateWS's bubble-phase handler
      function bindGenerateHook() {
        var btn = document.getElementById("btn-generate");
        if (!btn || btn.__pbzHooked) return;
        btn.__pbzHooked = true;
        btn.addEventListener("click", onGenerateCapture, true);
      }
      bus.on("app:ready", bindGenerateHook);
      bindGenerateHook();
      var hookTries = 0;
      var hookTimer = setInterval(function () {
        bindGenerateHook();
        if (document.getElementById("btn-generate") || ++hookTries > 25) clearInterval(hookTimer);
      }, 200);
      // also resolve when generation is triggered purely via the bus
      bus.on("generate:request", onGenerateCapture);

      // ================= UI ===================================================
      var panel, listEl, stackbar, filterIn, launchBtn, toast;

      function buildUI() {
        // tooldock launcher
        var dock = document.getElementById("tooldock");
        launchBtn = el("button", { class: "btn pbz-launch", type: "button", title: "Presets browser", html: "🗂" });
        launchBtn.addEventListener("click", togglePanel);
        if (dock) dock.appendChild(launchBtn);

        // floating panel
        panel = el("div", { id: "pbz-panel" });
        var head = el("div", { id: "pbz-head" }, [
          el("h3", { text: "Presets" }),
          el("button", { class: "btn pbz-x", type: "button", title: "Close", text: "✕",
            onclick: function () { showPanel(false); } }),
        ]);
        makeDraggable(panel, head);

        var tools = el("div", { id: "pbz-tools" });
        filterIn = el("input", { type: "text", class: "pbz-filter", placeholder: "Filter presets…" });
        filterIn.addEventListener("input", render);
        var saveBtn = el("button", { class: "btn btn-primary pbz-b", type: "button", text: "+ Save current",
          onclick: saveCurrent });
        var importBtn = el("button", { class: "btn pbz-b", type: "button", text: "Import",
          onclick: importDialog });
        var reloadBtn = el("button", { class: "btn pbz-b", type: "button", title: "Reload from server", text: "⟳",
          onclick: function () { reload(); } });
        tools.appendChild(filterIn);
        tools.appendChild(saveBtn);
        tools.appendChild(importBtn);
        tools.appendChild(reloadBtn);

        listEl = el("div", { id: "pbz-list" });

        stackbar = el("div", { id: "pbz-stackbar" });

        panel.appendChild(head);
        panel.appendChild(tools);
        panel.appendChild(listEl);
        panel.appendChild(stackbar);
        document.body.appendChild(panel);

        // drag-drop import onto the panel
        panel.addEventListener("dragover", function (e) { e.preventDefault(); panel.classList.add("drag"); });
        panel.addEventListener("dragleave", function (e) {
          if (e.target === panel) panel.classList.remove("drag");
        });
        panel.addEventListener("drop", function (e) {
          e.preventDefault(); panel.classList.remove("drag");
          var files = e.dataTransfer && e.dataTransfer.files;
          if (files && files.length) importFiles(files);
        });

        toast = el("div", { id: "pbz-toast" });
        document.body.appendChild(toast);
      }

      function showPanel(v) {
        panel.classList.toggle("show", v);
        launchBtn.classList.toggle("on", v);
        if (v) reload();
      }
      function togglePanel() { showPanel(!panel.classList.contains("show")); }

      function flash(msg) {
        if (!toast) return;
        toast.textContent = msg;
        toast.classList.add("show");
        clearTimeout(flash._t);
        flash._t = setTimeout(function () { toast.classList.remove("show"); }, 1800);
      }

      // ---- render the folder tree -------------------------------------------
      function render() {
        if (!listEl) return;
        renderStackbar();
        listEl.innerHTML = "";
        var q = (filterIn && filterIn.value.trim().toLowerCase()) || "";

        var groups = {};      // folder -> [entry]
        var order = [];       // folder insertion order
        presets.forEach(function (p) {
          var parts = splitName(p.name);
          if (q && p.name.toLowerCase().indexOf(q) < 0) return;
          if (!groups[parts.folder]) { groups[parts.folder] = []; order.push(parts.folder); }
          groups[parts.folder].push(p);
        });

        if (!order.length) {
          var msg = presets.length
            ? "No presets match the filter."
            : "No presets yet. Click “+ Save current”, drop a .json file here, or use <preset:name> in a prompt.";
          listEl.appendChild(el("div", { id: "pbz-empty", text: msg }));
          if (!useBackend && presets.length === 0) {
            listEl.appendChild(el("div", { id: "pbz-empty", style: "padding-top:0",
              text: "(server presets unavailable — using local storage)" }));
          }
          return;
        }

        // root folder first, then alpha
        order.sort(function (a, b) { return a === "" ? -1 : b === "" ? 1 : a.localeCompare(b); });
        order.forEach(function (folder) {
          var entries = groups[folder].sort(function (a, b) {
            return splitName(a.name).leaf.localeCompare(splitName(b.name).leaf);
          });
          if (folder === "") {
            // root presets render directly (no folder header)
            var rootItems = el("div", { class: "pbz-items", style: "padding-left:0" });
            entries.forEach(function (p) { rootItems.appendChild(presetRow(p)); });
            listEl.appendChild(rootItems);
          } else {
            var isColl = !!collapsed[folder];
            var folderEl = el("div", { class: "pbz-folder" + (isColl ? " collapsed" : "") });
            var activeCount = entries.filter(function (p) { return isActive(p.name); }).length;
            var head = el("div", { class: "pbz-foldhead" }, [
              el("span", { class: "pbz-caret", text: "▾" }),
              el("span", { text: "📁 " + folder }),
              el("span", { class: "pbz-meta", text: " (" + entries.length + ")" }),
              activeCount ? el("span", { class: "pbz-stackbadge", text: activeCount + " on" }) : null,
            ]);
            head.addEventListener("click", function () {
              collapsed[folder] = !collapsed[folder]; render();
            });
            var items = el("div", { class: "pbz-items" });
            entries.forEach(function (p) { items.appendChild(presetRow(p)); });
            folderEl.appendChild(head);
            folderEl.appendChild(items);
            listEl.appendChild(folderEl);
          }
        });
      }

      function presetRow(p) {
        var parts = splitName(p.name);
        var active = isActive(p.name);
        var row = el("div", { class: "pbz-row" + (active ? " active" : ""), title: p.name });
        var nkeys = p.params ? Object.keys(p.params).length : 0;

        var name = el("span", { class: "pbz-name", text: parts.leaf });
        var meta = el("span", { class: "pbz-meta", text: nkeys + " field" + (nkeys === 1 ? "" : "s") });

        // toggle-stack on row click (the core SwarmUI stacking gesture)
        row.addEventListener("click", function (e) {
          if (e.target.classList && e.target.classList.contains("pbz-mini")) return;
          toggleStack(p.name);
        });

        // apply-once (replace, not stack)
        var applyBtn = el("button", { class: "pbz-mini", type: "button", title: "Apply once (replace params)", text: "▶",
          onclick: function (e) { e.stopPropagation(); applyParams(p.params); flash("Applied “" + parts.leaf + "”"); } });
        // inject <preset:name> at the prompt cursor / end
        var injBtn = el("button", { class: "pbz-mini", type: "button", title: "Insert <preset:" + p.name + "> into prompt", text: "➕",
          onclick: function (e) { e.stopPropagation(); injectToken(p.name); } });
        // export
        var expBtn = el("button", { class: "pbz-mini", type: "button", title: "Export to .json", text: "⤓",
          onclick: function (e) { e.stopPropagation(); exportPreset(p); } });
        // delete
        var delBtn = el("button", { class: "pbz-mini", type: "button", title: "Delete preset", text: "✕",
          onclick: function (e) {
            e.stopPropagation();
            if (window.confirm("Delete preset “" + p.name + "”?")) remove(p.name).then(function () { flash("Deleted"); });
          } });

        row.appendChild(name);
        row.appendChild(meta);
        row.appendChild(applyBtn);
        row.appendChild(injBtn);
        row.appendChild(expBtn);
        row.appendChild(delBtn);
        return row;
      }

      function renderStackbar() {
        if (!stackbar) return;
        var act = get("presets.active") || [];
        if (!act.length) { stackbar.classList.remove("show"); stackbar.innerHTML = ""; return; }
        stackbar.classList.add("show");
        stackbar.innerHTML = "";
        var clear = el("button", { class: "btn pbz-clear", type: "button", text: "clear",
          onclick: function () { set("presets.active", []); render(); flash("Stack cleared"); } });
        var names = act.map(function (n) { return splitName(n).leaf; }).join(" + ");
        stackbar.appendChild(clear);
        stackbar.appendChild(el("span", { html: "<b>Stacked:</b> " + escapeHtml(names) }));
      }

      function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
          return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c];
        });
      }

      // insert <preset:name> at the prompt textarea cursor (falls back to append)
      function injectToken(name) {
        var token = "<preset:" + name + ">";
        var ta = document.querySelector("#prompt-bar textarea.pb-pos") ||
                 document.querySelector("#param-rail textarea.pr-prompt");
        if (ta && typeof ta.selectionStart === "number") {
          var s = ta.selectionStart, e = ta.selectionEnd;
          var v = ta.value;
          var ins = (s > 0 && v[s - 1] && v[s - 1] !== " " ? " " : "") + token + " ";
          ta.value = v.slice(0, s) + ins + v.slice(e);
          ta.dispatchEvent(new Event("input", { bubbles: true }));
          ta.focus();
          ta.selectionStart = ta.selectionEnd = s + ins.length;
        } else {
          var cur = get("params.prompt") || "";
          set("params.prompt", (cur ? cur + " " : "") + token);
        }
        flash("Inserted " + token);
      }

      // ---- save current params as a preset (prompts for a folder/name) ------
      function saveCurrent() {
        var def = (get("presets.lastName")) || "";
        var name = window.prompt(
          "Save current parameters as preset.\nUse “folder/name” to file it in a folder.",
          def);
        if (!name) return;
        name = name.trim();
        if (!name) return;
        if (indexOfName(name) >= 0 && !window.confirm("Overwrite existing preset “" + name + "”?")) return;
        set("presets.lastName", name);
        var params = captureCurrentParams();
        upsert(name, params).then(function () { flash("Saved “" + name + "”"); });
      }

      // ---- import / export ---------------------------------------------------
      function importDialog() {
        var inp = el("input", { type: "file", accept: ".json,application/json", multiple: "" });
        inp.addEventListener("change", function () { if (inp.files && inp.files.length) importFiles(inp.files); });
        inp.click();
      }

      function importFiles(fileList) {
        var files = Array.prototype.slice.call(fileList).filter(function (f) {
          return /\.json$/i.test(f.name) || /json/.test(f.type);
        });
        if (!files.length) { flash("Drop .json preset files"); return; }
        var imported = 0, pending = files.length;
        files.forEach(function (f) {
          var reader = new FileReader();
          reader.onload = function () {
            try {
              var doc = JSON.parse(reader.result);
              var entries = normalizeImport(doc, f.name);
              entries.forEach(function (en) { upsert(en.name, en.params); imported++; });
            } catch (e) {
              console.warn("[presetsBrowser] import parse failed for", f.name, e);
            }
            if (--pending === 0) { flash("Imported " + imported + " preset" + (imported === 1 ? "" : "s")); render(); }
          };
          reader.readAsText(f);
        });
      }

      // Accept several import shapes:
      //   {schema, presets:[{name,params}]}          (our export / daemon doc)
      //   {name, params}                              (single preset)
      //   {params:{...}}  or a bare params object     (file basename -> name)
      //   { "PresetName": {param object}, ... }       (a map of name->params)
      function normalizeImport(doc, filename) {
        var base = String(filename || "preset").replace(/\.json$/i, "");
        if (doc && Array.isArray(doc.presets)) {
          return doc.presets.filter(function (p) { return p && p.name; })
            .map(function (p) { return { name: String(p.name), params: p.params || {} }; });
        }
        if (doc && typeof doc.name === "string" && doc.params) {
          return [{ name: doc.name, params: doc.params }];
        }
        if (doc && doc.params && typeof doc.params === "object") {
          return [{ name: base, params: doc.params }];
        }
        if (doc && typeof doc === "object") {
          // map-of-presets vs a single bare params object: if every value is an
          // object AND no top-level value looks like a scalar param, treat as a map.
          var keys = Object.keys(doc);
          var looksLikeMap = keys.length > 0 && keys.every(function (k) {
            return doc[k] && typeof doc[k] === "object" && !Array.isArray(doc[k]);
          }) && keys.every(function (k) { return PARAM_KEYS.indexOf(k) < 0; });
          if (looksLikeMap) {
            return keys.map(function (k) { return { name: k, params: doc[k] }; });
          }
          // bare params object -> filter to known keys
          var params = {};
          PARAM_KEYS.forEach(function (k) { if (doc[k] !== undefined) params[k] = doc[k]; });
          if (Object.keys(params).length) return [{ name: base, params: params }];
        }
        return [];
      }

      function exportPreset(p) {
        var doc = { schema: "serenity.presets.v1", presets: [{ name: p.name, params: p.params || {} }] };
        var blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
        var a = el("a", { download: splitName(p.name).leaf.replace(/[^\w.-]+/g, "_") + ".json" });
        a.href = URL.createObjectURL(blob);
        document.body.appendChild(a); a.click();
        setTimeout(function () { try { URL.revokeObjectURL(a.href); } catch (_) {} a.remove(); }, 0);
        flash("Exported “" + splitName(p.name).leaf + "”");
      }

      // ---- drag the panel by its header -------------------------------------
      function makeDraggable(box, handle) {
        var ox = 0, oy = 0, dragging = false;
        handle.addEventListener("mousedown", function (e) {
          if (e.target.closest && e.target.closest("button")) return;
          dragging = true;
          var r = box.getBoundingClientRect();
          ox = e.clientX - r.left; oy = e.clientY - r.top;
          e.preventDefault();
        });
        document.addEventListener("mousemove", function (e) {
          if (!dragging) return;
          var x = Math.max(0, Math.min(window.innerWidth - 60, e.clientX - ox));
          var y = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - oy));
          box.style.left = x + "px"; box.style.top = y + "px";
        });
        document.addEventListener("mouseup", function () { dragging = false; });
      }

      // ================= wiring ===============================================
      buildUI();
      // let other modules open the browser (e.g. a future topbar/preset button)
      bus.on("presets:open", function () { showPanel(true); });
      bus.on("presets:reload", reload);
      // keep the active-stack badges fresh if another module mutates the stack
      bus.on("change:presets.active", function () { render(); });

      // prime the list quietly (so <preset:> resolution works even if the panel
      // was never opened) — backend probe, falls back to localStorage on failure
      reload();

      console.info("[presetsBrowser] ready (backend probe pending)");
    },
  });
})();
