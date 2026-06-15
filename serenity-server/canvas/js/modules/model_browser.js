/* model_browser.js — module 'modelBrowser'. A SwarmUI-style model + LoRA browser
   overlay that fills the "Models" nav tab (#view-models, created by nav_shell) and
   also opens as a floating overlay via a launcher button + the `modelBrowser:open` bus
   event. Fed by Serenity.api.models() (GET /v1/models -> serenity.models.v1 doc):
       { models:[{name,path,arch,size,loaded,thumbnail,preview,favorite,metadata,card}],
         loras: [{name,path,size,arch,target_arch,trigger,compatible,compatibility,...}] }
   Two tabs (Models / LoRAs). Card or list view. Folder tree built from each entry's
   path. Click a model card => set('params.model', name) (+ visual "selected"/"loaded").
   Click a LoRA card => append {name,weight:1.0} to params.loras (de-duped). A right-hand
   metadata pane shows the full per-entry detail (arch, size, path, trigger, compat,
   raw metadata). Thumbnails use entry.preview||entry.thumbnail (or the card's), and fall
   back to a generated SVG placeholder. Degrades gracefully when /v1/models 404s.
   Owns ONLY this file + the floating overlay it creates. Talks via state/get/set/bus/api. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") { console.warn("[modelBrowser] Serenity not ready"); return; }

  var STYLE_ID = "style-modelBrowser";
  var VIEW_PREF_KEY = "serenity.modelBrowser.view"; // 'cards' | 'list'

  // ---- tiny DOM helper (same shape as the other modules) -------------------
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

  function fmtBytes(n) {
    if (n == null || isNaN(n)) return "—";
    n = +n;
    if (n < 1024) return n + " B";
    var u = ["KB", "MB", "GB", "TB"], i = -1;
    do { n /= 1024; i++; } while (n >= 1024 && i < u.length - 1);
    return (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + " " + u[i];
  }

  // basename of a path (entry.name may already be a basename; path is full)
  function baseName(p) { return String(p || "").replace(/\\/g, "/").split("/").pop(); }

  // folder portion of a path RELATIVE to its filename (best-effort tree key).
  // We don't know the scan root, so we group on the immediate parent dir name(s).
  function folderOf(p) {
    var s = String(p || "").replace(/\\/g, "/");
    var parts = s.split("/");
    parts.pop(); // drop filename
    if (!parts.length) return "";
    // keep up to the last 3 dir segments so the tree is meaningful but compact
    return parts.slice(Math.max(0, parts.length - 3)).join("/");
  }

  // deterministic accent color from a string (arch/family) for the placeholder tile
  function hueOf(str) {
    var h = 0, s = String(str || "model");
    for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h % 360;
  }

  // data: URI SVG placeholder tile (no external assets, theme-neutral)
  function placeholder(label, sub) {
    var hue = hueOf(label);
    var initials = String(label || "?").replace(/[^A-Za-z0-9]+/g, " ").trim().split(" ")
      .slice(0, 2).map(function (w) { return w[0]; }).join("").toUpperCase() || "?";
    var bg1 = "hsl(" + hue + ",40%,28%)", bg2 = "hsl(" + ((hue + 40) % 360) + ",42%,18%)";
    var svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">' +
      '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">' +
      '<stop offset="0" stop-color="' + bg1 + '"/><stop offset="1" stop-color="' + bg2 + '"/>' +
      '</linearGradient></defs>' +
      '<rect width="240" height="240" fill="url(#g)"/>' +
      '<text x="120" y="118" font-family="system-ui,sans-serif" font-size="84" font-weight="700" ' +
      'fill="rgba(255,255,255,.86)" text-anchor="middle" dominant-baseline="middle">' + esc(initials) + '</text>' +
      (sub ? '<text x="120" y="186" font-family="system-ui,sans-serif" font-size="20" ' +
        'fill="rgba(255,255,255,.55)" text-anchor="middle">' + esc(sub) + '</text>' : "") +
      '</svg>';
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[<>&"']/g, function (c) {
      return ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  // Resolve a usable thumbnail URL for an entry; fall back to placeholder SVG.
  function thumbUrl(entry, api) {
    var card = entry && entry.card;
    var src = (entry && (entry.preview || entry.thumbnail)) ||
      (card && (card.preview || card.thumbnail)) || "";
    if (src) {
      // relative server paths -> route through viewUrl when it looks like a file
      if (/^https?:|^data:/.test(src)) return src;
      if (api && typeof api.viewUrl === "function") {
        try { return api.viewUrl(baseName(src)); } catch (_) {}
      }
      return src;
    }
    var sub = entry && (entry.arch || entry.target_arch || (entry.metadata && entry.metadata.family));
    return placeholder(entry && entry.name, sub);
  }

  // ---- scoped CSS ----------------------------------------------------------
  function injectCSS() {
    if (document.getElementById(STYLE_ID)) return;
    var P = ".mb-root";
    var css = [
      // overlay (used when opened as a floating modal, not as the nav view)
      "#mb-overlay{position:fixed;inset:0;z-index:120;background:rgba(8,9,12,.62);display:none}",
      "#mb-overlay.show{display:block}",
      "#mb-overlay .mb-modal{position:absolute;inset:40px;background:var(--panel);border:1px solid var(--line);" +
        "border-radius:12px;box-shadow:0 18px 60px rgba(0,0,0,.55);display:flex;flex-direction:column;overflow:hidden}",
      // root (shared by nav-view mode and modal mode)
      P + "{display:flex;flex-direction:column;height:100%;min-height:0;color:var(--text)}",
      // header / toolbar
      P + " .mb-bar{display:flex;align-items:center;gap:8px;padding:10px 12px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}",
      P + " .mb-title{font-size:14px;font-weight:700;margin:0 6px 0 0;letter-spacing:.02em}",
      P + " .mb-tabs{display:flex;gap:3px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:3px}",
      P + " .mb-tab{background:transparent;border:0;color:var(--muted);padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600}",
      P + " .mb-tab:hover{color:var(--text)}",
      P + " .mb-tab.active{background:var(--accent2);color:#fff}",
      P + " .mb-tab .mb-count{opacity:.7;font-weight:500;margin-left:5px}",
      P + " .mb-search{flex:1;min-width:140px}",
      P + " .mb-sp{flex:1}",
      P + " .mb-viewtog{display:flex;gap:3px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:3px}",
      P + " .mb-viewtog button{background:transparent;border:0;color:var(--muted);padding:4px 9px;border-radius:6px;cursor:pointer;font-size:13px;line-height:1}",
      P + " .mb-viewtog button.active{background:var(--accent2);color:#fff}",
      P + " .mb-iconbtn{padding:5px 9px;line-height:1}",
      // body: tree | grid | metadata
      P + " .mb-body{display:flex;flex:1;min-height:0}",
      P + " .mb-tree{width:188px;flex:0 0 auto;border-right:1px solid var(--line);overflow:auto;padding:8px 6px;background:var(--panel)}",
      P + " .mb-tree .mb-tnode{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
      P + " .mb-tree .mb-tnode:hover{background:var(--panel2);color:var(--text)}",
      P + " .mb-tree .mb-tnode.active{background:var(--accent2);color:#fff}",
      P + " .mb-tree .mb-tnode .mb-tcount{margin-left:auto;opacity:.7;font-size:11px}",
      P + " .mb-tree .mb-tnode.mb-tindent{padding-left:20px}",
      // grid area
      P + " .mb-main{flex:1;min-width:0;display:flex;flex-direction:column}",
      P + " .mb-scroll{flex:1;overflow:auto;padding:12px}",
      P + " .mb-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}",
      P + " .mb-card{position:relative;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer;background:var(--panel2);transition:border-color .12s,transform .08s}",
      P + " .mb-card:hover{border-color:var(--accent)}",
      P + " .mb-card.active{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent2) inset}",
      P + " .mb-card.loaded::after{content:'LOADED';position:absolute;top:6px;left:6px;background:var(--ok);color:#06210f;font-size:9px;font-weight:700;letter-spacing:.06em;padding:2px 6px;border-radius:5px}",
      P + " .mb-card.incompat{opacity:.62}",
      P + " .mb-card.incompat::after{content:'INCOMPATIBLE';position:absolute;top:6px;left:6px;background:var(--warn);color:#241c00;font-size:9px;font-weight:700;padding:2px 6px;border-radius:5px}",
      P + " .mb-thumb{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--panel)}",
      P + " .mb-meta{padding:7px 9px}",
      P + " .mb-name{font-size:12px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
      P + " .mb-sub{font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:6px;margin-top:2px}",
      P + " .mb-fav{position:absolute;top:6px;right:6px;color:var(--warn);font-size:14px;text-shadow:0 1px 2px #000}",
      // list view
      P + " .mb-list .mb-card{display:flex;align-items:center;gap:10px;aspect-ratio:auto}",
      P + " .mb-list .mb-thumb{width:46px;height:46px;flex:0 0 auto;aspect-ratio:1/1;border-radius:7px;margin:7px 0 7px 8px}",
      P + " .mb-list .mb-meta{flex:1;min-width:0;padding-right:10px}",
      P + " .mb-list.mb-grid{display:flex;flex-direction:column;gap:7px}",
      // metadata pane
      P + " .mb-detail{width:268px;flex:0 0 auto;border-left:1px solid var(--line);overflow:auto;padding:14px;background:var(--panel)}",
      P + " .mb-detail.empty{display:flex;align-items:center;justify-content:center;color:var(--muted);text-align:center;font-size:12px}",
      P + " .mb-detail .mb-dthumb{width:100%;border-radius:8px;border:1px solid var(--line);margin-bottom:10px;display:block}",
      P + " .mb-detail h3{font-size:14px;margin:0 0 2px;word-break:break-word}",
      P + " .mb-detail .mb-darch{color:var(--muted);font-size:12px;margin-bottom:12px}",
      P + " .mb-detail dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:5px 10px;font-size:12px}",
      P + " .mb-detail dt{color:var(--muted)}",
      P + " .mb-detail dd{margin:0;word-break:break-word;color:var(--text)}",
      P + " .mb-detail dd.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}",
      P + " .mb-detail .mb-pill{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;letter-spacing:.04em}",
      P + " .mb-detail .mb-pill.ok{background:var(--ok);color:#06210f}",
      P + " .mb-detail .mb-pill.no{background:var(--warn);color:#241c00}",
      P + " .mb-detail .mb-dact{margin-top:14px;display:flex;flex-direction:column;gap:7px}",
      P + " .mb-detail .mb-dact .btn{width:100%}",
      P + " .mb-detail .mb-raw{margin-top:12px}",
      P + " .mb-detail .mb-raw summary{cursor:pointer;color:var(--muted);font-size:11px;user-select:none}",
      P + " .mb-detail .mb-raw pre{margin:6px 0 0;font-size:10.5px;white-space:pre-wrap;word-break:break-word;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:8px;color:var(--text)}",
      // empty / status
      P + " .mb-status{padding:26px;color:var(--muted);text-align:center;font-size:13px}",
      P + " .mb-status .btn{margin-top:12px}",
    ].join("\n");
    var st = el("style", { id: STYLE_ID, text: css });
    document.head.appendChild(st);
  }

  // =========================================================================
  S.register("modelBrowser", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api;

      // ---- state ----------------------------------------------------------
      var data = { models: [], loras: [], loaded: false, error: null };
      var ui = {
        tab: "models",                 // 'models' | 'loras'
        view: loadView(),              // 'cards' | 'list'
        search: "",
        folder: "",                    // active folder filter ("" = all)
        selected: null,                // currently detailed entry
      };
      var mountedIn = null;            // 'view' | 'modal'

      // ---- build the root (reused for both the nav view and the modal) ----
      var root = el("div", { class: "mb-root" });

      // toolbar
      var tabModels = el("button", { class: "mb-tab active", type: "button", text: "Models" });
      var tabLoras = el("button", { class: "mb-tab", type: "button", text: "LoRAs" });
      var cntModels = el("span", { class: "mb-count", text: "" });
      var cntLoras = el("span", { class: "mb-count", text: "" });
      tabModels.appendChild(cntModels); tabLoras.appendChild(cntLoras);
      tabModels.addEventListener("click", function () { setTab("models"); });
      tabLoras.addEventListener("click", function () { setTab("loras"); });

      var searchIn = el("input", { class: "mb-search", type: "search", placeholder: "Search…" });
      searchIn.addEventListener("input", function () { ui.search = searchIn.value.trim().toLowerCase(); renderGrid(); });

      var viewCards = el("button", { type: "button", title: "Card view", html: "▦" });
      var viewList = el("button", { type: "button", title: "List view", html: "≣" });
      viewCards.addEventListener("click", function () { setView("cards"); });
      viewList.addEventListener("click", function () { setView("list"); });
      var viewTog = el("div", { class: "mb-viewtog" }, [viewCards, viewList]);

      var refreshBtn = el("button", { class: "btn mb-iconbtn", type: "button", title: "Reload model list", html: "⟳" });
      refreshBtn.addEventListener("click", function () { load(true); });

      var closeBtn = el("button", { class: "btn mb-iconbtn", type: "button", title: "Close", html: "✕" });
      closeBtn.style.display = "none"; // only shown in modal mode
      closeBtn.addEventListener("click", closeModal);

      var bar = el("div", { class: "mb-bar" }, [
        el("span", { class: "mb-title", text: "Browse" }),
        el("div", { class: "mb-tabs" }, [tabModels, tabLoras]),
        searchIn,
        viewTog,
        refreshBtn,
        closeBtn,
      ]);

      // body: tree | grid | detail
      var treeEl = el("div", { class: "mb-tree" });
      var gridEl = el("div", { class: "mb-grid" });
      var scrollEl = el("div", { class: "mb-scroll" }, [gridEl]);
      var mainEl = el("div", { class: "mb-main" }, [scrollEl]);
      var detailEl = el("div", { class: "mb-detail empty", text: "Select an item to see its metadata." });
      var bodyEl = el("div", { class: "mb-body" }, [treeEl, mainEl, detailEl]);

      root.appendChild(bar);
      root.appendChild(bodyEl);
      syncViewButtons();

      // ---- modal overlay (created lazily, reuses `root`) ------------------
      var overlay = null, modalHost = null;
      function ensureModal() {
        if (overlay) return;
        overlay = el("div", { id: "mb-overlay" });
        modalHost = el("div", { class: "mb-modal" });
        overlay.appendChild(modalHost);
        overlay.addEventListener("mousedown", function (e) { if (e.target === overlay) closeModal(); });
        document.body.appendChild(overlay);
      }
      function openModal() {
        // if the root is currently in the nav view, the modal is a no-op (already visible there)
        if (mountedIn === "view" && isNavModelsVisible()) return;
        ensureModal();
        modalHost.appendChild(root);          // move root into the modal
        mountedIn = "modal";
        closeBtn.style.display = "";
        overlay.classList.add("show");
        if (!data.loaded) load(false);
        searchIn.focus();
      }
      function closeModal() {
        if (overlay) overlay.classList.remove("show");
        // hand root back to the nav view container if it exists
        var host = document.getElementById("view-models");
        if (host) { host.innerHTML = ""; host.appendChild(root); mountedIn = "view"; closeBtn.style.display = "none"; }
      }
      function isNavModelsVisible() {
        var v = document.getElementById("view-models");
        var o = document.getElementById("wf-overlay");
        return !!(v && o && o.classList.contains("show") && v.classList.contains("show"));
      }

      // ---- entry points ---------------------------------------------------
      // Primary access is the existing nav "Models" tab (created by nav_shell).
      // Other modules can also open the browser via the `modelBrowser:open` bus
      // event; if nav_shell is absent we fall back to a self-contained modal.
      bus.on("modelBrowser:open", function (which) {
        if (which === "loras" || which === "models") ui.tab = which;
        if (S.nav && typeof S.nav.show === "function") S.nav.show("models");
        else openModal();
        setTab(ui.tab);
      });

      // ---- mount into the nav view (#view-models) -------------------------
      function mountInView() {
        var host = document.getElementById("view-models");
        if (!host) return false;
        host.innerHTML = "";
        host.appendChild(root);
        mountedIn = "view";
        closeBtn.style.display = "none";
        return true;
      }
      if (!mountInView()) {
        // nav_shell may init after us; retry once everything booted
        bus.on("app:ready", function () { mountInView(); });
      }

      // lazy-load when the Models tab is shown
      bus.on("nav:tab", function (t) {
        if (t !== "models") return;
        if (!mountInView()) { /* still no container; ignore */ }
        if (!data.loaded && !data.error) load(false);
        else renderAll();
      });

      // ---- data load ------------------------------------------------------
      function load(force) {
        if (data.loaded && !force) { renderAll(); return; }
        data.error = null;
        showStatus("Loading models…");
        if (!api || typeof api.models !== "function") { data.error = "API unavailable"; data.loaded = true; renderAll(); return; }
        var selModel = get("params.model") || "";
        Promise.resolve()
          .then(function () { return api.models(selModel ? { model: selModel } : undefined); })
          .then(function (doc) {
            data.models = normalizeEntries(doc, "models");
            data.loras = normalizeEntries(doc, "loras");
            data.loaded = true;
            renderAll();
          })
          .catch(function (e) {
            data.error = (e && e.message) || "failed to load";
            data.loaded = true;
            data.models = []; data.loras = [];
            renderAll();
          });
      }

      // api.models() may be called with no args (existing callers do). The shared
      // adapter ignores args, so passing {model} is harmless; we tolerate either.
      function normalizeEntries(doc, key) {
        var arr = null;
        if (doc && Array.isArray(doc[key])) arr = doc[key];
        else if (Array.isArray(doc)) arr = doc;               // bare array fallback
        else if (doc && Array.isArray(doc.data)) arr = doc.data;
        if (!arr) return [];
        return arr.map(function (e) {
          if (typeof e === "string") return { name: e, path: e, arch: "", size: null };
          var c = e.card || {};
          return {
            name: e.name || c.title || baseName(e.path) || "(unnamed)",
            path: e.path || c.path || e.name || "",
            arch: e.arch || e.target_arch || c.subtitle || (e.metadata && e.metadata.family) || "",
            target_arch: e.target_arch || "",
            size: e.size != null ? e.size : c.size,
            loaded: !!e.loaded,
            favorite: !!(e.favorite || c.favorite),
            trigger: e.trigger || (e.metadata && e.metadata.trigger) || "",
            compatible: e.compatible !== false,           // models have no compat -> true
            hasCompat: e.compatibility != null || e.compatible != null,
            incompatible_reason: e.incompatible_reason || (e.compatibility && e.compatibility.incompatible_reason) || "",
            preview: e.preview || "", thumbnail: e.thumbnail || "",
            card: c,
            metadata: e.metadata || c.metadata || {},
            raw: e,
          };
        });
      }

      // ---- rendering ------------------------------------------------------
      function currentList() { return ui.tab === "loras" ? data.loras : data.models; }

      function setTab(tab) {
        ui.tab = tab;
        tabModels.classList.toggle("active", tab === "models");
        tabLoras.classList.toggle("active", tab === "loras");
        ui.folder = "";              // reset folder filter per tab
        ui.selected = null;
        renderAll();
      }
      function setView(v) {
        ui.view = v; saveView(v); syncViewButtons();
        gridEl.classList.toggle("mb-list", v === "list");
        renderGrid();
      }
      function syncViewButtons() {
        viewCards.classList.toggle("active", ui.view === "cards");
        viewList.classList.toggle("active", ui.view === "list");
        gridEl.classList.toggle("mb-list", ui.view === "list");
      }

      function renderAll() {
        cntModels.textContent = data.models.length ? String(data.models.length) : "";
        cntLoras.textContent = data.loras.length ? String(data.loras.length) : "";
        renderTree();
        renderGrid();
        renderDetail();
      }

      function showStatus(msg, withRetry) {
        gridEl.innerHTML = "";
        treeEl.innerHTML = "";
        var box = el("div", { class: "mb-status", text: msg });
        if (withRetry) {
          box.appendChild(el("br"));
          var b = el("button", { class: "btn", type: "button", text: "Retry" });
          b.addEventListener("click", function () { load(true); });
          box.appendChild(b);
        }
        gridEl.appendChild(box);
      }

      function renderTree() {
        treeEl.innerHTML = "";
        var list = currentList();
        // build folder -> count
        var counts = {};
        list.forEach(function (e) { var f = folderOf(e.path) || "(root)"; counts[f] = (counts[f] || 0) + 1; });
        var folders = Object.keys(counts).sort();
        // "All" node
        var allNode = el("div", {
          class: "mb-tnode" + (ui.folder === "" ? " active" : ""),
          onclick: function () { ui.folder = ""; ui.selected = null; renderTree(); renderGrid(); },
        }, [el("span", { text: "All " + (ui.tab === "loras" ? "LoRAs" : "models") }),
            el("span", { class: "mb-tcount", text: String(list.length) })]);
        treeEl.appendChild(allNode);
        folders.forEach(function (f) {
          var node = el("div", {
            class: "mb-tnode mb-tindent" + (ui.folder === f ? " active" : ""),
            title: f,
            onclick: function () { ui.folder = (ui.folder === f ? "" : f); ui.selected = null; renderTree(); renderGrid(); },
          }, [el("span", { text: f }), el("span", { class: "mb-tcount", text: String(counts[f]) })]);
          treeEl.appendChild(node);
        });
      }

      function filtered() {
        var list = currentList();
        var q = ui.search, f = ui.folder;
        return list.filter(function (e) {
          if (f && (folderOf(e.path) || "(root)") !== f) return false;
          if (!q) return true;
          var hay = (e.name + " " + (e.arch || "") + " " + (e.trigger || "") + " " + (e.path || "")).toLowerCase();
          return hay.indexOf(q) >= 0;
        });
      }

      function renderGrid() {
        if (!data.loaded) return; // status already shown
        if (data.error) { showStatus("Could not load models: " + data.error, true); return; }
        var list = filtered();
        gridEl.innerHTML = "";
        if (!list.length) {
          gridEl.appendChild(el("div", { class: "mb-status", text: currentList().length ? "No matches." : "No " + (ui.tab === "loras" ? "LoRAs" : "models") + " found on disk." }));
          return;
        }
        var selModel = get("params.model") || "";
        var loras = get("params.loras") || [];
        list.forEach(function (e) { gridEl.appendChild(cardEl(e, selModel, loras)); });
      }

      function cardEl(e, selModel, loras) {
        var isModel = ui.tab === "models";
        var active = isModel
          ? (e.name === selModel || e.path === selModel)
          : loras.some(function (l) { return l && (l.name === e.name); });
        var incompat = !isModel && e.hasCompat && e.compatible === false;

        var card = el("div", { class: "mb-card" + (active ? " active" : "") + (e.loaded ? " loaded" : "") + (incompat ? " incompat" : "") });
        var img = el("img", { class: "mb-thumb", alt: e.name, loading: "lazy", decoding: "async", src: thumbUrl(e, api) });
        img.addEventListener("error", function () { img.src = placeholder(e.name, e.arch); }, { once: true });
        card.appendChild(img);
        if (e.favorite) card.appendChild(el("div", { class: "mb-fav", text: "★" }));
        card.appendChild(el("div", { class: "mb-meta" }, [
          el("div", { class: "mb-name", title: e.name, text: e.name }),
          el("div", { class: "mb-sub" }, [
            el("span", { text: e.arch || (isModel ? "model" : "lora") }),
            el("span", { text: fmtBytes(e.size) }),
          ]),
        ]));
        card.addEventListener("click", function () { ui.selected = e; renderDetail(); });
        card.addEventListener("dblclick", function () { applyEntry(e); });
        return card;
      }

      // ---- detail / metadata pane -----------------------------------------
      function renderDetail() {
        var e = ui.selected;
        detailEl.innerHTML = "";
        if (!e) {
          detailEl.classList.add("empty");
          detailEl.textContent = "Select an item to see its metadata.";
          return;
        }
        detailEl.classList.remove("empty");
        var isModel = ui.tab === "models";
        var selModel = get("params.model") || "";
        var loras = get("params.loras") || [];
        var active = isModel
          ? (e.name === selModel || e.path === selModel)
          : loras.some(function (l) { return l && l.name === e.name; });

        detailEl.appendChild(el("img", { class: "mb-dthumb", alt: e.name, src: thumbUrl(e, api),
          onerror: function () { this.src = placeholder(e.name, e.arch); } }));
        detailEl.appendChild(el("h3", { title: e.name, text: e.name }));
        detailEl.appendChild(el("div", { class: "mb-darch", text: (e.arch || "unknown arch") + (e.loaded ? " · loaded" : "") }));

        var dl = el("dl");
        function row(k, v, mono) {
          if (v == null || v === "") return;
          dl.appendChild(el("dt", { text: k }));
          dl.appendChild(el("dd", { class: mono ? "mono" : "", title: String(v), text: String(v) }));
        }
        row("Type", isModel ? (e.raw.type || "checkpoint") : "lora");
        row("Arch", e.arch);
        if (!isModel) row("Target arch", e.target_arch || e.arch);
        row("Size", fmtBytes(e.size));
        if (e.trigger) row("Trigger", e.trigger);
        row("Path", e.path, true);
        if (e.metadata && e.metadata.source) row("Source", e.metadata.source);
        if (e.metadata && e.metadata.notes) row("Notes", e.metadata.notes);
        detailEl.appendChild(dl);

        // LoRA compatibility pill
        if (!isModel && e.hasCompat) {
          var compatRow = el("div", { style: "margin-top:12px" });
          if (e.compatible) {
            compatRow.appendChild(el("span", { class: "mb-pill ok", text: "COMPATIBLE" }));
          } else {
            compatRow.appendChild(el("span", { class: "mb-pill no", text: "INCOMPATIBLE" }));
            if (e.incompatible_reason)
              compatRow.appendChild(el("div", { style: "color:var(--muted);font-size:11px;margin-top:6px", text: e.incompatible_reason }));
          }
          detailEl.appendChild(compatRow);
        }

        // actions
        var acts = el("div", { class: "mb-dact" });
        if (isModel) {
          var useBtn = el("button", { class: "btn" + (active ? "" : " btn-primary"), type: "button",
            text: active ? "✓ Current model" : "Use this model" });
          useBtn.disabled = active;
          useBtn.addEventListener("click", function () { applyEntry(e); });
          acts.appendChild(useBtn);
          // re-query compat for LoRAs against this model (refresh list with model arg)
          var refreshCompatBtn = el("button", { class: "btn", type: "button", text: "Set + check LoRAs" });
          refreshCompatBtn.title = "Select this model and re-check LoRA compatibility";
          refreshCompatBtn.addEventListener("click", function () { applyEntry(e); load(true); });
          acts.appendChild(refreshCompatBtn);
        } else {
          var addBtn = el("button", { class: "btn" + (active ? "" : " btn-primary"), type: "button",
            text: active ? "✓ Added" : "Add LoRA" });
          addBtn.disabled = active;
          addBtn.addEventListener("click", function () { applyEntry(e); });
          acts.appendChild(addBtn);
          if (active) {
            var rmBtn = el("button", { class: "btn", type: "button", text: "Remove from stack" });
            rmBtn.addEventListener("click", function () { removeLora(e); });
            acts.appendChild(rmBtn);
          }
        }
        detailEl.appendChild(acts);

        // raw JSON metadata
        var raw = el("details", { class: "mb-raw" }, [
          el("summary", { text: "Raw metadata" }),
          el("pre", { text: safeJSON(e.raw && (e.raw.metadata || e.raw)) }),
        ]);
        detailEl.appendChild(raw);
      }

      function safeJSON(o) {
        try { return JSON.stringify(o, null, 2); } catch (_) { return "(unserializable)"; }
      }

      // ---- selection actions ----------------------------------------------
      function applyEntry(e) {
        if (ui.tab === "models") { selectModel(e); }
        else { addLora(e); }
      }

      function selectModel(e) {
        set("params.model", e.name);
        flash("Model set: " + e.name);
        // also surface as the detailed item + repaint active states
        ui.selected = e;
        renderGrid(); renderDetail();
        bus.emit("modelBrowser:model", { name: e.name, arch: e.arch, path: e.path });
      }

      function addLora(e) {
        var loras = (get("params.loras") || []).slice();
        if (loras.some(function (l) { return l && l.name === e.name; })) { flash("Already in stack"); return; }
        loras.push({ name: e.name, weight: 1.0 });
        set("params.loras", loras);
        flash("LoRA added: " + e.name);
        ui.selected = e;
        renderGrid(); renderDetail();
        bus.emit("modelBrowser:lora", { name: e.name, weight: 1.0, list: loras });
      }

      function removeLora(e) {
        var loras = (get("params.loras") || []).filter(function (l) { return !(l && l.name === e.name); });
        set("params.loras", loras);
        flash("LoRA removed: " + e.name);
        renderGrid(); renderDetail();
        bus.emit("modelBrowser:lora", { name: e.name, removed: true, list: loras });
      }

      // repaint when model/loras change elsewhere (param_rail dropdown, gallery restore)
      bus.on("change:params.model", function () { if (data.loaded) { renderGrid(); renderDetail(); } });
      bus.on("change:params.loras", function () { if (data.loaded) { renderGrid(); renderDetail(); } });

      // ---- flash toast (scoped to root) -----------------------------------
      var flashTimer = null, flashEl = null;
      function flash(msg) {
        if (!flashEl) {
          flashEl = el("div", { style: "position:absolute;left:50%;bottom:14px;transform:translateX(-50%);" +
            "background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:6px 14px;" +
            "font-size:12px;color:var(--text);box-shadow:0 6px 20px rgba(0,0,0,.4);opacity:0;" +
            "transition:opacity .15s;pointer-events:none;z-index:5" });
          root.style.position = root.style.position || "relative";
          root.appendChild(flashEl);
        }
        flashEl.textContent = msg;
        flashEl.style.opacity = "1";
        clearTimeout(flashTimer);
        flashTimer = setTimeout(function () { if (flashEl) flashEl.style.opacity = "0"; }, 1500);
      }

      // ---- view pref persistence ------------------------------------------
      function loadView() {
        try { var v = localStorage.getItem(VIEW_PREF_KEY); return (v === "list") ? "list" : "cards"; }
        catch (_) { return "cards"; }
      }
      function saveView(v) { try { localStorage.setItem(VIEW_PREF_KEY, v); } catch (_) {} }

      // ---- kick off if the Models tab happens to be the initial view ------
      // (nav_shell defaults to 'generate'; we still preload lazily on first show.)
      // If something opened us before app:ready, ensure a load happens.
      if (isNavModelsVisible()) load(false);

      console.info("[modelBrowser] ready");
    },
  });
})();
