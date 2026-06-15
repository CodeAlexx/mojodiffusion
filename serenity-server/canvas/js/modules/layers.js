/* layers.js — module 'layers'. Owned by the layers team. ONE file.
   Responsibility (per CONTRACT.md):
     - The layer MODEL: state.layers (array) + state.activeLayerId.
       Layer types: raster | control | mask (inpaint) | regional | reference (IP).
     - The RIGHT layers panel UI (#layers-panel): one row per layer with
       visibility eye, type glyph, name, opacity, lock, delete. Click a row = set active.
       A [+ Layer] menu adds a layer of a chosen type.
     - Each layer maps to a Konva.Group on the canvas content layer. Panel order
       == z-order (top row = front). Group visibility/opacity/listening mirror the model.
     - Emits bus 'layers:changed' on any structural/visibility/order change.
   Talks to other modules ONLY via state/get/set + bus + api. Never edits shared files.
   The Konva stage is owned by canvasCore (may not exist yet) — degrade gracefully:
   we listen for bus 'canvas:stage' and also poll state.canvas.stage, then (re)bind groups. */
(function () {
  "use strict";
  if (!window.Serenity || !Serenity.register) return;

  Serenity.register("layers", {
    init(ctx) {
      const { state, get, set, bus, Konva, dom } = ctx;
      const ROOT = dom && dom.layersPanel;
      if (!ROOT) { console.warn("[layers] no #layers-panel mount"); return; }

      // ---- layer type metadata (glyph + human label + default props) ----
      const TYPES = {
        raster:    { glyph: "▦", label: "Raster",        title: "Raster image layer" },
        control:   { glyph: "⚙", label: "Control",       title: "ControlNet guidance layer" },
        mask:      { glyph: "◐", label: "Inpaint Mask",  title: "Inpaint mask layer" },
        regional:  { glyph: "▢", label: "Regional",      title: "Regional prompt layer" },
        reference: { glyph: "★", label: "Reference (IP)", title: "Image-prompt / IP-Adapter reference" },
      };
      const TYPE_ORDER = ["raster", "control", "mask", "regional", "reference"];

      // ---- scoped CSS (rules live under #layers-panel only) ----
      injectStyle();

      // ---- model helpers --------------------------------------------------
      let uid = 0;
      function newId(type) { uid += 1; return "ly_" + type + "_" + Date.now().toString(36) + "_" + uid; }

      function ensureModel() {
        if (!Array.isArray(state.layers)) state.layers = [];
        return state.layers;
      }

      function makeLayer(type, opts) {
        opts = opts || {};
        const meta = TYPES[type] || TYPES.raster;
        const existing = ensureModel().filter((l) => l.type === type).length;
        const name = opts.name || (meta.label + (existing ? " " + (existing + 1) : ""));
        return {
          id: opts.id || newId(type),
          type: TYPES[type] ? type : "raster",
          name: name,
          visible: opts.visible !== false,
          opacity: typeof opts.opacity === "number" ? opts.opacity : 1,
          locked: !!opts.locked,
        };
      }

      function findIndex(id) { return ensureModel().findIndex((l) => l.id === id); }
      function findLayer(id) { return ensureModel().find((l) => l.id === id) || null; }

      function emitChanged() { bus.emit("layers:changed", state.layers.slice()); }

      function setActive(id) {
        // activeLayerId is part of state -> use set() so others get change events
        set("activeLayerId", id);
      }

      function addLayer(type, opts) {
        const layer = makeLayer(type, opts);
        ensureModel();
        // newest on top (front) -> unshift so panel row order == z (top row front)
        state.layers.unshift(layer);
        setActive(layer.id);
        ensureGroups();
        render();
        emitChanged();
        return layer;
      }

      function removeLayer(id) {
        const i = findIndex(id);
        if (i < 0) return;
        const removed = state.layers.splice(i, 1)[0];
        // destroy its Konva group if present
        if (removed && groups[removed.id]) { try { groups[removed.id].destroy(); } catch (_) {} delete groups[removed.id]; }
        if (get("activeLayerId") === id) {
          const next = state.layers[Math.min(i, state.layers.length - 1)];
          setActive(next ? next.id : null);
        }
        ensureGroups();
        render();
        emitChanged();
      }

      function toggleVisible(id) {
        const l = findLayer(id); if (!l) return;
        l.visible = !l.visible;
        if (groups[id]) groups[id].visible(l.visible);
        batchDraw();
        render();
        emitChanged();
      }

      function toggleLock(id) {
        const l = findLayer(id); if (!l) return;
        l.locked = !l.locked;
        if (groups[id]) groups[id].listening(!l.locked);
        batchDraw();
        render();
        emitChanged();
      }

      function setOpacity(id, v) {
        const l = findLayer(id); if (!l) return;
        l.opacity = Math.max(0, Math.min(1, v));
        if (groups[id]) groups[id].opacity(l.opacity);
        batchDraw();
        // don't full re-render on every drag tick; just sync the % label
        const lbl = ROOT.querySelector('.ly-row[data-id="' + cssEsc(id) + '"] .ly-op-val');
        if (lbl) lbl.textContent = Math.round(l.opacity * 100) + "%";
        emitChanged();
      }

      function renameLayer(id, name) {
        const l = findLayer(id); if (!l) return;
        l.name = (name || "").trim() || l.name;
        render();
        emitChanged();
      }

      function moveLayer(id, dir) {
        const i = findIndex(id); if (i < 0) return;
        const j = i + dir;
        if (j < 0 || j >= state.layers.length) return;
        const arr = state.layers;
        const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        applyZOrder();
        render();
        emitChanged();
      }

      // ---- Konva binding (stage owned by canvasCore; may arrive later) ----
      const groups = Object.create(null); // id -> Konva.Group
      let contentLayer = null;
      let stageRef = null;

      function batchDraw() { if (contentLayer) { try { contentLayer.batchDraw(); } catch (_) {} } }

      function resolveStage() {
        const s = (state.canvas && state.canvas.stage) || stageRef;
        return s && typeof s.getLayers === "function" ? s : null;
      }

      // Pick the layer Konva groups should live on. canvasCore "owns layer
      // architecture"; we look for a content layer it exposes, else use the
      // first non-overlay Konva.Layer, else create our own dedicated layer.
      function resolveContentLayer(stage) {
        if (!stage || !Konva) return null;
        const c = state.canvas || {};
        if (c.contentLayer && typeof c.contentLayer.add === "function") return c.contentLayer;
        if (c.layer && typeof c.layer.add === "function") return c.layer;
        const ls = stage.getLayers ? stage.getLayers() : [];
        // prefer a layer explicitly named 'content'
        for (let k = 0; k < ls.length; k++) {
          const nm = ls[k].name && ls[k].name();
          if (nm && /content/i.test(nm)) return ls[k];
        }
        // otherwise the first layer that is not an overlay/transform layer
        for (let k = 0; k < ls.length; k++) {
          const nm = (ls[k].name && ls[k].name()) || "";
          if (!/overlay|transform|bg|background|grid|checker/i.test(nm)) return ls[k];
        }
        // nothing usable -> make our own
        const own = new Konva.Layer({ name: "layers-content" });
        stage.add(own);
        own.moveToBottom(); // keep below any overlay/transform layers if present
        return own;
      }

      function ensureGroups() {
        const stage = resolveStage();
        if (!stage) return false;
        stageRef = stage;
        if (!contentLayer || contentLayer.getStage() !== stage) {
          contentLayer = resolveContentLayer(stage);
          if (!contentLayer) return false;
        }
        ensureModel().forEach((l) => {
          let g = groups[l.id];
          if (!g) {
            g = new Konva.Group({
              id: "kg-" + l.id,
              name: "layer-group layer-" + l.type,
            });
            g.setAttr("serenityLayerId", l.id);
            g.setAttr("serenityLayerType", l.type);
            groups[l.id] = g;
            contentLayer.add(g);
          }
          g.visible(l.visible);
          g.opacity(l.opacity);
          g.listening(!l.locked);
        });
        // drop groups whose model is gone
        Object.keys(groups).forEach((id) => {
          if (findIndex(id) < 0) { try { groups[id].destroy(); } catch (_) {} delete groups[id]; }
        });
        applyZOrder();
        return true;
      }

      function applyZOrder() {
        if (!contentLayer) return;
        // panel order: index 0 = top/front. Konva zIndex: higher = front.
        // So reverse the array when assigning zIndex.
        const arr = ensureModel();
        for (let i = 0; i < arr.length; i++) {
          const g = groups[arr[i].id];
          if (g) { try { g.moveToTop(); } catch (_) {} } // moveToTop in array order => last unshifted ends on top
        }
        // The loop above leaves arr[last] on top; we want arr[0] on top.
        // Re-apply in reverse so arr[0] is the final moveToTop -> front.
        for (let i = arr.length - 1; i >= 0; i--) {
          const g = groups[arr[i].id];
          if (g) { try { g.moveToTop(); } catch (_) {} }
        }
        batchDraw();
      }

      // expose a tiny read-only accessor for other modules (brush/controlnet/sam)
      // to fetch the Konva.Group for a layer — via state, NOT a direct import.
      function groupFor(id) { return groups[id] || null; }
      state.canvas = state.canvas || {};
      state.canvas.layerGroup = groupFor;          // fn(id) -> Konva.Group|null
      state.canvas.activeLayerGroup = function () { // convenience for tools
        const id = get("activeLayerId");
        return id ? groupFor(id) : null;
      };

      // ---- UI render ------------------------------------------------------
      const elPanel = document.createElement("div");
      elPanel.id = "ly-root";
      elPanel.innerHTML =
        '<div class="ly-head">' +
          '<span class="ly-title">Layers</span>' +
          '<div class="ly-add-wrap">' +
            '<button class="ly-add btn" type="button" aria-haspopup="true" aria-expanded="false">+ Layer</button>' +
            '<div class="ly-menu" role="menu" hidden></div>' +
          '</div>' +
        '</div>' +
        '<div class="ly-list" role="list"></div>' +
        '<div class="ly-empty" hidden>No layers — add one with “+ Layer”.</div>';
      ROOT.appendChild(elPanel);

      const elList = elPanel.querySelector(".ly-list");
      const elEmpty = elPanel.querySelector(".ly-empty");
      const elAddBtn = elPanel.querySelector(".ly-add");
      const elMenu = elPanel.querySelector(".ly-menu");

      // build the +Layer menu once
      TYPE_ORDER.forEach((type) => {
        const m = TYPES[type];
        const item = document.createElement("button");
        item.type = "button";
        item.className = "ly-menu-item";
        item.setAttribute("role", "menuitem");
        item.dataset.type = type;
        item.innerHTML = '<span class="ly-glyph">' + m.glyph + "</span><span>" + m.label + "</span>";
        item.addEventListener("click", () => { closeMenu(); addLayer(type); });
        elMenu.appendChild(item);
      });

      function openMenu() { elMenu.hidden = false; elAddBtn.setAttribute("aria-expanded", "true"); }
      function closeMenu() { elMenu.hidden = true; elAddBtn.setAttribute("aria-expanded", "false"); }
      elAddBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (elMenu.hidden) openMenu(); else closeMenu();
      });
      document.addEventListener("click", (e) => { if (!elMenu.contains(e.target) && e.target !== elAddBtn) closeMenu(); });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

      function render() {
        const arr = ensureModel();
        const activeId = get("activeLayerId");
        elEmpty.hidden = arr.length > 0;
        elList.innerHTML = "";
        arr.forEach((l, idx) => {
          const meta = TYPES[l.type] || TYPES.raster;
          const row = document.createElement("div");
          row.className = "ly-row" + (l.id === activeId ? " is-active" : "") + (l.locked ? " is-locked" : "");
          row.dataset.id = l.id;
          row.setAttribute("role", "listitem");
          row.title = meta.title;

          row.innerHTML =
            '<button class="ly-eye ly-ico" type="button" title="Toggle visibility" aria-label="Toggle visibility">' +
              (l.visible ? "👁" : "🚫") + "</button>" +
            '<span class="ly-glyph" title="' + esc(meta.label) + '">' + meta.glyph + "</span>" +
            '<span class="ly-name" tabindex="0" title="Double-click to rename">' + esc(l.name) + "</span>" +
            '<span class="ly-spacer"></span>' +
            '<span class="ly-op">' +
              '<input class="ly-op-range" type="range" min="0" max="100" step="1" value="' + Math.round(l.opacity * 100) + '" title="Opacity" aria-label="Opacity">' +
              '<span class="ly-op-val">' + Math.round(l.opacity * 100) + "%</span>" +
            "</span>" +
            '<button class="ly-up ly-ico" type="button" title="Move up" aria-label="Move up"' + (idx === 0 ? " disabled" : "") + ">▲</button>" +
            '<button class="ly-dn ly-ico" type="button" title="Move down" aria-label="Move down"' + (idx === arr.length - 1 ? " disabled" : "") + ">▼</button>" +
            '<button class="ly-lock ly-ico" type="button" title="Lock layer" aria-label="Lock layer">' + (l.locked ? "🔒" : "🔓") + "</button>" +
            '<button class="ly-del ly-ico" type="button" title="Delete layer" aria-label="Delete layer">✕</button>';

          // row click (not on a control) => set active
          row.addEventListener("click", (e) => {
            if (e.target.closest("button") || e.target.closest("input") || e.target.classList.contains("ly-name")) return;
            setActive(l.id);
          });

          row.querySelector(".ly-eye").addEventListener("click", (e) => { e.stopPropagation(); toggleVisible(l.id); });
          row.querySelector(".ly-lock").addEventListener("click", (e) => { e.stopPropagation(); toggleLock(l.id); });
          row.querySelector(".ly-del").addEventListener("click", (e) => { e.stopPropagation(); removeLayer(l.id); });
          row.querySelector(".ly-up").addEventListener("click", (e) => { e.stopPropagation(); moveLayer(l.id, -1); });
          row.querySelector(".ly-dn").addEventListener("click", (e) => { e.stopPropagation(); moveLayer(l.id, +1); });

          const range = row.querySelector(".ly-op-range");
          range.addEventListener("input", (e) => { e.stopPropagation(); setOpacity(l.id, (+e.target.value) / 100); });
          range.addEventListener("click", (e) => e.stopPropagation());

          // double-click name -> inline rename
          const nameEl = row.querySelector(".ly-name");
          nameEl.addEventListener("dblclick", (e) => { e.stopPropagation(); startRename(l, nameEl); });

          elList.appendChild(row);
        });
      }

      function startRename(layer, nameEl) {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "ly-name-edit";
        input.value = layer.name;
        nameEl.replaceWith(input);
        input.focus(); input.select();
        const commit = () => { renameLayer(layer.id, input.value); };
        input.addEventListener("blur", commit);
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") { e.preventDefault(); input.blur(); }
          else if (e.key === "Escape") { e.preventDefault(); render(); }
        });
      }

      // ---- wire to stage lifecycle ---------------------------------------
      bus.on("canvas:stage", (s) => { stageRef = s || stageRef; if (ensureGroups()) batchDraw(); });
      bus.on("app:ready", () => { ensureGroups(); render(); });
      // if active layer changes from elsewhere, reflect highlight
      bus.on("change:activeLayerId", () => render());

      // poll briefly for a late stage (canvasCore may init after us, or expose
      // the stage on state without firing the bus). Cheap, bounded, self-stopping.
      let tries = 0;
      const poll = setInterval(() => {
        tries += 1;
        if (resolveStage()) { ensureGroups(); batchDraw(); clearInterval(poll); }
        else if (tries > 40) clearInterval(poll); // ~10s cap
      }, 250);

      // ---- seed default layers (one Raster + one Inpaint Mask) -----------
      // Raster is the base (bottom), Mask on top. unshift puts newest on top,
      // so add Raster first then Mask -> Mask ends on top, Raster active-last.
      if (ensureModel().length === 0) {
        const raster = makeLayer("raster", { name: "Raster" });
        const mask = makeLayer("mask", { name: "Inpaint Mask" });
        state.layers = [mask, raster]; // index0 (top) = mask, index1 = raster
        setActive(raster.id);          // raster active by default (the paint surface)
      }

      ensureGroups();
      render();
      emitChanged();

      // ---------------------------------------------------------------------
      function injectStyle() {
        if (document.getElementById("style-layers")) return;
        const css =
          "#layers-panel{padding:0}" +
          "#ly-root{display:flex;flex-direction:column;height:100%;font-size:13px}" +
          "#ly-root .ly-head{display:flex;align-items:center;justify-content:space-between;" +
            "padding:10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}" +
          "#ly-root .ly-title{font-weight:600;letter-spacing:.3px}" +
          "#ly-root .ly-add-wrap{position:relative}" +
          "#ly-root .ly-add{padding:4px 10px;font-size:12px}" +
          "#ly-root .ly-menu{position:absolute;right:0;top:calc(100% + 4px);min-width:170px;" +
            "background:var(--panel2);border:1px solid var(--line);border-radius:var(--radius);" +
            "box-shadow:0 8px 24px rgba(0,0,0,.4);padding:4px;z-index:30}" +
          "#ly-root .ly-menu[hidden]{display:none}" +
          "#ly-root .ly-menu-item{display:flex;align-items:center;gap:8px;width:100%;text-align:left;" +
            "background:none;border:0;color:var(--text);padding:7px 9px;border-radius:6px;cursor:pointer;font-size:13px}" +
          "#ly-root .ly-menu-item:hover{background:var(--accent2);color:#fff}" +
          "#ly-root .ly-menu-item .ly-glyph{width:18px;text-align:center;color:var(--accent)}" +
          "#ly-root .ly-list{flex:1;overflow-y:auto;padding:6px}" +
          "#ly-root .ly-empty{color:var(--muted);padding:14px;text-align:center;font-size:12px}" +
          "#ly-root .ly-empty[hidden]{display:none}" +
          "#ly-root .ly-row{display:flex;align-items:center;gap:6px;padding:6px;margin-bottom:4px;" +
            "border:1px solid var(--line);border-radius:6px;background:var(--panel2);cursor:pointer;user-select:none}" +
          "#ly-root .ly-row:hover{border-color:var(--accent)}" +
          "#ly-root .ly-row.is-active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);background:#262a36}" +
          "#ly-root .ly-row.is-locked .ly-name{opacity:.6;font-style:italic}" +
          "#ly-root .ly-glyph{width:18px;text-align:center;color:var(--accent);flex:0 0 auto}" +
          "#ly-root .ly-name{flex:0 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:96px}" +
          "#ly-root .ly-name-edit{flex:1;min-width:60px;padding:2px 4px;font-size:12px}" +
          "#ly-root .ly-spacer{flex:1 1 auto}" +
          "#ly-root .ly-op{display:flex;align-items:center;gap:4px;flex:0 0 auto}" +
          "#ly-root .ly-op-range{width:56px;flex:0 0 auto}" +
          "#ly-root .ly-op-val{color:var(--muted);font-size:10px;width:30px;text-align:right}" +
          "#ly-root .ly-ico{background:none;border:0;color:var(--muted);cursor:pointer;font-size:13px;" +
            "padding:2px;line-height:1;border-radius:4px;flex:0 0 auto}" +
          "#ly-root .ly-ico:hover:not(:disabled){color:var(--text);background:var(--panel)}" +
          "#ly-root .ly-ico:disabled{opacity:.3;cursor:default}" +
          "#ly-root .ly-del:hover{color:var(--danger)}";
        const tag = document.createElement("style");
        tag.id = "style-layers";
        tag.textContent = css;
        document.head.appendChild(tag);
      }

      function esc(s) {
        return String(s == null ? "" : s)
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
      }
      function cssEsc(s) { return String(s).replace(/["\\]/g, "\\$&"); }

      console.info("[layers] init ok (", state.layers.length, "layers )");
    },
  });
})();
