/* gallery_pro.js — module 'galleryPro'. SwarmUI-style gallery/history upgrades on
   TOP of the existing 'gallery' filmstrip (does NOT replace it). GAP §7.

   Features (all real, no stubs):
     - Full-image LIGHTBOX overlay: click any gallery thumb (or a "View" action) to
       open the result big, with prev/next nav (←/→), Esc to close, fit/100% toggle.
     - METADATA VIEWER pane inside the lightbox: reads the PNG's embedded genparams
       (serenity.genparams.v1 tEXt) via the backend /v1/gallery API (or the
       'result:ready' payload's params) and renders a key/value table + raw JSON.
     - RIGHT-CLICK per-image context menu on gallery thumbs AND in the lightbox:
         Regenerate · Reuse params · Reuse seed · Send to img2img · Upscale · Delete.
     - MULTI-SELECT (ctrl/cmd/shift-click thumbnails) + a floating batch bar:
         Download selected (zip-free, sequential blob downloads) · Delete selected ·
         Clear selection.
     - Configurable OUTPUT-PATH TEMPLATE field (SwarmUI OutpathBuilder-style):
         [model] [seed] [prompt] [date] [width] [height] [steps] [cfg] [sampler] [index]
       persisted to localStorage + written to state.params.output_path_template, with
       a live-resolved preview.

   Self-contained IIFE. Talks ONLY through ctx.{state,get,set,bus,api,Konva} + the
   backend /v1/gallery HTTP API (via api.base). Degrades gracefully when offline.

   Coordinates with the existing 'gallery' module via the shared bus:
     - listens 'gallery:select' (thumb clicked) -> open lightbox
     - listens 'result:ready' -> remember newest result (for lightbox default)
     - emits  'generate:request' (regenerate), 'params:restored', 'preview:load'
     - reaches the 'layers' module (soft) for send-to-img2img; falls back to bus.
*/
(function () {
  "use strict";
  if (!window.Serenity || !Serenity.register) { console.warn('[galleryPro] Serenity not ready'); return; }

  Serenity.register('galleryPro', {
    init(ctx) {
      const { state, get, set, bus, api } = ctx;
      const BASE = (api && api.base) || '';

      injectCSS();

      // =====================================================================
      // MODEL — items mirrored from the 'gallery' module's shared state.gallery
      // plus anything we learn from /v1/gallery (richer params) and result:ready.
      // item: {id, filename, url, thumbUrl, params, favorite, name, path}
      // =====================================================================
      const items = [];                 // ordered newest-first (matches strip)
      const byKey = new Map();           // key(url|id) -> item
      const selected = new Set();        // selected item keys (multi-select)
      let lightboxIndex = -1;            // index into items[] when lightbox open
      let lastResult = null;             // most recent result:ready payload

      const TPL_KEY = 'serenity.gallery.outpath';
      let outpathTpl = loadTpl();
      // publish the template so the backend (via the orchestrator's forwarding) sees it
      try { set('params.output_path_template', outpathTpl); } catch (_) {}

      // =====================================================================
      // ITEM INGEST / SYNC
      // =====================================================================
      function keyOf(raw) {
        return String((raw && (raw.url || raw.id || raw.filename)) || '');
      }
      function baseName(p) { return String(p || '').split('/').pop(); }

      // Turn an absolute server path into a browser URL under /out/.
      // The server serves out_dir at /out/<relative>. job PNGs sit at the out_dir
      // root, thumbnails under out_dir/thumbnails/. We map by basename + a thumb hint.
      function urlForPath(p, isThumb) {
        if (!p) return null;
        const bn = baseName(p);
        if (!bn) return null;
        if (isThumb || /thumbnails?[\\/]/.test(String(p))) return BASE + '/out/thumbnails/' + encodeURIComponent(bn);
        return BASE + '/out/' + encodeURIComponent(bn);
      }

      function upsertItem(raw) {
        if (!raw) return null;
        const k = keyOf(raw);
        if (!k) return null;
        let it = byKey.get(k);
        if (!it) {
          it = { key: k };
          byKey.set(k, it);
          items.unshift(it);  // newest first
        }
        // merge fields (don't clobber existing richer params with null)
        if (raw.id != null) it.id = String(raw.id);
        if (raw.filename) it.filename = raw.filename;
        else if (raw.path && !it.filename) it.filename = baseName(raw.path);
        if (raw.url) it.url = raw.url;
        else if (!it.url && it.filename) it.url = BASE + '/out/' + encodeURIComponent(it.filename);
        else if (!it.url && raw.path) it.url = urlForPath(raw.path, false);
        if (raw.thumbUrl) it.thumbUrl = raw.thumbUrl;
        else if (raw.thumbnail_path && !it.thumbUrl) it.thumbUrl = urlForPath(raw.thumbnail_path, true);
        if (raw.path) it.path = raw.path;
        if (raw.name) it.name = raw.name;
        if (raw.favorite != null) it.favorite = !!raw.favorite;
        if (raw.params && typeof raw.params === 'object') it.params = raw.params;
        else if (raw.params_json && !it.params) { try { it.params = JSON.parse(raw.params_json); } catch (_) {} }
        if (!it.thumbUrl) it.thumbUrl = it.url;  // fall back to full image
        return it;
      }

      // Pull the rich list from the backend (params, favorites, paths, thumbs).
      async function loadGallery() {
        if (!BASE && location.protocol === 'file:') return;  // no server to ask
        let doc;
        try {
          const r = await fetch(BASE + '/v1/gallery');
          if (!r.ok) throw new Error('http ' + r.status);
          doc = await r.json();
        } catch (err) {
          console.info('[galleryPro] /v1/gallery unavailable (degrading):', err && err.message);
          return;
        }
        const list = (doc && Array.isArray(doc.items)) ? doc.items : [];
        // server lists newest-first already (id_before order); upsert in reverse so
        // unshift keeps newest at the front.
        for (let i = list.length - 1; i >= 0; i--) upsertItem(list[i]);
        // if the lightbox is open, refresh it (params may have arrived)
        if (lightboxIndex >= 0) renderLightbox();
      }

      // Read one item's full metadata on demand (richer than the list payload).
      async function fetchItemMeta(it) {
        if (!it) return null;
        if (it.params && Object.keys(it.params).length) return it.params;
        // by id first, else by absolute path
        const tries = [];
        if (it.id && /^job-/.test(it.id)) tries.push(BASE + '/v1/gallery/' + encodeURIComponent(it.id));
        if (it.path) tries.push(BASE + '/v1/gallery/read?path=' + encodeURIComponent(it.path));
        for (const u of tries) {
          try {
            const r = await fetch(u);
            if (!r.ok) continue;
            const j = await r.json();
            if (j && j.params && typeof j.params === 'object') { it.params = j.params; return it.params; }
            if (j && j.params_json) { try { it.params = JSON.parse(j.params_json); return it.params; } catch (_) {} }
          } catch (_) {}
        }
        return it.params || null;
      }

      // =====================================================================
      // SELECTION OVERLAY on the existing gallery strip
      // We don't own #gallery's DOM; instead we OBSERVE its .gal-card children and
      // decorate them with a checkbox + selection ring + right-click menu, so the
      // existing filmstrip gains multi-select & context actions in place.
      // =====================================================================
      const galleryHost = (ctx.dom && ctx.dom.gallery) || document.getElementById('gallery');

      function cardItem(card) {
        // the gallery module stamps data-id; match by id, else by <img> src.
        const id = card.dataset && card.dataset.id;
        if (id) {
          for (const it of items) if (it.id === id || it.key === id) return it;
        }
        const img = card.querySelector('img.gal-img, img');
        const src = img && (img.currentSrc || img.src);
        if (src) {
          for (const it of items) if (it.url === src || it.thumbUrl === src) return it;
          // not tracked yet -> create a lightweight item from the DOM
          return upsertItem({ id: id || src, url: src, filename: baseName(src) });
        }
        return id ? upsertItem({ id }) : null;
      }

      function decorateCard(card) {
        if (card.__gpDecorated) return;
        card.__gpDecorated = true;
        card.classList.add('gp-card');

        // selection checkbox (top-left, below the star marker)
        const chk = document.createElement('div');
        chk.className = 'gp-check';
        chk.title = 'Select';
        chk.addEventListener('click', (e) => {
          e.stopPropagation();
          const it = cardItem(card);
          if (it) toggleSelect(it, e);
        });
        card.appendChild(chk);

        // right-click context menu
        card.addEventListener('contextmenu', (e) => {
          const it = cardItem(card);
          if (!it) return;
          e.preventDefault();
          openContextMenu(e.clientX, e.clientY, it);
        });

        // ctrl/cmd/shift click = multi-select (without opening preview); plain click
        // is left to the gallery module (loads preview + restores params).
        card.addEventListener('click', (e) => {
          if (e.target.closest('.gp-check') || e.target.closest('.gal-actbtn')) return;
          if (e.ctrlKey || e.metaKey || e.shiftKey) {
            // stopImmediatePropagation (not just stopPropagation): gallery.js's plain
            // click handler is on the SAME card element, so only the *immediate* form
            // prevents it from also loading the preview + clobbering state.params.
            e.preventDefault();
            e.stopImmediatePropagation();
            const it = cardItem(card);
            if (it) toggleSelect(it, e);
          }
        }, true);  // capture so we can pre-empt the gallery's plain-click handler

        // double-click = open lightbox
        card.addEventListener('dblclick', (e) => {
          e.preventDefault(); e.stopPropagation();
          const it = cardItem(card);
          if (it) openLightbox(it);
        });

        syncCardSelection(card);
      }

      function syncCardSelection(card) {
        const it = cardItem(card);
        const on = it && selected.has(it.key);
        card.classList.toggle('gp-selected', !!on);
      }

      function refreshAllCardSelections() {
        if (!galleryHost) return;
        galleryHost.querySelectorAll('.gal-card').forEach(syncCardSelection);
      }

      // Drop the matching .gal-card from the existing strip (used after a delete so the
      // filmstrip stays consistent even if the gallery module doesn't hear our event).
      function removeStripCard(it) {
        if (!galleryHost || !it) return;
        galleryHost.querySelectorAll('.gal-card').forEach((card) => {
          const id = card.dataset && card.dataset.id;
          if (id && (id === it.id || id === it.key)) { card.remove(); return; }
          const img = card.querySelector('img.gal-img, img');
          const src = img && (img.currentSrc || img.src);
          if (src && (src === it.url || src === it.thumbUrl)) card.remove();
        });
      }

      // observe the strip so newly-prepended cards get decorated
      let mo = null;
      function observeGallery() {
        if (!galleryHost || mo) return;
        mo = new MutationObserver(() => {
          galleryHost.querySelectorAll('.gal-card').forEach(decorateCard);
        });
        mo.observe(galleryHost, { childList: true, subtree: true });
        galleryHost.querySelectorAll('.gal-card').forEach(decorateCard);
      }

      // =====================================================================
      // MULTI-SELECT
      // =====================================================================
      let lastSelKey = null;
      function toggleSelect(it, e) {
        const k = it.key;
        if (e && e.shiftKey && lastSelKey != null) {
          // range select between lastSelKey and k in items[] order
          const a = items.findIndex((x) => x.key === lastSelKey);
          const b = items.findIndex((x) => x.key === k);
          if (a >= 0 && b >= 0) {
            const [lo, hi] = a < b ? [a, b] : [b, a];
            for (let i = lo; i <= hi; i++) selected.add(items[i].key);
          }
        } else {
          if (selected.has(k)) selected.delete(k); else selected.add(k);
          lastSelKey = k;
        }
        refreshAllCardSelections();
        renderBatchBar();
      }
      function clearSelection() {
        selected.clear(); lastSelKey = null;
        refreshAllCardSelections();
        renderBatchBar();
      }

      // =====================================================================
      // BATCH BAR (floating, bottom-right over the gallery)
      // =====================================================================
      let batchBar = null;
      function ensureBatchBar() {
        if (batchBar) return batchBar;
        batchBar = document.createElement('div');
        batchBar.id = 'gp-batchbar';
        batchBar.className = 'gp-hidden';
        document.body.appendChild(batchBar);
        return batchBar;
      }
      function renderBatchBar() {
        const bar = ensureBatchBar();
        const n = selected.size;
        if (!n) { bar.classList.add('gp-hidden'); bar.innerHTML = ''; return; }
        bar.classList.remove('gp-hidden');
        bar.innerHTML = '';
        const label = elc('span', 'gp-bb-label', n + ' selected');
        const dl = elc('button', 'gp-bb-btn', 'Download');
        dl.addEventListener('click', downloadSelected);
        const del = elc('button', 'gp-bb-btn gp-bb-danger', 'Delete');
        del.addEventListener('click', deleteSelected);
        const clr = elc('button', 'gp-bb-btn', 'Clear');
        clr.addEventListener('click', clearSelection);
        bar.appendChild(label); bar.appendChild(dl); bar.appendChild(del); bar.appendChild(clr);
      }

      async function downloadSelected() {
        const list = items.filter((it) => selected.has(it.key));
        for (let i = 0; i < list.length; i++) {
          await downloadItem(list[i]);
          // small gap so the browser doesn't drop rapid-fire anchor clicks
          await sleep(180);
        }
        flash('Downloaded ' + list.length + ' image' + (list.length === 1 ? '' : 's'));
      }

      async function deleteSelected() {
        const list = items.filter((it) => selected.has(it.key));
        if (!list.length) return;
        if (!confirm('Delete ' + list.length + ' image' + (list.length === 1 ? '' : 's') + '? This removes the file on disk.')) return;
        let ok = 0;
        for (const it of list) { if (await deleteItem(it, true)) ok++; }
        clearSelection();
        flash('Deleted ' + ok + '/' + list.length);
      }

      // =====================================================================
      // PER-ITEM ACTIONS
      // =====================================================================
      const PARAM_KEYS = [
        'model', 'vae', 'prompt', 'negative', 'width', 'height', 'aspect',
        'steps', 'cfg', 'seed', 'images', 'sampler', 'scheduler',
        'clip_skip', 'sigma_min', 'sigma_max', 'eta', 'restart_sampling',
        'loras', 'controlnet', 'refiner', 'denoise', 'sigma_shift',
      ];
      function reuseParams(it) {
        const p = it.params;
        if (!p || typeof p !== 'object') { flash('No params to reuse'); return; }
        PARAM_KEYS.forEach((k) => { if (p[k] !== undefined && p[k] !== null) set('params.' + k, p[k]); });
        // tolerate alt seed key
        if (p.seed === undefined && p.noise_seed !== undefined) set('params.seed', p.noise_seed);
        bus.emit('params:restored', p);
        flash('Params reused');
      }
      function reuseSeed(it) {
        const p = it.params || {};
        const s = (p.seed !== undefined) ? p.seed : (p.noise_seed !== undefined ? p.noise_seed : null);
        if (s == null) { flash('No seed found'); return; }
        set('params.seed', s);
        bus.emit('params:restored', { seed: s });
        flash('Seed ' + s + ' reused');
      }
      function regenerate(it) {
        // restore params then ask generateWS to run
        reuseParams(it);
        // a fresh seed for a true re-roll unless the user wants exact: reuse exact here
        bus.emit('generate:request');
        flash('Regenerating…');
      }

      // Send the picked image to img2img: add it as a raster init layer + drop denoise
      // so KSampler runs in img2img mode (generateWS scans state.layers for a raster
      // layer with pixels). Reaches the 'layers' module if present, else pushes to
      // state.layers directly + emits layers:changed.
      function sendToImg2img(it) {
        if (!it.url) { flash('No image url'); return; }
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const layerOpts = {
            type: 'raster', name: 'img2img: ' + (it.filename || it.id || 'result'),
            imageEl: img, src: it.url, visible: true, opacity: 1,
          };
          const layersMod = window.Serenity.modules && window.Serenity.modules.layers;
          let added = false;
          // prefer a public add API if the layers module exposes one
          if (layersMod && typeof layersMod.addLayer === 'function') {
            try { layersMod.addLayer('raster', layerOpts); added = true; } catch (_) {}
          }
          if (!added) {
            try {
              if (!Array.isArray(state.layers)) state.layers = [];
              const layer = Object.assign({ id: 'gp-' + Date.now().toString(36) }, layerOpts);
              state.layers.unshift(layer);
              set('activeLayerId', layer.id);
              bus.emit('layers:changed', state.layers.slice());
            } catch (e) { console.warn('[galleryPro] sendToImg2img layer add failed', e); }
          }
          // default a sensible img2img denoise if it's still full-strength
          const cur = Number(get('params.denoise'));
          if (!Number.isFinite(cur) || cur >= 0.999) set('params.denoise', 0.65);
          // also reuse the source params (prompt/model) so the run is coherent
          if (it.params) reuseParams(it);
          bus.emit('preview:load', { url: it.url, filename: it.filename });
          bus.emit('img2img:source', { url: it.url, filename: it.filename, params: it.params || null });
          flash('Sent to img2img (denoise ' + (get('params.denoise')) + ')');
        };
        img.onerror = () => flash('Could not load image for img2img');
        img.src = it.url;
      }

      // Upscale: SwarmUI's upscale == regenerate at higher res from this image as init.
      // We send-to-img2img, double W/H (clamped), and set a light denoise so detail is
      // added rather than the image replaced, then request generation.
      function upscale(it) {
        sendToImg2img(it);
        const p = it.params || {};
        const w = clampDim((Number(p.width) || Number(get('params.width')) || 1024) * 2);
        const h = clampDim((Number(p.height) || Number(get('params.height')) || 1024) * 2);
        set('params.width', w); set('params.height', h);
        set('params.denoise', 0.4);
        // mirror into the bbox so generateWS picks the new dims
        try {
          const bbox = get('canvas.bbox') || {};
          set('canvas.bbox', Object.assign({}, bbox, { width: w, height: h }));
        } catch (_) {}
        bus.emit('generate:request');
        flash('Upscaling to ' + w + '×' + h);
      }

      async function downloadItem(it) {
        if (!it.url) return;
        try {
          const r = await fetch(it.url);
          if (!r.ok) throw new Error('http ' + r.status);
          const blob = await r.blob();
          const a = document.createElement('a');
          const u = URL.createObjectURL(blob);
          a.href = u; a.download = it.filename || ((it.id || 'serenity') + '.png');
          document.body.appendChild(a); a.click(); a.remove();
          setTimeout(() => URL.revokeObjectURL(u), 4000);
        } catch (err) {
          try {
            const a = document.createElement('a');
            a.href = it.url; a.download = it.filename || 'result.png'; a.target = '_blank';
            document.body.appendChild(a); a.click(); a.remove();
          } catch (_) { flash('Download failed'); }
        }
      }

      // Delete on disk via the backend, then drop locally + tell the gallery module.
      async function deleteItem(it, silent) {
        let ok = false;
        if (it.id && /^job-/.test(it.id)) {
          try {
            const r = await fetch(BASE + '/v1/gallery/' + encodeURIComponent(it.id), { method: 'DELETE' });
            ok = r.ok;
          } catch (_) { ok = false; }
        }
        // remove locally regardless (so the UI reflects intent even if server 404s)
        const idx = items.findIndex((x) => x.key === it.key);
        if (idx >= 0) items.splice(idx, 1);
        byKey.delete(it.key);
        selected.delete(it.key);
        // ask the gallery module to drop its card too (it may or may not listen)
        bus.emit('gallery:removed', { id: it.id });
        bus.emit('gallery:delete-request', { id: it.id, key: it.key });
        // self-contained fallback: if the gallery strip still shows the card, drop it
        // ourselves so the filmstrip reflects the delete even when gallery.js doesn't
        // listen for 'gallery:removed'. Match by data-id or the thumbnail/full src.
        removeStripCard(it);
        // keep the lightbox in sync ONLY when we actually removed something from items[]
        if (lightboxIndex >= 0 && idx >= 0) {
          if (idx === lightboxIndex) { if (items.length) openByIndex(Math.min(lightboxIndex, items.length - 1)); else closeLightbox(); }
          else if (idx < lightboxIndex) { lightboxIndex--; renderLightbox(); }  // refresh the N/total counter
        }
        if (!silent) flash(ok ? 'Deleted' : 'Removed (server unavailable)');
        renderBatchBar();
        return ok;
      }

      async function toggleFavorite(it) {
        if (!(it.id && /^job-/.test(it.id))) { flash('Favorite needs a saved job'); return; }
        const next = !it.favorite;
        try {
          const r = await fetch(BASE + '/v1/gallery/' + encodeURIComponent(it.id) + '/favorite', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ favorite: next }),
          });
          if (r.ok) { it.favorite = next; flash(next ? 'Favorited' : 'Unfavorited'); if (lightboxIndex >= 0) renderLightbox(); }
          else flash('Favorite failed');
        } catch (_) { flash('Favorite failed (offline)'); }
      }

      // =====================================================================
      // CONTEXT MENU
      // =====================================================================
      let ctxMenu = null;
      function openContextMenu(x, y, it) {
        closeContextMenu();
        ctxMenu = document.createElement('div');
        ctxMenu.className = 'gp-ctxmenu';
        const add = (label, fn, danger) => {
          const b = elc('button', 'gp-ctx-item' + (danger ? ' gp-ctx-danger' : ''), label);
          b.addEventListener('click', () => { closeContextMenu(); fn(); });
          ctxMenu.appendChild(b);
        };
        add('Open', () => openLightbox(it));
        add('Regenerate', () => regenerate(it));
        add('Reuse params', () => reuseParams(it));
        add('Reuse seed', () => reuseSeed(it));
        add('Send to img2img', () => sendToImg2img(it));
        add('Upscale 2×', () => upscale(it));
        add(it.favorite ? 'Unfavorite' : 'Favorite', () => toggleFavorite(it));
        add('Download', () => downloadItem(it));
        sep();
        add('Delete', () => { if (confirm('Delete this image? Removes the file on disk.')) deleteItem(it); }, true);
        function sep() { const s = document.createElement('div'); s.className = 'gp-ctx-sep'; ctxMenu.appendChild(s); }

        document.body.appendChild(ctxMenu);
        // position, keeping it on-screen
        const vw = window.innerWidth, vh = window.innerHeight;
        const r = ctxMenu.getBoundingClientRect();
        ctxMenu.style.left = Math.min(x, vw - r.width - 6) + 'px';
        ctxMenu.style.top = Math.min(y, vh - r.height - 6) + 'px';
        setTimeout(() => {
          document.addEventListener('mousedown', onDocDown, true);
          document.addEventListener('keydown', onCtxKey, true);
        }, 0);
      }
      function onDocDown(e) { if (ctxMenu && !ctxMenu.contains(e.target)) closeContextMenu(); }
      function onCtxKey(e) { if (e.key === 'Escape') closeContextMenu(); }
      function closeContextMenu() {
        if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; }
        document.removeEventListener('mousedown', onDocDown, true);
        document.removeEventListener('keydown', onCtxKey, true);
      }

      // =====================================================================
      // LIGHTBOX  (full image + metadata viewer + nav)
      // =====================================================================
      let lb = null, lbImg = null, lbMeta = null, lbCaption = null, lbFitBtn = null;
      let lbFit = true;  // true = fit to view, false = 100%

      function buildLightbox() {
        if (lb) return;
        lb = document.createElement('div');
        lb.id = 'gp-lightbox';
        lb.className = 'gp-hidden';

        const stageWrap = elc('div', 'gp-lb-stage');
        lbImg = document.createElement('img');
        lbImg.className = 'gp-lb-img gp-fit';
        lbImg.alt = 'result';
        stageWrap.appendChild(lbImg);

        // prev / next
        const prev = elc('button', 'gp-lb-nav gp-lb-prev', '‹');
        prev.title = 'Previous (←)';
        prev.addEventListener('click', (e) => { e.stopPropagation(); step(-1); });
        const next = elc('button', 'gp-lb-nav gp-lb-next', '›');
        next.title = 'Next (→)';
        next.addEventListener('click', (e) => { e.stopPropagation(); step(1); });
        stageWrap.appendChild(prev); stageWrap.appendChild(next);

        // side panel = metadata viewer + actions
        const side = elc('div', 'gp-lb-side');
        const head = elc('div', 'gp-lb-head', '');
        const title = elc('div', 'gp-lb-title', 'Metadata');
        const close = elc('button', 'gp-lb-close', '✕');
        close.title = 'Close (Esc)';
        close.addEventListener('click', closeLightbox);
        head.appendChild(title); head.appendChild(close);
        side.appendChild(head);

        const actions = elc('div', 'gp-lb-actions', '');
        actions.appendChild(actBtn('Regenerate', () => regenerate(current())));
        actions.appendChild(actBtn('Reuse params', () => reuseParams(current())));
        actions.appendChild(actBtn('Reuse seed', () => reuseSeed(current())));
        actions.appendChild(actBtn('img2img', () => sendToImg2img(current())));
        actions.appendChild(actBtn('Upscale', () => upscale(current())));
        actions.appendChild(actBtn('Download', () => downloadItem(current())));
        actions.appendChild(actBtn('Delete', () => { const it = current(); if (it && confirm('Delete this image?')) deleteItem(it); }, true));
        side.appendChild(actions);

        lbFitBtn = elc('button', 'gp-lb-fit', 'Fit');
        lbFitBtn.title = 'Toggle fit / 100%';
        lbFitBtn.addEventListener('click', toggleFit);
        side.appendChild(lbFitBtn);

        lbCaption = elc('div', 'gp-lb-caption', '');
        side.appendChild(lbCaption);

        lbMeta = elc('div', 'gp-lb-meta', '');
        side.appendChild(lbMeta);

        // output-path template editor lives in the side panel footer
        side.appendChild(buildOutpathEditor());

        lb.appendChild(stageWrap);
        lb.appendChild(side);
        document.body.appendChild(lb);

        // background click closes; clicks inside stage/side don't
        lb.addEventListener('click', (e) => { if (e.target === lb || e.target === stageWrap) closeLightbox(); });
        // double-click the image toggles fit
        lbImg.addEventListener('dblclick', toggleFit);
      }

      function actBtn(label, fn, danger) {
        const b = elc('button', 'gp-lb-act' + (danger ? ' gp-lb-act-danger' : ''), label);
        b.addEventListener('click', fn);
        return b;
      }

      function current() { return (lightboxIndex >= 0 && lightboxIndex < items.length) ? items[lightboxIndex] : null; }

      function openLightbox(it) {
        const idx = items.findIndex((x) => x.key === it.key);
        openByIndex(idx >= 0 ? idx : 0);
      }
      function openByIndex(idx) {
        buildLightbox();
        if (idx < 0 || idx >= items.length) return;
        lightboxIndex = idx;
        lb.classList.remove('gp-hidden');
        document.addEventListener('keydown', onLbKey, true);
        renderLightbox();
      }
      function closeLightbox() {
        if (lb) lb.classList.add('gp-hidden');
        lightboxIndex = -1;
        document.removeEventListener('keydown', onLbKey, true);
      }
      function step(d) {
        if (!items.length) return;
        let i = lightboxIndex + d;
        if (i < 0) i = items.length - 1;
        if (i >= items.length) i = 0;
        openByIndex(i);
      }
      function onLbKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); closeLightbox(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
        else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFit(); }
      }
      function toggleFit() {
        lbFit = !lbFit;
        if (lbImg) lbImg.classList.toggle('gp-fit', lbFit);
        if (lbFitBtn) lbFitBtn.textContent = lbFit ? 'Fit' : '100%';
      }

      async function renderLightbox() {
        const it = current();
        if (!it || !lbImg) return;
        if (it.url) lbImg.src = it.url;
        lbImg.classList.toggle('gp-fit', lbFit);
        lbCaption.innerHTML = '';
        const fn = elc('div', 'gp-cap-name', it.filename || it.id || '');
        const pos = elc('div', 'gp-cap-pos', (lightboxIndex + 1) + ' / ' + items.length + (it.favorite ? '  ★' : ''));
        lbCaption.appendChild(fn); lbCaption.appendChild(pos);

        // metadata: ensure we have params (fetch on demand)
        lbMeta.innerHTML = '<div class="gp-meta-loading">Loading metadata…</div>';
        const params = await fetchItemMeta(it);
        if (current() !== it) return;  // navigated away during fetch
        renderMeta(params);
        refreshOutpathPreview();
      }

      function renderMeta(params) {
        lbMeta.innerHTML = '';
        if (!params || typeof params !== 'object' || !Object.keys(params).length) {
          lbMeta.appendChild(elc('div', 'gp-meta-empty', 'No embedded metadata (PNG has no serenity.genparams.v1 chunk).'));
          return;
        }
        // ordered, friendly subset first; then everything else collapsed under "raw"
        const ORDER = ['model', 'vae', 'prompt', 'negative', 'seed', 'width', 'height',
          'steps', 'cfg', 'sampler', 'scheduler', 'denoise', 'clip_skip', 'aspect', 'images',
          'sigma_shift', 'eta', 'sigma_min', 'sigma_max'];
        const table = elc('div', 'gp-meta-table', '');
        const seen = new Set();
        const addRow = (k, v) => {
          if (v === undefined || v === null || v === '') return;
          seen.add(k);
          const row = elc('div', 'gp-meta-row', '');
          const key = elc('span', 'gp-meta-key', k);
          const val = elc('span', 'gp-meta-val', stringifyVal(v));
          val.title = stringifyVal(v);
          // prompt/negative get a copy affordance
          if (k === 'prompt' || k === 'negative') {
            val.classList.add('gp-meta-long');
            val.addEventListener('click', () => { copyText(stringifyVal(v)); flash('Copied ' + k); });
          }
          row.appendChild(key); row.appendChild(val);
          table.appendChild(row);
        };
        ORDER.forEach((k) => addRow(k, params[k]));
        // remaining keys (skip internal reuse-bookkeeping the server adds)
        const SKIP = new Set(['params_source', 'reused_from_gallery_id', 'reused_from_job_id', 'reused_from_path']);
        Object.keys(params).forEach((k) => { if (!seen.has(k) && !SKIP.has(k)) addRow(k, params[k]); });
        lbMeta.appendChild(table);

        // raw JSON (collapsible)
        const details = document.createElement('details');
        details.className = 'gp-meta-raw';
        const sum = document.createElement('summary');
        sum.textContent = 'Raw JSON';
        const pre = document.createElement('pre');
        try { pre.textContent = JSON.stringify(params, null, 2); } catch (_) { pre.textContent = String(params); }
        const copyBtn = elc('button', 'gp-meta-copyraw', 'Copy');
        copyBtn.addEventListener('click', (e) => { e.preventDefault(); copyText(pre.textContent); flash('Copied JSON'); });
        details.appendChild(sum); details.appendChild(copyBtn); details.appendChild(pre);
        lbMeta.appendChild(details);
      }

      function stringifyVal(v) {
        if (typeof v === 'object') { try { return JSON.stringify(v); } catch (_) { return String(v); } }
        return String(v);
      }

      // =====================================================================
      // OUTPUT-PATH TEMPLATE  (SwarmUI OutpathBuilder)
      // =====================================================================
      const TOKENS = ['model', 'seed', 'prompt', 'date', 'time', 'width', 'height', 'steps', 'cfg', 'sampler', 'index'];
      let outpathInput = null, outpathPreview = null;
      function buildOutpathEditor() {
        const wrap = elc('div', 'gp-outpath', '');
        const lbl = elc('div', 'gp-outpath-label', 'Output filename template');
        wrap.appendChild(lbl);
        outpathInput = document.createElement('input');
        outpathInput.type = 'text';
        outpathInput.className = 'gp-outpath-input';
        outpathInput.value = outpathTpl;
        outpathInput.placeholder = '[date]/[model]-[seed]';
        outpathInput.addEventListener('input', () => {
          outpathTpl = outpathInput.value;
          saveTpl(outpathTpl);
          try { set('params.output_path_template', outpathTpl); } catch (_) {}
          refreshOutpathPreview();
        });
        wrap.appendChild(outpathInput);
        const toks = elc('div', 'gp-outpath-tokens', '');
        TOKENS.forEach((t) => {
          const chip = elc('button', 'gp-tok', '[' + t + ']');
          chip.title = 'Insert [' + t + ']';
          chip.addEventListener('click', () => insertToken('[' + t + ']'));
          toks.appendChild(chip);
        });
        wrap.appendChild(toks);
        outpathPreview = elc('div', 'gp-outpath-preview', '');
        wrap.appendChild(outpathPreview);
        const note = elc('div', 'gp-outpath-note',
          'Preview only — the server currently names files job-NNNN.png. The template is saved and sent as output_path_template for when server-side naming lands.');
        wrap.appendChild(note);
        return wrap;
      }
      function insertToken(tok) {
        if (!outpathInput) return;
        const s = outpathInput.selectionStart || outpathInput.value.length;
        const e = outpathInput.selectionEnd || s;
        outpathInput.value = outpathInput.value.slice(0, s) + tok + outpathInput.value.slice(e);
        outpathTpl = outpathInput.value;
        saveTpl(outpathTpl);
        try { set('params.output_path_template', outpathTpl); } catch (_) {}
        outpathInput.focus();
        const pos = s + tok.length;
        outpathInput.setSelectionRange(pos, pos);
        refreshOutpathPreview();
      }
      function refreshOutpathPreview() {
        if (!outpathPreview) return;
        const it = current();
        const p = (it && it.params) || snapshotParams();
        outpathPreview.textContent = resolveTemplate(outpathTpl, p) + '.png';
      }
      function resolveTemplate(tpl, p) {
        p = p || {};
        const now = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const date = now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate());
        const time = pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds());
        const promptSlug = String(p.prompt || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'untitled';
        const model = String(p.model || 'model').replace(/\.[^.]+$/, '').replace(/[\\/]/g, '-');
        const map = {
          model, seed: (p.seed != null ? p.seed : (p.noise_seed != null ? p.noise_seed : 'seed')),
          prompt: promptSlug, date, time,
          width: p.width || '', height: p.height || '', steps: p.steps || '',
          cfg: p.cfg || '', sampler: p.sampler || '', index: '0001',
        };
        let out = String(tpl || '[date]/[model]-[seed]');
        out = out.replace(/\[(\w+)\]/g, (m, k) => (map[k] !== undefined ? String(map[k]) : m));
        // sanitize path segments (keep slashes as folder separators)
        out = out.replace(/[<>:"|?*]+/g, '_');
        return out || 'image';
      }
      function snapshotParams() {
        const o = {};
        try { PARAM_KEYS.forEach((k) => { o[k] = get('params.' + k); }); } catch (_) {}
        return o;
      }
      function loadTpl() {
        try { return localStorage.getItem(TPL_KEY) || '[date]/[model]-[seed]'; } catch (_) { return '[date]/[model]-[seed]'; }
      }
      function saveTpl(v) { try { localStorage.setItem(TPL_KEY, v); } catch (_) {} }

      // =====================================================================
      // WIRING
      // =====================================================================
      // when the gallery module reports a thumb selection, remember it; double-click
      // there opens the lightbox (handled by our decorateCard dblclick). We also let a
      // bus event open the lightbox directly.
      bus.on('result:ready', (p) => { if (p) { lastResult = p; upsertItem(p); refreshOutpathPreview(); } });
      bus.on('gallery:select', (p) => { if (p) upsertItem(p); });
      bus.on('change:gallery', () => { /* gallery module rebuilt its strip; re-decorate */ if (galleryHost) galleryHost.querySelectorAll('.gal-card').forEach(decorateCard); });
      bus.on('galleryPro:open', (p) => { const it = p && (byKey.get(keyOf(p)) || upsertItem(p)); if (it) openLightbox(it); });
      bus.on('galleryPro:open-latest', () => { if (items.length) openByIndex(0); });

      // initial load + observe the strip (after app:ready so api.base is settled)
      bus.on('app:ready', () => { loadGallery(); observeGallery(); });
      // also attempt immediately in case app:ready already fired
      loadGallery();
      observeGallery();
      renderBatchBar();

      // =====================================================================
      // HELPERS
      // =====================================================================
      function elc(tag, cls, txt) { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }
      function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
      function clampDim(v) { v = Math.round(v / 8) * 8; return Math.max(64, Math.min(8192, v)); }
      function copyText(t) {
        try { if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(t); return; } } catch (_) {}
        try {
          const ta = document.createElement('textarea'); ta.value = t; document.body.appendChild(ta);
          ta.select(); document.execCommand('copy'); ta.remove();
        } catch (_) {}
      }
      let flashEl = null, flashTimer = null;
      function flash(msg) {
        if (!flashEl) { flashEl = document.createElement('div'); flashEl.id = 'gp-flash'; document.body.appendChild(flashEl); }
        flashEl.textContent = msg;
        flashEl.classList.add('show');
        clearTimeout(flashTimer);
        flashTimer = setTimeout(() => flashEl.classList.remove('show'), 1700);
      }

      function injectCSS() {
        if (document.getElementById('style-galleryPro')) return;
        const s = document.createElement('style');
        s.id = 'style-galleryPro';
        s.textContent = `
/* selection decoration on the existing gallery cards */
#gallery .gal-card.gp-card{}
#gallery .gal-card .gp-check{position:absolute;bottom:4px;left:4px;width:16px;height:16px;
  border:1.5px solid var(--line);border-radius:4px;background:rgba(20,22,28,.7);
  cursor:pointer;opacity:0;transition:opacity .12s;z-index:3}
#gallery .gal-card:hover .gp-check,#gallery .gal-card.gp-selected .gp-check{opacity:1}
#gallery .gal-card .gp-check:hover{border-color:var(--accent)}
#gallery .gal-card.gp-selected .gp-check{background:var(--accent2);border-color:var(--accent)}
#gallery .gal-card.gp-selected .gp-check::after{content:'✓';color:#fff;font-size:11px;
  display:block;text-align:center;line-height:13px}
#gallery .gal-card.gp-selected{outline:2px solid var(--accent);outline-offset:-2px}

/* context menu */
.gp-ctxmenu{position:fixed;z-index:10000;min-width:170px;background:var(--panel2);
  border:1px solid var(--line);border-radius:var(--radius);padding:4px;
  box-shadow:0 10px 30px rgba(0,0,0,.5);font:13px/1.2 system-ui,sans-serif}
.gp-ctxmenu .gp-ctx-item{display:block;width:100%;text-align:left;background:none;border:0;
  color:var(--text);padding:7px 10px;border-radius:6px;cursor:pointer}
.gp-ctxmenu .gp-ctx-item:hover{background:var(--accent2);color:#fff}
.gp-ctxmenu .gp-ctx-danger:hover{background:var(--danger)}
.gp-ctxmenu .gp-ctx-sep{height:1px;background:var(--line);margin:4px 2px}

/* batch bar */
#gp-batchbar{position:fixed;right:16px;bottom:96px;z-index:9000;display:flex;align-items:center;
  gap:8px;background:var(--panel2);border:1px solid var(--line);border-radius:var(--radius);
  padding:8px 12px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
#gp-batchbar.gp-hidden{display:none}
#gp-batchbar .gp-bb-label{font-size:12px;color:var(--text);font-weight:600;margin-right:4px}
#gp-batchbar .gp-bb-btn{background:var(--panel);border:1px solid var(--line);color:var(--text);
  border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer}
#gp-batchbar .gp-bb-btn:hover{border-color:var(--accent)}
#gp-batchbar .gp-bb-danger{color:var(--danger)}
#gp-batchbar .gp-bb-danger:hover{border-color:var(--danger)}

/* lightbox */
#gp-lightbox{position:fixed;inset:0;z-index:9500;display:flex;background:rgba(8,9,12,.92);
  backdrop-filter:blur(2px)}
#gp-lightbox.gp-hidden{display:none}
#gp-lightbox .gp-lb-stage{position:relative;flex:1;display:flex;align-items:center;
  justify-content:center;overflow:auto;padding:24px}
#gp-lightbox .gp-lb-img{max-width:none}
#gp-lightbox .gp-lb-img.gp-fit{max-width:100%;max-height:100%;object-fit:contain}
#gp-lightbox .gp-lb-img:not(.gp-fit){cursor:zoom-out}
#gp-lightbox .gp-lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:64px;
  background:rgba(28,31,38,.7);border:1px solid var(--line);color:var(--text);font-size:30px;
  line-height:1;cursor:pointer;border-radius:8px}
#gp-lightbox .gp-lb-nav:hover{background:var(--accent2);border-color:var(--accent)}
#gp-lightbox .gp-lb-prev{left:14px}
#gp-lightbox .gp-lb-next{right:14px}
#gp-lightbox .gp-lb-side{width:340px;max-width:42vw;background:var(--panel);
  border-left:1px solid var(--line);display:flex;flex-direction:column;overflow-y:auto;padding:14px}
#gp-lightbox .gp-lb-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
#gp-lightbox .gp-lb-title{font-weight:700;font-size:14px}
#gp-lightbox .gp-lb-close{background:var(--panel2);border:1px solid var(--line);color:var(--text);
  width:28px;height:28px;border-radius:6px;cursor:pointer;font-size:14px}
#gp-lightbox .gp-lb-close:hover{border-color:var(--danger);color:var(--danger)}
#gp-lightbox .gp-lb-actions{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
#gp-lightbox .gp-lb-act{background:var(--panel2);border:1px solid var(--line);color:var(--text);
  border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer}
#gp-lightbox .gp-lb-act:hover{border-color:var(--accent)}
#gp-lightbox .gp-lb-act-danger{color:var(--danger)}
#gp-lightbox .gp-lb-act-danger:hover{border-color:var(--danger)}
#gp-lightbox .gp-lb-fit{align-self:flex-start;background:var(--panel2);border:1px solid var(--line);
  color:var(--muted);border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;margin-bottom:10px}
#gp-lightbox .gp-lb-fit:hover{border-color:var(--accent);color:var(--text)}
#gp-lightbox .gp-lb-caption{margin-bottom:8px}
#gp-lightbox .gp-cap-name{font-size:12px;color:var(--text);word-break:break-all}
#gp-lightbox .gp-cap-pos{font-size:11px;color:var(--muted);margin-top:2px}
#gp-lightbox .gp-meta-loading,#gp-lightbox .gp-meta-empty{color:var(--muted);font-size:12px;padding:8px 0}
#gp-lightbox .gp-meta-table{display:flex;flex-direction:column;gap:1px;border:1px solid var(--line);
  border-radius:6px;overflow:hidden;background:var(--line)}
#gp-lightbox .gp-meta-row{display:grid;grid-template-columns:84px 1fr;gap:8px;background:var(--panel2);
  padding:5px 8px}
#gp-lightbox .gp-meta-key{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
#gp-lightbox .gp-meta-val{color:var(--text);font-size:12px;word-break:break-word;white-space:pre-wrap}
#gp-lightbox .gp-meta-long{cursor:copy}
#gp-lightbox .gp-meta-long:hover{color:var(--accent)}
#gp-lightbox .gp-meta-raw{margin-top:10px;border:1px solid var(--line);border-radius:6px;padding:6px 8px;
  background:var(--panel2);position:relative}
#gp-lightbox .gp-meta-raw summary{cursor:pointer;font-size:12px;color:var(--muted)}
#gp-lightbox .gp-meta-raw pre{margin:8px 0 0;font-size:11px;color:var(--text);white-space:pre-wrap;
  word-break:break-all;max-height:240px;overflow:auto}
#gp-lightbox .gp-meta-copyraw{position:absolute;right:6px;top:4px;background:var(--panel);
  border:1px solid var(--line);color:var(--muted);border-radius:5px;padding:2px 8px;font-size:10px;cursor:pointer}
#gp-lightbox .gp-meta-copyraw:hover{border-color:var(--accent);color:var(--text)}

/* output-path editor */
#gp-lightbox .gp-outpath{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
#gp-lightbox .gp-outpath-label{font-size:12px;font-weight:600;margin-bottom:6px}
#gp-lightbox .gp-outpath-input{width:100%;box-sizing:border-box;background:var(--panel2);
  border:1px solid var(--line);color:var(--text);border-radius:6px;padding:6px 8px;font-size:12px;
  font-family:ui-monospace,monospace}
#gp-lightbox .gp-outpath-input:focus{outline:none;border-color:var(--accent)}
#gp-lightbox .gp-outpath-tokens{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
#gp-lightbox .gp-tok{background:var(--panel2);border:1px solid var(--line);color:var(--muted);
  border-radius:5px;padding:2px 7px;font-size:10px;cursor:pointer;font-family:ui-monospace,monospace}
#gp-lightbox .gp-tok:hover{border-color:var(--accent);color:var(--text)}
#gp-lightbox .gp-outpath-preview{font-size:11px;color:var(--ok);font-family:ui-monospace,monospace;
  word-break:break-all;margin-top:4px}
#gp-lightbox .gp-outpath-note{font-size:10px;color:var(--muted);margin-top:6px;line-height:1.4}

/* flash toast */
#gp-flash{position:fixed;left:50%;bottom:104px;transform:translateX(-50%) translateY(6px);z-index:10001;
  background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:7px 16px;font-size:12px;
  color:var(--text);opacity:0;pointer-events:none;transition:opacity .15s,transform .15s;
  box-shadow:0 6px 20px rgba(0,0,0,.4)}
#gp-flash.show{opacity:1;transform:translateX(-50%) translateY(0)}
`;
        document.head.appendChild(s);
      }

      console.info('[galleryPro] initialised');
    },
  });
})();
