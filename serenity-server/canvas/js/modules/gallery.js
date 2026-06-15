/* gallery.js — module 'gallery'. Owns the bottom filmstrip (#gallery) + queue list
   (sibling inside #queue-strip, never stomping generateWS's progress bar).
   Reads: api.history (+ api.viewUrl) for thumbnails. Listens: 'result:ready' to prepend.
   Click a thumb => load into preview + RESTORE its params into state.params (SwarmUI sig).
   Hover actions: reuse-seed, download, star, delete. Queue list with interrupt.
   Talks ONLY through ctx.state/get/set/bus/api. Degrades gracefully when API 404s. */
(function () {
  "use strict";
  if (!window.Serenity || !Serenity.register) { console.warn('[gallery] Serenity not ready'); return; }

  Serenity.register('gallery', {
    init(ctx) {
      const { state, get, set, bus, api } = ctx;
      const mount = (ctx.dom && ctx.dom.gallery) || document.getElementById('gallery');
      const queueHost = (ctx.dom && ctx.dom.queueStrip) || document.getElementById('queue-strip');
      if (!mount) { console.warn('[gallery] no mount'); return; }

      // ---- scoped CSS (never touch the shared stylesheet) -------------------
      injectCSS();

      // ---- local model -----------------------------------------------------
      // items: [{id, filename, subfolder, type, url, params, starred}]
      const items = [];
      const seen = new Set();          // de-dupe by url
      const STAR_KEY = 'serenity.gallery.stars';
      const stars = loadStars();       // Set of filenames the user starred (persisted)

      // ---- DOM scaffold inside our mount ----------------------------------
      mount.classList.add('gallery-root');
      mount.innerHTML = '';
      const strip = el('div', 'gal-strip');
      const empty = el('div', 'gal-empty');
      empty.textContent = 'No results yet — generate something';
      mount.appendChild(empty);
      mount.appendChild(strip);

      // queue list lives as a SIBLING inside #queue-strip so we never collide
      // with generateWS's progress bar (which it appends/owns separately).
      let queuePanel = null;
      if (queueHost) {
        queuePanel = el('div', 'gal-queue');
        queuePanel.id = 'gallery-queue';        // unique id, ours only
        queueHost.appendChild(queuePanel);
      }

      // =====================================================================
      // RENDER
      // =====================================================================
      function render() {
        empty.style.display = items.length ? 'none' : 'block';
        strip.style.display = items.length ? 'flex' : 'none';
        // diff-light: rebuild strip (filmstrips are small; simplest correct path)
        strip.innerHTML = '';
        items.forEach((it) => strip.appendChild(thumbEl(it)));
      }

      function thumbEl(it) {
        const card = el('div', 'gal-card');
        card.dataset.id = it.id;
        if (it.starred) card.classList.add('is-starred');

        const img = new Image();
        img.className = 'gal-img';
        img.alt = it.filename || 'result';
        img.loading = 'lazy';
        img.decoding = 'async';
        if (it.url) { img.src = it.url; }
        img.onerror = () => { card.classList.add('gal-broken'); img.removeAttribute('src'); };
        card.appendChild(img);

        // star marker (top-left)
        const starMark = el('div', 'gal-starmark');
        starMark.textContent = '★';
        card.appendChild(starMark);

        // hover action bar
        const actions = el('div', 'gal-actions');
        actions.appendChild(actionBtn('seed', 'Reuse seed', () => reuseSeed(it)));
        actions.appendChild(actionBtn('dl', 'Download', () => download(it)));
        actions.appendChild(actionBtn('star', it.starred ? 'Unstar' : 'Star', () => toggleStar(it)));
        actions.appendChild(actionBtn('del', 'Delete', () => removeItem(it)));
        card.appendChild(actions);

        // click thumb => load into preview + restore params
        card.addEventListener('click', (e) => {
          if (e.target.closest('.gal-actbtn')) return; // action buttons handle themselves
          selectItem(it);
        });
        return card;
      }

      function actionBtn(kind, title, fn) {
        const b = el('button', 'gal-actbtn gal-act-' + kind);
        b.type = 'button';
        b.title = title;
        b.setAttribute('aria-label', title);
        b.textContent = glyph(kind);
        b.addEventListener('click', (e) => { e.stopPropagation(); fn(); });
        return b;
      }

      function glyph(kind) {
        return ({ seed: '🌱', dl: '⬇', star: '★', del: '🗑' })[kind] || '?';
      }

      // =====================================================================
      // ACTIONS
      // =====================================================================
      // Click = load into preview (broadcast for canvas/generateWS) + restore params.
      function selectItem(it) {
        // restore SwarmUI-signature params into state.params (only known keys)
        if (it.params && typeof it.params === 'object') {
          restoreParams(it.params);
        }
        set('gallery.selected', it.id);
        // ask whoever owns the stage to load this image as the preview.
        bus.emit('gallery:select', { id: it.id, url: it.url, filename: it.filename, params: it.params || null });
        bus.emit('preview:load', { url: it.url, filename: it.filename }); // canvas/generateWS may listen
        highlightSelected(it.id);
      }

      function highlightSelected(id) {
        strip.querySelectorAll('.gal-card').forEach((c) => {
          c.classList.toggle('is-selected', c.dataset.id === String(id));
        });
      }

      // restore params honoring the SwarmUI signature in state.params (state.js)
      const PARAM_KEYS = [
        'model', 'vae', 'prompt', 'negative', 'width', 'height', 'aspect',
        'steps', 'cfg', 'seed', 'images', 'sampler', 'scheduler',
        'clip_skip', 'sigma_min', 'sigma_max', 'eta', 'restart_sampling',
        'loras', 'controlnet', 'refiner', 'denoise',
      ];
      function restoreParams(p) {
        PARAM_KEYS.forEach((k) => {
          if (p[k] !== undefined && p[k] !== null) {
            set('params.' + k, p[k]);
          }
        });
        bus.emit('params:restored', p);
      }

      function reuseSeed(it) {
        const seed = pickSeed(it.params);
        if (seed === undefined || seed === null) return;
        set('params.seed', seed);
        bus.emit('params:restored', { seed });
        flash('Seed ' + seed + ' reused');
      }

      function pickSeed(params) {
        if (!params) return null;
        if (params.seed !== undefined) return params.seed;
        if (params.noise_seed !== undefined) return params.noise_seed;
        return null;
      }

      async function download(it) {
        if (!it.url) return;
        try {
          // fetch as blob so we can force a filename; fall back to anchor on CORS/404
          const resp = await fetch(it.url);
          if (!resp.ok) throw new Error('http ' + resp.status);
          const blob = await resp.blob();
          const a = document.createElement('a');
          const objUrl = URL.createObjectURL(blob);
          a.href = objUrl;
          a.download = it.filename || ('serenity-' + it.id + '.png');
          document.body.appendChild(a); a.click(); a.remove();
          setTimeout(() => URL.revokeObjectURL(objUrl), 4000);
        } catch (err) {
          // graceful fallback: plain anchor download attempt
          try {
            const a = document.createElement('a');
            a.href = it.url; a.download = it.filename || 'result.png';
            a.target = '_blank';
            document.body.appendChild(a); a.click(); a.remove();
          } catch (_) { flash('Download unavailable'); }
        }
      }

      function toggleStar(it) {
        it.starred = !it.starred;
        if (it.starred) stars.add(it.filename || it.id);
        else stars.delete(it.filename || it.id);
        saveStars();
        render();
        highlightSelected(get('gallery.selected'));
      }

      function removeItem(it) {
        const idx = items.findIndex((x) => x.id === it.id);
        if (idx >= 0) {
          items.splice(idx, 1);
          if (it.url) seen.delete(it.url);
          // mirror minimal list into shared state.gallery (id/filename/params)
          syncStateGallery();
          render();
          bus.emit('gallery:removed', { id: it.id });
        }
      }

      // =====================================================================
      // INGEST
      // =====================================================================
      // Build an internal item from a variety of plausible payload shapes so we
      // coordinate cleanly with generateWS regardless of its exact 'result:ready'.
      function makeItem(raw) {
        if (!raw) return null;
        const img = raw.image || raw;                 // sometimes nested under .image
        const filename = img.filename || raw.filename || raw.name || null;
        const subfolder = img.subfolder || raw.subfolder || '';
        const type = img.type || raw.type || 'output';
        let url = raw.url || img.url || null;
        if (!url && filename && api && typeof api.viewUrl === 'function') {
          url = api.viewUrl(filename, type, subfolder);
        }
        if (!url) return null;
        const id = raw.id || raw.prompt_id || (filename + '|' + subfolder + '|' + type);
        const params = raw.params || raw.meta || null;
        return {
          id: String(id), filename, subfolder, type, url,
          params, starred: stars.has(filename || id),
        };
      }

      function addItem(raw, opts) {
        const it = makeItem(raw);
        if (!it) return null;
        if (seen.has(it.url)) return null;            // de-dupe
        seen.add(it.url);
        if (opts && opts.prepend) items.unshift(it);
        else items.push(it);
        syncStateGallery();
        return it;
      }

      function syncStateGallery() {
        // shared state.gallery uses [{id, filename, params}] per state.js comment
        try {
          state.gallery = items.map((x) => ({ id: x.id, filename: x.filename, params: x.params }));
          bus.emit('change:gallery', state.gallery);
        } catch (_) {}
      }

      // 'result:ready' => one or many new results; prepend (newest first)
      function onResultReady(payload) {
        if (!payload) return;
        const list = normalizeResults(payload);
        let any = false;
        list.forEach((r) => {
          // attach current params snapshot if the result didn't carry its own
          if (!r.params) r.params = snapshotParams();
          if (addItem(r, { prepend: true })) any = true;
        });
        if (any) render();
      }

      function snapshotParams() {
        const out = {};
        try { PARAM_KEYS.forEach((k) => { out[k] = get('params.' + k); }); } catch (_) {}
        return out;
      }

      // Accept: array of images; {images:[...]}; {image:{...}}; single image obj;
      // ComfyUI history-style {outputs:{node:{images:[...]}}}; {filename:...}
      function normalizeResults(payload) {
        if (Array.isArray(payload)) return payload;
        if (payload.images && Array.isArray(payload.images)) {
          return payload.images.map((im) => ({ ...im, params: payload.params || payload.meta || null,
            prompt_id: payload.prompt_id, id: payload.id }));
        }
        if (payload.outputs && typeof payload.outputs === 'object') {
          return imagesFromOutputs(payload.outputs, payload.params, payload.prompt_id);
        }
        if (payload.image) return [payload.image];
        if (payload.filename || payload.url) return [payload];
        return [];
      }

      function imagesFromOutputs(outputs, params, promptId) {
        const out = [];
        Object.keys(outputs || {}).forEach((node) => {
          const o = outputs[node];
          (o && o.images ? o.images : []).forEach((im) => {
            out.push({ ...im, params, prompt_id: promptId });
          });
        });
        return out;
      }

      // =====================================================================
      // HISTORY (initial load)
      // =====================================================================
      async function loadHistory() {
        if (!api || typeof api.history !== 'function') return;
        let hist;
        try { hist = await api.history(); }
        catch (err) { console.info('[gallery] history unavailable (degrading):', err && err.message); return; }
        if (!hist || typeof hist !== 'object') return;

        // ComfyUI: { [prompt_id]: { prompt:[...], outputs:{...}, status:{...} } }
        // Iterate; newest typically last in object insertion order -> reverse so newest first.
        const ids = Object.keys(hist);
        let added = false;
        for (let i = ids.length - 1; i >= 0; i--) {
          const pid = ids[i];
          const entry = hist[pid] || {};
          const params = extractHistoryParams(entry);
          const imgs = imagesFromOutputs(entry.outputs || {}, params, pid);
          imgs.forEach((im) => { if (addItem(im, { prepend: false })) added = true; });
        }
        if (added) render();
      }

      // Best-effort: pull restorable params out of a ComfyUI prompt graph.
      function extractHistoryParams(entry) {
        try {
          const graph = entry && entry.prompt && entry.prompt[2]; // ComfyUI: prompt = [num, id, graph, ...]
          if (graph && typeof graph === 'object') {
            return paramsFromGraph(graph);
          }
        } catch (_) {}
        return null;
      }

      // Scan a ComfyUI graph for the common nodes to recover SwarmUI-style params.
      function paramsFromGraph(graph) {
        const p = {};
        try {
          Object.keys(graph).forEach((nid) => {
            const node = graph[nid] || {};
            const inp = node.inputs || {};
            const ct = node.class_type || '';
            if (ct === 'KSampler' || ct === 'KSamplerAdvanced' || ct.indexOf('SamplerCustom') >= 0) {
              if (inp.seed !== undefined) p.seed = inp.seed;
              if (inp.noise_seed !== undefined) p.seed = inp.noise_seed;
              if (inp.steps !== undefined) p.steps = inp.steps;
              if (inp.cfg !== undefined) p.cfg = inp.cfg;
              if (inp.sampler_name !== undefined) p.sampler = inp.sampler_name;
              if (inp.scheduler !== undefined) p.scheduler = inp.scheduler;
              if (inp.denoise !== undefined) p.denoise = inp.denoise;
            }
            if (ct === 'EmptyLatentImage' || ct === 'EmptySD3LatentImage') {
              if (inp.width !== undefined) p.width = inp.width;
              if (inp.height !== undefined) p.height = inp.height;
              if (inp.batch_size !== undefined) p.images = inp.batch_size;
            }
            if (ct === 'CheckpointLoaderSimple' || ct === 'UNETLoader') {
              if (inp.ckpt_name !== undefined) p.model = inp.ckpt_name;
              if (inp.unet_name !== undefined) p.model = inp.unet_name;
            }
            if (ct === 'CLIPTextEncode' && typeof inp.text === 'string') {
              // first positive-looking encode -> prompt; we can't always tell pos/neg,
              // so keep the first as prompt and second as negative.
              if (p.prompt === undefined) p.prompt = inp.text;
              else if (p.negative === undefined) p.negative = inp.text;
            }
          });
        } catch (_) {}
        return Object.keys(p).length ? p : null;
      }

      // =====================================================================
      // QUEUE LIST (sibling element in #queue-strip; interrupt via api.interrupt)
      // =====================================================================
      const queue = []; // [{id, label, kind:'running'|'pending'}]

      function renderQueue() {
        if (!queuePanel) return;
        if (!queue.length) { queuePanel.style.display = 'none'; queuePanel.innerHTML = ''; return; }
        queuePanel.style.display = 'block';
        queuePanel.innerHTML = '';
        const head = el('div', 'gal-queue-head');
        head.appendChild(text('span', 'gal-queue-title', 'Queue (' + queue.length + ')'));
        const stopAll = el('button', 'gal-queue-stop');
        stopAll.type = 'button';
        stopAll.textContent = 'Interrupt';
        stopAll.title = 'Interrupt the running job';
        stopAll.addEventListener('click', interrupt);
        head.appendChild(stopAll);
        queuePanel.appendChild(head);

        queue.forEach((q) => {
          const row = el('div', 'gal-queue-row gal-queue-' + q.kind);
          row.appendChild(text('span', 'gal-queue-dot', q.kind === 'running' ? '●' : '○'));
          row.appendChild(text('span', 'gal-queue-label', q.label || q.id));
          queuePanel.appendChild(row);
        });
      }

      async function interrupt() {
        if (!api || typeof api.interrupt !== 'function') return;
        try { await api.interrupt(); flash('Interrupt sent'); }
        catch (err) { console.info('[gallery] interrupt failed (degrading):', err && err.message); }
        bus.emit('queue:interrupt');
      }

      function setQueueFromProgress(prog) {
        // generateWS owns state.progress; we only mirror a lightweight queue view.
        queue.length = 0;
        if (prog && prog.running) {
          const lbl = prog.total ? ('Generating ' + (prog.step || 0) + '/' + prog.total) : 'Generating…';
          queue.push({ id: prog.jobId || 'job', label: lbl, kind: 'running' });
        }
        renderQueue();
      }

      // =====================================================================
      // WIRING
      // =====================================================================
      bus.on('result:ready', onResultReady);
      bus.on('result:image', onResultReady);     // alias some teams may emit
      bus.on('generate:result', onResultReady);  // alias
      bus.on('change:progress', setQueueFromProgress);
      bus.on('queue:update', (q) => {            // optional richer queue feed from generateWS
        if (Array.isArray(q)) { queue.length = 0; q.forEach((x) => queue.push(x)); renderQueue(); }
      });

      // initial render + history (after app:ready so api/base settled)
      render();
      renderQueue();
      setQueueFromProgress(state.progress);
      if (state.progress) { /* mirror any in-flight job */ }

      if (bus) bus.on('app:ready', () => { loadHistory(); });
      // also try immediately in case app:ready already fired
      loadHistory();

      // =====================================================================
      // HELPERS
      // =====================================================================
      function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
      function text(tag, cls, t) { const e = el(tag, cls); e.textContent = t; return e; }

      let flashTimer = null;
      function flash(msg) {
        let f = mount.querySelector('.gal-flash');
        if (!f) { f = el('div', 'gal-flash'); mount.appendChild(f); }
        f.textContent = msg;
        f.classList.add('show');
        clearTimeout(flashTimer);
        flashTimer = setTimeout(() => f.classList.remove('show'), 1600);
      }

      function loadStars() {
        try {
          const raw = localStorage.getItem(STAR_KEY);
          return new Set(raw ? JSON.parse(raw) : []);
        } catch (_) { return new Set(); }
      }
      function saveStars() {
        try { localStorage.setItem(STAR_KEY, JSON.stringify([...stars])); } catch (_) {}
      }

      function injectCSS() {
        if (document.getElementById('style-gallery')) return;
        const s = document.createElement('style');
        s.id = 'style-gallery';
        s.textContent = `
#gallery.gallery-root{position:relative;flex-direction:column;align-items:stretch;justify-content:center}
#gallery .gal-empty{color:var(--muted);font-size:12px;padding:0 4px;user-select:none}
#gallery .gal-strip{display:flex;align-items:center;gap:8px;height:100%;overflow-x:auto;overflow-y:hidden;padding:2px}
#gallery .gal-strip::-webkit-scrollbar{height:8px}
#gallery .gal-strip::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}
#gallery .gal-card{position:relative;flex:0 0 auto;height:100%;aspect-ratio:1/1;min-width:96px;
  border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;cursor:pointer;
  background:var(--panel2);transition:border-color .12s}
#gallery .gal-card:hover{border-color:var(--accent)}
#gallery .gal-card.is-selected{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent2) inset}
#gallery .gal-card.gal-broken{display:flex;align-items:center;justify-content:center}
#gallery .gal-card.gal-broken::after{content:'⚠';color:var(--muted);font-size:20px}
#gallery .gal-img{width:100%;height:100%;object-fit:cover;display:block}
#gallery .gal-starmark{position:absolute;top:3px;left:4px;color:var(--warn);font-size:13px;
  text-shadow:0 1px 2px #000;opacity:0;pointer-events:none;transition:opacity .12s}
#gallery .gal-card.is-starred .gal-starmark{opacity:1}
#gallery .gal-actions{position:absolute;top:3px;right:3px;display:flex;gap:3px;opacity:0;
  transition:opacity .12s}
#gallery .gal-card:hover .gal-actions{opacity:1}
#gallery .gal-actbtn{width:22px;height:22px;line-height:1;padding:0;font-size:12px;
  display:flex;align-items:center;justify-content:center;
  background:rgba(20,22,28,.82);border:1px solid var(--line);color:var(--text);
  border-radius:5px;cursor:pointer}
#gallery .gal-actbtn:hover{border-color:var(--accent);background:rgba(40,44,56,.95)}
#gallery .gal-act-del:hover{border-color:var(--danger);color:var(--danger)}
#gallery .gal-act-star:hover{border-color:var(--warn);color:var(--warn)}
#gallery .gal-flash{position:absolute;bottom:6px;right:8px;background:var(--panel2);
  border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:11px;color:var(--text);
  opacity:0;transform:translateY(4px);transition:opacity .15s,transform .15s;pointer-events:none}
#gallery .gal-flash.show{opacity:1;transform:translateY(0)}
/* queue panel lives in #queue-strip as a sibling; do not style the strip container itself */
#queue-strip #gallery-queue{min-width:180px;max-width:260px;background:var(--panel);
  border:1px solid var(--line);border-radius:var(--radius);padding:8px;
  box-shadow:0 6px 20px rgba(0,0,0,.35);margin-top:8px}
#queue-strip #gallery-queue .gal-queue-head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:6px}
#queue-strip #gallery-queue .gal-queue-title{font-weight:600;font-size:12px}
#queue-strip #gallery-queue .gal-queue-stop{background:var(--panel2);border:1px solid var(--line);
  color:var(--danger);border-radius:5px;padding:3px 8px;font-size:11px;cursor:pointer}
#queue-strip #gallery-queue .gal-queue-stop:hover{border-color:var(--danger)}
#queue-strip #gallery-queue .gal-queue-row{display:flex;align-items:center;gap:6px;
  font-size:11px;color:var(--muted);padding:2px 0}
#queue-strip #gallery-queue .gal-queue-running .gal-queue-dot{color:var(--ok)}
#queue-strip #gallery-queue .gal-queue-pending .gal-queue-dot{color:var(--muted)}
#queue-strip #gallery-queue .gal-queue-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
`;
        document.head.appendChild(s);
      }
    },
  });
})();
