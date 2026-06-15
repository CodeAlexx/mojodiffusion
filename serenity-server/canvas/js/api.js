/* api.js — SHARED CONTRACT. Adapter from the canvas's ComfyUI-shaped calls to
   serenity-server's PROVEN /v1/* API (POST /v1/generate -> worker -> PNG).
     submitPrompt -> POST /v1/generate (body from Serenity.state.params)
     connectWS    -> WS /v1/progress?job=<id>, {ev:...} -> ComfyUI-shaped messages
     viewUrl      -> /out/<filename>
   IMPORTANT: generate_ws opens the WS BEFORE submit, but serenity's /v1/progress is
   JOB-keyed and the job id only exists after submit. So connectWS DEFERS: it stashes
   the callback, and submitPrompt opens the real WS once it has the job id. */
(function () {
  "use strict";
  window.Serenity = window.Serenity || {};
  var BASE = new URLSearchParams(location.search).get('api') || '';
  var j = function (r) { if (!r.ok) throw new Error(r.status + ' ' + r.url); return r.json(); };
  var lastJobId = null;
  var pendingCb = null;   // callback waiting for the job id (connectWS called pre-submit)
  var activeWS = null;

  function baseName(p) { return String(p || '').split('/').pop(); }

  function generateBody() {
    var p = (window.Serenity.state && window.Serenity.state.params) || {};
    return {
      model: p.model && p.model.indexOf('—') < 0 ? p.model : 'z-image',
      prompt: p.prompt || '', negative: p.negative || '',
      width: p.width || 1024, height: p.height || 1024,
      steps: p.steps || 8, cfg: p.cfg != null ? p.cfg : 1.5,
      seed: p.seed != null ? p.seed : -1,
      sampler: p.sampler || 'euler', scheduler: p.scheduler || 'simple',
      images: p.images || 1, clip_skip: p.clip_skip || 0,
      eta: p.eta != null ? p.eta : -1,
      sigma_min: p.sigma_min != null ? p.sigma_min : -1,
      sigma_max: p.sigma_max != null ? p.sigma_max : -1,
      restart_sampling: !!p.restart_sampling, vae: p.vae || '',
      sigma_shift: p.sigma_shift != null ? p.sigma_shift : 3.0,
      cfg_override: p.cfg_override != null ? p.cfg_override : -1,
      cfg_override_start_percent: p.cfg_override_start_percent != null ? p.cfg_override_start_percent : 0.0,
      cfg_override_end_percent: p.cfg_override_end_percent != null ? p.cfg_override_end_percent : 1.0,
      // SwarmUI-parity fields from the 10 canvas modules. undefined values are
      // dropped by JSON.stringify, and the Rust GenerateRequest ignores unknown
      // keys, so forwarding is harmless where the backend doesn't model it yet.
      batch_size: p.batch_size, end_steps_early_pct: p.end_steps_early_pct,
      no_seed_increment: p.no_seed_increment, denoise: p.denoise,
      loras: p.loras, refiner: p.refiner,
      hires_scale: p.hires_scale, hires_denoise: p.hires_denoise,
      mask_data: p.mask_data, mask_channel: p.mask_channel, mask_mime: p.mask_mime,
      // Path-based inpaint/img2img: the worker reads init_image/mask_image as
      // FILESYSTEM PATHS. generate_ws uploads the painted mask / init layer via
      // api.uploadMask / api.uploadImage and stashes the returned path here.
      init_image: p.init_image, mask_image: p.mask_image,
      lanpaint_mask_channel: p.lanpaint_mask_channel,
      outpaint: p.outpaint, outpaint_enabled: p.outpaint_enabled,
      refiner_model: p.refiner_model, refiner_steps: p.refiner_steps,
      refiner_cfg: p.refiner_cfg, refiner_method: p.refiner_method,
      refiner_control: p.refiner_control, refiner_tiling: p.refiner_tiling,
      upscaler: p.upscaler, upscaler_model: p.upscaler_model, upscale_by: p.upscale_by,
    };
  }

  function closeProgressWS() { try { if (activeWS) activeWS.close(); } catch (_) {} activeWS = null; }

  // Read a Blob (or already-string base64/dataURL) as base64 (no data: prefix).
  function toBase64(blobOrStr) {
    if (typeof blobOrStr === 'string') {
      var s = blobOrStr;
      var i = s.indexOf('base64,');
      return Promise.resolve(i >= 0 ? s.slice(i + 7) : s);
    }
    if (!(blobOrStr instanceof Blob)) return Promise.reject(new Error('upload: need a Blob'));
    return new Promise(function (resolve, reject) {
      var fr = new FileReader();
      fr.onload = function () {
        var r = String(fr.result || '');
        var i = r.indexOf('base64,');
        resolve(i >= 0 ? r.slice(i + 7) : r);
      };
      fr.onerror = function () { reject(fr.error || new Error('FileReader failed')); };
      fr.readAsDataURL(blobOrStr);
    });
  }

  // POST a base64 PNG to an upload endpoint; resolve to {name, path, url}.
  function postUpload(endpoint, blobOrStr, name) {
    return toBase64(blobOrStr).then(function (b64) {
      return fetch(BASE + endpoint, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || 'layer.png', data: b64 }),
      }).then(j);
    });
  }

  // Open the job-keyed progress WS and translate {ev:...} -> ComfyUI-shaped cb calls.
  function openProgressWS(job, cb) {
    closeProgressWS();
    var wsBase = BASE ? BASE.replace(/^http/, 'ws')
      : ((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host);
    var url = wsBase + '/v1/progress?job=' + encodeURIComponent(job);
    var ws;
    try { ws = new WebSocket(url); } catch (e) { console.warn('[api] ws open failed', e); return; }
    activeWS = ws;
    ws.onmessage = function (ev) {
      var m; try { m = JSON.parse(ev.data); } catch (_) { return; }
      var which = (m.ev || '').toLowerCase();
      if (which === 'progress') {
        cb({ type: 'progress', data: { value: m.step || 0, max: m.total || 0, phase: m.phase || '', prompt_id: job } });
      } else if (which === 'done') {
        var fn = baseName(m.output_path);
        cb({ type: 'executed', data: { prompt_id: job, output: { images: [{ filename: fn, type: 'output', subfolder: '' }] } } });
        cb({ type: 'executing', data: { node: null, prompt_id: job } });
      } else if (which === 'failed') {
        cb({ type: 'execution_error', data: { prompt_id: job, error: m.error || 'failed' } });
      } else if (which === 'cancelled') {
        cb({ type: 'executing', data: { node: null, prompt_id: job } });
      }
    };
    ws.onerror = function (e) { console.warn('[api] ws error', e); };
  }

  var api = {
    base: BASE,
    submitPrompt: function (_graph, _clientId) {
      return fetch(BASE + '/v1/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(generateBody()),
      }).then(j).then(function (res) {
        lastJobId = res && (res.job_id || res.id || res.prompt_id);
        // open the deferred progress WS now that we have the job id
        if (pendingCb && lastJobId) { openProgressWS(lastJobId, pendingCb); pendingCb = null; }
        return { prompt_id: lastJobId, job_id: lastJobId };
      });
    },
    interrupt: function () {
      return fetch(BASE + '/v1/cancel', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job: lastJobId }),
      });
    },
    viewUrl: function (filename) { return BASE + '/out/' + encodeURIComponent(baseName(filename)); },
    models: function () { return fetch(BASE + '/v1/models').then(j); },
    systemStats: function () { return fetch(BASE + '/v1/health').then(j); },
    objectInfo: function () { return Promise.reject(new Error('objectInfo n/a (fallbacks)')); },
    loras: function () { return Promise.resolve([]); },
    embeddings: function () { return Promise.resolve([]); },
    history: function () { return Promise.resolve({}); },
    // uploadImage(blob, name) -> {name, path, url}. `path` is the worker-readable
    // absolute path; generate_ws stashes it into params.init_image.
    uploadImage: function (blob, name) { return postUpload('/upload/image', blob, name); },
    // uploadMask(blob, ref, name) -> {name, path, url}. `ref` (the base raster) is
    // accepted for ComfyUI-call-shape compatibility but unused: the worker's inpaint
    // path composites init_image+mask_image by path, so only the mask PNG is sent.
    uploadMask: function (blob, ref, name) {
      return postUpload('/upload/mask', blob, typeof ref === 'string' ? ref : name);
    },
    preprocess: function () { return Promise.reject(new Error('preprocess n/a')); },
    sam: function () { return Promise.reject(new Error('sam n/a')); },

    // Called by generate_ws BEFORE submit -> ALWAYS defer; submitPrompt opens the WS
    // once it has the (fresh) job id. (Always-defer keeps re-clicks on the right job.)
    connectWS: function (_clientId, cb) {
      pendingCb = cb;
      return { close: function () { pendingCb = null; closeProgressWS(); }, binaryType: 'arraybuffer' };
    },
  };
  Serenity.api = api;
  Serenity.clientId = 'serenity-' + Math.random().toString(36).slice(2);
})();
