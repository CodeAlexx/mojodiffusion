/* api.js — SHARED CONTRACT. Adapter from the canvas's ComfyUI-shaped calls to
   serenity-server's PROVEN /v1/* API (POST /v1/generate -> worker -> PNG, verified).
   Keeps the ComfyUI method names generate_ws/gallery already use, and translates:
     submitPrompt -> POST /v1/generate (body built from Serenity.state.params)
     connectWS    -> WS /v1/progress?job=<id>, {ev:...} -> ComfyUI-shaped messages
     viewUrl      -> /out/<filename>   (served by serenity-server)
   Same-origin by default (serenity-server serves this canvas); ?api= overrides. */
(function () {
  "use strict";
  window.Serenity = window.Serenity || {};
  var BASE = new URLSearchParams(location.search).get('api') || '';
  var j = function (r) { if (!r.ok) throw new Error(r.status + ' ' + r.url); return r.json(); };
  var lastJobId = null;

  function baseName(p) { return String(p || '').split('/').pop(); }

  function generateBody() {
    var p = (window.Serenity.state && window.Serenity.state.params) || {};
    return {
      model: p.model && p.model.indexOf('—') < 0 ? p.model : 'z-image',
      prompt: p.prompt || '',
      negative: p.negative || '',
      width: p.width || 1024,
      height: p.height || 1024,
      steps: p.steps || 8,
      cfg: p.cfg != null ? p.cfg : 1.5,
      seed: p.seed != null ? p.seed : -1,
      sampler: p.sampler || 'euler',
      scheduler: p.scheduler || 'simple',
      images: p.images || 1,
      clip_skip: p.clip_skip || 0,
      eta: p.eta != null ? p.eta : -1,
      sigma_min: p.sigma_min != null ? p.sigma_min : -1,
      sigma_max: p.sigma_max != null ? p.sigma_max : -1,
      restart_sampling: !!p.restart_sampling,
      vae: p.vae || '',
    };
  }

  var api = {
    base: BASE,
    // ComfyUI shape kept; ignores the assembled graph and submits the proven /v1/generate body
    submitPrompt: function (_graph, _clientId) {
      return fetch(BASE + '/v1/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(generateBody()),
      }).then(j).then(function (res) {
        lastJobId = res && (res.job_id || res.id || res.prompt_id);
        return { prompt_id: lastJobId, job_id: lastJobId };
      });
    },
    interrupt: function () {
      return fetch(BASE + '/v1/cancel', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job: lastJobId }),
      });
    },
    // serenity result image, served by serenity-server's /out/* route
    viewUrl: function (filename) { return BASE + '/out/' + encodeURIComponent(baseName(filename)); },
    // best-effort; paramRail/topbar degrade gracefully if these reject
    models: function () { return fetch(BASE + '/v1/models').then(j); },
    systemStats: function () { return fetch(BASE + '/v1/health').then(j); },
    objectInfo: function () { return Promise.reject(new Error('objectInfo n/a (using fallbacks)')); },
    loras: function () { return Promise.resolve([]); },
    embeddings: function () { return Promise.resolve([]); },
    history: function () { return Promise.resolve({}); },
    // Tier B (not wired yet) — reject so canvas degrades
    uploadImage: function () { return Promise.reject(new Error('upload n/a')); },
    uploadMask: function () { return Promise.reject(new Error('mask upload n/a')); },
    preprocess: function () { return Promise.reject(new Error('preprocess n/a')); },
    sam: function () { return Promise.reject(new Error('sam n/a')); },

    // WS: translate serenity {ev:...} -> the ComfyUI-shaped messages generate_ws expects
    connectWS: function (_clientId, cb) {
      var job = lastJobId;
      var wsBase = BASE ? BASE.replace(/^http/, 'ws') : ((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host);
      var url = wsBase + '/v1/progress' + (job ? ('?job=' + encodeURIComponent(job)) : '');
      var ws;
      try { ws = new WebSocket(url); } catch (e) { console.warn('[api] ws open failed', e); return { close: function () {} }; }
      ws.onmessage = function (ev) {
        var m; try { m = JSON.parse(ev.data); } catch (_) { return; }
        var which = (m.ev || '').toLowerCase();
        if (which === 'progress') {
          cb({ type: 'progress', data: { value: m.step || 0, max: m.total || 0, prompt_id: job } });
        } else if (which === 'done') {
          var fn = baseName(m.output_path);
          cb({ type: 'executed', data: { prompt_id: job, output: { images: [{ filename: fn, type: 'output', subfolder: '' }] } } });
          cb({ type: 'executing', data: { node: null, prompt_id: job } });
        } else if (which === 'failed') {
          cb({ type: 'execution_error', data: { prompt_id: job, error: m.error || 'failed' } });
        } else if (which === 'cancelled') {
          cb({ type: 'executing', data: { node: null, prompt_id: job } });
        }
        // 'ready' -> ignore
      };
      ws.onerror = function (e) { console.warn('[api] ws error', e); };
      return ws;
    },
  };
  Serenity.api = api;
  Serenity.clientId = 'serenity-' + Math.random().toString(36).slice(2);
})();
