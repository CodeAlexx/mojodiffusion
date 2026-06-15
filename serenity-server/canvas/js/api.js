/* api.js — SHARED CONTRACT. The ComfyUI HTTP/WS client. Owned by scaffold.
   Maps to the measured ComfyUI API the canvas drives. serenity-server will speak
   these (ComfyUI-API-compat, Tier A/B); until then calls may 404 — modules must
   degrade gracefully (show empty/placeholder, never crash the app).
   Base is same-origin by default; override with ?api=http://host:port. */
(function () {
  "use strict";
  window.Serenity = window.Serenity || {};
  const BASE = new URLSearchParams(location.search).get('api') || '';
  const j = async (r) => { if (!r.ok) throw new Error(r.status + ' ' + r.url); return r.json(); };

  const api = {
    base: BASE,
    // submit a ComfyUI workflow graph -> {prompt_id}
    submitPrompt: (graph, clientId) =>
      fetch(BASE + '/prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: graph, client_id: clientId }) }).then(j),
    interrupt: () => fetch(BASE + '/interrupt', { method: 'POST' }),
    history: (id) => fetch(BASE + '/history' + (id ? '/' + id : '')).then(j),
    objectInfo: (cls) => fetch(BASE + '/object_info' + (cls ? '/' + cls : '')).then(j),
    systemStats: () => fetch(BASE + '/system_stats').then(j),
    models: () => fetch(BASE + '/models').then(j),
    loras: () => fetch(BASE + '/models/loras').then(j),
    embeddings: () => fetch(BASE + '/embeddings').then(j),
    // image url for <img>/Konva.Image
    viewUrl: (filename, type = 'output', subfolder = '') =>
      BASE + '/view?filename=' + encodeURIComponent(filename) + '&type=' + type + '&subfolder=' + encodeURIComponent(subfolder),
    uploadImage: (blob, name = 'image.png') => { const f = new FormData(); f.append('image', blob, name);
      return fetch(BASE + '/upload/image', { method: 'POST', body: f }).then(j); },
    uploadMask: (blob, originalRef, name = 'mask.png') => { const f = new FormData();
      f.append('image', blob, name); if (originalRef) f.append('original_ref', JSON.stringify(originalRef));
      return fetch(BASE + '/upload/mask', { method: 'POST', body: f }).then(j); },
    // controlnet preprocessors + SAM masking (Tier B)
    preprocess: (method, payload) =>
      fetch(BASE + '/canvas/preprocess/' + method, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload) }).then(j),
    sam: (kind, payload) =>   // kind: 'text'|'points'|'exemplar'|'video'
      fetch(BASE + '/canvas/sam3/' + kind, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload) }).then(j),
    // websocket: progress + executing + b64 preview frames. cb({type,data})
    connectWS(clientId, cb) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = (BASE ? BASE.replace(/^http/, 'ws') : proto + '//' + location.host) + '/ws?clientId=' + clientId;
      let ws;
      try { ws = new WebSocket(url); } catch (e) { console.warn('[ws]', e); return { close() {} }; }
      ws.binaryType = 'arraybuffer';
      ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') { try { cb(JSON.parse(ev.data)); } catch (_) {} }
        else cb({ type: 'preview', data: ev.data });   // binary preview frame
      };
      ws.onerror = (e) => console.warn('[ws] error', e);
      return ws;
    },
  };
  Serenity.api = api;
  Serenity.clientId = 'serenity-' + Math.random().toString(36).slice(2);
})();
