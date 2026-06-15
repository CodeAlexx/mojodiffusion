/* state.js — SHARED CONTRACT. Owned by scaffold; do NOT rewrite in a module.
   A tiny reactive store + event bus that every module talks through, so modules
   never import each other directly (avoids 30-agent file coupling). */
(function () {
  "use strict";
  window.Serenity = window.Serenity || { modules: {} };

  // --- module registry: each module file calls Serenity.register('name', {init(ctx)}) ---
  Serenity.register = function (name, mod) { Serenity.modules[name] = mod; };

  // --- event bus ---
  const listeners = {};
  const bus = {
    on(evt, fn) { (listeners[evt] = listeners[evt] || []).push(fn); return () => bus.off(evt, fn); },
    off(evt, fn) { listeners[evt] = (listeners[evt] || []).filter(f => f !== fn); },
    emit(evt, payload) { (listeners[evt] || []).forEach(fn => { try { fn(payload); } catch (e) { console.error('[bus]', evt, e); } }); },
  };

  // --- shared application state (single source of truth) ---
  // params: the SwarmUI generate controls (left rail writes these; generate_ws reads them)
  // layers: the InvokeAI layer stack (layers/canvas modules own; generate_ws reads)
  // canvas: stage/bbox geometry (canvas_core/bbox own)
  const state = {
    params: {
      model: "", vae: "", prompt: "", negative: "",
      width: 1024, height: 1024, aspect: "1:1",
      steps: 8, cfg: 1.5, seed: -1, images: 1,
      sampler: "euler", scheduler: "simple",
      // advanced (revealed by [adv]; NOT a separate screen)
      clip_skip: 0, sigma_min: -1, sigma_max: -1, eta: -1, restart_sampling: false,
      loras: [], controlnet: null, refiner: null, denoise: 1.0,
    },
    layers: [],            // [{id,type:'raster'|'control'|'mask'|'regional'|'reference',name,visible,opacity,locked,...}]
    activeLayerId: null,
    canvas: { bbox: { x: 0, y: 0, width: 1024, height: 1024 }, zoom: 1 },
    progress: { running: false, step: 0, total: 0, jobId: null },
    gallery: [],           // [{id, filename, params}]
    ui: { advanced: false, activeTool: "move" },
  };

  // get/set with change events. set('params.steps', 8) emits 'change:params.steps' + 'change'
  function getPath(p) { return p.split('.').reduce((o, k) => (o == null ? o : o[k]), state); }
  function setPath(p, v) {
    const ks = p.split('.'); const last = ks.pop();
    const obj = ks.reduce((o, k) => (o[k] = o[k] || {}), state);
    obj[last] = v; bus.emit('change:' + p, v); bus.emit('change', { path: p, value: v });
  }

  Serenity.bus = bus;
  Serenity.state = state;
  Serenity.get = getPath;
  Serenity.set = setPath;
})();
