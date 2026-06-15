/* main.js — boot. Owned by scaffold. Builds the shared ctx and inits every
   registered module. Modules that aren't built yet are skipped (graceful). */
(function () {
  "use strict";
  const S = window.Serenity;
  const EXPECTED = ["topbar","paramRail","canvasCore","bbox","layers","toolsBrush","controlnet","sam","generateWS","gallery"];

  function boot() {
    const ctx = {
      state: S.state, get: S.get, set: S.set, bus: S.bus, api: S.api,
      Konva: window.Konva, clientId: S.clientId,
      dom: {
        topbar: document.getElementById('topbar'),
        tooldock: document.getElementById('tooldock'),
        paramRail: document.getElementById('param-rail'),
        canvasHost: document.getElementById('canvas-host'),
        canvasToolbar: document.getElementById('canvas-toolbar'),
        konvaStage: document.getElementById('konva-stage'),
        canvasStatusbar: document.getElementById('canvas-statusbar'),
        layersPanel: document.getElementById('layers-panel'),
        gallery: document.getElementById('gallery'),
        queueStrip: document.getElementById('queue-strip'),
      },
    };
    window.Serenity.ctx = ctx;
    EXPECTED.forEach((name) => {
      const mod = S.modules[name];
      if (!mod || typeof mod.init !== 'function') { console.warn('[boot] module not ready:', name); return; }
      try { mod.init(ctx); console.info('[boot] init', name); }
      catch (e) { console.error('[boot] init failed:', name, e); }
    });
    S.bus.emit('app:ready', ctx);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
