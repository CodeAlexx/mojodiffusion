/* api.js — SHARED CONTRACT. Adapter from the canvas's ComfyUI-shaped calls to
   serenity-server's PROVEN /v1/* API (POST /v1/generate -> worker -> PNG).
     submitPrompt -> POST /v1/generate (workflow graph when provided, else workflow params)
     connectWS    -> WS /v1/progress?job=<id>, {ev:...} -> ComfyUI-shaped messages
     viewUrl      -> /out/<filename>
   IMPORTANT: generate_ws opens the WS BEFORE submit, but serenity's /v1/progress is
   JOB-keyed and the job id only exists after submit. So connectWS DEFERS: it stashes
   the callback, and submitPrompt opens the real WS once it has the job id. */
(function () {
  "use strict";
  window.Serenity = window.Serenity || {};
  var BASE = new URLSearchParams(location.search).get('api') || '';
  var j = function (r) {
    if (r.ok) return r.json();
    return r.text().then(function (txt) {
      var parsed = null;
      var msg = txt || (r.status + ' ' + r.url);
      try {
        parsed = JSON.parse(txt);
        if (parsed && (parsed.error || parsed.detail)) msg = parsed.error || parsed.detail;
      } catch (_) {}
      var err = new Error(msg);
      err.status = r.status;
      err.responseText = txt;
      err.response = parsed;
      if (parsed && parsed.schema === "serenity.generate.error.v1") {
        err.name = "GenerateRejected";
        err.generateError = parsed;
        err.capabilityProfile = parsed.capability_profile || null;
      }
      throw err;
    });
  };
  var lastJobId = null;
  var pendingCb = null;   // callback waiting for the job id (connectWS called pre-submit)
  var activeWS = null;
  var capabilityCache = null;
  var capabilityPromise = null;

  function baseName(p) { return String(p || '').split('/').pop(); }
  function postJSON(path, body) {
    return fetch(BASE + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j);
  }

  function normalizeBackendName(name) {
    return String(name || "").trim().toLowerCase();
  }

  function backendForModelName(model) {
    var m = String(model || "").toLowerCase();
    if (m.indexOf("ideogram") >= 0) return "ideogram4";
    if (m.indexOf("qwen") >= 0) return "qwenimage";
    if (m.indexOf("sdxl") >= 0 || m.indexOf("sd_xl") >= 0 ||
        m.indexOf("sd-xl") >= 0 || m.indexOf("sd xl") >= 0 ||
        m.indexOf("stable-diffusion-xl") >= 0 || m.indexOf("animagine") >= 0) return "sdxl";
    if (m.indexOf("anima") >= 0) return "anima";
    if (m.indexOf("sd3") >= 0 || m.indexOf("sd35") >= 0 || m.indexOf("sd3.5") >= 0) return "sd3";
    if (m.indexOf("flux2") >= 0 || m.indexOf("flux-2") >= 0 ||
        m.indexOf("flux_2") >= 0 || m.indexOf("klein") >= 0) return "flux2";
    if (m.indexOf("flux") >= 0) return "flux";
    if (m.indexOf("zimage") >= 0 || m.indexOf("z-image") >= 0 || m.indexOf("z_image") >= 0) {
      return "zimage";
    }
    return "zimage";
  }

  function normalizeModelName(model) {
    var raw = String(model || "").trim();
    if (!raw || raw.indexOf("—") >= 0) return "z-image";
    var lower = raw.toLowerCase();
    if (/^z[\s_-]*image(?:\s*\([^)]*\))?$/.test(lower) || lower === "zimage") {
      return "z-image";
    }
    if (/^ideogram\s*4(?:\s*\([^)]*\))?$/.test(lower) || lower === "ideogram4") {
      return "ideogram4";
    }
    return raw;
  }

  function capabilities(force) {
    if (!force && capabilityCache) return Promise.resolve(capabilityCache);
    if (!force && capabilityPromise) return capabilityPromise;
    capabilityPromise = fetch(BASE + '/v1/capabilities').then(j).then(function (info) {
      capabilityCache = info || null;
      capabilityPromise = null;
      if (window.Serenity && window.Serenity.bus) {
        window.Serenity.bus.emit("capabilities:loaded", capabilityCache);
      }
      return capabilityCache;
    }, function (err) {
      capabilityPromise = null;
      throw err;
    });
    return capabilityPromise;
  }

  function capabilityForBackend(backend) {
    var b = normalizeBackendName(backend);
    var arr = (capabilityCache && capabilityCache.backends) || [];
    for (var i = 0; i < arr.length; i++) {
      if (arr[i] && normalizeBackendName(arr[i].backend) === b) return arr[i];
    }
    return null;
  }

  function capabilityForModel(model) {
    return capabilityForBackend(backendForModelName(model));
  }

  function featureForModel(model, featureName) {
    var cap = capabilityForModel(model);
    return cap && cap.features ? cap.features[featureName] || null : null;
  }

  function featureSupportedForModel(model, featureName, fallback) {
    var feature = featureForModel(model, featureName);
    if (feature && feature.supported === true) return true;
    if (feature && feature.supported === false) return false;
    return !!fallback;
  }

  function anyFeatureSupportedForModel(model, featureNames, fallback) {
    var cap = capabilityForModel(model);
    if (!cap || !cap.features) return !!fallback;
    var sawKnown = false;
    for (var i = 0; i < (featureNames || []).length; i++) {
      var feature = cap.features[featureNames[i]];
      if (!feature) continue;
      sawKnown = true;
      if (feature.supported === true) return true;
    }
    return sawKnown ? false : !!fallback;
  }

  function generateBody() {
    var p = (window.Serenity.state && window.Serenity.state.params) || {};
    return {
      model: normalizeModelName(p.model),
      prompt: p.prompt || '', negative: p.negative || '',
      prompt_raw: p.prompt_raw || '', negative_raw: p.negative_raw || '',
      prompt_json: p.prompt_json != null ? p.prompt_json : undefined,
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
      // SwarmUI-parity fields from the canvas modules. Production-admitted
      // values are forwarded; image-conditioning/refine/upscale surfaces use
      // explicit txt2img sentinels until the Rust route admits them.
      batch_size: p.batch_size, end_steps_early_pct: p.end_steps_early_pct,
      no_seed_increment: p.no_seed_increment,
      denoise: p.denoise != null ? p.denoise : 1.0,
      creativity: p.creativity != null ? p.creativity : (p.denoise != null ? p.denoise : 1.0),
      loras: p.loras, refiner: null,
      // Text-to-image production route: image-conditioning, inpaint, and hires
      // img2img refine are intentionally not forwarded in this slice.
      hires_scale: 1.0, hires_denoise: 0.4,
      init_image: '', mask_image: '',
      lanpaint_mask_channel: '',
      outpaint: null, outpaint_enabled: false,
      refiner_model: '', refiner_steps: 0,
      refiner_cfg: -1, refiner_method: '',
      refiner_control: -1, refiner_tiling: false,
      upscaler: null, upscaler_model: '', upscale_by: 1.0,
    };
  }

  function hasWorkflowGraph(graph) {
    return !!(graph && typeof graph === "object" && !Array.isArray(graph) &&
      Object.keys(graph).length);
  }

  function workflowSubmitBody(graph) {
    return {
      workflow: graph,
      workflow_client: "serenity.canvas.generate_ws",
    };
  }

  function workflowParamsSubmitBody(params) {
    return {
      workflow: { params: params },
      workflow_client: "serenity.canvas.generate_ws",
    };
  }

  function submitBody(graph) {
    return hasWorkflowGraph(graph) ? workflowSubmitBody(graph) : workflowParamsSubmitBody(generateBody());
  }

  function preflightMessage(r) {
    var bp = (r && r.block_profile) || {};
    var ap = (r && r.artifact_profile) || {};
    var cp = (r && r.capability_profile) || {};
    var parts = [];
    if (r && r.error) parts.push(r.error);
    if (r && r.backend) parts.push("backend " + r.backend);
    else if (cp.backend) parts.push("backend " + cp.backend);
    if (r && r.rejection_stage) parts.push("stage " + r.rejection_stage);
    if (cp.production_status) parts.push("status " + cp.production_status);
    if (bp.block_count != null) parts.push("blocks " + bp.block_count);
    if (ap.missing_count != null && ap.missing_count > 0) {
      parts.push("missing artifacts " + ap.missing_count);
    }
    return parts.length ? parts.join(" | ") : "request not admitted";
  }

  function requirePreflight(body) {
    return postJSON('/v1/preflight', body).then(function (r) {
      if (!r || r.admitted !== true) {
        var err = new Error("Preflight blocked: " + preflightMessage(r));
        err.name = "PreflightBlocked";
        err.preflight = r || null;
        throw err;
      }
      return r;
    });
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
      var body = submitBody(_graph);
      return requirePreflight(body).then(function (pf) {
        if (window.Serenity && window.Serenity.bus) window.Serenity.bus.emit("generate:preflight", pf);
        return postJSON('/v1/generate', body);
      }).then(function (res) {
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
    preflight: function (body) {
      return postJSON('/v1/preflight', body || submitBody(null));
    },
    models: function () { return fetch(BASE + '/v1/models').then(j); },
    samplers: function () { return fetch(BASE + '/v1/samplers').then(j); },
    capabilities: capabilities,
    backendForModelName: backendForModelName,
    normalizeModelName: normalizeModelName,
    capabilityForBackend: capabilityForBackend,
    capabilityForModel: capabilityForModel,
    featureForModel: featureForModel,
    featureSupportedForModel: featureSupportedForModel,
    anyFeatureSupportedForModel: anyFeatureSupportedForModel,
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
