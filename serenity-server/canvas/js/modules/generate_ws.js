/* generate_ws.js — module 'generateWS'.
 * Owns: state.progress. Mounts into the param-rail Generate button + #queue-strip,
 * plus a Konva preview overlay group on the shared stage.
 *
 * Responsibilities (per CONTRACT.md table):
 *  - Bind #btn-generate (rendered by paramRail) — found on bus 'app:ready'.
 *  - On click: assemble a ComfyUI workflow GRAPH from state.params + state.layers
 *      (checkpoint/clip/vae loaders, CLIPTextEncode pos+neg, KSampler, EmptyLatentImage
 *       from bbox dims, LoRA chain, ControlNetApply per control layer,
 *       SetLatentNoiseMask per mask layer, SaveImage).
 *  - Upload layer pixels only when the workflow graph contains image/mask/control
 *    nodes; the Rust workflow lowerer and capability gate decide admission.
 *  - POST via api.submitPrompt; open api.connectWS.
 *  - On 'progress': update state.progress + a progress bar in #queue-strip.
 *  - On binary 'preview': draw the frame into the stage overlay.
 *  - On done: fetch via api.viewUrl and emit 'result:ready'.
 *  - Interrupt button -> api.interrupt.
 *  - Degrade gracefully on 404 (show the assembled graph in console; never crash).
 *
 * Talks to the rest of the app ONLY through ctx.{state,get,set,bus,api,Konva,clientId,dom}.
 */
(function () {
  "use strict";
  const NAME = "generateWS";

  Serenity.register(NAME, {
    init(ctx) {
      const { get, set, bus, api, Konva, clientId, dom } = ctx;

      // ---- module-local CSS (scoped to ids we own) ---------------------------
      injectCSS();

      // ---- queue-strip UI (progress bar + interrupt) -------------------------
      const ui = buildQueueStrip(dom.queueStrip);

      // ---- Konva preview overlay (created lazily once the stage exists) -------
      let previewLayer = null;   // Konva.Layer dedicated to live previews
      let previewImage = null;   // Konva.Image showing the current b64/binary frame
      let stageRef = ctx.state && ctx.state.canvas ? ctx.state.canvas.stage : null;

      function ensurePreviewLayer() {
        const stage = stageRef || (ctx.state.canvas && ctx.state.canvas.stage);
        if (!stage || !Konva) return null;
        stageRef = stage;
        if (!previewLayer) {
          try {
            previewLayer = new Konva.Layer({ listening: false, name: "generate-preview" });
            stage.add(previewLayer);
          } catch (e) {
            console.warn("[generateWS] could not add preview layer", e);
            return null;
          }
        }
        return previewLayer;
      }

      // stage may come up after us; canvasCore announces it on this bus event
      bus.on("canvas:stage", (stage) => { if (stage) { stageRef = stage; } });

      // ---- runtime job state -------------------------------------------------
      let ws = null;          // active WebSocket (from api.connectWS)
      let wsHandle = null;    // the object connectWS returned (ws or {close})
      let currentPromptId = null;
      let submitting = false;

      // ---- find + bind the Generate button (rendered by paramRail) -----------
      let boundBtn = null;
      function bindGenerate() {
        const btn = document.getElementById("btn-generate");
        if (!btn || btn === boundBtn) return;
        boundBtn = btn;
        btn.addEventListener("click", onGenerateClick);
      }
      bus.on("app:ready", () => { bindGenerate(); });
      // also try immediately in case app:ready already fired before we listened
      bindGenerate();
      // The button might be (re)rendered later by paramRail; retry a few times.
      let retries = 0;
      const retryTimer = setInterval(() => {
        bindGenerate();
        if (boundBtn || ++retries > 20) clearInterval(retryTimer);
      }, 200);

      // also allow other modules to trigger generation via bus
      bus.on("generate:request", () => onGenerateClick());

      // ===== GENERATE =========================================================
      async function onGenerateClick() {
        if (submitting || get("progress.running")) {
          console.info("[generateWS] already running; ignoring click");
          return;
        }
        submitting = true;
        setBusy(true);
        try {
          const graph = await assembleGraph();
          console.info("[generateWS] assembled ComfyUI graph:", graph);

          // open the WS first so we don't miss early progress events
          openWS();

          let res;
          try {
            res = await api.submitPrompt(graph, clientId);
          } catch (e) {
            if (e && e.name === "PreflightBlocked") {
              console.warn("[generateWS] preflight blocked:", e.preflight || e);
              console.info("[generateWS] graph that was blocked before enqueue:\n" +
                JSON.stringify(graph, null, 2));
              ui.setStatus(e.message || "preflight blocked");
              bus.emit("generate:preflight_blocked", {
                error: e.message || String(e),
                preflight: e.preflight || null,
                graph,
              });
              setProgress({ running: false, step: 0, total: 0, jobId: null });
              closeWS();
              return;
            }
            if (e && e.name === "GenerateRejected") {
              const response = e.generateError || e.response || null;
              console.warn("[generateWS] generate rejected before enqueue:", response || e);
              console.info("[generateWS] graph rejected by /v1/generate:\n" +
                JSON.stringify(graph, null, 2));
              const stage = response && response.rejection_stage ? " · " + response.rejection_stage : "";
              ui.setStatus("generate blocked: " + (e.message || "request rejected") + stage);
              bus.emit("generate:rejected", {
                error: e.message || String(e),
                status: e.status || 0,
                response,
                graph,
              });
              setProgress({ running: false, step: 0, total: 0, jobId: null });
              closeWS();
              return;
            }
            // backend not up yet (404 / network) — degrade gracefully
            console.warn("[generateWS] submitPrompt failed (backend not ready?):", e);
            console.info("[generateWS] graph that WOULD have been submitted:\n" +
              JSON.stringify(graph, null, 2));
            ui.setStatus("backend offline — graph logged to console");
            setProgress({ running: false, step: 0, total: 0, jobId: null });
            closeWS();
            return;
          }

          currentPromptId = (res && (res.prompt_id || res.promptId)) || null;
          const total = Number(get("params.steps")) || 0;
          setProgress({ running: true, step: 0, total, jobId: currentPromptId });
          ui.setStatus("queued" + (currentPromptId ? " · " + shortId(currentPromptId) : ""));
          bus.emit("generate:submitted", { promptId: currentPromptId, graph });
        } catch (e) {
          console.error("[generateWS] generate failed:", e);
          ui.setStatus("error: " + (e && e.message ? e.message : e));
          setProgress({ running: false, step: 0, total: 0, jobId: null });
        } finally {
          submitting = false;
          setBusy(false);
        }
      }

      // ===== GRAPH ASSEMBLY ===================================================
      // Build a ComfyUI prompt graph (node-id -> {class_type, inputs}).
      async function assembleGraph() {
        const p = readParams();
        const layers = Array.isArray(get("layers")) ? get("layers") : [];
        const modelName = api && typeof api.normalizeModelName === "function"
          ? api.normalizeModelName(p.model)
          : (p.model || "z-image");
        const backendName = api && typeof api.backendForModelName === "function"
          ? api.backendForModelName(modelName)
          : "";

        // Ideogram4's production route is not the generic CLIPTextEncode graph:
        // its prompt contract is the structured prompt_json/bbox payload. Submit
        // the raw workflow params path so api.generateBody preserves prompt_json.
        if (backendName === "ideogram4" && p.prompt_json != null) {
          return null;
        }

        const g = {};
        let nid = 0;
        const id = () => String(++nid);

        // --- model loader: checkpoint (+ separate clip/vae if specified) ------
        const ckptId = id();
        g[ckptId] = {
          class_type: "CheckpointLoaderSimple",
          inputs: { ckpt_name: modelName || "z-image" },
        };
        // MODEL / CLIP / VAE source slots — may be rebound by loaders below.
        let modelSlot = [ckptId, 0];
        let clipSlot = [ckptId, 1];
        let vaeSlot = [ckptId, 2];

        // optional explicit VAE loader (overrides the checkpoint VAE)
        if (p.vae && p.vae !== "" && !/baked|default|automatic/i.test(p.vae)) {
          const vaeId = id();
          g[vaeId] = { class_type: "VAELoader", inputs: { vae_name: p.vae } };
          vaeSlot = [vaeId, 0];
        }

        // --- LoRA chain: each LoRA reroutes MODEL + CLIP -----------------------
        const loras = collectLoras(p, layers);
        for (const lo of loras) {
          const lid = id();
          g[lid] = {
            class_type: "LoraLoader",
            inputs: {
              lora_name: lo.name,
              strength_model: num(lo.strength_model, num(lo.strength, 1.0)),
              strength_clip: num(lo.strength_clip, num(lo.strength, 1.0)),
              model: modelSlot,
              clip: clipSlot,
            },
          };
          modelSlot = [lid, 0];
          clipSlot = [lid, 1];
        }

        // --- text conditioning -------------------------------------------------
        const posId = id();
        g[posId] = {
          class_type: "CLIPTextEncode",
          inputs: { text: p.prompt || "", clip: clipSlot },
        };
        const negId = id();
        g[negId] = {
          class_type: "CLIPTextEncode",
          inputs: { text: p.negative || "", clip: clipSlot },
        };
        let positiveSlot = [posId, 0];
        let negativeSlot = [negId, 0];

        // --- ControlNet: one ControlNetLoader + ControlNetApply per control layer
        const controlLayers = layers.filter(
          (l) => l && l.type === "control" && l.visible !== false
        );
        for (const cl of controlLayers) {
          const cnName = cl.controlnet || cl.model || cl.cnModel;
          if (!cnName) continue;
          // upload the (preprocessed) control image if we have pixels for it
          let imgRef = cl.uploadedPath || cl.uploadedName || null;
          if (!imgRef) imgRef = await maybeUploadLayerImage(cl, "control");
          if (!imgRef) continue;

          const loadImgId = id();
          g[loadImgId] = { class_type: "LoadImage", inputs: { image: imgRef } };

          const cnLoadId = id();
          g[cnLoadId] = {
            class_type: "ControlNetLoader",
            inputs: { control_net_name: cnName },
          };

          const cnApplyId = id();
          g[cnApplyId] = {
            class_type: "ControlNetApply",
            inputs: {
              conditioning: positiveSlot,
              control_net: [cnLoadId, 0],
              image: [loadImgId, 0],
              strength: num(cl.strength, 1.0),
            },
          };
          positiveSlot = [cnApplyId, 0];
        }

        // --- latent source: init image (img2img) OR empty latent --------------
        const rasterLayer = layers.find(
          (l) => l && (l.type === "raster" || l.type === "reference") &&
                 l.visible !== false && hasPixels(l)
        );
        let latentSlot;
        let denoise = num(p.denoise, 1.0);

        if (rasterLayer) {
          let initRef = rasterLayer.uploadedPath || rasterLayer.uploadedName || null;
          if (!initRef) initRef = await maybeUploadLayerImage(rasterLayer, "init");
          if (initRef) {
            const loadInitId = id();
            g[loadInitId] = { class_type: "LoadImage", inputs: { image: initRef } };
            const encId = id();
            g[encId] = {
              class_type: "VAEEncode",
              inputs: { pixels: [loadInitId, 0], vae: vaeSlot },
            };
            latentSlot = [encId, 0];
            // for true img2img the denoise should be < 1; respect explicit param,
            // otherwise leave it as-is (1.0 == full regen even on an init image).
          }
        }
        if (!latentSlot) {
          // EmptyLatentImage from the bbox dims (fall back to params w/h)
          const dims = readDims(p);
          const emptyId = id();
          g[emptyId] = {
            class_type: "EmptyLatentImage",
            inputs: {
              width: dims.width,
              height: dims.height,
              batch_size: Math.max(1, num(p.images, 1)),
            },
          };
          latentSlot = [emptyId, 0];
        }

        // --- SetLatentNoiseMask per mask layer (inpaint) ----------------------
        const maskLayers = layers.filter(
          (l) => l && l.type === "mask" && l.visible !== false
        );
        for (const ml of maskLayers) {
          let maskRef = ml.uploadedPath || ml.uploadedName || null;
          if (!maskRef) maskRef = await maybeUploadLayerMask(ml, rasterLayer);
          if (!maskRef) continue;
          const loadMaskId = id();
          g[loadMaskId] = { class_type: "LoadImage", inputs: { image: maskRef } };
          const setMaskId = id();
          g[setMaskId] = {
            class_type: "SetLatentNoiseMask",
            inputs: {
              samples: latentSlot,
              mask: [loadMaskId, 1], // LoadImage slot 1 == mask/alpha channel
            },
          };
          latentSlot = [setMaskId, 0];
        }

        // --- KSampler ----------------------------------------------------------
        const ksId = id();
        g[ksId] = {
          class_type: "KSampler",
          inputs: {
            seed: resolveSeed(p.seed),
            steps: Math.max(1, num(p.steps, 8)),
            cfg: num(p.cfg, 1.5),
            sampler_name: p.sampler || "euler",
            scheduler: p.scheduler || "simple",
            denoise: denoise,
            model: modelSlot,
            positive: positiveSlot,
            negative: negativeSlot,
            latent_image: latentSlot,
          },
        };

        // --- decode + save -----------------------------------------------------
        const decId = id();
        g[decId] = {
          class_type: "VAEDecode",
          inputs: { samples: [ksId, 0], vae: vaeSlot },
        };
        const saveId = id();
        g[saveId] = {
          class_type: "SaveImage",
          inputs: { images: [decId, 0], filename_prefix: "serenity" },
        };
        if (hasRefinerUpscaleIntent(p)) {
          const ruId = id();
          g[ruId] = {
            class_type: "SerenityRefinerUpscaleIntent",
            inputs: refinerUpscaleIntentInputs(p),
          };
        }

        return g;
      }

      // ===== layer image / mask upload helpers ===============================
      // Returns an uploaded on-disk path usable by the Rust workflow lowerer, or null.
      async function maybeUploadLayerImage(layer, kind) {
        const blob = await layerToBlob(layer);
        if (!blob) return null;
        try {
          const name = (layer.name || kind || "layer").replace(/\s+/g, "_") + ".png";
          const res = await api.uploadImage(blob, name);
          const ref = (res && (res.path || res.name || res.filename)) || name;
          layer.uploadedPath = ref;
          layer.uploadedName = (res && (res.name || res.filename)) || name;
          return ref;
        } catch (e) {
          console.warn("[generateWS] uploadImage failed for layer", layer && layer.id, e);
          return null;
        }
      }

      async function maybeUploadLayerMask(layer, rasterLayer) {
        const blob = await layerToBlob(layer);
        if (!blob) return null;
        try {
          const name = (layer.name || "mask").replace(/\s+/g, "_") + ".png";
          // if there's a base raster we uploaded, reference it for mask compositing
          const ref = rasterLayer && rasterLayer.uploadedName
            ? { filename: rasterLayer.uploadedName, type: "input", subfolder: "" }
            : null;
          const res = await api.uploadMask(blob, ref, name);
          const out = (res && (res.path || res.name || res.filename)) || name;
          layer.uploadedPath = out;
          layer.uploadedName = (res && (res.name || res.filename)) || name;
          return out;
        } catch (e) {
          console.warn("[generateWS] uploadMask failed for layer", layer && layer.id, e);
          return null;
        }
      }

      // Extract a PNG Blob from a layer. Supports several shapes a layer module
      // might expose: an explicit blob, a dataURL, an <img>/<canvas>/ImageData,
      // or a Konva node (group/image) we can rasterise.
      async function layerToBlob(layer) {
        if (!layer) return null;
        try {
          if (layer.blob instanceof Blob) return layer.blob;
          if (typeof layer.dataURL === "string") return dataURLToBlob(layer.dataURL);
          if (typeof layer.src === "string" && layer.src.startsWith("data:"))
            return dataURLToBlob(layer.src);

          const node = layer.konvaNode || layer.node || layer.group || layer.image;
          if (node && typeof node.toCanvas === "function") {
            const cv = node.toCanvas();
            return await canvasToBlob(cv);
          }
          if (node && typeof node.toDataURL === "function") {
            return dataURLToBlob(node.toDataURL());
          }
          // raw <canvas> / <img>
          if (layer.canvas && typeof layer.canvas.toBlob === "function")
            return await canvasToBlob(layer.canvas);
          if (layer.imageEl && layer.imageEl.tagName === "IMG")
            return await imgElToBlob(layer.imageEl);
        } catch (e) {
          console.warn("[generateWS] layerToBlob failed", e);
        }
        return null;
      }

      // ===== WebSocket: progress + previews ===================================
      function openWS() {
        closeWS();
        try {
          wsHandle = api.connectWS(clientId, onWSMessage);
          ws = (wsHandle && typeof wsHandle.close === "function" && wsHandle.binaryType !== undefined)
            ? wsHandle : (wsHandle || null);
        } catch (e) {
          console.warn("[generateWS] connectWS failed:", e);
          wsHandle = null;
        }
      }
      function closeWS() {
        try { if (wsHandle && typeof wsHandle.close === "function") wsHandle.close(); }
        catch (_) {}
        wsHandle = null; ws = null;
      }

      function onWSMessage(msg) {
        if (!msg) return;
        // binary preview frame (api.js wraps it as {type:'preview', data: ArrayBuffer})
        if (msg.type === "preview") { drawPreview(msg.data); return; }

        const type = msg.type;
        const data = msg.data || {};

        // filter to our prompt where possible (ComfyUI tags data.prompt_id)
        const sameJob = !currentPromptId || !data.prompt_id ||
          data.prompt_id === currentPromptId;

        if (type === "progress") {
          if (!sameJob) return;
          const step = num(data.value, 0);
          const total = num(data.max, get("progress.total") || 0);
          setProgress({ running: true, step, total, jobId: currentPromptId });
          ui.setBar(total ? step / total : 0);
          ui.setStatus("step " + step + "/" + total);
        } else if (type === "executing") {
          if (!sameJob) return;
          if (data.node == null) {
            // node === null => this prompt finished executing
            onDone();
          } else {
            ui.setStatus("running node " + data.node);
          }
        } else if (type === "executed") {
          if (!sameJob) return;
          handleExecuted(data);
        } else if (type === "execution_error") {
          if (!sameJob) return;
          console.error("[generateWS] execution_error", data);
          ui.setStatus("execution error");
          finishJob();
        } else if (type === "status") {
          const q = data.status && data.status.exec_info &&
                    data.status.exec_info.queue_remaining;
          if (q != null) ui.setQueue(q);
        }
      }

      // collect output images from an 'executed' message (SaveImage outputs)
      function handleExecuted(data) {
        const out = data.output || {};
        const imgs = out.images || [];
        for (const im of imgs) {
          if (!im || !im.filename) continue;
          const url = api.viewUrl(im.filename, im.type || "output", im.subfolder || "");
          bus.emit("result:ready", {
            url,
            filename: im.filename,
            type: im.type || "output",
            subfolder: im.subfolder || "",
            promptId: currentPromptId,
            params: readParams(),
          });
          ui.setStatus("done");
        }
      }

      function onDone() {
        ui.setBar(1);
        ui.setStatus("done");
        // executing(node=null) means finished; if no 'executed' carried images,
        // try fetching history for this prompt to recover output filenames.
        if (currentPromptId) {
          fetchResultsFromHistory(currentPromptId);
        }
        finishJob();
      }

      async function fetchResultsFromHistory(promptId) {
        if (typeof api.history !== "function") return;
        try {
          const h = await api.history(promptId);
          const entry = h && (h[promptId] || h);
          const outputs = entry && entry.outputs;
          if (!outputs) return;
          for (const nodeId of Object.keys(outputs)) {
            const imgs = (outputs[nodeId] && outputs[nodeId].images) || [];
            for (const im of imgs) {
              if (!im || !im.filename) continue;
              const url = api.viewUrl(im.filename, im.type || "output", im.subfolder || "");
              bus.emit("result:ready", {
                url, filename: im.filename, type: im.type || "output",
                subfolder: im.subfolder || "", promptId, params: readParams(),
              });
            }
          }
        } catch (e) {
          // history may 404 too — that's fine, the 'executed' path usually covers it
          console.warn("[generateWS] history fetch failed:", e);
        }
      }

      function finishJob() {
        setProgress({ running: false, step: 0, total: 0, jobId: null });
        // clear the live preview shortly after; the final result is the gallery's job
        clearPreview();
        closeWS();
        currentPromptId = null;
      }

      // ===== preview drawing into the stage overlay ===========================
      function drawPreview(payload) {
        const layer = ensurePreviewLayer();
        if (!layer || !Konva) return;
        toImageBitmapSource(payload).then((srcUrl) => {
          if (!srcUrl) return;
          const imgEl = new Image();
          imgEl.onload = () => {
            try {
              const stage = stageRef;
              const sw = stage ? stage.width() : imgEl.width;
              const sh = stage ? stage.height() : imgEl.height;
              const scale = Math.min(sw / imgEl.width, sh / imgEl.height) || 1;
              const w = imgEl.width * scale, h = imgEl.height * scale;
              const x = (sw - w) / 2, y = (sh - h) / 2;
              if (!previewImage) {
                previewImage = new Konva.Image({
                  image: imgEl, x, y, width: w, height: h, opacity: 0.95,
                  listening: false,
                });
                layer.add(previewImage);
              } else {
                previewImage.image(imgEl);
                previewImage.setAttrs({ x, y, width: w, height: h });
              }
              layer.batchDraw();
            } catch (e) {
              console.warn("[generateWS] drawPreview render failed", e);
            } finally {
              if (srcUrl.startsWith && srcUrl.startsWith("blob:")) {
                // revoke after the image is consumed
                setTimeout(() => { try { URL.revokeObjectURL(srcUrl); } catch (_) {} }, 0);
              }
            }
          };
          imgEl.onerror = () => {
            if (srcUrl.startsWith && srcUrl.startsWith("blob:")) {
              try { URL.revokeObjectURL(srcUrl); } catch (_) {}
            }
          };
          imgEl.src = srcUrl;
        });
      }

      function clearPreview() {
        try {
          if (previewImage) { previewImage.destroy(); previewImage = null; }
          if (previewLayer) previewLayer.batchDraw();
        } catch (_) {}
      }

      // ComfyUI binary previews are: 4-byte event + 4-byte image-type + JPEG/PNG.
      // We tolerate raw image bytes, a {data} ArrayBuffer, or a base64/string.
      async function toImageBitmapSource(payload) {
        try {
          if (!payload) return null;
          if (typeof payload === "string") {
            if (payload.startsWith("data:")) return payload;
            // assume base64 of an image
            return "data:image/png;base64," + payload;
          }
          let buf = payload;
          if (payload instanceof Blob) {
            return URL.createObjectURL(payload);
          }
          if (payload.data instanceof ArrayBuffer) buf = payload.data;
          if (buf instanceof ArrayBuffer) {
            const bytes = new Uint8Array(buf);
            // ComfyUI prepends an 8-byte header (event:uint32, format:uint32).
            // Detect a JPEG/PNG magic to decide whether to strip it.
            let start = 0;
            if (bytes.length > 8 && !isImageMagic(bytes, 0) && isImageMagic(bytes, 8)) {
              start = 8;
            }
            const view = bytes.subarray(start);
            const type = sniffMime(view);
            const blob = new Blob([view], { type });
            return URL.createObjectURL(blob);
          }
        } catch (e) {
          console.warn("[generateWS] toImageBitmapSource failed", e);
        }
        return null;
      }

      // ===== state.progress helpers ==========================================
      function setProgress(pr) {
        set("progress.running", !!pr.running);
        set("progress.step", num(pr.step, 0));
        set("progress.total", num(pr.total, 0));
        set("progress.jobId", pr.jobId != null ? pr.jobId : null);
        ui.setVisible(!!pr.running || submitting);
      }

      function setBusy(b) {
        if (boundBtn) {
          boundBtn.disabled = !!b;
          boundBtn.classList.toggle("is-busy", !!b);
        }
        ui.setVisible(b || get("progress.running"));
      }

      // ===== interrupt ========================================================
      function onInterrupt() {
        try {
          if (typeof api.interrupt === "function") {
            Promise.resolve(api.interrupt()).catch((e) =>
              console.warn("[generateWS] interrupt failed:", e));
          }
        } finally {
          ui.setStatus("interrupted");
          finishJob();
        }
      }
      ui.onInterrupt(onInterrupt);
      bus.on("generate:interrupt", onInterrupt);

      // ===== param / dims readers ============================================
      function readParams() {
        const params = get("params") || {};
        // shallow copy so downstream emits don't mutate the store
        return Object.assign({}, params);
      }
      function readDims(p) {
        // prefer the bbox geometry (the actual generation region), fall back to params
        const bbox = (get("canvas.bbox")) || null;
        let w = bbox && bbox.width ? bbox.width : p.width;
        let h = bbox && bbox.height ? bbox.height : p.height;
        w = clampDim(num(w, 1024));
        h = clampDim(num(h, 1024));
        return { width: w, height: h };
      }

      function collectLoras(p, layers) {
        const out = [];
        const fromParams = Array.isArray(p.loras) ? p.loras : [];
        for (const lo of fromParams) {
          if (!lo) continue;
          if (typeof lo === "string") { out.push({ name: lo, strength: 1.0 }); continue; }
          if (lo.enabled === false) continue;
          if (lo.name) out.push(lo);
        }
        return out;
      }

      function meaningfulObject(v) {
        return !!(v && typeof v === "object" && Object.keys(v).length > 0);
      }
      function hasRefinerUpscaleIntent(p) {
        const ref = meaningfulObject(p.refiner) ? p.refiner : null;
        const up = meaningfulObject(p.upscaler) ? p.upscaler : null;
        return !!(
          ref ||
          up ||
          num(p.hires_scale, 1.0) > 1.0 ||
          num(p.upscale_by, 1.0) > 1.0 ||
          hasText(p.refiner_model) ||
          hasText(p.refiner_method) ||
          num(p.refiner_steps, 0) > 0 ||
          num(p.refiner_cfg, -1) >= 0 ||
          num(p.refiner_control, -1) >= 0 ||
          !!p.refiner_tiling ||
          hasText(p.upscaler_model)
        );
      }
      function refinerUpscaleIntentInputs(p) {
        const ref = meaningfulObject(p.refiner) ? p.refiner : {};
        const up = meaningfulObject(p.upscaler) ? p.upscaler : {};
        const factor = num(
          up.factor != null ? up.factor : (p.upscale_by != null ? p.upscale_by : p.hires_scale),
          1.0
        );
        const control = num(
          ref.control != null ? ref.control : (p.refiner_control != null ? p.refiner_control : p.hires_denoise),
          0.4
        );
        return {
          enabled: ref.enabled !== false,
          refiner_model: ref.model || p.refiner_model || "",
          refiner_method: ref.method || p.refiner_method || "postapply",
          refiner_steps: num(ref.steps != null ? ref.steps : p.refiner_steps, 0),
          refiner_cfg: num(ref.cfg != null ? ref.cfg : p.refiner_cfg, -1),
          refiner_control: control,
          refiner_tiling: !!(ref.tiling != null ? ref.tiling : p.refiner_tiling),
          upscaler_model: up.model || p.upscaler_model || "",
          upscale_by: factor,
          hires_scale: num(p.hires_scale, factor),
          hires_denoise: num(p.hires_denoise, control),
        };
      }

      // ===== small utils ======================================================
      function readDimsSafe() { return readDims(readParams()); }
      function num(v, d) { const n = Number(v); return Number.isFinite(n) ? n : d; }
      function hasText(v) { return typeof v === "string" && v.trim().length > 0; }
      function clampDim(v) { v = Math.round(v / 8) * 8; return Math.max(64, Math.min(8192, v)); }
      function resolveSeed(seed) {
        const s = num(seed, -1);
        if (s < 0) return Math.floor(Math.random() * 0xffffffff);
        return s;
      }
      function hasPixels(l) {
        return !!(l && (l.blob || l.dataURL || l.src || l.konvaNode || l.node ||
          l.group || l.image || l.canvas || l.imageEl || l.uploadedPath || l.uploadedName));
      }
      function shortId(s) { return String(s).slice(0, 8); }

      function dataURLToBlob(dataURL) {
        try {
          const parts = dataURL.split(",");
          const mime = (parts[0].match(/data:([^;]+)/) || [, "image/png"])[1];
          const bin = atob(parts[1]);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          return new Blob([arr], { type: mime });
        } catch (e) { return null; }
      }
      function canvasToBlob(cv) {
        return new Promise((resolve) => {
          if (cv && typeof cv.toBlob === "function") cv.toBlob((b) => resolve(b), "image/png");
          else resolve(null);
        });
      }
      function imgElToBlob(img) {
        return new Promise((resolve) => {
          try {
            const cv = document.createElement("canvas");
            cv.width = img.naturalWidth || img.width;
            cv.height = img.naturalHeight || img.height;
            cv.getContext("2d").drawImage(img, 0, 0);
            cv.toBlob((b) => resolve(b), "image/png");
          } catch (_) { resolve(null); }
        });
      }
      function isImageMagic(bytes, off) {
        if (!bytes || bytes.length < off + 4) return false;
        // PNG 89 50 4E 47 ; JPEG FF D8 FF
        if (bytes[off] === 0x89 && bytes[off + 1] === 0x50 &&
            bytes[off + 2] === 0x4e && bytes[off + 3] === 0x47) return true;
        if (bytes[off] === 0xff && bytes[off + 1] === 0xd8 && bytes[off + 2] === 0xff) return true;
        return false;
      }
      function sniffMime(bytes) {
        if (bytes && bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png";
        return "image/jpeg";
      }

      // ===== queue-strip DOM ==================================================
      function buildQueueStrip(mount) {
        const root = document.createElement("div");
        root.id = "gen-queue";
        root.className = "gen-hidden";

        const bar = document.createElement("div");
        bar.className = "gen-bar";
        const fill = document.createElement("div");
        fill.className = "gen-bar-fill";
        bar.appendChild(fill);

        const meta = document.createElement("div");
        meta.className = "gen-meta";
        const status = document.createElement("span");
        status.className = "gen-status";
        status.textContent = "idle";
        const queue = document.createElement("span");
        queue.className = "gen-queue-count";
        queue.textContent = "";

        const stopBtn = document.createElement("button");
        stopBtn.className = "gen-interrupt";
        stopBtn.type = "button";
        stopBtn.title = "Interrupt generation";
        stopBtn.textContent = "■ Stop";

        meta.appendChild(status);
        meta.appendChild(queue);
        meta.appendChild(stopBtn);

        root.appendChild(meta);
        root.appendChild(bar);

        if (mount) mount.appendChild(root);
        else document.body.appendChild(root);

        return {
          el: root,
          setBar(frac) {
            const pct = Math.max(0, Math.min(1, num(frac, 0))) * 100;
            fill.style.width = pct.toFixed(1) + "%";
          },
          setStatus(txt) { status.textContent = txt == null ? "" : String(txt); },
          setQueue(n) {
            queue.textContent = (n && n > 0) ? ("· " + n + " queued") : "";
          },
          setVisible(v) { root.classList.toggle("gen-hidden", !v); },
          onInterrupt(fn) { stopBtn.addEventListener("click", fn); },
        };
      }

      // ===== scoped CSS =======================================================
      function injectCSS() {
        if (document.getElementById("style-" + NAME)) return;
        const css = `
#gen-queue{min-width:220px;background:var(--panel2);border:1px solid var(--line);
  border-radius:var(--radius);padding:8px 10px;box-shadow:0 6px 18px rgba(0,0,0,.35);
  font:12px/1.3 system-ui,sans-serif;color:var(--text)}
#gen-queue.gen-hidden{display:none}
#gen-queue .gen-meta{display:flex;align-items:center;gap:8px;margin-bottom:6px}
#gen-queue .gen-status{flex:1;color:var(--muted);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
#gen-queue .gen-queue-count{color:var(--muted);font-size:11px}
#gen-queue .gen-interrupt{background:var(--danger);border:1px solid var(--danger);
  color:#fff;border-radius:6px;padding:3px 8px;cursor:pointer;font-weight:600;font-size:11px}
#gen-queue .gen-interrupt:hover{filter:brightness(1.1)}
#gen-queue .gen-bar{height:6px;background:var(--line);border-radius:4px;overflow:hidden}
#gen-queue .gen-bar-fill{height:100%;width:0%;background:linear-gradient(90deg,
  var(--accent2),var(--accent));transition:width .12s linear}
#btn-generate.is-busy{opacity:.6;cursor:progress}
`;
        const styleEl = document.createElement("style");
        styleEl.id = "style-" + NAME;
        styleEl.textContent = css;
        document.head.appendChild(styleEl);
      }

      // keep a reference so linters don't flag the unused safe-reader
      void readDimsSafe;
      console.info("[generateWS] initialised");
    },
  });
})();
