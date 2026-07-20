"use strict";
/**
 * Workflow Builder — SerenityFlow
 * Model-aware ComfyUI workflow graph construction for all supported architectures.
 * Shared by Generate tab and Canvas tab.
 */
var WorkflowBuilder = (function () {
    'use strict';
    /**
     * Parse the UI's combined scheduler value (e.g. 'dpmpp_2m_k') into
     * separate sampler_name and scheduler for ComfyUI KSampler nodes.
     * Values ending in '_k' use karras noise schedule.
     */
    function parseSamplerScheduler(combined, defaultSampler, defaultScheduler) {
        var s = combined || defaultSampler || 'euler';
        var sched = defaultScheduler || 'normal';
        if (s.endsWith('_k')) {
            s = s.slice(0, -2);
            sched = 'karras';
        }
        // Map ancestral variants
        if (s === 'euler_ancestral') { s = 'euler_ancestral'; }
        return { sampler: s, scheduler: sched };
    }
    function build(params) {
        var arch = ModelUtils.detectArchFromFilename(params.model);
        var workflow;
        switch (arch) {
            case 'flux':
                workflow = buildFlux(params);
                break;
            case 'klein':
                workflow = buildKlein(params);
                break;
            case 'qwen':
                workflow = buildQwen(params);
                break;
            case 'sd3':
                workflow = buildSD3(params);
                break;
            case 'sdxl':
                workflow = buildSDXL(params);
                break;
            case 'anima':
                // Anima uses its own Mojo worker, but the product graph contract
                // is the same checkpoint/text/latent/sampler shape as SDXL.
                workflow = buildSDXL(params);
                break;
            case 'ltxv':
                workflow = buildLTXV(params);
                break;
            case 'wan':
                workflow = buildWan(params);
                break;
            case 'bernini':
                workflow = buildBernini(params);
                break;
            case 'scail2':
                workflow = buildScail2(params);
                break;
            case 'zimage':
                workflow = buildZImage(params);
                break;
            case 'ideogram4':
                workflow = buildIdeogram4(params);
                break;
            case 'krea2':
                workflow = buildKrea2(params);
                break;
            case 'sensenova':
                workflow = buildSensenova(params);
                break;
            case 'chroma':
                workflow = buildChroma(params);
                break;
            default:
                workflow = buildSD15(params);
                break;
        }
        if (params.loras && params.loras.length > 0) {
            workflow = injectLoRAs(workflow, params.loras);
        }
        if (params.upscale && params.upscale !== 'none') {
            workflow = injectUpscale(workflow, params.upscale, arch);
        }
        return workflow;
    }
    function buildImg2Img(params) {
        var arch = ModelUtils.detectArchFromFilename(params.model);
        var workflow;
        switch (arch) {
            case 'flux':
                workflow = buildFluxImg2Img(params);
                break;
            case 'qwen':
                workflow = buildQwenImg2Img(params);
                break;
            case 'sd3':
                workflow = buildSD3Img2Img(params);
                break;
            case 'sdxl':
                workflow = buildSDXLImg2Img(params);
                break;
            default:
                workflow = buildSD15Img2Img(params);
                break;
        }
        if (params.loras && params.loras.length > 0) {
            workflow = injectLoRAs(workflow, params.loras);
        }
        return workflow;
    }
    function buildInpaint(params) {
        var arch = ModelUtils.detectArchFromFilename(params.model);
        var w = ModelUtils.clampDimension(params.width);
        var h = ModelUtils.clampDimension(params.height);
        var seed = resolveSeed(params.seed);
        // Flux uses standard img2img (no VAEEncodeForInpaint support)
        if (arch === 'flux') {
            return buildFluxImg2Img(params);
        }
        if (arch === 'qwen') {
            return buildQwenImg2Img(params);
        }
        // Video models: fall back to txt2vid
        if (arch === 'ltxv' || arch === 'wan' || arch === 'bernini' || arch === 'scail2') {
            if (arch === 'ltxv') return buildLTXV(params);
            if (arch === 'bernini') return buildBernini(params);
            if (arch === 'scail2') return buildScail2(params);
            return buildWan(params);
        }
        var workflow = {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: params.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: params.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: params.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: params.initImageName } },
            '5': { class_type: 'LoadImage', inputs: { image: params.maskImageName } },
            '6': { class_type: 'VAEEncodeForInpaint', inputs: {
                    pixels: ['4', 0], vae: ['1', 2], mask: ['5', 0], grow_mask_by: 6
                } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: params.steps, cfg: params.cfg || 7.0,
                    sampler_name: parseSamplerScheduler(params.scheduler).sampler, scheduler: parseSamplerScheduler(params.scheduler).scheduler,
                    denoise: params.denoise || 0.75,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['1', 2] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'sf_inpaint' } }
        };
        if (params.loras && params.loras.length > 0) {
            workflow = injectLoRAs(workflow, params.loras);
        }
        return workflow;
    }
    function resolveSeed(seed) {
        return seed === -1 ? Math.floor(Math.random() * 4294967296) : seed;
    }
    /**
     * Inject LoRA loader nodes into a completed workflow.
     * Finds the model/clip source nodes, chains LoRA loaders, and rewires
     * downstream consumers to use the last LoRA output.
     */
    function injectLoRAs(workflow, loras) {
        // Find model and clip source nodes
        var modelNodeId = null;
        var clipNodeId = null;
        var modelIsCheckpoint = false;
        Object.keys(workflow).forEach(function (key) {
            var node = workflow[key];
            if (node.class_type === 'CheckpointLoaderSimple') {
                modelNodeId = key;
                clipNodeId = key;
                modelIsCheckpoint = true;
            }
            else if (node.class_type === 'UNETLoader' || node.class_type === 'LTXVLoader') {
                modelNodeId = key;
            }
            else if (node.class_type === 'DualCLIPLoader' || node.class_type === 'CLIPLoader') {
                clipNodeId = key;
            }
        });
        if (!modelNodeId)
            return workflow;
        var nextId = Math.max.apply(null, Object.keys(workflow).map(Number)) + 1;
        var origModelOut = [modelNodeId, 0];
        var origClipOut = clipNodeId ? [clipNodeId, modelIsCheckpoint ? 1 : 0] : null;
        var prevModelRef = origModelOut;
        var prevClipRef = origClipOut;
        loras.forEach(function (lora) {
            var id = String(nextId++);
            var inputs = {
                lora_name: lora.name,
                strength_model: lora.strength,
                strength_clip: lora.strength,
                model: prevModelRef
            };
            if (prevClipRef) {
                inputs.clip = prevClipRef;
            }
            workflow[id] = { class_type: 'LoraLoader', inputs: inputs };
            prevModelRef = [id, 0];
            if (prevClipRef)
                prevClipRef = [id, 1];
        });
        // Rewire nodes that referenced the original model/clip outputs
        Object.keys(workflow).forEach(function (key) {
            var node = workflow[key];
            if (node.class_type === 'LoraLoader')
                return;
            if (!node.inputs)
                return;
            Object.keys(node.inputs).forEach(function (k) {
                var v = node.inputs[k];
                if (!Array.isArray(v) || v.length < 2)
                    return;
                // Rewire model references
                if (v[0] === origModelOut[0] && v[1] === origModelOut[1] &&
                    (k === 'model' || k === 'ltxv_model')) {
                    node.inputs[k] = prevModelRef;
                }
                // Rewire clip references (only direct clip, not conditioning)
                if (origClipOut && v[0] === origClipOut[0] && v[1] === origClipOut[1] && k === 'clip') {
                    node.inputs[k] = prevClipRef;
                }
            });
        });
        return workflow;
    }
    // ─── FLUX ───────────────────────────────────────────────────────────────
    function buildFlux(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var guidance = p.guidance || 3.5;
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'DualCLIPLoader', inputs: {
                    clip_name1: 'clip_l.safetensors',
                    clip_name2: 't5xxl_fp16.safetensors',
                    type: 'flux'
                } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '5': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '6': { class_type: 'FluxGuidance', inputs: { conditioning: ['4', 0], guidance: guidance } },
            // flux is guidance-distilled: the family rejects real negatives, so
            // the negative input is a ConditioningZeroOut (klein pattern) —
            // wiring the positive there lowers to a rejected negative prompt.
            '10': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: 1.0,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                    model: ['1', 0], positive: ['6', 0], negative: ['10', 0], latent_image: ['5', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'sf_generate' } }
        };
    }
    function buildFluxImg2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var guidance = p.guidance || 3.5;
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'DualCLIPLoader', inputs: {
                    clip_name1: 'clip_l.safetensors',
                    clip_name2: 't5xxl_fp16.safetensors',
                    type: 'flux'
                } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '5': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '6': { class_type: 'VAEEncode', inputs: { pixels: ['5', 0], vae: ['3', 0] } },
            '7': { class_type: 'FluxGuidance', inputs: { conditioning: ['4', 0], guidance: guidance } },
            '8': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: 1.0,
                    sampler_name: 'euler', scheduler: 'simple', denoise: p.denoise || 0.75,
                    model: ['1', 0], positive: ['7', 0], negative: ['4', 0], latent_image: ['6', 0]
                } },
            '9': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
            '10': { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'sf_canvas' } }
        };
    }
    function buildKlein(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 35;
        var cfg = p.cfg || 3.5;
        var prompt = p.prompt || '';
        var negPrompt = typeof p.negPrompt === 'string' ? p.negPrompt.trim() : '';
        var hasNegPrompt = negPrompt.length > 0;
        var clipName = resolveKleinClipName(p.model);
        var negativeNode = hasNegPrompt
            ? { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negPrompt } }
            : { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } };
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: clipName, type: 'klein', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'flux2-vae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['2', 0] } },
            '5': negativeNode,
            '6': { class_type: 'EmptyFlux2LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                    model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'klein' } }
        };
    }
    function buildChroma(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 30;
        var cfg = p.cfg || 4.0;
        var prompt = p.prompt || '';
        var negPrompt = typeof p.negPrompt === 'string' ? p.negPrompt.trim() : '';
        // Chroma runs real CFG with a live T5-XXL encode in the Mojo worker;
        // an empty negative is a real empty-string uncond, not a zero-out.
        var negativeNode = negPrompt.length > 0
            ? { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negPrompt } }
            : { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } };
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 't5xxl_fp16.safetensors', type: 'chroma', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['2', 0] } },
            '5': negativeNode,
            '6': { class_type: 'EmptyFlux2LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                    model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'chroma' } }
        };
    }
    function buildSensenova(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 30;
        var cfg = p.cfg || 4.0;
        // SenseNova-U1 is pixel-space (no VAE) and the model IS its own text
        // encoder; loader nodes are pass-through tokens for the lowering.
        // Negative prompts are not admitted for this family.
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'sensenova_u1', type: 'sensenova', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'sensenova-pixel-space.no-vae' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt || '', clip: ['2', 0] } },
            '5': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } },
            '6': { class_type: 'EmptyFlux2LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                    model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'sensenova' } }
        };
    }
    function buildKrea2(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var turbo = /turbo/i.test(p.model || '');
        var steps = p.steps != null ? p.steps : (turbo ? 8 : 52);
        var cfg = p.cfg != null ? p.cfg : (turbo ? 0.0 : 3.5);
        var prompt = p.prompt || '';
        var negPrompt = typeof p.negPrompt === 'string' ? p.negPrompt.trim() : '';
        var negativeNode = negPrompt.length > 0
            ? { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negPrompt } }
            : { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } };
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'Qwen/Qwen3-VL-4B-Instruct', type: 'krea2', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_vae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['2', 0] } },
            '5': negativeNode,
            '6': { class_type: 'EmptyFlux2LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 1.0,
                    model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'krea2' } }
        };
    }
    function buildIdeogram4(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 48;
        var cfg = p.cfg || 4.5;
        var prompt = (p.prompt || '').trim();
        // Ideogram-4 is trained on structured JSON captions and the server
        // rejects plain text. A JSON object passes through verbatim; plain
        // text gets wrapped in the minimal caption template
        // (aspect_ratio / high_level_description / compositional_deconstruction).
        var caption = prompt;
        if (prompt.charAt(0) !== '{') {
            var g = (function gcd(a, b) { return b ? gcd(b, a % b) : a; })(w, h);
            caption = JSON.stringify({
                aspect_ratio: (w / g) + ':' + (h / g),
                high_level_description: prompt,
                compositional_deconstruction: {
                    background: prompt,
                    elements: []
                }
            });
        }
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'ideogram4', type: 'ideogram4', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ideogram4-vae.safetensors' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: caption, clip: ['2', 0] } },
            '5': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['4', 0] } },
            '6': { class_type: 'EmptyFlux2LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            // scheduler must be 'simple'/'ideogram_logitnormal'; denoise 0.5 is
            // the bounded production route's pinned creativity value.
            '7': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'euler', scheduler: 'simple', denoise: 0.5,
                    model: ['1', 0], positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0]
                } },
            '8': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } },
            '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'ideogram' } }
        };
    }
    function resolveKleinClipName(modelName) {
        if (matchesKleinSize(modelName, '4b')) {
            return 'qwen_3_4b.safetensors';
        }
        if (matchesKleinSize(modelName, '9b') || matchesKleinSize(modelName, '8b')) {
            return 'Qwen/Qwen3-8B';
        }
        return 'Qwen/Qwen3-8B';
    }
    function matchesKleinSize(modelName, size) {
        var lower = (modelName || '').toLowerCase();
        return new RegExp('klein[^a-z0-9]*' + size).test(lower);
    }
    function buildQwen(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 20;
        var cfg = p.cfg || 1;
        var prompt = p.prompt || '';
        var negPrompt = typeof p.negPrompt === 'string' ? p.negPrompt.trim() : '';
        var hasNegPrompt = negPrompt.length > 0;
        var negativeNode = hasNegPrompt
            ? { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negPrompt } }
            : { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['5', 0] } };
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen_2.5_vl_7b_fp8_scaled.safetensors', type: 'qwen', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_vae.safetensors' } },
            '4': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['1', 0], shift: 3 } },
            '5': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['2', 0] } },
            '6': negativeNode,
            '7': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '8': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    // The admitted Qwen product route is pinned to creativity
                    // 0.5 (the same contract used by the proven direct API job).
                    sampler_name: 'euler', scheduler: 'simple', denoise: 0.5,
                    model: ['4', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['7', 0]
                } },
            '9': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
            '10': { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'qwen_image' } }
        };
    }
    function buildQwenImg2Img(p) {
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 20;
        var denoise = p.denoise || 0.75;
        var prompt = p.prompt || '';
        var megapixels = Math.max(0.0625, (ModelUtils.clampDimension(p.width) * ModelUtils.clampDimension(p.height)) / 1000000);
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen_2.5_vl_7b_fp8_scaled.safetensors', type: 'qwen', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_vae.safetensors' } },
            '4': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['1', 0], shift: 3 } },
            '5': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '6': { class_type: 'ImageScaleToTotalPixels', inputs: {
                    image: ['5', 0],
                    upscale_method: 'bicubic',
                    megapixels: megapixels
                } },
            '7': { class_type: 'VAEEncode', inputs: { pixels: ['6', 0], vae: ['3', 0] } },
            '8': { class_type: 'TextEncodeQwenImageEditPlus', inputs: { clip: ['2', 0], text: prompt, image: ['6', 0] } },
            '9': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['8', 0] } },
            '10': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: 1,
                    sampler_name: 'euler', scheduler: 'simple', denoise: denoise,
                    model: ['4', 0], positive: ['8', 0], negative: ['9', 0], latent_image: ['7', 0]
                } },
            '11': { class_type: 'VAEDecode', inputs: { samples: ['10', 0], vae: ['3', 0] } },
            '12': { class_type: 'SaveImage', inputs: { images: ['11', 0], filename_prefix: 'qwen_edit' } }
        };
    }
    function buildZImage(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        var steps = p.steps || 4;
        var cfg = p.cfg || 1;
        var prompt = p.prompt || '';
        var negPrompt = typeof p.negPrompt === 'string' ? p.negPrompt.trim() : '';
        var hasNegPrompt = negPrompt.length > 0;
        var negativeNode = hasNegPrompt
            ? { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negPrompt } }
            : { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['5', 0] } };
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen_3_4b.safetensors', type: 'zimage', device: 'default' } },
            '3': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
            '4': { class_type: 'ModelSamplingAuraFlow', inputs: { model: ['1', 0], shift: 3 } },
            '5': { class_type: 'CLIPTextEncode', inputs: { text: prompt, clip: ['2', 0] } },
            '6': negativeNode,
            '7': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '8': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: steps, cfg: cfg,
                    sampler_name: 'flowmatch_euler', scheduler: 'simple', denoise: 1.0,
                    model: ['4', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['7', 0]
                } },
            '9': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
            '10': { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'zimage' } }
        };
    }
    // ─── SD3 ────────────────────────────────────────────────────────────────
    function buildSD3(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptySD3LatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSamplerAdvanced', inputs: {
                    add_noise: 'enable', noise_seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: 'euler', scheduler: 'simple', start_at_step: 0, end_at_step: 10000,
                    return_with_leftover_noise: 'disable', denoise: 1.0,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
                } },
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }
    function buildSD3Img2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSamplerAdvanced', inputs: {
                    add_noise: 'enable', noise_seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: 'euler', scheduler: 'simple', start_at_step: 0, end_at_step: 10000,
                    return_with_leftover_noise: 'disable', denoise: p.denoise || 0.75,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
                } },
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }
    // ─── SDXL ───────────────────────────────────────────────────────────────
    function buildSDXL(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptyLatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: parseSamplerScheduler(p.scheduler).sampler, scheduler: parseSamplerScheduler(p.scheduler).scheduler, denoise: 1.0,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
                } },
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }
    function buildSDXLImg2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: parseSamplerScheduler(p.scheduler).sampler, scheduler: parseSamplerScheduler(p.scheduler).scheduler, denoise: p.denoise || 0.75,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
                } },
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }
    // ─── SD1.5 ──────────────────────────────────────────────────────────────
    function buildSD15(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'EmptyLatentImage', inputs: { width: w, height: h, batch_size: 1 } },
            '5': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: parseSamplerScheduler(p.scheduler).sampler, scheduler: parseSamplerScheduler(p.scheduler).scheduler, denoise: 1.0,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0]
                } },
            '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
            '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'sf_generate' } }
        };
    }
    function buildSD15Img2Img(p) {
        var w = ModelUtils.clampDimension(p.width);
        var h = ModelUtils.clampDimension(p.height);
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.model } },
            '2': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['1', 1] } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['1', 1] } },
            '4': { class_type: 'LoadImage', inputs: { image: p.initImageName } },
            '5': { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['1', 2] } },
            '6': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps, cfg: p.cfg || 7.0,
                    sampler_name: parseSamplerScheduler(p.scheduler).sampler, scheduler: parseSamplerScheduler(p.scheduler).scheduler, denoise: p.denoise || 0.75,
                    model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['5', 0]
                } },
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
            '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'sf_canvas' } }
        };
    }
    // ─── LTX-V (Video) ──────────────────────────────────────────────────
    function buildLTXV(p) {
        // The production LTX2 arm is Creator-gated at one measured contract.
        // Keep the graph truthful; SerenityAPI routes it to /v1/video.
        var w = 1920;
        var h = 1088;
        var seed = resolveSeed(p.seed);
        var frames = 121;
        var fps = 24;
        var hasGuideImage = typeof p.initImageName === 'string' && p.initImageName.length > 0;
        var loaderInputs = {
            checkpoint_path: p.model,
            gemma_path: 'gemma-3-12b-it',
            dtype: 'bfloat16',
            quantization: 'auto',
            backend: 'mojo'
        };
        var samplerInputs = {
            ltxv_model: ['1', 0],
            prompt: p.prompt,
            negative_prompt: p.negPrompt || 'worst quality, inconsistent motion, blurry, jittery, distorted',
            width: w,
            height: h,
            num_frames: frames,
            steps: 15,
            cfg: 3.0,
            seed: seed,
            frame_rate: fps,
            stg_scale: 1.0,
            mode: 'dev'
        };
        var workflow = {
            '1': { class_type: 'LTXVLoader', inputs: loaderInputs },
            '2': { class_type: 'LTXVSampler', inputs: samplerInputs },
            '3': { class_type: 'SaveVideo', inputs: {
                    video: ['2', 1],
                    filename_prefix: 'sf_video',
                    fps: fps,
                    format: 'mp4'
                } }
        };
        if (hasGuideImage) {
            workflow['4'] = { class_type: 'LoadImage', inputs: { image: p.initImageName } };
            workflow['2'].inputs.guide_image = ['4', 0];
            workflow['2'].inputs.guide_strength = 1.0;
            workflow['2'].inputs.guide_frame_idx = 0;
        }
        return workflow;
    }
    // ─── WAN (Video) ──────────────────────────────────────────────────────
    function buildWan(p) {
        var w = ModelUtils.clampVideoDimension(p.width);
        var h = ModelUtils.clampVideoDimension(p.height);
        var seed = resolveSeed(p.seed);
        var frames = Math.max(9, p.frames || 121);
        return {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: p.model, weight_dtype: 'default' } },
            '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'umt5-xxl-enc-bf16.safetensors', type: 'wan' } },
            '3': { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['2', 0] } },
            '4': { class_type: 'CLIPTextEncode', inputs: { text: p.negPrompt || '', clip: ['2', 0] } },
            '5': { class_type: 'EmptyLatentVideo', inputs: { width: w, height: h, length: frames, batch_size: 1 } },
            '6': { class_type: 'KSampler', inputs: {
                    seed: seed, steps: p.steps || 50, cfg: p.cfg || 5.0,
                    // SerenityAPI translates this graph to the oracle-gated
                    // Mojo Flow-UniPC route; these labels keep saved workflow
                    // metadata aligned with the actual sampler contract.
                    sampler_name: 'uni_pc', scheduler: 'normal', denoise: 1.0,
                    model: ['1', 0], positive: ['3', 0], negative: ['4', 0], latent_image: ['5', 0]
                } },
            '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 1] } },
            '8': { class_type: 'SaveAnimatedWEBP', inputs: {
                    images: ['7', 0],
                    filename_prefix: 'sf_video',
                    fps: p.fps || 24
                } }
        };
    }
    // Bernini-R intentionally reuses the established Wan-shaped workflow
    // graph. SerenityAPI translates the selected model identity to the gated
    // dual-expert Mojo endpoint; the exact creator profile is locked here so a
    // saved Workflow and the Gen screen execute the same contract.
    function buildBernini(p) {
        var admitted = Object.assign({}, p, {
            width: 848,
            height: 480,
            frames: 81,
            fps: 16,
            steps: 40,
            cfg: 4.0,
            scheduler: 'uni_pc'
        });
        var workflow = buildWan(admitted);
        // Bernini's creator width is 848 (106 VAE latent columns), so the
        // generic Wan multiple-of-32 display clamp must not round it to 864.
        workflow['5'].inputs.width = 848;
        workflow['5'].inputs.height = 480;
        return workflow;
    }
    // SCAIL-2 is reference/driving-video animation, not text-to-video. Keep
    // its four media paths explicit in the saved workflow so API translation
    // cannot silently execute the wrong modality.
    function buildScail2(p) {
        var seed = resolveSeed(p.seed);
        return {
            '1': { class_type: 'SCAIL2Animation', inputs: {
                    model: p.model,
                    prompt: p.prompt,
                    negative_prompt: p.negPrompt || '',
                    mode: p.scail2Mode || 'animation',
                    reference_image: p.referenceImagePath || '',
                    reference_mask: p.referenceMaskPath || '',
                    driving_video: p.drivingVideoPath || '',
                    driving_mask_video: p.drivingMaskVideoPath || '',
                    additional_reference_images: p.additionalReferenceImagePaths || [],
                    additional_reference_masks: p.additionalReferenceMaskPaths || [],
                    width: 896,
                    height: 512,
                    frames: 65,
                    fps: 16,
                    steps: 40,
                    cfg: 5.0,
                    seed: seed,
                    quant: 'fp8'
                } },
            '2': { class_type: 'SaveVideo', inputs: {
                    video: ['1', 0],
                    filename_prefix: 'sf_scail2',
                    fps: 16,
                    format: 'mp4'
                } }
        };
    }
    function applyControlNetNodes(workflow, controlLayers) {
        if (!controlLayers || !controlLayers.length)
            return workflow;
        var nextId = Math.max.apply(null, Object.keys(workflow).map(Number)) + 1;
        // Find the positive conditioning node (output used by sampler)
        var samplerKey = null;
        var positiveRef = null;
        Object.keys(workflow).forEach(function (key) {
            var node = workflow[key];
            if (node.class_type === 'KSampler' || node.class_type === 'KSamplerAdvanced') {
                samplerKey = key;
                positiveRef = node.inputs.positive;
            }
        });
        if (!samplerKey)
            return workflow;
        var currentPositive = positiveRef;
        controlLayers.forEach(function (cl) {
            if (!cl.imageName)
                return;
            var loaderId = String(nextId++);
            var loadImgId = String(nextId++);
            var applyId = String(nextId++);
            workflow[loaderId] = {
                class_type: 'ControlNetLoader',
                inputs: { control_net_name: cl.controlNetModel || 'control_v11p_sd15_canny.safetensors' }
            };
            workflow[loadImgId] = {
                class_type: 'LoadImage',
                inputs: { image: cl.imageName }
            };
            workflow[applyId] = {
                class_type: 'ControlNetApplyAdvanced',
                inputs: {
                    positive: currentPositive,
                    negative: workflow[samplerKey].inputs.negative,
                    control_net: [loaderId, 0],
                    image: [loadImgId, 0],
                    strength: cl.weight || 1.0,
                    start_percent: cl.startStep || 0,
                    end_percent: cl.endStep || 1
                }
            };
            currentPositive = [applyId, 0];
        });
        // Rewire sampler positive
        workflow[samplerKey].inputs.positive = currentPositive;
        return workflow;
    }
    // ─── IP-Adapter ──────────────────────────────────────────────────────
    function applyIPAdapterNodes(workflow, ipaLayers) {
        if (!ipaLayers || !ipaLayers.length)
            return workflow;
        var nextId = Math.max.apply(null, Object.keys(workflow).map(Number)) + 1;
        // Find model reference used by sampler
        var samplerKey = null;
        Object.keys(workflow).forEach(function (key) {
            var node = workflow[key];
            if (node.class_type === 'KSampler' || node.class_type === 'KSamplerAdvanced') {
                samplerKey = key;
            }
        });
        if (!samplerKey)
            return workflow;
        var currentModel = workflow[samplerKey].inputs.model;
        ipaLayers.forEach(function (ipa) {
            if (!ipa.imageName)
                return;
            var ipaModelId = String(nextId++);
            var clipVisionId = String(nextId++);
            var loadImgId = String(nextId++);
            var applyId = String(nextId++);
            workflow[ipaModelId] = {
                class_type: 'IPAdapterModelLoader',
                inputs: { ipadapter_file: ipa.ipaModel || 'ip-adapter_sd15.safetensors' }
            };
            workflow[clipVisionId] = {
                class_type: 'CLIPVisionLoader',
                inputs: { clip_name: 'clip_vision_g.safetensors' }
            };
            workflow[loadImgId] = {
                class_type: 'LoadImage',
                inputs: { image: ipa.imageName }
            };
            workflow[applyId] = {
                class_type: 'IPAdapter',
                inputs: {
                    model: currentModel,
                    ipadapter: [ipaModelId, 0],
                    clip_vision: [clipVisionId, 0],
                    image: [loadImgId, 0],
                    weight: ipa.weight || 0.6,
                    noise: 0.0,
                    weight_type: 'original'
                }
            };
            currentModel = [applyId, 0];
        });
        workflow[samplerKey].inputs.model = currentModel;
        return workflow;
    }
    // ─── Post-generation upscale (RealESRGAN) ─────────────────────────────
    function injectUpscale(workflow, upscaleValue, arch) {
        var scale = upscaleValue === '4x' ? 4 : 2;
        // Find the Save node that receives decoded images/video
        var saveNodeId = null;
        var sourceRef = null;
        var isVideo = (arch === 'ltxv' || arch === 'wan' || arch === 'bernini' || arch === 'scail2');
        Object.keys(workflow).forEach(function (key) {
            var node = workflow[key];
            if (isVideo && node.class_type === 'SaveVideo') {
                saveNodeId = key;
                // SaveVideo input is 'video' which comes from sampler output 1
                sourceRef = node.inputs.video;
            }
            else if (!isVideo && (node.class_type === 'SaveImage' || node.class_type === 'SaveAnimatedWEBP')) {
                saveNodeId = key;
                sourceRef = node.inputs.images;
            }
        });
        if (!saveNodeId || !sourceRef)
            return workflow;
        var nextId = String(Math.max.apply(null, Object.keys(workflow).map(Number)) + 1);
        // Insert VideoUpscaleRealESRGAN between source and save
        workflow[nextId] = {
            class_type: 'VideoUpscaleRealESRGAN',
            inputs: {
                images: sourceRef,
                scale: scale,
                model_path: 'upscalers/realesrgan-x4plus/RealESRGAN_x4.pth'
            }
        };
        // Rewire save node to use upscaled output
        if (isVideo) {
            workflow[saveNodeId].inputs.video = [nextId, 0];
        }
        else {
            workflow[saveNodeId].inputs.images = [nextId, 0];
        }
        return workflow;
    }
    return { build: build, buildImg2Img: buildImg2Img, buildInpaint: buildInpaint, applyControlNetNodes: applyControlNetNodes, applyIPAdapterNodes: applyIPAdapterNodes };
})();
//# sourceMappingURL=workflow-builder.js.map
