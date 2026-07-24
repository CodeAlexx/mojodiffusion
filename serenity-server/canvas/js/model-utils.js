"use strict";
/**
 * Model Utilities — SerenityFlow
 * Architecture detection from model filenames and dimension clamping.
 */
var ModelUtils = (function () {
    'use strict';
    var objectInfoCache = null;
    var capabilitiesCache = null;
    function loadObjectInfo() {
        if (objectInfoCache)
            return Promise.resolve(objectInfoCache);
        var cachedEtag = localStorage.getItem('sf-object-info-etag');
        var headers = {};
        if (cachedEtag)
            headers['If-None-Match'] = cachedEtag;
        return fetch('/object_info', { headers: headers })
            .then(function (resp) {
            if (resp.status === 304) {
                try {
                    var cached = JSON.parse(localStorage.getItem('sf-object-info-data'));
                    if (cached) {
                        objectInfoCache = cached;
                        return cached;
                    }
                }
                catch (e) { }
            }
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json().then(function (data) {
                objectInfoCache = data;
                var etag = resp.headers.get('ETag');
                if (etag) {
                    localStorage.setItem('sf-object-info-etag', etag);
                    try {
                        localStorage.setItem('sf-object-info-data', JSON.stringify(data));
                    }
                    catch (e) {
                        localStorage.removeItem('sf-object-info-data');
                    }
                }
                return data;
            });
        });
    }
    // Detect architecture from model filename heuristics.
    // The backend does proper header detection — this is frontend-only best-effort
    // for choosing the right workflow graph before generation starts.
    function detectArchFromFilename(filename) {
        if (!filename)
            return 'sd15';
        var f = filename.toLowerCase();
        if (f.includes('scail'))
            return 'scail2';
        if (f.includes('bernini'))
            return 'bernini';
        if (f.includes('qwen'))
            return 'qwen';
        if (f.includes('zimage') || f.includes('z-image') || f.includes('z_image'))
            return 'zimage';
        // klein/krea2/ideogram BEFORE the generic flux match:
        // "flux-2-klein-base-9b" was detected as 'flux' and built a Flux1
        // graph (DualCLIP+t5xxl+FluxGuidance) that the klein family rejects.
        if (f.includes('klein'))
            return 'klein';
        if (f.includes('krea'))
            return 'krea2';
        if (f.includes('ideogram'))
            return 'ideogram4';
        if (f.includes('sensenova'))
            return 'sensenova';
        if (f.includes('chroma'))
            return 'chroma';
        if (f.includes('flux') || f.includes('flex') || f.includes('f1d') || f.includes('f1s'))
            return 'flux';
        if (f.includes('sd3') || f.includes('stable-diffusion-3') || f.includes('sd_3') || f.includes('stablediffusion3'))
            return 'sd3';
        if (f.includes('sdxl') || f.includes('xl') || f.includes('pony') || f.includes('illustrious'))
            return 'sdxl';
        if (f.includes('anima'))
            return 'anima';
        if (f.includes('ltx') || f.includes('ltxv'))
            return 'ltxv';
        if (f.includes('wan'))
            return 'wan';
        if (f.includes('hunyuan') || f.includes('capybara'))
            return 'hunyuan';
        return 'sd15';
    }
    // Check if a detected architecture is a video model
    function isVideoModel(filename) {
        var arch = detectArchFromFilename(filename);
        return arch === 'ltxv' || arch === 'wan' || arch === 'bernini' || arch === 'scail2';
    }
    // Standard video resolutions (smaller, snap to 32 for video VAE)
    var VIDEO_RESOLUTIONS = [
        { label: '1:1', width: 512, height: 512 },
        { label: '4:3', width: 768, height: 576 },
        { label: '16:9', width: 768, height: 432 },
        { label: '3:4', width: 576, height: 768 },
        { label: '9:16', width: 432, height: 768 }
    ];
    // Snap a dimension value to nearest multiple of 64
    function snapTo64(val) {
        return Math.max(64, Math.round(val / 64) * 64);
    }
    // Snap a dimension value to nearest multiple of 32 (video VAE requirement)
    function snapTo32(val) {
        return Math.max(64, Math.round(val / 32) * 32);
    }
    // Validate and clamp a dimension: min 256, max 4096, divisible by 64
    function clampDimension(val) {
        return Math.min(4096, Math.max(256, snapTo64(val)));
    }
    // Validate and clamp a video dimension: min 64, max 1280, divisible by 32
    function clampVideoDimension(val) {
        return Math.min(1280, Math.max(64, snapTo32(val)));
    }
    // Fetch all available models from /object_info, merging checkpoints and UNETs.
    // Returns a promise that resolves to an array of { name, loader } objects.
    // loader is 'checkpoint' or 'unet'.
    // Filter out sub-model components (text encoders, clips, sharded parts, upscalers, loras)
    function isMainModel(name) {
        var lower = name.toLowerCase();
        // LTX RefHQ has one production-admitted base. The other local LTX files
        // are legacy bases, parity oracles, LoRAs, and upscalers; exposing them
        // in Generate would falsely imply that /v1/video executes the selected
        // file. They remain visible in the model manager's /v1/models inventory.
        if (lower.includes('ltx') && lower !== 'ltx-2.3-22b-dev-fp8')
            return false;
        // Skip files inside subdirectories that are clearly sub-components
        if (/\/(text_encoder|clip|tokenizer|vae|scheduler|feature_extractor|vision_encoder|transformer)\//.test(name))
            return false;
        // Skip sharded model parts (model-00001-of-00004.safetensors)
        if (/model-\d+-of-\d+/.test(lower))
            return false;
        // Skip upscalers and loras mixed in
        if (/upscaler|upscale/.test(lower))
            return false;
        if (/[\-_]lora[\-_\.]/.test(lower))
            return false;
        // Skip individual training checkpoints (transformer-0001, etc.)
        if (/transformer-\d{4}/.test(lower))
            return false;
        // Skip diffusers subdirectory components (e.g. capybara/transformer/model.safetensors)
        if (/\/transformer\//.test(name))
            return false;
        // Qwen-VL files are text/vision encoders, not diffusion backbones.
        // Qwen Image Edit checkpoints belong to Canvas' Edit Models surface,
        // not the text-to-image Generate picker.
        if (/qwen[^/]*(?:vl|vision.*language)/.test(lower))
            return false;
        if (/qwen[^/]*image[^/]*edit|qwen[^/]*edit[^/]*image/.test(lower))
            return false;
        return true;
    }
    function fetchAllModels() {
        return loadObjectInfo()
            .then(function (data) {
            var models = [];
            var seen = {};
            // Checkpoints (SD1.5, SDXL, SD3, etc.)
            var ckptInfo = data && data.CheckpointLoaderSimple &&
                data.CheckpointLoaderSimple.input &&
                data.CheckpointLoaderSimple.input.required &&
                data.CheckpointLoaderSimple.input.required.ckpt_name;
            if (ckptInfo && Array.isArray(ckptInfo[0])) {
                ckptInfo[0].forEach(function (m) {
                    if (!seen[m] && isMainModel(m)) {
                        seen[m] = true;
                        models.push({ name: m, loader: 'checkpoint' });
                    }
                });
            }
            // UNETs (FLUX, Klein, WAN, etc.)
            var unetInfo = data && data.UNETLoader &&
                data.UNETLoader.input &&
                data.UNETLoader.input.required &&
                data.UNETLoader.input.required.unet_name;
            if (unetInfo && Array.isArray(unetInfo[0])) {
                unetInfo[0].forEach(function (m) {
                    if (!seen[m] && isMainModel(m)) {
                        seen[m] = true;
                        models.push({ name: m, loader: 'unet' });
                    }
                });
            }
            return models;
        });
    }
    function loadCapabilities() {
        if (capabilitiesCache)
            return Promise.resolve(capabilitiesCache);
        return fetch('/v1/capabilities', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('capabilities HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (data) {
            if (!data || data.schema !== 'serenity.capabilities.v1' || !Array.isArray(data.backends))
                throw new Error('invalid Serenity capability document');
            capabilitiesCache = data;
            return data;
        });
    }
    function backendForArch(arch) {
        var byArch = {
            zimage: 'zimage',
            qwen: 'qwenimage',
            ideogram4: 'ideogram4',
            sdxl: 'sdxl',
            anima: 'anima',
            sd3: 'sd3',
            flux: 'flux',
            klein: 'flux2',
            sensenova: 'sensenova',
            krea2: 'krea2',
            chroma: 'chroma'
        };
        return byArch[arch] || '';
    }
    function aspectsForArch(capabilities, arch) {
        var backend = backendForArch(arch);
        if (!backend || !capabilities || !Array.isArray(capabilities.backends))
            return [];
        var profile = capabilities.backends.find(function (entry) {
            return entry && entry.backend === backend;
        });
        var sizes = profile && profile.limits && profile.limits.sizes;
        if (!Array.isArray(sizes))
            return [];
        return sizes.map(function (size) {
            var width = Number(size.width);
            var height = Number(size.height);
            if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0)
                throw new Error('invalid capability size for ' + backend);
            return {
                label: width + '×' + height,
                w: width,
                h: height,
                vw: width / 64,
                vh: height / 64
            };
        });
    }
    function clearCache() {
        objectInfoCache = null;
        capabilitiesCache = null;
        localStorage.removeItem('sf-object-info-etag');
        localStorage.removeItem('sf-object-info-data');
    }
    return {
        detectArchFromFilename: detectArchFromFilename,
        isVideoModel: isVideoModel,
        VIDEO_RESOLUTIONS: VIDEO_RESOLUTIONS,
        snapTo64: snapTo64,
        snapTo32: snapTo32,
        clampDimension: clampDimension,
        clampVideoDimension: clampVideoDimension,
        fetchAllModels: fetchAllModels,
        loadCapabilities: loadCapabilities,
        backendForArch: backendForArch,
        aspectsForArch: aspectsForArch,
        loadObjectInfo: loadObjectInfo,
        clearCache: clearCache
    };
})();
//# sourceMappingURL=model-utils.js.map
