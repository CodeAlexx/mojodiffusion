"use strict";
/**
 * Model Utilities — SerenityFlow
 * Registry-first architecture lookup and dimension clamping.
 */
var ModelUtils = (function () {
    'use strict';
    var objectInfoCache = null;
    var modelRegistryCache = null;
    var modelByIdentity = {};
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
    // Legacy fallback for bundled profiles or a temporarily unavailable registry.
    // User-downloaded checkpoints route through archForModel(), whose primary
    // authority is the server's metadata/tensor-signature registry.
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
        if (f.includes('microsoft_lens') || f.includes('microsoft-lens') || f.includes('lenspipeline'))
            return 'lens';
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
    function normalizeRegistryArch(arch) {
        var value = String(arch || '').toLowerCase();
        var aliases = {
            'qwen-image': 'qwen',
            'flux-2': 'klein',
            'flux-2/klein': 'klein',
            'ltx2': 'ltxv',
            'wan2.2': 'wan'
        };
        return aliases[value] || value || 'unknown';
    }
    function rememberModel(model) {
        if (!model)
            return;
        var arch = normalizeRegistryArch(model.arch);
        [model.name, model.path].forEach(function (identity) {
            if (identity)
                modelByIdentity[String(identity)] = arch;
        });
    }
    function archForModel(identity) {
        return modelByIdentity[String(identity || '')] || detectArchFromFilename(identity);
    }
    function loadModelRegistry() {
        if (modelRegistryCache)
            return Promise.resolve(modelRegistryCache);
        return fetch('/v1/models?refresh=1', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('models HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (data) {
            if (!data || data.schema !== 'serenity.models.v1' || !Array.isArray(data.models))
                throw new Error('invalid Serenity model registry');
            modelRegistryCache = data;
            modelByIdentity = {};
            data.models.forEach(rememberModel);
            (data.loras || []).forEach(rememberModel);
            return data;
        });
    }
    // Check if a detected architecture is a video model
    function isVideoModel(filename) {
        var arch = archForModel(filename);
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
    // Normalize a video dimension to the model grid. The backend profile
    // registry owns admission; a shared UI clamp must not corrupt valid LTX2
    // profiles such as 1920x1088 into unsupported hybrid dimensions.
    function clampVideoDimension(val) {
        return Math.max(64, snapTo32(val));
    }
    // Fetch exact model cards from the disk registry. Architecture and format
    // come from safetensors metadata/key signatures; filenames are fallback-only.
    // Filter out sub-model components (text encoders, clips, sharded parts, upscalers, loras)
    function isMainModel(name) {
        var lower = name.toLowerCase();
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
        return loadModelRegistry()
            .then(function (data) {
            return data.models.filter(function (model) {
                return model && model.runtime_supported !== false && isMainModel(model.name);
            }).map(function (model) {
                rememberModel(model);
                return {
                    name: model.name,
                    path: model.path,
                    arch: normalizeRegistryArch(model.arch),
                    format: model.format || 'diffusion_model',
                    loader: model.format === 'full_checkpoint' ? 'checkpoint' : 'unet',
                    generationRoute: model.generation_route || 'image',
                    usesSelectedCheckpoint: model.uses_selected_checkpoint === true,
                    generationDefaults: model.generation_defaults || {}
                };
            });
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
            chroma: 'chroma',
            lens: 'lens'
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
        modelRegistryCache = null;
        modelByIdentity = {};
        capabilitiesCache = null;
        localStorage.removeItem('sf-object-info-etag');
        localStorage.removeItem('sf-object-info-data');
    }
    return {
        detectArchFromFilename: detectArchFromFilename,
        archForModel: archForModel,
        isVideoModel: isVideoModel,
        VIDEO_RESOLUTIONS: VIDEO_RESOLUTIONS,
        snapTo64: snapTo64,
        snapTo32: snapTo32,
        clampDimension: clampDimension,
        clampVideoDimension: clampVideoDimension,
        fetchAllModels: fetchAllModels,
        loadModelRegistry: loadModelRegistry,
        loadCapabilities: loadCapabilities,
        backendForArch: backendForArch,
        aspectsForArch: aspectsForArch,
        loadObjectInfo: loadObjectInfo,
        clearCache: clearCache
    };
})();
