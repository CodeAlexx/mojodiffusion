"use strict";
/**
 * SerenityAPI — Shared API helpers for SerenityFlow.
 * All /prompt, /upload, /interrupt calls go through here.
 * Registers jobs with QueueTab when available.
 */
var SerenityAPI = (function () {
    'use strict';
    function videoRequestFromWorkflow(workflow) {
        var nodes = workflow || {};
        var keys = Object.keys(nodes);
        for (var si = 0; si < keys.length; si++) {
            var scail = nodes[keys[si]];
            if (scail && scail.class_type === 'SCAIL2Animation') {
                var input = scail.inputs || {};
                return {
                    model: 'scail2',
                    prompt: input.prompt || '',
                    negative_prompt: input.negative_prompt || '',
                    mode: input.mode || 'animation',
                    reference_image: input.reference_image || '',
                    reference_mask: input.reference_mask || '',
                    driving_video: input.driving_video || '',
                    driving_mask_video: input.driving_mask_video || '',
                    additional_reference_images: Array.isArray(input.additional_reference_images) ? input.additional_reference_images : [],
                    additional_reference_masks: Array.isArray(input.additional_reference_masks) ? input.additional_reference_masks : [],
                    width: 896,
                    height: 512,
                    frames: 65,
                    fps: 16,
                    steps: 40,
                    guidance: 5.0,
                    seed: input.seed,
                    quant: 'fp8'
                };
            }
        }
        var h3Loader = null;
        for (var hi = 0; hi < keys.length; hi++) {
            var h3Candidate = nodes[keys[hi]];
            if (h3Candidate && h3Candidate.class_type === 'MiniMaxH3Loader')
                h3Loader = h3Candidate.inputs || {};
        }
        for (var hs = 0; hs < keys.length; hs++) {
            var h3Sampler = nodes[keys[hs]];
            if (h3Sampler && h3Sampler.class_type === 'MiniMaxH3Sampler') {
                var h3 = h3Sampler.inputs || {};
                return {
                    schema: 'serenity.genparams.v1',
                    model: 'minimax_h3',
                    runner: 'minimax_h3_mojo_request',
                    prompt: h3.prompt || '',
                    width: Number(h3.width) || 512,
                    height: Number(h3.height) || 320,
                    frames: Number(h3.num_frames) || 175,
                    fps: Number(h3.frame_rate) || 24,
                    steps: Number(h3.steps) || 20,
                    seed: Number(h3.seed) || 0,
                    quant: h3Loader && h3Loader.precision === 'bf16'
                        ? 'bf16'
                        : (h3Loader && h3Loader.precision === 'int8'
                            ? 'int8' : 'int8-fast'),
                    attention_backend: h3Loader && h3Loader.attention_backend === 'sage-int8'
                        ? 'sage-int8' : 'cudnn',
                    include_audio: true
                };
            }
        }
        var ltxCheckpoint = '';
        var ltxQuantization = '';
        for (var i = 0; i < keys.length; i++) {
            var candidate = nodes[keys[i]];
            if (candidate && candidate.class_type === 'LTXVLoader') {
                var loaderInputs = candidate.inputs || {};
                ltxCheckpoint = loaderInputs.checkpoint_path || '';
                ltxQuantization = loaderInputs.quantization || '';
            }
        }
        for (var i = 0; i < keys.length; i++) {
            var ltx = nodes[keys[i]];
            if (ltx && ltx.class_type === 'LTXVSampler') {
                var li = ltx.inputs || {};
                var loras = [];
                var modelRef = li.ltxv_model || li.model;
                var seen = {};
                while (Array.isArray(modelRef) && modelRef.length > 0) {
                    var modelId = String(modelRef[0]);
                    if (seen[modelId]) break;
                    seen[modelId] = true;
                    var modelNode = nodes[modelId];
                    if (!modelNode) break;
                    var modelInputs = modelNode.inputs || {};
                    if (modelNode.class_type === 'LoraLoaderModelOnly' ||
                        modelNode.class_type === 'LoraLoader') {
                        loras.push({
                            name: modelInputs.lora_name || '',
                            weight: modelInputs.strength_model
                        });
                    }
                    modelRef = modelInputs.model;
                }
                loras.reverse();
                var request = {
                    model: 'ltx2',
                    runner: 'ltx2_mojo_request',
                    checkpoint: ltxCheckpoint,
                    quant: ltxQuantization || 'fp8',
                    schema: 'serenity.genparams.v1',
                    prompt: li.prompt || '',
                    negative: li.negative_prompt || '',
                    steps: li.steps,
                    seed: li.seed,
                    width: li.width,
                    height: li.height,
                    frames: li.num_frames,
                    fps: li.frame_rate,
                    include_audio: li.include_audio === true,
                    audio_policy: li.audio_policy ||
                        (li.include_audio === true ? 'generate' : 'none'),
                    cfg: li.cfg,
                    guidance_mode: li.mode || 'distilled',
                    sampler: li.sampler,
                    scheduler: li.scheduler,
                    prompt_enhancer: li.prompt_enhancer || 'none',
                    caps_positive: li.caps_positive || '',
                    caps_negative: li.caps_negative || '',
                    noise_fixture: li.noise_fixture || '',
                    feature_id: li.feature_id || 'standard',
                    feature_weight: Number(li.feature_weight),
                    video_edit_mode: li.video_edit_mode || 'standard',
                    video_edit_start: Number(li.video_edit_start) || 0,
                    video_edit_end: Number(li.video_edit_end) || 0,
                    camera_motion: li.camera_motion || 'none',
                    lora: loras
                };
                if (String(li.workflow_profile || '').trim())
                    request.workflow_profile = String(li.workflow_profile).trim();
                if (request.feature_id !== 'standard' &&
                    !Number.isFinite(request.feature_weight))
                    throw new Error('LTX2 feature workflow requires numeric feature_weight');
                var postUpscaler = String(li.post_upscale_id || 'none');
                if (postUpscaler !== 'none') {
                    request.post_upscale = {
                        id: postUpscaler,
                        factor: Number(li.post_upscale_factor)
                    };
                }
                if (Array.isArray(li.guide_image)) {
                    var guideNode = nodes[String(li.guide_image[0])];
                    var guidePath = guideNode && guideNode.inputs
                        ? String(guideNode.inputs.image || guideNode.inputs.path || '').trim()
                        : '';
                    if (guidePath) {
                        request.image_path = guidePath;
                        request.image_strength = Number.isFinite(Number(li.guide_strength))
                            ? Number(li.guide_strength)
                            : 1.0;
                    }
                }
                if (Array.isArray(li.guide_video)) {
                    if (request.image_path)
                        throw new Error('LTX2 workflow cannot use guide_image and guide_video together');
                    var guideVideoNode = nodes[String(li.guide_video[0])];
                    var guideVideoPath = guideVideoNode && guideVideoNode.inputs
                        ? String(guideVideoNode.inputs.video || guideVideoNode.inputs.path || '').trim()
                        : '';
                    if (guideVideoPath) {
                        request.video_path = guideVideoPath;
                        request.video_strength = Number.isFinite(Number(li.guide_strength))
                            ? Number(li.guide_strength)
                            : 0.7;
                    }
                }
                if (Array.isArray(li.last_image)) {
                    if (request.video_path)
                        throw new Error('LTX2 workflow cannot use last_image and guide_video together');
                    var lastImageNode = nodes[String(li.last_image[0])];
                    var lastImagePath = lastImageNode && lastImageNode.inputs
                        ? String(lastImageNode.inputs.image || lastImageNode.inputs.path || '').trim()
                        : '';
                    if (lastImagePath) {
                        request.last_image_path = lastImagePath;
                        request.last_image_strength =
                            Number.isFinite(Number(li.last_image_strength))
                                ? Number(li.last_image_strength) : 1.0;
                    }
                }
                if (Array.isArray(li.guide_mask)) {
                    if (!request.video_path)
                        throw new Error('LTX2 guide_mask requires guide_video');
                    var guideMaskNode = nodes[String(li.guide_mask[0])];
                    var guideMaskPath = guideMaskNode && guideMaskNode.inputs
                        ? String(guideMaskNode.inputs.image || guideMaskNode.inputs.path || '').trim()
                        : '';
                    if (guideMaskPath)
                        request.video_mask_path = guideMaskPath;
                }
                return request;
            }
        }
        // The current Wan builder uses standard Comfy nodes. Detect it by the
        // selected UNET name and translate only the bounded 5B T2V controls.
        var wan = false;
        var wanA14b = false;
        var bernini = false;
        var sampler = null;
        var latent = null;
        var wanI2v = null;
        var videoLoras = [];
        for (var j = 0; j < keys.length; j++) {
            var node = nodes[keys[j]] || {};
            var inp = node.inputs || {};
            if (node.class_type === 'UNETLoader' && /bernini/i.test(inp.unet_name || ''))
                bernini = true;
            else if (node.class_type === 'UNETLoader' && /wan/i.test(inp.unet_name || '')) {
                wan = true;
                wanA14b = /a14b/i.test(inp.unet_name || '');
            }
            if (node.class_type === 'LoraLoader' || node.class_type === 'LoraLoaderModelOnly') {
                videoLoras.push({
                    name: inp.lora_name || '',
                    weight: Number(inp.strength_model == null ? 1 : inp.strength_model)
                });
            }
            if (node.class_type === 'KSampler')
                sampler = node;
            if (node.class_type === 'EmptyLatentVideo')
                latent = node;
            if (node.class_type === 'WanImageToVideo')
                wanI2v = node;
        }
        if ((wan || bernini) && sampler) {
            var si = sampler.inputs || {};
            function textFromRef(ref) {
                if (!Array.isArray(ref) || !nodes[String(ref[0])])
                    return '';
                var textInputs = nodes[String(ref[0])].inputs || {};
                if (typeof textInputs.text === 'string')
                    return textInputs.text;
                return textFromRef(textInputs.positive);
            }
            var request = {
                model: bernini ? 'bernini' : (wanA14b ? 'wan22_a14b' : 'wan22'),
                prompt: textFromRef(si.positive),
                negative_prompt: textFromRef(si.negative),
                width: latent ? ((latent.inputs || {}).width || (bernini ? 848 : (wanA14b ? 832 : 1280))) : (bernini ? 848 : (wanA14b ? 832 : 1280)),
                height: latent ? ((latent.inputs || {}).height || (bernini || wanA14b ? 480 : 704)) : (bernini || wanA14b ? 480 : 704),
                frames: latent ? ((latent.inputs || {}).length || ((bernini || wanA14b) ? 81 : 121)) : ((bernini || wanA14b) ? 81 : 121),
                steps: si.steps || (bernini || wanA14b ? 40 : 50),
                guidance: si.cfg || (bernini ? 4.0 : (wanA14b ? 3.0 : 5.0)),
                seed: si.seed || 0,
                fps: (bernini || wanA14b) ? 16 : 24,
                quant: (!bernini && !wanA14b) ? 'bf16' : 'fp8',
                camera_motion: si.camera_motion || 'none',
                lora: videoLoras
            };
            if (wanI2v && !wanA14b && !bernini) {
                var wi = wanI2v.inputs || {};
                var imageRef = wi.image;
                var imageNode = Array.isArray(imageRef)
                    ? nodes[String(imageRef[0])] : null;
                var imagePath = imageNode && imageNode.inputs
                    ? String(imageNode.inputs.image || imageNode.inputs.path || '').trim()
                    : '';
                if (!imagePath)
                    throw new Error('Wan first-frame workflow requires a loaded source image');
                request.image_path = imagePath;
                request.steps = 50;
            }
            return request;
        }
        return null;
    }
    function postVideo(request) {
        return fetch('/v1/video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        }).then(function (resp) {
            return resp.text().then(function (body) {
                var data = {};
                try { data = JSON.parse(body); }
                catch (e) { data = { error: body || ('HTTP ' + resp.status) }; }
                if (!resp.ok)
                    throw new Error(data.detail || data.error || ('HTTP ' + resp.status));
                return data;
            });
        });
    }
    function postPrompt(workflow, metadata) {
        var meta = metadata || {};
        var videoRequest = videoRequestFromWorkflow(workflow);
        if (videoRequest) {
            return postVideo(videoRequest).then(function (data) {
                return {
                    prompt_id: data.prompt_id || data.video_id || '',
                    video_pending: data.state === 'queued' || data.state === 'running',
                    video_result: data
                };
            });
        }
        return fetch('/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: workflow,
                client_id: SerenityWS.getClientId()
            })
        })
            .then(function (resp) {
            if (!resp.ok) {
                // Surface the server's structured error, not just the status
                return resp.text().then(function (body) {
                    var msg = 'HTTP ' + resp.status;
                    try {
                        var j = JSON.parse(body);
                        if (j && j.error)
                            msg = j.error;
                    }
                    catch (e) { /* non-JSON body */ }
                    throw new Error(msg);
                });
            }
            return resp.json();
        })
            .then(function (data) {
            // Register with Queue tab if available
            if (typeof QueueTab !== 'undefined' && QueueTab.registerPending) {
                QueueTab.registerPending({
                    promptId: data.prompt_id,
                    prompt: meta.prompt || '',
                    model: meta.model || '',
                    queuedAt: Date.now(),
                    batchLabel: meta.batchLabel || '',
                    promptData: {
                        workflow: workflow,
                        prompt: meta.prompt || '',
                        model: meta.model || '',
                        width: meta.width || null,
                        height: meta.height || null,
                        seed: meta.seed != null ? meta.seed : null,
                        scheduler: meta.scheduler || null,
                        steps: meta.steps || null,
                        cfg: meta.cfg != null ? meta.cfg : null
                    }
                });
            }
            return data;
        });
    }
    function postGenerate(request, metadata) {
        var meta = metadata || {};
        var body = JSON.stringify(request || {});
        function decodeResponse(resp) {
            return resp.text().then(function (responseBody) {
                var data = {};
                try { data = JSON.parse(responseBody); }
                catch (e) { data = { error: responseBody || ('HTTP ' + resp.status) }; }
                if (!resp.ok) {
                    var detail = data.detail || data.error;
                    if (detail && typeof detail === 'object')
                        detail = detail.message || JSON.stringify(detail);
                    throw new Error(detail || ('HTTP ' + resp.status));
                }
                return data;
            });
        }
        return fetch('/v1/preflight', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body
        }).then(decodeResponse).then(function (preflight) {
            if (preflight.admitted === false)
                throw new Error(preflight.error || preflight.reason || 'Generation request was not admitted');
            return fetch('/v1/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: body
            });
        }).then(decodeResponse).then(function (data) {
            return {
                prompt_id: data.prompt_id || data.job_id || '',
                job_id: data.job_id || data.prompt_id || ''
            };
        }).then(function (data) {
            if (typeof QueueTab !== 'undefined' && QueueTab.registerPending) {
                QueueTab.registerPending({
                    promptId: data.prompt_id,
                    prompt: meta.prompt || request.prompt || '',
                    model: meta.model || request.model || '',
                    queuedAt: Date.now(),
                    batchLabel: meta.batchLabel || '',
                    promptData: {
                        params: request,
                        prompt: meta.prompt || request.prompt || '',
                        model: meta.model || request.model || '',
                        width: request.width || null,
                        height: request.height || null,
                        seed: request.seed != null ? request.seed : null,
                        sampler: request.sampler || null,
                        scheduler: request.scheduler || null,
                        steps: request.steps || null,
                        cfg: request.cfg != null ? request.cfg : null
                    }
                });
            }
            return data;
        });
    }
    function interrupt() {
        return fetch('/interrupt', { method: 'POST' });
    }
    function uploadImageDetails(base64Data, prefix) {
        return fetch('data:image/png;base64,' + base64Data)
            .then(function (r) { return r.blob(); })
            .then(function (blob) {
            var form = new FormData();
            form.append('image', blob, (prefix || 'upload') + '.png');
            form.append('type', 'input');
            return fetch('/upload/image', { method: 'POST', body: form });
        })
            .then(function (resp) {
            if (!resp.ok) {
                // Surface the server's structured error, not just the status
                return resp.text().then(function (body) {
                    var msg = 'HTTP ' + resp.status;
                    try {
                        var j = JSON.parse(body);
                        if (j && j.error)
                            msg = j.error;
                    }
                    catch (e) { /* non-JSON body */ }
                    throw new Error(msg);
                });
            }
            return resp.json();
        });
    }
    function uploadImage(base64Data, prefix) {
        return uploadImageDetails(base64Data, prefix)
            .then(function (data) { return data.path || data.name; });
    }
    function uploadMediaDetails(file) {
        if (!file)
            return Promise.reject(new Error('No media file selected'));
        var form = new FormData();
        form.append('file', file, file.name || 'source.mp4');
        return fetch('/upload/media', { method: 'POST', body: form })
            .then(function (resp) {
            if (!resp.ok) {
                return resp.text().then(function (body) {
                    var msg = 'HTTP ' + resp.status;
                    try {
                        var data = JSON.parse(body);
                        if (data && (data.detail || data.error))
                            msg = data.detail || data.error;
                    }
                    catch (_) { }
                    throw new Error(msg);
                });
            }
            return resp.json();
        });
    }
    function uploadMedia(file) {
        return uploadMediaDetails(file)
            .then(function (data) { return data.path || data.name; });
    }
    function viewUrl(filename, subfolder, type) {
        return '/view?filename=' + encodeURIComponent(filename) +
            '&subfolder=' + encodeURIComponent(subfolder || '') +
            '&type=' + encodeURIComponent(type || 'output');
    }
    return {
        postPrompt: postPrompt,
        postGenerate: postGenerate,
        postVideo: postVideo,
        videoRequestFromWorkflow: videoRequestFromWorkflow,
        interrupt: interrupt,
        uploadImage: uploadImage,
        uploadImageDetails: uploadImageDetails,
        uploadMediaDetails: uploadMediaDetails,
        uploadMedia: uploadMedia,
        viewUrl: viewUrl
    };
})();
function SFApi() {
    this.connect = function () { };
    this.on = function (type, fn) { SerenityWS.on(type, fn); };
    this.off = function (type, fn) { SerenityWS.off(type, fn); };
    this.interrupt = function () { return SerenityAPI.interrupt(); };
    this.viewUrl = function (filename, subfolder, type) {
        return SerenityAPI.viewUrl(filename, subfolder, type);
    };
    this.getObjectInfo = function () {
        return fetch('/object_info').then(function (r) { return r.json(); });
    };
    this.queuePrompt = function (workflow) {
        return SerenityAPI.postPrompt(workflow);
    };
    this.getClientId = function () { return SerenityWS.getClientId(); };
}
