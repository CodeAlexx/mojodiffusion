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
        var ltxCheckpoint = '';
        for (var i = 0; i < keys.length; i++) {
            var candidate = nodes[keys[i]];
            if (candidate && candidate.class_type === 'LTXVLoader')
                ltxCheckpoint = (candidate.inputs || {}).checkpoint_path || '';
        }
        for (var i = 0; i < keys.length; i++) {
            var ltx = nodes[keys[i]];
            if (ltx && ltx.class_type === 'LTXVSampler') {
                var li = ltx.inputs || {};
                return {
                    model: 'ltx2',
                    runner: 'ltx2_refhq',
                    checkpoint: ltxCheckpoint,
                    prompt: li.prompt || '',
                    negative_prompt: li.negative_prompt || '',
                    steps: 15,
                    seed: li.seed,
                    width: li.width,
                    height: li.height,
                    frames: li.num_frames,
                    fps: li.frame_rate
                };
            }
        }
        // The current Wan builder uses standard Comfy nodes. Detect it by the
        // selected UNET name and translate only the bounded 5B T2V controls.
        var wan = false;
        var bernini = false;
        var sampler = null;
        var latent = null;
        for (var j = 0; j < keys.length; j++) {
            var node = nodes[keys[j]] || {};
            var inp = node.inputs || {};
            if (node.class_type === 'UNETLoader' && /bernini/i.test(inp.unet_name || ''))
                bernini = true;
            else if (node.class_type === 'UNETLoader' && /wan/i.test(inp.unet_name || ''))
                wan = true;
            if (node.class_type === 'KSampler')
                sampler = node;
            if (node.class_type === 'EmptyLatentVideo')
                latent = node;
        }
        if ((wan || bernini) && sampler) {
            var si = sampler.inputs || {};
            function textFromRef(ref) {
                if (!Array.isArray(ref) || !nodes[String(ref[0])])
                    return '';
                return (nodes[String(ref[0])].inputs || {}).text || '';
            }
            return {
                model: bernini ? 'bernini' : 'wan22',
                prompt: textFromRef(si.positive),
                negative_prompt: textFromRef(si.negative),
                width: latent ? ((latent.inputs || {}).width || (bernini ? 848 : 832)) : (bernini ? 848 : 832),
                height: latent ? ((latent.inputs || {}).height || 480) : 480,
                frames: latent ? ((latent.inputs || {}).length || (bernini ? 81 : 121)) : (bernini ? 81 : 121),
                steps: si.steps || (bernini ? 40 : 50),
                guidance: si.cfg || (bernini ? 4.0 : 5.0),
                seed: si.seed || 0,
                fps: bernini ? 16 : 24,
                quant: 'fp8'
            };
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
                return { prompt_id: data.video_id || '', video_result: data };
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
    function interrupt() {
        return fetch('/interrupt', { method: 'POST' });
    }
    function uploadImage(base64Data, prefix) {
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
        })
            .then(function (data) { return data.name; });
    }
    function viewUrl(filename, subfolder, type) {
        return '/view?filename=' + encodeURIComponent(filename) +
            '&subfolder=' + encodeURIComponent(subfolder || '') +
            '&type=' + encodeURIComponent(type || 'output');
    }
    return {
        postPrompt: postPrompt,
        postVideo: postVideo,
        videoRequestFromWorkflow: videoRequestFromWorkflow,
        interrupt: interrupt,
        uploadImage: uploadImage,
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
//# sourceMappingURL=api.js.map
