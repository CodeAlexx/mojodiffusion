// @ts-nocheck -- mirrored to js/h3-control-workflow.js for the browser build.
"use strict";
/** Native JSON workflow assembler for the dedicated MiniMax-H3 CT screen. */
var H3ControlWorkflow = (function () {
    'use strict';

    function finiteNumber(value, fallback) {
        var number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    }

    var h3ResolutionPresets = {
        '1536x672': true,
        '1344x768': true,
        '1024x768': true,
        '768x768': true,
        '768x1024': true,
        '768x1344': true
    };

    function h3Resolution(widthValue, heightValue) {
        var width = Number(widthValue);
        var height = Number(heightValue);
        if (!Number.isInteger(width) || !Number.isInteger(height) ||
            !h3ResolutionPresets[width + 'x' + height]) {
            throw new Error('H3 CT resolution must be one of the six supported presets');
        }
        return { width: width, height: height };
    }

    function activeControls(params) {
        if (!Array.isArray(params.controls)) return [{
            controlMedia: params.controlMedia,
            preprocessor: params.preprocessor,
            resizeMode: params.resizeMode,
            cannyLow: params.cannyLow,
            cannyHigh: params.cannyHigh,
            strength: params.strength,
            startPercent: params.startPercent,
            endPercent: params.endPercent
        }];
        return params.controls.filter(function (control) {
            return control && control.enabled !== false;
        });
    }

    function build(params) {
        var duration = finiteNumber(params.durationSeconds, 5);
        var visibleSteps = finiteNumber(params.steps, 40);
        var resolution = h3Resolution(
            params.width != null ? params.width : 1344,
            params.height != null ? params.height : 768
        );
        var width = resolution.width;
        var height = resolution.height;
        var controls = activeControls(params);
        var loras = (Array.isArray(params.loras) ? params.loras : []).filter(function (lora) {
            return lora && lora.enabled !== false;
        });
        var windowSize = finiteNumber(params.window, 2);
        var prefetch = finiteNumber(params.prefetch, 1);
        var videoShift = finiteNumber(params.videoShift, 12.0);
        var audioShift = finiteNumber(params.audioShift, 3.0);

        if (!params.model) throw new Error('Select a MiniMax H3 base checkpoint');
        if (!params.controlNet) throw new Error('Select the official H3 ControlNet checkpoint');
        if (!params.videoVae) throw new Error('MiniMax H3 video VAE was not found');
        if (!params.audioVae) throw new Error('MiniMax H3 audio VAE was not found');
        if (!controls.length || controls.some(function (control) { return !control.controlMedia; }))
            throw new Error('Add a control video');
        if (controls.length > 4)
            throw new Error('H3 CT supports up to 4 enabled control tracks');
        if (!Number.isInteger(duration) || duration < 5 || duration > 15)
            throw new Error('H3 CT duration must be a whole number from 5 through 15 seconds');
        if (!Number.isInteger(visibleSteps) || visibleSteps < 1 || visibleSteps > 100)
            throw new Error('H3 CT steps must be a whole number from 1 through 100');
        if (!Number.isInteger(windowSize) || windowSize < 2 || windowSize > 8)
            throw new Error('H3 CT streaming window must be a whole number from 2 through 8');
        if (!Number.isInteger(prefetch) || prefetch < 1 || prefetch > 4)
            throw new Error('H3 CT prefetch must be a whole number from 1 through 4');
        if (videoShift < 0 || videoShift > 30 || audioShift < 0 || audioShift > 30)
            throw new Error('H3 CT flow shifts must be from 0 through 30');
        if ((params.sourceMedia && !params.maskMedia) || (!params.sourceMedia && params.maskMedia))
            throw new Error('H3 CT inpainting needs both a source video and a mask');
        var normalizedControls = controls.map(function (control, index) {
            var controlPrefix = Array.isArray(params.controls) ? 'Control ' + (index + 1) + ' ' : '';
            var strength = finiteNumber(control.strength, 1.0);
            var start = finiteNumber(control.startPercent, 0.0);
            var end = finiteNumber(control.endPercent, 1.0);
            var cannyLow = Math.round(finiteNumber(control.cannyLow, 100));
            var cannyHigh = Math.round(finiteNumber(control.cannyHigh, 200));
            if (!(start >= 0 && start <= end && end <= 1))
                throw new Error((controlPrefix || 'Control ') +
                    'range must satisfy 0 ≤ start ≤ end ≤ 1');
            if ((control.preprocessor || 'prepared') === 'canny' &&
                    !(cannyLow >= 0 && cannyLow < cannyHigh && cannyHigh <= 255))
                throw new Error(controlPrefix + 'Canny thresholds must satisfy 0 ≤ low < high ≤ 255');
            return {
                controlMedia: control.controlMedia,
                preprocessor: control.preprocessor || 'prepared',
                resizeMode: control.resizeMode || 'crop',
                cannyLow: cannyLow,
                cannyHigh: cannyHigh,
                strength: strength,
                startPercent: start,
                endPercent: end
            };
        });

        // Preserve SerenityFlow's visual graph contract locally. Execution is
        // lowered by the shared API into one native Mojo H3 ControlNet request.
        var outputFrames = duration * 24;
        var internalFrames = outputFrames + (17 - ((outputFrames - 5) % 17)) % 17;
        var outputFormat = String(params.outputFormat || 'mp4')
            .trim().toLowerCase().replace(/^\.+/, '');
        if (['mp4', 'mov', 'mkv'].indexOf(outputFormat) < 0)
            outputFormat = 'mp4';
        var seed = Math.floor(finiteNumber(params.seed, -1));
        if (seed === -1)
            seed = Math.floor(Math.random() * 4294967296);
        var loaderInputs = {
            unet_name: params.model,
            window: windowSize,
            prefetch: prefetch,
            attention_backend: 'ck-int8'
        };
        var h3Loras = loras.map(function (lora) {
            return { name: String(lora.name || ''), strength: Number(lora.strength) };
        }).filter(function (lora) {
            return lora.name && Number.isFinite(lora.strength);
        });
        if (h3Loras.length)
            loaderInputs.loras = h3Loras;
        var workflow = {
            '1': {
                class_type: 'MiniMaxH3TextEncode',
                inputs: {
                    clip_name: String(params.clipName || '').trim(),
                    prompt: params.prompt || ''
                }
            },
            '2': { class_type: 'MiniMaxH3Loader', inputs: loaderInputs },
            '3': {
                class_type: 'MiniMaxH3EmptyLatent',
                inputs: {
                    width: width,
                    height: height,
                    length: internalFrames,
                    output_frames: outputFrames
                }
            },
            '4': {
                class_type: 'MiniMaxH3SigmaShift',
                inputs: { shift_video: videoShift, shift_audio: audioShift }
            },
            '6': {
                class_type: 'MiniMaxH3Sampler',
                inputs: {
                    model: ['2', 0],
                    streamer: ['2', 1],
                    conditioning: ['1', 0],
                    latent: ['3', 0],
                    shift: ['4', 0],
                    seed: seed,
                    // SerenityFlow stores N sigma points for N-1 model calls.
                    steps: visibleSteps + 1,
                    sampler_name: 'euler',
                    scheduler: 'normal'
                }
            },
            '91': {
                class_type: 'CreateVideo',
                inputs: { images: ['90', 0], audio: ['90', 1], fps: 24 }
            },
            '92': {
                class_type: 'SaveVideo',
                inputs: {
                    video: ['91', 0],
                    filename_prefix: 'minimax_h3_controlnet_union',
                    fps: 24,
                    format: outputFormat
                }
            }
        };

        workflow['5'] = {
            class_type: 'MiniMaxH3FunControlNetLoader',
            inputs: {
                model: ['2', 0],
                control_net_name: params.controlNet
            }
        };

        var conditioning = ['1', 0];
        normalizedControls.forEach(function (control, index) {
            var nodeId = String(7 + index);
            var applyInputs = {
                conditioning: conditioning,
                latent: ['3', 0],
                control_net: ['5', 0],
                video_vae_name: params.videoVae,
                control_media: control.controlMedia,
                preprocessor: control.preprocessor,
                resize_mode: control.resizeMode,
                canny_low: control.cannyLow,
                canny_high: control.cannyHigh,
                strength: control.strength,
                start_percent: control.startPercent,
                end_percent: control.endPercent
            };
            if (index === 0 && params.sourceMedia && params.maskMedia) {
                applyInputs.source_media = params.sourceMedia;
                applyInputs.mask_media = params.maskMedia;
                applyInputs.invert_mask = !!params.invertMask;
            }
            workflow[nodeId] = {
                class_type: 'MiniMaxH3FunControlNetApplyMedia',
                inputs: applyInputs
            };
            conditioning = [nodeId, 0];
        });
        workflow['6'].inputs.conditioning = conditioning;
        delete workflow['80'];
        delete workflow['81'];
        workflow['90'] = {
            class_type: 'MiniMaxH3DecodeRelease',
            inputs: {
                latent: ['6', 0],
                video_vae_name: params.videoVae,
                audio_vae_name: params.audioVae
            }
        };
        workflow['92'].inputs.filename_prefix = String(params.filenamePrefix || '').trim() ||
            'minimax_h3_controlnet_union';
        return workflow;
    }

    function nodesOfType(prompt, classType) {
        return Object.keys(prompt).filter(function (nodeId) {
            return prompt[nodeId] && prompt[nodeId].class_type === classType;
        });
    }

    function onlyNode(prompt, classType) {
        var matches = nodesOfType(prompt, classType);
        if (matches.length !== 1)
            throw new Error('H3 CT workflow requires exactly one ' + classType + ' node');
        return { id: matches[0], node: prompt[matches[0]] };
    }

    function isReference(value, nodeId, outputSlot) {
        return Array.isArray(value) && value.length === 2 &&
            String(value[0]) === String(nodeId) && Number(value[1]) === outputSlot;
    }

    /** Parse the exact visual H3 CT graph back into dedicated-screen settings. */
    function parse(prompt) {
        if (!prompt || typeof prompt !== 'object' || Array.isArray(prompt)) return null;
        var hasControlGraph = nodesOfType(prompt, 'MiniMaxH3FunControlNetLoader').length > 0 ||
            nodesOfType(prompt, 'MiniMaxH3FunControlNetApplyMedia').length > 0;
        if (!hasControlGraph) return null;

        var allowed = {
            MiniMaxH3TextEncode: true,
            MiniMaxH3Loader: true,
            MiniMaxH3EmptyLatent: true,
            MiniMaxH3SigmaShift: true,
            MiniMaxH3FunControlNetLoader: true,
            MiniMaxH3FunControlNetApplyMedia: true,
            MiniMaxH3Sampler: true,
            MiniMaxH3DecodeRelease: true,
            CreateVideo: true,
            SaveVideo: true
        };
        Object.keys(prompt).forEach(function (nodeId) {
            var classType = prompt[nodeId] && prompt[nodeId].class_type;
            if (!allowed[classType])
                throw new Error('H3 CT workflow contains unsupported node ' + String(classType || nodeId));
        });

        var text = onlyNode(prompt, 'MiniMaxH3TextEncode');
        var base = onlyNode(prompt, 'MiniMaxH3Loader');
        var latent = onlyNode(prompt, 'MiniMaxH3EmptyLatent');
        var shift = onlyNode(prompt, 'MiniMaxH3SigmaShift');
        var controlLoader = onlyNode(prompt, 'MiniMaxH3FunControlNetLoader');
        var sampler = onlyNode(prompt, 'MiniMaxH3Sampler');
        var decode = onlyNode(prompt, 'MiniMaxH3DecodeRelease');
        var createVideo = onlyNode(prompt, 'CreateVideo');
        var saveVideo = onlyNode(prompt, 'SaveVideo');
        var applyIds = nodesOfType(prompt, 'MiniMaxH3FunControlNetApplyMedia');
        if (applyIds.length < 1 || applyIds.length > 4)
            throw new Error('H3 CT workflow requires from 1 through 4 control tracks');

        var baseInputs = base.node.inputs || {};
        var latentInputs = latent.node.inputs || {};
        var shiftInputs = shift.node.inputs || {};
        var loaderInputs = controlLoader.node.inputs || {};
        var samplerInputs = sampler.node.inputs || {};
        var decodeInputs = decode.node.inputs || {};
        var createInputs = createVideo.node.inputs || {};
        var saveInputs = saveVideo.node.inputs || {};

        if (String(baseInputs.attention_backend || '') !== 'ck-int8')
            throw new Error('H3 CT workflow requires CK-INT8 attention');
        if (!isReference(loaderInputs.model, base.id, 0) ||
            !isReference(samplerInputs.model, base.id, 0) ||
            !isReference(samplerInputs.streamer, base.id, 1) ||
            !isReference(samplerInputs.latent, latent.id, 0) ||
            !isReference(samplerInputs.shift, shift.id, 0) ||
            !isReference(decodeInputs.latent, sampler.id, 0) ||
            !isReference(createInputs.images, decode.id, 0) ||
            !isReference(createInputs.audio, decode.id, 1) ||
            !isReference(saveInputs.video, createVideo.id, 0)) {
            throw new Error('H3 CT workflow connections do not match the native graph');
        }
        if (String(samplerInputs.sampler_name || '') !== 'euler' ||
            String(samplerInputs.scheduler || '') !== 'normal')
            throw new Error('H3 CT workflow requires Euler / Normal sampling');
        if (Number(createInputs.fps) !== 24 || Number(saveInputs.fps) !== 24)
            throw new Error('H3 CT workflow requires 24 FPS video output');

        var outputFrames = Number(latentInputs.output_frames);
        var duration = outputFrames / 24;
        var expectedInternalFrames = outputFrames + (17 - ((outputFrames - 5) % 17)) % 17;
        if (!Number.isInteger(duration) || duration < 5 || duration > 15 ||
            Number(latentInputs.length) !== expectedInternalFrames)
            throw new Error('H3 CT workflow has an invalid whole-second frame contract');

        var orderedReverse = [];
        var nextConditioning = samplerInputs.conditioning;
        var visited = {};
        while (Array.isArray(nextConditioning) && nextConditioning.length === 2) {
            var nodeId = String(nextConditioning[0]);
            var node = prompt[nodeId];
            if (!node || node.class_type !== 'MiniMaxH3FunControlNetApplyMedia') break;
            if (visited[nodeId] || Number(nextConditioning[1]) !== 0)
                throw new Error('H3 CT control chain is cyclic or uses the wrong output');
            visited[nodeId] = true;
            orderedReverse.push({ id: nodeId, node: node });
            nextConditioning = (node.inputs || {}).conditioning;
        }
        if (!isReference(nextConditioning, text.id, 0) || orderedReverse.length !== applyIds.length)
            throw new Error('H3 CT control tracks are not one connected conditioning chain');
        var ordered = orderedReverse.reverse();
        var videoVae = String(decodeInputs.video_vae_name || '');
        var controls = ordered.map(function (entry, index) {
            var inputs = entry.node.inputs || {};
            if (!isReference(inputs.latent, latent.id, 0) ||
                !isReference(inputs.control_net, controlLoader.id, 0) ||
                String(inputs.video_vae_name || '') !== videoVae)
                throw new Error('H3 CT control track ' + (index + 1) + ' is wired to the wrong asset');
            var start = finiteNumber(inputs.start_percent, 0);
            var end = finiteNumber(inputs.end_percent, 1);
            var cannyLow = Math.round(finiteNumber(inputs.canny_low, 100));
            var cannyHigh = Math.round(finiteNumber(inputs.canny_high, 200));
            var preprocessor = String(inputs.preprocessor || 'prepared');
            if (!(start >= 0 && start <= end && end <= 1))
                throw new Error('H3 CT control track ' + (index + 1) + ' has an invalid range');
            if (preprocessor === 'canny' &&
                !(cannyLow >= 0 && cannyLow < cannyHigh && cannyHigh <= 255))
                throw new Error('H3 CT control track ' + (index + 1) + ' has invalid Canny thresholds');
            return {
                enabled: true,
                controlMedia: String(inputs.control_media || ''),
                preprocessor: preprocessor,
                resizeMode: String(inputs.resize_mode || 'crop'),
                cannyLow: cannyLow,
                cannyHigh: cannyHigh,
                strength: finiteNumber(inputs.strength, 1),
                startPercent: start,
                endPercent: end
            };
        });
        if (controls.some(function (control) { return !control.controlMedia; }))
            throw new Error('H3 CT control track is missing media');

        var loras = Array.isArray(baseInputs.loras) ? baseInputs.loras.map(function (lora) {
            return {
                name: String(lora && lora.name || ''),
                strength: Number(lora && lora.strength),
                enabled: true
            };
        }) : [];
        var loraNames = {};
        if (loras.some(function (lora) {
            var invalid = !lora.name || !Number.isFinite(lora.strength) || loraNames[lora.name];
            loraNames[lora.name] = true;
            return invalid;
        })) throw new Error('H3 CT workflow contains an invalid or duplicate LoRA');

        var visibleSteps = Number(samplerInputs.steps) - 1;
        if (!Number.isInteger(visibleSteps) || visibleSteps < 1 || visibleSteps > 100)
            throw new Error('H3 CT workflow has an invalid step count');
        var firstInputs = ordered[0].node.inputs || {};
        var hasSource = !!String(firstInputs.source_media || '');
        var hasMask = !!String(firstInputs.mask_media || '');
        if (hasSource !== hasMask)
            throw new Error('H3 CT inpainting needs both a source video and a mask');
        if (ordered.slice(1).some(function (entry) {
            var inputs = entry.node.inputs || {};
            return Object.prototype.hasOwnProperty.call(inputs, 'source_media') ||
                Object.prototype.hasOwnProperty.call(inputs, 'mask_media') ||
                Object.prototype.hasOwnProperty.call(inputs, 'invert_mask');
        })) throw new Error('H3 CT inpainting inputs belong only on control track 1');
        var resolution = h3Resolution(latentInputs.width, latentInputs.height);
        var params = {
            model: String(baseInputs.unet_name || ''),
            controlNet: String(loaderInputs.control_net_name || ''),
            clipName: String((text.node.inputs || {}).clip_name || ''),
            videoVae: videoVae,
            audioVae: String(decodeInputs.audio_vae_name || ''),
            controlMedia: controls[0].controlMedia,
            controls: controls,
            loras: loras,
            sourceMedia: String(firstInputs.source_media || ''),
            maskMedia: String(firstInputs.mask_media || ''),
            invertMask: firstInputs.invert_mask === true,
            preprocessor: controls[0].preprocessor,
            resizeMode: controls[0].resizeMode,
            cannyLow: controls[0].cannyLow,
            cannyHigh: controls[0].cannyHigh,
            strength: controls[0].strength,
            startPercent: controls[0].startPercent,
            endPercent: controls[0].endPercent,
            prompt: String((text.node.inputs || {}).prompt || ''),
            width: resolution.width,
            height: resolution.height,
            durationSeconds: duration,
            steps: visibleSteps,
            seed: Number(samplerInputs.seed),
            outputFormat: String(saveInputs.format || 'mp4'),
            window: Number(baseInputs.window),
            prefetch: Number(baseInputs.prefetch),
            videoShift: Number(shiftInputs.shift_video),
            audioShift: Number(shiftInputs.shift_audio),
            filenamePrefix: String(saveInputs.filename_prefix || '')
        };
        if (!params.model || !params.controlNet || !params.clipName ||
            !params.videoVae || !params.audioVae)
            throw new Error('H3 CT workflow is missing a required installed asset');
        return params;
    }

    return { build: build, parse: parse };
})();
