"use strict";
/**
 * Generate Tab — SerenityFlow Phase 2
 * Prompt-to-image generation wired to ComfyUI-compatible backend.
 * SerenityFlow visual overhaul.
 */
var GenerateTab = (function () {
    'use strict';
    var ensureLibraryExpanded = function () { };
    var state = {
        model: null,
        prompt: '',
        negPrompt: '',
        width: 1024,
        height: 1024,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        sampler: 'euler',
        scheduler: 'simple',
        sigmaShift: 3.0,
        seed: -1,
        variationSeed: 0,
        variationStrength: 0,
        initImagePath: '',
        initImageName: '',
        initImageWidth: 0,
        initImageHeight: 0,
        creativity: 0.5,
        generating: false,
        externalActivity: null,
        currentImage: null,
        currentIsVideo: false,
        currentVideoFrames: null,
        currentVideoFps: null,
        gallery: [],
        assets: [],
        assetsLoaded: false,
        assetsLoading: false,
        assetsReloadPending: false,
        assetsError: '',
        arch: 'sd15',
        frames: 121,
        fps: 24,
        seconds: 10,
        lastSeed: null,
        batchCount: 1,
        loras: [], // [{name, strength, enabled}]
        aspectLocked: false,
        lockedRatio: 1,
        // New state for overhaul
        leftPanelVisible: true,
        rightPanelVisible: true,
        galleryTab: 'gallery', // 'layers' | 'gallery'
        gallerySubTab: 'images', // 'images' | 'assets'
        boardsVisible: true,
        selectedAspect: '1:1',
        pendingBatch: 0,
        // Prompt history
        promptHistory: [],
        promptHistoryIndex: -1,
        // Style preset
        stylePreset: 'none',
        // Model picker search
        modelPickerOpen: false,
        modelSearchQuery: '',
        allModels: [],
        capabilities: null,
        videoStatus: null,
        videoGuidanceMode: 'distilled',
        videoWorkflowProfile: '',
        videoPromptEnhancer: 'none',
        videoQuant: 'fp8',
        h3Quant: 'int8-fast',
        h3AttentionBackend: 'cudnn',
        h3StepCache: 'exact',
        cameraMotion: 'none',
        videoCheckpoint: 'ltx-2.3-22b-distilled',
        capsPositive: '',
        capsNegative: '',
        noiseFixture: '',
        includeAudio: false,
        audioPolicy: 'none',
        postUpscaler: 'none',
        postUpscaleFactor: 2,
        noSeedIncrement: false,
        continueAfterErrors: true,
        personalNote: '',
        showAdvanced: true,
        currentResultParams: null,
        currentGalleryIndex: -1,
        pendingVideoJobs: {},
        completedVideoJobs: {},
        pendingImageJobs: {},
        completedImageJobs: {},
        currentBatchKeys: {},
        videoPollToken: 0,
        // Phase 2: gallery enhancements
        selectedImages: [],
        lastSelectedIndex: -1,
        gallerySearch: '',
        thumbSize: 90,
        sortNewestFirst: true,
        starredFirst: false,
        autoSwitchNew: true,
        contextMenuIndex: -1,
        gallerySettingsOpen: false,
        metadataExpanded: false,
        // Phase 3: Advanced section
        vae: 'default',
        clipSkip: 0,
        cfgRescale: 0,
        seamlessX: false,
        seamlessY: false,
        // Phase 3: Refiner section (SDXL)
        refinerModel: '',
        refinerScheduler: 'euler',
        refinerSteps: 20,
        refinerCfg: 7.0,
        refinerStart: 0.8,
        aestheticScore: 6.0,
        negAestheticScore: 2.5,
        // Phase 3: Compositing section (inpaint)
        coherenceMode: 'gaussian',
        edgeSize: 32,
        minDenoise: 0.0,
        maskBlur: 4,
        infillMethod: 'patchmatch',
        referenceImagePath: '',
        referenceMaskPath: '',
        drivingVideoPath: '',
        drivingMaskVideoPath: '',
        scail2Mode: 'animation',
        additionalReferenceImagePaths: [],
        additionalReferenceMaskPaths: [],
        mediaUploadsInFlight: 0
    };
    var initialized = false;
    // Image samplers are not constrained by the bounded LTX request profile.
    // LTX temporarily rewrites these shared controls to its exact 8/20-step
    // schedules, so every switch back to an image model must restore them.
    var IMAGE_STEPS_SLIDER_MAX = 150;
    var IMAGE_STEPS_INPUT_MAX = 500;
    // DOM refs (set in init)
    var els = {};
    // Convert seconds + fps to frame count (round to nearest valid frame count)
    function secondsToFrames(seconds, fps) {
        // frames = seconds * fps + 1 (first frame is the start frame)
        return Math.max(9, Math.round(seconds * fps) + 1);
    }
    function ltx2SecondsToFrames(seconds, fps) {
        // LTX-2 requires (8 * K) + 1 frames. This matches Desktop's
        // _compute_num_frames rather than silently sending an invalid shape.
        return Math.max(9, Math.floor(seconds * fps / 8) * 8 + 1);
    }
    // ── Aspect ratio definitions ──
    var videoAspects = [
        { label: 'Free', w: 0, h: 0, vw: 16, vh: 16 },
        { label: '1:1', w: 768, h: 768, vw: 16, vh: 16 },
        { label: '4:3', w: 768, h: 576, vw: 18, vh: 14 },
        { label: '3:2', w: 768, h: 512, vw: 18, vh: 12 },
        { label: '16:9', w: 768, h: 432, vw: 20, vh: 11 },
        { label: '21:9', w: 768, h: 320, vw: 21, vh: 9 },
        { label: '3:4', w: 576, h: 768, vw: 14, vh: 18 },
        { label: '2:3', w: 512, h: 768, vw: 12, vh: 18 },
        { label: '9:16', w: 432, h: 768, vw: 11, vh: 20 },
        { label: '9:21', w: 320, h: 768, vw: 9, vh: 21 }
    ];
    // ── Style Presets ──
    var stylePresets = [
        { id: 'none', label: 'None', suffix: '' },
        { id: 'photo', label: 'Photo', suffix: ', photorealistic, high quality photo, detailed, sharp focus, natural lighting' },
        { id: 'anime', label: 'Anime', suffix: ', anime style, vibrant colors, detailed illustration, anime art' },
        { id: 'oil', label: 'Oil Paint', suffix: ', oil painting style, textured brushstrokes, rich colors, classical art' },
        { id: '3d', label: '3D Render', suffix: ', 3D render, octane render, detailed, volumetric lighting, CGI' },
        { id: 'cinematic', label: 'Cinematic', suffix: ', cinematic, dramatic lighting, film grain, anamorphic, movie still' },
        { id: 'watercolor', label: 'Watercolor', suffix: ', watercolor painting, soft colors, artistic, delicate brushwork' },
        { id: 'pixel', label: 'Pixel Art', suffix: ', pixel art, 8-bit, retro gaming style, low resolution aesthetic' },
        { id: 'sketch', label: 'Sketch', suffix: ', pencil sketch, hand drawn, detailed linework, graphite' },
        { id: 'fantasy', label: 'Fantasy', suffix: ', fantasy art, magical, ethereal, detailed illustration, concept art' },
        { id: 'neon', label: 'Neon', suffix: ', neon lights, cyberpunk, glowing, dark background, vibrant' },
        { id: 'minimal', label: 'Minimal', suffix: ', minimalist, clean, simple, modern design, white space' }
    ];
    // ── Scheduler/Sampler definitions ──
    // Only expose the sampler that every production worker currently admits.
    // Wider sampler inventory remains available through the capability API and
    // returns to this picker only after model-specific artifact gates pass.
    var schedulerOptions = [
        { value: 'euler', label: 'Euler' },
        { value: 'uni_pc', label: 'UniPC (Wan)' }
    ];
    function getActiveAspects() {
        var arch = ModelUtils.archForModel(state.model);
        if (arch === 'minimax_h3') {
            var h3Runner = activeMinimaxH3Runner();
            var h3Constraints = h3Runner && h3Runner.geometry_constraints || {};
            var h3Presets = Array.isArray(h3Constraints.resolutions)
                ? h3Constraints.resolutions : [];
            return h3Presets.map(function (profile) {
                return {
                    label: profile.width + '×' + profile.height,
                    w: Number(profile.width),
                    h: Number(profile.height),
                    vw: Number(profile.width) / 64,
                    vh: Number(profile.height) / 64
                };
            });
        }
        if (arch === 'ltxv') {
            var seen = {};
            return activeLtx2RequestProfiles().filter(function (profile) {
                var key = profile.width + 'x' + profile.height;
                if (seen[key])
                    return false;
                seen[key] = true;
                return true;
            }).map(function (profile) {
                return {
                    label: profile.width + '×' + profile.height,
                    w: Number(profile.width),
                    h: Number(profile.height),
                    vw: Number(profile.width) / 64,
                    vh: Number(profile.height) / 64
                };
            });
        }
        if (arch === 'wan') {
            var wanRunner = activeWan22Runner();
            var wanProfiles = wanRunner && Array.isArray(wanRunner.native_profiles)
                ? wanRunner.native_profiles : [];
            if (wanProfiles.length) {
                return wanProfiles.map(function (profile) {
                    return {
                        label: profile.width + '×' + profile.height +
                            (profile.mode === 'i2v_first_frame' ? ' I2V' : ''),
                        w: Number(profile.width),
                        h: Number(profile.height),
                        vw: Number(profile.width) / 64,
                        vh: Number(profile.height) / 64
                    };
                });
            }
            return [
                { label: '1280×704', w: 1280, h: 704, vw: 20, vh: 11 },
                { label: '704×1280', w: 704, h: 1280, vw: 11, vh: 20 },
                { label: '1248×704 I2V', w: 1248, h: 704, vw: 19.5, vh: 11 },
                { label: '704×1248 I2V', w: 704, h: 1248, vw: 11, vh: 19.5 }
            ];
        }
        if (arch === 'bernini')
            return [{ label: '848×480', w: 848, h: 480, vw: 53, vh: 30 }];
        if (arch === 'scail2')
            return [{ label: '896×512', w: 896, h: 512, vw: 28, vh: 16 }];
        return ModelUtils.aspectsForArch(state.capabilities, arch);
    }
    function activeLtx2RequestMode() {
        var candidates = state.videoStatus && state.videoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        var runner = candidates.find(function (entry) {
            return entry && entry.model === 'ltx2_t2v_av';
        });
        return runner && runner.modes && runner.modes.ltx2_mojo_request || null;
    }
    function refreshCameraMotionControls(motions) {
        if (!els.cameraMotion)
            return;
        var rows = Array.isArray(motions) ? motions.filter(function (motion) {
            return motion && typeof motion.id === 'string';
        }) : [];
        if (!rows.length)
            rows = [{ id: 'none', label: 'None', available: true }];
        els.cameraMotion.innerHTML = '';
        rows.forEach(function (motion) {
            var option = document.createElement('option');
            option.value = motion.id;
            option.textContent = motion.label || motion.id.replace(/_/g, ' ');
            if (motion.control === 'lora')
                option.textContent += ' · camera LoRA';
            else if (motion.control === 'prompt')
                option.textContent += ' · prompt guidance';
            option.disabled = motion.available === false;
            els.cameraMotion.appendChild(option);
        });
        var selected = rows.some(function (motion) {
            return motion.id === state.cameraMotion && motion.available !== false;
        }) ? state.cameraMotion : 'none';
        state.cameraMotion = selected;
        els.cameraMotion.value = selected;
        els.cameraMotion.disabled = false;
        els.cameraMotion.title = 'Camera LoRA controls are attached by the server; prompt-only controls are labeled explicitly.';
    }
    function ltx2CheckpointWorkflow(checkpoint) {
        var mode = activeLtx2RequestMode();
        var profiles = mode && Array.isArray(mode.checkpoint_workflows)
            ? mode.checkpoint_workflows : [];
        var normalized = String(checkpoint || '')
            .replace(/\.safetensors$/i, '').toLowerCase();
        return profiles.find(function (profile) {
            return profile && Array.isArray(profile.checkpoints) &&
                profile.checkpoints.some(function (name) {
                    return String(name || '').replace(/\.safetensors$/i, '')
                        .toLowerCase() === normalized;
                });
        }) || null;
    }
    function refreshVideoPromptEnhancer() {
        if (!els.videoPromptEnhancer)
            return;
        var workflow = ltx2CheckpointWorkflow(state.videoCheckpoint);
        var enhancer = workflow && workflow.prompt_enhancer;
        var available = !!(enhancer && workflow.prompt_enhancer_available === true);
        var enhancerId = enhancer ? String(enhancer.id || '') : '';
        els.videoPromptEnhancer.innerHTML =
            '<option value="none">Raw prompt (creator workflow default)</option>' +
            (enhancerId
                ? '<option value="' + escapeHtml(enhancerId) + '"' +
                    (available ? '' : ' disabled') + '>' +
                    escapeHtml(enhancerId) +
                    (available ? '' : ' (weights missing)') + '</option>'
                : '');
        if (!available || state.videoPromptEnhancer !== enhancerId)
            state.videoPromptEnhancer = 'none';
        els.videoPromptEnhancer.value = state.videoPromptEnhancer;
        els.videoPromptEnhancer.disabled = !enhancerId || !available;
        if (!els.videoPromptEnhancerNote)
            return;
        if (!enhancer) {
            els.videoPromptEnhancerNote.textContent =
                'The selected checkpoint has no registered creator prompt enhancer.';
        }
        else if (!available) {
            var filesAvailable = workflow.prompt_enhancer_files_available === true;
            els.videoPromptEnhancerNote.textContent = filesAvailable
                ? 'Creator enhancer files are installed, but its execution route is not implemented yet. Serenity will keep the prompt raw; it will not fake enhancement.'
                : 'Creator enhancer is unavailable: install ' +
                    String(enhancer.weights || '') + ' and ' +
                    String(enhancer.mmproj || '') +
                    '. Serenity will keep the prompt raw; it will not fake enhancement.';
        }
        else {
            els.videoPromptEnhancerNote.textContent =
                'Creator enhancer uses the raw prompt and optional source image with no system prompt.';
        }
    }
    function activeLtx2Checkpoints() {
        var mode = activeLtx2RequestMode();
        var checkpoints = mode && Array.isArray(mode.checkpoints)
            ? mode.checkpoints : [];
        return checkpoints.filter(function (checkpoint) {
            return checkpoint && checkpoint.installed === true;
        });
    }
    function activeLtx2Checkpoint() {
        var checkpoints = activeLtx2Checkpoints();
        return checkpoints.find(function (checkpoint) {
            return checkpoint.id === state.videoCheckpoint;
        }) || null;
    }
    function refreshLtx2CheckpointControls() {
        var mode = activeLtx2RequestMode() || {};
        var checkpoints = activeLtx2Checkpoints();
        var selected = activeLtx2Checkpoint();
        if (!selected) {
            selected = checkpoints.find(function (checkpoint) {
                return checkpoint.id === mode.default_checkpoint;
            }) || checkpoints[0] || null;
            if (selected)
                state.videoCheckpoint = selected.id;
        }
        if (els.videoCheckpoint) {
            els.videoCheckpoint.innerHTML = checkpoints.map(function (checkpoint) {
                var suffix = checkpoint.readiness_label === 'production_ready'
                    ? '' : ' · experimental';
                return '<option value="' + escapeHtml(String(checkpoint.id)) + '">' +
                    escapeHtml(String(checkpoint.label || checkpoint.id) + suffix) +
                    '</option>';
            }).join('');
            els.videoCheckpoint.value = state.videoCheckpoint;
            els.videoCheckpoint.disabled = checkpoints.length < 2;
        }
        var guidanceModes = selected && Array.isArray(selected.guidance_modes)
            ? selected.guidance_modes : ['distilled', 'dev'];
        if (guidanceModes.indexOf(state.videoGuidanceMode) < 0)
            state.videoGuidanceMode = guidanceModes[0] || 'distilled';
        if (els.videoGuidanceMode) {
            els.videoGuidanceMode.innerHTML = guidanceModes.map(function (modeName) {
                return '<option value="' + modeName + '">' +
                    (modeName === 'dev' ? 'Dev CFG' : 'Distilled') + '</option>';
            }).join('');
            els.videoGuidanceMode.value = state.videoGuidanceMode;
        }
        var quantModes = selected && Array.isArray(selected.quant_modes)
            ? selected.quant_modes : ['fp8'];
        if (quantModes.indexOf(state.videoQuant) < 0)
            state.videoQuant = quantModes[0] || 'fp8';
        if (els.videoQuant) {
            els.videoQuant.innerHTML = quantModes.map(function (modeName) {
                return '<option value="' + modeName + '">' +
                    modeName.toUpperCase() + '</option>';
            }).join('');
            els.videoQuant.value = state.videoQuant;
        }
    }
    function exactLtx2RequestProfile() {
        var mode = activeLtx2RequestMode();
        if (!mode)
            return null;
        var profiles = activeLtx2RequestProfiles();
        return profiles.find(function (profile) {
            return Number(profile.width) === Number(state.width) &&
                Number(profile.height) === Number(state.height) &&
                Number(profile.frames) === Number(state.frames) &&
                Number(profile.fps) === Number(state.fps);
        }) || null;
    }
    function activeLtx2RequestProfile() {
        var mode = activeLtx2RequestMode();
        if (!mode)
            return null;
        var profiles = activeLtx2RequestProfiles();
        var portrait = Number(state.height) > Number(state.width);
        var sameOrientation = profiles.find(function (profile) {
            return (Number(profile.height) > Number(profile.width)) === portrait;
        });
        return exactLtx2RequestProfile() || sameOrientation ||
            (mode.compiled_profile && mode.compiled_profile.available !== false
            ? mode.compiled_profile : null) || profiles[0] || null;
    }
    function updateLtx2ProfileStatus() {
        var profileNote = document.getElementById('gen-video-profile-note');
        if (!profileNote)
            return;
        var profiles = activeLtx2ProfilesForSize(state.width, state.height);
        var exact = exactLtx2RequestProfile();
        if (exact) {
            profileNote.classList.toggle('invalid', exact.available !== true);
            profileNote.textContent = (exact.available === true
                ? 'Available runtime profile: '
                : 'Registered profile; runtime currently unavailable: ') +
                state.width + '×' + state.height + ', ' +
                state.frames + ' frames at ' + state.fps +
                ' FPS.';
            return;
        }
        profileNote.classList.add('invalid');
        profileNote.textContent = 'No registered runtime profile matches ' +
            state.width + '×' + state.height + ', ' + state.frames +
            ' frames at ' + state.fps + ' FPS. Registered here: ' +
            profiles.map(function (profile) {
                return profile.duration + 's / ' + profile.frames +
                    'f @ ' + profile.fps;
            }).join(', ') + '.';
    }
    function activeLtx2PostUpscalers() {
        var mode = activeLtx2RequestMode();
        return mode && Array.isArray(mode.post_upscalers)
            ? mode.post_upscalers : [];
    }
    function refreshLtx2PostUpscaleControls() {
        if (!els.postUpscaler || !els.postUpscaleFactor)
            return;
        var isLtx2 = ModelUtils.archForModel(state.model) === 'ltxv';
        var upscalers = activeLtx2PostUpscalers();
        var available = upscalers.filter(function (entry) {
            return entry && entry.available === true;
        });
        els.postUpscaler.innerHTML =
            '<option value="none">Native output</option>' +
            upscalers.map(function (entry) {
                var disabled = entry.available === true ? '' : ' disabled';
                var suffix = entry.available !== true
                    ? ' (unavailable)'
                    : (entry.status === 'experimental_slow'
                        ? ' (experimental slow)' : '');
                return '<option value="' + escapeHtml(String(entry.id)) + '"' +
                    disabled + '>' + escapeHtml(String(entry.label || entry.id)) +
                    suffix + '</option>';
            }).join('');
        if (!isLtx2 || (state.postUpscaler !== 'none' &&
            !available.some(function (entry) {
                return entry.id === state.postUpscaler;
            }))) {
            state.postUpscaler = 'none';
        }
        els.postUpscaler.value = state.postUpscaler;
        els.postUpscaler.disabled = !isLtx2 || available.length === 0;
        var selected = available.find(function (entry) {
            return entry.id === state.postUpscaler;
        });
        var scales = selected && Array.isArray(selected.scales)
            ? selected.scales.map(Number).filter(function (value) {
                return value === 2 || value === 4;
            }) : [2, 4];
        if (scales.indexOf(Number(state.postUpscaleFactor)) < 0)
            state.postUpscaleFactor = scales[0] || 2;
        els.postUpscaleFactor.innerHTML = scales.map(function (factor) {
            return '<option value="' + factor + '">' + factor + '× · ' +
                (state.width * factor) + '×' + (state.height * factor) +
                '</option>';
        }).join('');
        els.postUpscaleFactor.value = String(state.postUpscaleFactor);
        els.postUpscaleFactor.disabled = !isLtx2 || state.postUpscaler === 'none';
        if (els.postUpscaleNote) {
            if (!isLtx2) {
                els.postUpscaleNote.textContent =
                    'Post-upscale is available on the LTX2 video route.';
            }
            else if (!available.length) {
                els.postUpscaleNote.textContent =
                    'No installed post-upscaler has both a Mojo runner and local weights.';
            }
            else if (!selected) {
                els.postUpscaleNote.textContent =
                    'Native output selected. Choose an admitted post-upscaler to render beyond the native profile.';
            }
            else {
                els.postUpscaleNote.textContent =
                    selected.label + ' will produce ' +
                    (state.width * state.postUpscaleFactor) + '×' +
                    (state.height * state.postUpscaleFactor) +
                    ' after native LTX2 decode.' +
                    (selected.status === 'experimental_slow'
                        ? ' Measured RRDB x4plus speed is 18.24 seconds/frame at 960×544; long clips are not production-speed.'
                        : '');
            }
        }
    }
    function activeLtx2RequestProfiles() {
        var mode = activeLtx2RequestMode();
        if (!mode)
            return [];
        var profiles = Array.isArray(mode.supported_profiles)
            ? mode.supported_profiles : [];
        return profiles.filter(function (profile) {
            // Keep registered standard profiles visible even while the single
            // runtime executable is absent. Availability controls queue
            // admission; it must not erase authored geometry or checkpoint
            // settings from the UI.
            return profile && Array.isArray(profile.modes) &&
                profile.modes.indexOf('standard') >= 0;
        });
    }
    function activeLtx2ProfilesForSize(width, height) {
        return activeLtx2RequestProfiles().filter(function (profile) {
            return Number(profile.width) === Number(width) &&
                Number(profile.height) === Number(height);
        });
    }
    function applyLtx2RequestProfile(profile) {
        if (!profile)
            return;
        if (ModelUtils.archForModel(state.model) === 'ltxv')
            state.videoCheckpoint = String(state.model).replace(/\.safetensors$/i, '');
        else
            state.videoCheckpoint = state.videoQuant === 'bf16'
                ? 'ltx-2.3-22b-dev-fp8-dequant-bf16'
                : String(profile.checkpoint || 'ltx-2.3-22b-dev-fp8');
        refreshLtx2CheckpointControls();
        state.width = Number(profile.width);
        state.height = Number(profile.height);
        state.frames = Number(profile.frames);
        state.fps = Number(profile.fps);
        state.seconds = Number(profile.duration);
        if (!Number.isFinite(state.seconds) || state.seconds <= 0)
            state.seconds = state.frames / state.fps;
        syncDimensionInputs();
        syncAspectDropdown();
        updateAspectPreview();
        refreshLtx2ProfileControls();
        refreshLtx2PostUpscaleControls();
        updateDurationHint();
    }
    function refreshLtx2ProfileControls() {
        var profiles = activeLtx2ProfilesForSize(state.width, state.height);
        if (!profiles.length)
            return;
        if (els.framesInput) {
            var frameValues = profiles.map(function (profile) {
                return Number(profile.frames);
            });
            els.framesInput.min = String(Math.min.apply(Math, frameValues));
            els.framesInput.max = String(Math.max.apply(Math, frameValues));
            els.framesInput.step = '8';
            els.framesInput.value = String(state.frames);
            els.framesInput.disabled = false;
            els.framesInput.title = 'Editable LTX2 frame count; valid native counts match 8*K+1 and a registered runtime profile';
        }
        if (els.secondsInput) {
            els.secondsInput.min = '0.1';
            els.secondsInput.max = '120';
            els.secondsInput.step = '0.01';
            els.secondsInput.value = String(state.seconds);
            els.secondsInput.disabled = false;
            els.secondsInput.title = 'Editable duration; LTX2 frame count follows the 8*K+1 rule without changing the selected resolution';
        }
        if (els.fpsInput) {
            els.fpsInput.value = String(state.fps);
            els.fpsInput.min = '1';
            els.fpsInput.max = '60';
            els.fpsInput.disabled = false;
            els.fpsInput.title = 'Editable FPS; the combination must match a registered runtime profile';
        }
        if (els.fpsRange) {
            els.fpsRange.value = String(state.fps);
            els.fpsRange.min = '1';
            els.fpsRange.max = '60';
            els.fpsRange.disabled = false;
            els.fpsRange.title = 'Editable FPS; unsupported combinations fail before model loading';
        }
        updateLtx2ProfileStatus();
    }
    function buildAspectOptions() {
        var aspects = getActiveAspects();
        var html = '';
        aspects.forEach(function (a) {
            html += '<option value="' + a.label + '">' + a.label + '</option>';
        });
        return html;
    }
    function buildStyleOptions() {
        var html = '';
        stylePresets.forEach(function (s) {
            html += '<option value="' + s.id + '">' + s.label + '</option>';
        });
        return html;
    }
    function buildSchedulerOptions() {
        var html = '';
        schedulerOptions.forEach(function (s) {
            html += '<option value="' + s.value + '">' + s.label + '</option>';
        });
        return html;
    }
    // ── Build DOM ──
    function buildUI() {
        var panel = document.getElementById('panel-generate');
        if (!panel)
            return;
        panel.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'gen-layout gen-workspace-layout';
        // Left panel
        var left = document.createElement('div');
        left.className = 'gen-left gen-workspace-parameters';
        left.id = 'gen-left-panel';
        left.innerHTML = buildGenerateLeftHTML();
        layout.appendChild(left);
        var leftResizer = document.createElement('div');
        leftResizer.id = 'gen-left-resizer';
        leftResizer.className = 'gen-layout-resizer gen-layout-resizer-vertical gen-left-resizer';
        leftResizer.title = 'Drag to resize Parameters';
        layout.appendChild(leftResizer);
        // Center panel
        var center = document.createElement('div');
        center.className = 'gen-center gen-workspace-stage';
        center.innerHTML = buildGenerateTopToolbarHTML() + buildCenterHTML();
        layout.appendChild(center);
        // Floating side toolbar (inside center so it positions relative to center)
        var floatBar = document.createElement('div');
        floatBar.className = 'gen-floating-toolbar';
        floatBar.id = 'gen-floating-toolbar';
        floatBar.innerHTML = buildFloatingToolbarHTML();
        center.appendChild(floatBar);
        var rightResizer = document.createElement('div');
        rightResizer.id = 'gen-right-resizer';
        rightResizer.className = 'gen-layout-resizer gen-layout-resizer-vertical gen-right-resizer';
        rightResizer.title = 'Drag to resize Current Batch';
        layout.appendChild(rightResizer);
        // Right panel
        var right = document.createElement('div');
        right.className = 'gen-right gen-workspace-batch-panel';
        right.id = 'gen-right-panel';
        right.innerHTML = buildGenerateBatchHTML();
        layout.appendChild(right);
        var promptResizer = document.createElement('div');
        promptResizer.id = 'gen-prompt-resizer';
        promptResizer.className = 'gen-layout-resizer gen-layout-resizer-horizontal gen-prompt-resizer';
        promptResizer.title = 'Drag down to enlarge the preview · double-click to hide/show prompts';
        layout.appendChild(promptResizer);
        var promptDock = document.createElement('div');
        promptDock.className = 'gen-workspace-prompt-dock';
        promptDock.innerHTML = buildGeneratePromptHTML();
        layout.appendChild(promptDock);
        var libraryResizer = document.createElement('div');
        libraryResizer.id = 'gen-library-resizer';
        libraryResizer.className = 'gen-layout-resizer gen-layout-resizer-horizontal gen-library-resizer';
        libraryResizer.title = 'Drag down to shrink History · double-click to show/hide its contents';
        layout.appendChild(libraryResizer);
        var library = document.createElement('div');
        library.className = 'gen-workspace-library';
        library.innerHTML = buildGenerateLibraryHTML();
        layout.appendChild(library);
        panel.appendChild(layout);
        cacheElements();
    }

    function generateGroup(id, title, body, open, help) {
        return '<section class="gen-workspace-group gen-param-group" data-param-search="' +
            (title + ' ' + (help || '')).toLowerCase() + '">' +
            '<button type="button" id="' + id + '" class="gen-accordion-header gen-workspace-group-title' +
            (open ? '' : ' closed') + '">' +
            '<span><i data-lucide="chevron-down"></i>' + title + '</span>' +
            (help ? '<span class="gen-workspace-help" title="' + help.replace(/"/g, '&quot;') + '">?</span>' : '') +
            '</button>' +
            '<div id="' + id.replace('-header', '-body') + '" class="gen-accordion-body gen-workspace-group-body' +
            (open ? '' : ' closed') + '">' + body + '</div></section>';
    }

    function disabledParamRow(label, value, search) {
        return '<div class="gen-setting-row gen-param-row gen-parity-disabled" data-param-search="' +
            String(search || label).toLowerCase() + '">' +
            '<label class="gen-setting-label">' + label + '</label>' +
            '<input class="gen-select" value="' + value + '" disabled title="Not admitted by the selected runtime">' +
            '</div>';
    }

    function buildGenerateLeftHTML() {
        var modelBody =
            '<div class="gen-param-row" data-param-search="model checkpoint search">' +
            '<label class="gen-label" for="gen-model-search">Model</label>' +
            '<div class="gen-model-row"><div class="gen-model-picker-wrap" id="gen-model-picker-wrap">' +
            '<input type="text" id="gen-model-search" class="gen-select gen-model-search-input" placeholder="Loading image models..." autocomplete="off">' +
            '<div id="gen-model-dropdown" class="gen-model-dropdown" style="display:none"><div id="gen-model-dropdown-list" class="gen-model-dropdown-list"></div></div>' +
            '</div><button id="gen-model-refresh" class="gen-model-action-btn" title="Refresh local models"><i data-lucide="refresh-cw"></i></button></div>' +
            '<input type="hidden" id="gen-model" value=""><div id="gen-model-warn" class="gen-model-warning"></div></div>';
        var coreBody =
            '<div class="gen-setting-row gen-param-row" id="gen-batch-section" data-param-search="images batch count">' +
            '<label class="gen-setting-label" for="gen-batch">Images</label><input id="gen-batch" type="number" class="gen-number-input" min="1" max="8" value="1">' +
            '<span class="gen-batch-hint">queued as separate jobs</span></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="seed random reuse">' +
            '<label class="gen-setting-label" for="gen-seed">Seed</label><input id="gen-seed" type="number" class="gen-number-input" value="-1">' +
            '<button id="gen-seed-shuffle" class="gen-seed-btn" title="Random seed">↻</button>' +
            '<button id="gen-seed-prev" class="gen-seed-btn" title="Reuse previous seed">↶</button>' +
            '<button id="gen-seed-random-toggle" class="gen-toggle on" title="Random seed"></button></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="steps sampling iterations">' +
            '<label class="gen-setting-label" for="gen-steps">Steps</label><input type="range" id="gen-steps-range" class="gen-range" min="1" max="100" value="20">' +
            '<input type="number" id="gen-steps" class="gen-number-input" min="1" max="500" value="20"></div>' +
            '<div class="gen-setting-row gen-param-row" id="gen-cfg-row" data-param-search="cfg classifier free guidance">' +
            '<label class="gen-setting-label" for="gen-cfg">CFG Scale</label><input type="range" id="gen-cfg-range" class="gen-range" min="0" max="20" step="0.1" value="7">' +
            '<input type="number" id="gen-cfg" class="gen-number-input" min="0" max="200" step="0.1" value="7"></div>' +
            '<div class="gen-setting-row gen-param-row" id="gen-guidance-row" style="display:none" data-param-search="distilled guidance">' +
            '<label class="gen-setting-label" for="gen-guidance">Guidance</label><input type="range" id="gen-guidance-range" class="gen-range" min="0" max="20" step="0.1" value="3.5">' +
            '<input type="number" id="gen-guidance" class="gen-number-input" min="0" max="200" step="0.1" value="3.5"></div>';
        var variationBody =
            '<div class="gen-setting-row gen-param-row" data-param-search="variation seed">' +
            '<label class="gen-setting-label" for="gen-variation-seed">Variation Seed</label><input id="gen-variation-seed" type="number" class="gen-number-input" value="0">' +
            '<button id="gen-variation-shuffle" class="gen-seed-btn" title="Random variation seed">↻</button></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="variation strength">' +
            '<label class="gen-setting-label" for="gen-variation-strength">Strength</label><input id="gen-variation-strength-range" type="range" class="gen-range" min="0" max="1" step="0.01" value="0">' +
            '<input id="gen-variation-strength" type="number" class="gen-number-input" min="0" max="1" step="0.01" value="0"></div>';
        var resolutionBody =
            '<div class="gen-param-row" data-param-search="aspect resolution size width height">' +
            '<div class="gen-aspect-row"><label class="gen-label" for="gen-aspect-dropdown">Aspect</label>' +
            '<select id="gen-aspect-dropdown" class="gen-select">' + buildAspectOptions() + '</select>' +
            '<button id="gen-swap-btn" class="gen-aspect-action" title="Swap admitted dimensions"><i data-lucide="arrow-left-right"></i></button>' +
            '<button id="gen-optimal-btn" class="gen-aspect-action" title="Restore model default"><i data-lucide="sparkles"></i></button>' +
            '<button id="gen-aspect-lock" class="gen-aspect-action" style="display:none" aria-hidden="true"></button></div>' +
            '<div class="gen-dim-row"><span class="gen-dim-label">Width</span><input type="range" id="gen-width-slider" class="gen-range" min="256" max="2048" step="64" value="1024" disabled>' +
            '<input type="number" id="gen-custom-width" class="gen-number-input" value="1024" disabled></div>' +
            '<div class="gen-dim-row"><span class="gen-dim-label">Height</span><input type="range" id="gen-height-slider" class="gen-range" min="256" max="2048" step="64" value="1024" disabled>' +
            '<input type="number" id="gen-custom-height" class="gen-number-input" value="1024" disabled></div>' +
            '<div id="gen-aspect-preview" class="gen-aspect-preview"><span>1024×1024</span></div></div>';
        var sourceBody =
            '<div id="gen-init-drop" class="gen-init-drop">' +
            '<input id="gen-init-image-input" type="file" accept="image/*" aria-label="Choose source image">' +
            '<div id="gen-init-empty"><i data-lucide="image-plus"></i><span>Choose or drop a source image</span></div>' +
            '<img id="gen-init-preview" style="display:none" alt="Selected source image"></div>' +
            '<div id="gen-init-name" class="gen-init-name">No source image</div>' +
            '<button id="gen-init-clear" type="button" class="gen-small-btn destructive" disabled>Clear source</button>' +
            '<div class="gen-capability-note">The selected source is used for admitted img2img and I2V routes. Source Assets can select the same field without uploading again.</div>';
        var samplingBody =
            '<div class="gen-param-row" data-param-search="sampler algorithm"><label class="gen-label" for="gen-sampler">Sampler</label><select id="gen-sampler" class="gen-select"></select></div>' +
            '<div class="gen-param-row" data-param-search="scheduler noise schedule"><label class="gen-label" for="gen-scheduler">Scheduler</label><select id="gen-scheduler" class="gen-select"></select></div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Zero Negative', 'Not admitted', 'zero negative empty negative prompt') +
            disabledParamRow('Seamless Tileable', 'Not admitted', 'seamless tileable x y texture') +
            '</div>';
        var videoBody =
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Video Model', 'Current model selection', 'video model') +
            disabledParamRow('Video Swap Model', 'No two-model runner', 'video swap model') +
            disabledParamRow('Video Swap Percent', 'Not admitted', 'video swap percent') +
            disabledParamRow('Video Resolution', 'Compiled profile', 'video resolution') +
            '</div>' +
            '<div id="gen-video-profile-note" class="gen-capability-note">Loading the admitted video profile…</div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="frames frame count video length">' +
            '<label class="gen-setting-label" for="gen-frames">Frames</label><input id="gen-frames" type="number" class="gen-number-input" min="9" max="481" step="8" value="121">' +
            '<span class="gen-batch-hint">native profile</span></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="fps frame rate">' +
            '<label class="gen-setting-label" for="gen-fps">FPS</label><input id="gen-fps-range" type="range" class="gen-range" min="1" max="60" step="1" value="25">' +
            '<input id="gen-fps" type="number" class="gen-number-input" min="1" max="120" step="1" value="25"></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="duration seconds">' +
            '<label class="gen-setting-label" for="gen-seconds">Duration</label><input id="gen-seconds" type="number" class="gen-number-input" min="0.1" max="120" step="0.01" value="4.84">' +
            '<span id="gen-duration-hint" class="gen-batch-hint">121 frames · 4.8s at 25fps</span></div>' +
            '<div class="gen-param-row" data-param-search="guidance mode distilled dev"><label class="gen-label" for="gen-video-guidance-mode">Guidance mode</label>' +
            '<select id="gen-video-guidance-mode" class="gen-select"><option value="distilled">Distilled</option><option value="dev">Dev CFG</option></select></div>' +
            '<div class="gen-param-row" data-param-search="quant bf16 fp8 int4"><label class="gen-label" for="gen-video-quant">Precision</label>' +
            '<select id="gen-video-quant" class="gen-select"><option value="bf16">BF16</option><option value="fp8">FP8</option><option value="int8">INT8</option><option value="int4">INT4</option></select></div>' +
            '<div id="gen-h3-attention-row" class="gen-param-row" data-param-search="attention cudnn sage int8"><label class="gen-label" for="gen-h3-attention">Attention</label>' +
            '<select id="gen-h3-attention" class="gen-select"><option value="cudnn">cU-DNN · quality default</option><option value="sage-int8">Sage INT8 · experimental</option></select></div>' +
            '<div id="gen-h3-step-cache-row" class="gen-param-row" data-param-search="denoise acceleration cache exact high speed"><label class="gen-label" for="gen-h3-step-cache">Denoise acceleration</label>' +
            '<select id="gen-h3-step-cache" class="gen-select"><option value="exact">Exact · quality default</option><option value="high">Experimental cached · quality loss</option></select></div>' +
            '<div class="gen-param-row" data-param-search="audio generate"><label class="gen-label" for="gen-audio-policy">Audio</label>' +
            '<select id="gen-audio-policy" class="gen-select"><option value="none">No audio</option><option value="generate">Generate audio</option></select></div>' +
            '<div class="gen-param-row" data-param-search="camera motion dolly jib focus static"><label class="gen-label" for="gen-camera-motion">Camera Motion</label>' +
            '<select id="gen-camera-motion" class="gen-select">' +
            '<option value="none">None</option></select></div>';
        var videoConditioningBody =
            '<div class="gen-capability-note">Prompt conditioning is generated automatically by the Mojo Gemma encoder. The path fields are optional expert overrides for an existing prompt-matched cache.</div>' +
            '<div class="gen-param-row" data-param-search="checkpoint compiled profile"><label class="gen-label" for="gen-video-checkpoint">Checkpoint</label>' +
            '<select id="gen-video-checkpoint" class="gen-select"><option value="ltx-2.3-22b-dev-fp8">LTX 2.3 Dev FP8</option></select></div>' +
            '<div class="gen-param-row" data-param-search="prompt enhancer sulphur qwen"><label class="gen-label" for="gen-video-prompt-enhancer">Prompt enhancer</label>' +
            '<select id="gen-video-prompt-enhancer" class="gen-select"><option value="none">Raw prompt (creator workflow default)</option></select></div>' +
            '<div id="gen-video-prompt-enhancer-note" class="gen-capability-note">The selected checkpoint has no registered creator prompt enhancer.</div>' +
            '<div class="gen-param-row" data-param-search="conditioning caps positive"><label class="gen-label" for="gen-caps-positive">Positive conditioning override</label>' +
            '<input id="gen-caps-positive" class="gen-select gen-path-input" placeholder="Automatic when blank"></div>' +
            '<div class="gen-param-row" data-param-search="conditioning caps negative"><label class="gen-label" for="gen-caps-negative">Negative conditioning override</label>' +
            '<input id="gen-caps-negative" class="gen-select gen-path-input" placeholder="Automatic when blank"></div>' +
            '<div class="gen-param-row" data-param-search="noise fixture"><label class="gen-label" for="gen-noise-fixture">Noise fixture</label>' +
            '<input id="gen-noise-fixture" class="gen-select gen-path-input" placeholder="Optional deterministic noise path"></div>';
        var refineBody =
            '<div id="gen-post-upscale-note" class="gen-capability-note">Loading installed post-upscalers…</div>' +
            '<div class="gen-param-row" data-param-search="post upscale super resolution realesrgan seedvr2"><label class="gen-label" for="gen-post-upscaler">Post Upscaler</label>' +
            '<select id="gen-post-upscaler" class="gen-select"><option value="none">Native output</option></select></div>' +
            '<div class="gen-param-row" data-param-search="post upscale factor resolution 2x 4x"><label class="gen-label" for="gen-post-upscale-factor">Output Scale</label>' +
            '<select id="gen-post-upscale-factor" class="gen-select"><option value="2">2×</option><option value="4">4×</option></select></div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Refiner Control Percentage', 'Not admitted', 'refiner control percentage') +
            disabledParamRow('Refiner Method', 'No second-pass runtime', 'refiner method') +
            disabledParamRow('Refiner Upscale', '1.0', 'refiner upscale') +
            disabledParamRow('Refiner Model', 'Use base only', 'refiner model override') +
            disabledParamRow('Refiner VAE', 'Use model VAE', 'refiner vae') +
            disabledParamRow('Refiner Sampler', 'Runtime default', 'refiner sampler') +
            disabledParamRow('Refiner Steps', 'Runtime unavailable', 'refiner steps') +
            disabledParamRow('Refiner CFG', 'Runtime unavailable', 'refiner cfg') +
            disabledParamRow('Refiner Do Tiling', 'Unavailable', 'refiner tiling') +
            '</div>';
        var runtimeInternalBody =
            '<label class="gen-check-row gen-param-row" data-param-search="no seed increment fixed batch seed"><input id="gen-no-seed-increment" type="checkbox"> No Seed Increment</label>' +
            '<label class="gen-check-row gen-param-row" data-param-search="continue after errors batch queue"><input id="gen-continue-after-errors" type="checkbox" checked> Continue After Errors</label>' +
            '<div class="gen-param-row" data-param-search="personal note metadata"><label class="gen-label" for="gen-personal-note">Personal Note</label>' +
            '<textarea id="gen-personal-note" class="gen-textarea gen-compact-textarea" rows="2" placeholder="Saved with reusable browser parameters"></textarea></div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Batch Size', 'Use Images in Core', 'batch size') +
            disabledParamRow('Alt Resolution Height Multiplier', 'Use admitted aspect', 'alternate resolution height multiplier') +
            disabledParamRow('Raw Resolution', 'Use admitted dimensions', 'raw resolution') +
            disabledParamRow('Output Intermediate Images', 'Worker does not emit intermediates', 'output intermediate images') +
            disabledParamRow('Do Not Save', 'Server result contract requires an artifact', 'do not save') +
            disabledParamRow('Do Not Save Intermediates', 'No intermediate artifacts emitted', 'do not save intermediates') +
            disabledParamRow('No Previews', 'Progress preview policy is runtime-owned', 'no previews') +
            disabledParamRow('No Load Models', 'Resident worker routing is automatic', 'no load models') +
            disabledParamRow('No Internal Special Handling', 'Not applicable', 'no internal special handling') +
            disabledParamRow('Webhooks', 'Not admitted', 'webhooks') +
            disabledParamRow('[Internal] Backend Type', 'Mojo worker', 'internal backend type') +
            disabledParamRow('Exact Backend ID', 'Automatic worker dispatch', 'backend id') +
            disabledParamRow('Wildcard Seed', 'Prompt wildcards unavailable', 'wildcard seed') +
            disabledParamRow('Wildcard Seed Behavior', 'Random', 'wildcard seed behavior') +
            disabledParamRow('Image Format', 'Runtime artifact contract', 'image format') +
            disabledParamRow('Color Depth', '8-bit RGB', 'color depth') +
            disabledParamRow('Override Outpath Format', 'Server output root', 'override output path format') +
            disabledParamRow('Model Specific Enhancements', 'Worker profile owns enhancements', 'model specific enhancements') +
            disabledParamRow('Custom Workflow', 'Use the Workflow tab', 'custom workflow') +
            '</div>';
        var advancedVideoBody =
            '<div class="gen-capability-note" id="gen-video-advanced-note">Select an admitted video model to use the video controls.</div>' +
            '<div class="gen-workspace-parity-list">' +
            '<div class="gen-setting-row gen-param-row" data-param-search="video format mp4"><label class="gen-setting-label">Video Format</label><input id="gen-video-format" class="gen-select" value="MP4" disabled></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="video preview type animate"><label class="gen-setting-label">Video Preview Type</label><input id="gen-video-preview-type" class="gen-select" value="Animate" disabled></div>' +
            disabledParamRow('Video Boomerang', 'Not admitted', 'video boomerang') +
            disabledParamRow('Video Audio Input', 'No audio-conditioning input route', 'video audio input') +
            disabledParamRow('Video Audio Reference', 'No reference-audio input route', 'video audio reference') +
            disabledParamRow('Video Min CFG', 'Not admitted', 'video minimum cfg') +
            disabledParamRow('Video Motion Bucket', 'Not admitted', 'video motion bucket') +
            disabledParamRow('Video Augmentation Level', 'Not admitted', 'video augmentation level') +
            disabledParamRow('Trim Video Start Frames', 'Not admitted', 'trim video start frames') +
            disabledParamRow('Trim Video End Frames', 'Not admitted', 'trim video end frames') +
            disabledParamRow('Frame Interpolation', 'Not admitted', 'video frame interpolation') +
            '</div>';
        var videoExtendBody =
            '<div class="gen-capability-note">Video extension is not exposed as generation because no Serenity request runner currently admits overlap-conditioned continuation.</div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Video Extend Model', 'No admitted runner', 'video extend model') +
            disabledParamRow('Video Extend Swap Model', 'No admitted runner', 'video extend swap model') +
            disabledParamRow('Video Extend Frame Overlap', '9', 'video extend frame overlap') +
            disabledParamRow('Video Extend Format', 'MP4', 'video extend format') +
            '</div>';
        var modelAddonsBody =
            '<div class="gen-capability-note">The selected checkpoint manifest owns these components. Overrides are never posted unless a model capability explicitly admits them.</div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('VAE', 'Automatic / checkpoint VAE', 'vae automatic override') +
            disabledParamRow('Pixel Decoder Model', 'Checkpoint-owned', 'pixel decoder model') +
            disabledParamRow('CLIP-L Model', 'Checkpoint-owned', 'clip l model') +
            disabledParamRow('CLIP-G Model', 'Checkpoint-owned', 'clip g model') +
            disabledParamRow('CLIP-Vision Model', 'Canvas-owned conditioning', 'clip vision model') +
            disabledParamRow('T5-XXL Model', 'Checkpoint-owned', 't5 xxl model') +
            disabledParamRow('LLaVA Model', 'Checkpoint-owned', 'llava model') +
            disabledParamRow('LLaMA Model', 'Checkpoint-owned', 'llama model') +
            disabledParamRow('Gemma Model', 'Checkpoint-owned', 'gemma model') +
            disabledParamRow('GPT-OSS Model', 'Checkpoint-owned', 'gpt oss model') +
            disabledParamRow('Mistral Model', 'Checkpoint-owned', 'mistral model') +
            disabledParamRow('Qwen Model', 'Checkpoint-owned', 'qwen model') +
            disabledParamRow('Torch Compile', 'Mojo compiled worker', 'torch compile') +
            disabledParamRow('Override Prediction Type', 'Model manifest default', 'prediction type') +
            disabledParamRow('Negative Model', 'Not admitted', 'negative model') +
            disabledParamRow('Negative Model Include LoRAs', 'Not applicable', 'negative model include loras') +
            '</div>';
        var dynamicThresholdBody =
            '<div class="gen-capability-note">Dynamic thresholding is not implemented in the current Mojo samplers.</div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Mimic Scale', '7.0', 'dynamic threshold mimic scale') +
            disabledParamRow('Threshold Percentile', '1.0', 'dynamic threshold percentile') +
            disabledParamRow('CFG Scale Mode', 'Constant', 'dynamic threshold cfg scale mode') +
            disabledParamRow('CFG Scale Minimum', '1.0', 'dynamic threshold cfg minimum') +
            disabledParamRow('Mimic Scale Mode', 'Constant', 'dynamic threshold mimic scale mode') +
            disabledParamRow('Mimic Scale Minimum', '1.0', 'dynamic threshold mimic minimum') +
            disabledParamRow('Scheduler Value', '4.0', 'dynamic threshold scheduler value') +
            disabledParamRow('Separate Feature Channels', 'Enabled', 'dynamic threshold separate feature channels') +
            disabledParamRow('Scaling Startpoint', 'MEAN', 'dynamic threshold scaling startpoint') +
            disabledParamRow('Variability Measure', 'AD', 'dynamic threshold variability measure') +
            disabledParamRow('Interpolate Phi', '1.0', 'dynamic threshold interpolate phi') +
            '</div>';
        var advancedSamplingBody =
            '<div id="gen-advanced-sampling-note" class="gen-capability-note">Advanced sampler controls are capability-driven. Unsupported fields stay disabled and are not posted.</div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="sigma shift flow schedule">' +
            '<label class="gen-setting-label" for="gen-sigma-shift">Sigma Shift</label><input id="gen-sigma-shift-range" type="range" class="gen-range" min="0.01" max="20" step="0.01" value="3" disabled>' +
            '<input id="gen-sigma-shift" type="number" class="gen-number-input" min="0.01" max="100" step="0.01" value="3" disabled></div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('Sampler Sigma Min', 'Not admitted', 'sampler sigma min') +
            disabledParamRow('Sampler Sigma Max', 'Not admitted', 'sampler sigma max') +
            disabledParamRow('Sampler Eta', 'Not admitted', 'sampler eta') +
            disabledParamRow('Sampler Rho', 'Not admitted', 'sampler rho') +
            disabledParamRow('CLIP Stop At Layer', 'Not admitted', 'clip stop layer clip skip') +
            disabledParamRow('VAE Tile Size', 'Runtime-owned', 'vae tile size') +
            disabledParamRow('VAE Tile Overlap', 'Runtime-owned', 'vae tile overlap') +
            disabledParamRow('VAE Temporal Tile Size', 'Runtime-owned', 'vae temporal tile size') +
            disabledParamRow('VAE Temporal Tile Overlap', 'Runtime-owned', 'vae temporal tile overlap') +
            disabledParamRow('Preferred DType', 'Model manifest default', 'preferred dtype') +
            disabledParamRow('End Steps Early', 'Not admitted', 'end steps early') +
            disabledParamRow('Shifted Latent Average Init', 'Not admitted', 'shifted latent average init') +
            disabledParamRow('Restart Sampling', 'Not admitted', 'restart sampling') +
            disabledParamRow('EasyCache Mode', 'Disabled', 'easycache') +
            disabledParamRow('Refiner HyperTile', 'Not admitted', 'refiner hypertile') +
            '</div>';
        var alternateGuidanceBody =
            '<div class="gen-capability-note">Alternate guidance implementations are not production-admitted by the selected sampler.</div>' +
            '<div class="gen-workspace-parity-list">' +
            disabledParamRow('FreeU Apply To', 'Both', 'freeu apply to') +
            disabledParamRow('FreeU Version', '1', 'freeu version') +
            disabledParamRow('FreeU Block One', '1.1', 'freeu block one') +
            disabledParamRow('FreeU Block Two', '1.2', 'freeu block two') +
            disabledParamRow('FreeU Skip One', '0.9', 'freeu skip one') +
            disabledParamRow('FreeU Skip Two', '0.2', 'freeu skip two') +
            '</div>';
        var outputBody =
            '<div class="gen-setting-row gen-param-row" data-param-search="output format png mp4"><label class="gen-setting-label">Format</label>' +
            '<input id="gen-output-format" class="gen-select" value="PNG" disabled><span class="gen-batch-hint">runtime-owned</span></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="color depth"><label class="gen-setting-label">Color Depth</label><input class="gen-select" value="8-bit RGB" disabled></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="output path format"><label class="gen-setting-label">Output Path</label><input class="gen-select" value="Server output root" disabled></div>' +
            '<div class="gen-capability-note">PNG for images and MP4 for video come from the admitted runtime artifact contract.</div>';
        var loraBody =
            '<div id="gen-lora-capability" class="gen-capability-note"></div><div id="gen-lora-list" class="gen-lora-list"></div>' +
            '<select id="gen-lora-picker" class="gen-lora-dropdown"><option value="" disabled selected>Select a LoRA…</option></select>';
        return '<div class="gen-workspace-parameter-head"><div><strong>Parameters</strong><span id="gen-arch-badge" class="gen-arch-badge">IMAGE</span></div>' +
            '<button id="gen-quick-reset" type="button" class="gen-small-btn" title="Reset selected model defaults">Reset</button></div>' +
            '<div class="gen-workspace-filter"><i data-lucide="search"></i><input id="gen-param-filter" type="search" placeholder="Filter parameters..."></div>' +
            '<div class="gen-workspace-param-scroll">' +
            generateGroup('gen-settings-header', 'Model', modelBody, true, 'Select an installed model admitted by the image or video runtime.') +
            generateGroup('gen-core-header', 'Core Parameters', coreBody, true, 'The controls used by every admitted generation backend.') +
            '<section id="gen-variation-section">' + generateGroup('gen-variation-header', 'Variation Seed', variationBody, false, 'Blend deterministic secondary noise into supported model families.') + '</section>' +
            generateGroup('gen-image-header', 'Resolution', resolutionBody, false, 'Only compiled, production-admitted shapes are listed.') +
            generateGroup('gen-source-header', 'Source Image', sourceBody, false, 'Upload once or reuse an Asset for admitted img2img and I2V generation.') +
            generateGroup('gen-sampling-header', 'Sampling', samplingBody, false, 'Sampler and scheduler values come from the selected backend capability report.') +
            '<section id="gen-video-section">' + generateGroup('gen-video-header', 'Video', videoBody, false, 'Video duration, frame rate, guidance, quantization, and audio parameters.') + '</section>' +
            '<section id="gen-video-conditioning-section">' + generateGroup('gen-video-conditioning-header', 'Video Conditioning', videoConditioningBody, false, 'Prompt-matched LTX2 conditioning and optional deterministic noise artifacts.') + '</section>' +
            '<section id="gen-lora-section">' + generateGroup('gen-lora-header', 'LoRAs', loraBody, false, 'Only backend-admitted LoRA counts are allowed.') + '</section>' +
            '<div class="gen-workspace-advanced-only">' +
            generateGroup('gen-refine-header', 'Refine / Upscale', refineBody, false, 'Refiner and upscale controls with truthful current runtime admission.') +
            generateGroup('gen-internal-header', 'Runtime & Output', runtimeInternalBody, false, 'Batch seed behavior, queue-error policy, local notes, and output/runtime controls.') +
            generateGroup('gen-advanced-video-header', 'Advanced Video', advancedVideoBody, false, 'Video output, audio-conditioning, trim, and interpolation controls.') +
            generateGroup('gen-video-extend-header', 'Video Extend', videoExtendBody, false, 'Video continuation and overlap parameters.') +
            generateGroup('gen-model-addons-header', 'Advanced Model Addons', modelAddonsBody, false, 'VAE, text encoder, decoder, dtype, and model override controls.') +
            generateGroup('gen-dynamic-threshold-header', 'Dynamic Thresholding', dynamicThresholdBody, false, 'Dynamic CFG thresholding parameters.') +
            generateGroup('gen-advanced-sampling-header', 'Advanced Sampling', advancedSamplingBody, false, 'Sigma schedule and advanced sampler parameters.') +
            generateGroup('gen-alternate-guidance-header', 'Alternate Guidance', alternateGuidanceBody, false, 'CFG override, FreeU, and alternate guidance parameters.') +
            '</div>' +
            generateGroup('gen-output-header', 'Output', outputBody, false, 'Output container is selected by the admitted image or video route.') +
            '</div>' +
            '<label class="gen-workspace-advanced-toggle"><input id="gen-show-advanced" type="checkbox" checked> Display Advanced Options <span id="gen-advanced-count"></span></label>' +
            '<div id="gen-left-progress" class="gen-progress gen-left-progress"><div id="gen-left-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-left-progress-label" class="gen-left-progress-label"></div>';
    }

    function buildGenerateTopToolbarHTML() {
        return '<div class="gen-workspace-topbar">' +
            '<div class="gen-workspace-model-info"><span id="gen-model-badge" class="gen-model-badge"></span><span id="gen-runtime-label">Serenity image runtime</span></div>' +
            '<div id="gen-activity-status" class="gen-activity-status" data-state="idle" role="status" aria-live="polite">' +
            '<span class="gen-activity-dot"></span><span id="gen-activity-text">Idle</span></div>' +
            '<div class="gen-workspace-quick-tools"><button id="gen-toolbar-copy" class="gen-toolbar-btn" title="Copy current image URL"><i data-lucide="copy"></i></button>' +
            '<button id="gen-toolbar-delete" class="gen-toolbar-btn" title="Clear preview"><i data-lucide="trash-2"></i></button>' +
            '<button id="gen-toolbar-toggle-gallery" class="gen-toolbar-btn active" title="Toggle batch panel"><i data-lucide="panel-right"></i></button></div></div>';
    }

    function buildGeneratePromptHTML() {
        return '<div class="gen-workspace-prompts"><div class="gen-workspace-prompt-main">' +
            '<div class="gen-prompt-label-row"><label class="gen-label" for="gen-prompt">Prompt</label><span id="gen-token-count" class="gen-token-count">~0 tokens</span></div>' +
            '<textarea id="gen-prompt" class="gen-textarea gen-workspace-prompt-textarea" rows="3" placeholder="Describe the image you want..."></textarea>' +
            '<div class="gen-workspace-prompt-options"><label class="gen-label" for="gen-style-preset">Style</label><select id="gen-style-preset" class="gen-select">' + buildStyleOptions() + '</select>' +
            '<div id="gen-style-preview" class="gen-style-preview" style="display:none"></div></div></div>' +
            '<div id="gen-neg-section" class="gen-workspace-negative"><label class="gen-label" for="gen-neg-prompt">Negative Prompt</label>' +
            '<textarea id="gen-neg-prompt" class="gen-textarea" rows="3" placeholder="What to avoid..."></textarea></div></div>' +
            '<div class="gen-workspace-generate-column"><button id="gen-btn" class="gen-btn"><i data-lucide="wand-2"></i><span>Generate</span></button>' +
            '<div class="gen-toolbar-batch"><label for="gen-toolbar-batch-input">Images</label><input type="number" id="gen-toolbar-batch-input" class="gen-toolbar-batch-input" min="1" max="8" value="1">' +
            '<button id="gen-toolbar-batch-up" title="Increase batch">+</button><button id="gen-toolbar-batch-down" title="Decrease batch">−</button></div></div>';
    }

    function buildGenerateBatchHTML() {
        return '<div class="gen-workspace-batch-head"><strong>Current Batch</strong><span id="gen-batch-status">Idle</span></div>' +
            '<div id="gen-batch-strip" class="gen-workspace-batch-strip"><div class="gen-workspace-batch-empty">New results appear here</div></div>';
    }

    function buildGenerateLibraryHTML() {
        return '<div class="gen-workspace-library-tabs">' +
            '<button class="gen-library-tab active" data-library="history">History</button>' +
            '<button class="gen-library-tab" data-library="assets">Assets</button>' +
            '<button class="gen-library-tab" data-library="presets">Presets</button>' +
            '<button class="gen-library-tab" data-library="workflows">Workflows</button>' +
            '<button class="gen-library-tab" data-library="models">Models</button>' +
            '<button class="gen-library-tab" data-library="loras">LoRAs</button>' +
            '<div class="gen-library-spacer"></div>' +
            '<button id="gen-gallery-upload-btn" class="gen-gallery-subtab-btn" title="Upload source image"><i data-lucide="upload"></i></button>' +
            '<input type="file" id="gen-gallery-upload-input" accept="image/*" multiple style="display:none">' +
            '<button id="gen-gallery-search-btn" class="gen-gallery-subtab-btn" title="Search current library"><i data-lucide="search"></i></button>' +
            '<input id="gen-gallery-search-input" class="gen-gallery-search" type="text" placeholder="Search current library...">' +
            '<button id="gen-gallery-settings-btn" class="gen-gallery-subtab-btn" title="History display settings"><i data-lucide="settings"></i></button>' +
            '<button id="gen-gallery-clear" class="gen-gallery-clear">Delete all</button></div>' +
            '<div id="gen-gallery-popover" class="gen-gallery-popover"><div class="gen-popover-row"><span class="gen-popover-label">Thumbnail size</span>' +
            '<input type="range" id="gen-thumb-size-slider" class="gen-range gen-popover-slider" min="45" max="200" step="5" value="90"><span id="gen-thumb-size-val">90</span></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Newest first</span><button id="gen-sort-direction-toggle" class="gen-toggle on"></button><span id="gen-sort-direction-label">Newest</span></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Starred first</span><button id="gen-starred-first-toggle" class="gen-toggle"></button></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Auto-switch new</span><button id="gen-auto-switch-toggle" class="gen-toggle on"></button></div></div>' +
            '<div class="gen-library-panel active" data-library-panel="history"><div id="gen-images-content"><div class="gen-gallery-grid-wrap">' +
            '<div id="gen-gallery-grid" class="gen-gallery-grid" tabindex="0" role="region" aria-label="Generation history grouped by date"></div><div id="gen-selection-badge" class="gen-selection-badge"></div></div></div>' +
            '<div id="gen-bulk-bar" class="gen-bulk-bar"><button id="gen-bulk-star" class="gen-bulk-btn">Star All</button><button id="gen-bulk-unstar" class="gen-bulk-btn">Unstar All</button>' +
            '<button id="gen-bulk-download" class="gen-bulk-btn">Download</button><button id="gen-bulk-delete" class="gen-bulk-btn destructive">Delete</button></div></div>' +
            '<div id="gen-assets-content" class="gen-library-panel" data-library-panel="assets">' +
            '<div id="gen-assets-grid" class="gen-assets-grid"></div>' +
            '<div id="gen-assets-placeholder" class="gen-assets-placeholder">Loading assets…</div></div>' +
            '<div class="gen-library-panel" data-library-panel="presets"><div class="gen-preset-toolbar"><input id="gen-preset-name" class="gen-select" placeholder="Preset name">' +
            '<button id="gen-preset-save" class="gen-small-btn">Save current</button></div><div id="gen-preset-list" class="gen-library-card-grid"></div></div>' +
            '<div class="gen-library-panel" data-library-panel="workflows"><div class="gen-workflow-library-note">Bundled and user workflows open in Canvas. Compatibility is checked against this Serenity runtime before you open them.</div>' +
            '<div id="gen-workflow-list" class="gen-library-card-grid"></div></div>' +
            '<div class="gen-library-panel" data-library-panel="models"><div id="gen-library-models" class="gen-library-card-grid"></div></div>' +
            '<div class="gen-library-panel" data-library-panel="loras"><div id="gen-library-loras" class="gen-library-card-grid"></div></div>';
    }
    function buildFloatingToolbarHTML() {
        return '' +
            '<button id="gen-float-toggle-left" class="gen-float-btn" title="Toggle left panel"><i data-lucide="panel-left"></i></button>' +
            '<button id="gen-float-generate" class="gen-float-btn gen-float-generate" title="Generate"><i data-lucide="sparkles"></i></button>' +
            '<button id="gen-float-cancel" class="gen-float-btn gen-float-cancel" title="Cancel"><i data-lucide="x"></i></button>' +
            '<button id="gen-float-delete" class="gen-float-btn gen-float-delete" title="Delete current"><i data-lucide="trash-2"></i></button>';
    }
    function buildCenterHTML() {
        return '' +
            '<div class="gen-center-content">' +
            '<div id="gen-empty" class="gen-empty">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>' +
            '<span class="gen-empty-text">Your image will appear here</span>' +
            '</div>' +
            '<img id="gen-preview-img" class="gen-preview-img" style="display:none" alt="Generated image">' +
            '<video id="gen-preview-video" class="gen-preview-video" style="display:none" autoplay loop playsinline controls></video>' +
            '<div id="gen-action-bar" class="gen-action-bar" style="display:none">' +
            '<button class="gen-action-btn" id="gen-download" title="Download"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>' +
            '<button class="gen-action-btn" id="gen-to-canvas" title="Coming in Canvas tab" disabled><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg></button>' +
            '<button class="gen-action-btn" id="gen-to-timeline" title="Send to Timeline"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="17" y1="17" x2="22" y2="17"/></svg></button>' +
            '<button class="gen-action-btn gen-action-btn-wide" id="gen-reuse-params" title="Restore every parameter used for this result"><i data-lucide="rotate-ccw"></i><span>Reuse parameters</span></button>' +
            '<button class="gen-action-btn" id="gen-clear-preview" title="Clear"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg></button>' +
            '</div>' +
            // Metadata panel (below action bar)
            '<div id="gen-metadata-panel" class="gen-metadata-panel">' +
            '<div id="gen-metadata-summary" class="gen-metadata-summary"></div>' +
            '<div id="gen-metadata-full" class="gen-metadata-full"></div>' +
            '</div>' +
            '</div>' +
            '<div id="gen-progress-label" class="gen-progress-label"></div>' +
            '<div id="gen-progress" class="gen-progress"><div id="gen-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-error-banner" class="gen-error-banner"></div>' +
            '<div id="gen-ws-indicator" class="gen-ws-indicator"><span class="gen-ws-dot"></span><span>Reconnecting...</span></div>';
    }
    function cacheElements() {
        els.model = document.getElementById('gen-model');
        els.modelSearch = document.getElementById('gen-model-search');
        els.modelDropdown = document.getElementById('gen-model-dropdown');
        els.modelDropdownList = document.getElementById('gen-model-dropdown-list');
        els.modelPickerWrap = document.getElementById('gen-model-picker-wrap');
        els.modelWarn = document.getElementById('gen-model-warn');
        els.prompt = document.getElementById('gen-prompt');
        els.negPrompt = document.getElementById('gen-neg-prompt');
        els.negSection = document.getElementById('gen-neg-section');
        els.customWidth = document.getElementById('gen-custom-width');
        els.customHeight = document.getElementById('gen-custom-height');
        els.layout = document.querySelector('#panel-generate .gen-workspace-layout');
        els.leftResizer = document.getElementById('gen-left-resizer');
        els.rightResizer = document.getElementById('gen-right-resizer');
        els.promptResizer = document.getElementById('gen-prompt-resizer');
        els.libraryResizer = document.getElementById('gen-library-resizer');
        els.videoSection = document.getElementById('gen-video-section');
        els.videoConditioningSection = document.getElementById('gen-video-conditioning-section');
        els.framesInput = document.getElementById('gen-frames');
        els.secondsInput = document.getElementById('gen-seconds');
        els.fpsInput = document.getElementById('gen-fps');
        els.fpsRange = document.getElementById('gen-fps-range');
        els.durationHint = document.getElementById('gen-duration-hint');
        els.videoGuidanceMode = document.getElementById('gen-video-guidance-mode');
        els.videoQuant = document.getElementById('gen-video-quant');
        els.h3Attention = document.getElementById('gen-h3-attention');
        els.h3AttentionRow = document.getElementById('gen-h3-attention-row');
        els.h3StepCache = document.getElementById('gen-h3-step-cache');
        els.h3StepCacheRow = document.getElementById('gen-h3-step-cache-row');
        els.cameraMotion = document.getElementById('gen-camera-motion');
        els.videoCheckpoint = document.getElementById('gen-video-checkpoint');
        els.videoPromptEnhancer = document.getElementById('gen-video-prompt-enhancer');
        els.videoPromptEnhancerNote = document.getElementById('gen-video-prompt-enhancer-note');
        els.capsPositive = document.getElementById('gen-caps-positive');
        els.capsNegative = document.getElementById('gen-caps-negative');
        els.noiseFixture = document.getElementById('gen-noise-fixture');
        els.audioPolicy = document.getElementById('gen-audio-policy');
        els.postUpscaler = document.getElementById('gen-post-upscaler');
        els.postUpscaleFactor = document.getElementById('gen-post-upscale-factor');
        els.postUpscaleNote = document.getElementById('gen-post-upscale-note');
        els.sigmaShift = document.getElementById('gen-sigma-shift');
        els.sigmaShiftRange = document.getElementById('gen-sigma-shift-range');
        els.noSeedIncrement = document.getElementById('gen-no-seed-increment');
        els.continueAfterErrors = document.getElementById('gen-continue-after-errors');
        els.personalNote = document.getElementById('gen-personal-note');
        els.showAdvanced = document.getElementById('gen-show-advanced');
        els.scail2Section = document.getElementById('gen-scail2-section');
        els.scail2Mode = document.getElementById('gen-scail2-mode');
        els.scail2UploadStatus = document.getElementById('gen-scail2-upload-status');
        els.imageHeader = document.getElementById('gen-image-header');
        els.imageBody = document.getElementById('gen-image-body');
        els.settingsHeader = document.getElementById('gen-settings-header');
        els.settingsBody = document.getElementById('gen-settings-body');
        els.steps = document.getElementById('gen-steps');
        els.stepsRange = document.getElementById('gen-steps-range');
        els.cfgRow = document.getElementById('gen-cfg-row');
        els.cfg = document.getElementById('gen-cfg');
        els.cfgRange = document.getElementById('gen-cfg-range');
        els.guidanceRow = document.getElementById('gen-guidance-row');
        els.guidance = document.getElementById('gen-guidance');
        els.guidanceRange = document.getElementById('gen-guidance-range');
        els.sampler = document.getElementById('gen-sampler');
        els.scheduler = document.getElementById('gen-scheduler');
        els.seed = document.getElementById('gen-seed');
        els.seedShuffle = document.getElementById('gen-seed-shuffle');
        els.btn = document.getElementById('gen-btn');
        els.empty = document.getElementById('gen-empty');
        els.previewImg = document.getElementById('gen-preview-img');
        els.previewVideo = document.getElementById('gen-preview-video');
        els.actionBar = document.getElementById('gen-action-bar');
        els.download = document.getElementById('gen-download');
        els.reuseParams = document.getElementById('gen-reuse-params');
        els.clearPreview = document.getElementById('gen-clear-preview');
        els.progress = document.getElementById('gen-progress');
        els.progressBar = document.getElementById('gen-progress-bar');
        els.progressLabel = document.getElementById('gen-progress-label');
        els.leftProgress = document.getElementById('gen-left-progress');
        els.leftProgressBar = document.getElementById('gen-left-progress-bar');
        els.leftProgressLabel = document.getElementById('gen-left-progress-label');
        els.activityStatus = document.getElementById('gen-activity-status');
        els.activityText = document.getElementById('gen-activity-text');
        els.errorBanner = document.getElementById('gen-error-banner');
        els.wsIndicator = document.getElementById('gen-ws-indicator');
        els.galleryGrid = document.getElementById('gen-gallery-grid');
        els.galleryClear = document.getElementById('gen-gallery-clear');
        els.assetsContent = document.getElementById('gen-assets-content');
        els.assetsGrid = document.getElementById('gen-assets-grid');
        els.assetsPlaceholder = document.getElementById('gen-assets-placeholder');
        // New elements
        els.leftPanel = document.getElementById('gen-left-panel');
        els.rightPanel = document.getElementById('gen-right-panel');
        els.aspectDropdown = document.getElementById('gen-aspect-dropdown');
        els.aspectPreview = document.getElementById('gen-aspect-preview');
        els.modelBadge = document.getElementById('gen-model-badge');
        els.toolbarBatchInput = document.getElementById('gen-toolbar-batch-input');
        els.batchStrip = document.getElementById('gen-batch-strip');
        els.batchStatus = document.getElementById('gen-batch-status');
    }
    // ── Aspect Preview Box ──
    function updateAspectPreview() {
        if (!els.aspectPreview)
            return;
        var maxDim = 100;
        var w = state.width;
        var h = state.height;
        var scale;
        if (w >= h) {
            scale = maxDim / w;
        }
        else {
            scale = maxDim / h;
        }
        var pw = Math.max(16, Math.round(w * scale));
        var ph = Math.max(16, Math.round(h * scale));
        els.aspectPreview.style.width = pw + 'px';
        els.aspectPreview.style.height = ph + 'px';
        // Determine label
        var label = state.width + '\u00d7' + state.height;
        var aspects = getActiveAspects();
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                label = aspects[i].label;
                break;
            }
        }
        els.aspectPreview.innerHTML = '<span>' + label + '</span>';
    }
    // ── Aspect Dropdown Sync ──
    function syncAspectDropdown() {
        if (!els.aspectDropdown)
            return;
        var aspects = getActiveAspects();
        var found = false;
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].label === 'Free')
                continue; // skip Free when matching
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                els.aspectDropdown.value = aspects[i].label;
                found = true;
                break;
            }
        }
        if (!found) {
            // If current dropdown is "Free", keep it; otherwise set to custom
            if (els.aspectDropdown.value !== 'Free') {
                els.aspectDropdown.value = 'custom';
            }
        }
    }
    // ── Event Binding ──
    function bindScail2Upload(inputId, stateKey, label) {
        var input = document.getElementById(inputId);
        if (!input)
            return;
        input.addEventListener('change', function () {
            var file = input.files && input.files[0];
            if (!file)
                return;
            var form = new FormData();
            form.append('file', file, file.name);
            state.mediaUploadsInFlight += 1;
            state[stateKey] = '';
            if (els.scail2UploadStatus)
                els.scail2UploadStatus.textContent = 'Uploading ' + label + '…';
            fetch('/upload/media', { method: 'POST', body: form })
                .then(function (response) {
                return response.text().then(function (body) {
                    var data = {};
                    try { data = JSON.parse(body); }
                    catch (error) { data = { detail: body }; }
                    if (!response.ok || !data.path)
                        throw new Error(data.detail || ('HTTP ' + response.status));
                    return data;
                });
            })
                .then(function (data) {
                state[stateKey] = data.path;
                if (els.scail2UploadStatus)
                    els.scail2UploadStatus.textContent = label + ' ready: ' + data.name;
            })
                .catch(function (error) {
                state[stateKey] = '';
                showError('Could not upload ' + label + ': ' + error.message);
            })
                .finally(function () {
                state.mediaUploadsInFlight = Math.max(0, state.mediaUploadsInFlight - 1);
            });
        });
    }
    function bindScail2MultiUpload(inputId, stateKey, label) {
        var input = document.getElementById(inputId);
        if (!input)
            return;
        input.addEventListener('change', function () {
            var files = Array.prototype.slice.call(input.files || []);
            if (!files.length)
                return;
            if (files.length > 3) {
                input.value = '';
                state[stateKey] = [];
                showError('SCAIL-2 accepts at most 3 ' + label);
                return;
            }
            state.mediaUploadsInFlight += files.length;
            state[stateKey] = [];
            Promise.all(files.map(function (file) {
                var form = new FormData();
                form.append('file', file, file.name);
                return fetch('/upload/media', { method: 'POST', body: form })
                    .then(function (response) {
                    return response.text().then(function (body) {
                        var data = {};
                        try { data = JSON.parse(body); }
                        catch (error) { data = { detail: body }; }
                        if (!response.ok || !data.path)
                            throw new Error(data.detail || ('HTTP ' + response.status));
                        return data.path;
                    });
                });
            }))
                .then(function (paths) {
                state[stateKey] = paths;
                if (els.scail2UploadStatus)
                    els.scail2UploadStatus.textContent = paths.length + ' ' + label + ' ready';
            })
                .catch(function (error) {
                state[stateKey] = [];
                showError('Could not upload ' + label + ': ' + error.message);
            })
                .finally(function () {
                state.mediaUploadsInFlight = Math.max(0, state.mediaUploadsInFlight - files.length);
            });
        });
    }
    function storedLayoutNumber(key, fallback) {
        var stored = localStorage.getItem(key);
        if (stored == null || stored === '')
            return fallback;
        var value = Number(stored);
        return Number.isFinite(value) ? value : fallback;
    }
    function syncPanelLayoutState() {
        if (!els.layout)
            return;
        els.layout.classList.toggle('gen-left-collapsed', !state.leftPanelVisible);
        els.layout.classList.toggle('gen-right-collapsed', !state.rightPanelVisible);
    }
    function bindLayoutResizers() {
        if (!els.layout)
            return;
        // Generate is a primary action and must remain reachable. The prompt
        // dock may collapse to one compact button row, but never to zero.
        var minPromptHeight = 54;
        var minLibraryHeight = 38;
        var minStageHeight = 160;
        var hasStoredVerticalLayout = localStorage.getItem('sf-gen-prompt-height') != null ||
            localStorage.getItem('sf-gen-library-height') != null;
        var leftWidth = Math.max(260, Math.min(560, storedLayoutNumber('sf-gen-left-width', 380)));
        var rightWidth = Math.max(180, Math.min(430, storedLayoutNumber('sf-gen-right-width', 250)));
        var promptHeight = Math.max(minPromptHeight, Math.min(300, storedLayoutNumber('sf-gen-prompt-height', 154)));
        var libraryHeight = Math.max(minLibraryHeight, Math.min(420, storedLayoutNumber('sf-gen-library-height', 230)));
        function constrainVerticalTracks(reserveDefaultPreview) {
            var height = els.layout.clientHeight;
            if (!height)
                return;
            // Use the larger preview only for the untouched startup layout.
            // Once the user resizes either lower panel, preserve that choice and
            // allow the preview to shrink to its real CSS minimum.
            var reservedPreview = reserveDefaultPreview
                ? Math.max(180, Math.min(420, Math.floor(height * 0.52)))
                : minStageHeight;
            var maxLowerTracks = Math.max(minPromptHeight + minLibraryHeight, height - 12 - reservedPreview);
            var overflow = promptHeight + libraryHeight - maxLowerTracks;
            if (overflow <= 0)
                return;
            var libraryReduction = Math.min(overflow, libraryHeight - minLibraryHeight);
            libraryHeight -= libraryReduction;
            overflow -= libraryReduction;
            if (overflow > 0)
                promptHeight = Math.max(minPromptHeight, promptHeight - overflow);
        }
        function apply(reserveDefaultPreview) {
            constrainVerticalTracks(Boolean(reserveDefaultPreview));
            els.layout.style.setProperty('--gen-left-width', leftWidth + 'px');
            els.layout.style.setProperty('--gen-right-width', rightWidth + 'px');
            els.layout.style.setProperty('--gen-prompt-height', promptHeight + 'px');
            els.layout.style.setProperty('--gen-library-height', libraryHeight + 'px');
            els.layout.classList.toggle('gen-prompt-compact', promptHeight < 132);
            els.layout.classList.remove('gen-prompt-hidden');
            els.layout.classList.toggle('gen-library-tabs-only', libraryHeight <= minLibraryHeight);
        }
        function drag(handle, axis, update, persist) {
            if (!handle)
                return;
            handle.addEventListener('pointerdown', function (event) {
                if (event.button !== 0)
                    return;
                event.preventDefault();
                var start = axis === 'x' ? event.clientX : event.clientY;
                var startValue = update();
                handle.classList.add('active');
                document.body.classList.add(axis === 'x' ? 'gen-resizing-x' : 'gen-resizing-y');
                function move(moveEvent) {
                    var current = axis === 'x' ? moveEvent.clientX : moveEvent.clientY;
                    update(startValue, current - start);
                    apply(false);
                }
                function done() {
                    handle.classList.remove('active');
                    document.body.classList.remove('gen-resizing-x', 'gen-resizing-y');
                    document.removeEventListener('pointermove', move);
                    document.removeEventListener('pointerup', done);
                    document.removeEventListener('pointercancel', done);
                    persist();
                }
                document.addEventListener('pointermove', move);
                document.addEventListener('pointerup', done);
                document.addEventListener('pointercancel', done);
            });
        }
        drag(els.leftResizer, 'x', function (start, delta) {
            if (start == null)
                return leftWidth;
            var max = Math.max(260, Math.min(560, els.layout.clientWidth - 620));
            leftWidth = Math.max(260, Math.min(max, start + delta));
        }, function () {
            localStorage.setItem('sf-gen-left-width', String(Math.round(leftWidth)));
        });
        drag(els.rightResizer, 'x', function (start, delta) {
            if (start == null)
                return rightWidth;
            rightWidth = Math.max(180, Math.min(430, start - delta));
        }, function () {
            localStorage.setItem('sf-gen-right-width', String(Math.round(rightWidth)));
        });
        drag(els.promptResizer, 'y', function (start, delta) {
            if (start == null)
                return { prompt: promptHeight, library: libraryHeight };
            var max = Math.max(
                minPromptHeight,
                els.layout.clientHeight - 12 - minStageHeight - start.library
            );
            if (delta >= 0) {
                promptHeight = Math.max(minPromptHeight, start.prompt - delta);
                var consumed = start.prompt - promptHeight;
                libraryHeight = Math.max(minLibraryHeight, start.library - Math.max(0, delta - consumed));
            }
            else {
                promptHeight = Math.max(minPromptHeight, Math.min(max, start.prompt - delta));
                libraryHeight = start.library;
            }
        }, function () {
            localStorage.setItem('sf-gen-prompt-height', String(Math.round(promptHeight)));
            localStorage.setItem('sf-gen-library-height', String(Math.round(libraryHeight)));
        });
        drag(els.libraryResizer, 'y', function (start, delta) {
            if (start == null)
                return libraryHeight;
            var max = Math.max(
                minLibraryHeight,
                els.layout.clientHeight - 12 - minStageHeight - promptHeight
            );
            libraryHeight = Math.max(minLibraryHeight, Math.min(max, start - delta));
        }, function () {
            localStorage.setItem('sf-gen-library-height', String(Math.round(libraryHeight)));
        });
        if (els.promptResizer) {
            els.promptResizer.addEventListener('dblclick', function () {
                promptHeight = promptHeight <= minPromptHeight ? 154 : minPromptHeight;
                apply(false);
                localStorage.setItem('sf-gen-prompt-height', String(Math.round(promptHeight)));
            });
        }
        if (els.libraryResizer) {
            els.libraryResizer.addEventListener('dblclick', function () {
                libraryHeight = libraryHeight <= minLibraryHeight ? 230 : minLibraryHeight;
                apply(false);
                localStorage.setItem('sf-gen-library-height', String(Math.round(libraryHeight)));
            });
        }
        ensureLibraryExpanded = function () {
            if (libraryHeight > minLibraryHeight)
                return;
            var max = Math.max(
                minLibraryHeight,
                els.layout.clientHeight - 12 - minStageHeight - promptHeight
            );
            libraryHeight = Math.max(minLibraryHeight, Math.min(230, max));
            apply(false);
            localStorage.setItem('sf-gen-library-height', String(Math.round(libraryHeight)));
        };
        window.addEventListener('resize', function () { apply(false); });
        apply(!hasStoredVerticalLayout);
        syncPanelLayoutState();
    }
    function bindEvents() {
        bindLayoutResizers();
        bindScail2Upload('gen-scail2-reference-image', 'referenceImagePath', 'reference image');
        bindScail2Upload('gen-scail2-reference-mask', 'referenceMaskPath', 'reference mask');
        bindScail2Upload('gen-scail2-driving-video', 'drivingVideoPath', 'driving video');
        bindScail2Upload('gen-scail2-driving-mask-video', 'drivingMaskVideoPath', 'driving mask video');
        bindScail2MultiUpload('gen-scail2-additional-images', 'additionalReferenceImagePaths', 'additional reference images');
        bindScail2MultiUpload('gen-scail2-additional-masks', 'additionalReferenceMaskPaths', 'additional reference masks');
        if (els.scail2Mode) {
            els.scail2Mode.addEventListener('change', function () {
                state.scail2Mode = this.value === 'replacement' ? 'replacement' : 'animation';
            });
        }
        // Auto-grow prompt textarea
        els.prompt.addEventListener('input', function () {
            state.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
            updateTokenCount();
            updateStylePreview();
        });
        els.negPrompt.addEventListener('input', function () {
            state.negPrompt = this.value;
        });
        // Image accordion
        els.imageHeader.addEventListener('click', function () {
            this.classList.toggle('closed');
            els.imageBody.classList.toggle('closed');
        });
        // Generation Settings accordion
        els.settingsHeader.addEventListener('click', function () {
            this.classList.toggle('closed');
            els.settingsBody.classList.toggle('closed');
        });
        // Section accordions
        bindAccordion('gen-compositing-header', 'gen-compositing-body');
        bindAccordion('gen-refiner-header', 'gen-refiner-body');
        bindAccordion('gen-advanced-header', 'gen-advanced-body');
        // ── Phase 3: Compositing controls ──
        bindSliderPair('gen-edge-size-range', 'gen-edge-size', function (v) { state.edgeSize = parseInt(v); });
        bindSliderPair('gen-min-denoise-range', 'gen-min-denoise', function (v) { state.minDenoise = parseFloat(v); });
        bindSliderPair('gen-mask-blur-range', 'gen-mask-blur', function (v) { state.maskBlur = parseInt(v); });
        var coherenceEl = document.getElementById('gen-coherence-mode');
        if (coherenceEl)
            coherenceEl.addEventListener('change', function () { state.coherenceMode = this.value; });
        var infillEl = document.getElementById('gen-infill-method');
        if (infillEl)
            infillEl.addEventListener('change', function () { state.infillMethod = this.value; });
        // ── Phase 3: Refiner controls ──
        bindSliderPair('gen-refiner-steps-range', 'gen-refiner-steps', function (v) { state.refinerSteps = parseInt(v); });
        bindSliderPair('gen-refiner-cfg-range', 'gen-refiner-cfg', function (v) { state.refinerCfg = parseFloat(v); });
        bindSliderPair('gen-refiner-start-range', 'gen-refiner-start', function (v) { state.refinerStart = parseFloat(v); });
        bindSliderPair('gen-aesthetic-pos-range', 'gen-aesthetic-pos', function (v) { state.aestheticScore = parseFloat(v); });
        bindSliderPair('gen-aesthetic-neg-range', 'gen-aesthetic-neg', function (v) { state.negAestheticScore = parseFloat(v); });
        var refSchedEl = document.getElementById('gen-refiner-scheduler');
        if (refSchedEl)
            refSchedEl.addEventListener('change', function () { state.refinerScheduler = this.value; });
        var refModelEl = document.getElementById('gen-refiner-model');
        if (refModelEl)
            refModelEl.addEventListener('change', function () { state.refinerModel = this.value; });
        // ── Phase 3: Advanced controls ──
        var vaeEl = document.getElementById('gen-vae-picker');
        if (vaeEl)
            vaeEl.addEventListener('change', function () { state.vae = this.value; });
        bindSliderPair('gen-clip-skip-range', 'gen-clip-skip', function (v) { state.clipSkip = parseInt(v); });
        bindSliderPair('gen-cfg-rescale-range', 'gen-cfg-rescale', function (v) { state.cfgRescale = parseFloat(v); });
        var seamXBtn = document.getElementById('gen-seamless-x');
        var seamYBtn = document.getElementById('gen-seamless-y');
        if (seamXBtn)
            seamXBtn.addEventListener('click', function () {
                state.seamlessX = !state.seamlessX;
                this.classList.toggle('on', state.seamlessX);
            });
        if (seamYBtn)
            seamYBtn.addEventListener('click', function () {
                state.seamlessY = !state.seamlessY;
                this.classList.toggle('on', state.seamlessY);
            });
        // Image Advanced Options disclosure
        var imgAdvDisc = document.getElementById('gen-image-adv-disclosure');
        var imgAdvBody = document.getElementById('gen-image-adv-body');
        if (imgAdvDisc && imgAdvBody) {
            imgAdvDisc.addEventListener('click', function () {
                this.classList.toggle('open');
                imgAdvBody.classList.toggle('open');
            });
        }
        // Generation Advanced Options disclosure
        var genAdvDisc = document.getElementById('gen-gen-adv-disclosure');
        var genAdvBody = document.getElementById('gen-gen-adv-body');
        if (genAdvDisc && genAdvBody) {
            genAdvDisc.addEventListener('click', function () {
                this.classList.toggle('open');
                genAdvBody.classList.toggle('open');
            });
        }
        // Steps sync
        els.steps.addEventListener('input', function () {
            state.steps = parseInt(this.value) || 20;
            els.stepsRange.value = this.value;
        });
        els.stepsRange.addEventListener('input', function () {
            state.steps = parseInt(this.value);
            els.steps.value = this.value;
        });
        // CFG sync
        els.cfg.addEventListener('input', function () {
            var parsed = parseFloat(this.value);
            state.cfg = Number.isFinite(parsed) ? parsed : 7.0;
            els.cfgRange.value = this.value;
        });
        els.cfgRange.addEventListener('input', function () {
            state.cfg = parseFloat(this.value);
            els.cfg.value = this.value;
        });
        // Scheduler
        if (els.sampler) {
            els.sampler.addEventListener('change', function () {
                state.sampler = this.value;
            });
        }
        els.scheduler.addEventListener('change', function () {
            state.scheduler = this.value;
        });
        bindSliderPair('gen-variation-strength-range', 'gen-variation-strength', function (v) {
            state.variationStrength = Math.max(0, Math.min(1, parseFloat(v) || 0));
        });
        var variationSeed = document.getElementById('gen-variation-seed');
        if (variationSeed) {
            variationSeed.addEventListener('input', function () {
                state.variationSeed = Math.max(0, parseInt(this.value) || 0);
            });
        }
        var variationShuffle = document.getElementById('gen-variation-shuffle');
        if (variationShuffle) {
            variationShuffle.addEventListener('click', function () {
                state.variationSeed = Math.floor(Math.random() * 4294967296);
                if (variationSeed)
                    variationSeed.value = String(state.variationSeed);
            });
        }
        bindSliderPair('gen-creativity-range', 'gen-creativity', function (v) {
            state.creativity = Math.max(0, Math.min(1, parseFloat(v) || 0));
        });
        bindInitImage();
        // Model searchable picker
        bindModelPicker();
        // Model refresh button
        var modelRefresh = document.getElementById('gen-model-refresh');
        if (modelRefresh) {
            modelRefresh.addEventListener('click', function (e) {
                e.stopPropagation();
                ModelUtils.clearCache();
                loadModels();
            });
        }
        // Guidance sync (FLUX)
        els.guidance.addEventListener('input', function () {
            state.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function () {
            state.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });
        // Aspect dropdown
        if (els.aspectDropdown) {
            els.aspectDropdown.addEventListener('change', function () {
                var val = this.value;
                if (val === 'custom' || val === 'Free')
                    return; // Free = keep current W/H
                var aspects = getActiveAspects();
                for (var i = 0; i < aspects.length; i++) {
                    if (aspects[i].label === val) {
                        state.width = aspects[i].w;
                        state.height = aspects[i].h;
                        if (ModelUtils.archForModel(state.model) === 'minimax_h3') {
                            syncDimensionInputs();
                            updateAspectPreview();
                        }
                        else if (ModelUtils.archForModel(state.model) === 'ltxv') {
                            var sizeProfiles = activeLtx2ProfilesForSize(
                                state.width, state.height
                            );
                            var nextProfile = sizeProfiles.find(function (profile) {
                                return Number(profile.frames) === Number(state.frames);
                            }) || sizeProfiles[0];
                            applyLtx2RequestProfile(nextProfile);
                        }
                        else {
                            syncDimensionInputs();
                            updateAspectPreview();
                        }
                        break;
                    }
                }
            });
        }
        // Swap button
        var swapBtn = document.getElementById('gen-swap-btn');
        if (swapBtn) {
            swapBtn.addEventListener('click', function () {
                var tmp = state.width;
                state.width = state.height;
                state.height = tmp;
                if (ModelUtils.archForModel(state.model) === 'ltxv') {
                    var swappedProfiles = activeLtx2ProfilesForSize(
                        state.width, state.height
                    );
                    var swappedProfile = swappedProfiles.find(function (profile) {
                        return Number(profile.frames) === Number(state.frames);
                    }) || swappedProfiles[0];
                    applyLtx2RequestProfile(swappedProfile);
                }
                else {
                    syncDimensionInputs();
                    syncAspectDropdown();
                    updateAspectPreview();
                }
            });
        }
        // Aspect lock
        var lockBtn = document.getElementById('gen-aspect-lock');
        if (lockBtn) {
            lockBtn.addEventListener('click', function () {
                state.aspectLocked = !state.aspectLocked;
                this.classList.toggle('active', state.aspectLocked);
                if (state.aspectLocked) {
                    state.lockedRatio = state.width / state.height;
                }
            });
        }
        // Optimal size button
        var optBtn = document.getElementById('gen-optimal-btn');
        if (optBtn) {
            optBtn.addEventListener('click', function () {
                var defaults = getDefaultsForArch(state.arch);
                state.width = defaults.w;
                state.height = defaults.h;
                syncDimensionInputs();
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        // Custom resolution inputs
        els.customWidth.addEventListener('blur', function () {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = clampAuthoredDimension(
                parseInt(this.value), 'width', isVideo ? 512 : 1024
            );
            this.value = String(v);
            state.width = v;
            var wsl = document.getElementById('gen-width-slider');
            if (wsl)
                wsl.value = String(v);
            if (state.aspectLocked && state.lockedRatio) {
                var newH = clampAuthoredDimension(
                    Math.round(v / state.lockedRatio), 'height', isVideo ? 512 : 1024
                );
                state.height = newH;
                if (els.customHeight)
                    els.customHeight.value = String(newH);
                var hsl = document.getElementById('gen-height-slider');
                if (hsl)
                    hsl.value = String(newH);
            }
            syncAspectDropdown();
            updateAspectPreview();
        });
        els.customHeight.addEventListener('blur', function () {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = clampAuthoredDimension(
                parseInt(this.value), 'height', isVideo ? 512 : 1024
            );
            this.value = String(v);
            state.height = v;
            var hsl = document.getElementById('gen-height-slider');
            if (hsl)
                hsl.value = String(v);
            if (state.aspectLocked && state.lockedRatio) {
                var newW = clampAuthoredDimension(
                    Math.round(v * state.lockedRatio), 'width', isVideo ? 512 : 1024
                );
                state.width = newW;
                if (els.customWidth)
                    els.customWidth.value = String(newW);
                var wsl = document.getElementById('gen-width-slider');
                if (wsl)
                    wsl.value = String(newW);
            }
            syncAspectDropdown();
            updateAspectPreview();
        });
        // Width slider sync
        var widthSlider = document.getElementById('gen-width-slider');
        if (widthSlider) {
            widthSlider.addEventListener('input', function () {
                var v = parseInt(this.value);
                state.width = v;
                if (els.customWidth)
                    els.customWidth.value = String(v);
                if (state.aspectLocked && state.lockedRatio) {
                    var isVideo = ModelUtils.isVideoModel(state.model);
                    var newH = clampAuthoredDimension(
                        Math.round(v / state.lockedRatio), 'height', isVideo ? 512 : 1024
                    );
                    state.height = newH;
                    if (els.customHeight)
                        els.customHeight.value = String(newH);
                    var hs = document.getElementById('gen-height-slider');
                    if (hs)
                        hs.value = String(newH);
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        var heightSlider = document.getElementById('gen-height-slider');
        if (heightSlider) {
            heightSlider.addEventListener('input', function () {
                var v = parseInt(this.value);
                state.height = v;
                if (els.customHeight)
                    els.customHeight.value = String(v);
                if (state.aspectLocked && state.lockedRatio) {
                    var isVideo = ModelUtils.isVideoModel(state.model);
                    var newW = clampAuthoredDimension(
                        Math.round(v * state.lockedRatio), 'width', isVideo ? 512 : 1024
                    );
                    state.width = newW;
                    if (els.customWidth)
                        els.customWidth.value = String(newW);
                    var ws = document.getElementById('gen-width-slider');
                    if (ws)
                        ws.value = String(newW);
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        // Seed
        els.seed.addEventListener('input', function () {
            state.seed = parseInt(this.value);
        });
        els.seedShuffle.addEventListener('click', function () {
            var s = Math.floor(Math.random() * 4294967296);
            state.seed = s;
            if (els.seed)
                els.seed.value = String(s);
        });
        // Seed previous
        var seedPrev = document.getElementById('gen-seed-prev');
        if (seedPrev) {
            seedPrev.addEventListener('click', function () {
                if (state.lastSeed !== null) {
                    state.seed = state.lastSeed;
                    if (els.seed)
                        els.seed.value = String(state.lastSeed);
                }
            });
        }
        // Seed random toggle
        var randomToggle = document.getElementById('gen-seed-random-toggle');
        if (randomToggle) {
            randomToggle.addEventListener('click', function () {
                var isRandom = state.seed !== -1;
                if (isRandom) {
                    state.seed = -1;
                    if (els.seed)
                        els.seed.value = String(-1);
                }
                else {
                    state.seed = Math.floor(Math.random() * 4294967296);
                    if (els.seed)
                        els.seed.value = String(state.seed);
                }
                this.classList.toggle('on', state.seed === -1);
            });
        }
        // Batch count (left panel)
        var batchInput = document.getElementById('gen-batch');
        if (batchInput) {
            batchInput.addEventListener('input', function () {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
            });
        }
        // Batch count (toolbar)
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.addEventListener('input', function () {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        var batchUp = document.getElementById('gen-toolbar-batch-up');
        var batchDown = document.getElementById('gen-toolbar-batch-down');
        if (batchUp) {
            batchUp.addEventListener('click', function () {
                state.batchCount = Math.min(8, state.batchCount + 1);
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        if (batchDown) {
            batchDown.addEventListener('click', function () {
                state.batchCount = Math.max(1, state.batchCount - 1);
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        // LoRA picker (concepts dropdown)
        var loraPicker = document.getElementById('gen-lora-picker');
        if (loraPicker) {
            loraPicker.addEventListener('change', function () {
                if (this.value && this.selectedIndex > 0) {
                    addLora(this.value);
                    this.selectedIndex = 0;
                }
            });
        }
        if (els.framesInput) {
            els.framesInput.addEventListener('input', function () {
                if (ModelUtils.archForModel(state.model) === 'ltxv') {
                    state.frames = Math.max(9, parseInt(this.value) || 9);
                    var frameProfile = exactLtx2RequestProfile();
                    state.seconds = frameProfile
                        ? Number(frameProfile.duration)
                        : Math.max(0.1, (state.frames - 1) / Math.max(1, state.fps));
                    if (els.secondsInput)
                        els.secondsInput.value = String(Number(state.seconds.toFixed(3)));
                    updateDurationHint();
                    return;
                }
                if (ModelUtils.archForModel(state.model) === 'minimax_h3') {
                    state.frames = Math.max(1, Math.min(1800, parseInt(this.value) || 1));
                    state.seconds = Math.max(1, Math.min(15,
                        state.frames / Math.max(1, state.fps)));
                    state.frames = Math.max(1, Math.round(state.seconds * state.fps));
                    this.value = String(state.frames);
                    if (els.secondsInput)
                        els.secondsInput.value = String(Number(state.seconds.toFixed(3)));
                    updateDurationHint();
                    return;
                }
                state.frames = Math.max(1, parseInt(this.value) || 1);
                state.seconds = state.frames / Math.max(1, state.fps);
                if (els.secondsInput)
                    els.secondsInput.value = String(Number(state.seconds.toFixed(3)));
                updateDurationHint();
            });
        }
        // Seconds → compute frames
        els.secondsInput.addEventListener('input', function () {
            if (ModelUtils.archForModel(state.model) === 'ltxv') {
                state.seconds = Math.max(
                    0.1, Math.min(120, parseFloat(this.value) || 0.1)
                );
                state.frames = ltx2SecondsToFrames(state.seconds, state.fps);
                if (els.framesInput)
                    els.framesInput.value = String(state.frames);
                updateDurationHint();
                return;
            }
            if (ModelUtils.archForModel(state.model) === 'minimax_h3') {
                state.seconds = Math.max(1, Math.min(15, parseFloat(this.value) || 1));
                state.frames = Math.max(1, Math.round(state.seconds * state.fps));
                if (els.framesInput)
                    els.framesInput.value = String(state.frames);
                updateDurationHint();
                return;
            }
            state.seconds = Math.max(0.1, Math.min(120, parseFloat(this.value) || 1));
            state.frames = secondsToFrames(state.seconds, state.fps);
            if (els.framesInput)
                els.framesInput.value = String(state.frames);
            updateDurationHint();
        });
        // FPS sync — recompute frames from seconds
        els.fpsInput.addEventListener('input', function () {
            var activeArch = ModelUtils.archForModel(state.model);
            var fpsMax = activeArch === 'minimax_h3' ? 120 : 60;
            state.fps = Math.max(1, Math.min(fpsMax, parseInt(this.value) || 1));
            els.fpsRange.value = String(state.fps);
            state.frames = activeArch === 'minimax_h3'
                ? Math.max(1, Math.round(state.seconds * state.fps))
                : activeArch === 'ltxv'
                ? ltx2SecondsToFrames(state.seconds, state.fps)
                : secondsToFrames(state.seconds, state.fps);
            if (els.framesInput)
                els.framesInput.value = String(state.frames);
            updateDurationHint();
        });
        els.fpsRange.addEventListener('input', function () {
            state.fps = parseInt(this.value);
            els.fpsInput.value = this.value;
            var activeArch = ModelUtils.archForModel(state.model);
            state.frames = activeArch === 'minimax_h3'
                ? Math.max(1, Math.round(state.seconds * state.fps))
                : activeArch === 'ltxv'
                ? ltx2SecondsToFrames(state.seconds, state.fps)
                : secondsToFrames(state.seconds, state.fps);
            if (els.framesInput)
                els.framesInput.value = String(state.frames);
            updateDurationHint();
        });
        if (els.videoGuidanceMode) {
            els.videoGuidanceMode.addEventListener('change', function () {
                state.videoGuidanceMode = this.value === 'dev' ? 'dev' : 'distilled';
                var creatorWorkflow = ltx2CheckpointWorkflow(state.videoCheckpoint);
                state.videoWorkflowProfile =
                    state.videoGuidanceMode === 'distilled' && creatorWorkflow
                        ? String(creatorWorkflow.id || '') : '';
                applyVideoGuidanceMode();
            });
        }
        if (els.videoQuant)
            els.videoQuant.addEventListener('change', function () {
                state.videoQuant = this.value;
                if (ModelUtils.archForModel(state.model) === 'minimax_h3') {
                    state.h3Quant = state.videoQuant;
                    if (state.videoQuant === 'int8-fast') {
                        state.h3AttentionBackend = 'cudnn';
                        if (els.h3Attention)
                            els.h3Attention.value = 'cudnn';
                    }
                    if (els.h3Attention)
                        els.h3Attention.disabled = state.videoQuant === 'int8-fast';
                    return;
                }
                var selectedIsLtx = ModelUtils.archForModel(state.model) === 'ltxv';
                var checkpoint = selectedIsLtx
                    ? String(state.model).replace(/\.safetensors$/i, '')
                    : (state.videoQuant === 'bf16'
                        ? 'ltx-2.3-22b-dev-fp8-dequant-bf16'
                        : 'ltx-2.3-22b-dev-fp8');
                state.videoCheckpoint = checkpoint;
                if (els.videoCheckpoint)
                    els.videoCheckpoint.value = checkpoint;
                if (!selectedIsLtx && (state.allModels || []).some(function (model) {
                    return model.name.replace(/\.safetensors$/i, '') === checkpoint;
                }))
                    selectModel(checkpoint);
            });
        if (els.h3Attention)
            els.h3Attention.addEventListener('change', function () {
                if (state.videoQuant === 'int8-fast') {
                    state.h3AttentionBackend = 'cudnn';
                    this.value = 'cudnn';
                    return;
                }
                state.h3AttentionBackend = this.value === 'sage-int8'
                    ? 'sage-int8' : 'cudnn';
            });
        if (els.h3StepCache)
            els.h3StepCache.addEventListener('change', function () {
                state.h3StepCache = this.value === 'high' ? 'high' : 'exact';
            });
        if (els.videoCheckpoint)
            els.videoCheckpoint.addEventListener('change', function () {
                state.videoCheckpoint = this.value;
                refreshLtx2CheckpointControls();
                var creatorWorkflow = ltx2CheckpointWorkflow(
                    state.videoCheckpoint
                );
                state.videoWorkflowProfile = creatorWorkflow
                    ? String(creatorWorkflow.id || '') : '';
                if (creatorWorkflow)
                    state.videoGuidanceMode = String(
                        creatorWorkflow.guidance_mode || 'distilled'
                    );
                applyVideoGuidanceMode();
                refreshVideoPromptEnhancer();
            });
        if (els.videoPromptEnhancer)
            els.videoPromptEnhancer.addEventListener('change', function () {
                state.videoPromptEnhancer = this.value || 'none';
            });
        if (els.capsPositive)
            els.capsPositive.addEventListener('input', function () { state.capsPositive = this.value; });
        if (els.capsNegative)
            els.capsNegative.addEventListener('input', function () { state.capsNegative = this.value; });
        if (els.noiseFixture)
            els.noiseFixture.addEventListener('input', function () { state.noiseFixture = this.value; });
        if (els.audioPolicy)
            els.audioPolicy.addEventListener('change', function () {
                state.audioPolicy = this.value;
                state.includeAudio = this.value === 'generate';
            });
        if (els.postUpscaler) {
            els.postUpscaler.addEventListener('change', function () {
                state.postUpscaler = this.value || 'none';
                refreshLtx2PostUpscaleControls();
            });
        }
        if (els.cameraMotion)
            els.cameraMotion.addEventListener('change', function () {
                state.cameraMotion = this.value || 'none';
            });
        if (els.postUpscaleFactor) {
            els.postUpscaleFactor.addEventListener('change', function () {
                state.postUpscaleFactor = Number(this.value) === 4 ? 4 : 2;
                refreshLtx2PostUpscaleControls();
            });
        }
        if (els.sigmaShift) {
            els.sigmaShift.addEventListener('input', function () {
                var value = Number(this.value);
                if (!Number.isFinite(value))
                    return;
                state.sigmaShift = value;
                if (els.sigmaShiftRange)
                    els.sigmaShiftRange.value = String(Math.min(Number(els.sigmaShiftRange.max), value));
            });
        }
        if (els.sigmaShiftRange) {
            els.sigmaShiftRange.addEventListener('input', function () {
                state.sigmaShift = Number(this.value);
                if (els.sigmaShift)
                    els.sigmaShift.value = this.value;
            });
        }
        if (els.noSeedIncrement)
            els.noSeedIncrement.addEventListener('change', function () { state.noSeedIncrement = this.checked; });
        if (els.continueAfterErrors)
            els.continueAfterErrors.addEventListener('change', function () { state.continueAfterErrors = this.checked; });
        if (els.personalNote)
            els.personalNote.addEventListener('input', function () { state.personalNote = this.value; });
        if (els.showAdvanced) {
            els.showAdvanced.addEventListener('change', function () {
                state.showAdvanced = this.checked;
                updateAdvancedVisibility();
            });
        }
        // Generate (left panel button)
        els.btn.addEventListener('click', function () {
            generate();
        });
        // Action bar
        els.download.addEventListener('click', function () {
            if (!state.currentImage)
                return;
            var a = document.createElement('a');
            a.href = state.currentImage;
            var ext = state.currentIsVideo ? '.mp4' : '.png';
            a.download = 'serenityflow_' + Date.now() + ext;
            a.click();
        });
        els.clearPreview.addEventListener('click', function () {
            clearPreview();
        });
        if (els.reuseParams) {
            els.reuseParams.addEventListener('click', function () {
                var params = state.currentResultParams;
                if (!params && state.currentGalleryIndex >= 0 && state.gallery[state.currentGalleryIndex])
                    params = reusableParamsForItem(state.gallery[state.currentGalleryIndex]);
                if (!params) {
                    showError('This result has no saved parameters to reuse');
                    return;
                }
                applyParams(params);
            });
        }
        // Send to Timeline
        var toTimelineBtn = document.getElementById('gen-to-timeline');
        if (toTimelineBtn) {
            toTimelineBtn.addEventListener('click', function () {
                if (!state.currentImage) return;
                if (typeof VideoEditTab === 'undefined') {
                    alert('Open the Video Edit tab first.');
                    return;
                }
                var pid = VideoEditTab.getActiveProjectId();
                if (!pid) {
                    alert('Open the Video Edit tab first to create a project.');
                    return;
                }
                var label = 'Generated';
                try {
                    var promptEl = document.getElementById('gen-prompt');
                    if (promptEl && promptEl.value) {
                        label = promptEl.value.slice(0, 30).trim() || 'Generated';
                    }
                } catch (e) {}
                var timelineFps = Math.max(
                    1,
                    Math.min(120, Math.round(Number(state.currentVideoFps || state.fps) || 30))
                );
                var durationFrames = state.currentIsVideo
                    ? Math.max(1, Math.round(Number(state.currentVideoFrames) || Number(state.frames) || timelineFps))
                    : timelineFps * 3;

                // Resolve /view URL to a real filesystem path before passing to timeline
                var viewUrl = state.currentImage;
                var params = {};
                try {
                    var urlObj = new URL(viewUrl, window.location.origin);
                    urlObj.searchParams.forEach(function (v, k) { params[k] = v; });
                } catch (e) {
                    // Not a URL — pass as-is (may be a direct path)
                    VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                    if (typeof switchTab === 'function') switchTab('video-edit');
                    return;
                }

                if (params.filename) {
                    fetch('/video_edit/resolve_view_path', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename: params.filename,
                            subfolder: params.subfolder || '',
                            type: params.type || 'output',
                        }),
                    })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        if (data.path) {
                            VideoEditTab.addClipFromExternal(data.path, label, durationFrames, timelineFps);
                        } else {
                            // Fallback: use the URL as-is
                            VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                        }
                        if (typeof switchTab === 'function') switchTab('video-edit');
                    })
                    .catch(function () {
                        VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                        if (typeof switchTab === 'function') switchTab('video-edit');
                    });
                } else {
                    VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                    if (typeof switchTab === 'function') switchTab('video-edit');
                }
            });
        }
        // Gallery clear
        els.galleryClear.addEventListener('click', function () {
            if (state.gallery.length === 0)
                return;
            if (!confirm('Permanently delete all ' + state.gallery.length + ' history items?'))
                return;
            var doomed = state.gallery.slice();
            Promise.all(doomed.map(deleteHistoryArtifact)).then(function () {
                state.gallery = [];
                state.selectedImages = [];
                renderGallery();
                updateSelectionUI();
                updateMetadataPanel();
                clearPreview();
            }).catch(function (error) {
                showError('Could not delete every output: ' + error.message);
                refreshHistory();
            });
        });
        // ── Top Toolbar ──
        var toolbarGenerate = document.getElementById('gen-toolbar-generate');
        if (toolbarGenerate) {
            toolbarGenerate.addEventListener('click', function () { generate(); });
        }
        var toolbarDelete = document.getElementById('gen-toolbar-delete');
        if (toolbarDelete) {
            toolbarDelete.addEventListener('click', function () { clearPreview(); });
        }
        var toolbarToggleGallery = document.getElementById('gen-toolbar-toggle-gallery');
        if (toolbarToggleGallery) {
            toolbarToggleGallery.addEventListener('click', function () {
                state.rightPanelVisible = !state.rightPanelVisible;
                els.rightPanel.classList.toggle('gen-panel-hidden', !state.rightPanelVisible);
                this.classList.toggle('active', state.rightPanelVisible);
                syncPanelLayoutState();
            });
        }
        var toolbarCopy = document.getElementById('gen-toolbar-copy');
        if (toolbarCopy) {
            toolbarCopy.addEventListener('click', function () {
                if (state.currentImage) {
                    var fullUrl = window.location.origin + state.currentImage;
                    navigator.clipboard.writeText(fullUrl).catch(function () { });
                }
            });
        }
        // ── Floating Toolbar ──
        var floatToggleLeft = document.getElementById('gen-float-toggle-left');
        if (floatToggleLeft) {
            floatToggleLeft.addEventListener('click', function () {
                state.leftPanelVisible = !state.leftPanelVisible;
                els.leftPanel.classList.toggle('gen-panel-hidden', !state.leftPanelVisible);
                this.classList.toggle('active', state.leftPanelVisible);
                syncPanelLayoutState();
            });
        }
        var floatGenerate = document.getElementById('gen-float-generate');
        if (floatGenerate) {
            floatGenerate.addEventListener('click', function () { generate(); });
        }
        var floatCancel = document.getElementById('gen-float-cancel');
        if (floatCancel) {
            floatCancel.addEventListener('click', function () {
                if (state.generating) {
                    state.pendingBatch = 0;
                    state.videoPollToken++;
                    state.pendingVideoJobs = {};
                    state.pendingImageJobs = {};
                    SerenityAPI.interrupt();
                    setGenerating(false);
                }
            });
        }
        var floatDelete = document.getElementById('gen-float-delete');
        if (floatDelete) {
            floatDelete.addEventListener('click', function () { clearPreview(); });
        }
        // ── Gallery Tabs ──
        var tabLayers = document.getElementById('gen-tab-layers');
        var tabGallery = document.getElementById('gen-tab-gallery');
        var layersContent = document.getElementById('gen-layers-content');
        var galleryContent = document.getElementById('gen-gallery-content');
        if (tabLayers && tabGallery) {
            tabLayers.addEventListener('click', function () {
                state.galleryTab = 'layers';
                tabLayers.classList.add('active');
                tabGallery.classList.remove('active');
                layersContent.classList.add('active');
                galleryContent.classList.remove('active');
            });
            tabGallery.addEventListener('click', function () {
                state.galleryTab = 'gallery';
                tabGallery.classList.add('active');
                tabLayers.classList.remove('active');
                galleryContent.classList.add('active');
                layersContent.classList.remove('active');
            });
        }
        // Gallery close
        var galleryClose = document.getElementById('gen-gallery-close');
        if (galleryClose) {
            galleryClose.addEventListener('click', function () {
                state.rightPanelVisible = false;
                els.rightPanel.classList.add('gen-panel-hidden');
                syncPanelLayoutState();
                var galleryToggle = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle)
                    galleryToggle.classList.remove('active');
            });
        }
        // Boards toggle
        var boardsHeader = document.getElementById('gen-boards-header');
        var boardsBody = document.getElementById('gen-boards-body');
        if (boardsHeader && boardsBody) {
            boardsHeader.addEventListener('click', function () {
                state.boardsVisible = !state.boardsVisible;
                boardsBody.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.querySelector('span:first-child').textContent = state.boardsVisible ? 'Hide Boards' : 'Show Boards';
            });
        }
        // Sub-tabs
        var subtabImages = document.getElementById('gen-subtab-images');
        var subtabAssets = document.getElementById('gen-subtab-assets');
        var imagesContent = document.getElementById('gen-images-content');
        var assetsContent = document.getElementById('gen-assets-content');
        if (subtabImages && subtabAssets) {
            subtabImages.addEventListener('click', function () {
                setGallerySubTab('images');
            });
            subtabAssets.addEventListener('click', function () {
                setGallerySubTab('assets');
            });
        }
        var assetsRefresh = document.getElementById('gen-assets-refresh');
        if (assetsRefresh)
            assetsRefresh.addEventListener('click', function () {
                loadAssets(true);
            });
        // Gallery search toggle + filtering
        var searchBtn = document.getElementById('gen-gallery-search-btn');
        var searchInput = document.getElementById('gen-gallery-search-input');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', function () {
                searchInput.classList.toggle('open');
                if (searchInput.classList.contains('open')) {
                    searchInput.focus();
                }
                else {
                    searchInput.value = '';
                    state.gallerySearch = '';
                    if (state.gallerySubTab === 'assets')
                        renderAssets();
                    else
                        renderGallery();
                }
            });
            searchInput.addEventListener('input', function () {
                state.gallerySearch = this.value;
                if (state.gallerySubTab === 'assets')
                    renderAssets();
                else
                    renderGallery();
            });
        }
        // ── Style Preset ──
        var styleSelect = document.getElementById('gen-style-preset');
        if (styleSelect) {
            styleSelect.addEventListener('change', function () {
                state.stylePreset = this.value;
                updateStylePreview();
            });
        }
        // ── Prompt History (Alt+Up/Down) ──
        if (els.prompt) {
            els.prompt.addEventListener('keydown', function (e) {
                // Prompt attention weight (Ctrl+Up/Down)
                if (e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    handlePromptWeight(this, e.key === 'ArrowUp');
                    return;
                }
                // Prompt history (Alt+Up/Down)
                if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    if (e.key === 'ArrowUp') {
                        navigatePromptHistory(-1);
                    }
                    else {
                        navigatePromptHistory(1);
                    }
                }
            });
        }
        bindGenerateControls();
    }
    function bindInitImage() {
        var input = document.getElementById('gen-init-image-input');
        var drop = document.getElementById('gen-init-drop');
        var clear = document.getElementById('gen-init-clear');
        if (!input || !drop)
            return;
        function upload(file) {
            if (!file || !/^image\//.test(file.type || '')) {
                showError('Choose a readable image file');
                return;
            }
            var form = new FormData();
            form.append('image', file, file.name || 'init.png');
            var preview = document.getElementById('gen-init-preview');
            var empty = document.getElementById('gen-init-empty');
            var name = document.getElementById('gen-init-name');
            if (preview) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
            if (empty)
                empty.style.display = 'none';
            if (name)
                name.textContent = 'Uploading ' + (file.name || 'image') + '…';
            if (typeof createImageBitmap === 'function') {
                createImageBitmap(file).then(function (bitmap) {
                    state.initImageWidth = bitmap.width;
                    state.initImageHeight = bitmap.height;
                    if (typeof bitmap.close === 'function')
                        bitmap.close();
                    syncWanI2vSteps();
                }).catch(function () {
                    state.initImageWidth = 0;
                    state.initImageHeight = 0;
                });
            }
            fetch('/upload/image', { method: 'POST', body: form })
                .then(function (resp) {
                return resp.text().then(function (body) {
                    var data = {};
                    try { data = JSON.parse(body); }
                    catch (e) { data = { error: body }; }
                    if (!resp.ok)
                        throw new Error(data.error || data.detail || ('HTTP ' + resp.status));
                    return data;
                });
            })
                .then(function (data) {
                state.initImagePath = data.path || data.name || '';
                state.initImageName = data.name || file.name || state.initImagePath;
                if (!state.initImagePath)
                    throw new Error('upload returned no image path');
                syncWanI2vSteps();
                if (name)
                    name.textContent = state.initImageName;
                if (clear)
                    clear.disabled = false;
            })
                .catch(function (err) {
                clearInitImage();
                showError('Init image upload failed: ' + err.message);
            });
        }
        input.addEventListener('change', function () {
            upload(this.files && this.files[0]);
        });
        ['dragenter', 'dragover'].forEach(function (eventName) {
            drop.addEventListener(eventName, function (event) {
                event.preventDefault();
                drop.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(function (eventName) {
            drop.addEventListener(eventName, function (event) {
                event.preventDefault();
                drop.classList.remove('dragover');
            });
        });
        drop.addEventListener('drop', function (event) {
            upload(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
        });
        if (clear)
            clear.addEventListener('click', clearInitImage);
    }
    function clearInitImage() {
        state.initImagePath = '';
        state.initImageName = '';
        state.initImageWidth = 0;
        state.initImageHeight = 0;
        syncWanI2vSteps();
        var input = document.getElementById('gen-init-image-input');
        var preview = document.getElementById('gen-init-preview');
        var empty = document.getElementById('gen-init-empty');
        var name = document.getElementById('gen-init-name');
        var clear = document.getElementById('gen-init-clear');
        if (input)
            input.value = '';
        if (preview) {
            if (preview.src && preview.src.indexOf('blob:') === 0)
                URL.revokeObjectURL(preview.src);
            preview.removeAttribute('src');
            preview.style.display = 'none';
        }
        if (empty)
            empty.style.display = '';
        if (name)
            name.textContent = 'No source image';
        if (clear)
            clear.disabled = true;
    }
    function bindGenerateControls() {
        ['gen-core', 'gen-variation', 'gen-sampling', 'gen-video',
            'gen-source', 'gen-video-conditioning', 'gen-lora', 'gen-advanced-runtime',
            'gen-output'].forEach(function (prefix) {
            bindAccordion(prefix + '-header', prefix + '-body');
        });
        var filter = document.getElementById('gen-param-filter');
        if (filter) {
            filter.addEventListener('input', function () {
                var query = this.value.trim().toLowerCase();
                document.querySelectorAll('#gen-left-panel .gen-param-group').forEach(function (group) {
                    var haystack = group.getAttribute('data-param-search') || '';
                    var rows = Array.prototype.slice.call(group.querySelectorAll('.gen-param-row'));
                    var groupMatch = !query || haystack.indexOf(query) >= 0;
                    var rowMatch = false;
                    rows.forEach(function (row) {
                        var visible = !query || groupMatch ||
                            (row.getAttribute('data-param-search') || '').indexOf(query) >= 0 ||
                            row.textContent.toLowerCase().indexOf(query) >= 0;
                        row.style.display = visible ? '' : 'none';
                        rowMatch = rowMatch || visible;
                    });
                    group.style.display = (!query || groupMatch || rowMatch) ? '' : 'none';
                    if (query && (groupMatch || rowMatch)) {
                        var body = group.querySelector('.gen-workspace-group-body');
                        var header = group.querySelector('.gen-workspace-group-title');
                        if (body) body.classList.remove('closed');
                        if (header) header.classList.remove('closed');
                    }
                });
            });
        }
        var reset = document.getElementById('gen-quick-reset');
        if (reset) {
            reset.addEventListener('click', function () {
                state.variationSeed = 0;
                state.variationStrength = 0;
                clearInitImage();
                updateUIForArch(state.arch);
                var variationSeed = document.getElementById('gen-variation-seed');
                var variationStrength = document.getElementById('gen-variation-strength');
                var variationRange = document.getElementById('gen-variation-strength-range');
                if (variationSeed) variationSeed.value = '0';
                if (variationStrength) variationStrength.value = '0';
                if (variationRange) variationRange.value = '0';
            });
        }
        document.querySelectorAll('.gen-library-tab').forEach(function (button) {
            button.addEventListener('click', function () {
                activateLibraryPanel(this.dataset.library);
            });
        });
        var presetSave = document.getElementById('gen-preset-save');
        if (presetSave) {
            presetSave.addEventListener('click', function () {
                var input = document.getElementById('gen-preset-name');
                var name = input ? input.value.trim() : '';
                if (!name) {
                    showError('Enter a preset name');
                    return;
                }
                fetch('/v1/presets', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, params: getParams() })
                }).then(function (resp) {
                    if (!resp.ok)
                        return resp.text().then(function (text) { throw new Error(text || ('HTTP ' + resp.status)); });
                    return resp.json();
                }).then(function () {
                    if (input) input.value = '';
                    loadGenerationPresets();
                }).catch(function (err) {
                    showError('Preset save failed: ' + err.message);
                });
            });
        }
    }
    function loadGenerationPresets() {
        fetch('/v1/presets', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (doc) {
            var list = document.getElementById('gen-preset-list');
            if (!list)
                return;
            list.innerHTML = '';
            var presets = Array.isArray(doc.presets) ? doc.presets : [];
            if (!presets.length) {
                list.innerHTML = '<div class="gen-library-empty">No saved presets</div>';
                return;
            }
            presets.forEach(function (preset) {
                var card = document.createElement('div');
                card.className = 'gen-library-card';
                var title = document.createElement('strong');
                title.textContent = preset.name || 'Unnamed';
                var load = document.createElement('button');
                load.className = 'gen-small-btn';
                load.textContent = 'Load';
                load.addEventListener('click', function () { applyParams(preset.params || {}); });
                var del = document.createElement('button');
                del.className = 'gen-small-btn destructive';
                del.textContent = 'Delete';
                del.addEventListener('click', function () {
                    fetch('/v1/presets/' + encodeURIComponent(preset.name), { method: 'DELETE' })
                        .then(function (resp) {
                        if (!resp.ok)
                            throw new Error('HTTP ' + resp.status);
                        loadGenerationPresets();
                    }).catch(function (err) { showError('Preset delete failed: ' + err.message); });
                });
                card.appendChild(title);
                card.appendChild(load);
                card.appendChild(del);
                list.appendChild(card);
            });
        })
            .catch(function (err) {
            showError('Preset load failed: ' + err.message);
        });
    }
    function workflowGraphStats(graph, objectInfo) {
        var definitions = {};
        var subgraphs = graph && graph.definitions && Array.isArray(graph.definitions.subgraphs)
            ? graph.definitions.subgraphs : [];
        subgraphs.forEach(function (subgraph) {
            if (subgraph && subgraph.id)
                definitions[subgraph.id] = subgraph;
        });
        var skipped = new Set(['MarkdownNote', 'Note']);
        var topLevelNodes = graph && Array.isArray(graph.nodes) ? graph.nodes : [];
        var typesSeen = new Set();
        function collectTypes(nodes, visiting) {
            (nodes || []).forEach(function (node) {
                if (!node || node.id < 0 || !node.type || skipped.has(node.type))
                    return;
                var subgraph = definitions[node.type];
                if (subgraph && !visiting.has(node.type)) {
                    var next = new Set(visiting);
                    next.add(node.type);
                    collectTypes(subgraph.nodes, next);
                    return;
                }
                typesSeen.add(String(node.type));
            });
        }
        collectTypes(topLevelNodes, new Set());
        var types = Array.from(typesSeen).sort(function (a, b) {
            return a.localeCompare(b);
        });
        var missing = types.filter(function (type) {
            return !objectInfo || !Object.prototype.hasOwnProperty.call(objectInfo, type);
        });
        return { nodeCount: topLevelNodes.length, nodeTypes: types, missing: missing };
    }
    function openGenerationWorkflow(template) {
        if (!template) {
            showError('Workflow template is missing');
            return;
        }
        if (typeof switchTab !== 'function') {
            showError('Canvas navigation is unavailable');
            return;
        }
        if (template.preset) {
            applyParams(template.preset);
            switchTab('workflows');
            var presetNameInput = document.getElementById('workflow-name');
            if (presetNameInput)
                presetNameInput.value = template.name || 'Untitled Workflow';
            return;
        }
        if (!template.url) {
            showError('Workflow template URL is missing');
            return;
        }
        switchTab('workflows');
        var nameInput = document.getElementById('workflow-name');
        if (nameInput)
            nameInput.value = template.name || 'Untitled Workflow';
        requestAnimationFrame(function () {
            if (typeof sfToolbar !== 'undefined' && sfToolbar) {
                sfToolbar.loadWorkflowFromUrl(template.url);
                return;
            }
            showError('Canvas is still initializing; try opening the workflow again');
        });
    }
    function loadGenerationWorkflows() {
        var list = document.getElementById('gen-workflow-list');
        if (!list)
            return;
        list.innerHTML = '<div class="gen-library-empty">Inspecting workflow compatibility…</div>';
        Promise.all([
            fetch('/templates', { cache: 'no-store' }).then(function (resp) {
                if (!resp.ok)
                    throw new Error('templates HTTP ' + resp.status);
                return resp.json();
            }),
            ModelUtils.loadObjectInfo()
        ]).then(function (values) {
            var templates = Array.isArray(values[0]) ? values[0] : [];
            var objectInfo = values[1] || {};
            var availableModels = new Set((state.allModels || []).map(function (model) {
                return model.name;
            }));
            var builtins = (typeof SerenityWorkflowPresets !== 'undefined'
                ? SerenityWorkflowPresets : []).filter(function (template) {
                return !template.preset || availableModels.has(template.preset.model);
            });
            var merged = new Map();
            builtins.concat(templates).forEach(function (template) {
                merged.set(template.name || template.file || template.url, template);
            });
            templates = Array.from(merged.values());
            return Promise.all(templates.map(function (template) {
                if (template.preset) {
                    return Promise.resolve({
                        template: template,
                        stats: { nodeCount: 0, nodeTypes: [], missing: [] },
                        error: '',
                        builtin: true
                    });
                }
                return fetch(template.url, { cache: 'no-store' })
                    .then(function (resp) {
                    if (!resp.ok)
                        throw new Error('HTTP ' + resp.status);
                    return resp.json();
                }).then(function (graph) {
                    return {
                        template: template,
                        stats: workflowGraphStats(graph, objectInfo),
                        error: ''
                    };
                }).catch(function (error) {
                    return { template: template, stats: null, error: error.message };
                });
            }));
        }).then(function (entries) {
            list.innerHTML = '';
            entries.forEach(function (entry) {
                var template = entry.template;
                var card = document.createElement('div');
                card.className = 'gen-library-card gen-workflow-card';
                var copy = document.createElement('div');
                copy.className = 'gen-workflow-card-copy';
                var title = document.createElement('strong');
                title.textContent = template.name || template.file || 'Workflow';
                copy.appendChild(title);
                var status = document.createElement('span');
                status.className = 'gen-workflow-card-status';
                if (entry.error) {
                    status.classList.add('is-missing');
                    status.textContent = 'Could not inspect · ' + entry.error;
                }
                else if (entry.builtin) {
                    status.classList.add('is-ready');
                    status.textContent = 'Built-in product workflow · current runtime';
                }
                else if (entry.stats.missing.length) {
                    status.classList.add('is-missing');
                    status.textContent = 'Reference workflow · ' + entry.stats.nodeCount +
                        ' nodes · ' + entry.stats.missing.length + ' custom node type' +
                        (entry.stats.missing.length === 1 ? '' : 's') + ' missing';
                    status.title = 'Missing: ' + entry.stats.missing.join(', ');
                }
                else {
                    status.classList.add('is-ready');
                    status.textContent = entry.stats.nodeCount +
                        ' nodes · all node types registered';
                }
                copy.appendChild(status);
                var open = document.createElement('button');
                open.type = 'button';
                open.className = 'gen-small-btn';
                open.textContent = 'Open in Canvas';
                open.addEventListener('click', function () {
                    openGenerationWorkflow(template);
                });
                card.appendChild(copy);
                card.appendChild(open);
                list.appendChild(card);
            });
            if (!list.children.length)
                list.innerHTML = '<div class="gen-library-empty">No workflow templates found</div>';
        }).catch(function (error) {
            list.innerHTML = '<div class="gen-library-empty">Workflow catalog failed: ' +
                escapeHtml(error.message) + '</div>';
        });
    }
    function renderModelLibrary() {
        var list = document.getElementById('gen-library-models');
        if (!list)
            return;
        list.innerHTML = '';
        (state.allModels || []).forEach(function (model) {
            var card = document.createElement('button');
            card.className = 'gen-library-card gen-library-model-card';
            card.type = 'button';
            card.textContent = model.name;
            card.title = model.name;
            card.classList.toggle('active', model.name === state.model);
            card.addEventListener('click', function () { selectModel(model.name); renderModelLibrary(); });
            list.appendChild(card);
        });
        if (!list.children.length)
            list.innerHTML = '<div class="gen-library-empty">No admitted image models found</div>';
    }
    function renderLoraLibrary() {
        var list = document.getElementById('gen-library-loras');
        var picker = document.getElementById('gen-lora-picker');
        if (!list)
            return;
        list.innerHTML = '';
        if (picker) {
            Array.prototype.slice.call(picker.options).forEach(function (option, index) {
                if (index === 0 || !option.value)
                    return;
                var card = document.createElement('button');
                card.className = 'gen-library-card gen-library-model-card';
                card.type = 'button';
                card.textContent = option.textContent;
                card.addEventListener('click', function () { addLora(option.value); });
                list.appendChild(card);
            });
        }
        if (!list.children.length)
            list.innerHTML = '<div class="gen-library-empty">No local LoRAs found</div>';
    }
    function bindAccordion(headerId, bodyId) {
        var header = document.getElementById(headerId);
        var body = document.getElementById(bodyId);
        if (header && body) {
            header.addEventListener('click', function () {
                this.classList.toggle('closed');
                body.classList.toggle('closed');
            });
        }
    }
    // Helper: bind a range slider ↔ number input pair with state callback
    function bindSliderPair(rangeId, inputId, onUpdate) {
        var range = document.getElementById(rangeId);
        var input = document.getElementById(inputId);
        if (range && input) {
            range.addEventListener('input', function () {
                input.value = this.value;
                if (onUpdate)
                    onUpdate(this.value);
            });
            input.addEventListener('input', function () {
                var v = parseFloat(this.value);
                if (!isNaN(v)) {
                    var min = parseFloat(range.min);
                    var max = parseFloat(range.max);
                    range.value = String(Math.min(max, Math.max(min, v)));
                    if (onUpdate)
                        onUpdate(this.value);
                }
            });
        }
    }
    function syncDimensionInputs() {
        if (els.customWidth)
            els.customWidth.value = String(state.width);
        if (els.customHeight)
            els.customHeight.value = String(state.height);
        var ws = document.getElementById('gen-width-slider');
        var hs = document.getElementById('gen-height-slider');
        if (ws)
            ws.value = String(state.width);
        if (hs)
            hs.value = String(state.height);
    }
    function getDefaultsForArch(arch) {
        var defaults = {
            sd15: { w: 512, h: 512 },
            sdxl: { w: 1024, h: 1024 },
            sd3: { w: 1024, h: 1024 },
            flux: { w: 1024, h: 1024 },
            klein: { w: 1024, h: 1024 },
            krea2: { w: 1024, h: 1024 },
            qwen: { w: 1024, h: 1024 },
            ltxv: { w: 1920, h: 1088 },
            minimax_h3: { w: 1344, h: 768 },
            wan: { w: 1280, h: 704 },
            bernini: { w: 848, h: 480 },
            scail2: { w: 896, h: 512 }
        };
        return defaults[arch] || { w: 1024, h: 1024 };
    }
    function activeCapabilityProfile(arch) {
        var resolvedArch = arch || state.arch;
        var backend = ModelUtils.backendForArch(resolvedArch);
        if (!backend || !state.capabilities || !Array.isArray(state.capabilities.backends))
            return null;
        return state.capabilities.backends.find(function (entry) {
            return entry && entry.backend === backend;
        }) || null;
    }
    function featureSupported(profile, name) {
        return !!(profile && profile.features && profile.features[name] &&
            profile.features[name].supported === true);
    }
    function advancedParameterCapability(profile, name) {
        var advanced = profile && profile.features && profile.features.advanced_sampling;
        var parameters = advanced && advanced.parameters;
        return parameters && parameters[name] || null;
    }
    function updateAdvancedSamplingUI(profile) {
        var capability = advancedParameterCapability(profile, 'sigma_shift');
        var supported = !!(capability && capability.supported === true);
        var defaultValue = Number(capability && capability.default);
        if (!Number.isFinite(defaultValue))
            defaultValue = 3.0;
        if (!supported || !Number.isFinite(state.sigmaShift))
            state.sigmaShift = defaultValue;
        if (els.sigmaShift) {
            els.sigmaShift.disabled = !supported;
            els.sigmaShift.value = String(state.sigmaShift);
            els.sigmaShift.min = String(Number(capability && capability.min) || 0.01);
            els.sigmaShift.max = String(Number(capability && capability.max) || 100);
            els.sigmaShift.step = String(Number(capability && capability.step) || 0.01);
            els.sigmaShift.title = supported
                ? String(capability.reason || 'Applied by the selected runtime')
                : String(capability && capability.reason || 'Not admitted by the selected runtime');
        }
        if (els.sigmaShiftRange) {
            els.sigmaShiftRange.disabled = !supported;
            els.sigmaShiftRange.value = String(Math.min(20, state.sigmaShift));
            els.sigmaShiftRange.min = String(Number(capability && capability.min) || 0.01);
            els.sigmaShiftRange.max = String(Math.min(20, Number(capability && capability.max) || 20));
            els.sigmaShiftRange.step = String(Number(capability && capability.step) || 0.01);
        }
        var note = document.getElementById('gen-advanced-sampling-note');
        if (note) {
            note.textContent = supported
                ? 'Sigma Shift is applied by this Mojo worker. The remaining advanced sampler fields are not admitted.'
                : 'This runtime admits no user-adjustable advanced sampler fields; unsupported values stay disabled and are not posted.';
        }
    }
    function setVideoControlsForMode(enabled) {
        [els.videoSection, els.videoConditioningSection].forEach(function (section) {
            if (!section)
                return;
            section.style.display = '';
            Array.from(section.querySelectorAll('input, select, textarea')).forEach(function (control) {
                control.disabled = !enabled || !!control.closest('.gen-parity-disabled');
            });
        });
        var note = document.getElementById('gen-video-advanced-note');
        if (note) {
            note.textContent = enabled
                ? 'The admitted LTX2 runner outputs MP4. Input audio, boomerang, trimming, interpolation, and extension remain unavailable.'
                : 'Select the admitted LTX2 video model to enable its compiled video request parameters.';
        }
        // Native LTX2 frame counts and durations are discrete profile choices.
        // FPS is part of each profile and changes with the selected profile.
        [els.fpsInput, els.fpsRange].forEach(function (control) {
            if (control)
                control.disabled = true;
        });
        if (els.h3AttentionRow)
            els.h3AttentionRow.style.display = state.arch === 'minimax_h3' ? '' : 'none';
        if (els.h3StepCacheRow)
            els.h3StepCacheRow.style.display = state.arch === 'minimax_h3' ? '' : 'none';
        if (state.arch !== 'minimax_h3') {
            if (els.videoGuidanceMode && els.videoGuidanceMode.closest('.gen-param-row'))
                els.videoGuidanceMode.closest('.gen-param-row').style.display = '';
            if (els.cameraMotion && els.cameraMotion.closest('.gen-param-row'))
                els.cameraMotion.closest('.gen-param-row').style.display = '';
        }
    }
    function updateAdvancedVisibility() {
        var visible = state.showAdvanced !== false;
        document.querySelectorAll('#panel-generate .gen-workspace-advanced-only').forEach(function (node) {
            node.style.display = visible ? '' : 'none';
        });
        if (els.showAdvanced)
            els.showAdvanced.checked = visible;
        var count = document.getElementById('gen-advanced-count');
        if (count) {
            var rows = document.querySelectorAll('#panel-generate .gen-workspace-advanced-only .gen-param-row').length;
            count.textContent = '(' + rows + ')';
        }
    }
    function variationSupportedForArch(arch) {
        // These are the worker implementations that actually blend secondary
        // noise. Other families reject or ignore the fields and must not expose
        // the control.
        return ['zimage', 'sdxl', 'sd3', 'flux', 'chroma'].indexOf(arch) >= 0;
    }
    function displaySamplerName(name) {
        var names = {
            euler: 'Euler',
            flowmatch_euler: 'FlowMatch Euler',
            dpmpp_2m: 'DPM++ 2M',
            uni_pc: 'UniPC',
            uni_pc_bh2: 'UniPC BH2'
        };
        return names[name] || String(name || '').replace(/_/g, ' ');
    }
    function displaySchedulerName(name) {
        var names = {
            simple: 'Simple',
            normal: 'Normal',
            sgm_uniform: 'SGM Uniform',
            ideogram_logitnormal: 'Ideogram Logit-Normal',
            flux2: 'FLUX.2'
        };
        return names[name] || String(name || '').replace(/_/g, ' ');
    }
    // ── Model Loading ──
    function loadModels() {
        var videoReadiness = fetch('/v1/video', { cache: 'no-store' })
            .then(function (response) {
                if (!response.ok)
                    throw new Error('video readiness HTTP ' + response.status);
                return response.json();
            })
            .catch(function () { return null; });
        Promise.all([ModelUtils.fetchAllModels(), ModelUtils.loadCapabilities(), videoReadiness])
            .then(function (loaded) {
            state.videoStatus = loaded[2];
            // Keep installed LTX/Sulphur checkpoints selectable even when the
            // readiness request is transiently unavailable. Admission still
            // reports precise missing runtime artifacts when Generate is used.
            var models = loaded[0];
            state.capabilities = loaded[1];
            if (!models.length)
                throw new Error('empty');
            state.allModels = models;
            // Default to a KNOWN-GOOD image family, not models[0] blindly —
            // an alphabetical/scan-ordered first entry can be a VIDEO model
            // (LTX/Wan), whose graph the image queue rejects with 501
            // "workflow route 'video'" on every Generate click.
            // Krea Raw remains selectable as the training/base checkpoint, but
            // the creator's distilled Turbo checkpoint is the inference default.
            var preferred = ['krea2-turbo', 'krea2', 'klein', 'flux2', 'zimage', 'ideogram', 'sdxl', 'sd3', 'qwen'];
            var pick = state.model ? models.find(function (model) { return model.name === state.model; }) : null;
            if (!pick) {
                pick = models[0];
                outer: for (var pi = 0; pi < preferred.length; pi++) {
                    for (var mi = 0; mi < models.length; mi++) {
                        var nm = (models[mi].name + ' ' + (models[mi].arch || '')).toLowerCase();
                        if (nm.indexOf(preferred[pi]) >= 0) { pick = models[mi]; break outer; }
                    }
                }
            }
            els.model.value = pick.name;
            state.model = pick.name;
            var globalBadge = document.querySelector('#topbar .model-badge');
            if (globalBadge) {
                globalBadge.textContent = pick.name;
                globalBadge.title = pick.name;
            }
            if (els.modelSearch) {
                els.modelSearch.value = pick.name;
                els.modelSearch.placeholder = 'Search models...';
            }
            updateUIForArch(ModelUtils.archForModel(pick.name));
            els.modelWarn.classList.remove('visible');
        })
            .catch(function () {
            state.allModels = [];
            if (els.modelSearch) {
                els.modelSearch.value = '';
                els.modelSearch.placeholder = 'No admitted models found';
            }
            els.modelWarn.textContent = 'Could not load models or server capabilities';
            els.modelWarn.classList.add('visible');
        });
    }
    // ── Load VAEs and refiner models from /object_info ──
    function loadAdvancedOptions() {
        fetch('/object_info', { cache: 'no-store' })
            .then(function (r) { return r.ok ? r.json() : {}; })
            .then(function (info) {
            // Populate VAE picker
            var vaePicker = document.getElementById('gen-vae-picker');
            if (vaePicker && info.VAELoader && info.VAELoader.input && info.VAELoader.input.required && info.VAELoader.input.required.vae_name) {
                var vaeList = info.VAELoader.input.required.vae_name[0];
                if (Array.isArray(vaeList)) {
                    vaeList.forEach(function (name) {
                        var opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        vaePicker.appendChild(opt);
                    });
                }
            }
            // Populate refiner model picker (SDXL checkpoints)
            var refinerPicker = document.getElementById('gen-refiner-model');
            if (refinerPicker && info.CheckpointLoaderSimple && info.CheckpointLoaderSimple.input && info.CheckpointLoaderSimple.input.required && info.CheckpointLoaderSimple.input.required.ckpt_name) {
                var ckptList = info.CheckpointLoaderSimple.input.required.ckpt_name[0];
                if (Array.isArray(ckptList)) {
                    ckptList.forEach(function (name) {
                        var opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        refinerPicker.appendChild(opt);
                    });
                }
            }
        })
            .catch(function () { });
    }
    function setPreviewModelBadges(model, arch) {
        // Callers pass either the active request or the previewed History item.
        // Prefer that explicit identity; do not let another selected model
        // overwrite the output being inspected.
        var resolvedModel = model || state.model || '';
        var resolvedArch = arch ||
            (resolvedModel && ModelUtils.archForModel(resolvedModel)) ||
            state.arch || 'other';
        var archBadge = document.getElementById('gen-arch-badge');
        if (archBadge) {
            var archNames = { flux: 'FLUX', sdxl: 'SDXL', anima: 'ANIMA', sd3: 'SD3', sd15: 'SD1.5', ltxv: 'LTX-V', minimax_h3: 'MINIMAX-H3', wan: 'WAN', bernini: 'BERNINI-R', scail2: 'SCAIL-2', klein: 'KLEIN', krea2: 'KREA2', chroma: 'CHROMA', lens: 'MICROSOFT LENS', qwen: 'QWEN', zimage: 'Z-IMAGE', ideogram4: 'IDEOGRAM 4', sensenova: 'SENSENOVA' };
            archBadge.textContent = archNames[resolvedArch] || resolvedArch.toUpperCase();
            archBadge.dataset.arch = resolvedArch;
        }
        if (els.modelBadge) {
            var shortName = resolvedModel.length > 20 ? resolvedModel.substring(0, 18) + '...' : resolvedModel;
            els.modelBadge.textContent = shortName;
            els.modelBadge.style.display = resolvedModel ? '' : 'none';
            els.modelBadge.title = resolvedModel;
        }
    }
    // ── Arch-aware UI ──
    function updateUIForArch(arch) {
        updateGenerateUIForArch(arch);
        return;
    }
    function applyVideoGuidanceMode(preserveSteps) {
        refreshLtx2CheckpointControls();
        var creatorWorkflow = state.videoWorkflowProfile
            ? ltx2CheckpointWorkflow(state.videoCheckpoint) : null;
        if (creatorWorkflow &&
            creatorWorkflow.id === state.videoWorkflowProfile) {
            state.videoGuidanceMode = String(
                creatorWorkflow.guidance_mode || 'distilled'
            );
            state.sampler = String(
                creatorWorkflow.sampler || 'euler_ancestral_cfg_pp'
            );
            state.scheduler = String(
                creatorWorkflow.scheduler || 'sulphur_creator_8_3'
            );
            state.steps = Number(creatorWorkflow.steps) || 8;
            if (els.sampler) {
                els.sampler.innerHTML = '<option value="' + state.sampler + '">' +
                    displaySamplerName(state.sampler) + '</option>';
                els.sampler.value = state.sampler;
            }
            if (els.scheduler) {
                els.scheduler.innerHTML = '<option value="' + state.scheduler + '">' +
                    displaySchedulerName(state.scheduler) + '</option>';
                els.scheduler.value = state.scheduler;
            }
            if (els.steps) {
                els.steps.max = String(state.steps);
                els.steps.value = String(state.steps);
            }
            if (els.stepsRange) {
                els.stepsRange.max = String(state.steps);
                els.stepsRange.value = String(state.steps);
            }
            if (els.videoGuidanceMode)
                els.videoGuidanceMode.value = state.videoGuidanceMode;
            return;
        }
        var profile = activeLtx2RequestProfile() || {};
        var modes = profile.guidance_modes || {};
        var selected = modes[state.videoGuidanceMode] || {};
        state.sampler = selected.sampler ||
            (state.videoGuidanceMode === 'dev' ? 'res2s' : 'euler');
        state.scheduler = selected.scheduler ||
            (state.videoGuidanceMode === 'dev' ? 'ltx2' : 'ltx2_distilled');
        if (!preserveSteps) {
            state.steps = Number(
                state.videoGuidanceMode === 'dev'
                    ? selected.default_steps
                    : selected.steps
            ) || (state.videoGuidanceMode === 'dev' ? 15 : 8);
        }
        if (els.sampler) {
            els.sampler.innerHTML = '<option value="' + state.sampler + '">' +
                displaySamplerName(state.sampler) + '</option>';
            els.sampler.value = state.sampler;
        }
        if (els.scheduler) {
            els.scheduler.innerHTML = '<option value="' + state.scheduler + '">' +
                displaySchedulerName(state.scheduler) + '</option>';
            els.scheduler.value = state.scheduler;
        }
        var maxSteps = state.videoGuidanceMode === 'dev'
            ? (Number(selected.max_steps) || 20)
            : state.steps;
        if (els.steps) {
            els.steps.max = String(maxSteps);
            els.steps.value = String(state.steps);
        }
        if (els.stepsRange) {
            els.stepsRange.max = String(maxSteps);
            els.stepsRange.value = String(state.steps);
        }
        if (els.videoGuidanceMode)
            els.videoGuidanceMode.value = state.videoGuidanceMode;
    }
    function updateVideoUIForArch(arch) {
        state.arch = arch;
        var mode = activeLtx2RequestMode();
        var profile = activeLtx2RequestProfile();
        if (!mode || !profile) {
            if (els.modelWarn) {
                els.modelWarn.textContent = 'The server did not return registered LTX2 request profiles';
                els.modelWarn.classList.add('visible');
            }
            return;
        }
        var runnerReady = activeLtx2RequestProfiles().some(function (entry) {
            return entry.available === true;
        });
        if (els.modelWarn) {
            if (runnerReady) {
                els.modelWarn.classList.remove('visible');
            }
            else {
                els.modelWarn.textContent =
                    'LTX2 settings are available, but generation is blocked until the single runtime runner is installed';
                els.modelWarn.classList.add('visible');
            }
        }
        if (els.videoSection)
            els.videoSection.style.display = '';
        if (els.videoConditioningSection)
            els.videoConditioningSection.style.display = '';
        setVideoControlsForMode(true);
        var variationSection = document.getElementById('gen-variation-section');
        if (variationSection)
            variationSection.style.display = 'none';
        if (els.negSection)
            els.negSection.style.display = '';
        if (els.cfgRow)
            els.cfgRow.style.display = 'none';
        if (els.guidanceRow)
            els.guidanceRow.style.display = 'none';
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection)
            batchSection.style.display = 'none';
        state.batchCount = 1;
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.value = '1';
            els.toolbarBatchInput.disabled = true;
        }
        applyLtx2RequestProfile(profile);
        var profileSelectReason = 'Choose a registered native size and duration. The single runtime runner validates the request at queue time.';
        refreshLtx2CheckpointControls();
        if (els.videoCheckpoint)
            els.videoCheckpoint.value = state.videoCheckpoint;
        if (els.videoQuant)
            els.videoQuant.value = state.videoQuant;
        if (els.audioPolicy)
            els.audioPolicy.value = state.audioPolicy;
        if (els.capsPositive)
            els.capsPositive.value = state.capsPositive;
        if (els.capsNegative)
            els.capsNegative.value = state.capsNegative;
        if (els.noiseFixture)
            els.noiseFixture.value = state.noiseFixture;
        refreshCameraMotionControls(mode.camera_motions);
        applyVideoGuidanceMode(false);
        refreshVideoPromptEnhancer();
        syncDimensionInputs();
        if (els.customWidth) {
            els.customWidth.disabled = true;
            els.customWidth.title = profileSelectReason;
        }
        if (els.customHeight) {
            els.customHeight.disabled = true;
            els.customHeight.title = profileSelectReason;
        }
        var widthSlider = document.getElementById('gen-width-slider');
        var heightSlider = document.getElementById('gen-height-slider');
        if (widthSlider)
            widthSlider.disabled = true;
        if (heightSlider)
            heightSlider.disabled = true;
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML = buildAspectOptions();
            els.aspectDropdown.value = profile.width + '×' + profile.height;
        }
        updateAspectPreview();
        updateDurationHint();
        var loraSection = document.getElementById('gen-lora-section');
        if (loraSection)
            loraSection.style.display = '';
        var loraNote = document.getElementById('gen-lora-capability');
        if (loraNote)
            loraNote.textContent = 'LTX2 LoRAs are validated against the model registry before GPU work';
        var output = document.getElementById('gen-output-format');
        if (output)
            output.value = String(profile.output_format || 'mp4').toUpperCase();
        if (els.btn)
            els.btn.innerHTML = '<i data-lucide="wand-2"></i><span>Generate Video</span>';
        setPreviewModelBadges(state.model, arch);
        var runtimeLabel = document.getElementById('gen-runtime-label');
        if (runtimeLabel)
            runtimeLabel.textContent = runnerReady
                ? 'LTX2 · single Mojo runtime runner · ready'
                : 'LTX2 · single Mojo runtime runner · unavailable';
        updateAdvancedSamplingUI(null);
        renderModelLibrary();
        refreshLtx2PostUpscaleControls();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }

    function activeMinimaxH3Runner() {
        var candidates = state.videoStatus && state.videoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        return candidates.find(function (entry) {
            return entry && entry.model === 'minimax_h3_t2va';
        }) || null;
    }

    function clampMinimaxH3Dimension(value, axis) {
        var runner = activeMinimaxH3Runner();
        var constraints = runner && runner.geometry_constraints || {};
        var step = Number(constraints.dimension_step) || 32;
        var minimum = Number(axis === 'height'
            ? constraints.height_min : constraints.width_min) || 32;
        var maximum = Number(axis === 'height'
            ? constraints.height_max : constraints.width_max) || 2048;
        var numeric = Number(value);
        if (!Number.isFinite(numeric))
            numeric = axis === 'height' ? 768 : 1344;
        return Math.min(maximum, Math.max(minimum,
            Math.round(numeric / step) * step));
    }

    function clampAuthoredDimension(value, axis, fallback) {
        if (ModelUtils.archForModel(state.model) === 'minimax_h3')
            return clampMinimaxH3Dimension(value, axis);
        if (ModelUtils.isVideoModel(state.model))
            return ModelUtils.clampVideoDimension(Number(value) || fallback);
        return ModelUtils.clampDimension(Number(value) || fallback);
    }

    function updateMinimaxH3VideoUI(arch) {
        state.arch = arch;
        var runner = activeMinimaxH3Runner();
        var runnerReady = !!(runner && runner.available === true);
        var constraints = runner && runner.geometry_constraints || {};
        var profiles = Array.isArray(constraints.resolutions)
            ? constraints.resolutions : [];
        var dimensionStep = Number(constraints.dimension_step) || 32;
        var widthMin = Number(constraints.width_min) || 32;
        var widthMax = Number(constraints.width_max) || 2048;
        var heightMin = Number(constraints.height_min) || 32;
        var heightMax = Number(constraints.height_max) || 2048;
        var maxPixels = Number(constraints.max_pixels) || 768 * 1344;
        var validDimensions = Number.isInteger(Number(state.width)) &&
            Number.isInteger(Number(state.height)) &&
            state.width >= widthMin && state.width <= widthMax &&
            state.height >= heightMin && state.height <= heightMax &&
            state.width % dimensionStep === 0 && state.height % dimensionStep === 0 &&
            state.width * state.height <= maxPixels;
        if (!validDimensions) {
            state.width = 1344;
            state.height = 768;
        }
        state.seconds = Math.max(1, Math.min(15, Number(state.seconds) || 5));
        state.fps = Math.max(1, Math.min(120, Math.round(Number(state.fps) || 24)));
        state.frames = Math.max(1, Math.round(state.seconds * state.fps));
        state.steps = Math.max(2, Math.min(50, Math.round(Number(state.steps) || 20)));
        state.h3Quant = ['int8-fast', 'int8', 'bf16'].indexOf(state.h3Quant) >= 0
            ? state.h3Quant : 'int8-fast';
        state.videoQuant = state.h3Quant;
        state.h3AttentionBackend = state.h3AttentionBackend === 'sage-int8'
            ? 'sage-int8' : 'cudnn';
        state.h3StepCache = state.h3StepCache === 'high' ? 'high' : 'exact';
        state.includeAudio = true;
        state.audioPolicy = 'generate';
        state.batchCount = 1;
        if (!String(state.prompt || '').trim() && runner && typeof runner.test_prompt === 'string') {
            state.prompt = runner.test_prompt;
            if (els.prompt)
                els.prompt.value = runner.test_prompt;
        }
        setVideoControlsForMode(true);
        if (els.videoSection)
            els.videoSection.style.display = '';
        if (els.videoConditioningSection)
            els.videoConditioningSection.style.display = 'none';
        if (els.negSection)
            els.negSection.style.display = 'none';
        if (els.cfgRow)
            els.cfgRow.style.display = 'none';
        if (els.guidanceRow)
            els.guidanceRow.style.display = 'none';
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection)
            batchSection.style.display = 'none';
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.value = '1';
            els.toolbarBatchInput.disabled = true;
        }
        if (els.videoQuant) {
            var modes = runner && Array.isArray(runner.quant_modes)
                ? runner.quant_modes : [];
            els.videoQuant.innerHTML = modes.map(function (mode) {
                return '<option value="' + mode.id + '"' +
                    (mode.available === true ? '' : ' disabled') + '>' +
                    mode.label + '</option>';
            }).join('') || '<option value="int8-fast">INT8 Fast · W8A8 48 + 2</option><option value="int8">INT8 Quality · groupwise 43 + BF16 tail 7</option><option value="bf16">BF16 DiT quality · streamed</option>';
            if (!Array.from(els.videoQuant.options).some(function (option) {
                return option.value === state.videoQuant && !option.disabled;
            })) {
                var firstReadyMode = Array.from(els.videoQuant.options).find(function (option) {
                    return !option.disabled;
                });
                state.videoQuant = firstReadyMode ? firstReadyMode.value : 'int8-fast';
                state.h3Quant = state.videoQuant;
            }
            els.videoQuant.value = state.videoQuant;
        }
        if (els.h3Attention) {
            var attentionBackends = runner && Array.isArray(runner.attention_backends)
                ? runner.attention_backends : [];
            els.h3Attention.innerHTML = attentionBackends.map(function (backend) {
                return '<option value="' + backend.id + '"' +
                    (backend.available === true ? '' : ' disabled') + '>' +
                    backend.label + '</option>';
            }).join('') || '<option value="cudnn">cU-DNN · quality default</option><option value="sage-int8">Sage INT8 · experimental</option>';
            if (state.videoQuant === 'int8-fast')
                state.h3AttentionBackend = 'cudnn';
            els.h3Attention.value = state.h3AttentionBackend;
            els.h3Attention.disabled = state.videoQuant === 'int8-fast';
        }
        if (els.h3StepCache)
            els.h3StepCache.value = state.h3StepCache;
        if (els.audioPolicy) {
            els.audioPolicy.innerHTML = '<option value="generate">Generate synchronized audio</option>';
            els.audioPolicy.value = 'generate';
            els.audioPolicy.disabled = true;
        }
        [els.framesInput, els.secondsInput, els.fpsInput, els.fpsRange].forEach(function (control) {
            if (control)
                control.disabled = false;
        });
        if (els.framesInput) {
            els.framesInput.min = '1';
            els.framesInput.max = '1800';
            els.framesInput.step = '1';
            els.framesInput.value = String(state.frames);
        }
        if (els.secondsInput) {
            els.secondsInput.min = '1';
            els.secondsInput.max = '15';
            els.secondsInput.step = '0.01';
            els.secondsInput.value = state.seconds.toFixed(3);
        }
        if (els.fpsInput) {
            els.fpsInput.min = '1';
            els.fpsInput.max = '120';
            els.fpsInput.value = String(state.fps);
        }
        if (els.fpsRange) {
            els.fpsRange.min = '1';
            els.fpsRange.max = '120';
            els.fpsRange.value = String(state.fps);
        }
        if (els.steps) {
            els.steps.min = '2';
            els.steps.max = '50';
            els.steps.value = String(state.steps);
        }
        if (els.stepsRange) {
            els.stepsRange.min = '2';
            els.stepsRange.max = '50';
            els.stepsRange.value = String(state.steps);
        }
        syncDimensionInputs();
        [els.customWidth, els.customHeight,
            document.getElementById('gen-width-slider'),
            document.getElementById('gen-height-slider')].forEach(function (control) {
            if (!control)
                return;
            control.disabled = false;
            control.min = control === els.customHeight || control.id === 'gen-height-slider'
                ? String(heightMin) : String(widthMin);
            control.max = control === els.customHeight || control.id === 'gen-height-slider'
                ? String(heightMax) : String(widthMax);
            control.step = String(dimensionStep);
        });
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML = profiles.map(function (candidate) {
                var value = candidate.width + '×' + candidate.height;
                return '<option value="' + value + '">' +
                    String(candidate.aspect_ratio || 'Preset') + ' · ' + value + '</option>';
            }).join('') + '<option value="Free">Custom width × height</option>';
            var currentPreset = profiles.some(function (candidate) {
                return Number(candidate.width) === state.width &&
                    Number(candidate.height) === state.height;
            });
            els.aspectDropdown.value = currentPreset
                ? state.width + '×' + state.height : 'Free';
            els.aspectDropdown.disabled = false;
        }
        if (els.videoGuidanceMode && els.videoGuidanceMode.closest('.gen-param-row'))
            els.videoGuidanceMode.closest('.gen-param-row').style.display = 'none';
        if (els.cameraMotion && els.cameraMotion.closest('.gen-param-row'))
            els.cameraMotion.closest('.gen-param-row').style.display = 'none';
        var profileNote = document.getElementById('gen-video-profile-note');
        if (profileNote)
            profileNote.textContent = runnerReady
                ? 'H3 width and height are adjustable from 32–2048 in 32-pixel steps, up to 1,032,192 pixels. The listed sizes are tested presets, not a lock. Seconds are adjustable from 1–15; INT8 Fast, INT8 Quality, and BF16 remain switchable.'
                : 'MiniMax-H3 controls remain visible, but the selected local binaries, INT8 encoder cache, or quality gate are unavailable.';
        var advancedNote = document.getElementById('gen-video-advanced-note');
        if (advancedNote)
            advancedNote.textContent = 'H3 runs entirely on GPU, exits after denoising, decodes in a fresh GPU process, and muxes with NVENC.';
        if (els.modelWarn) {
            els.modelWarn.textContent = runnerReady
                ? '' : 'MiniMax-H3 is not admitted by the current machine-local quality gate';
            els.modelWarn.classList.toggle('visible', !runnerReady);
        }
        if (els.btn)
            els.btn.innerHTML = '<i data-lucide="wand-2"></i><span>Generate H3 Video + Audio</span>';
        var output = document.getElementById('gen-output-format');
        if (output)
            output.value = 'MP4';
        var runtimeLabel = document.getElementById('gen-runtime-label');
        if (runtimeLabel)
            runtimeLabel.textContent = runnerReady
                ? 'MiniMax-H3 · Mojo GPU runtime · ready'
                : 'MiniMax-H3 · Mojo GPU runtime · unavailable';
        setPreviewModelBadges(state.model, arch);
        updateAspectPreview();
        updateDurationHint();
        updateAdvancedSamplingUI(null);
        renderModelLibrary();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }

    function activeWan22Runner() {
        var candidates = state.videoStatus && state.videoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        return candidates.find(function (entry) {
            return entry && entry.model === 'wan22_t2v';
        }) || null;
    }

    function activeWan22LoraMode() {
        var runner = activeWan22Runner();
        return runner && runner.modes && runner.modes.lora || null;
    }

    function wanCreatorI2vSize(sourceWidth, sourceHeight) {
        // Exact port of Wan creator best_output_size(): the advertised
        // 1280×704/704×1280 value is a max-area bucket. I2V follows the source
        // aspect ratio on the model's 32-pixel grid.
        var area = 1280 * 704;
        var ratio = sourceWidth / sourceHeight;
        var outputWidth = Math.sqrt(area * ratio);
        var outputHeight = area / outputWidth;
        var widthFirst = Math.floor(outputWidth / 32) * 32;
        var heightFromWidth = Math.floor(area / widthFirst / 32) * 32;
        var ratioWidthFirst = widthFirst / heightFromWidth;
        var heightFirst = Math.floor(outputHeight / 32) * 32;
        var widthFromHeight = Math.floor(area / heightFirst / 32) * 32;
        var ratioHeightFirst = widthFromHeight / heightFirst;
        var distortionWidth = Math.max(
            ratio / ratioWidthFirst, ratioWidthFirst / ratio);
        var distortionHeight = Math.max(
            ratio / ratioHeightFirst, ratioHeightFirst / ratio);
        return distortionWidth < distortionHeight
            ? { width: widthFirst, height: heightFromWidth }
            : { width: widthFromHeight, height: heightFirst };
    }

    function syncWanI2vSteps() {
        if (state.arch !== 'wan')
            return;
        if (state.initImagePath &&
            state.initImageWidth > 0 && state.initImageHeight > 0) {
            var creatorSize = wanCreatorI2vSize(
                state.initImageWidth, state.initImageHeight);
            state.width = creatorSize.width;
            state.height = creatorSize.height;
        }
        else {
            state.width = 1280;
            state.height = 704;
        }
        state.steps = 50;
        if (els.steps) {
            els.steps.min = String(state.steps);
            els.steps.max = String(state.steps);
            els.steps.value = String(state.steps);
            els.steps.disabled = true;
        }
        if (els.stepsRange) {
            els.stepsRange.min = String(state.steps);
            els.stepsRange.max = String(state.steps);
            els.stepsRange.value = String(state.steps);
            els.stepsRange.disabled = true;
        }
        var note = document.getElementById('gen-video-profile-note');
        if (note) {
            note.textContent = state.initImagePath
                ? 'Wan2.2 TI2V-5B I2V · creator source-aspect sizing and first-frame conditioning · ' +
                    state.width + '×' + state.height +
                    ' · BF16 · 121 frames · 24 FPS · 50 UniPC steps · shift 5'
                : 'Wan2.2 TI2V-5B T2V · 1280×704 · BF16 · 121 frames · 24 FPS · 50 UniPC steps · shift 5';
        }
        syncDimensionInputs();
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML =
                '<option value="' + state.width + '×' + state.height + '">' +
                state.width + '×' + state.height + '</option>';
            els.aspectDropdown.value = state.width + '×' + state.height;
        }
    }

    function updateWanVideoUI(arch) {
        state.arch = arch;
        var runner = activeWan22Runner();
        var runnerReady = !!runner && runner.status === 'quality_profile_ready';
        if (!runnerReady) {
            if (els.modelWarn) {
                els.modelWarn.textContent = runner
                    ? 'Wan2.2 request controls are available; generation will report any missing runner prerequisites'
                    : 'Wan2.2 request controls are available; no runner status was returned';
                els.modelWarn.classList.add('visible');
            }
        }
        else if (els.modelWarn)
            els.modelWarn.classList.remove('visible');
        runner = runner || {};
        setVideoControlsForMode(true);
        if (els.videoSection)
            els.videoSection.style.display = '';
        // These controls are LTX2-only conditioning cache overrides. Wan owns
        // its UMT5 encode automatically and uses the shared source-image picker.
        if (els.videoConditioningSection)
            els.videoConditioningSection.style.display = 'none';
        var variationSection = document.getElementById('gen-variation-section');
        if (variationSection)
            variationSection.style.display = 'none';
        if (els.negSection)
            els.negSection.style.display = '';
        if (els.cfgRow)
            els.cfgRow.style.display = 'flex';
        if (els.guidanceRow)
            els.guidanceRow.style.display = 'none';
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection)
            batchSection.style.display = 'none';
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.value = '1';
            els.toolbarBatchInput.disabled = true;
        }
        state.batchCount = 1;
        state.width = Number(runner.target_width) || 1280;
        state.height = Number(runner.target_height) || 704;
        state.frames = Number(runner.target_frame_count) || 121;
        state.fps = 24;
        state.seconds = state.frames / state.fps;
        state.cfg = Number(runner.default_guidance) || 5;
        state.guidance = state.cfg;
        state.videoQuant = 'bf16';
        state.sampler = 'uni_pc';
        state.scheduler = 'normal';
        if (els.cfg) els.cfg.value = String(state.cfg);
        if (els.cfgRange) els.cfgRange.value = String(state.cfg);
        if (els.guidance) els.guidance.value = String(state.guidance);
        if (els.guidanceRange) els.guidanceRange.value = String(state.guidance);
        if (els.framesInput) {
            els.framesInput.value = String(state.frames);
            els.framesInput.disabled = true;
        }
        if (els.fpsInput) els.fpsInput.value = String(state.fps);
        if (els.fpsRange) els.fpsRange.value = String(state.fps);
        if (els.secondsInput) {
            els.secondsInput.value = state.seconds.toFixed(3);
            els.secondsInput.disabled = true;
        }
        if (els.sampler) {
            els.sampler.innerHTML = '<option value="uni_pc">UniPC (Wan creator)</option>';
            els.sampler.value = state.sampler;
        }
        if (els.scheduler) {
            els.scheduler.innerHTML = '<option value="normal">Wan flow schedule</option>';
            els.scheduler.value = state.scheduler;
        }
        refreshCameraMotionControls(runner.camera_motions);
        syncWanI2vSteps();
        syncDimensionInputs();
        if (els.customWidth) els.customWidth.disabled = true;
        if (els.customHeight) els.customHeight.disabled = true;
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML =
                '<option value="' + state.width + '×' + state.height + '">' +
                state.width + '×' + state.height + '</option>';
            els.aspectDropdown.value = state.width + '×' + state.height;
            els.aspectDropdown.disabled = true;
        }
        updateAspectPreview();
        updateDurationHint();
        var loraSection = document.getElementById('gen-lora-section');
        if (loraSection)
            loraSection.style.display = '';
        var loraMode = activeWan22LoraMode();
        var loraNote = document.getElementById('gen-lora-capability');
        if (loraNote)
            loraNote.textContent = loraMode && loraMode.available
                ? 'One Wan2.2 TI2V-5B LoRA · 5B tensor shapes are checked before GPU work'
                : 'Wan2.2 TI2V-5B LoRA runtime is not ready';
        if (state.loras.length > 1)
            state.loras = state.loras.slice(0, 1);
        renderLoraList();
        var output = document.getElementById('gen-output-format');
        if (output)
            output.value = 'MP4';
        if (els.btn)
            els.btn.innerHTML = '<i data-lucide="wand-2"></i><span>Generate Wan Video</span>';
        setPreviewModelBadges(state.model, arch);
        var runtimeLabel = document.getElementById('gen-runtime-label');
        if (runtimeLabel)
            runtimeLabel.textContent = runnerReady
                ? 'Wan2.2 TI2V-5B · Mojo creator T2V/I2V runner · ready'
                : 'Wan2.2 TI2V-5B · Mojo creator request route';
        updateAdvancedSamplingUI(null);
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }
    function restoreImageStepBounds(defaultSteps) {
        var selectedDefault = Math.max(1, Number(defaultSteps) || 1);
        var inputMax = Math.max(IMAGE_STEPS_INPUT_MAX, selectedDefault);
        var sliderMax = Math.min(
            inputMax,
            Math.max(IMAGE_STEPS_SLIDER_MAX, selectedDefault)
        );
        if (els.steps) {
            els.steps.min = '1';
            els.steps.max = String(inputMax);
        }
        if (els.stepsRange) {
            els.stepsRange.min = '1';
            els.stepsRange.max = String(sliderMax);
        }
    }
    function selectedModelGenerationDefaults() {
        var selected = state.allModels.find(function (model) {
            return model && model.name === state.model;
        });
        return selected && selected.generationDefaults
            ? selected.generationDefaults : {};
    }
    function updateGenerateUIForArch(arch) {
        if (arch === 'ltxv') {
            updateVideoUIForArch(arch);
            return;
        }
        if (arch === 'minimax_h3') {
            updateMinimaxH3VideoUI(arch);
            return;
        }
        if (arch === 'wan') {
            updateWanVideoUI(arch);
            return;
        }
        state.arch = arch;
        var profile = activeCapabilityProfile(arch);
        if (!profile) {
            if (els.modelWarn) {
                els.modelWarn.textContent = 'This model family is not admitted by the image runtime';
                els.modelWarn.classList.add('visible');
            }
            return;
        }
        if (els.modelWarn)
            els.modelWarn.classList.remove('visible');
        setVideoControlsForMode(false);
        if (els.toolbarBatchInput)
            els.toolbarBatchInput.disabled = false;
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection)
            batchSection.style.display = 'flex';
        var output = document.getElementById('gen-output-format');
        if (output)
            output.value = 'PNG';
        var defaults = Object.assign(
            {},
            profile.defaults || {},
            selectedModelGenerationDefaults()
        );
        var defaultSteps = Number(defaults.steps) || 20;
        restoreImageStepBounds(defaultSteps);
        state.steps = defaultSteps;
        state.cfg = Number(defaults.cfg);
        if (!Number.isFinite(state.cfg))
            state.cfg = 4.5;
        state.sampler = defaults.sampler || 'euler';
        state.scheduler = defaults.scheduler || 'simple';
        var sigmaCapability = advancedParameterCapability(profile, 'sigma_shift');
        var sigmaDefault = Number(sigmaCapability && sigmaCapability.default);
        state.sigmaShift = Number.isFinite(sigmaDefault) ? sigmaDefault : 3.0;
        updateAdvancedSamplingUI(profile);
        if (els.steps) els.steps.value = String(state.steps);
        if (els.stepsRange) els.stepsRange.value = String(state.steps);
        if (els.cfg) els.cfg.value = String(state.cfg);
        if (els.cfgRange) els.cfgRange.value = String(state.cfg);
        if (els.cfg) els.cfg.min = '0';
        if (els.cfgRange) els.cfgRange.min = '0';
        var samplers = profile.samplers || {};
        var supportedSamplers = Array.isArray(samplers.supported_samplers)
            ? samplers.supported_samplers : [state.sampler];
        var supportedSchedulers = Array.isArray(samplers.supported_schedulers)
            ? samplers.supported_schedulers : [state.scheduler];
        if (els.sampler) {
            els.sampler.innerHTML = '';
            supportedSamplers.forEach(function (name) {
                var option = document.createElement('option');
                option.value = name;
                option.textContent = displaySamplerName(name);
                els.sampler.appendChild(option);
            });
            if (supportedSamplers.indexOf(state.sampler) < 0)
                state.sampler = supportedSamplers[0] || 'euler';
            els.sampler.value = state.sampler;
        }
        if (els.scheduler) {
            els.scheduler.innerHTML = '';
            supportedSchedulers.forEach(function (name) {
                var option = document.createElement('option');
                option.value = name;
                option.textContent = displaySchedulerName(name);
                els.scheduler.appendChild(option);
            });
            if (supportedSchedulers.indexOf(state.scheduler) < 0)
                state.scheduler = supportedSchedulers[0] || 'simple';
            els.scheduler.value = state.scheduler;
        }
        var negativeSupported = featureSupported(profile, 'negative_prompt');
        if (els.negSection)
            els.negSection.style.display = negativeSupported ? '' : 'none';
        if (!negativeSupported) {
            state.negPrompt = '';
            if (els.negPrompt) els.negPrompt.value = '';
        }
        if (els.cfgRow)
            els.cfgRow.style.display = featureSupported(profile, 'cfg') ? 'flex' : 'none';
        if (els.guidanceRow)
            els.guidanceRow.style.display = 'none';
        var variationSection = document.getElementById('gen-variation-section');
        if (variationSection)
            variationSection.style.display = variationSupportedForArch(arch) ? '' : 'none';
        if (!variationSupportedForArch(arch))
            state.variationStrength = 0;
        var loraFeature = profile.features && profile.features.lora;
        var loraSupported = !!(loraFeature && loraFeature.supported);
        var loraSection = document.getElementById('gen-lora-section');
        if (loraSection)
            loraSection.style.display = loraSupported ? '' : 'none';
        var loraNote = document.getElementById('gen-lora-capability');
        var maxLoras = loraFeature && loraFeature.max_count;
        if (loraNote)
            loraNote.textContent = maxLoras == null
                ? 'Multiple LoRAs supported'
                : ('Maximum ' + maxLoras + ' LoRA' + (maxLoras === 1 ? '' : 's'));
        if (!loraSupported) {
            state.loras = [];
            renderLoraList();
        } else if (Number.isInteger(maxLoras) && state.loras.length > maxLoras) {
            state.loras = state.loras.slice(0, maxLoras);
            renderLoraList();
        }
        if (els.btn)
            els.btn.innerHTML = '<i data-lucide="wand-2"></i><span>Generate</span>';
        setPreviewModelBadges(state.model, arch);
        var runtimeLabel = document.getElementById('gen-runtime-label');
        if (runtimeLabel)
            runtimeLabel.textContent = profile.backend + ' · Mojo image worker · admitted';
        var aspects = getActiveAspects();
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML = buildAspectOptions();
        }
        if (aspects.length) {
            var defaultWidth = Number(defaults.width);
            var defaultHeight = Number(defaults.height);
            var selected = aspects.find(function (size) {
                return size.w === defaultWidth && size.h === defaultHeight;
            }) || aspects[0];
            state.width = selected.w;
            state.height = selected.h;
            syncDimensionInputs();
            syncAspectDropdown();
            updateAspectPreview();
        }
        var swap = document.getElementById('gen-swap-btn');
        if (swap) {
            swap.style.display = aspects.some(function (size) {
                return aspects.some(function (other) {
                    return size.w === other.h && size.h === other.w;
                });
            }) ? '' : 'none';
        }
        renderModelLibrary();
        refreshLtx2PostUpscaleControls();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }
    function updateDurationHint() {
        if (!els.durationHint)
            return;
        els.durationHint.textContent = state.frames + ' frames \u00b7 ' +
            Number(state.seconds.toFixed(3)) + 's at ' + state.fps + 'fps';
        if (ModelUtils.archForModel(state.model) === 'ltxv')
            updateLtx2ProfileStatus();
    }
    // ── Token Counter ──
    function updateTokenCount() {
        var count = state.prompt.trim() ? state.prompt.trim().split(/\s+/).length : 0;
        var el = document.getElementById('gen-token-count');
        if (el)
            el.textContent = '~' + count + ' tokens';
    }
    // ── LoRA Management ──
    function loadLoras() {
        fetch('/models/loras')
            .then(function (r) { return r.ok ? r.json() : []; })
            .then(function (list) {
            var picker = document.getElementById('gen-lora-picker');
            if (!picker)
                return;
            if (!list || !list.length) {
                picker.innerHTML = '<option value="" disabled selected>No LoRAs installed</option>';
                return;
            }
            picker.innerHTML = '<option value="" disabled selected>Select a LoRA…</option>';
            list.forEach(function (name) {
                var opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                picker.appendChild(opt);
            });
            updateLoraPickerPrompt();
        })
            .catch(function () { });
    }
    function updateLoraPickerPrompt() {
        var picker = document.getElementById('gen-lora-picker');
        if (!picker || !picker.options.length)
            return;
        var prompt = picker.options[0];
        prompt.value = '';
        prompt.disabled = true;
        prompt.textContent = state.loras.length
            ? (state.loras.length + ' LoRA' + (state.loras.length === 1 ? '' : 's') + ' loaded · add another…')
            : 'Select a LoRA…';
        picker.selectedIndex = 0;
    }
    function addLora(name) {
        if (state.loras.some(function (l) { return l.name === name; }))
            return;
        var profile = activeCapabilityProfile();
        var feature = state.arch === 'ltxv'
            ? { supported: true, max_count: null }
            : (state.arch === 'wan'
                ? {
                    supported: !!(activeWan22LoraMode() &&
                        activeWan22LoraMode().available),
                    max_count: 1
                }
                : (profile && profile.features && profile.features.lora));
        if (!feature || feature.supported !== true) {
            showError('The selected model does not admit LoRA overlays');
            return;
        }
        var maxCount = feature.max_count;
        if (Number.isInteger(maxCount) && state.loras.length >= maxCount) {
            showError('The selected model admits at most ' + maxCount + ' LoRA' + (maxCount === 1 ? '' : 's'));
            return;
        }
        state.loras.push({
            name: name,
            strength: 1.0,
            enabled: true,
            role: state.arch === 'ltxv' && /distill/i.test(name)
                ? 'distillation' : 'overlay'
        });
        renderLoraList();
    }
    function removeLora(idx) {
        state.loras.splice(idx, 1);
        renderLoraList();
    }
    function renderLoraList() {
        var list = document.getElementById('gen-lora-list');
        if (!list)
            return;
        list.innerHTML = '';
        state.loras.forEach(function (lora, idx) {
            // Ensure enabled property exists (migration from old format)
            if (lora.enabled === undefined)
                lora.enabled = true;
            if (lora.role !== 'distillation')
                lora.role = 'overlay';
            var disabledClass = lora.enabled ? '' : ' gen-lora-disabled';
            var row = document.createElement('div');
            row.className = 'gen-lora-row' + disabledClass;
            var roleControl = state.arch === 'ltxv'
                ? ('<select class="gen-select gen-lora-role"' +
                    (lora.enabled ? '' : ' disabled') + '>' +
                    '<option value="overlay"' +
                    (lora.role === 'overlay' ? ' selected' : '') +
                    '>Overlay</option>' +
                    '<option value="distillation"' +
                    (lora.role === 'distillation' ? ' selected' : '') +
                    '>Distillation</option></select>')
                : '';
            row.innerHTML =
                '<button class="gen-toggle gen-lora-toggle' + (lora.enabled ? ' on' : '') + '" data-idx="' + idx + '"></button>' +
                    '<span class="gen-lora-name">' + lora.name + '</span>' +
                    roleControl +
                    '<input type="range" class="gen-range gen-lora-strength" min="-1" max="2" step="0.05" value="' + lora.strength + '"' + (lora.enabled ? '' : ' disabled') + '>' +
                    '<input type="number" class="gen-number-input gen-lora-val-input" min="-10" max="10" step="0.05" value="' + lora.strength.toFixed(2) + '"' + (lora.enabled ? '' : ' disabled') + '>' +
                    '<button class="gen-lora-remove" data-idx="' + idx + '">&times;</button>';
            list.appendChild(row);
        });
        // Bind events
        list.querySelectorAll('.gen-lora-strength').forEach(function (slider, idx) {
            slider.addEventListener('input', function () {
                state.loras[idx].strength = parseFloat(this.value);
                var valInput = this.parentElement.querySelector('.gen-lora-val-input');
                if (valInput)
                    valInput.value = state.loras[idx].strength.toFixed(2);
            });
        });
        list.querySelectorAll('.gen-lora-val-input').forEach(function (input, idx) {
            input.addEventListener('change', function () {
                var v = Math.max(-10, Math.min(10, parseFloat(this.value) || 0));
                this.value = v.toFixed(2);
                state.loras[idx].strength = v;
                var slider = this.parentElement.querySelector('.gen-lora-strength');
                if (slider)
                    slider.value = String(Math.max(-1, Math.min(2, v))); // clamp slider visual
            });
        });
        list.querySelectorAll('.gen-lora-role').forEach(function (select, idx) {
            select.addEventListener('change', function () {
                var role = this.value === 'distillation'
                    ? 'distillation' : 'overlay';
                if (role === 'distillation') {
                    state.loras.forEach(function (candidate, candidateIdx) {
                        if (candidateIdx !== idx)
                            candidate.role = 'overlay';
                    });
                }
                state.loras[idx].role = role;
                renderLoraList();
            });
        });
        list.querySelectorAll('.gen-lora-toggle').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var i = parseInt(this.dataset.idx);
                state.loras[i].enabled = !state.loras[i].enabled;
                renderLoraList();
            });
        });
        list.querySelectorAll('.gen-lora-remove').forEach(function (btn) {
            btn.addEventListener('click', function () {
                removeLora(parseInt(this.dataset.idx));
            });
        });
        updateLoraPickerPrompt();
    }
    // ── Workflow Builder ──
    function buildWorkflow() {
        // Append style suffix to prompt
        var finalPrompt = state.prompt;
        if (state.stylePreset && state.stylePreset !== 'none') {
            for (var i = 0; i < stylePresets.length; i++) {
                if (stylePresets[i].id === state.stylePreset) {
                    finalPrompt += stylePresets[i].suffix;
                    break;
                }
            }
        }
        // Filter disabled LoRAs
        var enabledLoras = state.loras.filter(function (l) { return l.enabled !== false; });
        return WorkflowBuilder.build({
            model: state.model || '',
            prompt: finalPrompt,
            negPrompt: state.negPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            sampler: state.sampler,
            scheduler: state.scheduler,
            sigmaShift: state.sigmaShift,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps,
            referenceImagePath: state.referenceImagePath,
            referenceMaskPath: state.referenceMaskPath,
            drivingVideoPath: state.drivingVideoPath,
            drivingMaskVideoPath: state.drivingMaskVideoPath,
            scail2Mode: state.scail2Mode,
            additionalReferenceImagePaths: state.additionalReferenceImagePaths.slice(),
            additionalReferenceMaskPaths: state.additionalReferenceMaskPaths.slice(),
            initImageName: state.initImagePath || state.initImageName || '',
            ltx2Mode: state.videoGuidanceMode,
            quantization: state.videoQuant,
            h3AttentionBackend: state.h3AttentionBackend,
            h3StepCache: state.h3StepCache,
            ltx2WorkflowProfile: state.videoWorkflowProfile,
            ltx2PromptEnhancer: state.videoPromptEnhancer,
            ltx2AudioPolicy: state.audioPolicy,
            ltx2PostUpscaler: state.postUpscaler,
            ltx2PostUpscaleFactor: state.postUpscaleFactor,
            ltx2CameraMotion: state.cameraMotion || 'none',
            loras: enabledLoras,
            // Phase 3: Advanced
            vae: state.vae,
            clipSkip: state.clipSkip,
            cfgRescale: state.cfgRescale,
            seamlessX: state.seamlessX,
            seamlessY: state.seamlessY,
            // Phase 3: Refiner
            refinerModel: state.refinerModel,
            refinerScheduler: state.refinerScheduler,
            refinerSteps: state.refinerSteps,
            refinerCfg: state.refinerCfg,
            refinerStart: state.refinerStart,
            aestheticScore: state.aestheticScore,
            negAestheticScore: state.negAestheticScore,
            // Phase 3: Compositing
            coherenceMode: state.coherenceMode,
            edgeSize: state.edgeSize,
            minDenoise: state.minDenoise,
            maskBlur: state.maskBlur,
            infillMethod: state.infillMethod
        });
    }
    function buildGenerateRequest(seed) {
        var finalPrompt = state.prompt;
        if (state.stylePreset && state.stylePreset !== 'none') {
            for (var i = 0; i < stylePresets.length; i++) {
                if (stylePresets[i].id === state.stylePreset) {
                    finalPrompt += stylePresets[i].suffix;
                    break;
                }
            }
        }
        var request = {
            model: state.model || '',
            prompt: finalPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            seed: seed,
            sampler: state.sampler || 'euler',
            scheduler: state.scheduler || 'simple',
            cfg: state.cfg,
            images: 1
        };
        var profile = activeCapabilityProfile();
        if (featureSupported(profile, 'negative_prompt') && state.negPrompt)
            request.negative = state.negPrompt;
        if (variationSupportedForArch(state.arch) && state.variationStrength > 0) {
            request.variation_seed = state.variationSeed;
            request.variation_strength = state.variationStrength;
        }
        var sigmaCapability = advancedParameterCapability(profile, 'sigma_shift');
        if (sigmaCapability && sigmaCapability.supported === true)
            request.sigma_shift = state.sigmaShift;
        var enabledLoras = state.loras.filter(function (lora) { return lora.enabled !== false; });
        if (enabledLoras.length) {
            request.lora = enabledLoras.map(function (lora) {
                return { name: lora.name, weight: Number(lora.strength) };
            });
        }
        return request;
    }
    function buildVideoRequest(seed) {
        var finalPrompt = state.prompt.trim();
        if (state.stylePreset && state.stylePreset !== 'none') {
            for (var i = 0; i < stylePresets.length; i++) {
                if (stylePresets[i].id === state.stylePreset) {
                    finalPrompt += stylePresets[i].suffix;
                    break;
                }
            }
        }
        if (state.arch === 'minimax_h3') {
            return {
                schema: 'serenity.genparams.v1',
                model: 'minimax_h3',
                runner: 'minimax_h3_mojo_request',
                prompt: finalPrompt.trim(),
                width: state.width,
                height: state.height,
                frames: state.frames,
                fps: state.fps,
                duration_seconds: Math.max(1, Math.min(15,
                    Number(state.seconds) || state.frames / Math.max(1, state.fps))),
                steps: Number(state.steps) || 20,
                seed: seed,
                quant: state.videoQuant === 'bf16'
                    ? 'bf16'
                    : (state.videoQuant === 'int8' ? 'int8' : 'int8-fast'),
                attention_backend: state.h3AttentionBackend === 'sage-int8'
                    ? 'sage-int8' : 'cudnn',
                step_cache: state.h3StepCache === 'high' ? 'high' : 'exact',
                include_audio: true
            };
        }
        if (state.arch === 'wan') {
            var wanI2v = !!state.initImagePath;
            return {
                schema: 'serenity.genparams.v1',
                model: 'wan22',
                prompt: finalPrompt.trim(),
                negative_prompt: (state.negPrompt || '').trim(),
                width: state.width,
                height: state.height,
                frames: 121,
                steps: 50,
                seed: seed,
                fps: 24,
                guidance: 5,
                quant: 'bf16',
                sampler: 'uni_pc',
                scheduler: 'normal',
                camera_motion: state.cameraMotion || 'none',
                image_path: wanI2v ? state.initImagePath : '',
                lora: state.loras.filter(function (lora) {
                    return lora.enabled !== false;
                }).slice(0, 1).map(function (lora) {
                    return {
                        name: lora.name,
                        weight: Number(lora.strength)
                    };
                })
            };
        }
        // Guidance mode owns the complete sampler tuple. Re-normalize at the
        // submission boundary so a reused dev request cannot leave distilled
        // mode paired with the stale dev step count (or vice versa).
        applyVideoGuidanceMode(false);
        var request = {
            schema: 'serenity.genparams.v1',
            model: 'ltx2',
            runner: 'ltx2_mojo_request',
            checkpoint: state.videoCheckpoint,
            quant: state.videoQuant,
            prompt: finalPrompt.trim(),
            negative: (state.negPrompt || '').trim(),
            width: state.width,
            height: state.height,
            frames: state.frames,
            steps: state.steps,
            seed: seed,
            fps: state.fps,
            guidance_mode: state.videoGuidanceMode,
            workflow_profile: state.videoWorkflowProfile || '',
            prompt_enhancer: state.videoPromptEnhancer || 'none',
            sampler: state.sampler,
            scheduler: state.scheduler,
            caps_positive: state.capsPositive.trim(),
            caps_negative: state.capsNegative.trim(),
            noise_fixture: state.noiseFixture.trim(),
            include_audio: state.includeAudio === true,
            audio_policy: state.audioPolicy ||
                (state.includeAudio === true ? 'generate' : 'none'),
            camera_motion: state.cameraMotion || 'none',
            post_upscale: state.postUpscaler === 'none'
                ? { id: 'none', factor: 1 }
                : {
                    id: state.postUpscaler,
                    factor: Number(state.postUpscaleFactor) === 4 ? 4 : 2
                },
            lora: state.loras.filter(function (lora) {
                return lora.enabled !== false;
            }).map(function (lora) {
                return {
                    name: lora.name,
                    weight: Number(lora.strength),
                    role: lora.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
            })
        };
        // The source picker is shared by image and video generation, but its
        // uploaded path previously never reached the LTX2 request. That made
        // the visible "I2V" source a preview-only decoration and silently ran
        // T2V. LTX2's server performs the creator CRF-33 preparation and owns
        // the exact strength, so send the source only and let it author 1.0.
        if (state.initImagePath)
            request.image_path = state.initImagePath;
        return request;
    }
    function videoResultUrl(videoId, manifest) {
        if (manifest && manifest.mp4_url)
            return String(manifest.mp4_url);
        var artifact = String(manifest && manifest.artifact_path || '');
        var filename = artifact.split('/').pop();
        return filename
            ? ('/out/' + encodeURIComponent(videoId) + '/' + encodeURIComponent(filename))
            : '';
    }
    function pollVideoGeneration(job, request, reusable) {
        var videoId = String(job.video_id || job.prompt_id || '');
        var statusUrl = String(job.status_url || ('/out/' + encodeURIComponent(videoId) + '/status.json'));
        var resultUrl = String(job.result_url || ('/out/' + encodeURIComponent(videoId) + '/result.json'));
        var token = ++state.videoPollToken;
        state.pendingVideoJobs[videoId] = true;
        function poll() {
            if (token !== state.videoPollToken || !state.pendingVideoJobs[videoId])
                return;
            fetch(statusUrl, { cache: 'no-store' })
                .then(function (response) {
                if (!response.ok)
                    throw new Error('status HTTP ' + response.status);
                return response.json();
            })
                .then(function (status) {
                var step = Number(status.step) || 0;
                var total = Number(status.total) || 0;
                var phase = String(status.message || status.phase || 'LTX2 running');
                var pct = total > 0 ? Math.max(0, Math.min(100, step / total * 100)) : 4;
                updateGenerationActivity(phase, step, total);
                els.progressBar.style.width = pct + '%';
                els.progressLabel.textContent = phase + (total > 0 ? ' · Step ' + step + ' / ' + total : '');
                els.progressLabel.classList.add('visible');
                if (els.leftProgressBar)
                    els.leftProgressBar.style.width = pct + '%';
                if (els.leftProgressLabel)
                    els.leftProgressLabel.textContent = els.progressLabel.textContent;
                if (status.state === 'failed' || status.state === 'error')
                    throw new Error(phase);
                if (status.state !== 'done') {
                    setTimeout(poll, 500);
                    return null;
                }
                return fetch(resultUrl, { cache: 'no-store' })
                    .then(function (response) {
                    if (!response.ok)
                        throw new Error('result HTTP ' + response.status);
                    return response.json();
                });
            })
                .then(function (manifest) {
                if (!manifest)
                    return;
                var src = videoResultUrl(videoId, manifest);
                if (!src)
                    throw new Error('completed video manifest has no playable MP4');
                var metadata = Object.assign({}, reusable, {
                    params: reusable,
                    frame_count: request.frames,
                    fps: request.fps
                });
                displayVideo(src, metadata);
                addToGallery(src, true, metadata);
                state.completedVideoJobs[videoId] = true;
                var completedIds = Object.keys(state.completedVideoJobs);
                if (completedIds.length > 256)
                    delete state.completedVideoJobs[completedIds[0]];
                delete state.pendingVideoJobs[videoId];
                state.pendingBatch = 0;
                setGenerating(false);
            })
                .catch(function (error) {
                if (token !== state.videoPollToken)
                    return;
                if (/status HTTP 404/.test(error.message)) {
                    setTimeout(poll, 500);
                    return;
                }
                delete state.pendingVideoJobs[videoId];
                state.pendingBatch = 0;
                showError('Video generation failed: ' + error.message);
                setGenerating(false);
            });
        }
        setTimeout(poll, 250);
    }
    function generateVideo() {
        if (ModelUtils.archForModel(state.model) === 'ltxv' &&
            !exactLtx2RequestProfile()) {
            var note = document.getElementById('gen-video-profile-note');
            showError(note
                ? note.textContent
                : 'This LTX2 frame/FPS combination has no registered runtime profile.');
            return;
        }
        var seed = state.seed === -1
            ? Math.floor(Math.random() * 4294967296)
            : state.seed;
        state.lastSeed = seed;
        beginCurrentBatch();
        state.pendingBatch = 1;
        var request = buildVideoRequest(seed);
        var reusable = getParams();
        reusable.seed = seed;
        reusable.arch = ModelUtils.archForModel(reusable.model);
        reusable.width = request.width;
        reusable.height = request.height;
        reusable.steps = request.steps;
        reusable.sampler = request.sampler;
        reusable.scheduler = request.scheduler;
        reusable.frames = request.frames;
        reusable.fps = request.fps;
        if (Number.isFinite(Number(request.guidance))) {
            reusable.cfg = Number(request.guidance);
            reusable.guidance = Number(request.guidance);
        }
        setGenerating(true);
        updateGenerationActivity('Preparing GPU · unloading image model if resident', 0, 0);
        SerenityAPI.postVideo(request)
            .then(function (job) {
            if (job && job.accepted_video_artifact === true && job.mp4_url) {
                var videoId = String(job.video_id || '');
                var metadata = Object.assign({}, reusable, {
                    params: reusable,
                    frame_count: request.frames,
                    fps: request.fps,
                    mode: job.mode ||
                        (request.image_path ? 'i2v_first_frame' : 't2v')
                });
                displayVideo(String(job.mp4_url), metadata);
                addToGallery(String(job.mp4_url), true, metadata);
                if (videoId)
                    state.completedVideoJobs[videoId] = true;
                state.pendingBatch = 0;
                setGenerating(false);
                return;
            }
            if (!job || !(job.video_id || job.prompt_id))
                throw new Error('server did not return a video job id');
            pollVideoGeneration(job, request, reusable);
        })
            .catch(function (error) {
            state.pendingBatch = 0;
            showError('Failed to queue video: ' + error.message);
            setGenerating(false);
        });
    }
    // ── Generate ──
    function generate() {
        if (state.generating)
            return;
        if (!state.model) {
            showError('No model selected');
            return;
        }
        if (!state.prompt.trim()) {
            showError('Enter a prompt');
            return;
        }
        if (ModelUtils.isVideoModel(state.model)) {
            pushPromptHistory(state.prompt);
            generateVideo();
            return;
        }
        // Save to prompt history
        pushPromptHistory(state.prompt);
        beginCurrentBatch();
        var batchN = state.batchCount || 1;
        state.pendingBatch = batchN;
        setGenerating(true);
        var resolvedSeed = state.seed === -1 ? Math.floor(Math.random() * 4294967296) : state.seed;
        state.lastSeed = resolvedSeed;
        // Build and queue all batch workflows upfront
        var originalSeed = state.seed;
        var submissions = [];
        for (var i = 0; i < batchN; i++) {
            state.seed = state.noSeedIncrement
                ? resolvedSeed
                : (resolvedSeed + i) % 4294967296;
            // Bind completion metadata to this immutable submission. Video
            // requests block until the render completes, so reading `state`
            // in their completion handler would mislabel the gallery item if
            // the user selected another model while the render was running.
            submissions.push({
                request: buildGenerateRequest(state.seed),
                prompt: state.prompt,
                model: state.model,
                seed: state.seed,
                steps: state.steps,
                cfg: state.cfg,
                guidance: state.guidance,
                scheduler: state.scheduler,
                width: state.width,
                height: state.height,
                arch: ModelUtils.archForModel(state.model)
            });
        }
        state.seed = originalSeed;
        // Submit in order so "Continue After Errors" has real behavior instead
        // of being a decorative checkbox. The server still owns execution order.
        function submitAt(i) {
            if (i >= submissions.length)
                return;
            var submission = submissions[i];
            if (batchN > 1) {
                els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating ' + (i + 1) + ' / ' + batchN + '...';
                if (typeof lucide !== 'undefined')
                    lucide.createIcons({ nameAttr: 'data-lucide' });
            }
            SerenityAPI.postGenerate(submission.request, {
                prompt: submission.prompt,
                model: submission.model,
                batchLabel: batchN > 1 ? ('(' + (i + 1) + '/' + batchN + ')') : ''
            })
                .then(function (result) {
                if (!result || !result.prompt_id)
                    throw new Error('server did not return a job id');
                state.pendingImageJobs[String(result.prompt_id)] = true;
                submitAt(i + 1);
            })
                .catch(function (err) {
                showError('Failed to queue item ' + (i + 1) + ': ' + err.message);
                if (state.continueAfterErrors)
                    submitAt(i + 1);
                else
                    setGenerating(false);
            });
        }
        submitAt(0);
    }
    // ── State Helpers ──
    function updateGenerationActivity(phase, step, total) {
        if (!els.activityStatus || !els.activityText)
            return;
        var raw = String(phase || '').trim();
        var lower = raw.toLowerCase();
        var label = raw;
        var activityState = 'working';
        if (/queued|waiting.*gpu|gpu.*wait/.test(lower)) {
            label = 'Queued · waiting for GPU';
            activityState = 'waiting';
        }
        else if (/unload|evict|releas.*model/.test(lower)) {
            label = raw || 'Unloading current model';
            activityState = 'loading';
        }
        else if (/load/.test(lower)) {
            label = raw === '' || lower === 'loading'
                ? 'Loading ' + (state.model || 'model')
                : raw;
            activityState = 'loading';
        }
        else if (/complet|done/.test(lower)) {
            label = 'Complete';
            activityState = 'done';
        }
        else if (/sav|mux|writ.*result/.test(lower)) {
            label = 'Saving result';
            activityState = 'saving';
        }
        else if (/decod|vae/.test(lower)) {
            label = ModelUtils.isVideoModel(state.model) ? 'Decoding video' : 'Decoding image';
            activityState = 'decoding';
        }
        else if (/token|encod|condition|caption/.test(lower)) {
            if (/gemma layer|\b\d+\s*\/\s*\d+/.test(lower))
                label = raw;
            else if (/^token/.test(lower))
                label = raw || 'Tokenizing prompt';
            else if (/^encod/.test(lower))
                label = raw || 'Encoding prompt';
            else
                label = /condition/.test(lower)
                    ? 'Encoding prompt conditioning'
                    : 'Tokenizing + encoding prompt';
            activityState = 'encoding';
        }
        else if (/sampl|denois/.test(lower) || Number(step) > 0) {
            label = 'Sampling';
            activityState = 'sampling';
        }
        else if (/prepar|start|admit/.test(lower)) {
            label = raw || 'Preparing generation';
            activityState = 'working';
        }
        else if (!label) {
            label = state.generating ? 'Preparing generation' : 'Idle';
            activityState = state.generating ? 'working' : 'idle';
        }
        if (Number(total) > 0 && Number(step) > 0 &&
            activityState === 'sampling') {
            label += ' · Step ' + Number(step) + ' / ' + Number(total);
        }
        else if (Number(total) > 0 && Number(step) > 0 &&
            activityState === 'encoding' &&
            !new RegExp('(?:^|\\D)' + Number(step) + '\\s*\\/\\s*' + Number(total) + '(?:\\D|$)').test(label)) {
            label += ' · Step ' + Number(step) + ' / ' + Number(total);
        }
        els.activityText.textContent = label;
        els.activityStatus.dataset.state = activityState;
        els.activityStatus.title = raw && raw !== label ? raw : label;
    }
    function setGenerationIdle() {
        if (!els.activityStatus || !els.activityText)
            return;
        els.activityText.textContent = 'Idle';
        els.activityStatus.dataset.state = 'idle';
        els.activityStatus.title = 'No generation is running';
    }
    function renderExternalActivity() {
        var activity = state.externalActivity;
        if (!activity || activity.active !== true) {
            if (!state.generating)
                setGenerationIdle();
            return;
        }
        updateGenerationActivity(
            activity.label || activity.phase || 'Workflow running',
            activity.step || 0,
            activity.total || 0
        );
        if (els.activityText)
            els.activityText.textContent = 'Workflow · ' + els.activityText.textContent;
        if (els.activityStatus)
            els.activityStatus.title = activity.label || activity.phase || 'Workflow running';
    }
    function setExternalActivity(activity) {
        state.externalActivity = activity && activity.active === true
            ? {
                active: true,
                promptId: String(activity.promptId || ''),
                phase: String(activity.phase || ''),
                label: String(activity.label || ''),
                step: Number(activity.step) || 0,
                total: Number(activity.total) || 0
            }
            : null;
        if (initialized && !state.generating)
            renderExternalActivity();
    }
    function setGenerating(v) {
        state.generating = v;
        var isVideo = ModelUtils.isVideoModel(state.model);
        els.btn.disabled = v;
        if (v) {
            els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating...';
        }
        else {
            els.btn.innerHTML = isVideo
                ? '<i data-lucide="wand-2"></i> Generate Video'
                : '<i data-lucide="wand-2"></i> Generate';
        }
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
        els.btn.classList.toggle('generating', v);
        if (els.batchStatus)
            els.batchStatus.textContent = v
                ? ('Running · ' + Math.max(1, state.pendingBatch) + ' queued')
                : 'Idle';
        if (v) {
            updateGenerationActivity('Queued · waiting for GPU', 0, 0);
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
            if (els.leftProgress) {
                els.leftProgress.classList.add('active');
                els.leftProgressBar.style.width = '100%';
            }
        }
        else {
            renderExternalActivity();
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
            if (els.leftProgress) {
                els.leftProgress.classList.remove('active');
                els.leftProgressBar.style.width = '0%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = '';
            }
        }
    }
    function reusableParamsForItem(item) {
        if (!item)
            return null;
        var base = item.params && typeof item.params === 'object'
            ? JSON.parse(JSON.stringify(item.params))
            : {};
        var values = {
            model: item.model,
            prompt: item.prompt,
            negPrompt: item.negPrompt,
            width: item.width,
            height: item.height,
            steps: item.steps,
            cfg: item.cfg,
            guidance: item.guidance,
            sampler: item.sampler,
            scheduler: item.scheduler,
            sigmaShift: item.sigmaShift,
            seed: item.seed,
            frames: item.frame_count,
            fps: item.fps,
            stylePreset: item.stylePreset,
            loras: item.loras,
            videoGuidanceMode: item.videoGuidanceMode,
            videoWorkflowProfile: item.videoWorkflowProfile,
            videoPromptEnhancer: item.videoPromptEnhancer,
            videoQuant: item.videoQuant,
            videoCheckpoint: item.videoCheckpoint,
            capsPositive: item.capsPositive,
            capsNegative: item.capsNegative,
            noiseFixture: item.noiseFixture,
            includeAudio: item.includeAudio,
            audioPolicy: item.audioPolicy,
            postUpscaler: item.postUpscaler,
            postUpscaleFactor: item.postUpscaleFactor,
            noSeedIncrement: item.noSeedIncrement,
            continueAfterErrors: item.continueAfterErrors,
            personalNote: item.personalNote
        };
        Object.keys(values).forEach(function (key) {
            if (values[key] != null && values[key] !== '' && base[key] == null)
                base[key] = values[key];
        });
        return Object.keys(base).length ? base : null;
    }
    function displayImage(src, metadata) {
        state.currentImage = src;
        state.currentIsVideo = false;
        state.currentResultParams = reusableParamsForItem(metadata);
        els.previewImg.src = src;
        els.previewImg.style.display = 'block';
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
    }
    function displayVideo(src, metadata) {
        state.currentImage = src;
        state.currentIsVideo = true;
        state.currentVideoFrames = Number(
            metadata && (metadata.frame_count != null ? metadata.frame_count : metadata.frames)
        ) || Number(state.frames) || null;
        state.currentVideoFps = Number(metadata && metadata.fps) || Number(state.fps) || null;
        state.currentResultParams = reusableParamsForItem(metadata);
        els.previewVideo.pause();
        els.previewVideo.src = src;
        els.previewVideo.muted = true;
        els.previewVideo.loop = true;
        els.previewVideo.controls = true;
        els.previewVideo.playsInline = true;
        els.previewVideo.load();
        els.previewVideo.style.display = 'block';
        els.previewImg.style.display = 'none';
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
        var playback = els.previewVideo.play();
        if (playback && typeof playback.catch === 'function')
            playback.catch(function () { /* controls remain available if autoplay is blocked */ });
    }
    function clearPreview() {
        state.currentImage = null;
        state.currentIsVideo = false;
        state.currentVideoFrames = null;
        state.currentVideoFps = null;
        state.currentResultParams = null;
        state.currentGalleryIndex = -1;
        els.previewImg.style.display = 'none';
        els.previewImg.removeAttribute('src');
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.previewVideo.removeAttribute('src');
        els.actionBar.style.display = 'none';
        els.empty.style.display = 'flex';
        // Deselect gallery thumbs
        state.selectedImages = [];
        state.lastSelectedIndex = -1;
        updateSelectionUI();
        updateMetadataPanel();
        setPreviewModelBadges(state.model, state.arch);
    }
    function mediaArtifactKey(src) {
        if (!src)
            return '';
        try {
            var url = new URL(String(src), window.location.origin);
            var filename = '';
            var subfolder = '';
            if (url.pathname === '/view') {
                filename = url.searchParams.get('filename') || '';
                subfolder = url.searchParams.get('subfolder') || '';
            }
            else {
                var parts = url.pathname.split('/').filter(Boolean);
                if (parts.length >= 3 && parts[0] === 'out') {
                    subfolder = parts[1];
                    filename = parts.slice(2).join('/');
                }
            }
            if (!filename)
                return '';
            return 'output:' + subfolder.replace(/^\/+|\/+$/g, '') + '/' +
                filename.replace(/^\/+/, '');
        }
        catch (error) {
            return '';
        }
    }
    function galleryItemKey(item) {
        if (!item || !item.src)
            return '';
        return mediaArtifactKey(item.src) || 'media:' + String(item.src);
    }
    function deleteHistoryArtifact(item) {
        return fetch('/v1/history/artifacts', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: String(item && item.historyId || ''),
                url: String(item && item.src || '')
            })
        }).then(function (response) {
            return response.text().then(function (body) {
                var result = {};
                try { result = JSON.parse(body); }
                catch (error) { result = { detail: body }; }
                if (!response.ok || result.deleted !== true)
                    throw new Error(result.detail || ('HTTP ' + response.status));
                return result;
            });
        });
    }
    function beginCurrentBatch() {
        state.currentBatchKeys = {};
        renderGallery();
    }
    function addToGallery(src, isVideo, metadata) {
        function value(key, fallback) {
            return metadata && Object.prototype.hasOwnProperty.call(metadata, key)
                ? metadata[key]
                : fallback;
        }
        var item = {
            src: src,
            isVideo: !!isVideo,
            prompt: value('prompt', state.prompt || ''),
            model: value('model', state.model || ''),
            seed: value('seed', state.lastSeed),
            steps: value('steps', state.steps),
            cfg: value('cfg', state.cfg),
            guidance: value('guidance', state.guidance),
            sampler: value('sampler', state.sampler),
            scheduler: value('scheduler', state.scheduler),
            width: value('width', state.width),
            height: value('height', state.height),
            frame_count: value('frame_count', state.currentVideoFrames),
            fps: value('fps', state.currentVideoFps),
            arch: value('arch', state.arch),
            negPrompt: value('negPrompt', value('negative', '')),
            stylePreset: value('stylePreset', 'none'),
            loras: value('loras', []),
            videoGuidanceMode: value('videoGuidanceMode', ''),
            videoWorkflowProfile: value('videoWorkflowProfile',
                value('workflow_profile', '')),
            videoPromptEnhancer: value('videoPromptEnhancer',
                value('prompt_enhancer', 'none')),
            videoQuant: value('videoQuant', ''),
            videoCheckpoint: value('videoCheckpoint', ''),
            capsPositive: value('capsPositive', ''),
            capsNegative: value('capsNegative', ''),
            noiseFixture: value('noiseFixture', ''),
            includeAudio: value('includeAudio', false),
            audioPolicy: value('audioPolicy',
                value('audio_policy', value('includeAudio', false) ? 'generate' : 'none')),
            postUpscaler: value('postUpscaler', state.postUpscaler),
            postUpscaleFactor: value('postUpscaleFactor', state.postUpscaleFactor),
            sigmaShift: value('sigmaShift', value('sigma_shift', state.sigmaShift)),
            noSeedIncrement: value('noSeedIncrement', state.noSeedIncrement),
            continueAfterErrors: value('continueAfterErrors', state.continueAfterErrors),
            personalNote: value('personalNote', state.personalNote),
            params: metadata && metadata.params
                ? JSON.parse(JSON.stringify(metadata.params))
                : null,
            timestamp: Number(value('timestamp', Date.now())) || Date.now(),
            starred: false
        };
        var artifactKey = mediaArtifactKey(item.src);
        if (artifactKey) {
            var existingIndex = state.gallery.findIndex(function (existing) {
                return mediaArtifactKey(existing.src) === artifactKey;
            });
            if (existingIndex >= 0) {
                if (!state.gallery[existingIndex].params && item.params)
                    state.gallery[existingIndex] = item;
                if (state.generating || state.pendingBatch > 0)
                    state.currentBatchKeys[galleryItemKey(state.gallery[existingIndex])] = true;
                renderGallery();
                return;
            }
        }
        state.gallery.unshift(item);
        if (state.generating || state.pendingBatch > 0)
            state.currentBatchKeys[galleryItemKey(item)] = true;
        renderGallery();
        // Auto-switch to new image
        if (state.autoSwitchNew) {
            state.selectedImages = [0];
            state.lastSelectedIndex = 0;
            updateSelectionUI();
            updateMetadataPanel();
            setPreviewModelBadges(item.model, item.arch);
            state.currentGalleryIndex = 0;
            state.currentResultParams = reusableParamsForItem(item);
        }
    }
    function createThumb(item, galleryIndex) {
        var src = item.src;
        var isVideo = item.isVideo;
        var starred = item.starred;
        var isSelected = state.selectedImages.indexOf(galleryIndex) >= 0;
        var wrap = document.createElement('div');
        wrap.className = 'gen-thumb-wrap' + (isSelected ? ' gen-selected' : '') + (isVideo ? ' gen-thumb-video' : '');
        wrap.dataset.galleryIndex = String(galleryIndex);
        var starClass = starred ? ' starred' : '';
        var starChar = starred ? '\u2605' : '\u2606';
        if (isVideo) {
            wrap.innerHTML =
                '<button class="gen-thumb-star' + starClass + '" data-idx="' + galleryIndex + '">' + starChar + '</button>' +
                    '<video src="' + src + '#t=0.1" muted preload="metadata" style="width:100%;height:100%;object-fit:cover;display:block;"></video>' +
                    '<div class="gen-thumb-play">\u25b6</div>';
        }
        else {
            wrap.innerHTML =
                '<button class="gen-thumb-star' + starClass + '" data-idx="' + galleryIndex + '">' + starChar + '</button>' +
                    '<img src="' + src + '" alt="thumbnail" loading="lazy">' +
                    '<div class="gen-thumb-overlay">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>' +
                    '</div>';
        }
        var media = wrap.querySelector(isVideo ? 'video' : 'img');
        if (media) {
            media.addEventListener('error', function () {
                if (wrap.querySelector('.gen-thumb-unavailable-label'))
                    return;
                media.style.display = 'none';
                wrap.classList.add('gen-thumb-unavailable');
                var label = document.createElement('span');
                label.className = 'gen-thumb-unavailable-label';
                label.textContent = 'Output unavailable';
                wrap.appendChild(label);
            });
        }
        // Star click
        wrap.querySelector('.gen-thumb-star').addEventListener('click', function (e) {
            e.stopPropagation();
            var idx = parseInt(this.dataset.idx);
            state.gallery[idx].starred = !state.gallery[idx].starred;
            renderGallery();
        });
        // Click with multi-select support
        wrap.addEventListener('click', function (e) {
            handleThumbClick(galleryIndex, e);
        });
        // Right-click context menu
        wrap.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            showContextMenu(e, galleryIndex);
        });
        return wrap;
    }
    function galleryMetadataFromJob(job) {
        var meta = (job && job.metadata) || {};
        var params = meta.params || (job && job.params) || {};
        var model = (job && job.model) || meta.model || params.model || '';
        var arch = params.arch || ModelUtils.archForModel(model) || '';
        var executedCfg = meta.cfg != null ? meta.cfg : params.cfg;
        var reusable = {
            model: model,
            prompt: meta.prompt != null ? meta.prompt : (params.prompt || ''),
            negPrompt: params.negative || params.negPrompt || '',
            width: meta.width != null ? meta.width : params.width,
            height: meta.height != null ? meta.height : params.height,
            steps: meta.steps != null ? meta.steps : params.steps,
            cfg: executedCfg,
            guidance: arch === 'flux' ? executedCfg : params.guidance,
            sampler: meta.sampler || params.sampler || '',
            scheduler: meta.scheduler || params.scheduler || '',
            sigmaShift: params.sigma_shift,
            seed: meta.seed != null ? meta.seed : params.seed,
            variationSeed: params.variation_seed,
            variationStrength: params.variation_strength,
            stylePreset: meta.stylePreset || 'none',
            loras: Array.isArray(params.lora) ? params.lora.map(function (row) {
                return {
                    name: row.name || '',
                    strength: Number(row.weight == null ? 1 : row.weight),
                    enabled: true,
                    role: row.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
            }) : []
        };
        return {
            prompt: meta.prompt != null ? meta.prompt : (params.prompt || ''),
            model: model,
            seed: meta.seed != null ? meta.seed : params.seed,
            steps: meta.steps != null ? meta.steps : params.steps,
            cfg: executedCfg,
            // FLUX lowers its distilled guidance into JobParams.cfg. Preserve
            // that executed value under the UI's Guidance label too.
            guidance: arch === 'flux' ? executedCfg : params.guidance,
            sampler: meta.sampler || params.sampler || '',
            scheduler: meta.scheduler || params.scheduler || '',
            sigmaShift: reusable.sigmaShift,
            width: meta.width != null ? meta.width : params.width,
            height: meta.height != null ? meta.height : params.height,
            frame_count: meta.frame_count != null ? meta.frame_count : params.frames,
            fps: meta.fps != null ? meta.fps : params.fps,
            arch: arch,
            negPrompt: reusable.negPrompt,
            stylePreset: reusable.stylePreset,
            loras: reusable.loras,
            timestamp: Date.parse(job && job.created || '') || Date.now(),
            params: reusable
        };
    }
    function refreshHistory() {
        fetch('/v1/history/artifacts', { cache: 'no-store' })
            .then(function (response) {
            if (!response.ok)
                throw new Error('HTTP ' + response.status);
            return response.json();
        })
            .then(function (history) {
            var byKey = {};
            state.gallery.forEach(function (item) {
                var key = galleryItemKey(item);
                if (key)
                    byKey[key] = item;
            });
            var discovered = [];
            var artifacts = history && Array.isArray(history.items)
                ? history.items : [];
            artifacts.forEach(function (artifact) {
                var src = String(artifact && artifact.url || '');
                if (!src || !/\.(png|jpe?g|webp|gif|mp4|webm)$/i.test(src))
                    return;
                var key = mediaArtifactKey(src) || ('media:' + src);
                var params = artifact.params || {};
                var timestamp = Number(artifact.timestamp) || 0;
                var metadata = galleryMetadataFromJob({
                    model: params.model || '',
                    params: params,
                    created: timestamp ? new Date(timestamp).toISOString() : ''
                });
                metadata.timestamp = timestamp || metadata.timestamp;
                var isVideo = artifact.media_type === 'video' ||
                    /\.(gif|mp4|webm)$/i.test(src);
                var existing = byKey[key];
                var item = Object.assign({
                    src: src,
                    isVideo: isVideo,
                    historyId: String(artifact.id || ''),
                    starred: existing && existing.starred === true
                }, metadata);
                discovered.push(item);
            });
            // The server inventory is authoritative. In particular, discard
            // stale /out/job-* aliases created when a benchmark directory was
            // incorrectly used as the product output root.
            state.gallery = discovered;
            return state.gallery;
        })
            .then(function (items) {
            state.gallery = items.sort(function (left, right) {
                return (Number(right.timestamp) || 0) -
                    (Number(left.timestamp) || 0);
            });
            renderGallery();
        })
            .catch(function (error) {
            console.error('History output sync failed: ' + error.message);
        });
    }
    function loadHistory() {
        state.gallery = [];
        renderGallery();
        refreshHistory();
        // Restore thumb size
        try {
            var ts = parseInt(localStorage.getItem('sf-thumb-size'));
            if (ts >= 45 && ts <= 200) {
                state.thumbSize = ts;
                applyThumbSize();
            }
        }
        catch (e) { /* ignore */ }
    }
    // ── Prompt History ──
    function loadPromptHistory() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-prompt-history'));
            if (saved && Array.isArray(saved)) {
                state.promptHistory = saved.slice(0, 20);
            }
        }
        catch (e) { /* ignore */ }
    }
    function savePromptHistory() {
        try {
            localStorage.setItem('sf-prompt-history', JSON.stringify(state.promptHistory));
        }
        catch (e) { /* quota exceeded */ }
    }
    function pushPromptHistory(prompt) {
        if (!prompt || !prompt.trim())
            return;
        var trimmed = prompt.trim();
        // Don't duplicate the last entry
        if (state.promptHistory.length > 0 && state.promptHistory[state.promptHistory.length - 1] === trimmed)
            return;
        state.promptHistory.push(trimmed);
        if (state.promptHistory.length > 20)
            state.promptHistory.shift();
        state.promptHistoryIndex = -1;
        savePromptHistory();
    }
    function navigatePromptHistory(direction) {
        if (state.promptHistory.length === 0)
            return;
        if (direction < 0) {
            // Up = older
            if (state.promptHistoryIndex === -1) {
                // Save current prompt as temp
                state._tempPrompt = state.prompt;
                state.promptHistoryIndex = state.promptHistory.length - 1;
            }
            else if (state.promptHistoryIndex > 0) {
                state.promptHistoryIndex--;
            }
        }
        else {
            // Down = newer
            if (state.promptHistoryIndex === -1)
                return;
            if (state.promptHistoryIndex < state.promptHistory.length - 1) {
                state.promptHistoryIndex++;
            }
            else {
                // Back to current
                state.promptHistoryIndex = -1;
                state.prompt = state._tempPrompt || '';
                if (els.prompt) {
                    els.prompt.value = state.prompt;
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = els.prompt.scrollHeight + 'px';
                }
                updateTokenCount();
                return;
            }
        }
        if (state.promptHistoryIndex >= 0 && state.promptHistoryIndex < state.promptHistory.length) {
            state.prompt = state.promptHistory[state.promptHistoryIndex];
            if (els.prompt) {
                els.prompt.value = state.prompt;
                els.prompt.style.height = 'auto';
                els.prompt.style.height = els.prompt.scrollHeight + 'px';
            }
            updateTokenCount();
        }
    }
    // ── Style Preview ──
    function updateStylePreview() {
        var el = document.getElementById('gen-style-preview');
        if (!el)
            return;
        if (state.stylePreset === 'none' || !state.stylePreset) {
            el.style.display = 'none';
            return;
        }
        var suffix = '';
        for (var i = 0; i < stylePresets.length; i++) {
            if (stylePresets[i].id === state.stylePreset) {
                suffix = stylePresets[i].suffix;
                break;
            }
        }
        if (suffix) {
            var preview = state.prompt ? state.prompt + suffix : '(your prompt)' + suffix;
            el.textContent = 'Full prompt: ' + preview;
            el.style.display = 'block';
        }
        else {
            el.style.display = 'none';
        }
    }
    // ── Searchable Model Picker ──
    function bindModelPicker() {
        if (!els.modelSearch)
            return;
        els.modelSearch.addEventListener('focus', function () {
            this.select();
            openModelDropdown();
        });
        els.modelSearch.addEventListener('input', function () {
            state.modelSearchQuery = this.value.toLowerCase();
            renderModelDropdown();
            if (els.modelDropdown)
                els.modelDropdown.style.display = '';
        });
        els.modelSearch.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                closeModelDropdown();
                this.blur();
            }
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (els.modelPickerWrap && !els.modelPickerWrap.contains(e.target)) {
                closeModelDropdown();
            }
        });
    }
    function openModelDropdown() {
        if (!els.modelDropdown)
            return;
        state.modelSearchQuery = '';
        renderModelDropdown();
        els.modelDropdown.style.display = '';
    }
    function closeModelDropdown() {
        if (els.modelDropdown)
            els.modelDropdown.style.display = 'none';
        // Restore display name
        if (els.modelSearch && state.model) {
            els.modelSearch.value = state.model;
        }
    }
    function selectModel(name) {
        state.model = name;
        els.model.value = name;
        if (els.modelSearch)
            els.modelSearch.value = name;
        closeModelDropdown();
        var checkpoint = name.replace(/\.safetensors$/i, '');
        if (ModelUtils.archForModel(name) === 'ltxv') {
            var normalizedCheckpoint = checkpoint.toLowerCase();
            if (/dequant-bf16|bf16/.test(normalizedCheckpoint))
                state.videoQuant = 'bf16';
            else if (/fp8/.test(normalizedCheckpoint) ||
                normalizedCheckpoint === 'ltx-2.3-22b-dev' ||
                normalizedCheckpoint === 'ltx-2.3-22b-distilled')
                state.videoQuant = 'fp8';
            state.videoCheckpoint = checkpoint;
            var creatorWorkflow = ltx2CheckpointWorkflow(checkpoint);
            state.videoWorkflowProfile = creatorWorkflow
                ? String(creatorWorkflow.id || '') : '';
            if (creatorWorkflow) {
                state.videoGuidanceMode = String(
                    creatorWorkflow.guidance_mode || 'distilled'
                );
            }
            else {
                state.videoGuidanceMode = /distill/i.test(checkpoint)
                    ? 'distilled' : 'dev';
            }
            if (els.videoQuant)
                els.videoQuant.value = state.videoQuant;
            if (els.videoCheckpoint)
                els.videoCheckpoint.value = checkpoint;
        }
        updateUIForArch(ModelUtils.archForModel(name));
        var globalBadge = document.querySelector('#topbar .model-badge');
        if (globalBadge) {
            globalBadge.textContent = name;
            globalBadge.title = name;
        }
    }
    function renderModelDropdown() {
        if (!els.modelDropdownList)
            return;
        var models = state.allModels || [];
        var query = state.modelSearchQuery || '';
        var filtered = models.filter(function (m) {
            return !query || m.name.toLowerCase().indexOf(query) >= 0;
        });
        // Group by architecture
        var groups = {};
        var groupOrder = ['krea2', 'klein', 'flux2', 'flux', 'ideogram4', 'lens', 'sdxl', 'sd3', 'zimage', 'qwen', 'hunyuan', 'sd15', 'ltxv', 'minimax_h3', 'wan', 'bernini', 'scail2', 'other'];
        var groupLabels = { krea2: 'KREA2', klein: 'KLEIN', flux2: 'FLUX 2', flux: 'FLUX', ideogram4: 'IDEOGRAM 4', lens: 'MICROSOFT LENS', sdxl: 'SDXL', sd3: 'SD3', zimage: 'Z-IMAGE', qwen: 'QWEN', hunyuan: 'HUNYUAN', sd15: 'SD 1.5', ltxv: 'Video (LTX)', minimax_h3: 'Video (MiniMax-H3)', wan: 'Video (WAN)', bernini: 'Video (BERNINI-R)', scail2: 'Video (SCAIL-2)', other: 'Other' };
        filtered.forEach(function (m) {
            var arch = ModelUtils.archForModel(m.name) || 'other';
            if (!groups[arch])
                groups[arch] = [];
            groups[arch].push(m);
        });
        var html = '';
        if (filtered.length === 0) {
            html = '<div class="gen-model-dropdown-empty">No models match</div>';
        }
        else {
            // Any arch not in groupOrder still renders (never silently drop a model)
            Object.keys(groups).forEach(function (arch) {
                if (groupOrder.indexOf(arch) < 0)
                    groupOrder.push(arch);
            });
            groupOrder.forEach(function (arch) {
                if (!groups[arch] || groups[arch].length === 0)
                    return;
                html += '<div class="gen-model-dropdown-group">' + (groupLabels[arch] || arch.toUpperCase()) + '</div>';
                groups[arch].forEach(function (m) {
                    var archVal = ModelUtils.archForModel(m.name) || 'other';
                    html += '<div class="gen-model-dropdown-item" data-model="' + m.name + '">' +
                        '<span class="gen-model-dropdown-name">' + m.name + '</span>' +
                        '<span class="gen-arch-badge" data-arch="' + archVal + '">' + (groupLabels[archVal] || archVal.toUpperCase()) + '</span>' +
                        '</div>';
                });
            });
        }
        els.modelDropdownList.innerHTML = html;
        // Bind click
        els.modelDropdownList.querySelectorAll('.gen-model-dropdown-item').forEach(function (item) {
            item.addEventListener('click', function () {
                selectModel(this.dataset.model);
            });
        });
    }
    // ── Prompt Attention Weight (Ctrl+Up/Down) ──
    function handlePromptWeight(textarea, increase) {
        var start = textarea.selectionStart;
        var end = textarea.selectionEnd;
        if (start === end)
            return; // No selection
        var text = textarea.value;
        var selected = text.substring(start, end);
        var newText;
        var newStart, newEnd;
        if (increase) {
            // Check if already weighted with explicit value: (word:N.N)
            var weightMatch = selected.match(/^\((.+):(\d+\.?\d*)\)$/);
            if (weightMatch) {
                var newWeight = (parseFloat(weightMatch[2]) + 0.1).toFixed(1);
                newText = '(' + weightMatch[1] + ':' + newWeight + ')';
            }
            else if (/^\(+[^)]+\)+$/.test(selected)) {
                // Already in parens, add another layer
                newText = '(' + selected + ')';
            }
            else {
                // Wrap in parens
                newText = '(' + selected + ')';
            }
        }
        else {
            // Decrease
            var weightMatch2 = selected.match(/^\((.+):(\d+\.?\d*)\)$/);
            if (weightMatch2) {
                var decreased = (parseFloat(weightMatch2[2]) - 0.1).toFixed(1);
                if (parseFloat(decreased) <= 1.0) {
                    newText = '(' + weightMatch2[1] + ')';
                }
                else {
                    newText = '(' + weightMatch2[1] + ':' + decreased + ')';
                }
            }
            else if (/^\(\((.+)\)\)$/.test(selected)) {
                // Remove one layer of parens
                newText = selected.substring(1, selected.length - 1);
            }
            else if (/^\((.+)\)$/.test(selected)) {
                // Remove parens entirely
                newText = selected.substring(1, selected.length - 1);
            }
            else {
                return; // Nothing to decrease
            }
        }
        textarea.value = text.substring(0, start) + newText + text.substring(end);
        state.prompt = textarea.value;
        // Restore selection
        newStart = start;
        newEnd = start + newText.length;
        textarea.setSelectionRange(newStart, newEnd);
        updateTokenCount();
    }
    // ── Error Display ──
    function showError(msg) {
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function () {
            els.errorBanner.classList.remove('visible');
        }, 5000);
    }
    // ── WebSocket (via shared SerenityWS) ──
    function connectWS() {
        SerenityWS.on('connected', function () {
            els.wsIndicator.classList.remove('visible');
        });
        SerenityWS.on('disconnected', function () {
            els.wsIndicator.classList.add('visible');
        });
        SerenityWS.on('execution_start', function (data) {
            var startPromptId = String(data && data.prompt_id || '');
            if (!startPromptId ||
                (!state.pendingImageJobs[startPromptId] &&
                    !state.pendingVideoJobs[startPromptId]))
                return;
            setGenerating(true);
        });
        SerenityWS.on('progress', function (data) {
            if (!data)
                return;
            var progressPromptId = String(data.prompt_id || '');
            if (progressPromptId &&
                !state.pendingImageJobs[progressPromptId] &&
                !state.pendingVideoJobs[progressPromptId])
                return;
            var pct = data.max > 0 ? (data.value / data.max * 100).toFixed(0) : '0';
            els.progressBar.style.width = pct + '%';
            var phase = data.phase || data.message || '';
            updateGenerationActivity(phase, data.value, data.max);
            els.progressLabel.textContent = (phase ? phase + ' · ' : '') +
                'Step ' + data.value + ' / ' + data.max;
            els.progressLabel.classList.add('visible');
            if (els.leftProgressBar) {
                els.leftProgressBar.style.width = pct + '%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = els.progressLabel.textContent;
            }
        });
        SerenityWS.on('preview', function (data) {
            if (!data || !data.blob || !state.generating)
                return;
            var previewPromptId = String(data.prompt_id || '');
            if (previewPromptId &&
                !state.pendingImageJobs[previewPromptId] &&
                !state.pendingVideoJobs[previewPromptId])
                return;
            var url = URL.createObjectURL(data.blob);
            if (els.previewImg) {
                // Revoke old preview URL to prevent memory leaks
                if (els.previewImg._previewUrl)
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                els.previewImg._previewUrl = url;
                els.previewImg.src = url;
                els.previewImg.style.display = 'block';
                if (els.empty)
                    els.empty.style.display = 'none';
                els.previewImg.classList.add('gen-preview-live');
            }
        });
        SerenityWS.on('executed', function (data) {
            if (!data || !data.output)
                return;
            var eventPromptId = String(data.prompt_id || '');
            // LTX2 is completed by HTTP status/result polling. Its WorkerEvent::Done
            // also becomes a Comfy-style `executed` WebSocket event. Whichever
            // arrives first owns gallery insertion; suppress both the in-flight
            // and late-event cases so one render produces one gallery movie.
            if (eventPromptId &&
                (state.pendingVideoJobs[eventPromptId] ||
                    state.completedVideoJobs[eventPromptId]))
                return;
            // This server WebSocket is shared by Canvas, diagnostics, and other
            // API clients. Only a prompt id submitted by this Generate tab may
            // replace its preview or enter its Current Batch gallery.
            if (!eventPromptId || !state.pendingImageJobs[eventPromptId])
                return;
            delete state.pendingImageJobs[eventPromptId];
            state.completedImageJobs[eventPromptId] = true;
            var completedImageIds = Object.keys(state.completedImageJobs);
            if (completedImageIds.length > 256)
                delete state.completedImageJobs[completedImageIds[0]];
            // Clean up live preview state
            if (els.previewImg) {
                els.previewImg.classList.remove('gen-preview-live');
                if (els.previewImg._previewUrl) {
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                    els.previewImg._previewUrl = undefined;
                }
            }
            // Image outputs
            var out = data.output.ui || data.output;
            var items = out.images;
            var isVideo = false;
            // Video outputs (SaveAnimatedWEBP, SaveVideo)
            if (!items && out.videos) {
                items = out.videos;
                isVideo = true;
            }
            if (!items || !items.length)
                return;
            var file = items[0];
            var src = '/view?filename=' + encodeURIComponent(file.filename) +
                '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                '&type=' + encodeURIComponent(file.type || 'output');
            // Also detect video from filename extension
            if (!isVideo)
                isVideo = /\.(webp|mp4|gif)$/i.test(file.filename);
            if (isVideo) {
                displayVideo(src);
            }
            else {
                displayImage(src);
            }
            // The selected model can change while a queued job is running. Bind
            // gallery metadata to the completed job record, never mutable UI state.
            // /v1/job/:id exposes the PNG-derived params captured at Done-time.
            var promptId = data.prompt_id || '';
            fetch('/v1/job/' + encodeURIComponent(promptId))
                .then(function (resp) {
                if (!resp.ok)
                    throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
                .then(function (job) {
                addToGallery(src, isVideo, galleryMetadataFromJob(job));
            })
                .catch(function (err) {
                console.error('Completed job metadata unavailable for ' + promptId + ': ' + err.message);
                addToGallery(src, isVideo, {
                    prompt: '',
                    model: '',
                    seed: null,
                    steps: null,
                    cfg: null,
                    scheduler: '',
                    width: null,
                    height: null,
                    arch: ''
                });
            })
                .finally(function () {
                state.pendingBatch = Math.max(0, state.pendingBatch - 1);
                if (state.pendingBatch <= 0) {
                    setGenerating(false);
                }
            });
        });
        SerenityWS.on('execution_error', function (data) {
            var errorPromptId = String(data && data.prompt_id || '');
            if (errorPromptId &&
                !state.pendingImageJobs[errorPromptId] &&
                !state.pendingVideoJobs[errorPromptId])
                return;
            if (errorPromptId)
                delete state.pendingImageJobs[errorPromptId];
            state.pendingBatch = 0;
            var errMsg = (data && data.exception_message) || 'Generation failed';
            showError(errMsg);
            setGenerating(false);
        });
    }
    // ── Phase 2: Gallery Rendering ──
    function getFilteredGallery() {
        // Completed artifacts belong in persistent History immediately. The
        // Current Batch strip is a convenient second view, not a holding area
        // that makes the gallery appear empty after generation completes.
        var items = state.gallery.slice();
        // Search filter
        if (state.gallerySearch) {
            var q = state.gallerySearch.toLowerCase();
            items = items.filter(function (item) {
                return (item.prompt && item.prompt.toLowerCase().indexOf(q) >= 0);
            });
        }
        // Sort from timestamps rather than insertion order so restored and
        // server-discovered outputs land under the correct date.
        items.sort(function (left, right) {
            var delta = (Number(right.timestamp) || 0) -
                (Number(left.timestamp) || 0);
            return state.sortNewestFirst ? delta : -delta;
        });
        // Starred first
        if (state.starredFirst) {
            var starred = items.filter(function (i) { return i.starred; });
            var unstarred = items.filter(function (i) { return !i.starred; });
            items = starred.concat(unstarred);
        }
        return items;
    }
    function getGalleryIndexMap() {
        // Maps filtered display position to actual gallery index
        var map = [];
        var filtered = getFilteredGallery();
        filtered.forEach(function (item) {
            map.push(state.gallery.indexOf(item));
        });
        return map;
    }
    function galleryDateInfo(item) {
        var timestamp = Number(item && item.timestamp) || 0;
        if (timestamp <= 0)
            return { key: 'unknown', label: 'Date unavailable' };
        var date = new Date(timestamp);
        if (Number.isNaN(date.getTime()))
            return { key: 'unknown', label: 'Date unavailable' };
        var key = date.getFullYear() + '-' +
            String(date.getMonth() + 1).padStart(2, '0') + '-' +
            String(date.getDate()).padStart(2, '0');
        var today = new Date();
        var todayStart = new Date(
            today.getFullYear(), today.getMonth(), today.getDate()).getTime();
        var dateStart = new Date(
            date.getFullYear(), date.getMonth(), date.getDate()).getTime();
        var dayDelta = Math.round((todayStart - dateStart) / 86400000);
        var label = dayDelta === 0
            ? 'Today'
            : (dayDelta === 1
                ? 'Yesterday'
                : date.toLocaleDateString(undefined, {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                }));
        return { key: key, label: label };
    }
    function renderGallery() {
        if (!els.galleryGrid)
            return;
        var filtered = getFilteredGallery();
        var indexMap = getGalleryIndexMap();
        els.galleryGrid.innerHTML = '';
        var groups = {};
        filtered.forEach(function (item, i) {
            var date = galleryDateInfo(item);
            var group = groups[date.key];
            if (!group) {
                var section = document.createElement('section');
                section.className = 'gen-gallery-date-group';
                var heading = document.createElement('h3');
                heading.className = 'gen-gallery-date-heading';
                heading.textContent = date.label;
                var items = document.createElement('div');
                items.className = 'gen-gallery-date-items';
                section.appendChild(heading);
                section.appendChild(items);
                els.galleryGrid.appendChild(section);
                group = groups[date.key] = items;
            }
            group.appendChild(createThumb(item, indexMap[i]));
        });
        applyThumbSize();
        renderBatchStrip();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }
    function renderBatchStrip() {
        if (!els.batchStrip)
            return;
        els.batchStrip.innerHTML = '';
        var recent = state.gallery.filter(function (item) {
            return !!state.currentBatchKeys[galleryItemKey(item)];
        }).slice(0, 12);
        if (!recent.length) {
            els.batchStrip.innerHTML = '<div class="gen-workspace-batch-empty">New results appear here</div>';
            return;
        }
        recent.forEach(function (item) {
            var button = document.createElement('button');
            button.type = 'button';
            button.className = 'gen-workspace-batch-thumb' + (item.isVideo ? ' gen-thumb-video' : '');
            if (item.isVideo) {
                var video = document.createElement('video');
                video.src = item.src + '#t=0.1';
                video.muted = true;
                video.playsInline = true;
                video.preload = 'metadata';
                button.appendChild(video);
            }
            else {
                var image = document.createElement('img');
                image.src = item.src;
                image.alt = item.prompt || 'Generated image';
                button.appendChild(image);
            }
            var caption = document.createElement('span');
            caption.textContent = (item.width && item.height)
                ? (item.width + '×' + item.height + (item.isVideo ? ' · Video' : ''))
                : (item.isVideo ? 'Video' : 'Image');
            button.appendChild(caption);
            button.addEventListener('click', function () {
                var index = state.gallery.indexOf(item);
                if (index >= 0)
                    handleThumbClick(index, {});
            });
            els.batchStrip.appendChild(button);
        });
    }
    function applyThumbSize() {
        if (!els.galleryGrid)
            return;
        els.galleryGrid.style.setProperty(
            '--gen-history-thumb-size', state.thumbSize + 'px');
    }
    // ── Phase 2: Multi-select ──
    function handleThumbClick(galleryIndex, e) {
        var item = state.gallery[galleryIndex];
        if (!item)
            return;
        if (e.ctrlKey || e.metaKey) {
            // Toggle this item in selection
            var pos = state.selectedImages.indexOf(galleryIndex);
            if (pos >= 0) {
                state.selectedImages.splice(pos, 1);
            }
            else {
                state.selectedImages.push(galleryIndex);
            }
            state.lastSelectedIndex = galleryIndex;
        }
        else if (e.shiftKey && state.lastSelectedIndex >= 0) {
            // Range select
            var start = Math.min(state.lastSelectedIndex, galleryIndex);
            var end = Math.max(state.lastSelectedIndex, galleryIndex);
            state.selectedImages = [];
            for (var i = start; i <= end; i++) {
                if (state.selectedImages.indexOf(i) < 0) {
                    state.selectedImages.push(i);
                }
            }
        }
        else {
            // Normal click: select only this
            state.selectedImages = [galleryIndex];
            state.lastSelectedIndex = galleryIndex;
        }
        // Display the clicked item in preview
        if (item.isVideo) {
            displayVideo(item.src, item);
        }
        else {
            displayImage(item.src, item);
        }
        state.currentGalleryIndex = galleryIndex;
        state.currentResultParams = reusableParamsForItem(item);
        setPreviewModelBadges(item.model, item.arch);
        updateSelectionUI();
        updateMetadataPanel();
    }
    function updateSelectionUI() {
        // Update thumb selection classes
        els.galleryGrid.querySelectorAll('.gen-thumb-wrap').forEach(function (wrap) {
            var idx = parseInt(wrap.dataset.galleryIndex);
            wrap.classList.toggle('gen-selected', state.selectedImages.indexOf(idx) >= 0);
            wrap.classList.remove('active'); // remove old style
        });
        // Selection badge
        var badge = document.getElementById('gen-selection-badge');
        if (badge) {
            if (state.selectedImages.length > 1) {
                badge.textContent = state.selectedImages.length + ' selected';
                badge.classList.add('visible');
            }
            else {
                badge.classList.remove('visible');
            }
        }
        // Bulk action bar
        var bulkBar = document.getElementById('gen-bulk-bar');
        if (bulkBar) {
            bulkBar.classList.toggle('visible', state.selectedImages.length > 1);
        }
    }
    // ── Phase 2: Metadata Panel ──
    function updateMetadataPanel() {
        var panel = document.getElementById('gen-metadata-panel');
        var summary = document.getElementById('gen-metadata-summary');
        var full = document.getElementById('gen-metadata-full');
        if (!panel || !summary || !full)
            return;
        // Only show for single selection
        if (state.selectedImages.length !== 1) {
            panel.classList.remove('visible');
            return;
        }
        var item = state.gallery[state.selectedImages[0]];
        if (!item || (!item.prompt && !item.model && !item.seed)) {
            panel.classList.remove('visible');
            return;
        }
        var truncPrompt = item.prompt ? (item.prompt.length > 100 ? item.prompt.substring(0, 97) + '...' : item.prompt) : '';
        var pairs = [];
        if (truncPrompt)
            pairs.push('<span class="gen-metadata-key">Prompt:</span> <span class="gen-metadata-val">' + escapeHtml(truncPrompt) + '</span>');
        if (item.model)
            pairs.push('<span class="gen-metadata-key">Model:</span> <span class="gen-metadata-val">' + escapeHtml(item.model) + '</span>');
        if (item.seed != null)
            pairs.push('<span class="gen-metadata-key">Seed:</span> <span class="gen-metadata-val">' + item.seed + '</span>');
        if (item.width && item.height)
            pairs.push('<span class="gen-metadata-key">Size:</span> <span class="gen-metadata-val">' + item.width + '\u00d7' + item.height + '</span>');
        if (item.steps)
            pairs.push('<span class="gen-metadata-key">Steps:</span> <span class="gen-metadata-val">' + item.steps + '</span>');
        if (item.cfg != null)
            pairs.push('<span class="gen-metadata-key">' + (item.arch === 'flux' ? 'Guidance' : 'CFG') + ':</span> <span class="gen-metadata-val">' + item.cfg + '</span>');
        if (item.isVideo && item.frame_count && item.fps)
            pairs.push('<span class="gen-metadata-key">Video:</span> <span class="gen-metadata-val">' +
                item.frame_count + 'f @ ' + item.fps + 'fps</span>');
        summary.innerHTML = pairs.join(' ');
        panel.classList.add('visible');
        // Full metadata
        var fullPairs = [];
        if (item.prompt)
            fullPairs.push('<span class="gen-metadata-key">Full Prompt:</span> <span class="gen-metadata-val">' + escapeHtml(item.prompt) + '</span>');
        if (item.sampler)
            fullPairs.push('<span class="gen-metadata-key">Sampler:</span> <span class="gen-metadata-val">' + escapeHtml(item.sampler) + '</span>');
        if (item.scheduler)
            fullPairs.push('<span class="gen-metadata-key">Scheduler:</span> <span class="gen-metadata-val">' + escapeHtml(item.scheduler) + '</span>');
        if (item.guidance && item.arch !== 'flux')
            fullPairs.push('<span class="gen-metadata-key">Guidance:</span> <span class="gen-metadata-val">' + item.guidance + '</span>');
        if (item.arch)
            fullPairs.push('<span class="gen-metadata-key">Arch:</span> <span class="gen-metadata-val">' + item.arch + '</span>');
        if (item.videoGuidanceMode)
            fullPairs.push('<span class="gen-metadata-key">Guidance mode:</span> <span class="gen-metadata-val">' + escapeHtml(item.videoGuidanceMode) + '</span>');
        if (item.videoQuant)
            fullPairs.push('<span class="gen-metadata-key">Quantization:</span> <span class="gen-metadata-val">' + escapeHtml(item.videoQuant) + '</span>');
        if (item.isVideo)
            fullPairs.push('<span class="gen-metadata-key">Audio:</span> <span class="gen-metadata-val">' +
                escapeHtml(item.audioPolicy || (item.includeAudio ? 'generate' : 'none')) + '</span>');
        if (item.timestamp)
            fullPairs.push('<span class="gen-metadata-key">Time:</span> <span class="gen-metadata-val">' + new Date(item.timestamp).toLocaleString() + '</span>');
        full.innerHTML = fullPairs.join('<br>');
        // Toggle expand on click
        summary.onclick = function () {
            state.metadataExpanded = !state.metadataExpanded;
            full.classList.toggle('open', state.metadataExpanded);
        };
    }
    function escapeHtml(str) {
        if (!str)
            return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    // ── Phase 2: Context Menu ──
    function showContextMenu(e, galleryIndex) {
        state.contextMenuIndex = galleryIndex;
        var menu = document.getElementById('gen-context-menu');
        if (!menu) {
            menu = document.createElement('div');
            menu.id = 'gen-context-menu';
            menu.className = 'gen-context-menu';
            document.body.appendChild(menu);
        }
        var item = state.gallery[galleryIndex] || {};
        var starLabel = item.starred ? 'Unstar' : 'Star';
        menu.innerHTML =
            '<button class="gen-context-item" data-action="download">Download</button>' +
                '<button class="gen-context-item" data-action="star">' + starLabel + '</button>' +
                '<div class="gen-context-sep"></div>' +
                '<button class="gen-context-item" data-action="use-prompt">Use Prompt</button>' +
                '<button class="gen-context-item" data-action="use-seed">Use Seed</button>' +
                '<button class="gen-context-item" data-action="use-all">Use All Parameters</button>' +
                '<div class="gen-context-sep"></div>' +
                '<button class="gen-context-item destructive" data-action="delete">Delete</button>';
        // Position
        menu.style.left = e.clientX + 'px';
        menu.style.top = e.clientY + 'px';
        menu.classList.add('visible');
        // Ensure menu stays in viewport
        requestAnimationFrame(function () {
            var rect = menu.getBoundingClientRect();
            if (rect.right > window.innerWidth)
                menu.style.left = (window.innerWidth - rect.width - 4) + 'px';
            if (rect.bottom > window.innerHeight)
                menu.style.top = (window.innerHeight - rect.height - 4) + 'px';
        });
        // Bind actions
        menu.querySelectorAll('.gen-context-item').forEach(function (btn) {
            btn.addEventListener('click', function () {
                handleContextAction(this.dataset.action, galleryIndex);
                hideContextMenu();
            });
        });
    }
    function hideContextMenu() {
        var menu = document.getElementById('gen-context-menu');
        if (menu)
            menu.classList.remove('visible');
    }
    function handleContextAction(action, idx) {
        var item = state.gallery[idx];
        if (!item)
            return;
        switch (action) {
            case 'download':
                var a = document.createElement('a');
                a.href = item.src;
                var ext = item.isVideo ? '.mp4' : '.png';
                a.download = 'serenityflow_' + Date.now() + ext;
                a.click();
                break;
            case 'star':
                item.starred = !item.starred;
                renderGallery();
                break;
            case 'use-prompt':
                if (item.prompt && els.prompt) {
                    els.prompt.value = item.prompt;
                    state.prompt = item.prompt;
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = els.prompt.scrollHeight + 'px';
                    updateTokenCount();
                }
                break;
            case 'use-seed':
                if (item.seed != null && els.seed) {
                    els.seed.value = String(item.seed);
                    state.seed = item.seed;
                    var toggle = document.getElementById('gen-seed-random-toggle');
                    if (toggle)
                        toggle.classList.toggle('on', state.seed === -1);
                }
                break;
            case 'use-all':
                applyParams(reusableParamsForItem(item));
                break;
            case 'delete':
                deleteHistoryArtifact(item).then(function () {
                    state.gallery.splice(idx, 1);
                    state.selectedImages = state.selectedImages.filter(function (si) { return si !== idx; }).map(function (si) { return si > idx ? si - 1 : si; });
                    renderGallery();
                    updateSelectionUI();
                    updateMetadataPanel();
                }).catch(function (error) {
                    showError('Could not delete output: ' + error.message);
                });
                break;
        }
    }
    // ── Phase 2: Bulk Operations ──
    function bindBulkActions() {
        var bulkStar = document.getElementById('gen-bulk-star');
        var bulkUnstar = document.getElementById('gen-bulk-unstar');
        var bulkDownload = document.getElementById('gen-bulk-download');
        var bulkDelete = document.getElementById('gen-bulk-delete');
        if (bulkStar)
            bulkStar.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    if (state.gallery[idx])
                        state.gallery[idx].starred = true;
                });
                renderGallery();
            });
        if (bulkUnstar)
            bulkUnstar.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    if (state.gallery[idx])
                        state.gallery[idx].starred = false;
                });
                renderGallery();
            });
        if (bulkDownload)
            bulkDownload.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    var item = state.gallery[idx];
                    if (!item)
                        return;
                    var a = document.createElement('a');
                    a.href = item.src;
                    a.download = 'serenityflow_' + (item.timestamp || Date.now()) + (item.isVideo ? '.mp4' : '.png');
                    a.click();
                });
            });
        if (bulkDelete)
            bulkDelete.addEventListener('click', function () {
                if (!confirm('Delete ' + state.selectedImages.length + ' images?'))
                    return;
                var selected = state.selectedImages.map(function (idx) {
                    return state.gallery[idx];
                }).filter(Boolean);
                Promise.all(selected.map(deleteHistoryArtifact)).then(function () {
                    state.gallery = state.gallery.filter(function (item) {
                        return selected.indexOf(item) < 0;
                    });
                    state.selectedImages = [];
                    renderGallery();
                    updateSelectionUI();
                    updateMetadataPanel();
                    clearPreview();
                }).catch(function (error) {
                    showError('Could not delete every output: ' + error.message);
                    refreshHistory();
                });
            });
    }
    // ── Phase 7: Gallery Upload & Drag-Drop ──
    function activateLibraryPanel(target) {
        if (target !== 'history')
            ensureLibraryExpanded();
        document.querySelectorAll('.gen-library-tab').forEach(function (tab) {
            tab.classList.toggle('active', tab.dataset.library === target);
        });
        document.querySelectorAll('.gen-library-panel').forEach(function (panel) {
            panel.classList.toggle('active', panel.dataset.libraryPanel === target);
        });
        if (els.galleryClear)
            els.galleryClear.style.display = target === 'history' ? '' : 'none';
        var gallerySettings = document.getElementById('gen-gallery-settings-btn');
        if (gallerySettings)
            gallerySettings.style.display = target === 'history' ? '' : 'none';
        state.gallerySubTab = target === 'assets' ? 'assets' : 'images';
        if (target === 'assets')
            loadAssets(true);
        if (target === 'history')
            renderGallery();
        if (target === 'presets')
            loadGenerationPresets();
        if (target === 'workflows')
            loadGenerationWorkflows();
        if (target === 'models')
            renderModelLibrary();
        if (target === 'loras')
            renderLoraLibrary();
    }
    function setGallerySubTab(tab) {
        var assets = tab === 'assets';
        state.gallerySubTab = assets ? 'assets' : 'images';
        var subtabImages = document.getElementById('gen-subtab-images');
        var subtabAssets = document.getElementById('gen-subtab-assets');
        var imagesContent = document.getElementById('gen-images-content');
        var assetsContent = document.getElementById('gen-assets-content');
        var bulkBar = document.getElementById('gen-bulk-bar');
        if (subtabImages)
            subtabImages.classList.toggle('active', !assets);
        if (subtabAssets)
            subtabAssets.classList.toggle('active', assets);
        if (imagesContent)
            imagesContent.style.display = assets ? 'none' : '';
        if (assetsContent)
            assetsContent.style.display = assets ? '' : 'none';
        if (bulkBar && assets)
            bulkBar.classList.remove('visible');
        if (assets)
            loadAssets(false);
        else
            renderGallery();
    }
    function loadAssets(force) {
        if (state.assetsLoading) {
            if (force)
                state.assetsReloadPending = true;
            renderAssets();
            return Promise.resolve(state.assets);
        }
        if (state.assetsLoaded && !force) {
            renderAssets();
            return Promise.resolve(state.assets);
        }
        state.assetsLoading = true;
        state.assetsError = '';
        if (els.assetsPlaceholder) {
            els.assetsPlaceholder.style.display = '';
            els.assetsPlaceholder.textContent = 'Loading assets…';
        }
        return fetch('/v1/assets', { cache: 'no-store' })
            .then(function (response) {
            return response.text().then(function (body) {
                var data = {};
                try {
                    data = JSON.parse(body);
                }
                catch (error) {
                    data = { detail: body };
                }
                if (!response.ok)
                    throw new Error(data.detail || ('HTTP ' + response.status));
                if (data.schema !== 'serenity.assets.v1' || !Array.isArray(data.assets))
                    throw new Error('server returned an invalid assets inventory');
                return data.assets;
            });
        })
            .then(function (assets) {
            state.assets = assets;
            state.assetsLoaded = true;
            state.assetsError = '';
            return assets;
        })
            .catch(function (error) {
            state.assets = [];
            state.assetsLoaded = false;
            state.assetsError = error.message;
            return [];
        })
            .finally(function () {
            state.assetsLoading = false;
            renderAssets();
            if (state.assetsReloadPending) {
                state.assetsReloadPending = false;
                loadAssets(true);
            }
        });
    }
    function assetSizeLabel(bytes) {
        var size = Number(bytes) || 0;
        if (size >= 1024 * 1024)
            return (size / (1024 * 1024)).toFixed(size >= 10 * 1024 * 1024 ? 0 : 1) + ' MB';
        if (size >= 1024)
            return Math.round(size / 1024) + ' KB';
        return size + ' B';
    }
    function useAssetAsSource(asset) {
        if (!asset || asset.media_type !== 'image') {
            showError('Only image assets can be used as an I2V/img2img source here');
            return;
        }
        state.initImagePath = String(asset.path || '');
        state.initImageName = String(asset.name || state.initImagePath);
        state.initImageWidth = Number(asset.width) || 0;
        state.initImageHeight = Number(asset.height) || 0;
        if (!state.initImagePath) {
            showError('This asset has no worker-readable source path');
            return;
        }
        var preview = document.getElementById('gen-init-preview');
        var empty = document.getElementById('gen-init-empty');
        var name = document.getElementById('gen-init-name');
        var clear = document.getElementById('gen-init-clear');
        if (preview) {
            preview.src = String(asset.url || '');
            preview.style.display = 'block';
        }
        if (empty)
            empty.style.display = 'none';
        if (name)
            name.textContent = state.initImageName;
        if (clear)
            clear.disabled = false;
        if ((!state.initImageWidth || !state.initImageHeight) && asset.url) {
            var probe = new Image();
            probe.onload = function () {
                state.initImageWidth = probe.naturalWidth;
                state.initImageHeight = probe.naturalHeight;
                syncWanI2vSteps();
            };
            probe.src = asset.url;
        }
        syncWanI2vSteps();
        renderAssets();
    }
    function deleteAsset(asset) {
        if (!asset || !asset.name)
            return;
        if (!confirm('Delete source asset ' + asset.name + '?'))
            return;
        fetch('/v1/assets/' + encodeURIComponent(asset.name), {
            method: 'DELETE'
        })
            .then(function (response) {
            return response.text().then(function (body) {
                var data = {};
                try {
                    data = JSON.parse(body);
                }
                catch (error) {
                    data = { detail: body };
                }
                if (!response.ok)
                    throw new Error(data.detail || ('HTTP ' + response.status));
                return data;
            });
        })
            .then(function () {
            if (state.initImagePath === asset.path)
                clearInitImage();
            return loadAssets(true);
        })
            .catch(function (error) {
            showError('Asset delete failed: ' + error.message);
        });
    }
    function renderAssets() {
        if (!els.assetsGrid || !els.assetsPlaceholder)
            return;
        var query = String(state.gallerySearch || '').toLowerCase();
        var assets = (state.assets || []).filter(function (asset) {
            return !query || String(asset.name || '').toLowerCase().indexOf(query) >= 0;
        });
        els.assetsGrid.innerHTML = '';
        els.assetsPlaceholder.style.display = assets.length ? 'none' : '';
        if (!state.assetsLoading) {
            els.assetsPlaceholder.textContent = state.assetsError
                ? 'Could not load assets: ' + state.assetsError
                : state.assetsLoaded
                ? (query ? 'No assets match this search' : 'No source assets uploaded')
                : 'Assets are unavailable';
        }
        assets.forEach(function (asset) {
            var card = document.createElement('div');
            card.className = 'gen-asset-card' +
                (state.initImagePath === asset.path ? ' selected' : '');
            var media = asset.media_type === 'video'
                ? document.createElement('video')
                : document.createElement('img');
            media.className = 'gen-asset-preview';
            media.src = String(asset.url || '');
            if (asset.media_type === 'video') {
                media.muted = true;
                media.preload = 'metadata';
            }
            else {
                media.loading = 'lazy';
                media.alt = String(asset.name || 'source asset');
                card.addEventListener('dblclick', function () {
                    useAssetAsSource(asset);
                });
            }
            card.appendChild(media);
            var name = document.createElement('div');
            name.className = 'gen-asset-name';
            name.textContent = String(asset.name || 'asset');
            name.title = name.textContent;
            card.appendChild(name);
            var meta = document.createElement('div');
            meta.className = 'gen-asset-meta';
            var dimensions = asset.width && asset.height
                ? asset.width + '×' + asset.height + ' · '
                : '';
            meta.textContent = dimensions + assetSizeLabel(asset.size);
            card.appendChild(meta);
            var actions = document.createElement('div');
            actions.className = 'gen-asset-actions';
            if (asset.media_type === 'image') {
                var use = document.createElement('button');
                use.type = 'button';
                use.className = 'gen-asset-use';
                use.textContent = state.initImagePath === asset.path
                    ? 'Source selected'
                    : 'Use as source';
                use.addEventListener('click', function () {
                    useAssetAsSource(asset);
                });
                actions.appendChild(use);
            }
            var remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'gen-asset-delete';
            remove.textContent = 'Delete';
            remove.addEventListener('click', function () {
                deleteAsset(asset);
            });
            actions.appendChild(remove);
            card.appendChild(actions);
            els.assetsGrid.appendChild(card);
        });
    }
    function bindGalleryUpload() {
        var uploadBtn = document.getElementById('gen-gallery-upload-btn');
        var uploadInput = document.getElementById('gen-gallery-upload-input');
        if (uploadBtn && uploadInput) {
            uploadBtn.addEventListener('click', function () { uploadInput.click(); });
            uploadInput.addEventListener('change', function () {
                if (this.files && this.files.length > 0) {
                    handleUploadFiles(this.files);
                    this.value = '';
                }
            });
        }
        // Drag-and-drop on gallery grid
        var galleryGrid = els.galleryGrid;
        if (galleryGrid) {
            galleryGrid.addEventListener('dragover', function (e) {
                e.preventDefault();
                e.stopPropagation();
                galleryGrid.classList.add('gen-drag-over');
            });
            galleryGrid.addEventListener('dragleave', function (e) {
                e.preventDefault();
                galleryGrid.classList.remove('gen-drag-over');
            });
            galleryGrid.addEventListener('drop', function (e) {
                e.preventDefault();
                e.stopPropagation();
                galleryGrid.classList.remove('gen-drag-over');
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    handleUploadFiles(e.dataTransfer.files);
                }
            });
        }
    }
    function handleUploadFiles(files) {
        Array.prototype.forEach.call(files, function (file) {
            if (!file.type.startsWith('image/')) {
                showError('Assets currently accept source images');
                return;
            }
            // Upload to server
            var formData = new FormData();
            formData.append('image', file);
            fetch('/upload/image', { method: 'POST', body: formData })
                .then(function (response) {
                return response.text().then(function (body) {
                    var data = {};
                    try {
                        data = JSON.parse(body);
                    }
                    catch (error) {
                        data = { detail: body };
                    }
                    if (!response.ok)
                        throw new Error(data.detail || ('HTTP ' + response.status));
                    return data;
                });
            })
                .then(function (data) {
                if (!data || !data.path || !data.url)
                    throw new Error('upload returned no persistent asset path');
                activateLibraryPanel('assets');
                return loadAssets(true);
            })
                .catch(function (err) {
                showError('Asset upload failed: ' + err.message);
            });
        });
    }
    // ── Phase 2: Gallery Settings Popover ──
    function bindGallerySettings() {
        var settingsBtn = document.getElementById('gen-gallery-settings-btn');
        var popover = document.getElementById('gen-gallery-popover');
        if (!settingsBtn || !popover)
            return;
        settingsBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            state.gallerySettingsOpen = !state.gallerySettingsOpen;
            popover.classList.toggle('visible', state.gallerySettingsOpen);
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (popover && !popover.contains(e.target) && e.target !== settingsBtn) {
                state.gallerySettingsOpen = false;
                popover.classList.remove('visible');
            }
        });
        // Thumb size slider
        var thumbSlider = document.getElementById('gen-thumb-size-slider');
        var thumbVal = document.getElementById('gen-thumb-size-val');
        if (thumbSlider) {
            thumbSlider.value = String(state.thumbSize);
            if (thumbVal)
                thumbVal.textContent = String(state.thumbSize);
            thumbSlider.addEventListener('input', function () {
                state.thumbSize = parseInt(this.value);
                if (thumbVal)
                    thumbVal.textContent = String(state.thumbSize);
                applyThumbSize();
                try {
                    localStorage.setItem('sf-thumb-size', String(state.thumbSize));
                }
                catch (e) { }
            });
        }
        // Sort direction toggle
        var sortToggle = document.getElementById('gen-sort-direction-toggle');
        var sortLabel = document.getElementById('gen-sort-direction-label');
        if (sortToggle) {
            sortToggle.addEventListener('click', function () {
                state.sortNewestFirst = !state.sortNewestFirst;
                this.classList.toggle('on', state.sortNewestFirst);
                if (sortLabel)
                    sortLabel.textContent = state.sortNewestFirst ? 'Newest' : 'Oldest';
                renderGallery();
            });
        }
        // Starred first toggle
        var starredToggle = document.getElementById('gen-starred-first-toggle');
        if (starredToggle) {
            starredToggle.addEventListener('click', function () {
                state.starredFirst = !state.starredFirst;
                this.classList.toggle('on', state.starredFirst);
                renderGallery();
            });
        }
        // Auto-switch toggle
        var autoToggle = document.getElementById('gen-auto-switch-toggle');
        if (autoToggle) {
            autoToggle.addEventListener('click', function () {
                state.autoSwitchNew = !state.autoSwitchNew;
                this.classList.toggle('on', state.autoSwitchNew);
            });
        }
    }
    // ── Phase 2: Global Event Listeners ──
    function bindGlobalPhase2() {
        // Close context menu on click outside or ESC
        document.addEventListener('click', function () {
            hideContextMenu();
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                hideContextMenu();
            }
        });
    }

    function getParams() {
        return {
            model: state.model || '',
            prompt: state.prompt || '',
            negPrompt: state.negPrompt || '',
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            sigmaShift: state.sigmaShift,
            sampler: state.sampler,
            scheduler: state.scheduler,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps,
            seconds: state.seconds,
            variationSeed: state.variationSeed,
            variationStrength: state.variationStrength,
            creativity: state.creativity,
            batchCount: state.batchCount,
            stylePreset: state.stylePreset,
            videoGuidanceMode: state.videoGuidanceMode,
            ltx2Mode: state.videoGuidanceMode,
            videoWorkflowProfile: state.videoWorkflowProfile,
            ltx2WorkflowProfile: state.videoWorkflowProfile,
            videoPromptEnhancer: state.videoPromptEnhancer,
            ltx2PromptEnhancer: state.videoPromptEnhancer,
            videoQuant: state.videoQuant,
            quantization: state.videoQuant,
            h3AttentionBackend: state.h3AttentionBackend,
            h3StepCache: state.h3StepCache,
            videoCheckpoint: state.videoCheckpoint,
            capsPositive: state.capsPositive,
            capsNegative: state.capsNegative,
            noiseFixture: state.noiseFixture,
            includeAudio: state.includeAudio,
            audioPolicy: state.audioPolicy,
            ltx2AudioPolicy: state.audioPolicy,
            postUpscaler: state.postUpscaler,
            ltx2PostUpscaler: state.postUpscaler,
            postUpscaleFactor: state.postUpscaleFactor,
            ltx2PostUpscaleFactor: state.postUpscaleFactor,
            cameraMotion: state.cameraMotion,
            ltx2CameraMotion: state.cameraMotion,
            initImagePath: state.initImagePath,
            initImageName: state.initImageName,
            noSeedIncrement: state.noSeedIncrement,
            continueAfterErrors: state.continueAfterErrors,
            personalNote: state.personalNote,
            loras: state.loras.map(function (lora) {
                return {
                    name: lora.name,
                    strength: lora.strength,
                    enabled: lora.enabled !== false,
                    role: lora.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
            })
        };
    }

    function applyParams(params) {
        if (!params)
            return;
        if (!initialized)
            init();
        var normalized = Object.assign({}, params);
        if (normalized.negPrompt == null && typeof normalized.negative === 'string')
            normalized.negPrompt = normalized.negative;
        if (!Array.isArray(normalized.loras) && Array.isArray(normalized.lora)) {
            normalized.loras = normalized.lora.map(function (row) {
                return {
                    name: row.name || '',
                    strength: Number(row.weight == null ? 1 : row.weight),
                    enabled: true,
                    role: row.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
            });
        }
        params = normalized;
        if (typeof params.model === 'string' && params.model) {
            var requestedModel = params.model === 'ltx2'
                ? String(params.videoCheckpoint || params.checkpoint ||
                    (((params.videoQuant || params.quant) === 'bf16')
                        ? 'ltx-2.3-22b-distilled-fp8-dequant-bf16'
                        : 'ltx-2.3-22b-distilled'))
                : (params.model === 'minimax_h3' ? 'MiniMax-H3-Mojo' : params.model);
            state.model = requestedModel;
            if (els.model)
                els.model.value = requestedModel;
            if (els.modelSearch)
                els.modelSearch.value = requestedModel;
            updateUIForArch(ModelUtils.archForModel(requestedModel));
        }
        if (typeof params.prompt === 'string')
            state.prompt = params.prompt;
        if (typeof params.negPrompt === 'string')
            state.negPrompt = params.negPrompt;
        ['width', 'height', 'steps', 'cfg', 'guidance', 'sigmaShift', 'seed', 'frames', 'fps',
            'seconds', 'variationSeed', 'variationStrength', 'creativity',
            'batchCount'].forEach(function (key) {
            if (params[key] != null && Number.isFinite(Number(params[key])))
                state[key] = Number(params[key]);
        });
        var isVideo = ModelUtils.isVideoModel(state.model);
        var profile = activeCapabilityProfile();
        var supportedSamplers = profile && profile.samplers &&
            Array.isArray(profile.samplers.supported_samplers)
            ? profile.samplers.supported_samplers : [];
        var supportedSchedulers = profile && profile.samplers &&
            Array.isArray(profile.samplers.supported_schedulers)
            ? profile.samplers.supported_schedulers : [];
        if (typeof params.sampler === 'string' &&
            (isVideo || supportedSamplers.indexOf(params.sampler) >= 0))
            state.sampler = params.sampler;
        if (typeof params.scheduler === 'string' && params.scheduler) {
            if (isVideo || supportedSchedulers.indexOf(params.scheduler) >= 0)
                state.scheduler = params.scheduler;
            else if (supportedSamplers.indexOf(params.scheduler) >= 0)
                state.sampler = params.scheduler;
        }
        if (typeof params.stylePreset === 'string')
            state.stylePreset = params.stylePreset;
        if (typeof params.videoGuidanceMode === 'string')
            state.videoGuidanceMode = params.videoGuidanceMode === 'dev' ? 'dev' : 'distilled';
        else if (typeof params.guidance_mode === 'string')
            state.videoGuidanceMode = params.guidance_mode === 'dev' ? 'dev' : 'distilled';
        if (typeof params.videoWorkflowProfile === 'string')
            state.videoWorkflowProfile = params.videoWorkflowProfile;
        else if (typeof params.workflow_profile === 'string')
            state.videoWorkflowProfile = params.workflow_profile;
        if (typeof params.videoPromptEnhancer === 'string')
            state.videoPromptEnhancer = params.videoPromptEnhancer;
        else if (typeof params.prompt_enhancer === 'string')
            state.videoPromptEnhancer = params.prompt_enhancer;
        if (typeof params.videoQuant === 'string')
            state.videoQuant = params.videoQuant;
        else if (typeof params.quant === 'string')
            state.videoQuant = params.quant;
        if (ModelUtils.archForModel(state.model) === 'minimax_h3')
            state.h3Quant = ['int8-fast', 'int8', 'bf16'].indexOf(state.videoQuant) >= 0
                ? state.videoQuant : 'int8-fast';
        if (typeof params.h3AttentionBackend === 'string')
            state.h3AttentionBackend = params.h3AttentionBackend === 'sage-int8'
                ? 'sage-int8' : 'cudnn';
        else if (typeof params.attention_backend === 'string')
            state.h3AttentionBackend = params.attention_backend === 'sage-int8'
                ? 'sage-int8' : 'cudnn';
        if (typeof params.h3StepCache === 'string')
            state.h3StepCache = params.h3StepCache === 'high' ? 'high' : 'exact';
        else if (typeof params.step_cache === 'string')
            state.h3StepCache = params.step_cache === 'high' ? 'high' : 'exact';
        if (typeof params.videoCheckpoint === 'string')
            state.videoCheckpoint = params.videoCheckpoint;
        else if (typeof params.checkpoint === 'string')
            state.videoCheckpoint = params.checkpoint;
        if (typeof params.capsPositive === 'string')
            state.capsPositive = params.capsPositive;
        else if (typeof params.caps_positive === 'string')
            state.capsPositive = params.caps_positive;
        if (typeof params.capsNegative === 'string')
            state.capsNegative = params.capsNegative;
        else if (typeof params.caps_negative === 'string')
            state.capsNegative = params.caps_negative;
        if (typeof params.noiseFixture === 'string')
            state.noiseFixture = params.noiseFixture;
        else if (typeof params.noise_fixture === 'string')
            state.noiseFixture = params.noise_fixture;
        if (typeof params.includeAudio === 'boolean')
            state.includeAudio = params.includeAudio;
        else if (typeof params.include_audio === 'boolean')
            state.includeAudio = params.include_audio;
        if (typeof params.audioPolicy === 'string')
            state.audioPolicy = params.audioPolicy;
        else if (typeof params.audio_policy === 'string')
            state.audioPolicy = params.audio_policy;
        else
            state.audioPolicy = state.includeAudio ? 'generate' : 'none';
        var postUpscale = params.post_upscale && typeof params.post_upscale === 'object'
            ? params.post_upscale : null;
        if (typeof params.postUpscaler === 'string')
            state.postUpscaler = params.postUpscaler;
        else if (postUpscale && typeof postUpscale.id === 'string')
            state.postUpscaler = postUpscale.id;
        if (params.postUpscaleFactor != null &&
            (Number(params.postUpscaleFactor) === 2 || Number(params.postUpscaleFactor) === 4))
            state.postUpscaleFactor = Number(params.postUpscaleFactor);
        else if (postUpscale &&
            (Number(postUpscale.factor) === 2 || Number(postUpscale.factor) === 4))
            state.postUpscaleFactor = Number(postUpscale.factor);
        if (params.sigmaShift == null && params.sigma_shift != null &&
            Number.isFinite(Number(params.sigma_shift)))
            state.sigmaShift = Number(params.sigma_shift);
        if (typeof params.noSeedIncrement === 'boolean')
            state.noSeedIncrement = params.noSeedIncrement;
        if (typeof params.continueAfterErrors === 'boolean')
            state.continueAfterErrors = params.continueAfterErrors;
        if (typeof params.personalNote === 'string')
            state.personalNote = params.personalNote;
        if (typeof params.cameraMotion === 'string')
            state.cameraMotion = params.cameraMotion;
        else if (typeof params.camera_motion === 'string')
            state.cameraMotion = params.camera_motion;
        if (Array.isArray(params.loras)) {
            state.loras = params.loras.map(function (lora) {
                return {
                    name: lora.name || '',
                    strength: Number(lora.strength == null ? 1 : lora.strength),
                    enabled: lora.enabled !== false,
                    role: lora.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
            }).filter(function (lora) { return !!lora.name; });
            renderLoraList();
        }
        if (ModelUtils.archForModel(state.model) === 'ltxv') {
            var reusedProfile = activeLtx2RequestProfiles().find(function (entry) {
                return Number(entry.width) === Number(state.width) &&
                    Number(entry.height) === Number(state.height) &&
                    Number(entry.frames) === Number(state.frames) &&
                    Number(entry.fps) === Number(state.fps);
            });
            if (reusedProfile)
                applyLtx2RequestProfile(reusedProfile);
        }
        if (ModelUtils.archForModel(state.model) === 'ltxv')
            applyVideoGuidanceMode(false);
        if (els.prompt) {
            els.prompt.value = state.prompt;
            els.prompt.style.height = 'auto';
            els.prompt.style.height = els.prompt.scrollHeight + 'px';
        }
        if (els.negPrompt)
            els.negPrompt.value = state.negPrompt;
        if (els.steps)
            els.steps.value = String(state.steps);
        if (els.stepsRange)
            els.stepsRange.value = String(state.steps);
        if (els.cfg)
            els.cfg.value = String(state.cfg);
        if (els.cfgRange)
            els.cfgRange.value = String(state.cfg);
        if (els.guidance)
            els.guidance.value = String(state.guidance);
        if (els.guidanceRange)
            els.guidanceRange.value = String(state.guidance);
        if (els.sampler)
            els.sampler.value = state.sampler;
        if (els.scheduler)
            els.scheduler.value = state.scheduler;
        if (els.seed)
            els.seed.value = String(state.seed);
        if (els.framesInput)
            els.framesInput.value = String(state.frames);
        if (els.fpsInput)
            els.fpsInput.value = String(state.fps);
        if (els.fpsRange)
            els.fpsRange.value = String(state.fps);
        if (els.secondsInput)
            els.secondsInput.value = String(Number(state.seconds.toFixed(3)));
        if (els.videoGuidanceMode)
            els.videoGuidanceMode.value = state.videoGuidanceMode;
        if (els.videoQuant)
            els.videoQuant.value = state.videoQuant;
        if (els.h3Attention)
            els.h3Attention.value = state.h3AttentionBackend;
        if (els.h3StepCache)
            els.h3StepCache.value = state.h3StepCache;
        if (els.videoCheckpoint)
            els.videoCheckpoint.value = state.videoCheckpoint;
        if (els.capsPositive)
            els.capsPositive.value = state.capsPositive;
        if (els.capsNegative)
            els.capsNegative.value = state.capsNegative;
        if (els.noiseFixture)
            els.noiseFixture.value = state.noiseFixture;
        if (els.audioPolicy)
            els.audioPolicy.value = state.audioPolicy;
        refreshLtx2PostUpscaleControls();
        if (els.sigmaShift)
            els.sigmaShift.value = String(state.sigmaShift);
        if (els.sigmaShiftRange)
            els.sigmaShiftRange.value = String(Math.min(Number(els.sigmaShiftRange.max) || 20, state.sigmaShift));
        if (els.noSeedIncrement)
            els.noSeedIncrement.checked = state.noSeedIncrement;
        if (els.continueAfterErrors)
            els.continueAfterErrors.checked = state.continueAfterErrors;
        if (els.personalNote)
            els.personalNote.value = state.personalNote;
        var variationSeed = document.getElementById('gen-variation-seed');
        var variationStrength = document.getElementById('gen-variation-strength');
        var variationRange = document.getElementById('gen-variation-strength-range');
        var creativity = document.getElementById('gen-creativity');
        var creativityRange = document.getElementById('gen-creativity-range');
        var batch = document.getElementById('gen-batch');
        var style = document.getElementById('gen-style-preset');
        if (variationSeed) variationSeed.value = String(state.variationSeed);
        if (variationStrength) variationStrength.value = String(state.variationStrength);
        if (variationRange) variationRange.value = String(state.variationStrength);
        if (creativity) creativity.value = String(state.creativity);
        if (creativityRange) creativityRange.value = String(state.creativity);
        if (batch) batch.value = String(state.batchCount);
        if (els.toolbarBatchInput) els.toolbarBatchInput.value = String(state.batchCount);
        if (style) style.value = state.stylePreset;
        syncDimensionInputs();
        syncAspectDropdown();
        updateAspectPreview();
        updateTokenCount();
        updateStylePreview();
        closeModelDropdown();
    }

    // ── Public API ──
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        bindEvents();
        loadModels();
        loadLoras();
        loadAdvancedOptions();
        loadHistory();
        loadPromptHistory();
        connectWS();
        renderExternalActivity();
        updateAspectPreview();
        bindBulkActions();
        bindGalleryUpload();
        bindGallerySettings();
        bindGlobalPhase2();
        updateAdvancedVisibility();
        // Render lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons({ nameAttr: 'data-lucide' });
        }
    }
    return {
        state: state,
        init: init,
        selectModel: selectModel,
        generate: generate,
        buildWorkflow: buildWorkflow,
        getParams: getParams,
        applyParams: applyParams,
        setExternalActivity: setExternalActivity,
        displayResult: function (src, isVideo, metadata) {
            if (!initialized)
                init();
            if (isVideo)
                displayVideo(src, metadata);
            else
                displayImage(src, metadata);
        },
        displayCompletedJob: function (src, isVideo, promptId) {
            if (!initialized)
                init();
            var id = String(promptId || '');
            if (!id) {
                if (isVideo)
                    displayVideo(src);
                else
                    displayImage(src);
                return Promise.resolve();
            }
            return fetch('/v1/job/' + encodeURIComponent(id), { cache: 'no-store' })
                .then(function (response) {
                if (!response.ok)
                    throw new Error('HTTP ' + response.status);
                return response.json();
            })
                .then(function (job) {
                var metadata = galleryMetadataFromJob(job);
                if (isVideo)
                    displayVideo(src, metadata);
                else
                    displayImage(src, metadata);
                addToGallery(src, isVideo, metadata);
            })
                .catch(function (error) {
                console.error('Completed Workflow result metadata unavailable for ' +
                    id + ': ' + error.message);
                if (isVideo)
                    displayVideo(src);
                else
                    displayImage(src);
                refreshHistory();
            });
        }
    };
})();
