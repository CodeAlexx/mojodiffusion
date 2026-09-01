"use strict";
/**
 * H3ProjectContracts — pure project, prompt, Director, and render contracts for
 * the production Serenity browser UI. No DOM, no GPU work, and no training.
 */
var H3ProjectContracts = (function () {
    'use strict';

    var PROJECT_SCHEMA = 'serenity.h3.movie.v1';
    var CAPTION_SCHEMA = 'serenity.h3.caption.v2';
    var NATIVE_FPS = 24;
    var MIN_SECONDS = 5;
    var MAX_SECONDS = 15;
    var MAX_ENDLESS_SECONDS = 3600;
    var MIN_SEGMENT_FRAMES = MIN_SECONDS * NATIVE_FPS;
    var MAX_SEGMENT_FRAMES = MAX_SECONDS * NATIVE_FPS;
    var MAX_TARGET_FRAMES = MAX_ENDLESS_SECONDS * NATIVE_FPS;
    var ENDLESS_SCHEMA = 'serenity.h3.endless.v1';
    var MAX_PIXELS = 1032192;
    var QWEN_8B = 'Qwen/Qwen3-VL-8B-Instruct';
    var QWEN_32B = 'Qwen/Qwen3-VL-32B-Instruct';
    var CHARACTER_SOURCE_REPO = 'PoopMan333/H3_Character_Sheet_Generator';
    var CHARACTER_SOURCE_URL = 'https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator';
    var CHARACTER_SOURCE_REVISION = 'ccc9d411b6b7056b43edf2690503e063560a5acd';

    var RESOLUTIONS = [
        { id: '21:9', label: '21:9 · 1536×672', width: 1536, height: 672 },
        { id: '16:9', label: '16:9 · 1344×768', width: 1344, height: 768 },
        { id: '4:3', label: '4:3 · 1024×768', width: 1024, height: 768 },
        { id: '1:1', label: '1:1 · 768×768', width: 768, height: 768 },
        { id: '3:4', label: '3:4 · 768×1024', width: 768, height: 1024 },
        { id: '9:16', label: '9:16 · 768×1344', width: 768, height: 1344 }
    ];

    var ACTIONS = [
        { id: 'caption_shot', group: 'ANALYZE', label: 'Caption shot', help: 'Describe an existing clip faithfully, then emit a generation-ready H3 prompt.', needs: 'Video reference or selected rendered take' },
        { id: 'prompt_doctor', group: 'ANALYZE', label: 'Prompt doctor', help: 'Repair a weak prompt without changing its intent.', needs: 'Existing prompt or detailed intent' },
        { id: 'retake_doctor', group: 'ANALYZE', label: 'Retake doctor', help: 'Compare a generated take with the intent and prescribe a focused retry.', needs: 'Generated take or source video plus intended prompt' },
        { id: 'replace_restage', group: 'TRANSFORM', label: 'Replace / restage', help: 'Regenerate a source clip while changing only named people, clothes, props, or setting.', needs: 'Source-video role, replacement visual, and exact change request' },
        { id: 'reference_director', group: 'TRANSFORM', label: 'Reference director', help: 'Assign identity, motion, camera, environment, keyframe, and audio roles across references.', needs: 'One or more ordered image, video, or audio references' },
        { id: 'keyframe_bridge', group: 'TRANSFORM', label: 'Keyframe bridge', help: 'Build an I2VA, FL2VA, or L2VA motion path around supplied frames.', needs: 'Opening frame, ending frame, or keyframe image' },
        { id: 'dream_project', group: 'CREATE', label: 'Dream project', help: 'Expand an outline into beats, assets, exact generations, and 5–15 second H3 shots.', needs: 'Story, film, scene, or commercial outline plus target runtime' },
        { id: 'script_to_coverage', group: 'CREATE', label: 'Script to coverage', help: 'Turn scripted scenes and exact dialogue into editable camera coverage.', needs: 'Screenplay, scene text, dialogue, or commercial copy' },
        { id: 'continue_story', group: 'CREATE', label: 'Continue story', help: 'Continue from a clip while preserving motion, identity, sound, and story state.', needs: 'Source clip or approved take plus direction for what follows' },
        { id: 'sound_to_scene', group: 'CREATE', label: 'Sound to scene', help: 'Stage visuals against dialogue, music, beats, or an audio reference.', needs: 'Audio reference plus visual or story direction' },
        { id: 'character_sheet', group: 'BUILD IDENTITY', label: 'Character sheet', help: 'Turn 1–9 ordered images into an identity-locked H3 orbit sheet and reusable reference prompt.', needs: '1–9 ordered image references; notes say what each contributes or excludes' }
    ];

    function copy(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function secondsText(seconds) {
        return Math.max(0, Number(seconds) || 0).toFixed(2);
    }

    function normalizeAttentionBackend(quant, backend) {
        if (backend === 'ck-int8') return 'ck-int8';
        if (quant !== 'bf16' && backend === 'sage-int8') return 'sage-int8';
        return 'cudnn';
    }

    function createReference(kind, path) {
        return {
            kind: kind || 'image', path: path || '', audio_use: 'reference',
            duration_seconds: 5, note: '', role: 'subject'
        };
    }

    function createShot(id, title) {
        return {
            id: Number(id) || 1,
            title: title || ('Shot ' + (Number(id) || 1)),
            duration_seconds: 8,
            width: 1344,
            height: 768,
            steps: 20,
            seed: 42,
            quant: 'int8-fast',
            attention_backend: 'ck-int8',
            step_cache: 'exact',
            brief: '',
            shot_description: '',
            soundscape: 'N/A',
            music: 'N/A',
            subject_definitions: '',
            summary: '',
            retention_analysis: '',
            prompt_override: '',
            first_frame: '',
            last_frame: '',
            continue_from: '',
            motion_context_frames: 22,
            references: [],
            take_job_ids: [],
            take_states: [],
            take_output_paths: [],
            selected_take: -1,
            status: 'Unrendered',
            output_path: '',
            locked: false
        };
    }

    function createProject() {
        return {
            schema: PROJECT_SCHEMA,
            title: 'Untitled H3 project',
            project_kind: 1,
            delivery_fps: 24,
            adult_mode: false,
            caption_tier: 0,
            target_duration_seconds: 120,
            takes_per_shot: 1,
            director_brief: '',
            continuity_bible: '',
            brand_bible: '',
            last_director_result: null,
            assets: [],
            character_sheets: [],
            endless: createEmptyEndless(120),
            shots: [createShot(1, 'Opening shot')],
            next_asset_id: 1,
            next_shot_id: 2,
            updated_at: new Date().toISOString()
        };
    }

    function createEmptyEndless(targetSeconds) {
        var targetFrames = normalizeFrameCount(targetSeconds, MIN_SEGMENT_FRAMES, MAX_TARGET_FRAMES, 120 * NATIVE_FPS);
        return {
            schema: ENDLESS_SCHEMA,
            status: 'idle',
            target_seconds: targetFrames / NATIVE_FPS,
            target_frames: targetFrames,
            segment_seconds: 10,
            segment_frames: 10 * NATIVE_FPS,
            continuation_direction: 'Continue the same scene naturally. Preserve identity, vehicles, wardrobe, environment, camera direction, motion, lighting, synchronized sound, and story state.',
            base_shot_id: 0,
            segment_durations: [],
            segment_output_frames: [],
            completed_job_ids: [],
            completed_output_paths: [],
            segment_shot_ids: [],
            active_job: null,
            base_snapshot: null,
            fingerprint: '',
            error: '',
            started_at: '',
            updated_at: '',
            stopped_at: ''
        };
    }

    function normalizeFrameCount(seconds, minimum, maximum, fallback) {
        var frames = Math.round(Number(seconds) * NATIVE_FPS);
        if (!Number.isFinite(frames)) frames = fallback;
        return Math.max(minimum, Math.min(maximum, frames));
    }

    function videoJobUrls(videoId) {
        var id = String(videoId || '');
        if (!/^video-\d+$/.test(id)) return null;
        var encoded = encodeURIComponent(id);
        return { video_id: id, status_url: '/out/' + encoded + '/status.json', result_url: '/out/' + encoded + '/result.json' };
    }

    function normalizeEndless(value, targetSeconds) {
        var out = Object.assign(createEmptyEndless(targetSeconds), value || {});
        out.schema = ENDLESS_SCHEMA;
        out.status = ['idle', 'submitting', 'running', 'stopping', 'stopped', 'completed', 'failed'].indexOf(out.status) >= 0 ? out.status : 'idle';
        out.target_frames = normalizeFrameCount(
            value && value.target_frames != null ? Number(value.target_frames) / NATIVE_FPS : out.target_seconds,
            MIN_SEGMENT_FRAMES, MAX_TARGET_FRAMES,
            normalizeFrameCount(targetSeconds, MIN_SEGMENT_FRAMES, MAX_TARGET_FRAMES, 120 * NATIVE_FPS));
        out.segment_frames = normalizeFrameCount(
            value && value.segment_frames != null ? Number(value.segment_frames) / NATIVE_FPS : out.segment_seconds,
            MIN_SEGMENT_FRAMES, MAX_SEGMENT_FRAMES, 10 * NATIVE_FPS);
        out.target_seconds = out.target_frames / NATIVE_FPS;
        out.segment_seconds = out.segment_frames / NATIVE_FPS;
        out.base_shot_id = Number(out.base_shot_id) || 0;
        var planned = Array.isArray(out.segment_output_frames)
            ? out.segment_output_frames.map(Number)
            : (Array.isArray(out.segment_durations) ? out.segment_durations.map(function (seconds) { return Math.round(Number(seconds) * NATIVE_FPS); }) : []);
        var coherentPlan = planned.length > 0 && planned.every(function (frames) {
            return Number.isInteger(frames) && frames >= MIN_SEGMENT_FRAMES && frames <= MAX_SEGMENT_FRAMES;
        }) && planned.reduce(function (sum, frames) { return sum + frames; }, 0) === out.target_frames;
        out.segment_output_frames = coherentPlan ? planned : [];
        out.segment_durations = out.segment_output_frames.map(function (frames) { return frames / NATIVE_FPS; });
        out.completed_job_ids = Array.isArray(out.completed_job_ids) ? out.completed_job_ids.map(String) : [];
        out.completed_output_paths = Array.isArray(out.completed_output_paths) ? out.completed_output_paths.map(String) : [];
        out.segment_shot_ids = Array.isArray(out.segment_shot_ids) ? out.segment_shot_ids.map(Number) : [];
        var completionCoherent = out.completed_job_ids.length <= out.segment_output_frames.length &&
            out.completed_job_ids.every(function (id) { return !!videoJobUrls(id); }) &&
            out.completed_output_paths.length === out.completed_job_ids.length &&
            out.segment_shot_ids.length >= out.completed_job_ids.length &&
            out.segment_shot_ids.slice(0, out.completed_job_ids.length).every(function (id) { return Number.isInteger(id) && id > 0; });
        if (!completionCoherent) {
            out.status = 'failed'; out.error = 'Imported endless run has an inconsistent completed segment chain.';
            out.completed_job_ids = []; out.completed_output_paths = []; out.segment_shot_ids = [];
        }
        var rawActive = out.active_job && typeof out.active_job === 'object' ? out.active_job : null;
        var activeUrls = rawActive ? videoJobUrls(rawActive.video_id) : null;
        var activeIndex = rawActive ? Number(rawActive.segment_index) : -1;
        if (rawActive && (!activeUrls || !Number.isInteger(activeIndex) || activeIndex !== out.completed_job_ids.length || activeIndex >= out.segment_output_frames.length)) {
            out.status = 'failed'; out.error = 'Imported endless run has an invalid active video job.'; rawActive = null;
        }
        out.active_job = rawActive ? Object.assign({}, rawActive, activeUrls, {
            segment_index: activeIndex,
            not_found_count: Math.max(0, Number(rawActive.not_found_count) || 0)
        }) : null;
        out.base_snapshot = out.base_snapshot && typeof out.base_snapshot === 'object' ? out.base_snapshot : null;
        out.continuation_direction = String(out.continuation_direction || '');
        out.fingerprint = String(out.fingerprint || '');
        out.error = String(out.error || '');
        if (out.active_job && ['submitting', 'running', 'stopping', 'failed'].indexOf(out.status) < 0) {
            out.status = 'failed'; out.error = 'Imported endless run has an active job in a terminal state.'; out.active_job = null;
        }
        if (['idle', 'failed'].indexOf(out.status) < 0 && (!out.base_snapshot || !out.segment_output_frames.length)) {
            out.status = 'failed'; out.error = 'Imported endless run is missing its immutable frame plan or base snapshot.'; out.active_job = null;
        }
        if (out.status === 'completed' && out.completed_job_ids.length !== out.segment_output_frames.length) {
            out.status = 'failed'; out.error = 'Imported endless run claims completion without the full frame plan.'; out.active_job = null;
        }
        if (out.base_snapshot && out.status !== 'idle') {
            try {
                var replay = endlessBaseSnapshot(
                    Object.assign(createShot(1, 'Imported endless base'), copy(out.base_snapshot.shot || {})),
                    out.target_seconds, out.segment_seconds, out.continuation_direction);
                if (!endlessSnapshotsEqual(replay, out.base_snapshot) ||
                    canonicalEndlessSnapshot(out.segment_output_frames) !== canonicalEndlessSnapshot(out.base_snapshot.segment_output_frames))
                    throw new Error('snapshot mismatch');
            } catch (_) {
                out.status = 'failed'; out.error = 'Imported endless run failed its immutable snapshot contract.'; out.active_job = null;
            }
        }
        return out;
    }

    function normalizeReference(value) {
        var out = Object.assign(createReference(), value || {});
        out.kind = ['image', 'video', 'audio'].indexOf(out.kind) >= 0 ? out.kind : 'image';
        out.role = ['subject', 'source_video', 'keyframe', 'motion_camera', 'environment_style'].indexOf(out.role) >= 0 ? out.role : 'subject';
        out.audio_use = ['reference', 'reuse', 'voice_timbre'].indexOf(out.audio_use) >= 0 ? out.audio_use : 'reference';
        out.duration_seconds = Number(out.duration_seconds) || 5;
        return out;
    }

    function normalizeShot(value, fallbackId) {
        var out = Object.assign(createShot(fallbackId), value || {});
        out.id = Number(out.id) || fallbackId;
        out.duration_seconds = Number(out.duration_seconds) || 8;
        out.width = Number(out.width) || 1344;
        out.height = Number(out.height) || 768;
        out.steps = Number(out.steps) || 20;
        out.seed = Number.isFinite(Number(out.seed)) ? Number(out.seed) : 42;
        out.quant = ['int8-fast', 'int8', 'bf16'].indexOf(out.quant) >= 0 ? out.quant : 'int8-fast';
        out.attention_backend = normalizeAttentionBackend(
            out.quant, out.attention_backend);
        out.step_cache = out.step_cache === 'high' ? 'high' : 'exact';
        out.motion_context_frames = [5, 22, 39].indexOf(Number(out.motion_context_frames)) >= 0 ? Number(out.motion_context_frames) : 22;
        out.references = Array.isArray(out.references) ? out.references.map(normalizeReference) : [];
        out.take_job_ids = Array.isArray(out.take_job_ids) ? out.take_job_ids : [];
        out.take_states = Array.isArray(out.take_states) ? out.take_states : [];
        out.take_output_paths = Array.isArray(out.take_output_paths) ? out.take_output_paths : [];
        out.selected_take = Number.isFinite(Number(out.selected_take)) ? Number(out.selected_take) : -1;
        out.locked = out.locked === true;
        return out;
    }

    function normalizeProject(value) {
        if (!value || value.schema !== PROJECT_SCHEMA)
            throw new Error('Unsupported H3 project schema');
        var base = createProject();
        var out = Object.assign(base, value);
        out.project_kind = [0, 1, 2].indexOf(Number(out.project_kind)) >= 0 ? Number(out.project_kind) : 1;
        out.caption_tier = Number(out.caption_tier) === 1 ? 1 : 0;
        out.delivery_fps = [24, 25, 30].indexOf(Number(out.delivery_fps)) >= 0 ? Number(out.delivery_fps) : 24;
        out.target_duration_seconds = Math.max(5, Number(out.target_duration_seconds) || 120);
        out.takes_per_shot = Math.max(1, Math.min(8, Number(out.takes_per_shot) || 1));
        out.adult_mode = out.adult_mode === true;
        out.assets = Array.isArray(out.assets) ? out.assets : [];
        out.character_sheets = Array.isArray(out.character_sheets) ? out.character_sheets : [];
        out.shots = Array.isArray(out.shots) && out.shots.length
            ? out.shots.map(function (shot, index) { return normalizeShot(shot, index + 1); })
            : [createShot(1, 'Opening shot')];
        out.next_shot_id = Number(out.next_shot_id) || (Math.max.apply(null, out.shots.map(function (shot) { return shot.id; })) + 1);
        out.next_asset_id = Number(out.next_asset_id) || 1;
        out.endless = normalizeEndless(out.endless, out.target_duration_seconds);
        if (out.endless.completed_job_ids.length) {
            var completedChainValid = out.endless.completed_job_ids.every(function (jobId, index) {
                var mapped = out.shots.find(function (shot) { return shot.id === Number(out.endless.segment_shot_ids[index]); });
                return mapped && mapped.take_job_ids.indexOf(jobId) >= 0 && mapped.selected_take >= 0;
            });
            if (!completedChainValid) {
                out.endless.status = 'failed'; out.endless.error = 'Imported endless run is missing a completed shot from its continuity spine.'; out.endless.active_job = null;
            }
        }
        if (['submitting', 'running', 'stopping'].indexOf(out.endless.status) >= 0) {
            var endlessBase = out.shots.find(function (shot) { return shot.id === Number(out.endless.base_shot_id); });
            try {
                if (!endlessBase || !endlessSnapshotsEqual(
                    endlessBaseSnapshot(endlessBase, out.endless.target_seconds, out.endless.segment_seconds, out.endless.continuation_direction),
                    out.endless.base_snapshot)) throw new Error('base mismatch');
            } catch (_) {
                out.endless.status = 'failed'; out.endless.error = 'Imported endless run no longer matches its base inference snapshot.'; out.endless.active_job = null;
            }
        }
        return out;
    }

    function validateEndlessSettings(targetSeconds, segmentSeconds, direction) {
        var target = Number(targetSeconds);
        var segment = Number(segmentSeconds);
        if (!Number.isFinite(target) || target < MIN_SECONDS || target > MAX_ENDLESS_SECONDS)
            return 'Endless target must stay between 5 seconds and 60 minutes.';
        if (!Number.isFinite(segment) || segment < MIN_SECONDS || segment > MAX_SECONDS)
            return 'Each endless segment must stay between 5 and 15 seconds.';
        if (!String(direction || '').trim())
            return 'Write an explicit continuation direction before starting.';
        return '';
    }

    function planEndlessDurations(targetSeconds, segmentSeconds) {
        var issue = validateEndlessSettings(targetSeconds, segmentSeconds, 'direction');
        if (issue) throw new Error(issue);
        return planEndlessFrames(targetSeconds, segmentSeconds).map(function (frames) { return frames / NATIVE_FPS; });
    }

    function planEndlessFrames(targetSeconds, segmentSeconds) {
        var issue = validateEndlessSettings(targetSeconds, segmentSeconds, 'direction');
        if (issue) throw new Error(issue);
        var target = Math.round(Number(targetSeconds) * NATIVE_FPS);
        var preferred = Math.round(Number(segmentSeconds) * NATIVE_FPS);
        var count = Math.ceil(target / preferred);
        count = Math.min(count, Math.floor(target / MIN_SEGMENT_FRAMES));
        count = Math.max(count, Math.ceil(target / MAX_SEGMENT_FRAMES), 1);
        var remaining = target;
        var frames = [];
        for (var i = 0; i < count; i++) {
            var slots = count - i - 1;
            var low = Math.max(MIN_SEGMENT_FRAMES, remaining - slots * MAX_SEGMENT_FRAMES);
            var high = Math.min(MAX_SEGMENT_FRAMES, remaining - slots * MIN_SEGMENT_FRAMES);
            var next = Math.max(low, Math.min(high, preferred));
            frames.push(next); remaining -= next;
        }
        if (remaining !== 0 || frames.some(function (value) { return !Number.isInteger(value) || value < MIN_SEGMENT_FRAMES || value > MAX_SEGMENT_FRAMES; }))
            throw new Error('Endless duration cannot be divided into valid 120–360 frame H3 segments.');
        return frames;
    }

    function stableValue(value) {
        if (Array.isArray(value)) return value.map(stableValue);
        if (value && typeof value === 'object') {
            var out = {};
            Object.keys(value).sort().forEach(function (key) { out[key] = stableValue(value[key]); });
            return out;
        }
        return value;
    }

    function endlessFingerprint(snapshot) {
        var text = canonicalEndlessSnapshot(snapshot);
        var hash = 2166136261;
        for (var i = 0; i < text.length; i++) {
            hash ^= text.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }
        return 'fnv1a32-' + (hash >>> 0).toString(16).padStart(8, '0');
    }

    function canonicalEndlessSnapshot(snapshot) {
        return JSON.stringify(stableValue(snapshot));
    }

    function endlessSnapshotsEqual(left, right) {
        return canonicalEndlessSnapshot(left) === canonicalEndlessSnapshot(right);
    }

    function endlessBaseSnapshot(shot, targetSeconds, segmentSeconds, direction) {
        var issue = validateEndlessSettings(targetSeconds, segmentSeconds, direction);
        if (issue) throw new Error(issue);
        var frames = planEndlessFrames(targetSeconds, segmentSeconds);
        var durations = frames.map(function (value) { return value / NATIVE_FPS; });
        var base = {
            width: Number(shot.width), height: Number(shot.height),
            steps: Number(shot.steps), seed: Number(shot.seed), quant: String(shot.quant || ''),
            attention_backend: String(shot.attention_backend || ''), step_cache: String(shot.step_cache || ''),
            brief: String(shot.brief || ''), shot_description: String(shot.shot_description || ''),
            soundscape: String(shot.soundscape || ''), music: String(shot.music || ''),
            subject_definitions: String(shot.subject_definitions || ''), summary: String(shot.summary || ''),
            retention_analysis: String(shot.retention_analysis || ''), prompt_override: String(shot.prompt_override || ''),
            first_frame: String(shot.first_frame || ''), last_frame: String(shot.last_frame || ''),
            continue_from: String(shot.continue_from || ''), motion_context_frames: Number(shot.motion_context_frames) || 22,
            references: Array.isArray(shot.references) ? shot.references.map(copy) : []
        };
        var first = Object.assign(createShot(base.id, base.title), copy(base));
        first.duration_seconds = durations[0];
        var shotIssue = validateShot(first);
        if (shotIssue) throw new Error(shotIssue);
        return {
            schema: ENDLESS_SCHEMA,
            target_frames: frames.reduce(function (sum, value) { return sum + value; }, 0),
            segment_frames: Math.round(Number(segmentSeconds) * NATIVE_FPS),
            segment_output_frames: frames,
            target_seconds: frames.reduce(function (sum, value) { return sum + value; }, 0) / NATIVE_FPS,
            segment_seconds: Math.round(Number(segmentSeconds) * NATIVE_FPS) / NATIVE_FPS,
            segment_durations: durations,
            continuation_direction: String(direction).trim(),
            first_segment_mode: detectMode(first), first_segment_prompt: compilePrompt(first),
            shot: base
        };
    }

    function createEndlessRun(shot, targetSeconds, segmentSeconds, direction) {
        var snapshot = endlessBaseSnapshot(shot, targetSeconds, segmentSeconds, direction);
        var out = createEmptyEndless(targetSeconds);
        out.status = 'running';
        out.target_seconds = snapshot.target_seconds;
        out.target_frames = snapshot.target_frames;
        out.segment_seconds = snapshot.segment_seconds;
        out.segment_frames = snapshot.segment_frames;
        out.continuation_direction = snapshot.continuation_direction;
        out.base_shot_id = Number(shot.id);
        out.segment_durations = snapshot.segment_durations.slice();
        out.segment_output_frames = snapshot.segment_output_frames.slice();
        out.base_snapshot = snapshot;
        out.fingerprint = endlessFingerprint(snapshot);
        out.started_at = new Date().toISOString();
        out.updated_at = out.started_at;
        for (var i = 0; i < out.segment_output_frames.length; i++) {
            var probe = copy(out);
            probe.completed_job_ids = [];
            for (var j = 0; j < i; j++) probe.completed_job_ids.push('video-' + String(j + 1).padStart(4, '0'));
            renderRequest(endlessSegmentShot(probe, i));
        }
        return out;
    }

    function endlessSegmentShot(run, segmentIndex) {
        if (!run || !run.base_snapshot || !run.base_snapshot.shot)
            throw new Error('Endless run has no saved base snapshot.');
        var index = Number(segmentIndex);
        if (!Number.isInteger(index) || index < 0 || index >= run.segment_durations.length)
            throw new Error('Endless segment index is outside the saved plan.');
        var base = run.base_snapshot.shot;
        var shot;
        if (index === 0) {
            shot = Object.assign(createShot(run.base_shot_id, 'Endless opening'), copy(base));
        } else {
            shot = Object.assign(createShot(0, 'Endless continuation ' + (index + 1)), copy(base));
            shot.width = Number(base.width); shot.height = Number(base.height);
            shot.steps = Number(base.steps); shot.quant = base.quant;
            shot.attention_backend = base.attention_backend; shot.step_cache = base.step_cache;
            shot.brief = run.continuation_direction;
            shot.shot_description = run.continuation_direction;
            shot.soundscape = base.soundscape; shot.music = base.music;
            shot.references = Array.isArray(base.references) ? base.references.map(copy) : [];
            shot.prompt_override = '';
            shot.first_frame = ''; shot.last_frame = '';
            shot.continue_from = run.completed_job_ids[index - 1] || '';
            shot.motion_context_frames = 22;
        }
        shot.duration_seconds = Number(run.segment_output_frames[index]) / NATIVE_FPS;
        shot.seed = (Number(base.seed) + index) >>> 0;
        return shot;
    }

    function recordEndlessSegmentTake(project, run, segmentIndex, videoId, src) {
        var urls = videoJobUrls(videoId);
        if (!urls) throw new Error('Endless completion has an invalid video job id.');
        var index = Number(segmentIndex);
        if (!Number.isInteger(index) || index < 0 || index >= run.segment_output_frames.length)
            throw new Error('Endless completion is outside the saved frame plan.');
        var base = project.shots.find(function (item) { return item.id === Number(run.base_shot_id); });
        if (!base) throw new Error('The base shot for this endless run no longer exists.');
        var generated = endlessSegmentShot(run, index);
        var shot;
        if (index === 0) {
            shot = base;
            generated.id = shot.id; generated.title = shot.title;
        } else {
            var mappedId = Number(run.segment_shot_ids[index]);
            shot = project.shots.find(function (item) { return item.id === mappedId; }) || null;
            if (!shot) {
                generated.id = project.next_shot_id++;
                var previousId = Number(run.segment_shot_ids[index - 1] || run.base_shot_id);
                var previousIndex = project.shots.findIndex(function (item) { return item.id === previousId; });
                if (previousIndex < 0) throw new Error('The prior endless shot is missing from the continuity spine.');
                shot = generated;
                project.shots.splice(previousIndex + 1, 0, shot);
            }
            generated.id = shot.id;
            generated.title = base.title + ' · continuation ' + (index + 1);
        }
        var takeIds = Array.isArray(shot.take_job_ids) ? shot.take_job_ids : [];
        var takeStates = Array.isArray(shot.take_states) ? shot.take_states : [];
        var takePaths = Array.isArray(shot.take_output_paths) ? shot.take_output_paths : [];
        Object.assign(shot, generated);
        shot.take_job_ids = takeIds; shot.take_states = takeStates; shot.take_output_paths = takePaths;
        var take = shot.take_job_ids.indexOf(urls.video_id);
        if (take < 0) {
            shot.take_job_ids.push(urls.video_id); shot.take_states.push('done'); shot.take_output_paths.push(String(src || ''));
            take = shot.take_job_ids.length - 1;
        } else {
            shot.take_states[take] = 'done'; shot.take_output_paths[take] = String(src || '');
        }
        shot.selected_take = take; shot.status = 'Ready'; shot.output_path = String(src || '');
        run.segment_shot_ids[index] = shot.id;
        return shot;
    }

    function projectKindLabel(kind) {
        return Number(kind) === 2 ? 'Brand commercial' : (Number(kind) === 0 ? 'Single clip' : 'Movie');
    }

    function outputFrames(shot) {
        return Math.round(NATIVE_FPS * Number(shot.duration_seconds));
    }

    function internalFrames(shot) {
        var frames = outputFrames(shot);
        if (String(shot.continue_from || '').trim())
            frames += Number(shot.motion_context_frames) || 22;
        while (frames % 17 !== 5)
            frames += 1;
        return frames;
    }

    function detectMode(shot) {
        if (String(shot.continue_from || '').trim()) return 'continue';
        if (shot.references && shot.references.length) return 'ref2va';
        if (String(shot.first_frame || '').trim() && String(shot.last_frame || '').trim()) return 'fl2va';
        if (String(shot.first_frame || '').trim()) return 'i2va';
        if (String(shot.last_frame || '').trim()) return 'l2va';
        return 't2va';
    }

    function modeLabel(mode) {
        var labels = {
            t2va: 'From story', i2va: 'Opening frame', l2va: 'Ending frame',
            fl2va: 'Frame-to-frame', ref2va: 'Directed references', continue: 'Native continuation'
        };
        return labels[mode] || labels.t2va;
    }

    function referenceSourceLabel(references, index) {
        var counts = { image: 0, video: 0, audio: 0 };
        for (var i = 0; i <= index; i++)
            counts[references[i].kind] = (counts[references[i].kind] || 0) + 1;
        var kind = references[index].kind;
        return kind === 'image' ? '<Picture ' + counts.image + '>'
            : (kind === 'video' ? '<Video ' + counts.video + '>' : '<Audio ' + counts.audio + '>');
    }

    function referenceLabel(references, index) {
        var item = references[index];
        if (item.kind === 'audio' || (item.role !== 'subject' && item.role !== 'environment_style'))
            return referenceSourceLabel(references, index);
        var subject = 0;
        for (var i = 0; i <= index; i++) {
            if (references[i].kind !== 'audio' && (references[i].role === 'subject' || references[i].role === 'environment_style'))
                subject += 1;
        }
        return '<Subject ' + subject + '>';
    }

    function planText(shot) {
        return String(shot.shot_description || shot.brief || 'Describe the visible action, camera movement, dialogue, and synchronized sound for this shot.');
    }

    function refDefinitions(shot) {
        if (String(shot.subject_definitions || '').trim()) return shot.subject_definitions;
        return (shot.references || []).map(function (item, index) {
            var label = referenceLabel(shot.references, index);
            var source = referenceSourceLabel(shot.references, index);
            var note = String(item.note || '').trim() || ('the ordered ' + item.kind + ' reference at ' + item.path);
            if (label !== source) return label + ' is ' + note + ' from ' + source + ', used as reusable visible content in the target video.';
            if (item.kind === 'image') return label + ' is ' + note + ', used as a concrete keyframe or composition anchor.';
            if (item.kind === 'video' && item.role === 'source_video') return label + ' is the source video for the target-video edit: ' + note + '.';
            if (item.kind === 'video') return label + ' provides whole-video motion, camera, cut, or pacing structure: ' + note + '.';
            return label + ' is the audio ' + item.audio_use + ' reference: ' + note + '.';
        }).join('\n');
    }

    function refRetention(shot) {
        if (String(shot.retention_analysis || '').trim()) return shot.retention_analysis;
        return (shot.references || []).map(function (item, index) {
            var label = referenceLabel(shot.references, index);
            var relation = 'fully_preserved';
            var explanation = 'the referenced visual identity and defining attributes remain consistent in the target shot.';
            if (item.kind === 'video' && item.role === 'source_video') {
                relation = 'partially_preserved';
                explanation = 'the source duration, staging, camera, timing, lighting, and unchanged content remain locked while only the requested elements are regenerated.';
            } else if (item.kind === 'video' && item.role !== 'subject' && item.role !== 'environment_style') {
                relation = 'weak_reference';
                explanation = 'the source movement, rhythm, and camera behavior guide the target shot without copying the complete source video.';
            } else if (item.kind === 'audio') {
                relation = item.audio_use === 'reuse' ? 'partially_copy' : 'reference';
                explanation = item.audio_use === 'reuse'
                    ? 'the referenced signal is reused where instructed in the target soundtrack.'
                    : 'the signal guides timbre, delivery, rhythm, or sound texture without copying unintended content.';
            }
            return label + ': ' + relation + ' - ' + explanation;
        }).join('\n');
    }

    function compilePrompt(shot) {
        if (String(shot.prompt_override || '').trim()) return shot.prompt_override;
        var mode = detectMode(shot);
        var plan = planText(shot);
        var sound = String(shot.soundscape || '').trim() || 'N/A';
        var music = String(shot.music || '').trim() || 'N/A';
        if (mode === 'ref2va' || (mode === 'continue' && shot.references.length)) {
            var editing = shot.references.some(function (item) { return item.kind === 'video' && item.role === 'source_video'; });
            var summary = String(shot.summary || '').trim() || ((editing ? '[video editing + reference generation] The target video is an edited version of <Video 1>. ' : '[reference generation] ') + String(shot.brief || ''));
            return 'subject_definitions:\n' + refDefinitions(shot) +
                '\n\nsummary:\n' + summary +
                '\n\nretention_analysis:\n' + refRetention(shot) +
                '\n\ndetailed_description:\n[Shot 1] ' + plan +
                '\n\noverall_soundscape:\n' + sound +
                '\n\nnon_diegetic_music:\n' + music;
        }
        var prefix = '';
        var endpoint = secondsText(shot.duration_seconds);
        if (mode === 'i2va')
            prefix = 'For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n';
        else if (mode === 'fl2va')
            prefix = 'How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the ' + endpoint + '-second mark of the target video.\n\n';
        else if (mode === 'l2va')
            prefix = 'How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the ' + endpoint + '-second mark of the target video.\n\n';
        return prefix + 'integrated_multimodal_description: [Shot 1] ' + plan +
            '\n\noverall_soundscape: ' + sound +
            '\n\nnon_diegetic_music: ' + music;
    }

    function occurrences(text, needle) {
        return String(text).split(needle).length - 1;
    }

    function promptComplianceIssue(prompt, mode, duration) {
        if (!String(prompt || '').trim()) return 'Prompt is empty.';
        var refSections = ['subject_definitions:', 'summary:', 'retention_analysis:', 'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:'];
        var baseSections = ['integrated_multimodal_description:', 'overall_soundscape:', 'non_diegetic_music:'];
        var sections = mode === 'ref2va' || prompt.indexOf('subject_definitions:') === 0 ? refSections : baseSections;
        if (sections.some(function (section) { return occurrences(prompt, section) !== 1; }))
            return sections === refSections ? 'Ref2VA requires each canonical section exactly once.' : 'Base H3 modes require each canonical section exactly once.';
        var positions = sections.map(function (section) { return prompt.indexOf(section); });
        for (var i = 1; i < positions.length; i++) {
            if (positions[i] <= positions[i - 1]) return (sections === refSections ? 'Ref2VA' : 'Base H3') + ' sections are out of canonical order.';
        }
        if (prompt.indexOf('[Shot 1]') < 0) return 'The prompt must begin its playback description with [Shot 1].';
        var endpoint = secondsText(duration);
        if (mode === 'i2va' && prompt.indexOf('For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.') !== 0)
            return 'I2VA first-frame alignment instruction is missing or altered.';
        if (mode === 'fl2va' && (prompt.indexOf('How the reference pictures align with the target video') !== 0 || prompt.indexOf(endpoint + '-second mark of the target video.') < 0))
            return 'FL2VA opening/final alignment instruction is missing or has the wrong endpoint.';
        if (mode === 'l2va' && (prompt.indexOf('How the reference pictures align with the target video') !== 0 || prompt.indexOf(endpoint + '-second mark of the target video.') < 0))
            return 'L2VA final-frame alignment instruction is missing or has the wrong endpoint.';
        var soundBody = String(prompt).split('overall_soundscape:')[1];
        if (soundBody && soundBody.split('non_diegetic_music:')[0].indexOf('<d>') >= 0)
            return 'Dialogue must stay in the timed description, not overall_soundscape.';
        if (occurrences(prompt, '<d>') !== occurrences(prompt, '</d>')) return 'Dialogue tags are unbalanced.';
        return '';
    }

    function validateShot(shot) {
        var seconds = Number(shot.duration_seconds);
        if (seconds < MIN_SECONDS || seconds > MAX_SECONDS) return 'Shot duration must stay between 5 and 15 seconds.';
        if (shot.width < 32 || shot.height < 32 || shot.width % 32 || shot.height % 32) return 'Shot dimensions must be positive multiples of 32.';
        if (shot.width * shot.height > MAX_PIXELS) return 'Shot resolution exceeds H3’s 24 GB product envelope.';
        if (!String(shot.brief || shot.shot_description || shot.prompt_override || '').trim()) return 'Write a shot brief or detailed shot plan before rendering.';
        var mode = detectMode(shot);
        var compliance = promptComplianceIssue(compilePrompt(shot), mode, seconds);
        if (compliance) return 'H3 prompt contract: ' + compliance;
        if (mode === 'continue' && String(shot.continue_from).indexOf('video-') !== 0) return 'Continuation source must be an H3 video job such as video-0100.';
        if ((shot.references || []).length > 12) return 'H3 accepts at most 12 ordered references.';
        var counts = { image: 0, video: 0, audio: 0 }, durations = { video: 0, audio: 0 };
        for (var i = 0; i < shot.references.length; i++) {
            var item = shot.references[i];
            if (!String(item.path || '').trim()) return 'Every ordered reference needs a server-uploaded file path.';
            counts[item.kind] = (counts[item.kind] || 0) + 1;
            if (item.kind === 'video' || item.kind === 'audio') {
                if (Number(item.duration_seconds) < 2 || Number(item.duration_seconds) > 15) return 'Each ' + item.kind + ' reference must be 2 through 15 seconds.';
                durations[item.kind] += Number(item.duration_seconds);
            }
        }
        if (counts.image > 9 || counts.video > 3 || counts.audio > 3) return 'Reference budget is 9 images, 3 videos, and 3 audio files.';
        if (counts.audio && !counts.image && !counts.video) return 'Audio references require at least one visual reference.';
        if (durations.video > 15) return 'Combined video-reference duration must not exceed 15 seconds.';
        if (durations.audio > 15) return 'Combined audio-reference duration must not exceed 15 seconds.';
        return '';
    }

    function renderRequest(shot) {
        var issue = validateShot(shot);
        if (issue) throw new Error(issue);
        var mode = detectMode(shot);
        var request = {
            schema: 'serenity.h3.render.v1', model: 'minimax_h3', runner: 'minimax_h3_mojo_request',
            task: mode, prompt: compilePrompt(shot), width: shot.width, height: shot.height,
            duration_seconds: Number(shot.duration_seconds), fps: NATIVE_FPS, frames: internalFrames(shot),
            output_frames: outputFrames(shot), steps: Number(shot.steps) || 20, seed: Number(shot.seed),
            include_audio: true, quant: shot.quant || 'int8-fast',
            attention_backend: normalizeAttentionBackend(
                shot.quant, shot.attention_backend),
            step_cache: shot.step_cache === 'high' ? 'high' : 'exact'
        };
        if (shot.first_frame) request.source_image = shot.first_frame;
        if (shot.last_frame) request.last_frame = shot.last_frame;
        if (shot.continue_from) {
            request.continue_from = shot.continue_from;
            request.motion_context_frames = Number(shot.motion_context_frames) || 22;
            request.trim_start_frames = request.motion_context_frames;
        }
        if (shot.references.length) request.references = shot.references.map(copy);
        return request;
    }

    function directorMinimumShots(project) {
        return Math.ceil(Number(project.target_duration_seconds) / 15);
    }

    function directorMaximumShots(project) {
        return Math.ceil(Number(project.target_duration_seconds) / 5);
    }

    function actionById(id) {
        return ACTIONS.find(function (action) { return action.id === id; }) || ACTIONS[0];
    }

    function hasKind(shot, kind) {
        return shot.references.some(function (item) { return item.kind === kind; });
    }

    function validateDirector(project, shot, actionId) {
        var hasVideo = hasKind(shot, 'video') || !!shot.output_path || !!shot.continue_from;
        var hasImage = hasKind(shot, 'image') || !!shot.first_frame || !!shot.last_frame;
        var hasAudio = hasKind(shot, 'audio');
        var hasSourceVideo = !!shot.output_path || !!shot.continue_from || shot.references.some(function (item) { return item.kind === 'video' && item.role === 'source_video'; });
        var hasReplacement = shot.references.some(function (item) { return (item.kind === 'image' || item.kind === 'video') && item.role !== 'source_video'; });
        var hasDirection = !!String(project.director_brief || shot.brief || shot.shot_description || shot.prompt_override || '').trim();
        if (actionId === 'character_sheet') {
            if (shot.references.some(function (item) { return item.kind !== 'image'; })) return 'Character Sheet accepts image references only; remove video/audio references from this shot first.';
            if (shot.references.length < 1 || shot.references.length > 9) return 'Character Sheet needs between 1 and 9 ordered image references.';
        }
        if (actionId === 'caption_shot' && !hasVideo) return 'Caption Shot needs a video reference or selected rendered take.';
        if (actionId === 'prompt_doctor' && !hasDirection) return 'Prompt Doctor needs an existing prompt or detailed intent.';
        if (actionId === 'retake_doctor' && (!hasVideo || !hasDirection)) return 'Retake Doctor needs a take/source video and the intended prompt.';
        if (actionId === 'replace_restage' && (!hasSourceVideo || !hasReplacement || !hasDirection)) return 'Replace / Restage needs a source-video role, a replacement visual, and an exact change request.';
        if (actionId === 'reference_director' && !shot.references.length) return 'Reference Director needs at least one ordered reference.';
        if (actionId === 'keyframe_bridge' && !hasImage) return 'Keyframe Bridge needs an opening, ending, or keyframe image.';
        if ((actionId === 'dream_project' || actionId === 'script_to_coverage') && !String(project.director_brief || '').trim()) return 'Write the project outline or script in Director input first.';
        if (actionId === 'continue_story' && (!hasVideo || !hasDirection)) return 'Continue Story needs a source clip/take and direction for what follows.';
        if (actionId === 'sound_to_scene' && (!hasAudio || !hasDirection)) return 'Sound to Scene needs an audio reference and visual or story direction.';
        return '';
    }

    function responseContract(actionId) {
        if (actionId === 'character_sheet') return 'Return one JSON object with schema serenity.h3.character_sheet.result.v1, operation=character_sheet, name, panel_count, style_mode, source_repo, source_revision, source_template, reference_descriptions, identity_lock, a_prompt, b_prompt, h3_prompt, internal_frames, frame_indices, planned_seconds, render_admitted, sheet_layout, reusable_reference_plan, compliance, and warnings. reference_descriptions must contain exactly one ordered entry per supplied image. a_prompt must contain exactly one <Picture N> line per image; each line must state what to keep/use, what to ignore/remove, and concrete clothing/material details when visible. h3_prompt must use the canonical Ref2VA six-section order and identity_lock must be suitable for the project continuity bible.';
        var common = 'Return one JSON object with schema serenity.h3.director.result.v2, operation, summary, continuity_bible, reference_plan, shot_count, takes_per_shot, generation_count, shots, compliance, and warnings. Each shots item must contain order, title, duration_seconds, mode, brief, references, h3_prompt, continuity_in, continuity_out, and takes. generation_count must equal the sum of takes across shots.';
        var extras = {
            caption_shot: ' Also return literal_caption, source_shot_map, dialogue_ledger, and source_fidelity_notes.',
            prompt_doctor: ' Also return prompt_diagnosis, preserved_intent, changes, and revised_prompt.',
            retake_doctor: ' Also return mismatch_diagnosis, failure_timeline, locked_elements, and retry_changes_only.',
            replace_restage: ' Also return edit_target, preservation_locks, replacement_bindings, and changed_elements.',
            reference_director: ' Also return subject_definitions, role_map, retention_analysis, and unresolved_references.',
            keyframe_bridge: ' Also return alignment_instruction, start_state, transition_path, and end_state.',
            dream_project: ' Also return logline, story_beats, scene_plan, character_bible, asset_plan, and edit_plan.',
            script_to_coverage: ' Also return scene_breakdown, dialogue_ledger, coverage_plan, and edit_plan.',
            continue_story: ' Also return source_end_state, seam_plan, next_beats, and continuity_changes.',
            sound_to_scene: ' Also return audio_timeline, beat_map, dialogue_ledger, visual_sync_plan, and audio_reuse_plan.'
        };
        return common + (extras[actionId] || '');
    }

    function systemPrompt(project, shot, actionId) {
        var action = actionById(actionId);
        var operationRules = action.help;
        if (actionId === 'character_sheet') operationRules += ' Analyze the same identity across every ordered image. Write exactly one A-prompt line per <Picture N>. Each line must explicitly say which face, body, hair, clothing, accessory, prop, color, material, or style attributes to keep/use and which background, extra person, wrong hair, unwanted accessory, pose, or artifact to ignore/remove. Describe wardrobe and materials in words instead of merely saying to copy a picture. Merge contributions into one <Subject 1>, flag real conflicts instead of silently averaging them, and preserve the supplied upstream B prompt verbatim. Do not invent a LoRA, training step, or additional model.';
        if (actionId === 'replace_restage') operationRules += ' Treat the first source_video role as the clip being regenerated. Lock its duration, cuts, camera path, blocking, lighting, environment, sound, and every unrequested identity or attribute. Change only the named person, clothing, prop, or setting.';
        if (actionId === 'dream_project' || actionId === 'script_to_coverage') operationRules += ' Design a complete editable project, not one oversized prompt. Divide the target runtime into coherent 5–15 second render units, calculate the exact number of shots and generations, and give every unit its own standalone compliant H3 prompt plus continuity-in and continuity-out state.';
        var adult = project.adult_mode
            ? 'Lawful consensual adult 18+ content is in scope. Describe visible adult anatomy, clothing state, intimacy, camera, action, and sound concretely when requested. Never create sexual content involving minors, coercion, or exploitative non-consent.'
            : 'Do not introduce sexual content. Describe sensitive visible content neutrally when needed for fidelity.';
        return 'You are Serenity Director, a Qwen3-VL filmmaking captioner and prompt architect for MiniMax H3. ' + operationRules + '\n\nReturn only valid JSON matching the response contract. Do not wrap it in Markdown. Do not propose LoRAs, training, model changes, or unsupported controls. Write rewrite sections in English while preserving dialogue, lyrics, and visible text in their original language. Every render unit must be 5–15 seconds at H3’s native 24 fps. Choose T2VA, I2VA, FL2VA, L2VA, or Ref2VA from the actual inputs. Base prompts use integrated_multimodal_description, overall_soundscape, non_diegetic_music exactly once and in that order. Ref2VA uses subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music exactly once and in that order. Use <Subject N> for reusable visible content, <Picture N> for concrete frames, <Video N> for temporal structure, and <Audio N> for signals. Start playback with [Shot 1]; later cuts use increasing timestamps. Track face, body, wardrobe, props, location, lighting, eyeline, screen direction, pose, motion, camera, dialogue, ambience, and score across shots. ' + adult;
    }

    function characterSheetBPrompt(panelCount, styleMode) {
        var style = styleMode === 'anime_to_real'
            ? '[STYLE]\nFully photorealistic live-action, unretouched studio photograph. Completely natural bare face and body — zero makeup of any kind. Natural lip color, natural eyelashes and eyebrows only. Skin shows real texture. Mild natural film grain. Style never drifts.'
            : '[STYLE]\nThe output is matches the style of <Picture 1>. Sharp detail on eyes and face. The style never changes and never drifts between shots. No shadows.';
        var staging = '[STAGING]\nSolid light grey seamless backdrop, one flat uniform tone edge to edge, with no gradient, no vignette, no texture and no floor line. Nothing else is in frame. the subject casts no shadow onto the backdrop and no contact shadow on the ground beneath it, and it does not sit in its own shadow. Soft form shading on the subject itself is fine and should read its shape. Long telephoto lens, near-orthographic.\n\nThe character holds one relaxed A-pose throughout: arms hanging slightly away from the body, palms toward the thighs, feet shoulder-width apart, head level, calm neutral expression, eyes open and looking forward.\nThe subject is completely frozen, as rigid and motionless as a statue. Only the camera moves. Hair, fabric, cloaks, skirts, sleeves, straps, ribbons, chains, tassels, fur and feathers are all locked solid: every strand and every fold sits in exactly the same position in every frame. There is no wind, no breeze, no air movement, no breathing, no settling, no sway, no secondary motion of any kind. Orientation, surfaces and lighting are identical in every shot, and the subject stays the same size in frame.';
        var orbit = Number(panelCount) === 4
            ? '[0-2 seconds] tight full shot of the subject. The camera makes one smooth fixed-speed orbit to the left around it, sweeping 180 degrees: starting square on the back, passing the left side a third of the way through this move, and ending square on the front at 2 seconds. The subject does not move at all. Ends on the back view at 2 seconds.\n\n[2-3 seconds] Camera whip-pans front and snaps into a fast push-in on the character\'s upper body. Locked-off head and shoulders close-up, face square to camera, eyes into the lens. Ends on a sharp front-on face at 3 seconds.\n\n[CAMERA] One constant-speed 180-degree orbit in beat 1, then a hard whip-pan back to front and a push-in for beat 2. No zoom, no dolly, no tilt, no roll, no handheld shake, no motion blur, no dissolves beyond the one whip-pan cut.\n[AUDIO] Silence. No music, no room tone, no voices.'
            : '[0-3 seconds] tight full shot of the subject. The camera makes one smooth fixed-speed orbit right around it, a full 360 degrees: starting square on the front, passing the left side a quarter of the way round, directly behind at halfway, the right side three quarters of the way round, and returning to the front. The subject does not move at all. Ends back on the front view at 3 seconds.\n\n[3-4 seconds] Camera snaps into a fast push-in on the character\'s face. Locked-off head and shoulders close-up, face square to camera, eyes into the lens. Ends on a sharp front-on face.\n\n[4-5 seconds] camera whip-pans and rotates to a orthogonal angle. Locked-off head and shoulders close-up, head turned to a three-quarter angle, eyes still forward. Ends on a clean three-quarter face.\n\n[CAMERA] One constant-speed orbit in beat 1, then locked off and static. The camera is the only thing in the scene that moves at any point. No zoom, no push in, no dolly, no tilt, no roll, no handheld shake, no motion blur, no dissolves.\n[AUDIO] Silence. No music, no room tone, no voices.';
        return style + '\n\n\n' + staging + '\n\n' + orbit;
    }

    function characterSheetFromShot(shot, panelCount, styleMode, notes) {
        var count = Number(panelCount) === 4 ? 4 : 6;
        var refs = shot.references.filter(function (item) { return item.kind === 'image'; });
        var aPrompt = refs.map(function (item, index) {
            var note = String(item.note || '').trim() || 'the clearly visible face, body proportions, hair, wardrobe, accessories, props, colors, materials, and source style that belong to the intended character';
            return '<Picture ' + (index + 1) + '> - use ' + note + '. Keep only the named identity and attributes, describing clothing and materials concretely. Ignore the background, unrelated people, framing, pose, text, and unrequested accessories or artifacts.';
        }).join('\n');
        var identityLock = 'One identity across every panel: preserve the same face geometry, body proportions, skin tone, hair, wardrobe construction, colors, materials, accessories, props, and source style unless a per-picture directive explicitly replaces an attribute.' + (notes ? ' Director notes: ' + notes : '');
        var frames = count === 4 ? 73 : 124;
        var indices = count === 4 ? [2, 24, 45, 68] : [2, 21, 42, 63, 84, 113];
        var template = (styleMode === 'anime_to_real' ? 'Anime2Real' : 'Standard Orbit') + ' - ' + count + ' panel.txt';
        var bPrompt = characterSheetBPrompt(count, styleMode);
        var pictureSources = refs.map(function (_, index) { return '<Picture ' + (index + 1) + '>'; }).join(', ');
        var h3Prompt = 'subject_definitions:\n<Subject 1> is the single identity assembled from ' + pictureSources + ' under these per-picture extraction directives:\n' + aPrompt +
            '\n\nsummary:\n[reference generation] Generate one silent ' + count + '-panel identity-locked character reference sheet from <Subject 1>, using one continuous camera-orbit generation and the documented extraction plan.' +
            '\n\nretention_analysis:\n<Subject 1> (appears throughout [Shot 1]): fully_preserved - ' + identityLock +
            '\n\ndetailed_description:\nThe target uses the source-faithful ' + template + ' orbit preset from ' + CHARACTER_SOURCE_REPO + ' revision ' + CHARACTER_SOURCE_REVISION + '.\n[Shot 1] ' + bPrompt +
            '\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A';
        return {
            schema: 'serenity.h3.character_sheet.v1', source_shot_id: shot.id, name: shot.title,
            panel_count: count, style_mode: styleMode, director_notes: notes || '',
            source_repo: CHARACTER_SOURCE_REPO, source_url: CHARACTER_SOURCE_URL,
            source_revision: CHARACTER_SOURCE_REVISION, source_template: template,
            internal_frames: frames, frame_indices: indices,
            sheet_layout: count === 4 ? 'experimental upstream extraction labels: front@2, left@24, right@45, face-front@68' : 'front@2, left@21, back@42, right@63, face-front@84, face-three-quarter@113',
            plan_status: count === 4 ? 'experimental_metadata_only_not_render_admitted' : 'canonical_source_faithful_render_admitted',
            render_admitted: count !== 4, planned_seconds: frames / 24,
            reference_paths: refs.map(function (item) { return item.path; }),
            a_prompt: aPrompt, b_prompt: bPrompt, identity_lock: identityLock, h3_prompt: h3Prompt,
            warning: count === 4 ? 'Upstream mismatch: the README and extraction plan describe 73 frames, while the workflow JSON still requests 124; its orbit wording and extraction labels also conflict. At 24 fps, 73 frames is about 3.04 seconds, below Serenity’s 5-second product minimum. This preset is planning metadata only and is not render-admitted.' : '',
            caption_result_json: ''
        };
    }

    function upsertCharacterSheet(project, sheet) {
        var index = project.character_sheets.findIndex(function (item) { return item.source_shot_id === sheet.source_shot_id; });
        if (index >= 0) project.character_sheets[index] = copy(sheet);
        else project.character_sheets.push(copy(sheet));
    }

    function firstText() {
        for (var i = 0; i < arguments.length; i++) {
            var value = arguments[i];
            if (typeof value === 'string' && value.trim()) return value.trim();
        }
        return '';
    }

    function clampSeconds(value) {
        var seconds = Number(value) || 8;
        return Math.max(5, Math.min(15, Math.round(seconds)));
    }

    function directorShotFromItem(project, model, item, index) {
        var shot = createShot(project.next_shot_id++, firstText(item.title) || ('Shot ' + (index + 1)));
        shot.width = model.width; shot.height = model.height; shot.steps = model.steps; shot.seed = model.seed;
        shot.quant = model.quant; shot.attention_backend = model.attention_backend; shot.step_cache = model.step_cache;
        shot.duration_seconds = clampSeconds(item.duration_seconds);
        shot.brief = firstText(item.brief, item.beat, item.summary);
        var plan = firstText(item.shot_plan, item.plan, item.shot_description);
        var continuity = [];
        if (firstText(item.continuity_in)) continuity.push('Continuity in: ' + firstText(item.continuity_in));
        if (firstText(item.continuity_out)) continuity.push('Continuity out: ' + firstText(item.continuity_out));
        shot.shot_description = [plan].concat(continuity).filter(Boolean).join('\n');
        shot.prompt_override = firstText(item.h3_prompt, item.prompt);
        if (firstText(item.soundscape, item.overall_soundscape)) shot.soundscape = firstText(item.soundscape, item.overall_soundscape);
        if (firstText(item.music, item.non_diegetic_music)) shot.music = firstText(item.music, item.non_diegetic_music);
        return shot;
    }

    // Apply a serenity.h3.director.result.v2 object (or a single-shot doctor
    // result) to the project. Returns { project, message, focusShotId }.
    function applyDirectorResult(project, selectedShotId, actionId, result) {
        var r = result && typeof result === 'object' ? result : null;
        if (!r) throw new Error('Director result is not a JSON object; nothing to apply');
        var index = project.shots.findIndex(function (shot) { return shot.id === selectedShotId; });
        if (index < 0) index = 0;
        var model = project.shots[index];
        var messages = [];
        var focusShotId = model.id;
        var bible = firstText(r.continuity_bible, r.identity_lock);
        if (bible && (!String(project.continuity_bible || '').trim() || ['dream_project', 'script_to_coverage', 'continue_story', 'sound_to_scene'].indexOf(actionId) >= 0)) {
            project.continuity_bible = bible; messages.push('continuity bible');
        }
        var items = Array.isArray(r.shots) ? r.shots.filter(function (item) { return item && typeof item === 'object'; }) : [];
        if (items.length) {
            var made = items.map(function (item, i) { return directorShotFromItem(project, model, item, i); });
            var untouched = project.shots.length === 1 && !model.take_job_ids.length
                && !String(model.brief || model.shot_description || model.prompt_override || '').trim();
            if (untouched) project.shots = made;
            else Array.prototype.splice.apply(project.shots, [index + 1, 0].concat(made));
            if (actionId === 'continue_story' && model.take_job_ids.length) {
                var takeIndex = model.selected_take >= 0 ? model.selected_take : model.take_job_ids.length - 1;
                made[0].continue_from = model.take_job_ids[takeIndex] || ''; made[0].motion_context_frames = 22;
            }
            focusShotId = made[0].id;
            messages.push(made.length + ' shot' + (made.length === 1 ? '' : 's') + (untouched ? ' (replaced the empty opening shot)' : ' inserted after ' + model.title));
        } else {
            var prompt = firstText(r.revised_prompt, r.h3_prompt, r.prompt);
            if (prompt) { model.prompt_override = prompt; messages.push('H3 prompt on ' + model.title); }
            var brief = firstText(r.brief, r.literal_caption);
            if (brief && !String(model.brief || '').trim()) { model.brief = brief; messages.push('brief on ' + model.title); }
        }
        if (!messages.length) throw new Error('Director result has no shots or prompt to apply');
        return { project: project, message: 'Applied ' + messages.join(', '), focusShotId: focusShotId };
    }

    function captionRequest(project, shot, actionId, options) {
        var issue = validateDirector(project, shot, actionId);
        if (issue) throw new Error(issue);
        var action = actionById(actionId);
        var sheet = null;
        if (actionId === 'character_sheet') {
            var opts = options || {};
            sheet = characterSheetFromShot(shot, Number(opts.panel_count) === 4 ? 4 : 6, opts.style_mode === 'anime_to_real' ? 'anime_to_real' : 'standard_orbit', project.director_brief);
            upsertCharacterSheet(project, sheet);
        }
        return {
            schema: CAPTION_SCHEMA, profile: 'h3-director', operation: actionId,
            workflow_group: action.group,
            model: Number(project.caption_tier) === 1 ? QWEN_32B : QWEN_8B,
            model_tier: Number(project.caption_tier) === 1 ? '32b' : '8b',
            adult_content: project.adult_mode === true,
            content_scope: project.adult_mode ? 'lawful_consensual_adult_18_plus' : 'general_audience',
            system_prompt: systemPrompt(project, shot, actionId),
            response_contract: responseContract(actionId),
            project_kind: projectKindLabel(project.project_kind), project_title: project.title,
            target_duration_seconds: Number(project.target_duration_seconds),
            minimum_shots: directorMinimumShots(project), maximum_shots: directorMaximumShots(project),
            takes_per_shot: Number(project.takes_per_shot),
            generation_count_rule: 'generation_count must equal the sum of takes across all planned 5–15 second shots',
            director_brief: project.director_brief, duration_seconds: Number(shot.duration_seconds),
            brief: shot.brief, current_shot_plan: shot.shot_description,
            current_h3_prompt: sheet ? sheet.h3_prompt : compilePrompt(shot),
            continuity_bible: project.continuity_bible, brand_bible: project.brand_bible,
            first_frame: shot.first_frame, last_frame: shot.last_frame,
            source_take: shot.output_path, continue_from: shot.continue_from,
            inputs: shot.references.map(function (item, index) {
                return Object.assign({ order: index + 1, label: referenceLabel(shot.references, index), source_label: referenceSourceLabel(shot.references, index) }, copy(item));
            }),
            project_assets: copy(project.assets),
            character_sheet: sheet
        };
    }

    function deliveryManifest(project) {
        return {
            schema: 'serenity.h3.edit.v1', project_schema: PROJECT_SCHEMA,
            project_title: project.title, source_fps: NATIVE_FPS, delivery_fps: project.delivery_fps,
            shots: project.shots.filter(function (shot) { return shot.selected_take >= 0; }).map(function (shot, index) {
                return { order: index + 1, shot_id: shot.id, title: shot.title, locked: shot.locked, job_id: shot.take_job_ids[shot.selected_take] || '', path: shot.take_output_paths[shot.selected_take] || shot.output_path || '', duration_seconds: shot.duration_seconds };
            })
        };
    }

    return {
        PROJECT_SCHEMA: PROJECT_SCHEMA,
        CAPTION_SCHEMA: CAPTION_SCHEMA,
        NATIVE_FPS: NATIVE_FPS,
        MIN_SECONDS: MIN_SECONDS,
        MAX_SECONDS: MAX_SECONDS,
        MAX_ENDLESS_SECONDS: MAX_ENDLESS_SECONDS,
        ENDLESS_SCHEMA: ENDLESS_SCHEMA,
        MAX_PIXELS: MAX_PIXELS,
        RESOLUTIONS: RESOLUTIONS,
        ACTIONS: ACTIONS,
        CHARACTER_SOURCE_URL: CHARACTER_SOURCE_URL,
        CHARACTER_SOURCE_REVISION: CHARACTER_SOURCE_REVISION,
        createReference: createReference,
        createShot: createShot,
        createProject: createProject,
        createEmptyEndless: createEmptyEndless,
        createEndlessRun: createEndlessRun,
        normalizeProject: normalizeProject,
        copy: copy,
        secondsText: secondsText,
        projectKindLabel: projectKindLabel,
        outputFrames: outputFrames,
        internalFrames: internalFrames,
        detectMode: detectMode,
        modeLabel: modeLabel,
        referenceSourceLabel: referenceSourceLabel,
        referenceLabel: referenceLabel,
        compilePrompt: compilePrompt,
        promptComplianceIssue: promptComplianceIssue,
        validateShot: validateShot,
        renderRequest: renderRequest,
        validateEndlessSettings: validateEndlessSettings,
        planEndlessDurations: planEndlessDurations,
        planEndlessFrames: planEndlessFrames,
        endlessBaseSnapshot: endlessBaseSnapshot,
        endlessFingerprint: endlessFingerprint,
        canonicalEndlessSnapshot: canonicalEndlessSnapshot,
        endlessSnapshotsEqual: endlessSnapshotsEqual,
        videoJobUrls: videoJobUrls,
        endlessSegmentShot: endlessSegmentShot,
        recordEndlessSegmentTake: recordEndlessSegmentTake,
        directorMinimumShots: directorMinimumShots,
        directorMaximumShots: directorMaximumShots,
        actionById: actionById,
        validateDirector: validateDirector,
        characterSheetFromShot: characterSheetFromShot,
        upsertCharacterSheet: upsertCharacterSheet,
        captionRequest: captionRequest,
        applyDirectorResult: applyDirectorResult,
        deliveryManifest: deliveryManifest
    };
})();
