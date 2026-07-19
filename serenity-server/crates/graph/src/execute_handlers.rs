// Larger per-type handlers for execute_typed_graph, `include!`-d into execute.rs.
// Each is a faithful port of the matching Mojo `elif` block in
// `apply_typed_workflow_graph` (1904-2851). Latent geometry that the Mojo
// executor kept in parallel `latent_*` lists travels here in the LATENT
// payload's {width,height,batch,init_image,mask_image} fields.

/// Geometry view of a LATENT payload (Mojo latent_* row at a given link).
struct LatentGeom {
    width: i64,
    height: i64,
    batch: i64,
    init_image: String,
    mask_image: String,
}

fn latent_geom_of(store: &ValueStore, link: &WorkflowLink) -> Option<LatentGeom> {
    match latent_payload(store, link) {
        Some(ValuePayload::Latent { width, height, batch, init_image, mask_image }) => {
            Some(LatentGeom {
                width: *width,
                height: *height,
                batch: *batch,
                init_image: init_image.clone().unwrap_or_default(),
                mask_image: mask_image.clone().unwrap_or_default(),
            })
        }
        _ => None,
    }
}

fn latent_payload_from(g: &LatentGeom) -> ValuePayload {
    ValuePayload::Latent {
        width: g.width,
        height: g.height,
        batch: g.batch,
        init_image: opt_nonempty(&g.init_image),
        mask_image: opt_nonempty(&g.mask_image),
    }
}

fn opt_nonempty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn exec_ltxv_sampler(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let mut model_link = links.input(id, "ltxv_model");
    if !model_link.found {
        model_link = links.input(id, "model");
    }
    let audio_link = links.input(id, "audio");
    if !model_link.found {
        return Err(GraphError::unsupported(
            "workflow graph LTXVSampler missing ltxv_model input",
        )
        .with_node(id));
    }
    if !(ready(store, &model_link) && optional_ready(store, &audio_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "ltxv_model")?;
    if audio_link.found {
        require_value_type(store, &audio_link, "AUDIO", "audio")?;
    }
    let model_name = model_name_of(store, &model_link)?;
    if !model_name.is_empty() {
        set_if_missing(out, "model", json!(model_name));
    }
    let prompt = wf_string(fields, "prompt");
    if !prompt.is_empty() {
        set_if_missing(out, "prompt", json!(prompt));
    }
    let negative = wf_string(fields, "negative_prompt");
    if !negative.is_empty() {
        set_if_missing(out, "negative", json!(negative));
    }
    let width = opt_int(fields, "width", 768, 16, 4096)?;
    let height = opt_int(fields, "height", 512, 16, 4096)?;
    let frames = opt_int(fields, "num_frames", 97, 1, 4096)?;
    let steps = opt_int(fields, "steps", 25, 1, 1000)?;
    let seed = opt_int(fields, "seed", 0, 0, 4294967295)?;
    let fps = opt_int(fields, "frame_rate", 24, 1, 240)?;
    let cfg = wf_float(fields, "cfg", 3.0, 0.0, 100.0)?;
    set_if_missing(out, "width", json!(width));
    set_if_missing(out, "height", json!(height));
    set_if_missing(out, "num_frames", json!(frames));
    set_if_missing(out, "frame_count", json!(frames));
    set_if_missing(out, "steps", json!(steps));
    set_if_missing(out, "seed", json!(seed));
    set_if_missing(out, "cfg", json!(cfg));
    set_if_missing(out, "fps", json!(fps));
    set_if_missing(out, "frame_rate", json!(fps));
    copy_field_if_missing(out, fields, "mode", "video_mode");
    copy_field_if_missing(out, fields, "stg_scale", "stg_scale");
    copy_field_if_missing(out, fields, "audio_start_time", "audio_start_time");
    copy_field_if_missing(out, fields, "audio_duration", "audio_duration");
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width,
            height,
            batch: 1,
            init_image: None,
            mask_image: None,
        },
    )?;
    add_value(store, id, "VIDEO", ValuePayload::Video { path: String::new() })?;
    add_value(store, id, "AUDIO", ValuePayload::Audio { path: String::new() })?;
    Ok(Fire::Done)
}

fn exec_wan_image_to_video(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let positive_link = links.input(id, "positive");
    let vae_link = links.input(id, "vae");
    let image_link = links.input(id, "image");
    let clip_vision_link = links.input(id, "clip_vision_output");
    if !positive_link.found || !vae_link.found || !image_link.found {
        return Err(GraphError::unsupported(
            "workflow graph WanImageToVideo missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &positive_link)
        && ready(store, &vae_link)
        && ready(store, &image_link)
        && optional_ready(store, &clip_vision_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &positive_link, "CONDITIONING", "positive")?;
    require_value_type(store, &vae_link, "VAE", "vae")?;
    require_value_type(store, &image_link, "IMAGE", "image")?;
    if clip_vision_link.found {
        require_value_type(
            store,
            &clip_vision_link,
            "CLIP_VISION_OUTPUT",
            "clip_vision_output",
        )?;
    }
    let width = opt_int(fields, "width", 832, 16, 4096)?;
    let height = opt_int(fields, "height", 480, 16, 4096)?;
    let frames = opt_int(fields, "length", 81, 1, 4096)?;
    let image_path = image_path_of(store, &image_link)?;
    set_if_missing(out, "width", json!(width));
    set_if_missing(out, "height", json!(height));
    set_if_missing(out, "num_frames", json!(frames));
    set_if_missing(out, "frame_count", json!(frames));
    set_if_missing(out, "video_conditioning_image", json!(image_path.clone()));
    add_value(
        store,
        id,
        "CONDITIONING",
        ValuePayload::Cond {
            text: cond_text_of(store, &positive_link)?,
        },
    )?;
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width,
            height,
            batch: 1,
            init_image: opt_nonempty(&image_path),
            mask_image: None,
        },
    )?;
    Ok(Fire::Done)
}

// --- EmptyLatentImage / EmptySD3 / EmptyFlux2 / video latent (Mojo 1904) -------

fn exec_empty_latent(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let width_link = links.input(id, "width");
    let height_link = links.input(id, "height");
    let batch_link = links.input(id, "batch_size");
    let length_link = links.input(id, "length");
    if !(optional_ready(store, &width_link)
        && optional_ready(store, &height_link)
        && optional_ready(store, &batch_link)
        && optional_ready(store, &length_link))
    {
        return Ok(Fire::NotReady);
    }
    let mut width = opt_int(fields, "width", 512, 16, 2048)?;
    let mut height = opt_int(fields, "height", 512, 16, 2048)?;
    let mut images = opt_int(fields, "batch_size", 1, 1, 64)?;
    if width_link.found {
        width = scalar_int_of(store, &width_link, "width")?;
    }
    if height_link.found {
        height = scalar_int_of(store, &height_link, "height")?;
    }
    if batch_link.found {
        images = scalar_int_of(store, &batch_link, "batch_size")?;
    }
    if node.type_id == "EmptyHunyuanLatentVideo" {
        let mut frames = opt_int(fields, "length", 1, 1, 4096)?;
        if length_link.found {
            frames = scalar_int_of(store, &length_link, "length")?;
        }
        if !(1..=4096).contains(&frames) {
            return Err(GraphError::unsupported(
                "workflow graph EmptyHunyuanLatentVideo length out of range",
            )
            .with_node(id));
        }
        set_if_missing(out, "num_frames", json!(frames));
        set_if_missing(out, "frame_count", json!(frames));
    }
    if !(16..=2048).contains(&width) || !(16..=2048).contains(&height) {
        return Err(GraphError::unsupported(
            "workflow graph EmptyLatentImage scalar dimensions out of range",
        )
        .with_node(id));
    }
    if !(1..=64).contains(&images) {
        return Err(GraphError::unsupported(
            "workflow graph EmptyLatentImage scalar batch_size out of range",
        )
        .with_node(id));
    }
    set_if_missing(out, "width", json!(width));
    set_if_missing(out, "height", json!(height));
    set_if_missing(out, "images", json!(images));
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width,
            height,
            batch: images,
            init_image: None,
            mask_image: None,
        },
    )?;
    Ok(Fire::Done)
}

// --- ConditioningSetMask (Mojo 2001) -------------------------------------------

fn exec_conditioning_set_mask(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let cond_link = links.input(id, "conditioning");
    let mask_link = links.input(id, "mask");
    if !cond_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ConditioningSetMask missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &cond_link) && ready(store, &mask_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &cond_link, "CONDITIONING", "conditioning")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let text = cond_text_of(store, &cond_link)?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    let strength = wf_float(fields, "strength", 1.0, 0.0, 10.0)?;
    let set_cond_area = wf_string(fields, "set_cond_area").to_lowercase();
    let set_area_to_bounds = !set_cond_area.is_empty() && set_cond_area != "default";
    set_if_missing(out, "conditioning_mask_image", json!(mask_path));
    set_if_missing(out, "conditioning_mask_channel", json!(mask_source));
    set_if_missing(out, "conditioning_mask_strength", json!(strength));
    set_if_missing(out, "conditioning_mask_set_area_to_bounds", json!(set_area_to_bounds));
    add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
    Ok(Fire::Done)
}

// --- KSampler family (Mojo 2638) -----------------------------------------------

fn exec_ksampler(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    saw_prompt: &mut bool,
) -> GraphResult<Fire> {
    let id = node.id;
    let t = node.type_id.as_str();
    let fields = &node.fields;
    let advanced = t == "KSamplerAdvanced";
    let model_link = links.input(id, "model");
    let pos_link = links.input(id, "positive");
    let neg_link = links.input(id, "negative");
    let latent_link = links.input(id, "latent_image");
    // KSamplerAdvanced seeds from `noise_seed`; KSampler from `seed`. Resolve the
    // seed input link from whichever field the node carries.
    let seed_link = {
        let l = links.input(id, "noise_seed");
        if advanced && l.found { l } else { links.input(id, "seed") }
    };
    let steps_link = links.input(id, "steps");
    let cfg_link = links.input(id, "cfg");
    let sampler_name_link = links.input(id, "sampler_name");
    let scheduler_link = links.input(id, "scheduler");
    let denoise_link = links.input(id, "denoise");
    if !model_link.found || !pos_link.found || !neg_link.found || !latent_link.found {
        return Err(GraphError::unsupported(format!(
            "workflow graph {t} missing required typed input"
        ))
        .with_node(id));
    }
    if !(ready(store, &model_link)
        && ready(store, &pos_link)
        && ready(store, &neg_link)
        && ready(store, &latent_link)
        && optional_ready(store, &seed_link)
        && optional_ready(store, &steps_link)
        && optional_ready(store, &cfg_link)
        && optional_ready(store, &sampler_name_link)
        && optional_ready(store, &scheduler_link)
        && optional_ready(store, &denoise_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "model")?;
    require_value_type(store, &pos_link, "CONDITIONING", "positive")?;
    require_value_type(store, &neg_link, "CONDITIONING", "negative")?;
    require_value_type(store, &latent_link, "LATENT", "latent_image")?;
    let model_name = model_name_of(store, &model_link)?;
    if !model_name.is_empty() {
        set_if_missing(out, "model", json!(model_name));
    }
    let prompt = cond_text_of(store, &pos_link)?;
    set_if_missing(out, "prompt", json!(prompt));
    *saw_prompt = true;
    let negative = cond_text_of(store, &neg_link)?;
    set_if_missing(out, "negative", json!(negative));

    let geom = latent_geom_of(store, &latent_link);
    if let Some(g) = &geom {
        if g.width > 0 {
            set_if_missing(out, "width", json!(g.width));
        }
        if g.height > 0 {
            set_if_missing(out, "height", json!(g.height));
        }
        set_if_missing(out, "images", json!(g.batch));
        if !g.init_image.is_empty() {
            set_if_missing(out, "init_image", json!(g.init_image));
        }
        if !g.mask_image.is_empty() {
            set_if_missing(out, "mask_image", json!(g.mask_image));
        }
    }

    // Track the resolved step count so KSamplerAdvanced can derive `creativity`
    // from its `start_at_step`/`steps` denoise window.
    let mut resolved_steps: Option<i64> = None;
    if steps_link.found {
        let steps = scalar_int_of(store, &steps_link, "steps")?;
        if !(1..=4096).contains(&steps) {
            return Err(GraphError::unsupported(
                "workflow graph KSampler scalar steps out of range",
            )
            .with_node(id));
        }
        set_if_missing(out, "steps", json!(steps));
        resolved_steps = Some(steps);
    } else {
        copy_field_if_missing(out, fields, "steps", "steps");
        if let Some(n) = fields.as_object().and_then(|o| o.get("steps")).and_then(JsonValue::as_i64) {
            resolved_steps = Some(n);
        }
    }

    if seed_link.found {
        let seed_name = if advanced { "noise_seed" } else { "seed" };
        let seed = scalar_int_of(store, &seed_link, seed_name)?;
        if seed >= 0 {
            set_if_missing(out, "seed", json!(seed));
        }
    } else if advanced {
        // KSamplerAdvanced names the seed `noise_seed`; fall back to `seed`.
        if fields.as_object().and_then(|o| o.get("noise_seed")).map(|v| !v.is_null()).unwrap_or(false) {
            set_field_if_nonneg_int(out, fields, "noise_seed", "seed")?;
        } else {
            set_field_if_nonneg_int(out, fields, "seed", "seed")?;
        }
    } else {
        set_field_if_nonneg_int(out, fields, "seed", "seed")?;
    }

    if cfg_link.found {
        let cfg = scalar_float_of(store, &cfg_link, "cfg")?;
        set_if_missing(out, "cfg", json!(cfg));
    } else {
        copy_field_if_missing(out, fields, "cfg", "cfg");
    }

    if sampler_name_link.found {
        let s = scalar_string_of(store, &sampler_name_link, "sampler_name")?;
        set_if_missing(out, "sampler", json!(s));
    } else {
        copy_field_if_missing(out, fields, "sampler_name", "sampler");
    }

    if scheduler_link.found {
        let s = scalar_string_of(store, &scheduler_link, "scheduler")?;
        set_if_missing(out, "scheduler", json!(s));
    } else {
        copy_field_if_missing(out, fields, "scheduler", "scheduler");
    }

    if advanced {
        // KSamplerAdvanced has no `denoise` field. Its denoise window is the
        // [start_at_step, end_at_step] slice of the `steps` schedule, with
        // `add_noise` toggling whether the latent is re-noised at the start.
        // zimage's flat model only represents the START of the window
        // (creativity = denoise start). An early `end_at_step`
        // (return_with_leftover_noise) or `add_noise="disable"` (continue an
        // existing latent without re-noising) CANNOT be faithfully lowered — so
        // fail loud rather than silently emit a full-txt2img creativity and
        // render the wrong image.
        let steps = resolved_steps.filter(|&s| s > 0).unwrap_or(0);
        let add_noise = wf_string(fields, "add_noise").to_lowercase();
        if add_noise == "disable" {
            return Err(GraphError::unsupported(
                "workflow graph KSamplerAdvanced add_noise=disable (continue without \
                 re-noising) is not representable in the zimage flat denoise model",
            )
            .with_node(id));
        }
        // Default end_at_step to a large sentinel (Comfy's "run to the end"); only
        // an explicit early end is rejected.
        let end_at_step = opt_int(fields, "end_at_step", 1_000_000, 0, 10_000_000)?;
        if steps > 0 && end_at_step < steps {
            return Err(GraphError::unsupported(
                "workflow graph KSamplerAdvanced early end_at_step \
                 (return_with_leftover_noise) is not representable in the zimage \
                 flat denoise model",
            )
            .with_node(id));
        }
        let start_at_step = opt_int(fields, "start_at_step", 0, 0, 4096)?;
        let creativity = if steps > 0 {
            let raw = 1.0 - (start_at_step as f64) / (steps as f64);
            raw.clamp(0.0, 1.0)
        } else {
            // No usable step count: fall back to full denoise.
            1.0
        };
        set_if_missing(out, "creativity", json!(creativity));
    } else if denoise_link.found {
        let d = scalar_float_of(store, &denoise_link, "denoise")?;
        if !(0.0..=1.0).contains(&d) {
            return Err(GraphError::unsupported(
                "workflow graph KSampler scalar denoise out of range",
            )
            .with_node(id));
        }
        set_if_missing(out, "creativity", json!(d));
    } else {
        copy_field_if_missing(out, fields, "denoise", "creativity");
    }

    if t == "LanPaint_KSampler" || t == "LanPaint_KSamplerAdvanced" {
        copy_lanpaint_sampler_fields(out, fields);
    }

    // Output LATENT carries the input geometry, or a 512x512 default.
    let payload = match &geom {
        Some(g) => latent_payload_from(g),
        None => ValuePayload::Latent {
            width: 512,
            height: 512,
            batch: 1,
            init_image: None,
            mask_image: None,
        },
    };
    add_value(store, id, "LATENT", payload)?;
    Ok(Fire::Done)
}

// --- CFGGuider (Mojo 2490) -----------------------------------------------------

fn exec_cfg_guider(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    saw_prompt: &mut bool,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let model_link = links.input(id, "model");
    let pos_link = links.input(id, "positive");
    let neg_link = links.input(id, "negative");
    let cfg_link = links.input(id, "cfg");
    if !model_link.found || !pos_link.found || !neg_link.found {
        return Err(GraphError::unsupported(
            "workflow graph CFGGuider missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &model_link)
        && ready(store, &pos_link)
        && ready(store, &neg_link)
        && optional_ready(store, &cfg_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "model")?;
    require_value_type(store, &pos_link, "CONDITIONING", "positive")?;
    require_value_type(store, &neg_link, "CONDITIONING", "negative")?;
    let model_name = model_name_of(store, &model_link)?;
    if !model_name.is_empty() {
        set_if_missing(out, "model", json!(model_name));
    }
    let prompt = cond_text_of(store, &pos_link)?;
    set_if_missing(out, "prompt", json!(prompt));
    *saw_prompt = true;
    let negative = cond_text_of(store, &neg_link)?;
    set_if_missing(out, "negative", json!(negative));
    if cfg_link.found {
        let cfg = scalar_float_of(store, &cfg_link, "cfg")?;
        set_if_missing(out, "cfg", json!(cfg));
    } else {
        copy_field_if_missing(out, fields, "cfg", "cfg");
    }
    add_value(store, id, "GUIDER", ValuePayload::Guider { cfg: None })?;
    Ok(Fire::Done)
}

// --- BasicGuider (Mojo 2529) ---------------------------------------------------

fn exec_basic_guider(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    saw_prompt: &mut bool,
) -> GraphResult<Fire> {
    let id = node.id;
    let model_link = links.input(id, "model");
    let cond_link = links.input(id, "conditioning");
    if !model_link.found || !cond_link.found {
        return Err(GraphError::unsupported(
            "workflow graph BasicGuider missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &model_link) && ready(store, &cond_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "model")?;
    require_value_type(store, &cond_link, "CONDITIONING", "conditioning")?;
    let model_name = model_name_of(store, &model_link)?;
    if !model_name.is_empty() {
        set_if_missing(out, "model", json!(model_name));
    }
    let prompt = cond_text_of(store, &cond_link)?;
    set_if_missing(out, "prompt", json!(prompt));
    *saw_prompt = true;
    add_value(store, id, "GUIDER", ValuePayload::Guider { cfg: None })?;
    Ok(Fire::Done)
}

// --- Flux2Scheduler (Mojo 2549) ------------------------------------------------

fn exec_flux2_scheduler(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    _out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let steps_link = links.input(id, "steps");
    if !optional_ready(store, &steps_link) {
        return Ok(Fire::NotReady);
    }
    let mut steps = opt_int(fields, "steps", 20, 1, 4096)?;
    if steps_link.found {
        steps = scalar_int_of(store, &steps_link, "steps")?;
    }
    if !(1..=4096).contains(&steps) {
        return Err(GraphError::unsupported(
            "workflow graph Flux2Scheduler scalar steps out of range",
        )
        .with_node(id));
    }
    add_value(
        store,
        id,
        "SIGMAS",
        ValuePayload::Sigmas { steps, scheduler: "flux2".to_string(), denoise: 1.0 },
    )?;
    Ok(Fire::Done)
}

// --- BasicScheduler (Mojo 2567) ------------------------------------------------

fn exec_basic_scheduler(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    _out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let model_link = links.input(id, "model");
    if !model_link.found {
        return Err(GraphError::unsupported(
            "workflow graph BasicScheduler missing model input",
        )
        .with_node(id));
    }
    let scheduler_link = links.input(id, "scheduler");
    let steps_link = links.input(id, "steps");
    let denoise_link = links.input(id, "denoise");
    if !(ready(store, &model_link)
        && optional_ready(store, &scheduler_link)
        && optional_ready(store, &steps_link)
        && optional_ready(store, &denoise_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "model")?;
    let mut scheduler = wf_string(fields, "scheduler");
    if scheduler.is_empty() {
        scheduler = "simple".to_string();
    }
    let mut steps = opt_int(fields, "steps", 20, 1, 4096)?;
    let mut denoise = wf_float(fields, "denoise", 1.0, 0.0, 1.0)?;
    if scheduler_link.found {
        scheduler = scalar_string_of(store, &scheduler_link, "scheduler")?;
    }
    if steps_link.found {
        steps = scalar_int_of(store, &steps_link, "steps")?;
    }
    if denoise_link.found {
        denoise = scalar_float_of(store, &denoise_link, "denoise")?;
    }
    if !(1..=4096).contains(&steps) {
        return Err(GraphError::unsupported(
            "workflow graph BasicScheduler scalar steps out of range",
        )
        .with_node(id));
    }
    if !(0.0..=1.0).contains(&denoise) {
        return Err(GraphError::unsupported(
            "workflow graph BasicScheduler scalar denoise out of range",
        )
        .with_node(id));
    }
    add_value(
        store,
        id,
        "SIGMAS",
        ValuePayload::Sigmas { steps, scheduler, denoise },
    )?;
    Ok(Fire::Done)
}

// --- SamplerCustomAdvanced family (Mojo 2767) ----------------------------------

fn exec_sampler_custom_advanced(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let t = node.type_id.as_str();
    let fields = &node.fields;
    let noise_link = links.input(id, "noise");
    let guider_link = links.input(id, "guider");
    let sampler_link = links.input(id, "sampler");
    let sigmas_link = links.input(id, "sigmas");
    let latent_link = links.input(id, "latent_image");
    if !noise_link.found
        || !guider_link.found
        || !sampler_link.found
        || !sigmas_link.found
        || !latent_link.found
    {
        return Err(GraphError::unsupported(format!(
            "workflow graph {t} missing required typed input"
        ))
        .with_node(id));
    }
    if !(ready(store, &noise_link)
        && ready(store, &guider_link)
        && ready(store, &sampler_link)
        && ready(store, &sigmas_link)
        && ready(store, &latent_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &noise_link, "NOISE", "noise")?;
    require_value_type(store, &guider_link, "GUIDER", "guider")?;
    require_value_type(store, &sampler_link, "SAMPLER", "sampler")?;
    require_value_type(store, &sigmas_link, "SIGMAS", "sigmas")?;
    require_value_type(store, &latent_link, "LATENT", "latent_image")?;

    if let Some(ValuePayload::Noise { seed }) = store.get(noise_link.node_id, &noise_link.port).map(|v| &v.payload) {
        set_if_missing(out, "seed", json!(*seed));
    }
    if let Some(ValuePayload::Sampler { name }) = store.get(sampler_link.node_id, &sampler_link.port).map(|v| &v.payload) {
        set_if_missing(out, "sampler", json!(name));
    }
    if let Some(ValuePayload::Sigmas { steps, scheduler, denoise }) = store.get(sigmas_link.node_id, &sigmas_link.port).map(|v| &v.payload) {
        set_if_missing(out, "steps", json!(*steps));
        set_if_missing(out, "scheduler", json!(scheduler));
        set_if_missing(out, "creativity", json!(*denoise));
    }
    if t == "LanPaint_SamplerCustomAdvanced" {
        copy_lanpaint_sampler_fields(out, fields);
    }
    let geom = latent_geom_of(store, &latent_link);
    if let Some(g) = &geom {
        if g.width > 0 {
            set_if_missing(out, "width", json!(g.width));
        }
        if g.height > 0 {
            set_if_missing(out, "height", json!(g.height));
        }
        set_if_missing(out, "images", json!(g.batch));
        if !g.init_image.is_empty() {
            set_if_missing(out, "init_image", json!(g.init_image));
        }
        if !g.mask_image.is_empty() {
            set_if_missing(out, "mask_image", json!(g.mask_image));
        }
    }
    let payload = match &geom {
        Some(g) => latent_payload_from(g),
        None => ValuePayload::Latent { width: 0, height: 0, batch: 1, init_image: None, mask_image: None },
    };
    add_value(store, id, "LATENT", payload)?;
    Ok(Fire::Done)
}

// --- Named SIGMAS scheduler nodes (Karras/Exponential/Polyexp/SDTurbo) ----------

/// Named SIGMAS producer: the scheduler name is the node TYPE. Like
/// BasicScheduler, lowers to scheduler= + steps (+ denoise for SDTurboScheduler).
/// Gated on the worker's supported list — an unsupported name fails loud.
fn exec_named_scheduler(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let t = node.type_id.as_str();
    let fields = &node.fields;
    let scheduler = crate::named_scheduler_name(t).to_string();
    if !crate::worker_supports_scheduler(&scheduler) {
        return Err(GraphError::unsupported(format!(
            "workflow graph {t} lowers to unsupported scheduler '{scheduler}'; \
             the worker supports only simple/flowmatch/flow_match/sgm_uniform"
        ))
        .with_node(id));
    }
    let steps_link = links.input(id, "steps");
    if !optional_ready(store, &steps_link) {
        return Ok(Fire::NotReady);
    }
    let mut steps = opt_int(fields, "steps", 20, 1, 4096)?;
    if steps_link.found {
        steps = scalar_int_of(store, &steps_link, "steps")?;
    }
    if !(1..=4096).contains(&steps) {
        return Err(GraphError::unsupported(format!(
            "workflow graph {t} scalar steps out of range"
        ))
        .with_node(id));
    }
    let denoise = if t == "SDTurboScheduler" {
        wf_float(fields, "denoise", 1.0, 0.0, 1.0)?
    } else {
        1.0
    };
    add_value(
        store,
        id,
        "SIGMAS",
        ValuePayload::Sigmas { steps, scheduler, denoise },
    )?;
    Ok(Fire::Done)
}

// --- SamplerCustom (single-output sibling of SamplerCustomAdvanced) -------------

/// Single-output sibling of SamplerCustomAdvanced. Unlike the Advanced node it
/// has NO noise/guider input ports — it builds CFG guidance and noise itself
/// from its model/positive/negative inputs and add_noise/noise_seed/cfg widgets,
/// then samples with the SAMPLER + SIGMAS inputs. Reuses the SAME flat
/// sampler/scheduler/steps/seed/cfg path.
fn exec_sampler_custom(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    saw_prompt: &mut bool,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let model_link = links.input(id, "model");
    let pos_link = links.input(id, "positive");
    let neg_link = links.input(id, "negative");
    let sampler_link = links.input(id, "sampler");
    let sigmas_link = links.input(id, "sigmas");
    let latent_link = links.input(id, "latent_image");
    if !model_link.found
        || !pos_link.found
        || !neg_link.found
        || !sampler_link.found
        || !sigmas_link.found
        || !latent_link.found
    {
        return Err(GraphError::unsupported(
            "workflow graph SamplerCustom missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &model_link)
        && ready(store, &pos_link)
        && ready(store, &neg_link)
        && ready(store, &sampler_link)
        && ready(store, &sigmas_link)
        && ready(store, &latent_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &model_link, "MODEL", "model")?;
    require_value_type(store, &pos_link, "CONDITIONING", "positive")?;
    require_value_type(store, &neg_link, "CONDITIONING", "negative")?;
    require_value_type(store, &sampler_link, "SAMPLER", "sampler")?;
    require_value_type(store, &sigmas_link, "SIGMAS", "sigmas")?;
    require_value_type(store, &latent_link, "LATENT", "latent_image")?;

    let model_name = model_name_of(store, &model_link)?;
    if !model_name.is_empty() {
        set_if_missing(out, "model", json!(model_name));
    }
    let prompt = cond_text_of(store, &pos_link)?;
    set_if_missing(out, "prompt", json!(prompt));
    *saw_prompt = true;
    let negative = cond_text_of(store, &neg_link)?;
    set_if_missing(out, "negative", json!(negative));

    // cfg/seed come from the node's own widgets (SamplerCustom has no
    // CFGGuider/RandomNoise inputs).
    copy_field_if_missing(out, fields, "cfg", "cfg");
    set_field_if_nonneg_int(out, fields, "noise_seed", "seed")?;

    if let Some(ValuePayload::Sampler { name }) =
        store.get(sampler_link.node_id, &sampler_link.port).map(|v| &v.payload)
    {
        set_if_missing(out, "sampler", json!(name));
    }
    if let Some(ValuePayload::Sigmas { steps, scheduler, denoise }) =
        store.get(sigmas_link.node_id, &sigmas_link.port).map(|v| &v.payload)
    {
        set_if_missing(out, "steps", json!(*steps));
        set_if_missing(out, "scheduler", json!(scheduler));
        set_if_missing(out, "creativity", json!(*denoise));
    }

    let geom = latent_geom_of(store, &latent_link);
    if let Some(g) = &geom {
        if g.width > 0 {
            set_if_missing(out, "width", json!(g.width));
        }
        if g.height > 0 {
            set_if_missing(out, "height", json!(g.height));
        }
        set_if_missing(out, "images", json!(g.batch));
        if !g.init_image.is_empty() {
            set_if_missing(out, "init_image", json!(g.init_image));
        }
        if !g.mask_image.is_empty() {
            set_if_missing(out, "mask_image", json!(g.mask_image));
        }
    }
    let payload = match &geom {
        Some(g) => latent_payload_from(g),
        None => ValuePayload::Latent { width: 0, height: 0, batch: 1, init_image: None, mask_image: None },
    };
    add_value(store, id, "LATENT", payload)?;
    Ok(Fire::Done)
}

// --- ImageToMask (Mojo 2167) ---------------------------------------------------

fn exec_image_to_mask(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let image_link = links.input(id, "image");
    if !image_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ImageToMask missing image input",
        )
        .with_node(id));
    }
    if !ready(store, &image_link) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image_link, "IMAGE", "image")?;
    let image_path = image_path_of(store, &image_link)?;
    let requested_channel = imagetomask_channel(fields)?;
    let mut mask_source = image_mask_source_of(store, &image_link)?;
    if mask_source.is_empty() {
        mask_source = requested_channel;
    }
    set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    add_value(
        store,
        id,
        "MASK",
        ValuePayload::Mask { path: image_path, source: Some(mask_source) },
    )?;
    Ok(Fire::Done)
}

// --- MaskToImage (Mojo 2184) ---------------------------------------------------

fn exec_mask_to_image(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let mask_link = links.input(id, "mask");
    if !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph MaskToImage missing mask input",
        )
        .with_node(id));
    }
    if !ready(store, &mask_link) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: mask_path, mask_source: opt_nonempty(&mask_source) },
    )?;
    Ok(Fire::Done)
}

// --- ThresholdMask (Mojo 2197) -------------------------------------------------

fn exec_threshold_mask(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let mask_link = links.input(id, "mask");
    if !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ThresholdMask missing mask input",
        )
        .with_node(id));
    }
    if !ready(store, &mask_link) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    let threshold = wf_float(fields, "value", 0.5, 0.0, 1.0)?;
    set_if_missing(out, "threshold_mask_value", json!(threshold));
    set_if_missing(out, "threshold_mask_operator", json!("gt"));
    add_value(
        store,
        id,
        "MASK",
        ValuePayload::Mask { path: mask_path, source: opt_nonempty(&mask_source) },
    )?;
    Ok(Fire::Done)
}

// --- ImageScale / ImageScaleToTotalPixels (Mojo 2224) --------------------------

fn exec_image_scale(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let t = node.type_id.as_str();
    let image_link = links.input(id, "image");
    if !image_link.found {
        return Err(GraphError::unsupported(format!(
            "workflow graph {t} missing image input"
        ))
        .with_node(id));
    }
    let width_link = links.input(id, "width");
    let height_link = links.input(id, "height");
    if !(ready(store, &image_link)
        && optional_ready(store, &width_link)
        && optional_ready(store, &height_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image_link, "IMAGE", "image")?;
    if width_link.found {
        let width = scalar_int_of(store, &width_link, "width")?;
        if !(1..=8192).contains(&width) {
            return Err(GraphError::unsupported(
                "workflow graph ImageScale scalar width out of range",
            )
            .with_node(id));
        }
    }
    if height_link.found {
        let height = scalar_int_of(store, &height_link, "height")?;
        if !(1..=8192).contains(&height) {
            return Err(GraphError::unsupported(
                "workflow graph ImageScale scalar height out of range",
            )
            .with_node(id));
        }
    }
    let image_path = image_path_of(store, &image_link)?;
    let mask_source = image_mask_source_of(store, &image_link)?;
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: image_path, mask_source: opt_nonempty(&mask_source) },
    )?;
    Ok(Fire::Done)
}

// --- ImageScaleBy (ComfyUI built-in) -------------------------------------------

/// ImageScaleBy multiplies the SOURCE image dims by the `scale_by` widget
/// (`out_w = round(src_w * scale_by)`). In the flat single-image model the source
/// image dims are NOT resolvable (LoadImage carries only a path, never dims), so
/// the scaled output dims cannot be represented on a worker-supported grid. Fail
/// loud [501] — never silently emit a wrong size. (This mirrors the Rust/Mojo
/// fail-loud choice for nodes whose output is not representable in the flat
/// single-pass model.) The `scale_by` widget is range-validated first so the
/// error is precise rather than a generic parse failure.
fn exec_image_scale_by(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let image_link = links.input(id, "image");
    if !image_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ImageScaleBy missing image input",
        )
        .with_node(id));
    }
    if !ready(store, &image_link) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image_link, "IMAGE", "image")?;
    // Validate scale_by (ComfyUI range [0.01, 8.0]) so the 501 is specific.
    let _scale_by = wf_float(fields, "scale_by", 1.0, 0.01, 8.0)?;
    Err(GraphError::unsupported(
        "workflow graph ImageScaleBy scales the source image dims by scale_by, but \
         the source image dimensions are not resolvable in the flat single-image \
         model (LoadImage carries no dims); use ImageResizeKJ with explicit \
         width/height, or an EmptyLatentImage, to set a worker-supported size",
    )
    .with_node(id))
}

// --- ImageResizeKJ (KJ) --------------------------------------------------------

/// ImageResizeKJ resizes an image to EXPLICIT `width`/`height` widgets. Unlike
/// ImageScaleBy, the target dims are knowable — but only when the resize is a
/// plain explicit resize: `keep_proportion=false`, both width AND height nonzero,
/// and no `get_image_size` IMAGE input (all three of those make the output dims
/// depend on the un-knowable source dims). In the representable case we resolve
/// the explicit width/height into the flat params (range-validated like the
/// ImageScale scalar path), pass the IMAGE handle through, and emit width/height
/// INT outputs (slots 1/2). Otherwise fail loud [501].
fn exec_image_resize_kj(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let image_link = links.input(id, "image");
    if !image_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ImageResizeKJ missing image input",
        )
        .with_node(id));
    }
    // The width/height widgets may also arrive over typed links.
    let width_link = links.input(id, "width");
    let height_link = links.input(id, "height");
    // A connected `get_image_size` IMAGE input makes the output dims copy that
    // image's (un-knowable) dims — not representable.
    let get_size_link = links.input(id, "get_image_size");
    if !(ready(store, &image_link)
        && optional_ready(store, &width_link)
        && optional_ready(store, &height_link)
        && optional_ready(store, &get_size_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image_link, "IMAGE", "image")?;
    if get_size_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ImageResizeKJ get_image_size input copies the source \
             image dims, which are not resolvable in the flat single-image model",
        )
        .with_node(id));
    }
    // keep_proportion derives one dim from the source aspect → not representable.
    if wf_bool(fields, "keep_proportion", false)? {
        return Err(GraphError::unsupported(
            "workflow graph ImageResizeKJ keep_proportion=true derives a dimension \
             from the source image aspect, which is not resolvable in the flat \
             single-image model; set explicit width and height instead",
        )
        .with_node(id));
    }
    // Resolve the explicit width/height (link overrides widget).
    let mut width = opt_int(fields, "width", 512, 0, 8192)?;
    let mut height = opt_int(fields, "height", 512, 0, 8192)?;
    if width_link.found {
        width = scalar_int_of(store, &width_link, "width")?;
    }
    if height_link.found {
        height = scalar_int_of(store, &height_link, "height")?;
    }
    // A zero dim means "keep the source dim" (ComfyUI semantics) → not knowable.
    if width == 0 || height == 0 {
        return Err(GraphError::unsupported(
            "workflow graph ImageResizeKJ width/height of 0 keeps the source \
             dimension, which is not resolvable in the flat single-image model; \
             set explicit nonzero width and height",
        )
        .with_node(id));
    }
    if !(1..=8192).contains(&width) || !(1..=8192).contains(&height) {
        return Err(GraphError::unsupported(
            "workflow graph ImageResizeKJ width/height out of range",
        )
        .with_node(id));
    }
    // Explicit resize: resolve into the flat params (first-writer-wins).
    set_if_missing(out, "width", json!(width));
    set_if_missing(out, "height", json!(height));
    let image_path = image_path_of(store, &image_link)?;
    let mask_source = image_mask_source_of(store, &image_link)?;
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: image_path, mask_source: opt_nonempty(&mask_source) },
    )?;
    // Resolved width/height INT outputs (slots 1/2 of the KJ node).
    add_value(store, id, "width", ValuePayload::ScalarInt(width))?;
    add_value(store, id, "height", ValuePayload::ScalarInt(height))?;
    Ok(Fire::Done)
}

// --- ImagePadForOutpaint (Mojo 2257) -------------------------------------------

fn exec_image_pad(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let image_link = links.input(id, "image");
    if !image_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ImagePadForOutpaint missing image input",
        )
        .with_node(id));
    }
    if !ready(store, &image_link) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image_link, "IMAGE", "image")?;
    let image_path = image_path_of(store, &image_link)?;
    let left = opt_int(fields, "left", 0, 0, 4096)?;
    let top = opt_int(fields, "top", 0, 0, 4096)?;
    let right = opt_int(fields, "right", 0, 0, 4096)?;
    let bottom = opt_int(fields, "bottom", 0, 0, 4096)?;
    let feathering = opt_int(fields, "feathering", 40, 0, 4096)?;
    set_if_missing(out, "outpaint_left", json!(left));
    set_if_missing(out, "outpaint_top", json!(top));
    set_if_missing(out, "outpaint_right", json!(right));
    set_if_missing(out, "outpaint_bottom", json!(bottom));
    set_if_missing(out, "outpaint_feathering", json!(feathering));
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: image_path.clone(), mask_source: None },
    )?;
    add_value(
        store,
        id,
        "MASK",
        ValuePayload::Mask { path: image_path, source: Some("image_pad_for_outpaint".to_string()) },
    )?;
    Ok(Fire::Done)
}

// --- VAEEncode (Mojo 2282) -----------------------------------------------------

fn exec_vae_encode(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let pixels_link = links.input(id, "pixels");
    let vae_link = links.input(id, "vae");
    if !pixels_link.found || !vae_link.found {
        return Err(GraphError::unsupported(
            "workflow graph VAEEncode missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &pixels_link) && ready(store, &vae_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &pixels_link, "IMAGE", "pixels")?;
    require_value_type(store, &vae_link, "VAE", "vae")?;
    let init_path = image_path_of(store, &pixels_link)?;
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width: 0,
            height: 0,
            batch: 1,
            init_image: opt_nonempty(&init_path),
            mask_image: None,
        },
    )?;
    Ok(Fire::Done)
}

// --- VAEEncodeForInpaint (ComfyUI built-in) ------------------------------------

/// VAEEncodeForInpaint encodes `pixels` to a LATENT and attaches the inpaint
/// `mask` (grown by `grow_mask_by` in ComfyUI). In the flat single-pass model
/// this is the same effect as InpaintModelConditioning's mask half — it aliases
/// to the inpaint_* params + mask_image. There is no conditioning here (the cond
/// stays whatever the prompt/CLIPTextEncode supplies), so only the image/mask
/// keys are set. `grow_mask_by` has no flat representation and is ignored (the
/// worker grows the inpaint mask internally); it is read only to validate range.
fn exec_vae_encode_for_inpaint(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let pixels_link = links.input(id, "pixels");
    let vae_link = links.input(id, "vae");
    let mask_link = links.input(id, "mask");
    if !pixels_link.found || !vae_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph VAEEncodeForInpaint missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &pixels_link) && ready(store, &vae_link) && ready(store, &mask_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &pixels_link, "IMAGE", "pixels")?;
    require_value_type(store, &vae_link, "VAE", "vae")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    // grow_mask_by: read only for range validation; no flat key carries it.
    let _grow = opt_int(fields, "grow_mask_by", 6, 0, 64)?;
    let init_path = image_path_of(store, &pixels_link)?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    set_if_missing(out, "init_image", json!(init_path));
    set_if_missing(out, "mask_image", json!(mask_path));
    set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width: 0,
            height: 0,
            batch: 1,
            init_image: opt_nonempty(&init_path),
            mask_image: opt_nonempty(&mask_path),
        },
    )?;
    Ok(Fire::Done)
}

// --- RepeatLatentBatch (ComfyUI built-in) --------------------------------------

/// RepeatLatentBatch mutates a Comfy latent tensor batch. The flat daemon
/// `images=N` key is serial fanout, not latent-batch execution, so this node
/// must fail loud until a real batched latent path exists.
fn exec_repeat_latent_batch(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    _out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let samples_link = links.input(id, "samples");
    let amount_link = links.input(id, "amount");
    if !samples_link.found {
        return Err(GraphError::unsupported(
            "workflow graph RepeatLatentBatch missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &samples_link) && optional_ready(store, &amount_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &samples_link, "LATENT", "samples")?;
    let mut amount = opt_int(fields, "amount", 1, 1, 64)?;
    if amount_link.found {
        amount = scalar_int_of(store, &amount_link, "amount")?;
    }
    if !(1..=64).contains(&amount) {
        return Err(GraphError::unsupported(
            "workflow graph RepeatLatentBatch scalar amount out of range",
        )
        .with_node(id));
    }
    Err(GraphError::unsupported(
        "workflow graph RepeatLatentBatch requires real Comfy latent-batch execution; use flat images=N for serial product fanout",
    )
    .with_node(id))
}

// --- SetLatentNoiseMask (Mojo 2301) --------------------------------------------

fn exec_set_latent_noise_mask(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let samples_link = links.input(id, "samples");
    let mask_link = links.input(id, "mask");
    if !samples_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph SetLatentNoiseMask missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &samples_link) && ready(store, &mask_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &samples_link, "LATENT", "samples")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let geom = latent_geom_of(store, &samples_link);
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    set_if_missing(out, "mask_image", json!(mask_path));
    set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    let payload = match geom {
        Some(g) => ValuePayload::Latent {
            width: g.width,
            height: g.height,
            batch: g.batch,
            init_image: opt_nonempty(&g.init_image),
            mask_image: opt_nonempty(&mask_path),
        },
        None => ValuePayload::Latent {
            width: 0,
            height: 0,
            batch: 1,
            init_image: None,
            mask_image: opt_nonempty(&mask_path),
        },
    };
    add_value(store, id, "LATENT", payload)?;
    Ok(Fire::Done)
}

// --- InpaintModelConditioning (Mojo 2331) --------------------------------------

fn exec_inpaint_model_conditioning(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let pos_link = links.input(id, "positive");
    let neg_link = links.input(id, "negative");
    let vae_link = links.input(id, "vae");
    let pixels_link = links.input(id, "pixels");
    let mask_link = links.input(id, "mask");
    if !pos_link.found || !neg_link.found || !vae_link.found || !pixels_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph InpaintModelConditioning missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &pos_link)
        && ready(store, &neg_link)
        && ready(store, &vae_link)
        && ready(store, &pixels_link)
        && ready(store, &mask_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &pos_link, "CONDITIONING", "positive")?;
    require_value_type(store, &neg_link, "CONDITIONING", "negative")?;
    require_value_type(store, &vae_link, "VAE", "vae")?;
    require_value_type(store, &pixels_link, "IMAGE", "pixels")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let positive_text = cond_text_of(store, &pos_link)?;
    let negative_text = cond_text_of(store, &neg_link)?;
    let image_path = image_path_of(store, &pixels_link)?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    let noise_mask = wf_bool(fields, "noise_mask", true)?;
    set_if_missing(out, "init_image", json!(image_path));
    set_if_missing(out, "inpaint_conditioning_image", json!(image_path));
    set_if_missing(out, "inpaint_conditioning_mask", json!(mask_path));
    set_if_missing(out, "inpaint_conditioning_noise_mask", json!(noise_mask));
    if noise_mask {
        set_if_missing(out, "mask_image", json!(mask_path));
        set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    }
    add_value(store, id, "positive", ValuePayload::Cond { text: positive_text })?;
    add_value(store, id, "negative", ValuePayload::Cond { text: negative_text })?;
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width: 0,
            height: 0,
            batch: 1,
            init_image: opt_nonempty(&image_path),
            mask_image: if noise_mask { opt_nonempty(&mask_path) } else { None },
        },
    )?;
    Ok(Fire::Done)
}

// --- LanPaint preprocessing subgraph f07d... (Mojo 2380) -----------------------

fn exec_lanpaint_preproc(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let vae_link = links.input(id, "vae");
    let image_link = links.input(id, "image");
    let mask_link = links.input(id, "mask");
    if !vae_link.found || !image_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph LanPaint preprocessing subgraph missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &vae_link) && ready(store, &image_link) && ready(store, &mask_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &vae_link, "VAE", "vae")?;
    require_value_type(store, &image_link, "IMAGE", "image")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let init_path = image_path_of(store, &image_link)?;
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    set_if_missing(out, "init_image", json!(init_path));
    set_if_missing(out, "mask_image", json!(mask_path));
    set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    add_value(
        store,
        id,
        "LATENT",
        ValuePayload::Latent {
            width: 0,
            height: 0,
            batch: 1,
            init_image: opt_nonempty(&init_path),
            mask_image: opt_nonempty(&mask_path),
        },
    )?;
    add_value(
        store,
        id,
        "MASK",
        ValuePayload::Mask { path: mask_path, source: opt_nonempty(&mask_source) },
    )?;
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: init_path, mask_source: None },
    )?;
    Ok(Fire::Done)
}

// --- ReferenceLatent (Mojo 2413) -----------------------------------------------

fn exec_reference_latent(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    reference_latent_count: &mut i64,
) -> GraphResult<Fire> {
    let id = node.id;
    let cond_link = links.input(id, "conditioning");
    if !cond_link.found {
        return Err(GraphError::unsupported(
            "workflow graph ReferenceLatent missing conditioning input",
        )
        .with_node(id));
    }
    let latent_link = links.input(id, "latent");
    let cond_ready = ready(store, &cond_link);
    let latent_ready = if latent_link.found {
        ready(store, &latent_link)
    } else {
        true
    };
    if !(cond_ready && latent_ready) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &cond_link, "CONDITIONING", "conditioning")?;
    let text = cond_text_of(store, &cond_link)?;
    if latent_link.found {
        require_value_type(store, &latent_link, "LATENT", "latent")?;
        *reference_latent_count += 1;
        add_value_typed(store, id, "CONDITIONING", "COND_LATENT", ValuePayload::Cond { text })?;
        // Mojo 2430-2444: the same (node_id, "CONDITIONING") port is ALSO
        // appended to the latent_* lists, COPYING the source latent's
        // width/height/batch/init_image/mask_image (or zeros if the source has
        // no latent row). Register that copied geometry in the side-table so a
        // downstream sampler reading this node's latent_image re-emits init_image.
        let copied = match latent_geom_of(store, &latent_link) {
            Some(g) => {
                if !g.init_image.is_empty() {
                    set_if_missing(out, "reference_image", json!(g.init_image));
                    set_if_missing(out, "reference_latent_method", json!("index"));
                }
                latent_payload_from(&g)
            }
            None => ValuePayload::Latent {
                width: 0,
                height: 0,
                batch: 1,
                init_image: None,
                mask_image: None,
            },
        };
        store.insert_latent_geom(id, "CONDITIONING", copied);
    } else {
        add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
    }
    Ok(Fire::Done)
}

// --- Reference Conditioning subgraph 6007... (Mojo 2449) -----------------------

fn exec_reference_conditioning(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    reference_latent_count: &mut i64,
) -> GraphResult<Fire> {
    let id = node.id;
    let pos_link = links.input(id, "conditioning");
    let neg_link = links.input(id, "conditioning_1");
    let pixels_link = links.input(id, "pixels");
    let vae_link = links.input(id, "vae");
    if !pos_link.found || !neg_link.found || !pixels_link.found || !vae_link.found {
        return Err(GraphError::unsupported(
            "workflow graph Reference Conditioning subgraph missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &pos_link)
        && ready(store, &neg_link)
        && ready(store, &pixels_link)
        && ready(store, &vae_link))
    {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &pos_link, "CONDITIONING", "conditioning")?;
    require_value_type(store, &neg_link, "CONDITIONING", "conditioning_1")?;
    require_value_type(store, &pixels_link, "IMAGE", "pixels")?;
    require_value_type(store, &vae_link, "VAE", "vae")?;
    let pos_text = cond_text_of(store, &pos_link)?;
    let neg_text = cond_text_of(store, &neg_link)?;
    let reference_path = image_path_of(store, &pixels_link)?;
    set_if_missing(out, "reference_image", json!(reference_path));
    set_if_missing(out, "reference_latent_method", json!("index"));
    *reference_latent_count += 2;
    add_value_typed(store, id, "CONDITIONING", "COND_LATENT", ValuePayload::Cond { text: pos_text })?;
    add_value_typed(store, id, "CONDITIONING_1", "COND_LATENT", ValuePayload::Cond { text: neg_text })?;
    Ok(Fire::Done)
}

// --- SetNode / GetNode / Reroute / ComfySwitchNode (bus + passthrough) ---------

fn exec_setnode(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let set_link = links.setnode_input(id)?;
    if !set_link.found {
        return Err(GraphError::unsupported("workflow graph SetNode missing input").with_node(id));
    }
    let src = match store.get(set_link.node_id, &set_link.port) {
        Some(v) => v,
        None => return Ok(Fire::NotReady),
    };
    let name = setget_name(&node.fields);
    let actual = src.typ.clone();
    if !setget_supported_type(&actual) {
        return Err(GraphError::unsupported(format!(
            "workflow graph SetNode unsupported bus type: {actual}"
        ))
        .with_node(id));
    }
    let payload = src.payload.clone();
    add_value_typed(store, id, "SET", &actual, payload)?;
    // Register the bus name -> this SetNode's SET handle.
    store.set_named(name, (id, "SET".to_string()));
    Ok(Fire::Done)
}

fn exec_getnode(node: &WorkflowNode, store: &mut ValueStore) -> GraphResult<Fire> {
    let id = node.id;
    let name = setget_name(&node.fields);
    let source = match store.get_named(&name) {
        Some(s) => s.clone(),
        None => return Ok(Fire::NotReady),
    };
    let src = match store.get(source.0, &source.1) {
        Some(v) => v,
        None => return Ok(Fire::NotReady),
    };
    let actual = src.typ.clone();
    let declared = wf_string(&node.fields, "output_type");
    if !type_accepts(&declared, &actual) {
        return Err(GraphError::unsupported(format!(
            "workflow graph GetNode output type mismatch: {declared} vs {actual}"
        ))
        .with_node(id));
    }
    let payload = src.payload.clone();
    add_value_typed(store, id, "GET", &actual, payload)?;
    Ok(Fire::Done)
}

fn exec_reroute(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let input_link = links.reroute_input(id)?;
    if !input_link.found {
        return Err(GraphError::unsupported("workflow graph Reroute missing input").with_node(id));
    }
    let src = match store.get(input_link.node_id, &input_link.port) {
        Some(v) => v,
        None => return Ok(Fire::NotReady),
    };
    let actual = src.typ.clone();
    let payload = src.payload.clone();
    add_value_typed(store, id, "REROUTE", &actual, payload)?;
    Ok(Fire::Done)
}

fn exec_switch(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let false_link = links.input(id, "on_false");
    let true_link = links.input(id, "on_true");
    let switch_link = links.input(id, "switch");

    // Audit-parity fix (graph_runtime_audit_2026-06-13): ComfyUI's SwitchNode
    // uses check_lazy_status, so only the SELECTED branch is required — the
    // other branch may be absent/muted and the prompt still validates. Resolve
    // `switch_value` FIRST, then require only the selected branch's link.
    if !optional_ready(store, &switch_link) {
        return Ok(Fire::NotReady);
    }
    let mut switch_value = wf_bool(fields, "switch", false)?;
    if switch_link.found {
        switch_value = scalar_bool_of(store, &switch_link, "switch")?;
    }
    let selected = if switch_value { &true_link } else { &false_link };
    if !selected.found {
        // Only the selected branch is required; a missing selected branch is a
        // genuinely under-specified graph.
        return Err(GraphError::unsupported(
            "workflow graph ComfySwitchNode missing required typed input",
        )
        .with_node(id));
    }
    let src = match store.get(selected.node_id, &selected.port) {
        Some(v) => v,
        None => return Ok(Fire::NotReady),
    };
    let actual = src.typ.clone();
    let payload = src.payload.clone();
    add_value_typed(store, id, "output", &actual, payload)?;
    Ok(Fire::Done)
}

// --- LanPaint_MaskBlend (Mojo 2825) --------------------------------------------

fn exec_mask_blend(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
) -> GraphResult<Fire> {
    let id = node.id;
    let fields = &node.fields;
    let image1_link = links.input(id, "image1");
    let image2_link = links.input(id, "image2");
    let mask_link = links.input(id, "mask");
    if !image1_link.found || !image2_link.found || !mask_link.found {
        return Err(GraphError::unsupported(
            "workflow graph LanPaint_MaskBlend missing required typed input",
        )
        .with_node(id));
    }
    if !(ready(store, &image1_link) && ready(store, &image2_link) && ready(store, &mask_link)) {
        return Ok(Fire::NotReady);
    }
    require_value_type(store, &image1_link, "IMAGE", "image1")?;
    require_value_type(store, &image2_link, "IMAGE", "image2")?;
    require_value_type(store, &mask_link, "MASK", "mask")?;
    let mut image_path = image_path_of(store, &image2_link)?;
    if image_path.is_empty() {
        image_path = image_path_of(store, &image1_link)?;
    }
    let mask_path = image_path_of(store, &mask_link)?;
    let mask_source = mask_source_of(store, &mask_link)?;
    set_if_missing(out, "mask_image", json!(mask_path));
    set_if_missing(out, "lanpaint_mask_channel", json!(mask_source));
    copy_field_if_missing(out, fields, "blend_overlap", "lanpaint_mask_blend_overlap");
    add_value(
        store,
        id,
        "IMAGE",
        ValuePayload::Image { path: image_path, mask_source: None },
    )?;
    Ok(Fire::Done)
}
