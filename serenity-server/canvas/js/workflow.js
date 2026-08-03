"use strict";
/**
 * Serialize/deserialize workflow JSON.
 * Supports two formats:
 *   1. API format: { "id": { class_type, inputs: { name: value|[src,slot] } } }
 *   2. ComfyUI graph format: { nodes: [...], links: [...] }
 */
function serializeWorkflow(canvas) {
    const prompt = {};
    const nodePositions = {};
    canvas.nodes.forEach((node, id) => {
        const inputs = {};
        node.inputs.forEach((input, idx) => {
            const conn = canvas.connections.find((c) => c.targetNode === id && c.targetSlot === idx);
            if (conn) {
                inputs[input.name] = [conn.sourceNode, conn.sourceSlot];
            }
            else if (node.widgetValues[input.name] !== undefined) {
                inputs[input.name] = node.widgetValues[input.name];
            }
        });
        prompt[id] = {
            class_type: node.nodeType,
            inputs: inputs,
        };
        nodePositions[id] = {
            type: node.nodeType,
            pos: [node.x, node.y],
            size: [node.width, node.height],
            widgets_values: { ...node.widgetValues },
            collapsed: node._collapsed || false,
        };
    });
    return { prompt, nodePositions };
}
// Workflow metadata (stored alongside canvas)
var workflowMeta = {
    name: 'Untitled Workflow',
    author: '',
    description: '',
    tags: [],
    version: '1.0',
};
/**
 * Detect format and load accordingly.
 */
function loadWorkflow(canvas, data, nodeInfo) {
    try {
        var loaded = false;
        if (data.version && data.prompt) {
            console.log('[workflow] Loading native format v' + data.version);
            if (data.metadata && typeof workflowMeta !== 'undefined') {
                Object.assign(workflowMeta, data.metadata);
            }
            deserializeWorkflow(canvas, data.prompt, data.nodes || null, nodeInfo);
            loaded = true;
        }
        else if (data.nodes && Array.isArray(data.nodes)) {
            console.log('[workflow] Loading ComfyUI graph format,', data.nodes.length, 'top-level nodes');
            loadComfyUIGraph(canvas, data, nodeInfo);
            loaded = true;
        }
        else if (typeof data === 'object' && !Array.isArray(data)) {
            const firstVal = Object.values(data)[0];
            if (firstVal && firstVal.class_type) {
                console.log('[workflow] Loading raw API format');
                deserializeWorkflow(canvas, data, null, nodeInfo);
                loaded = true;
            }
            else {
                console.warn('[workflow] Unknown format, keys:', Object.keys(data).slice(0, 5));
            }
        }
        if (loaded && typeof WorkflowSync !== 'undefined') {
            // Keep the exact source graph for the Generate handoff. A ComfyUI
            // subgraph is intentionally represented by one outer canvas node,
            // so serializing only the visible canvas would discard the loader,
            // prompt, sampler, and latent nodes stored in definitions.subgraphs.
            WorkflowSync.onWorkflowLoaded(data, serializeWorkflow(canvas).prompt);
        }
    }
    catch (err) {
        console.error('[workflow] Load error:', err);
        throw err;
    }
}
/**
 * Load ComfyUI graph format (nodes[] + links[]).
 * Comfy subgraphs stay as their serialized outer nodes. Expanding their
 * definitions without translating negative interface IDs and link namespaces
 * corrupts the graph; the outer node already carries the correct ports.
 */
function loadComfyUIGraph(canvas, graphData, nodeInfo) {
    // Clear
    const existingIds = [...canvas.nodes.keys()];
    existingIds.forEach(id => canvas.removeNode(id));
    canvas.connections = [];
    const allNodes = Array.isArray(graphData.nodes) ? graphData.nodes : [];
    const allLinks = Array.isArray(graphData.links) ? graphData.links : [];
    // Build link map: link_id -> { srcId, srcSlot, dstId, dstSlot, type }
    const linkMap = {};
    for (const link of allLinks) {
        let id, srcId, srcSlot, dstId, dstSlot, ltype;
        if (Array.isArray(link)) {
            [id, srcId, srcSlot, dstId, dstSlot, ltype] = link;
        }
        else {
            id = link.id;
            srcId = link.origin_id;
            srcSlot = link.origin_slot;
            dstId = link.target_id;
            dstSlot = link.target_slot;
            ltype = link.type || '*';
        }
        linkMap[String(id)] = { srcId: String(srcId), srcSlot: srcSlot, dstId: String(dstId), dstSlot: dstSlot, type: String(ltype) };
    }
    // Create canvas nodes
    const idMap = {}; // graph node id -> canvas node id
    let created = 0;
    for (const node of allNodes) {
        const nid = String(node.id);
        const ntype = node.type || '';
        if (!ntype || node.id < 0)
            continue;
        const registeredInfo = nodeInfo ? nodeInfo[ntype] : null;
        const info = _comfyNodeInfo(node, registeredInfo);
        // Position
        let px = 0, py = 0;
        const pos = node.pos;
        if (Array.isArray(pos) && pos.length >= 2) {
            px = pos[0];
            py = pos[1];
        }
        else if (pos && typeof pos === 'object') {
            px = pos['0'] || 0;
            py = pos['1'] || 0;
        }
        try {
            const canvasNode = canvas.addNode(ntype, px, py, info);
            idMap[nid] = canvasNode.id;
            created++;
            // Apply widgets_values
            if (node.widgets_values && info) {
                _applyWidgetValues(canvasNode, node.widgets_values, info);
            }
        }
        catch (err) {
            console.error('[workflow] Failed to create node', nid, ntype, err);
        }
    }
    console.log('[workflow] Created', created, 'nodes');
    // Create connections using graph node input link references
    let connCount = 0;
    for (const node of allNodes) {
        const nid = String(node.id);
        const dstCanvasId = idMap[nid];
        if (!dstCanvasId)
            continue;
        const dstNode = canvas.nodes.get(dstCanvasId);
        if (!dstNode)
            continue;
        const inputs = node.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            const inp = inputs[i];
            const linkId = inp.link;
            if (linkId == null)
                continue;
            const link = linkMap[linkId];
            if (!link)
                continue;
            const srcCanvasId = idMap[link.srcId];
            if (!srcCanvasId)
                continue;
            // Find the target slot by name match in the canvas node
            const inputName = inp.name || inp.label || '';
            let targetSlot = dstNode.getInputIndex(inputName);
            // Fallback: use positional index if name doesn't match
            if (targetSlot < 0 && i < dstNode.inputs.length) {
                targetSlot = i;
            }
            if (targetSlot >= 0) {
                try {
                    canvas.addConnection(srcCanvasId, link.srcSlot, dstCanvasId, targetSlot);
                    connCount++;
                }
                catch (err) {
                    console.error('[workflow] Failed to connect', link.srcId, '->', nid, err);
                }
            }
        }
    }
    console.log('[workflow] Created', connCount, 'connections');
    canvas.nodeLayer.batchDraw();
    canvas.connectionLayer.batchDraw();
    canvas.fitView(true);
    // Update topbar model badge from the primary loader node
    _updateModelBadgeFromWorkflow(allNodes);
}
/**
 * Merge registered object_info with the ports serialized by ComfyUI.
 *
 * Imported custom nodes are often absent from Serenity's runtime. Drawing them
 * with an empty info object drops every input/output and therefore every link.
 * The graph JSON already has authoritative port names, types, and ordering, so
 * use those for visual fidelity while retaining registered widget definitions.
 */
function _comfyNodeInfo(node, registeredInfo) {
    const registeredInput = registeredInfo && registeredInfo.input
        ? registeredInfo.input : {};
    const required = Object.assign({}, registeredInput.required || {});
    const optional = Object.assign({}, registeredInput.optional || {});
    (node.inputs || []).forEach(function (input, index) {
        const name = input.name || input.label || ('input_' + index);
        if (required[name] || optional[name])
            return;
        required[name] = [String(input.type || '*')];
    });
    const serializedOutputs = Array.isArray(node.outputs) ? node.outputs : [];
    return {
        input: { required: required, optional: optional },
        output: serializedOutputs.length
            ? serializedOutputs.map(function (output) {
                return String(output.type || '*');
            })
            : (registeredInfo && registeredInfo.output || []),
        output_name: serializedOutputs.length
            ? serializedOutputs.map(function (output, index) {
                return String(output.name || output.label ||
                    output.type || ('output_' + index));
            })
            : (registeredInfo && registeredInfo.output_name || []),
        display_name: node.title ||
            (registeredInfo && registeredInfo.display_name) || node.type,
        category: registeredInfo && registeredInfo.category ||
            'Imported Workflow'
    };
}
/**
 * Map widgets_values array to named input fields.
 * ComfyUI stores widget values as a flat array. Widget-type inputs appear
 * in the order defined by INPUT_TYPES (required then optional).
 */
function _applyWidgetValues(canvasNode, widgetValues, info) {
    if (!Array.isArray(widgetValues) || widgetValues.length === 0)
        return;
    // Get widget-type inputs in order
    const widgetInputs = [];
    for (const inp of canvasNode.inputs) {
        if (['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'].includes(inp.type)) {
            widgetInputs.push(inp.name);
        }
    }
    let vi = 0;
    for (let i = 0; i < widgetInputs.length && vi < widgetValues.length; i++) {
        const name = widgetInputs[i];
        const value = widgetValues[vi];
        canvasNode.setWidgetValue(name, value);
        if (canvasNode.widgets[name]) {
            canvasNode.widgets[name].setValue(value);
        }
        vi++;
        // Some widgets consume an extra value (seed + control_after_generate)
        if ((name.includes('seed') || name === 'noise_seed') && vi < widgetValues.length) {
            const next = widgetValues[vi];
            if (typeof next === 'string' && ['fixed', 'increment', 'decrement', 'randomize'].includes(next)) {
                vi++;
            }
        }
    }
}
/**
 * Deserialize API format prompt.
 */
function deserializeWorkflow(canvas, prompt, nodePositions, nodeInfo) {
    const nodeIds = [...canvas.nodes.keys()];
    nodeIds.forEach(id => canvas.removeNode(id));
    canvas.connections = [];
    if (!prompt || typeof prompt !== 'object')
        return;
    const hasSavedPositions = !!nodePositions && Object.keys(nodePositions).some(function (id) {
        const entry = nodePositions[id];
        return entry && Array.isArray(entry.pos) && entry.pos.length >= 2;
    });
    let x = 100, y = 100;
    const idMap = {};
    for (const [origId, nodeData] of Object.entries(prompt)) {
        if (!nodeData || !nodeData.class_type)
            continue;
        const info = nodeInfo ? (nodeInfo[nodeData.class_type] || {}) : {};
        let posX = x, posY = y;
        if (hasSavedPositions && nodePositions[origId]) {
            const pos = nodePositions[origId].pos;
            if (pos) {
                posX = pos[0];
                posY = pos[1];
            }
        }
        const node = canvas.addNode(nodeData.class_type, posX, posY, info);
        idMap[origId] = node.id;
        for (const [name, value] of Object.entries(nodeData.inputs || {})) {
            if (!Array.isArray(value)) {
                node.setWidgetValue(name, value);
                if (node.widgets[name]) {
                    node.widgets[name].setValue(value);
                }
            }
        }
        const widgetVals = nodePositions && nodePositions[origId]
            ? nodePositions[origId].widgets_values
            : null;
        if (widgetVals) {
            for (const [name, value] of Object.entries(widgetVals)) {
                node.setWidgetValue(name, value);
                if (node.widgets[name]) {
                    node.widgets[name].setValue(value);
                }
            }
        }
        // Restore collapsed state
        if (nodePositions && nodePositions[origId] && nodePositions[origId].collapsed) {
            node.toggleCollapse();
        }
        if (!hasSavedPositions) {
            y += 200;
            if (y > 1200) {
                y = 100;
                x += 300;
            }
        }
    }
    for (const [origId, nodeData] of Object.entries(prompt)) {
        if (!nodeData || !nodeData.inputs)
            continue;
        for (const [name, value] of Object.entries(nodeData.inputs)) {
            if (Array.isArray(value) && value.length === 2) {
                const sourceId = idMap[String(value[0])];
                const targetId = idMap[origId];
                const sourceSlot = value[1];
                if (!sourceId || !targetId)
                    continue;
                const targetNode = canvas.nodes.get(targetId);
                if (!targetNode)
                    continue;
                const targetSlot = targetNode.getInputIndex(name);
                if (targetSlot < 0)
                    continue;
                canvas.addConnection(sourceId, sourceSlot, targetId, targetSlot);
            }
        }
    }
    if (!hasSavedPositions && canvas.autoLayout) {
        canvas.autoLayout();
    }
    else {
        // Saved coordinates define the arrangement, not the zoom level. Fit
        // the whole graph so wide workflows remain visible after loading.
        canvas.fitView(true);
    }
    // Update topbar model badge from the workflow
    _updateModelBadgeFromPrompt(prompt);
}
/**
 * Scan litegraph nodes for a model loader and update the topbar badge.
 */
function _updateModelBadgeFromWorkflow(nodes) {
    var loaderTypes = ['CheckpointLoaderSimple', 'UNETLoader', 'LTXVLoader'];
    for (var i = 0; i < nodes.length; i++) {
        var ntype = nodes[i].type || '';
        if (loaderTypes.indexOf(ntype) < 0)
            continue;
        var wvals = nodes[i].widgets_values;
        if (wvals && wvals.length > 0 && typeof wvals[0] === 'string') {
            _setModelBadge(wvals[0]);
            return;
        }
    }
}
/**
 * Scan API-format prompt for a model loader and update the topbar badge.
 */
function _updateModelBadgeFromPrompt(prompt) {
    var loaderTypes = ['CheckpointLoaderSimple', 'UNETLoader', 'LTXVLoader'];
    for (var id in prompt) {
        var nd = prompt[id];
        if (!nd || !nd.class_type)
            continue;
        if (loaderTypes.indexOf(nd.class_type) < 0)
            continue;
        var inputs = nd.inputs || {};
        var modelName = inputs.ckpt_name || inputs.unet_name || inputs.checkpoint_path;
        if (modelName && typeof modelName === 'string') {
            _setModelBadge(modelName);
            return;
        }
    }
}
/**
 * Set the topbar model badge text.
 */
function _setModelBadge(modelName) {
    var badge = document.querySelector('.model-badge');
    if (!badge)
        return;
    var short = modelName.split('/').pop().replace(/\.\w+$/, '');
    badge.textContent = short;
}

/**
 * Keep the Generate and Workflow screens on one product request.  The graph is
 * authoritative while it is being edited; Generate is authoritative after its
 * controls change.  A stable signature prevents tab switches from replacing a
 * custom graph when the shared controls have not changed.
 */
var WorkflowSync = (function () {
    var lastSyncedSignature = '';
    var suppressLoadedSync = false;
    var stagedWorkflowPending = false;
    var lastLoadedWorkflowSource = null;
    var lastLoadedCanvasSignature = '';

    function canonicalModelName(name) {
        var original = String(name || '').trim();
        var normalized = original.toLowerCase().replace(/\\/g, '/');
        if (!normalized)
            return '';
        if (normalized.indexOf('ideogram') >= 0) {
            if (normalized.indexOf('unconditional') >= 0)
                return '';
            if (normalized === 'ideogram-4-bf16-diffusers' ||
                normalized.indexOf('bf16') >= 0)
                return 'ideogram-4-bf16-diffusers';
            // Comfy's creator workflow names the component
            // diffusion_models/ideogram4_fp8_scaled.safetensors, while the
            // Serenity model registry exposes the admitted product identity.
            return 'ideogram-4-fp8';
        }
        return original;
    }

    function comfyWidgetValues(node) {
        return node && Array.isArray(node.widgets_values)
            ? node.widgets_values : [];
    }

    function activeComfyNodes(graph) {
        var nodes = [];
        function append(list, subgraphName) {
            (Array.isArray(list) ? list : []).forEach(function (node) {
                if (!node || node.mode === 4)
                    return;
                nodes.push({ node: node, subgraphName: subgraphName || '' });
            });
        }
        append(graph && graph.nodes, '');
        var definitions = graph && graph.definitions;
        (definitions && Array.isArray(definitions.subgraphs)
            ? definitions.subgraphs : []).forEach(function (subgraph) {
            append(subgraph.nodes, subgraph.name || '');
        });
        return nodes;
    }

    function usableComfyPrompt(value) {
        if (typeof value !== 'string')
            return '';
        var text = value.trim();
        if (!text || text.length > 20000)
            return '';
        if (/^IGNORE AND DELETE ALL PREVIOUS INSTRUCTIONS/i.test(text) ||
            (text.indexOf('[SYSTEM]') >= 0 && text.indexOf('OUTPUT CONTRACT') >= 0))
            return '';
        return text;
    }

    /**
     * Read product parameters from a full ComfyUI graph without expanding its
     * subgraphs on the visual canvas. The creator graph remains authoritative;
     * this is analysis-only and never rewrites its topology.
     */
    function extractComfyGraph(graph) {
        var params = { loras: [] };
        var entries = activeComfyNodes(graph);
        var promptCandidates = [];
        var negativeCandidates = [];
        var seenLoras = {};
        var modelLoader = null;
        var latentNode = null;
        var samplerNode = null;
        var basicScheduler = null;
        var samplerSelect = null;
        var guider = null;
        var randomNoise = null;
        var seedNode = null;
        var qualityChoice = null;

        entries.forEach(function (entry) {
            var node = entry.node;
            var type = String(node.type || '');
            var values = comfyWidgetValues(node);
            if (!modelLoader &&
                (type === 'CheckpointLoaderSimple' || type === 'UNETLoader' ||
                    type === 'LTXVLoader') &&
                typeof values[0] === 'string' &&
                values[0].toLowerCase().indexOf('unconditional') < 0) {
                modelLoader = node;
            }
            if (!latentNode && /^Empty.*(?:Latent|Video)/.test(type))
                latentNode = node;
            if (!samplerNode &&
                (type === 'KSampler' || type === 'KSamplerAdvanced' ||
                    type === 'LTXVSampler'))
                samplerNode = node;
            if (!basicScheduler && type === 'BasicScheduler')
                basicScheduler = node;
            if (!samplerSelect && type === 'KSamplerSelect')
                samplerSelect = node;
            if (!guider && (type === 'DualModelGuider' || type === 'CFGGuider'))
                guider = node;
            if (!randomNoise && type === 'RandomNoise')
                randomNoise = node;
            if (!seedNode && type === 'Seed (rgthree)')
                seedNode = node;
            if (!qualityChoice && type === 'CustomCombo' &&
                typeof values[0] === 'string')
                qualityChoice = values[0];

            if (type === 'CLIPTextEncode' || type === 'TextEncodeQwenImageEditPlus') {
                var encoded = usableComfyPrompt(values[0]);
                if (encoded)
                    promptCandidates.push({ priority: 100, text: encoded });
            }
            if (type === 'PrimitiveStringMultiline') {
                var primitive = usableComfyPrompt(values[0]);
                if (primitive) {
                    promptCandidates.push({
                        priority: /caption prompt/i.test(entry.subgraphName) ? 80 : 40,
                        text: primitive
                    });
                }
            }
            if ((type === 'LoraLoader' || type === 'LoraLoaderModelOnly') &&
                typeof values[0] === 'string' && values[0]) {
                var loraName = values[0];
                if (!seenLoras[loraName]) {
                    seenLoras[loraName] = true;
                    params.loras.push({
                        name: loraName,
                        strength: Number(values[1] == null ? 1 : values[1]),
                        enabled: true
                    });
                }
            }
        });

        // Subgraph instance widgets can hold the concrete prompt while the
        // definition contains only a placeholder/default.
        var definitionsById = {};
        var definitions = graph && graph.definitions;
        (definitions && Array.isArray(definitions.subgraphs)
            ? definitions.subgraphs : []).forEach(function (subgraph) {
            definitionsById[String(subgraph.id)] = subgraph;
        });
        (Array.isArray(graph && graph.nodes) ? graph.nodes : []).forEach(function (node) {
            var definition = definitionsById[String(node.type || '')];
            if (!definition)
                return;
            var values = comfyWidgetValues(node);
            var widgetInputs = (definition.inputs || []).filter(function (input) {
                return !!input.widget;
            });
            widgetInputs.forEach(function (input, index) {
                if (!/prompt|text|value/i.test(String(input.name || '') + ' ' +
                    String(input.label || '')))
                    return;
                var candidate = usableComfyPrompt(values[index]);
                if (candidate) {
                    promptCandidates.push({
                        priority: /caption prompt/i.test(definition.name || '') ? 120 : 90,
                        text: candidate
                    });
                }
            });
        });

        if (modelLoader)
            params.model = canonicalModelName(comfyWidgetValues(modelLoader)[0]);
        if (latentNode) {
            var latentValues = comfyWidgetValues(latentNode);
            if (Number.isFinite(Number(latentValues[0])))
                params.width = Number(latentValues[0]);
            if (Number.isFinite(Number(latentValues[1])))
                params.height = Number(latentValues[1]);
            if (Number.isFinite(Number(latentValues[2])) &&
                /Video/.test(String(latentNode.type || '')))
                params.frames = Number(latentValues[2]);
        }
        if (samplerNode) {
            var samplerValues = comfyWidgetValues(samplerNode);
            if (samplerNode.type === 'KSampler' ||
                samplerNode.type === 'KSamplerAdvanced') {
                params.seed = Number(samplerValues[0]);
                params.steps = Number(samplerValues[2]);
                params.cfg = Number(samplerValues[3]);
                params.sampler = samplerValues[4] || 'euler';
                params.noiseScheduler = samplerValues[5] || 'simple';
                params.scheduler = params.sampler;
            }
        }
        if (basicScheduler) {
            var schedulerValues = comfyWidgetValues(basicScheduler);
            params.noiseScheduler = schedulerValues[0] || 'simple';
            params.steps = Number(schedulerValues[1]);
        }
        if (qualityChoice && params.model === 'ideogram-4-fp8') {
            var ideogramSteps = { Quality: 48, Default: 20, Turbo: 12 };
            if (ideogramSteps[qualityChoice] != null)
                params.steps = ideogramSteps[qualityChoice];
        }
        if (samplerSelect) {
            params.sampler = comfyWidgetValues(samplerSelect)[0] || 'euler';
            params.scheduler = params.sampler;
        }
        if (guider)
            params.cfg = Number(comfyWidgetValues(guider)[0]);
        if (seedNode)
            params.seed = Number(comfyWidgetValues(seedNode)[0]);
        else if (randomNoise)
            params.seed = Number(comfyWidgetValues(randomNoise)[0]);

        promptCandidates.sort(function (a, b) { return b.priority - a.priority; });
        negativeCandidates.sort(function (a, b) { return b.priority - a.priority; });
        params.prompt = promptCandidates.length ? promptCandidates[0].text : '';
        params.negPrompt = negativeCandidates.length ? negativeCandidates[0].text : '';
        Object.keys(params).forEach(function (key) {
            if (typeof params[key] === 'number' && !Number.isFinite(params[key]))
                delete params[key];
        });
        return params;
    }

    function valueRef(prompt, ref) {
        if (!Array.isArray(ref) || ref.length < 1)
            return null;
        return prompt[String(ref[0])] || null;
    }

    function findText(prompt, ref, seen) {
        var node = valueRef(prompt, ref);
        if (!node)
            return '';
        seen = seen || {};
        var key = String(ref[0]);
        if (seen[key])
            return '';
        seen[key] = true;
        var inputs = node.inputs || {};
        if (typeof inputs.text === 'string')
            return inputs.text;
        if (typeof inputs.prompt === 'string')
            return inputs.prompt;
        if (typeof inputs.positive_prompt === 'string')
            return inputs.positive_prompt;
        var refs = ['conditioning', 'positive', 'clip', 'source'];
        for (var i = 0; i < refs.length; i++) {
            if (Array.isArray(inputs[refs[i]])) {
                var text = findText(prompt, inputs[refs[i]], seen);
                if (text)
                    return text;
            }
        }
        return '';
    }

    function extract(prompt) {
        if (prompt && prompt.workflow && Array.isArray(prompt.workflow.nodes))
            return extractComfyGraph(prompt.workflow);
        if (prompt && Array.isArray(prompt.nodes))
            return extractComfyGraph(prompt);
        if (prompt && prompt.version && prompt.prompt)
            return extract(prompt.prompt);
        var params = { loras: [] };
        var nodes = prompt || {};
        var sampler = null;
        var textNodes = [];
        Object.keys(nodes).forEach(function (id) {
            var node = nodes[id] || {};
            var inputs = node.inputs || {};
            if (!params.model && (node.class_type === 'CheckpointLoaderSimple' || node.class_type === 'UNETLoader' || node.class_type === 'LTXVLoader')) {
                params.model = inputs.ckpt_name || inputs.unet_name || inputs.checkpoint_path || '';
            }
            if (!sampler && (node.class_type === 'KSampler' || node.class_type === 'KSamplerAdvanced' || node.class_type === 'LTXVSampler')) {
                sampler = node;
            }
            if (node.class_type === 'CLIPTextEncode' || node.class_type === 'TextEncodeQwenImageEditPlus') {
                textNodes.push(node);
            }
            if (node.class_type === 'LoraLoader' && typeof inputs.lora_name === 'string') {
                params.loras.push({
                    name: inputs.lora_name,
                    strength: Number(inputs.strength_model == null ? 1 : inputs.strength_model),
                    enabled: true
                });
            }
            if (inputs.width != null && inputs.height != null &&
                (/^Empty.*(?:Latent|Video)/.test(node.class_type || '') || node.class_type === 'LTXVSampler')) {
                params.width = Number(inputs.width);
                params.height = Number(inputs.height);
            }
            if (node.class_type === 'FluxGuidance' && inputs.guidance != null) {
                params.guidance = Number(inputs.guidance);
            }
            if (node.class_type === 'SaveVideo' || node.class_type === 'SaveAnimatedWEBP') {
                if (inputs.fps != null)
                    params.fps = Number(inputs.fps);
            }
        });
        if (sampler) {
            var sinputs = sampler.inputs || {};
            if (sampler.class_type === 'LTXVSampler') {
                params.prompt = typeof sinputs.prompt === 'string' ? sinputs.prompt : '';
                params.negPrompt = typeof sinputs.negative_prompt === 'string' ? sinputs.negative_prompt : '';
                params.width = Number(sinputs.width);
                params.height = Number(sinputs.height);
                params.frames = Number(sinputs.num_frames);
                params.fps = Number(sinputs.frame_rate);
                params.steps = Number(sinputs.steps);
                params.cfg = Number(sinputs.cfg);
                params.seed = Number(sinputs.seed);
                params.sampler = sinputs.sampler || 'euler';
                params.scheduler = sinputs.scheduler || 'simple';
            }
            else {
                params.prompt = findText(nodes, sinputs.positive);
                var negativeNode = valueRef(nodes, sinputs.negative);
                params.negPrompt = negativeNode && negativeNode.class_type === 'ConditioningZeroOut'
                    ? '' : findText(nodes, sinputs.negative);
                params.steps = Number(sinputs.steps);
                params.cfg = Number(sinputs.cfg);
                params.seed = Number(sinputs.seed != null ? sinputs.seed : sinputs.noise_seed);
                params.sampler = sinputs.sampler_name || 'euler';
                params.scheduler = sinputs.scheduler || 'simple';
                var latent = valueRef(nodes, sinputs.latent_image);
                if (latent && latent.inputs) {
                    if (latent.inputs.width != null)
                        params.width = Number(latent.inputs.width);
                    if (latent.inputs.height != null)
                        params.height = Number(latent.inputs.height);
                    if (latent.inputs.length != null)
                        params.frames = Number(latent.inputs.length);
                }
            }
        }
        if (typeof params.prompt !== 'string' || !params.prompt) {
            params.prompt = textNodes.length && typeof textNodes[0].inputs.text === 'string'
                ? textNodes[0].inputs.text : '';
        }
        if (typeof params.negPrompt !== 'string') {
            params.negPrompt = textNodes.length > 1 && typeof textNodes[1].inputs.text === 'string'
                ? textNodes[1].inputs.text : '';
        }
        if (params.model)
            params.model = canonicalModelName(params.model);
        Object.keys(params).forEach(function (key) {
            if (typeof params[key] === 'number' && !Number.isFinite(params[key]))
                delete params[key];
        });
        return params;
    }

    function signature(params) {
        var p = params || {};
        return JSON.stringify({
            model: p.model || '', prompt: p.prompt || '', negPrompt: p.negPrompt || '',
            width: Number(p.width), height: Number(p.height), steps: Number(p.steps),
            cfg: Number(p.cfg), guidance: Number(p.guidance),
            sampler: p.sampler || '', scheduler: p.scheduler || '',
            seed: Number(p.seed), frames: Number(p.frames), fps: Number(p.fps),
            loras: Array.isArray(p.loras) ? p.loras.map(function (l) {
                return { name: l.name || '', strength: Number(l.strength), enabled: l.enabled !== false };
            }) : []
        });
    }

    function applyToGenerate(params) {
        if (typeof GenerateTab === 'undefined' || !GenerateTab.applyParams)
            return;
        GenerateTab.applyParams(params, { source: 'workflow' });
    }

    function markSynced(params) {
        lastSyncedSignature = signature(params);
    }

    function onWorkflowLoaded(source, serializedPrompt) {
        lastLoadedWorkflowSource = source || null;
        lastLoadedCanvasSignature = JSON.stringify(serializedPrompt || {});
        if (suppressLoadedSync)
            return;
        var params = extract(source);
        applyToGenerate(params);
        markSynced(typeof GenerateTab !== 'undefined' && GenerateTab.getParams
            ? GenerateTab.getParams() : params);
    }

    function syncGenerateFromCanvas() {
        if (typeof sfCanvas === 'undefined' || !sfCanvas)
            return false;
        var serialized = serializeWorkflow(sfCanvas).prompt;
        if (!Object.keys(serialized).length)
            return false;
        var serializedSignature = JSON.stringify(serialized);
        var source = lastLoadedWorkflowSource &&
            serializedSignature === lastLoadedCanvasSignature
            ? lastLoadedWorkflowSource : serialized;
        if (source === serialized)
            lastLoadedWorkflowSource = null;
        var params = extract(source);
        applyToGenerate(params);
        markSynced(typeof GenerateTab !== 'undefined' && GenerateTab.getParams
            ? GenerateTab.getParams() : params);
        return true;
    }

    function syncWorkflowFromGenerate(params) {
        if (!params || !params.model || typeof sfCanvas === 'undefined' || !sfCanvas ||
            typeof WorkflowBuilder === 'undefined')
            return false;
        var nextSignature = signature(params);
        var hasNodes = sfCanvas.nodes && sfCanvas.nodes.size > 0;
        if (hasNodes && nextSignature === lastSyncedSignature)
            return false;
        var graph = WorkflowBuilder.build(params);
        suppressLoadedSync = true;
        try {
            loadWorkflow(sfCanvas, graph, sfCanvas.nodeInfo);
        }
        finally {
            suppressLoadedSync = false;
        }
        markSynced(params);
        return true;
    }

    /**
     * Put an already-built product graph into the Workflow editor without
     * translating it through Generate controls. The next visit to Workflow
     * consumes this handoff so tab switching cannot replace the exact graph.
     */
    function stageWorkflow(workflow, options) {
        if (!workflow || typeof workflow !== 'object' ||
            typeof sfCanvas === 'undefined' || !sfCanvas ||
            typeof loadWorkflow === 'undefined')
            return false;
        suppressLoadedSync = true;
        try {
            loadWorkflow(sfCanvas, workflow, sfCanvas.nodeInfo);
        }
        finally {
            suppressLoadedSync = false;
        }
        stagedWorkflowPending = true;
        options = options || {};
        if (options.name) {
            var nameInput = document.getElementById('workflow-name');
            if (nameInput)
                nameInput.value = options.name;
            if (typeof workflowMeta !== 'undefined')
                workflowMeta.name = options.name;
        }
        return true;
    }

    function hasStagedWorkflow() {
        return stagedWorkflowPending;
    }

    function consumeStagedWorkflow() {
        var hadPending = stagedWorkflowPending;
        stagedWorkflowPending = false;
        return hadPending;
    }

    return {
        extract: extract,
        markSynced: markSynced,
        onWorkflowLoaded: onWorkflowLoaded,
        syncGenerateFromCanvas: syncGenerateFromCanvas,
        syncWorkflowFromGenerate: syncWorkflowFromGenerate,
        stageWorkflow: stageWorkflow,
        hasStagedWorkflow: hasStagedWorkflow,
        consumeStagedWorkflow: consumeStagedWorkflow
    };
})();
