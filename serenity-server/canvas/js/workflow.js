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
            WorkflowSync.onWorkflowLoaded(serializeWorkflow(canvas).prompt);
        }
    }
    catch (err) {
        console.error('[workflow] Load error:', err);
        throw err;
    }
}
/**
 * Load ComfyUI graph format (nodes[] + links[]).
 * Handles subgraph expansion.
 */
function loadComfyUIGraph(canvas, graphData, nodeInfo) {
    // Clear
    const existingIds = [...canvas.nodes.keys()];
    existingIds.forEach(id => canvas.removeNode(id));
    canvas.connections = [];
    // Collect subgraph definitions
    const subgraphDefs = {};
    if (graphData.definitions && graphData.definitions.subgraphs) {
        for (const sg of graphData.definitions.subgraphs) {
            subgraphDefs[sg.id] = sg;
        }
    }
    console.log('[workflow] Subgraph defs:', Object.keys(subgraphDefs).length);
    // Flatten: expand subgraphs into top-level nodes
    const allNodes = [];
    const allLinks = [];
    const skipTypes = new Set(['MarkdownNote', 'Reroute', 'Note']);
    for (const node of graphData.nodes) {
        const ntype = node.type || '';
        if (subgraphDefs[ntype]) {
            // Expand subgraph
            const sg = subgraphDefs[ntype];
            for (const inner of (sg.nodes || [])) {
                if (inner.id >= 0)
                    allNodes.push(inner);
            }
            for (const link of (sg.links || [])) {
                allLinks.push(link);
            }
            console.log('[workflow] Expanded subgraph:', ntype.substring(0, 8) + '...');
        }
        else {
            allNodes.push(node);
        }
    }
    // Add outer links
    for (const link of (graphData.links || [])) {
        allLinks.push(link);
    }
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
    // Build a map of graph node inputs: graphNodeId -> [{name, link_id, slot}]
    const graphNodeInputs = {};
    for (const node of allNodes) {
        const nid = String(node.id);
        const inputs = node.inputs || [];
        graphNodeInputs[nid] = inputs.map((inp, idx) => ({
            name: inp.name || inp.label || '',
            link: inp.link,
            slot: idx,
        }));
    }
    // Create canvas nodes
    const idMap = {}; // graph node id -> canvas node id
    let created = 0;
    for (const node of allNodes) {
        const nid = String(node.id);
        const ntype = node.type || '';
        if (skipTypes.has(ntype) || !ntype || node.id < 0)
            continue;
        const info = nodeInfo ? (nodeInfo[ntype] || {}) : {};
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

    function combinedScheduler(inputs) {
        var sampler = inputs.sampler_name || 'euler';
        if (sampler === 'euler' || sampler === 'flowmatch_euler')
            return 'euler';
        return inputs.scheduler === 'karras' ? sampler + '_k' : sampler;
    }

    function extract(prompt) {
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
                params.scheduler = 'euler';
            }
            else {
                params.prompt = findText(nodes, sinputs.positive);
                var negativeNode = valueRef(nodes, sinputs.negative);
                params.negPrompt = negativeNode && negativeNode.class_type === 'ConditioningZeroOut'
                    ? '' : findText(nodes, sinputs.negative);
                params.steps = Number(sinputs.steps);
                params.cfg = Number(sinputs.cfg);
                params.seed = Number(sinputs.seed != null ? sinputs.seed : sinputs.noise_seed);
                params.scheduler = combinedScheduler(sinputs);
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
            cfg: Number(p.cfg), guidance: Number(p.guidance), scheduler: p.scheduler || '',
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

    function onWorkflowLoaded(prompt) {
        if (suppressLoadedSync)
            return;
        var params = extract(prompt);
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
        var params = extract(serialized);
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

    return {
        extract: extract,
        markSynced: markSynced,
        onWorkflowLoaded: onWorkflowLoaded,
        syncGenerateFromCanvas: syncGenerateFromCanvas,
        syncWorkflowFromGenerate: syncWorkflowFromGenerate
    };
})();
//# sourceMappingURL=workflow.js.map
