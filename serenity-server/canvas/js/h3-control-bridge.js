"use strict";
/**
 * Native bridge for the dedicated H3 ControlNet screen.
 *
 * The screen preserves the reference product's interaction contract, while
 * every operation is delegated to the Mojo inference shell's shared API and
 * WebSocket. There is no Python route or fallback here.
 */
var SerenityH3WS = SerenityWS;

var SerenityH3API = (function () {
    'use strict';
    function createPromptId() {
        return crypto.randomUUID ? crypto.randomUUID() :
            'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
    }
    return {
        createPromptId: createPromptId,
        postPrompt: function (workflow, metadata) {
            return SerenityAPI.postPrompt(workflow, metadata);
        },
        interrupt: function () { return SerenityAPI.interrupt(); },
        uploadMediaDetails: function (file) {
            return SerenityAPI.uploadMediaDetails(file);
        },
        uploadMedia: function (file) { return SerenityAPI.uploadMedia(file); },
        viewUrl: function (filename, subfolder, type) {
            return SerenityAPI.viewUrl(filename, subfolder, type);
        }
    };
})();

// Project the native H3 graph schemas into the visible workflow canvas.
var SerenityH3WorkflowRegistry = (function () {
    'use strict';
    var nodeTypes = [
        'MiniMaxH3TextEncode',
        'MiniMaxH3Loader',
        'MiniMaxH3EmptyLatent',
        'MiniMaxH3SigmaShift',
        'MiniMaxH3FunControlNetLoader',
        'MiniMaxH3FunControlNetApplyMedia',
        'MiniMaxH3Sampler',
        'MiniMaxH3DecodeRelease',
        'CreateVideo',
        'SaveVideo'
    ];
    function install(info) {
        var projected = {};
        nodeTypes.forEach(function (nodeType) {
            if (info && info[nodeType]) projected[nodeType] = info[nodeType];
        });
        if (typeof sfCanvas !== 'undefined' && sfCanvas)
            sfCanvas.nodeInfo = Object.assign({}, sfCanvas.nodeInfo || {}, projected);
        if (typeof sfSidebar !== 'undefined' && sfSidebar) {
            sfSidebar.nodeTypes = Object.assign({}, sfSidebar.nodeTypes || {}, projected);
            if (sfSidebar._buildCategories && sfSidebar._render) {
                sfSidebar._buildCategories();
                sfSidebar._render(sfSidebar.searchInput ? sfSidebar.searchInput.value : '');
            }
        }
        return Object.keys(projected).length;
    }
    return { install: install, nodeTypes: nodeTypes.slice() };
})();

if (typeof mirrorPromptToWorkflow === 'undefined') {
    var mirrorPromptToWorkflow = function (prompt, metadata) {
        if (typeof WorkflowSync === 'undefined' || !WorkflowSync.stageWorkflow)
            return false;
        return WorkflowSync.stageWorkflow(prompt, {
            name: metadata && metadata.name || 'MiniMax H3 ControlNet Union'
        });
    };
}
