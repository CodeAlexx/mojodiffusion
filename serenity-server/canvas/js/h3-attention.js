"use strict";
/**
 * Shared MiniMax-H3 attention admission for Generate, Canvas, and H3 Studio.
 * The server manifest is authoritative. When it is missing or the requested
 * GPU-tuned backend is unavailable, the portable cU-DNN path wins.
 */
var H3AttentionContracts = (function () {
    'use strict';

    var FALLBACKS = [
        { id: 'ck-int8', label: 'CK INT8 · GPU-tuned', available: false,
            quant_modes: ['int8-fast', 'int8', 'bf16'] },
        { id: 'cudnn', label: 'cU-DNN · portable quality default', available: true,
            quant_modes: ['int8-fast', 'int8', 'bf16'] },
        { id: 'sage-int8', label: 'Sage INT8 · experimental', available: false,
            quant_modes: ['int8-fast', 'int8'] }
    ];

    function definitions(backends) {
        return Array.isArray(backends) && backends.length ? backends : FALLBACKS;
    }

    function definition(backends, id) {
        return definitions(backends).find(function (backend) {
            return backend && backend.id === id;
        }) || null;
    }

    function isAvailable(quant, id, backends) {
        var backend = definition(backends, id);
        if (!backend || backend.available !== true)
            return false;
        var modes = Array.isArray(backend.quant_modes) ? backend.quant_modes : [];
        return modes.length === 0 || modes.indexOf(quant) >= 0;
    }

    function resolveBackend(quant, requested, backends) {
        var candidates = [requested, 'cudnn'].concat(definitions(backends).map(function (backend) {
            return backend && backend.id;
        }));
        for (var index = 0; index < candidates.length; index += 1) {
            var id = candidates[index];
            if (id && isAvailable(quant, id, backends))
                return id;
        }
        // Keep the request syntax safe even when the whole H3 runner is down.
        // The server will reject missing runtime prerequisites before GPU work.
        return 'cudnn';
    }

    return {
        fallbackDefinitions: function () { return FALLBACKS.map(function (row) {
            return Object.assign({}, row, { quant_modes: row.quant_modes.slice() });
        }); },
        definitions: definitions,
        definition: definition,
        isAvailable: isAvailable,
        resolveBackend: resolveBackend
    };
})();
