"use strict";
/**
 * Wire type definitions. Colors match by type for visual consistency.
 * Type compatibility rules for connection validation.
 */
const SF_TYPES = {
    MODEL: { color: '#b39ddb', name: 'Model' },
    CLIP: { color: '#f06292', name: 'CLIP' },
    VAE: { color: '#ef5350', name: 'VAE' },
    CONDITIONING: { color: '#ffa726', name: 'Conditioning' },
    LATENT: { color: '#ff7043', name: 'Latent' },
    IMAGE: { color: '#64b5f6', name: 'Image' },
    MASK: { color: '#81c784', name: 'Mask' },
    CONTROL_NET: { color: '#4db6ac', name: 'ControlNet' },
    CLIP_VISION: { color: '#ce93d8', name: 'CLIP Vision' },
    CLIP_VISION_OUTPUT: { color: '#ce93d8', name: 'CLIP Vision Output' },
    UPSCALE_MODEL: { color: '#a1887f', name: 'Upscale Model' },
    AUDIO: { color: '#90caf9', name: 'Audio' },
    VIDEO: { color: '#80deea', name: 'Video' },
    NOISE: { color: '#b0bec5', name: 'Noise' },
    GUIDER: { color: '#ffcc80', name: 'Guider' },
    SAMPLER: { color: '#ef9a9a', name: 'Sampler' },
    SIGMAS: { color: '#c5e1a5', name: 'Sigmas' },
    STRING: { color: '#a5d6a7', name: 'String' },
    INT: { color: '#90caf9', name: 'Int' },
    FLOAT: { color: '#90caf9', name: 'Float' },
    BOOLEAN: { color: '#ffab91', name: 'Boolean' },
    COMBO: { color: '#b0bec5', name: 'Combo' },
    '*': { color: '#808080', name: 'Any' },
};
// Category colors for node headers
const SF_CATEGORY_COLORS = {
    'sampling': '#4a3a6a',
    'conditioning': '#3a4a3a',
    'loaders': '#3a3a5a',
    'latent': '#4a3a3a',
    'image': '#3a4a5a',
    'mask': '#3a5a3a',
    'advanced': '#4a4a3a',
    'utils': '#3a3a4a',
    '_default': '#2a2a5a',
};
function getTypeColor(typeName) {
    if (!typeName)
        return '#808080';
    const key = String(typeName).toUpperCase().replace(/ /g, '_');
    const t = SF_TYPES[key] || SF_TYPES['*'];
    return t.color;
}
function getCategoryColor(category) {
    if (!category)
        return SF_CATEGORY_COLORS._default;
    const base = category.split('/')[0].toLowerCase();
    return SF_CATEGORY_COLORS[base] || SF_CATEGORY_COLORS._default;
}
function typesCompatible(outputType, inputType) {
    const outputTypes = String(outputType || '').split(',').map(function (t) { return t.trim(); }).filter(Boolean);
    const inputTypes = String(inputType || '').split(',').map(function (t) { return t.trim(); }).filter(Boolean);
    if (outputTypes.length === 0 || inputTypes.length === 0)
        return false;
    if (outputTypes.includes('*') || inputTypes.includes('*'))
        return true;
    if (outputTypes.some(function (outType) { return inputTypes.includes(outType); }))
        return true;
    if (outputType === '*' || inputType === '*')
        return true;
    if (outputType === inputType)
        return true;
    const compatible = {
        'INT': ['FLOAT'],
        'FLOAT': ['INT'],
    };
    return outputTypes.some(function (outType) {
        return inputTypes.some(function (inType) {
            return (compatible[outType] || []).includes(inType);
        });
    });
}
//# sourceMappingURL=types.js.map