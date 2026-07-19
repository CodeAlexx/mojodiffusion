"use strict";
/**
 * Port rendering and hit detection for input/output ports on nodes.
 */
const PORT_RADIUS = 5;
const PORT_HIT_RADIUS = 10;
function createOutputPort(node, slotIndex, x, y, typeName, label) {
    const color = getTypeColor(typeName);
    const group = new Konva.Group({ x: x, y: y });
    // Hit area (larger, invisible)
    const hitCircle = new Konva.Circle({
        radius: PORT_HIT_RADIUS,
        fill: 'transparent',
        listening: true,
    });
    // Visual circle
    const circle = new Konva.Circle({
        radius: PORT_RADIUS,
        fill: color,
        stroke: '#fff',
        strokeWidth: 0,
        listening: false,
    });
    // Label
    const text = new Konva.Text({
        x: -200,
        y: -6,
        width: 190,
        text: label || typeName,
        fontSize: 10,
        fill: '#c0c0d0',
        align: 'right',
        listening: false,
    });
    group.add(hitCircle);
    group.add(circle);
    group.add(text);
    // Hover effect
    hitCircle.on('mouseenter', () => {
        circle.strokeWidth(1.5);
        circle.radius(PORT_RADIUS + 1);
        node.canvas.nodeLayer.batchDraw();
        document.body.style.cursor = 'crosshair';
    });
    hitCircle.on('mouseleave', () => {
        circle.strokeWidth(0);
        circle.radius(PORT_RADIUS);
        node.canvas.nodeLayer.batchDraw();
        document.body.style.cursor = '';
    });
    // Start connection drag
    hitCircle.on('mousedown', (e) => {
        e.cancelBubble = true;
        const absPos = group.getAbsolutePosition();
        const stageTransform = node.canvas.stage.getAbsoluteTransform().copy().invert();
        const worldPos = stageTransform.point(absPos);
        node.canvas.startConnectionDrag(node.id, slotIndex, worldPos);
    });
    return group;
}
function createInputPort(node, slotIndex, x, y, typeName, label) {
    const color = getTypeColor(typeName);
    const group = new Konva.Group({ x: x, y: y });
    // Hit area
    const hitCircle = new Konva.Circle({
        radius: PORT_HIT_RADIUS,
        fill: 'transparent',
        listening: true,
    });
    // Visual circle
    const circle = new Konva.Circle({
        radius: PORT_RADIUS,
        fill: color,
        stroke: '#fff',
        strokeWidth: 0,
        listening: false,
    });
    // Label (skip if empty — widget provides its own label)
    const labelText = label || typeName;
    const text = new Konva.Text({
        x: 10,
        y: -6,
        text: label === '' ? '' : labelText,
        fontSize: 10,
        fill: '#c0c0d0',
        listening: false,
    });
    group.add(hitCircle);
    group.add(circle);
    if (label !== '')
        group.add(text);
    // Hover + connection drag highlight
    hitCircle.on('mouseenter', () => {
        if (node.canvas.draggingConnection) {
            // Show compatibility feedback
            const srcNode = node.canvas.nodes.get(node.canvas.draggingConnection.sourceNode);
            if (srcNode) {
                const srcType = srcNode.getOutputType(node.canvas.draggingConnection.sourceSlot);
                const compatible = typesCompatible(srcType, typeName);
                circle.stroke(compatible ? '#4aff4a' : '#ff4a4a');
                circle.strokeWidth(2);
                circle.radius(PORT_RADIUS + 2);
            }
        }
        else {
            circle.strokeWidth(1.5);
            circle.radius(PORT_RADIUS + 1);
        }
        node.canvas.nodeLayer.batchDraw();
        document.body.style.cursor = 'crosshair';
    });
    hitCircle.on('mouseleave', () => {
        circle.stroke('#fff');
        circle.strokeWidth(0);
        circle.radius(PORT_RADIUS);
        node.canvas.nodeLayer.batchDraw();
        document.body.style.cursor = '';
    });
    // Drop target for connection drag
    hitCircle.on('mouseup', (e) => {
        e.cancelBubble = true;
        if (node.canvas.draggingConnection) {
            node.canvas.endConnectionDrag(node.id, slotIndex);
        }
    });
    return group;
}
//# sourceMappingURL=port.js.map