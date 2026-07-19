"use strict";
/**
 * Properties panel for selected node.
 */
class SFProperties {
    constructor(canvas) {
        this.canvas = canvas;
        this.panel = document.getElementById('properties-panel');
        this.title = document.getElementById('properties-title');
        this.content = document.getElementById('properties-content');
        this.closeBtn = document.getElementById('properties-close');
        this.currentNodeId = null;
        this.closeBtn.addEventListener('click', () => this.hide());
    }
    show(nodeId) {
        const id = String(nodeId);
        const node = this.canvas.nodes.get(id);
        if (!node)
            return;
        this.currentNodeId = id;
        this.title.textContent = node.info.display_name || node.nodeType;
        this.content.innerHTML = '';
        // Node info group
        const infoGroup = this._createGroup('Node Info');
        this._addRow(infoGroup, 'ID', node.id);
        this._addRow(infoGroup, 'Type', node.nodeType);
        this._addRow(infoGroup, 'Category', node.info.category || 'none');
        // Widget values group
        if (Object.keys(node.widgetValues).length > 0) {
            const valGroup = this._createGroup('Values');
            for (const [name, value] of Object.entries(node.widgetValues)) {
                const inp = node.inputs.find((i) => i.name === name);
                if (!inp)
                    continue;
                const type = inp.type;
                if (type === 'STRING') {
                    const config = Array.isArray(inp.config) && inp.config[1] ? inp.config[1] : {};
                    if (config.multiline) {
                        this._addTextarea(valGroup, name, value, (v) => {
                            node.setWidgetValue(name, v);
                            if (node.widgets[name])
                                node.widgets[name].setValue(v);
                            this.canvas.nodeLayer.batchDraw();
                        });
                    }
                    else {
                        this._addInput(valGroup, name, value, 'text', (v) => {
                            node.setWidgetValue(name, v);
                            if (node.widgets[name])
                                node.widgets[name].setValue(v);
                            this.canvas.nodeLayer.batchDraw();
                        });
                    }
                }
                else if (type === 'INT' || type === 'FLOAT') {
                    this._addInput(valGroup, name, value, 'number', (v) => {
                        const num = type === 'INT' ? parseInt(v) : parseFloat(v);
                        if (!isNaN(num)) {
                            node.setWidgetValue(name, num);
                            if (node.widgets[name])
                                node.widgets[name].setValue(num);
                            this.canvas.nodeLayer.batchDraw();
                        }
                    });
                }
                else if (type === 'BOOLEAN') {
                    this._addCheckbox(valGroup, name, value, (v) => {
                        node.setWidgetValue(name, v);
                        if (node.widgets[name])
                            node.widgets[name].setValue(v);
                        this.canvas.nodeLayer.batchDraw();
                    });
                }
                else if (type === 'COMBO') {
                    const options = Array.isArray(inp.config) && Array.isArray(inp.config[0]) ? inp.config[0] : [];
                    this._addSelect(valGroup, name, value, options, (v) => {
                        node.setWidgetValue(name, v);
                        if (node.widgets[name])
                            node.widgets[name].setValue(v);
                        this.canvas.nodeLayer.batchDraw();
                    });
                }
                else if (type === 'IMAGE_PICKER') {
                    const options = Array.isArray(inp.config) && Array.isArray(inp.config[0]) ? inp.config[0] : [];
                    this._addImagePicker(valGroup, name, value, options, (v) => {
                        node.setWidgetValue(name, v);
                        if (node.widgets[name])
                            node.widgets[name].setValue(v);
                        this.canvas.nodeLayer.batchDraw();
                    });
                }
            }
        }
        this.panel.classList.remove('hidden');
    }
    hide() {
        this.panel.classList.add('hidden');
        this.currentNodeId = null;
    }
    _createGroup(title) {
        const div = document.createElement('div');
        div.className = 'prop-group';
        const t = document.createElement('div');
        t.className = 'prop-group-title';
        t.textContent = title;
        div.appendChild(t);
        this.content.appendChild(div);
        return div;
    }
    _addRow(parent, label, value) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;
        const v = document.createElement('span');
        v.className = 'prop-value';
        v.textContent = String(value);
        v.style.color = 'var(--text-secondary)';
        v.style.fontSize = '12px';
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }
    _addInput(parent, label, value, type, onChange) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;
        const v = document.createElement('div');
        v.className = 'prop-value';
        const input = document.createElement('input');
        input.className = 'prop-input';
        input.type = type;
        input.value = String(value);
        input.addEventListener('change', () => onChange(input.value));
        v.appendChild(input);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }
    _addTextarea(parent, label, value, onChange) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        row.style.flexDirection = 'column';
        row.style.alignItems = 'stretch';
        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;
        const ta = document.createElement('textarea');
        ta.className = 'prop-textarea';
        ta.value = value;
        ta.addEventListener('change', () => onChange(ta.value));
        row.appendChild(l);
        row.appendChild(ta);
        parent.appendChild(row);
    }
    _addCheckbox(parent, label, value, onChange) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;
        const v = document.createElement('div');
        v.className = 'prop-value';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = Boolean(value);
        input.addEventListener('change', () => onChange(input.checked));
        v.appendChild(input);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }
    _addSelect(parent, label, value, options, onChange) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        const l = document.createElement('span');
        l.className = 'prop-label';
        l.textContent = label;
        const v = document.createElement('div');
        v.className = 'prop-value';
        const select = document.createElement('select');
        select.className = 'prop-select';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.text = opt;
            if (opt === value)
                o.selected = true;
            select.appendChild(o);
        });
        select.addEventListener('change', () => onChange(select.value));
        v.appendChild(select);
        row.appendChild(l);
        row.appendChild(v);
        parent.appendChild(row);
    }
    _addImagePicker(parent, label, value, options, onChange) {
        const container = document.createElement('div');
        container.className = 'prop-image-picker';
        // Label
        const l = document.createElement('div');
        l.className = 'prop-label';
        l.textContent = label;
        l.style.marginBottom = '6px';
        container.appendChild(l);
        // Preview image
        const preview = document.createElement('img');
        preview.className = 'prop-image-preview';
        if (value && !value.startsWith('(')) {
            preview.src = '/view?filename=' + encodeURIComponent(value) + '&type=input';
        }
        preview.onerror = () => { preview.style.display = 'none'; };
        container.appendChild(preview);
        // Dropdown selector
        const select = document.createElement('select');
        select.className = 'prop-select';
        select.style.marginTop = '6px';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.text = opt;
            if (opt === value)
                o.selected = true;
            select.appendChild(o);
        });
        select.addEventListener('change', () => {
            const newVal = select.value;
            if (newVal && !newVal.startsWith('(')) {
                preview.src = '/view?filename=' + encodeURIComponent(newVal) + '&type=input';
                preview.style.display = '';
            }
            else {
                preview.style.display = 'none';
            }
            onChange(newVal);
        });
        container.appendChild(select);
        // Upload button
        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'prop-upload-btn';
        uploadBtn.textContent = 'Upload...';
        uploadBtn.addEventListener('click', () => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*,video/*';
            fileInput.addEventListener('change', () => {
                if (!fileInput.files || fileInput.files.length === 0)
                    return;
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('image', file);
                fetch('/upload/image', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                    if (data.name) {
                        // Add to dropdown
                        const o = document.createElement('option');
                        o.value = data.name;
                        o.text = data.name;
                        select.appendChild(o);
                        select.value = data.name;
                        preview.src = '/view?filename=' + encodeURIComponent(data.name) + '&type=input';
                        preview.style.display = '';
                        onChange(data.name);
                    }
                })
                    .catch(() => { });
            });
            fileInput.click();
        });
        container.appendChild(uploadBtn);
        parent.appendChild(container);
    }
}
//# sourceMappingURL=properties.js.map