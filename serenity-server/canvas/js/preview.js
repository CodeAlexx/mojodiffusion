"use strict";
/**
 * Preview panel: shows output images/videos and latent previews.
 */
class SFPreview {
    constructor(api) {
        this.api = api;
        this.panel = document.getElementById('preview-panel');
        this.content = document.getElementById('preview-content');
        this.closeBtn = document.getElementById('preview-close');
        this._currentBlobUrl = null;
        this.closeBtn.addEventListener('click', () => this.hide());
        // Make header draggable
        this._initDrag();
        // Listen for binary preview (latent)
        this.api.on('preview', (data) => {
            if (data && data.url) {
                this.showImage(data.url, true);
            }
        });
    }
    _initDrag() {
        const header = document.getElementById('preview-header');
        let dragging = false, startX = 0, startY = 0, startLeft = 0, startTop = 0;
        header.addEventListener('mousedown', (e) => {
            if (e.target.tagName === 'BUTTON')
                return;
            dragging = true;
            const rect = this.panel.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            // Switch from bottom/right positioning to top/left
            this.panel.style.left = rect.left + 'px';
            this.panel.style.top = rect.top + 'px';
            this.panel.style.bottom = 'auto';
            this.panel.style.right = 'auto';
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!dragging)
                return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            this.panel.style.left = (startLeft + dx) + 'px';
            this.panel.style.top = (startTop + dy) + 'px';
        });
        document.addEventListener('mouseup', () => {
            dragging = false;
        });
    }
    showImage(url, isBlob) {
        this.content.innerHTML = '';
        this._revokeBlobUrl();
        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Preview';
        if (isBlob)
            this._currentBlobUrl = url;
        this.content.appendChild(img);
        this.panel.classList.remove('hidden');
    }
    showVideo(url) {
        this.content.innerHTML = '';
        this._revokeBlobUrl();
        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        this.content.appendChild(video);
        this.panel.classList.remove('hidden');
    }
    hide() {
        this.panel.classList.add('hidden');
        this._revokeBlobUrl();
    }
    _revokeBlobUrl() {
        if (this._currentBlobUrl) {
            URL.revokeObjectURL(this._currentBlobUrl);
            this._currentBlobUrl = null;
        }
    }
}
//# sourceMappingURL=preview.js.map