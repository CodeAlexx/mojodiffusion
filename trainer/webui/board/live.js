/**
 * live.js -- WebSocket client for SerenityBoard live updates.
 *
 * Provides a LiveConnection class that manages WebSocket connectivity
 * with auto-reconnect, subscription management, and status reporting.
 */

/**
 * WebSocket client for live training data updates.
 *
 * Usage:
 *   var live = new LiveConnection();
 *   live.onStatus(function(status) { ... });
 *   live.onMessage(function(msg) { ... });
 *   live.connect();
 *   live.subscribe(["run1"], ["loss/*"]);
 */
function LiveConnection() {
    this._ws = null;
    this._url = null;
    this._status = 'disconnected';
    this._statusHandlers = [];
    this._messageHandlers = [];
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._reconnectTimer = null;
    this._shouldReconnect = true;
    this._pendingSubscription = null;
}

/**
 * Connect to the WebSocket endpoint.
 * Defaults to ws://{current host}/ws/live
 *
 * @param {string} [url] - WebSocket URL. Auto-detected if omitted.
 */
LiveConnection.prototype.connect = function(url) {
    var self = this;

    if (!url) {
        var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        url = proto + '//' + location.host + '/ws/live';
    }
    this._url = url;
    this._shouldReconnect = true;

    this._setStatus('connecting');

    try {
        this._ws = new WebSocket(url);
    } catch (e) {
        this._setStatus('disconnected');
        this._scheduleReconnect();
        return;
    }

    this._ws.onopen = function() {
        self._reconnectDelay = 1000;  // reset backoff
        self._setStatus('connected');

        // Re-send pending subscription if any
        if (self._pendingSubscription) {
            self._sendSubscription(self._pendingSubscription.runs, self._pendingSubscription.tags, self._pendingSubscription.kinds);
        }
    };

    this._ws.onmessage = function(event) {
        var msg;
        try {
            msg = JSON.parse(event.data);
        } catch (e) {
            return;
        }
        for (var i = 0; i < self._messageHandlers.length; i++) {
            try {
                self._messageHandlers[i](msg);
            } catch (e) {
                // Handler error should not break the connection
            }
        }
    };

    this._ws.onclose = function() {
        self._ws = null;
        self._setStatus('disconnected');
        if (self._shouldReconnect) {
            self._scheduleReconnect();
        }
    };

    this._ws.onerror = function() {
        // onclose will fire after onerror, which handles reconnect
        if (self._ws) {
            self._ws.close();
        }
    };
};

/**
 * Subscribe to runs and tag patterns.
 * Sends the subscription message and stores it for reconnect.
 *
 * @param {string[]} runs - Run names to subscribe to.
 * @param {string[]} tagPatterns - Tag patterns (supports * glob).
 * @param {string[]} [kinds] - Data kinds to subscribe to (default: ["scalar"]).
 */
LiveConnection.prototype.subscribe = function(runs, tagPatterns, kinds) {
    this._pendingSubscription = { runs: runs, tags: tagPatterns, kinds: kinds || ['scalar'] };

    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._sendSubscription(runs, tagPatterns, kinds);
    }
};

/**
 * Unsubscribe (clear subscription and send empty subscribe).
 */
LiveConnection.prototype.unsubscribe = function() {
    this._pendingSubscription = null;
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._sendSubscription([], []);
    }
};

/**
 * Disconnect and stop reconnecting.
 */
LiveConnection.prototype.disconnect = function() {
    this._shouldReconnect = false;
    if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
    }
    if (this._ws) {
        this._ws.close();
        this._ws = null;
    }
    this._setStatus('disconnected');
};

/**
 * Register a callback for connection status changes.
 * @param {function} handler - Called with status string: "connected", "connecting", "disconnected"
 */
LiveConnection.prototype.onStatus = function(handler) {
    this._statusHandlers.push(handler);
};

/**
 * Register a callback for incoming messages.
 * @param {function} handler - Called with parsed JSON message object.
 */
LiveConnection.prototype.onMessage = function(handler) {
    this._messageHandlers.push(handler);
};

/**
 * Get current connection status.
 * @returns {string} "connected", "connecting", or "disconnected"
 */
LiveConnection.prototype.getStatus = function() {
    return this._status;
};

// ─── Private methods ──────────────────────────────────────────────────

LiveConnection.prototype._sendSubscription = function(runs, tags, kinds) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        var sub = { runs: runs, tags: tags };
        if (kinds && kinds.length) {
            sub.kinds = kinds;
        }
        this._ws.send(JSON.stringify({ subscribe: sub }));
    }
};

LiveConnection.prototype._setStatus = function(status) {
    if (this._status !== status) {
        this._status = status;
        for (var i = 0; i < this._statusHandlers.length; i++) {
            try {
                this._statusHandlers[i](status);
            } catch (e) {
                // Ignore handler errors
            }
        }
    }
};

LiveConnection.prototype._scheduleReconnect = function() {
    var self = this;

    if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
    }

    this._reconnectTimer = setTimeout(function() {
        self._reconnectTimer = null;
        if (self._shouldReconnect) {
            self.connect(self._url);
        }
    }, this._reconnectDelay);

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (max)
    this._reconnectDelay = Math.min(
        this._reconnectDelay * 2,
        this._maxReconnectDelay
    );
};
