"use strict";
/**
 * SerenityWS — Shared WebSocket client for SerenityFlow.
 * Singleton pub/sub with exponential backoff reconnect.
 */
var SerenityWS = (function () {
    'use strict';
    var socket = null;
    var listeners = {};
    var connected = false;
    var reconnectAttempts = 0;
    var reconnectTimer = null;
    var MAX_RECONNECT_DELAY = 30000;
    var BASE_DELAY = 1000;
    // Persistent client ID
    var clientId = localStorage.getItem('sf-client-id') || '';
    if (!clientId) {
        clientId = crypto.randomUUID ? crypto.randomUUID() :
            'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        localStorage.setItem('sf-client-id', clientId);
    }
    function getReconnectDelay() {
        var exp = Math.min(BASE_DELAY * Math.pow(2, reconnectAttempts), MAX_RECONNECT_DELAY);
        var jitter = Math.random() * 0.3 * exp;
        return Math.round(exp + jitter);
    }
    function connect() {
        if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
            return;
        }
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        var host = location.host || 'localhost:8188';
        var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(protocol + '//' + host + '/ws?clientId=' + clientId);
        socket.onopen = function () {
            connected = true;
            reconnectAttempts = 0;
            emit('connected', {});
        };
        socket.onmessage = function (event) {
            if (event.data instanceof Blob) {
                emit('preview', { blob: event.data });
                return;
            }
            try {
                var msg = JSON.parse(event.data);
                var type = msg.type;
                var data = msg.data || msg;
                emit(type, data);
            }
            catch (e) {
                // Ignore parse errors
            }
        };
        socket.onclose = function (event) {
            connected = false;
            emit('disconnected', {});
            // Only reconnect on abnormal close
            if (event.code !== 1000) {
                var delay = getReconnectDelay();
                reconnectAttempts++;
                reconnectTimer = setTimeout(connect, delay);
            }
        };
        socket.onerror = function () {
            // onclose will fire after this
        };
    }
    function emit(type, data) {
        var handlers = listeners[type];
        if (handlers) {
            handlers.forEach(function (fn) {
                try {
                    fn(data);
                }
                catch (e) {
                    console.error('WS listener error:', e);
                }
            });
        }
        // Wildcard listeners
        var wildcard = listeners['*'];
        if (wildcard && type !== '*') {
            wildcard.forEach(function (fn) {
                try {
                    fn({ type: type, data: data });
                }
                catch (e) { }
            });
        }
    }
    function on(type, fn) {
        if (!listeners[type])
            listeners[type] = [];
        listeners[type].push(fn);
    }
    function off(type, fn) {
        if (!listeners[type])
            return;
        listeners[type] = listeners[type].filter(function (f) { return f !== fn; });
    }
    function send(data) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(data));
        }
    }
    function getClientId() { return clientId; }
    function isConnected() { return connected; }
    // Page visibility — reconnect when tab becomes visible
    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'visible') {
            if (!socket || socket.readyState === WebSocket.CLOSED) {
                reconnectAttempts = 0;
                connect();
            }
        }
    });
    // Auto-connect
    connect();
    return {
        on: on,
        off: off,
        send: send,
        getClientId: getClientId,
        isConnected: isConnected
    };
})();
//# sourceMappingURL=ws.js.map