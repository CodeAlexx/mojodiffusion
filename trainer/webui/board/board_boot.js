/**
 * board_boot.js — namespaces SerenityBoard's frontend under the serenity web
 * trainer supervisor. The ported frontend calls "/api/..." and "ws(s)://host/ws/live"
 * verbatim; those paths collide with the supervisor's own "/api/runs" (training
 * launch). This shim rewrites, at the fetch + WebSocket layer, so the frontend
 * source stays a faithful 1:1 port:
 *   /api/<x>        -> /api/board/<x>
 *   .../ws/live     -> .../api/board/ws/live
 * Loaded synchronously in <head> before app.js/live.js run.
 */
(function () {
    'use strict';

    var API_PREFIX = '/api/board';
    var WS_PATH = '/api/board/ws/live';

    function rewriteHttp(u) {
        if (typeof u !== 'string') return u;
        if (u.indexOf('/api/') === 0) return API_PREFIX + u.slice(4); // "/api/runs" -> "/api/board/runs"
        return u;
    }

    function rewriteWs(u) {
        if (typeof u !== 'string') return u;
        return u.replace('/ws/live', WS_PATH);
    }

    var _fetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        if (typeof input === 'string') input = rewriteHttp(input);
        return _fetch(input, init);
    };

    var _WS = window.WebSocket;
    function BoardWS(url, protocols) {
        url = rewriteWs(url);
        return protocols === undefined ? new _WS(url) : new _WS(url, protocols);
    }
    BoardWS.prototype = _WS.prototype;
    BoardWS.CONNECTING = _WS.CONNECTING;
    BoardWS.OPEN = _WS.OPEN;
    BoardWS.CLOSING = _WS.CLOSING;
    BoardWS.CLOSED = _WS.CLOSED;
    window.WebSocket = BoardWS;
})();
