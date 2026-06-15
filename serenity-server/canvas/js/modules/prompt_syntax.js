/* prompt_syntax.js — module 'promptSyntax'. SwarmUI-style client-side prompt-syntax
   engine + a resolver invoked BEFORE every submit, plus a wildcard manager UI and a
   prompt autocomplete dropdown. GAP section 5.

   What it resolves (all deterministic per the job seed where randomness applies):
     <wildcard:name>           pick a line from an in-memory wildcard store (seeded);
                               <wildcard:name[2]> picks N distinct lines joined by ", ".
     <random:a|b|c>            uniform seeded pick of one option (nested ok, outer-first).
     <random:1-5>              numeric range pick (integer lo..hi inclusive, seeded).
     <random[2]:a|b|c>         pick N distinct options, joined by ", ".
     <lora:name:w>             extracted -> state.params.loras (REMOVED from text).
                               <lora:name:wModel:wClip> splits unet/tenc weights.
     <embed:file>              textual-inversion ref -> rewritten to embedding:file
                               (a1111-style) so the encoder can pick it up; passed through.
     <setvar[name]:value> / <var:name>      define + substitute a variable.
     <setmacro[name]:body> / <macro:name>   define a macro (re-resolved on use).
     <comment:...>             stripped entirely (a note for the human).
     (word:1.2)                attention weighting — VALIDATED + PASSED THROUGH verbatim
                               (worker doesn't consume weights yet; never rewritten).

   HOW it runs at submit: we WRAP Serenity.api.submitPrompt from inside init() (we only
   add code in our own module; we do not edit api.js). On each submit we read
   state.params.prompt + .negative, resolve them against the concrete seed, write the
   resolved text back into state.params.{prompt,negative}, stash the originals in
   state.params.{prompt_raw,negative_raw}, MERGE extracted LoRAs into state.params.loras
   (UI stack wins on name conflict), then defer to the original submitPrompt. After the
   POST resolves we RESTORE the raw prompt into state.params so the editor keeps showing
   what the user typed (and the next run re-resolves with a fresh seed).

   Malformed syntax is NEVER fatal: the broken span passes through verbatim and a
   human-readable note is surfaced via bus 'promptSyntax:notes' (and console).

   Ported from MojoUI/mojoui/app/prompt_syntax.mojo (parity-tested there) and extended
   to the full SwarmUI set above. Self-contained; talks via ctx.{state,get,set,bus,api}.
*/
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  // daemon LoRA weight bounds (serenity_daemon parse_generate lora.weight) — clamp+note
  var LORA_MIN = -10.0, LORA_MAX = 10.0;
  var MAX_PASSES = 32; // nesting cap for var/macro/wildcard/random expansion

  // ---------------------------------------------------------------------------
  // deterministic per-seed RNG (splitmix64, matches the Mojo reference)
  // JS has no 64-bit ints; emulate with BigInt for bit-parity with the daemon side.
  // ---------------------------------------------------------------------------
  var M64 = (1n << 64n) - 1n;
  function mix64(stateRef) {
    var s = (stateRef.v + 0x9E3779B97F4A7C15n) & M64;
    stateRef.v = s;
    var z = s;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M64;
    z = (z ^ (z >> 31n)) & M64;
    return z;
  }
  function rngPick(stateRef, n) {
    if (n <= 0) return 0;
    return Number(mix64(stateRef) % BigInt(n));
  }
  function makeRng(seed) {
    var s = (seed >= 0) ? seed : -seed;
    return { v: BigInt(Math.floor(s)) & M64 };
  }

  // ---------------------------------------------------------------------------
  // wildcard store (in-memory). { name -> [line, line, ...] }
  // Persisted to localStorage so the manager survives reloads (no backend needed).
  // ---------------------------------------------------------------------------
  var LS_KEY = "serenity.wildcards";
  var wildcards = {};
  function loadWildcards() {
    try {
      var raw = localStorage.getItem(LS_KEY);
      if (raw) {
        var obj = JSON.parse(raw);
        if (obj && typeof obj === "object") wildcards = obj;
      }
    } catch (_) {}
    // seed a couple of helpful defaults the FIRST time only
    if (!Object.keys(wildcards).length) {
      wildcards = {
        season: ["spring", "summer", "autumn", "winter"],
        timeofday: ["dawn", "morning", "noon", "golden hour", "dusk", "midnight"],
      };
      saveWildcards();
    }
  }
  function saveWildcards() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(wildcards)); } catch (_) {}
  }
  function parseLines(text) {
    return String(text || "").split(/\r?\n/).map(function (l) { return l.trim(); })
      .filter(function (l) { return l.length && l[0] !== "#"; });
  }

  // ---------------------------------------------------------------------------
  // parse result
  // ---------------------------------------------------------------------------
  function newResult() {
    return { resolved: "", loras: [], notes: [], hadSyntax: false };
  }

  // strict float: [+-]?digits[.digits]; ok=false on junk. (mirrors Mojo _parse_float)
  function parseFloatStrict(s) {
    var t = String(s == null ? "" : s).trim();
    if (!/^[+-]?(\d+(\.\d*)?|\.\d+)$/.test(t)) return { ok: false, v: 0 };
    var v = parseFloat(t);
    return isNaN(v) ? { ok: false, v: 0 } : { ok: true, v: v };
  }
  function parseIntStrict(s) {
    var t = String(s == null ? "" : s).trim();
    if (!/^[+-]?\d+$/.test(t)) return { ok: false, v: 0 };
    return { ok: true, v: parseInt(t, 10) };
  }

  // find matching '>' for a tag opened at i (depth-counts nested <...>). -1 if none.
  function matchClose(text, bodyLo) {
    var depth = 1;
    for (var j = bodyLo; j < text.length; j++) {
      var c = text[j];
      if (c === "<") depth++;
      else if (c === ">") { depth--; if (depth === 0) return j; }
    }
    return -1;
  }

  // split body on TOP-LEVEL '|' (nested <...> keep their own '|')
  function splitTopLevel(body, sep) {
    var out = [], d = 0, start = 0;
    for (var k = 0; k < body.length; k++) {
      var c = body[k];
      if (c === "<") d++;
      else if (c === ">") d--;
      else if (c === sep && d === 0) { out.push(body.slice(start, k)); start = k + 1; }
    }
    out.push(body.slice(start));
    return out;
  }

  // ---- generic tag opener: returns {prefix,count} for "<random[2]:" style heads ----
  // matches  name  or  name[N]  at the start, returns null if not this tag.
  function readTagHead(text, i, tagName) {
    // text[i..] should start with "<" + tagName
    var head = "<" + tagName;
    if (text.substr(i, head.length) !== head) return null;
    var p = i + head.length;
    var count = 1, namedKey = null;
    // optional [N] count  OR  [name] key (for setvar/setmacro)
    if (text[p] === "[") {
      var rb = text.indexOf("]", p);
      if (rb < 0) return null;
      var inside = text.slice(p + 1, rb);
      var ci = parseIntStrict(inside);
      if (ci.ok) count = Math.max(1, ci.v);
      else namedKey = inside.trim();
      p = rb + 1;
    }
    if (text[p] !== ":") return null; // must be "<tag...:"
    return { bodyLo: p + 1, count: count, namedKey: namedKey };
  }

  // ---------------------------------------------------------------------------
  // pass: <comment:...>  -> stripped
  // ---------------------------------------------------------------------------
  function passComment(text, notes, flag) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text.substr(i, 9) === "<comment:") {
        var close = matchClose(text, i + 9);
        if (close < 0) { notes.push("unterminated <comment:...> passed through"); out += text.slice(i); break; }
        i = close + 1; flag.changed = true;
        i = collapseSeam(out, text, i);
        continue;
      }
      out += text[i++];
    }
    return out;
  }

  // After removing a span (comment/var/macro/setvar), avoid leaving a double space
  // or a leading space: if `out` ends in a space (or is empty), swallow leading
  // spaces from the remaining text. Returns the (possibly advanced) index.
  function collapseSeam(out, text, i) {
    if (out.length === 0 || out[out.length - 1] === " ") {
      while (i < text.length && text[i] === " ") i++;
    }
    return i;
  }

  // ---------------------------------------------------------------------------
  // pass: <setvar[name]:value> define, then <var:name> substitute.
  //       <setmacro[name]:body> define, then <macro:name> expand (re-resolved).
  // vars/macros live in `env`; one pass each, repeated by the outer loop.
  // ---------------------------------------------------------------------------
  function passDefineAndUse(text, env, notes, flag) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text[i] !== "<") { out += text[i++]; continue; }

      // ----- <setvar[name]:value>  /  <setmacro[name]:body> -----
      var isVar = text.substr(i, 8) === "<setvar[";
      var isMac = text.substr(i, 10) === "<setmacro[";
      if (isVar || isMac) {
        var tag = isVar ? "setvar" : "setmacro";
        var head = readTagHead(text, i, tag);
        if (head && head.namedKey) {
          var close = matchClose(text, head.bodyLo);
          if (close < 0) { notes.push("unterminated <" + tag + "[...]> passed through"); out += text.slice(i); break; }
          var value = text.slice(head.bodyLo, close);
          if (isVar) env.vars[head.namedKey] = value;
          else env.macros[head.namedKey] = value;
          i = close + 1; flag.changed = true;
          // collapse a leading/dangling double space left by the directive
          i = collapseSeam(out, text, i);
          continue;
        }
      }

      // ----- <var:name>  substitute -----
      if (text.substr(i, 5) === "<var:") {
        var vc = matchClose(text, i + 5);
        if (vc >= 0) {
          var vname = text.slice(i + 5, vc).trim();
          if (Object.prototype.hasOwnProperty.call(env.vars, vname)) {
            out += env.vars[vname]; i = vc + 1; flag.changed = true; continue;
          }
          notes.push("undefined <var:" + vname + "> passed through");
        }
      }
      // ----- <macro:name>  expand -----
      if (text.substr(i, 7) === "<macro:") {
        var mc = matchClose(text, i + 7);
        if (mc >= 0) {
          var mname = text.slice(i + 7, mc).trim();
          if (Object.prototype.hasOwnProperty.call(env.macros, mname)) {
            out += env.macros[mname]; i = mc + 1; flag.changed = true; continue;
          }
          notes.push("undefined <macro:" + mname + "> passed through");
        }
      }
      out += text[i++];
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // pass: <wildcard:name>  /  <wildcard[N]:name>  — seeded pick(s) from the store
  // ---------------------------------------------------------------------------
  function passWildcards(text, rng, notes, flag) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text[i] !== "<" || text.substr(i, 10) !== "<wildcard:" && text.substr(i, 10) !== "<wildcard[") {
        out += text[i++]; continue;
      }
      var head = readTagHead(text, i, "wildcard");
      if (!head) { out += text[i++]; continue; }
      var close = matchClose(text, head.bodyLo);
      if (close < 0) { notes.push("unterminated <wildcard:...> passed through"); out += text.slice(i); break; }
      var name = text.slice(head.bodyLo, close).trim();
      var lines = wildcards[name];
      if (!lines || !lines.length) {
        notes.push("unknown wildcard '" + name + "' passed through");
        out += text.slice(i, close + 1); i = close + 1; continue;
      }
      var picks = pickDistinct(lines, head.count, rng);
      out += picks.join(", ");
      flag.changed = true;
      i = close + 1;
    }
    return out;
  }

  function pickDistinct(arr, count, rng) {
    var n = arr.length;
    count = Math.min(count, n);
    if (count <= 1) return [arr[rngPick(rng, n)]];
    // seeded partial Fisher-Yates over a copy of indices
    var idx = arr.map(function (_, k) { return k; });
    for (var k = 0; k < count; k++) {
      var j = k + rngPick(rng, n - k);
      var tmp = idx[k]; idx[k] = idx[j]; idx[j] = tmp;
    }
    var out = [];
    for (var m = 0; m < count; m++) out.push(arr[idx[m]]);
    return out;
  }

  // ---------------------------------------------------------------------------
  // pass: <random:...>  — one outer-first, left-to-right pass.
  //   <random:a|b|c>      pick one
  //   <random[N]:a|b|c>   pick N distinct (joined ", ")
  //   <random:1-5>        numeric range (a single "lo-hi" option, integer inclusive)
  // ---------------------------------------------------------------------------
  function passRandom(text, rng, notes, flag) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text[i] !== "<" || (text.substr(i, 8) !== "<random:" && text.substr(i, 8) !== "<random[")) {
        out += text[i++]; continue;
      }
      var head = readTagHead(text, i, "random");
      if (!head) { out += text[i++]; continue; }
      var close = matchClose(text, head.bodyLo);
      if (close < 0) { notes.push("unterminated <random:...> passed through"); out += text.slice(i); break; }
      if (close === head.bodyLo) { notes.push("empty <random:> passed through"); out += text.slice(i, close + 1); i = close + 1; continue; }
      var body = text.slice(head.bodyLo, close);

      // numeric range "lo-hi" (only if there's no top-level '|' and it matches)
      var options = splitTopLevel(body, "|");
      if (options.length === 1) {
        var rng2 = /^\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*$/.exec(body);
        if (rng2) {
          var lo = parseInt(rng2[1], 10), hi = parseInt(rng2[2], 10);
          if (lo > hi) { var t = lo; lo = hi; hi = t; }
          var span = hi - lo + 1;
          out += String(lo + rngPick(rng, span));
          flag.changed = true; i = close + 1; continue;
        }
      }
      var picks = pickDistinct(options, head.count, rng).map(function (s) { return s; });
      out += picks.join(", ");
      flag.changed = true;
      i = close + 1;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // pass: <embed:file>  -> a1111-style "embedding:file" token, passed through to encoder
  // ---------------------------------------------------------------------------
  function passEmbeds(text, notes, flag) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text.substr(i, 7) !== "<embed:") { out += text[i++]; continue; }
      var close = matchClose(text, i + 7);
      if (close < 0) { notes.push("unterminated <embed:...> passed through"); out += text.slice(i); break; }
      var name = text.slice(i + 7, close).trim();
      if (!name) { notes.push("empty <embed:> passed through"); out += text.slice(i, close + 1); i = close + 1; continue; }
      out += "embedding:" + name;
      flag.changed = true;
      i = close + 1;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // pass: <lora:name[:w[:wClip]]>  — extracted -> loras[], removed from text.
  //   single weight => strength (both unet+tenc). two weights => model/clip split.
  // ---------------------------------------------------------------------------
  function extractLoras(text, loras, notes) {
    var out = "", i = 0;
    while (i < text.length) {
      if (text.substr(i, 6) !== "<lora:") { out += text[i++]; continue; }
      // find '>' but treat a '<' before it as malformed (mirrors Mojo)
      var close = -1, nested = false;
      for (var j = i + 6; j < text.length; j++) {
        if (text[j] === ">") { close = j; break; }
        if (text[j] === "<") { nested = true; break; }
      }
      if (close < 0 || nested) {
        notes.push("malformed <lora:...> passed through verbatim");
        out += text[i++]; continue;
      }
      var body = text.slice(i + 6, close);
      var parts = body.split(":");
      var name = parts[0].trim();
      var wModel = 1.0, wClip = null;
      var bad = false;
      if (parts.length >= 2) {
        var w1 = parseFloatStrict(parts[1]);
        if (!w1.ok) bad = true; else wModel = w1.v;
      }
      if (parts.length >= 3) {
        var w2 = parseFloatStrict(parts[2]);
        if (!w2.ok) bad = true; else wClip = w2.v;
      }
      if (parts.length > 3) bad = true;
      if (!name || bad) {
        notes.push("malformed <lora:" + body + "> passed through verbatim");
        out += text.slice(i, close + 1); i = close + 1; continue;
      }
      wModel = clampLora(wModel, name, notes);
      var entry = { name: name, strength: wModel, strength_model: wModel,
                    strength_clip: (wClip == null ? wModel : clampLora(wClip, name, notes)),
                    enabled: true, fromPrompt: true };
      loras.push(entry);
      // collapse the seam so removal doesn't leave a double space
      i = close + 1;
      if (out.length && out[out.length - 1] === " ") {
        while (i < text.length && text[i] === " ") i++;
      } else if (!out.length) {
        while (i < text.length && text[i] === " ") i++;
      }
    }
    return out;
  }
  function clampLora(w, name, notes) {
    if (w < LORA_MIN) { notes.push("<lora:" + name + ":" + w + "> clamped to " + LORA_MIN + " (daemon range [" + LORA_MIN + "," + LORA_MAX + "])"); return LORA_MIN; }
    if (w > LORA_MAX) { notes.push("<lora:" + name + ":" + w + "> clamped to " + LORA_MAX + " (daemon range [" + LORA_MIN + "," + LORA_MAX + "])"); return LORA_MAX; }
    return w;
  }

  // ---------------------------------------------------------------------------
  // (word:1.2) — validate paren balance + numeric weight tails. Never rewrites.
  // ---------------------------------------------------------------------------
  function validateWeights(text, notes) {
    var depth = 0, gstart = [], gcolon = [];
    for (var i = 0; i < text.length; i++) {
      var c = text[i];
      if (c === "(") { depth++; gstart.push(i); gcolon.push(-1); }
      else if (c === ":" && depth > 0) { gcolon[gcolon.length - 1] = i; }
      else if (c === ")") {
        if (depth === 0) { notes.push("unbalanced ')' in prompt — weights passed through"); return; }
        depth--; var gs = gstart.pop(); var gc = gcolon.pop();
        if (gc >= 0) {
          var tail = text.slice(gc + 1, i);
          if (!parseFloatStrict(tail).ok) {
            notes.push("weight tag '" + text.slice(gs, i + 1) + "' not numeric — passed through");
          }
        }
      }
    }
    if (depth !== 0) notes.push("unbalanced '(' in prompt — weights passed through");
  }

  // ---------------------------------------------------------------------------
  // PUBLIC: resolve a prompt against a concrete seed. Never throws.
  // ---------------------------------------------------------------------------
  function resolvePrompt(prompt, seed) {
    var r = newResult();
    var rng = makeRng(seed);
    var env = { vars: {}, macros: {} };
    var text = String(prompt == null ? "" : prompt);

    // outer loop: comment -> vars/macros -> wildcards -> random, until stable
    var anyExpand = false;
    for (var pass = 0; pass < MAX_PASSES; pass++) {
      var flag = { changed: false };
      text = passComment(text, r.notes, flag);
      text = passDefineAndUse(text, env, r.notes, flag);
      text = passWildcards(text, rng, r.notes, flag);
      text = passRandom(text, rng, r.notes, flag);
      if (!flag.changed) break;
      anyExpand = true;
    }
    // embeds (one shot; output token has no '<' so no re-expansion needed)
    var embFlag = { changed: false };
    text = passEmbeds(text, r.notes, embFlag);

    var beforeLoras = r.loras.length;
    text = extractLoras(text, r.loras, r.notes);
    validateWeights(text, r.notes);

    r.hadSyntax = anyExpand || embFlag.changed || (r.loras.length > beforeLoras);
    r.resolved = text;
    return r;
  }

  // ---------------------------------------------------------------------------
  // submit-time resolver: wraps Serenity.api.submitPrompt (in our module only).
  // ---------------------------------------------------------------------------
  function installSubmitResolver(ctx) {
    var api = ctx.api, get = ctx.get, set = ctx.set, bus = ctx.bus;
    if (!api || typeof api.submitPrompt !== "function" || api.__promptSyntaxWrapped) return;
    var original = api.submitPrompt.bind(api);
    api.__promptSyntaxWrapped = true;

    api.submitPrompt = function (graph, clientId) {
      var rawPos = get("params.prompt") || "";
      var rawNeg = get("params.negative") || "";
      // concrete seed: -1 means "random per run"; pick a real one so resolution is
      // deterministic w.r.t. what we send, and write it back so the worker uses it too.
      var seed = Number(get("params.seed"));
      if (!Number.isFinite(seed) || seed < 0) {
        seed = Math.floor(Math.random() * 0xffffffff);
      }

      var rPos = resolvePrompt(rawPos, seed);
      var rNeg = resolvePrompt(rawNeg, seed ^ 0x5bd1e995);

      var notes = rPos.notes.concat(rNeg.notes);
      if (notes.length) {
        console.info("[promptSyntax] notes:", notes.join("; "));
        bus.emit("promptSyntax:notes", notes);
      }

      // merge prompt LoRAs into the UI stack; UI stack WINS on name conflict.
      var promptLoras = rPos.loras.concat(rNeg.loras);
      if (promptLoras.length) {
        var existing = Array.isArray(get("params.loras")) ? get("params.loras").slice() : [];
        var byName = {};
        existing.forEach(function (l) { if (l && l.name) byName[l.name] = true; });
        promptLoras.forEach(function (l) { if (!byName[l.name]) { existing.push(l); byName[l.name] = true; } });
        set("params.loras", existing);
      }

      // stash raw + write resolved so api.generateBody picks up the resolved text.
      set("params.prompt_raw", rawPos);
      set("params.negative_raw", rawNeg);
      if (rPos.hadSyntax || rPos.resolved !== rawPos) set("params.prompt", rPos.resolved);
      if (rNeg.hadSyntax || rNeg.resolved !== rawNeg) set("params.negative", rNeg.resolved);
      // pin the concrete seed we resolved against (so wildcards/randoms match the gen)
      set("params.seed", seed);

      bus.emit("promptSyntax:resolved", {
        seed: seed, prompt: rPos.resolved, negative: rNeg.resolved,
        loras: promptLoras, notes: notes,
      });

      var restore = function () {
        // put the human-typed prompt back so the editor isn't clobbered; the next
        // run re-resolves with a fresh seed. (LoRAs stay merged — that matches Swarm.)
        set("params.prompt", rawPos);
        set("params.negative", rawNeg);
      };
      var p;
      try { p = Promise.resolve(original(graph, clientId)); }
      catch (e) { restore(); throw e; }
      return p.then(function (res) { restore(); return res; },
                    function (err) { restore(); throw err; });
    };
    console.info("[promptSyntax] submit resolver installed");
  }

  // ===========================================================================
  // UI: wildcard manager (modal) + a small <…> autocomplete dropdown on the prompt
  // ===========================================================================
  function injectCSS() {
    if (document.getElementById("style-promptSyntax")) return;
    var css = [
      // autocomplete dropdown
      ".ps-ac{position:absolute;z-index:120;min-width:180px;max-width:340px;max-height:240px;overflow:auto;",
      "  background:var(--panel2);border:1px solid var(--line);border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.45);padding:4px;font:12px/1.3 system-ui,sans-serif}",
      ".ps-ac .ps-ac-item{padding:6px 9px;border-radius:5px;cursor:pointer;color:var(--text);display:flex;justify-content:space-between;gap:10px}",
      ".ps-ac .ps-ac-item .ps-ac-hint{color:var(--muted);font-size:11px}",
      ".ps-ac .ps-ac-item.sel,.ps-ac .ps-ac-item:hover{background:var(--accent2);color:#fff}",
      ".ps-ac .ps-ac-item.sel .ps-ac-hint,.ps-ac .ps-ac-item:hover .ps-ac-hint{color:#dfe3ff}",
      // manager modal
      "#ps-modal{position:fixed;inset:0;z-index:200;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.55)}",
      "#ps-modal.show{display:flex}",
      "#ps-modal .ps-card{width:min(720px,92vw);max-height:86vh;display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.5)}",
      "#ps-modal .ps-head{display:flex;align-items:center;gap:10px;padding:12px 14px;border-bottom:1px solid var(--line)}",
      "#ps-modal .ps-head h3{margin:0;font-size:14px;flex:1}",
      "#ps-modal .ps-body{display:flex;gap:12px;padding:12px 14px;min-height:0;flex:1}",
      "#ps-modal .ps-list{width:200px;border-right:1px solid var(--line);padding-right:10px;overflow:auto}",
      "#ps-modal .ps-list .ps-wc{padding:6px 8px;border-radius:6px;cursor:pointer;color:var(--text);display:flex;justify-content:space-between;gap:8px;font-size:12px}",
      "#ps-modal .ps-list .ps-wc .ps-wc-n{color:var(--muted);font-variant-numeric:tabular-nums}",
      "#ps-modal .ps-list .ps-wc.sel{background:var(--accent2);color:#fff}",
      "#ps-modal .ps-edit{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0}",
      "#ps-modal .ps-edit textarea{flex:1;min-height:220px;resize:vertical;font:12px/1.45 ui-monospace,monospace}",
      "#ps-modal .ps-row{display:flex;gap:8px;align-items:center}",
      "#ps-modal .ps-row input{flex:1}",
      "#ps-modal .ps-foot{padding:10px 14px;border-top:1px solid var(--line);display:flex;gap:8px;align-items:center}",
      "#ps-modal .ps-foot .ps-sp{flex:1}",
      "#ps-modal .ps-hint{color:var(--muted);font-size:11px}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-promptSyntax"; st.textContent = css;
    document.head.appendChild(st);
  }

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k.slice(0, 2) === "on" && typeof attrs[k] === "function") n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    }
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
      if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  // ---- autocomplete: triggers after '<' (directive helper) and on a tag CSV list ----
  // Built-in directive list + dynamic wildcard names + (when available) api.loras()/embeddings().
  function buildAutocomplete(ctx) {
    var get = ctx.get, set = ctx.set, api = ctx.api;
    var DIRECTIVES = [
      { v: "wildcard:", h: "pick a stored line" },
      { v: "random:", h: "a|b|c random pick" },
      { v: "lora:", h: "inline LoRA name:weight" },
      { v: "embed:", h: "textual inversion" },
      { v: "setvar[name]:", h: "define a variable" },
      { v: "var:", h: "use a variable" },
      { v: "setmacro[name]:", h: "define a macro" },
      { v: "macro:", h: "expand a macro" },
      { v: "comment:", h: "stripped at submit" },
    ];
    var loraNames = [], embedNames = [];
    if (api && typeof api.loras === "function") {
      Promise.resolve().then(function () { return api.loras(); })
        .then(function (l) { loraNames = normNames(l); }).catch(function () {});
    }
    if (api && typeof api.embeddings === "function") {
      Promise.resolve().then(function () { return api.embeddings(); })
        .then(function (l) { embedNames = normNames(l); }).catch(function () {});
    }

    var pop = el("div", { class: "ps-ac" });
    pop.style.display = "none";
    document.body.appendChild(pop);
    var state = { ta: null, items: [], sel: 0, tokenStart: -1, mode: "dir" };

    function normNames(list) {
      if (!list) return [];
      if (Array.isArray(list)) return list.map(function (x) { return typeof x === "string" ? x : (x && (x.name || x.filename || x.title)); }).filter(Boolean);
      if (typeof list === "object") {
        var a = list.loras || list.embeddings || list.items || list.data;
        if (Array.isArray(a)) return normNames(a);
      }
      return [];
    }

    function hide() { pop.style.display = "none"; state.ta = null; state.items = []; state.tokenStart = -1; }

    function render(items, anchorRect) {
      pop.innerHTML = "";
      if (!items.length) { hide(); return; }
      items.forEach(function (it, idx) {
        var row = el("div", { class: "ps-ac-item" + (idx === state.sel ? " sel" : "") }, [
          el("span", { text: it.label }),
          it.hint ? el("span", { class: "ps-ac-hint", text: it.hint }) : null,
        ]);
        row.addEventListener("mousedown", function (e) { e.preventDefault(); accept(idx); });
        pop.appendChild(row);
      });
      pop.style.left = anchorRect.left + "px";
      pop.style.top = (anchorRect.top) + "px";
      pop.style.display = "block";
    }

    // figure out the token being typed and which completion list applies
    function compute(ta) {
      var pos = ta.selectionStart;
      var text = ta.value.slice(0, pos);
      var lt = text.lastIndexOf("<");
      if (lt < 0) return null;
      // bail if the '<' is already closed before the cursor
      if (text.indexOf(">", lt) >= 0) return null;
      var frag = text.slice(lt + 1); // after '<'
      // directive head not yet completed (no ':') => suggest directives
      var colon = frag.indexOf(":");
      if (colon < 0) {
        var q = frag.toLowerCase();
        var items = DIRECTIVES.filter(function (d) { return d.v.toLowerCase().indexOf(q) === 0; })
          .map(function (d) { return { label: d.v, hint: d.h, insert: d.v, replaceFrom: lt + 1 }; });
        return { mode: "dir", items: items, tokenStart: lt + 1 };
      }
      // after the colon — context list for wildcard / lora / embed / var / macro
      var head = frag.slice(0, colon).replace(/\[.*?\]/, "");
      var arg = frag.slice(colon + 1);
      var argStart = lt + 1 + colon + 1;
      var pool = [];
      if (head === "wildcard") pool = Object.keys(wildcards).map(function (n) { return { label: n, hint: (wildcards[n] || []).length + " lines" }; });
      else if (head === "lora") pool = loraNames.map(function (n) { return { label: n }; });
      else if (head === "embed") pool = embedNames.map(function (n) { return { label: n }; });
      else return null;
      var qa = arg.toLowerCase();
      var items2 = pool.filter(function (it) { return it.label.toLowerCase().indexOf(qa) >= 0; }).slice(0, 40)
        .map(function (it) { return { label: it.label, hint: it.hint, insert: it.label, replaceFrom: argStart, closeTag: true }; });
      return { mode: "arg", items: items2, tokenStart: argStart };
    }

    function accept(idx) {
      var ta = state.ta; if (!ta) return;
      var it = state.items[idx]; if (!it) return;
      var pos = ta.selectionStart;
      var before = ta.value.slice(0, it.replaceFrom);
      var after = ta.value.slice(pos);
      var ins = it.insert;
      if (it.closeTag && after[0] !== ">") ins += ">";
      ta.value = before + ins + after;
      var caret = (before + ins).length;
      ta.setSelectionRange(caret, caret);
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.focus();
      // chain: after choosing a directive head, immediately recompute (arg list)
      setTimeout(function () { onType(ta); }, 0);
    }

    function caretRect(ta) {
      // approximate: anchor under the textarea's bottom-left (good enough, no mirror div)
      var r = ta.getBoundingClientRect();
      return { left: r.left, top: r.bottom + 2 };
    }

    function onType(ta) {
      var res = compute(ta);
      if (!res || !res.items.length) { hide(); return; }
      state.ta = ta; state.items = res.items; state.tokenStart = res.tokenStart; state.mode = res.mode;
      if (state.sel >= res.items.length) state.sel = 0;
      render(res.items, caretRect(ta));
    }

    function onKey(e) {
      if (pop.style.display === "none") return;
      if (e.key === "ArrowDown") { e.preventDefault(); state.sel = (state.sel + 1) % state.items.length; render(state.items, caretRect(state.ta)); }
      else if (e.key === "ArrowUp") { e.preventDefault(); state.sel = (state.sel - 1 + state.items.length) % state.items.length; render(state.items, caretRect(state.ta)); }
      else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); accept(state.sel); }
      else if (e.key === "Escape") { hide(); }
    }

    // attach to any prompt/negative textarea; rescan on DOM changes (prompt bar may
    // mount after us). We bind by capturing input on the document, scoped to textareas
    // that live in the prompt bar or param rail.
    function isPromptField(t) {
      if (!t || t.tagName !== "TEXTAREA") return false;
      return !!(t.closest("#prompt-bar") || t.closest("#param-rail"));
    }
    document.addEventListener("input", function (e) { if (isPromptField(e.target)) onType(e.target); }, true);
    document.addEventListener("keydown", function (e) { if (isPromptField(e.target)) onKey(e); }, true);
    document.addEventListener("click", function (e) { if (e.target !== pop && !pop.contains(e.target)) hide(); });
    return { refreshLoras: function () { if (api && api.loras) Promise.resolve(api.loras()).then(function (l) { loraNames = normNames(l); }).catch(function () {}); } };
  }

  // ---- wildcard manager modal ----
  function buildManager(ctx) {
    var modal = el("div", { id: "ps-modal" });
    var listEl, nameIn, linesTa, curName = null;

    function refreshList() {
      listEl.innerHTML = "";
      var names = Object.keys(wildcards).sort();
      names.forEach(function (n) {
        var row = el("div", { class: "ps-wc" + (n === curName ? " sel" : "") }, [
          el("span", { text: n }),
          el("span", { class: "ps-wc-n", text: String((wildcards[n] || []).length) }),
        ]);
        row.addEventListener("click", function () { selectWC(n); });
        listEl.appendChild(row);
      });
    }
    function selectWC(n) {
      curName = n; nameIn.value = n; linesTa.value = (wildcards[n] || []).join("\n"); refreshList();
    }
    function saveCurrent() {
      var nm = nameIn.value.trim();
      if (!nm) return;
      var lines = parseLines(linesTa.value);
      // rename support: if curName changed, drop the old key
      if (curName && curName !== nm) delete wildcards[curName];
      wildcards[nm] = lines; curName = nm; saveWildcards(); refreshList();
    }
    function newWC() { curName = null; nameIn.value = ""; linesTa.value = ""; refreshList(); nameIn.focus(); }
    function delWC() { if (curName && wildcards[curName]) { delete wildcards[curName]; saveWildcards(); newWC(); } }

    listEl = el("div", { class: "ps-list" });
    nameIn = el("input", { type: "text", placeholder: "wildcard name (use as <wildcard:name>)" });
    linesTa = el("textarea", { placeholder: "one option per line\n# lines starting with # are comments" });
    var card = el("div", { class: "ps-card" }, [
      el("div", { class: "ps-head" }, [
        el("h3", { text: "Wildcards" }),
        el("button", { class: "btn", type: "button", text: "+ New", onclick: newWC }),
        el("button", { class: "btn", type: "button", text: "✕", title: "Close",
          onclick: function () { modal.classList.remove("show"); } }),
      ]),
      el("div", { class: "ps-body" }, [
        listEl,
        el("div", { class: "ps-edit" }, [
          el("div", { class: "ps-row" }, [nameIn]),
          linesTa,
          el("div", { class: "ps-hint", text: "Used in prompts as <wildcard:name> (or <wildcard[2]:name> for 2 distinct picks). Picks are seeded by the job seed." }),
        ]),
      ]),
      el("div", { class: "ps-foot" }, [
        el("button", { class: "btn", type: "button", text: "Import .txt",
          onclick: function () { importTxt(); } }),
        el("button", { class: "btn", type: "button", text: "Delete", onclick: delWC }),
        el("span", { class: "ps-sp" }),
        el("button", { class: "btn btn-primary", type: "button", text: "Save", onclick: saveCurrent }),
      ]),
    ]);
    modal.appendChild(card);
    modal.addEventListener("click", function (e) { if (e.target === modal) modal.classList.remove("show"); });
    document.body.appendChild(modal);

    function importTxt() {
      var inp = el("input", { type: "file", accept: ".txt,.csv,text/plain" });
      inp.addEventListener("change", function () {
        var f = inp.files && inp.files[0]; if (!f) return;
        var fr = new FileReader();
        fr.onload = function () {
          var base = f.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "_");
          wildcards[base] = parseLines(String(fr.result)); saveWildcards(); selectWC(base);
        };
        fr.readAsText(f);
      });
      inp.click();
    }

    refreshList();
    return {
      open: function () { modal.classList.add("show"); if (!curName) newWC(); },
    };
  }

  // ---------------------------------------------------------------------------
  S.register("promptSyntax", {
    init: function (ctx) {
      injectCSS();
      loadWildcards();
      installSubmitResolver(ctx);
      var ac = buildAutocomplete(ctx);
      var mgr = buildManager(ctx);

      // expose a tiny API other modules / the topbar can call to open the manager
      // and to resolve/preview a prompt without submitting.
      S.promptSyntax = {
        resolve: function (prompt, seed) { return resolvePrompt(prompt, seed == null ? 0 : seed); },
        openWildcardManager: function () { mgr.open(); },
        getWildcards: function () { return wildcards; },
        refreshLoras: ac.refreshLoras,
      };
      // allow opening via bus too (e.g. a topbar button) without coupling files
      ctx.bus.on("promptSyntax:openWildcards", function () { mgr.open(); });

      // keyboard shortcut: Ctrl/Cmd+Shift+W opens the wildcard manager
      document.addEventListener("keydown", function (e) {
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "W" || e.key === "w")) {
          e.preventDefault(); mgr.open();
        }
      });

      console.info("[promptSyntax] ready (resolver + wildcard manager + autocomplete)");
    },
  });
})();
