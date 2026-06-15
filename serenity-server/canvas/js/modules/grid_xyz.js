/* grid_xyz.js — module 'gridXyz'. SwarmUI-style X/Y/Z multi-axis grid (XYZ-plot).
   A self-contained launcher button (bottom-right) opens a panel where the user picks
   up to three axes (X required, Y/Z optional) + a value list per axis, then runs a
   parameter sweep via the PROVEN server endpoint POST /v1/grid. The base params for
   every cell come from the CURRENT Serenity.state.params (model/prompt/negative/
   width/height/steps/cfg/seed/sampler/scheduler + advanced knobs + loras), so the grid
   sweeps exactly what the Generate tab is configured for, overriding only the swept
   axes. The composited grid PNG (one page per Z value) is shown in a results viewer
   with per-page download + "open in canvas" (broadcasts result:ready so resultView/
   gallery pick it up).

   Axes (match grid.rs Axis::parse): seed, cfg, steps, sampler, scheduler, model,
   prompt, negative, width, height, resolution, lora.

   Value syntax (server-side normalize_values): a comma list, a `||` list (use this when
   values contain commas, e.g. prompts), numeric range `1,..,10` / `1,3,..,9` /
   `10,..,1`, and `SKIP:<v>` to drop an entry. Examples:
     cfg        →  3,5,7        or  3,..,8
     resolution →  1024x1024 || 768x1024 || 1024x768
     lora       →  none || styleA.safetensors:0.8 || styleB.safetensors
     prompt     →  a red car || a blue car

   Self-contained: registers window.Serenity.modules.gridXyz, talks ONLY through
   ctx.state/get/set/bus/api. Never edits shared files. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") { console.warn("[gridXyz] Serenity not ready"); return; }

  // ── axis catalog (kept in sync with grid.rs Axis::parse) ──────────────────
  // numeric:true axes take numbers / numeric ranges; the rest take strings.
  var AXES = [
    { v: "", label: "— none —", numeric: false, none: true },
    { v: "cfg", label: "CFG", numeric: true, ph: "3,5,7  or  3,..,8" },
    { v: "steps", label: "Steps", numeric: true, ph: "8,16,24  or  8,..,12" },
    { v: "seed", label: "Seed", numeric: true, ph: "1,2,3  or  100,..,104" },
    { v: "sampler", label: "Sampler", numeric: false, ph: "euler, dpmpp_2m, heun" },
    { v: "scheduler", label: "Scheduler", numeric: false, ph: "simple, karras, beta" },
    { v: "model", label: "Model", numeric: false, ph: "z-image || ideogram4" },
    { v: "resolution", label: "Resolution (WxH)", numeric: false, ph: "1024x1024 || 768x1024" },
    { v: "width", label: "Width", numeric: true, ph: "768,1024,1280" },
    { v: "height", label: "Height", numeric: true, ph: "768,1024,1280" },
    { v: "lora", label: "LoRA", numeric: false, ph: "none || styleA.safetensors:0.8" },
    { v: "prompt", label: "Prompt", numeric: false, ph: "a red car || a blue car" },
    { v: "negative", label: "Negative", numeric: false, ph: "blurry || lowres || (use || )" },
  ];

  function axisInfo(v) {
    for (var i = 0; i < AXES.length; i++) if (AXES[i].v === v) return AXES[i];
    return AXES[0];
  }

  // ── tiny DOM helper (mirrors the existing module style) ───────────────────
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k.slice(0, 2) === "on" && typeof attrs[k] === "function") n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    }
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  function injectCSS() {
    if (document.getElementById("style-gridXyz")) return;
    var css = [
      /* launcher button (bottom-right; sits ABOVE #queue-strip(bottom:138px) and
         gallery_pro's #gp-batchbar(bottom:96px) so the floating buttons don't overlap) */
      "#gx-launch{position:fixed;right:14px;bottom:188px;z-index:64;display:flex;align-items:center;gap:6px;",
      "  background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:9px;",
      "  padding:8px 12px;font-size:12px;font-weight:600;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.35)}",
      "#gx-launch:hover{border-color:var(--accent)}",
      /* modal overlay + panel */
      "#gx-overlay{position:fixed;inset:0;z-index:70;background:rgba(8,9,12,.55);display:none;",
      "  align-items:center;justify-content:center}",
      "#gx-overlay.show{display:flex}",
      "#gx-panel{width:min(880px,94vw);max-height:90vh;overflow:auto;background:var(--panel);",
      "  border:1px solid var(--line);border-radius:12px;box-shadow:0 18px 50px rgba(0,0,0,.5);padding:0}",
      "#gx-panel .gx-head{display:flex;align-items:center;gap:10px;padding:14px 16px;",
      "  border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}",
      "#gx-panel .gx-head h2{margin:0;font-size:15px;font-weight:700}",
      "#gx-panel .gx-head .gx-sp{flex:1}",
      "#gx-panel .gx-x{background:transparent;border:0;color:var(--muted);font-size:18px;cursor:pointer;padding:2px 8px;border-radius:6px}",
      "#gx-panel .gx-x:hover{background:var(--panel2);color:var(--text)}",
      "#gx-panel .gx-body{padding:14px 16px}",
      "#gx-panel .gx-axis{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin-bottom:10px;background:var(--panel2)}",
      "#gx-panel .gx-axis .gx-arow{display:flex;align-items:center;gap:10px;margin-bottom:8px}",
      "#gx-panel .gx-axis .gx-tag{font-weight:700;font-size:13px;width:18px;color:var(--accent)}",
      "#gx-panel .gx-axis select{min-width:170px}",
      "#gx-panel .gx-axis .gx-count{margin-left:auto;font-size:11px;color:var(--muted)}",
      "#gx-panel .gx-axis textarea{width:100%;min-height:44px;font-family:ui-monospace,Menlo,Consolas,monospace;",
      "  font-size:12px;line-height:1.4;resize:vertical;box-sizing:border-box}",
      "#gx-panel .gx-axis textarea:disabled{opacity:.4}",
      "#gx-panel .gx-syntax{font-size:11px;color:var(--muted);margin:6px 2px 12px;line-height:1.5}",
      "#gx-panel .gx-syntax code{background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:0 4px}",
      "#gx-panel .gx-base{font-size:11px;color:var(--muted);background:var(--panel2);border:1px solid var(--line);",
      "  border-radius:8px;padding:8px 10px;margin-bottom:12px;line-height:1.5}",
      "#gx-panel .gx-base b{color:var(--text);font-weight:600}",
      "#gx-panel .gx-foot{display:flex;align-items:center;gap:10px;padding:12px 16px;border-top:1px solid var(--line);",
      "  position:sticky;bottom:0;background:var(--panel)}",
      "#gx-panel .gx-foot .gx-msg{flex:1;font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
      "#gx-panel .gx-foot .gx-msg.err{color:var(--danger)}",
      "#gx-panel .gx-foot .gx-msg.ok{color:var(--ok)}",
      "#gx-run{padding:9px 18px;font-size:13px}",
      "#gx-run:disabled{opacity:.55;cursor:default}",
      /* results viewer */
      "#gx-panel .gx-results{margin-top:6px}",
      "#gx-panel .gx-results .gx-pages{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}",
      "#gx-panel .gx-results .gx-pagebtn{font-size:11px;padding:4px 10px}",
      "#gx-panel .gx-results .gx-pagebtn.active{background:var(--accent2);border-color:var(--accent);color:#fff}",
      "#gx-panel .gx-results .gx-imgwrap{border:1px solid var(--line);border-radius:8px;overflow:auto;background:#0d0e12;",
      "  max-height:54vh;text-align:center}",
      "#gx-panel .gx-results img.gx-img{max-width:100%;display:inline-block}",
      "#gx-panel .gx-results .gx-acts{display:flex;gap:8px;margin-top:8px}",
      "#gx-panel .gx-spin{display:inline-block;width:13px;height:13px;border:2px solid var(--line);",
      "  border-top-color:var(--accent);border-radius:50%;animation:gx-spin 0.8s linear infinite;vertical-align:-2px;margin-right:6px}",
      "@keyframes gx-spin{to{transform:rotate(360deg)}}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-gridXyz";
    st.textContent = css;
    document.head.appendChild(st);
  }

  S.register("gridXyz", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus, api = ctx.api, state = ctx.state;
      var apiBase = (api && api.base) || "";

      // ── launcher ──────────────────────────────────────────────────────────
      var launch = el("button", { id: "gx-launch", type: "button", title: "Open the X/Y/Z parameter-grid generator" },
        [el("span", { text: "▦" }), el("span", { text: "Grid (XYZ)" })]);
      document.body.appendChild(launch);

      // ── modal scaffold ─────────────────────────────────────────────────────
      var overlay = el("div", { id: "gx-overlay" });
      var panel = el("div", { id: "gx-panel" });
      overlay.appendChild(panel);
      document.body.appendChild(overlay);

      // header
      var closeBtn = el("button", { class: "gx-x", type: "button", title: "Close", text: "✕" });
      panel.appendChild(el("div", { class: "gx-head" }, [
        el("h2", { text: "▦  Parameter Grid (X / Y / Z)" }),
        el("div", { class: "gx-sp" }),
        closeBtn,
      ]));

      var body = el("div", { class: "gx-body" });
      panel.appendChild(body);

      // base-params summary (read live from state.params when the panel opens)
      var baseLine = el("div", { class: "gx-base" });
      body.appendChild(baseLine);

      // three axis rows
      var axisRows = {};
      ["x", "y", "z"].forEach(function (dim) { axisRows[dim] = buildAxisRow(dim); body.appendChild(axisRows[dim].root); });

      // syntax hint
      body.appendChild(el("div", { class: "gx-syntax", html:
        "Values: comma list, or <code>||</code> list (use when values contain commas — prompts). " +
        "Numeric ranges: <code>1,..,10</code> · <code>1,3,..,9</code> · <code>10,..,1</code>. " +
        "Drop one with <code>SKIP:val</code>. Max 16 per axis, 64 cells total." }));

      // results viewer (hidden until a run completes)
      var results = el("div", { class: "gx-results", style: "display:none" });
      var pagesBar = el("div", { class: "gx-pages" });
      var imgWrap = el("div", { class: "gx-imgwrap" });
      var gridImg = el("img", { class: "gx-img", alt: "grid result" });
      imgWrap.appendChild(gridImg);
      var dlBtn = el("button", { class: "btn gx-pagebtn", type: "button", text: "⬇ Download page" });
      var openBtn = el("button", { class: "btn gx-pagebtn", type: "button", text: "↗ Show in canvas" });
      var acts = el("div", { class: "gx-acts" }, [dlBtn, openBtn]);
      results.appendChild(pagesBar);
      results.appendChild(imgWrap);
      results.appendChild(acts);
      body.appendChild(results);

      // footer
      var msg = el("div", { class: "gx-msg", text: "Configure axes, then Run." });
      var runBtn = el("button", { id: "gx-run", class: "btn btn-primary", type: "button", text: "▶ Run grid" });
      panel.appendChild(el("div", { class: "gx-foot" }, [msg, runBtn]));

      // ── axis row factory ────────────────────────────────────────────────────
      function buildAxisRow(dim) {
        var sel = el("select");
        AXES.forEach(function (a) { sel.appendChild(el("option", { value: a.v, text: a.label })); });
        var count = el("span", { class: "gx-count", text: "" });
        var ta = el("textarea", { placeholder: "(pick an axis)", spellcheck: "false" });
        ta.disabled = true;

        function onAxisChange() {
          var info = axisInfo(sel.value);
          ta.disabled = info.none;
          ta.placeholder = info.none ? "(pick an axis)" : info.ph;
          updateCount();
          recomputeOptions();
        }
        function updateCount() {
          var info = axisInfo(sel.value);
          if (info.none) { count.textContent = ""; return; }
          var n = previewValues(ta.value, info).length;
          count.textContent = n ? (n + " value" + (n === 1 ? "" : "s")) : "—";
        }
        sel.addEventListener("change", onAxisChange);
        ta.addEventListener("input", function () { updateCount(); recomputeTotal(); });

        var tag = el("span", { class: "gx-tag", text: dim.toUpperCase() });
        var arow = el("div", { class: "gx-arow" }, [tag, el("span", { text: "axis:" }), sel, count]);
        var root = el("div", { class: "gx-axis" }, [arow, ta]);
        return { root: root, sel: sel, ta: ta, count: count, dim: dim, updateCount: updateCount };
      }

      // disallow picking the same axis on two dims (purely a UX guard; server also rejects)
      function recomputeOptions() {
        var chosen = ["x", "y", "z"].map(function (d) { return axisRows[d].sel.value; });
        ["x", "y", "z"].forEach(function (d) {
          var sel = axisRows[d].sel, mine = sel.value;
          Array.prototype.forEach.call(sel.options, function (opt) {
            var taken = opt.value && opt.value !== mine && chosen.indexOf(opt.value) >= 0;
            opt.disabled = !!taken;
          });
        });
        recomputeTotal();
      }

      function recomputeTotal() {
        var counts = ["x", "y", "z"].map(function (d) {
          var info = axisInfo(axisRows[d].sel.value);
          if (info.none) return null;
          return previewValues(axisRows[d].ta.value, info).length;
        }).filter(function (n) { return n != null; });
        if (!counts.length) { setMsg("Pick at least the X axis."); return; }
        var total = counts.reduce(function (a, b) { return a * (b || 0); }, 1);
        if (counts.some(function (n) { return n === 0; })) { setMsg("Some axis has no valid values."); return; }
        setMsg(total + " cell" + (total === 1 ? "" : "s") + " (" + counts.join(" × ") + ")" +
          (total > 64 ? " — over the 64-cell cap" : ""));
      }

      // CLIENT-SIDE value preview (best-effort mirror of server normalize_values, only to
      // show a count/validation; the SERVER is authoritative and does the real parse).
      function previewValues(text, info) {
        var s = (text || "").trim();
        if (!s) return [];
        var parts;
        if (s.indexOf("||") >= 0) parts = s.split("||");
        else parts = s.split(",");
        var out = [];
        var hasRange = s.indexOf("..") >= 0;
        if (info.numeric && s.indexOf("||") < 0 && hasRange) {
          // expand a single comma range "a,b,..,c" (preview only; cap at 16)
          out = expandNumericPreview(s);
        } else {
          parts.forEach(function (p) {
            p = p.trim();
            if (!p) return;
            if (p.indexOf("SKIP:") === 0) return;
            if (info.numeric && s.indexOf("||") < 0) {
              // plain numeric csv
              if (!isNaN(parseFloat(p))) out.push(p);
            } else if (info.numeric) {
              // "||" numeric entry (may itself be a range)
              if (p.indexOf("..") >= 0) out = out.concat(expandNumericPreview(p));
              else if (!isNaN(parseFloat(p))) out.push(p);
            } else {
              out.push(p);
            }
          });
        }
        return out.slice(0, 16);
      }
      function expandNumericPreview(spec) {
        var toks = spec.split(",").map(function (t) { return t.trim(); }).filter(Boolean);
        var out = [];
        for (var i = 0; i < toks.length; i++) {
          if (toks[i] === "..") {
            var start = parseFloat(out.length ? out[out.length - 1] : "0");
            var end = parseFloat(toks[i + 1]);
            if (isNaN(end)) { i++; continue; }
            var step = 1;
            if (out.length >= 2) {
              var d = parseFloat(out[out.length - 1]) - parseFloat(out[out.length - 2]);
              if (d !== 0) step = d;
            }
            if (end < start && step > 0) step = -1;
            var cur = start + step, guard = 0;
            while (((step > 0) ? cur <= end + 1e-9 : cur >= end - 1e-9) && guard < 20) {
              out.push(String(Math.round(cur * 1e6) / 1e6)); cur += step; guard++;
            }
            i++;
          } else if (!isNaN(parseFloat(toks[i]))) {
            out.push(toks[i]);
          }
        }
        return out;
      }

      // ── base-params summary, refreshed when opening ─────────────────────────
      function refreshBase() {
        var p = (state && state.params) || {};
        var model = (p.model && String(p.model).indexOf("—") < 0) ? p.model : "z-image";
        var loraTxt = (Array.isArray(p.loras) && p.loras.length)
          ? p.loras.map(function (l) { return (l.name || "") + (l.weight != null && l.weight !== 1 ? ":" + l.weight : ""); }).join(", ")
          : "none";
        baseLine.innerHTML =
          "Cells inherit the current Generate params, overridden by the swept axes:&nbsp; " +
          "<b>model</b> " + esc(model) + " · <b>" + (p.width || 1024) + "×" + (p.height || 1024) + "</b>" +
          " · <b>steps</b> " + (p.steps || 8) + " · <b>cfg</b> " + (p.cfg != null ? p.cfg : 1.5) +
          " · <b>seed</b> " + (p.seed != null ? p.seed : -1) +
          " · <b>sampler</b> " + esc(p.sampler || "euler") + "/" + esc(p.scheduler || "simple") +
          " · <b>lora</b> " + esc(loraTxt) +
          "<br><b>prompt:</b> " + esc(truncate(p.prompt || "(empty)", 120));
        ["x", "y", "z"].forEach(function (d) { axisRows[d].updateCount(); });
        recomputeOptions();
      }

      // ── build the /v1/grid request body from state.params + the axis rows ───
      function buildGridBody() {
        var p = (state && state.params) || {};
        var model = (p.model && String(p.model).indexOf("—") < 0) ? p.model : "z-image";
        var body = {
          // base generate fields (server's base_params reads these flat keys)
          model: model,
          prompt: p.prompt || "",
          negative: p.negative || "",
          width: p.width || 1024,
          height: p.height || 1024,
          steps: p.steps || 8,
          seed: p.seed != null ? p.seed : -1,
          cfg: p.cfg != null ? p.cfg : 1.5,
          sampler: p.sampler || "euler",
          scheduler: p.scheduler || "simple",
          // advanced knobs (honored-or-warned by the worker; lifted onto every cell)
          clip_skip: p.clip_skip || 0,
          eta: p.eta != null ? p.eta : -1,
          sigma_min: p.sigma_min != null ? p.sigma_min : -1,
          sigma_max: p.sigma_max != null ? p.sigma_max : -1,
          restart_sampling: !!p.restart_sampling,
          vae: p.vae || "",
        };
        if (Array.isArray(p.loras) && p.loras.length) {
          body.lora = p.loras.map(function (l) { return { name: l.name, weight: l.weight != null ? l.weight : 1.0 }; });
        }
        // axes: send the RAW value string per dim — the server normalize_values does the
        // range/`||`/SKIP expansion (single source of truth). X is required.
        var dims = ["x", "y", "z"];
        var n = 0;
        for (var i = 0; i < dims.length; i++) {
          var d = dims[i], info = axisInfo(axisRows[d].sel.value);
          if (info.none) continue;
          var raw = (axisRows[d].ta.value || "").trim();
          if (!raw) throw new Error(d.toUpperCase() + " axis '" + info.v + "' has no values");
          body[d + "_axis"] = info.v;
          body[d + "_values"] = raw;     // STRING form — server expands it
          n++;
        }
        if (n === 0) throw new Error("Pick at least the X axis");
        // If only Y/Z set (no X), promote the first present dim to X (server requires X).
        if (!body.x_axis) {
          for (var j = 0; j < dims.length; j++) {
            var dd = dims[j];
            if (body[dd + "_axis"]) {
              body.x_axis = body[dd + "_axis"]; body.x_values = body[dd + "_values"];
              delete body[dd + "_axis"]; delete body[dd + "_values"];
              break;
            }
          }
        }
        return body;
      }

      // ── run ─────────────────────────────────────────────────────────────────
      var lastPages = [];     // [{path, url}]
      var activePage = 0;
      var running = false;

      function setMsg(t, cls) {
        msg.textContent = t;
        msg.className = "gx-msg" + (cls ? " " + cls : "");
      }

      function run() {
        if (running) return;
        var bodyObj;
        try { bodyObj = buildGridBody(); }
        catch (e) { setMsg(e.message || String(e), "err"); return; }

        running = true;
        runBtn.disabled = true;
        results.style.display = "none";
        setMsg("Running grid… cells run serially on the GPU; this can take a while.");
        var spin = el("span", { class: "gx-spin" });
        msg.insertBefore(spin, msg.firstChild);

        fetch(apiBase + "/v1/grid", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(bodyObj),
        }).then(function (r) {
          return r.json().then(function (doc) { return { status: r.status, doc: doc }; })
            .catch(function () { return { status: r.status, doc: null }; });
        }).then(function (res) {
          running = false;
          runBtn.disabled = false;
          var doc = res.doc || {};
          if (res.status === 200) {
            showPages(doc);
            setMsg("Grid done: " + (doc.cells ? doc.cells.length : "?") + " cells, " +
              ((doc.paths && doc.paths.length) || 1) + " page(s).", "ok");
          } else if (res.status === 504) {
            // partial grid — still has a (partial) composite path
            showPages(doc);
            var pend = (doc.pending && doc.pending.length) || 0;
            setMsg("Timed out; " + pend + " cell(s) unfinished. Showing the partial grid.", "err");
          } else {
            var detail = (doc && doc.detail) || ("HTTP " + res.status);
            setMsg("Grid failed: " + detail, "err");
          }
        }).catch(function (e) {
          running = false;
          runBtn.disabled = false;
          setMsg("Grid request failed: " + (e && e.message || e), "err");
        });
      }

      function showPages(doc) {
        var paths = (doc && doc.paths && doc.paths.length) ? doc.paths : (doc && doc.path ? [doc.path] : []);
        if (!paths.length) { results.style.display = "none"; return; }
        lastPages = paths.map(function (pth) {
          return { path: pth, url: (api && api.viewUrl) ? api.viewUrl(pth) : (apiBase + "/out/" + encodeURIComponent(baseName(pth))) };
        });
        activePage = 0;
        // page buttons (one per Z page)
        pagesBar.innerHTML = "";
        if (lastPages.length > 1) {
          lastPages.forEach(function (pg, i) {
            var b = el("button", { class: "btn gx-pagebtn" + (i === 0 ? " active" : ""), type: "button",
              text: "Page " + (i + 1) });
            b.addEventListener("click", function () { selectPage(i); });
            pagesBar.appendChild(b);
          });
          pagesBar.style.display = "flex";
        } else {
          pagesBar.style.display = "none";
        }
        selectPage(0);
        results.style.display = "block";
      }

      function selectPage(i) {
        if (i < 0 || i >= lastPages.length) return;
        activePage = i;
        gridImg.src = lastPages[i].url + (lastPages[i].url.indexOf("?") < 0 ? "?t=" : "&t=") + Date.now();
        Array.prototype.forEach.call(pagesBar.children, function (b, idx) {
          b.classList.toggle("active", idx === i);
        });
      }

      // ── result actions ──────────────────────────────────────────────────────
      dlBtn.addEventListener("click", function () {
        var pg = lastPages[activePage];
        if (!pg) return;
        var a = document.createElement("a");
        a.href = pg.url;
        a.download = baseName(pg.path) || "grid.png";
        document.body.appendChild(a); a.click(); a.remove();
      });
      openBtn.addEventListener("click", function () {
        var pg = lastPages[activePage];
        if (!pg) return;
        // broadcast like a normal result so resultView shows it big + gallery prepends it
        bus.emit("result:ready", { url: pg.url, filename: baseName(pg.path), params: null });
        close();
      });

      // ── open / close ────────────────────────────────────────────────────────
      function open() { refreshBase(); overlay.classList.add("show"); }
      function close() { overlay.classList.remove("show"); }
      launch.addEventListener("click", open);
      closeBtn.addEventListener("click", close);
      overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && overlay.classList.contains("show")) close();
      });
      runBtn.addEventListener("click", run);

      // let other modules open the grid (e.g. a future toolbar button)
      bus.on("grid:open", open);
      S.gridXyz = { open: open, close: close, run: run };

      // hide the launcher while a non-generate tab overlay is up, so it doesn't float
      // over Workflows/Models/etc.; show it again on the Generate tab.
      bus.on("nav:tab", function (t) {
        launch.style.display = (t === "generate" || t == null) ? "flex" : "none";
        if (t !== "generate") close();
      });

      // ── helpers ─────────────────────────────────────────────────────────────
      function baseName(p) { return String(p || "").split(/[\\/]/).pop(); }
      function truncate(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
      function esc(s) {
        return String(s == null ? "" : s)
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }

      console.info("[gridXyz] ready");
    },
  });
})();
