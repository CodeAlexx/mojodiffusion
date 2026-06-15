/* nav_shell.js — module 'navShell'. Top-center tab bar (like serenityflow-v2's menu):
   Generate · Workflows · Models · Queue · Settings. Additive + low-risk: the Generate
   screen is the base app; other tabs show a fixed OVERLAY over the content area, so
   switching never disturbs the working Generate grid. Creates #wf-overlay + per-view
   containers; workflows.js fills #view-workflows. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;

  var TABS = [
    { id: "generate", label: "Generate", icon: "✦" },
    { id: "ideogram", label: "Ideogram", icon: "📐" },
    { id: "workflows", label: "Workflows", icon: "⌗" },
    { id: "models", label: "Models", icon: "▦" },
    { id: "queue", label: "Queue", icon: "≣" },
    { id: "settings", label: "Settings", icon: "⚙" },
  ];

  function injectCSS() {
    if (document.getElementById("style-navShell")) return;
    var css = [
      "#nav-tabs{position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:60;",
      "  display:flex;gap:4px;background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:3px}",
      "#nav-tabs .nt{background:transparent;border:0;color:var(--muted);padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600}",
      "#nav-tabs .nt:hover{color:var(--text)}",
      "#nav-tabs .nt.active{background:var(--accent2);color:#fff}",
      "#wf-overlay{position:fixed;top:48px;left:0;right:0;bottom:0;z-index:50;background:var(--bg);display:none}",
      "#wf-overlay.show{display:block}",
      "#wf-overlay .view{position:absolute;inset:0;display:none}",
      "#wf-overlay .view.show{display:flex;flex-direction:column}",
      "#wf-overlay .view-pad{padding:18px;color:var(--muted);overflow:auto}",
    ].join("\n");
    var st = document.createElement("style"); st.id = "style-navShell"; st.textContent = css;
    document.head.appendChild(st);
  }

  S.register("navShell", {
    init: function (ctx) {
      injectCSS();
      var bus = ctx.bus;

      // tab bar
      var bar = document.createElement("div"); bar.id = "nav-tabs";
      var btns = {};
      TABS.forEach(function (t) {
        var b = document.createElement("button");
        b.className = "nt"; b.dataset.tab = t.id; b.textContent = t.icon + "  " + t.label;
        b.addEventListener("click", function () { show(t.id); });
        bar.appendChild(b); btns[t.id] = b;
      });
      document.body.appendChild(bar);

      // overlay + per-view containers (generate has no view -> overlay hidden)
      var overlay = document.createElement("div"); overlay.id = "wf-overlay";
      var views = {};
      ["workflows", "ideogram", "models", "queue", "settings"].forEach(function (id) {
        var v = document.createElement("div"); v.className = "view"; v.id = "view-" + id;
        if (id !== "workflows" && id !== "ideogram") {
          var pad = document.createElement("div"); pad.className = "view-pad";
          pad.innerHTML = "<h2 style='color:var(--text);margin:0 0 8px'>" + id[0].toUpperCase() + id.slice(1) + "</h2>" +
            "<p>This panel is a placeholder. Generate + Workflows are the live tabs.</p>";
          v.appendChild(pad);
        }
        overlay.appendChild(v); views[id] = v;
      });
      document.body.appendChild(overlay);

      function show(tab) {
        TABS.forEach(function (t) { btns[t.id].classList.toggle("active", t.id === tab); });
        if (tab === "generate") {
          overlay.classList.remove("show");
        } else {
          overlay.classList.add("show");
          Object.keys(views).forEach(function (id) { views[id].classList.toggle("show", id === tab); });
        }
        bus.emit("nav:tab", tab);
      }

      S.nav = { show: show };           // let other modules switch tabs
      show("generate");                 // default
      console.info("[navShell] ready");
    },
  });
})();
