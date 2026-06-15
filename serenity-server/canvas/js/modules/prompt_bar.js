/* prompt_bar.js — module 'promptBar'. SwarmUI-style bottom-center prompt bar.
   Owns #prompt-bar: positive prompt (wide) + char counter + "+" image button,
   negative prompt, and the primary Generate button (id 'btn-generate', which
   generateWS binds). Writes state.params.prompt / state.params.negative. */
(function () {
  "use strict";
  var S = window.Serenity;
  if (!S || typeof S.register !== "function") return;
  var ROOT_ID = "prompt-bar";

  function injectCSS() {
    if (document.getElementById("style-promptBar")) return;
    var css = [
      "#" + ROOT_ID + "{display:flex;align-items:stretch;gap:8px;padding:8px 10px;background:var(--panel);border-top:1px solid var(--line)}",
      "#" + ROOT_ID + " .pb-add{align-self:flex-start;font-size:18px;line-height:1;padding:8px 12px}",
      "#" + ROOT_ID + " .pb-cols{flex:1;display:flex;flex-direction:column;gap:6px;min-width:0}",
      "#" + ROOT_ID + " .pb-pos-wrap{position:relative}",
      "#" + ROOT_ID + " textarea{width:100%;resize:none;font-size:13px;line-height:1.4}",
      "#" + ROOT_ID + " textarea.pb-pos{min-height:46px}",
      "#" + ROOT_ID + " textarea.pb-neg{min-height:30px;color:var(--muted)}",
      "#" + ROOT_ID + " .pb-count{position:absolute;right:8px;bottom:4px;font-size:10px;color:var(--muted);pointer-events:none}",
      "#" + ROOT_ID + " .pb-gen{display:flex;align-items:stretch}",
      "#" + ROOT_ID + " #btn-generate{min-width:140px;font-size:15px;font-weight:700;padding:0 22px}",
    ].join("\n");
    var st = document.createElement("style");
    st.id = "style-promptBar"; st.textContent = css; document.head.appendChild(st);
  }

  function elx(tag, attrs, kids) {
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

  S.register("promptBar", {
    init: function (ctx) {
      injectCSS();
      var get = ctx.get, set = ctx.set, bus = ctx.bus;
      var root = (ctx.dom && ctx.dom.promptBar) || document.getElementById(ROOT_ID);
      if (!root) { console.warn("[promptBar] mount not found:", ROOT_ID); return; }
      root.innerHTML = "";

      // "+" attach image -> uploads via api and emits a bus event the canvas can pick up
      var addBtn = elx("button", { class: "btn pb-add", type: "button", title: "Attach init image", text: "+" });
      addBtn.addEventListener("click", function () {
        var inp = elx("input", { type: "file", accept: "image/*" });
        inp.addEventListener("change", function () {
          if (inp.files && inp.files[0]) bus.emit("prompt:image", inp.files[0]);
        });
        inp.click();
      });

      var pos = elx("textarea", { class: "pb-pos", rows: 2,
        placeholder: "Type your prompt here… (or drag/paste an image to use image prompting)" });
      pos.value = get("params.prompt") || "";
      var count = elx("span", { class: "pb-count", text: "0/75" });
      function updateCount() {
        var words = pos.value.trim() ? pos.value.trim().split(/\s+/).length : 0;
        count.textContent = words + "/75";
      }
      pos.addEventListener("input", function () { set("params.prompt", pos.value); updateCount(); });
      updateCount();
      bus.on("change:params.prompt", function (v) { if (document.activeElement !== pos && v != null) { pos.value = v; updateCount(); } });

      var neg = elx("textarea", { class: "pb-neg", rows: 1, placeholder: "Optionally, type a negative prompt here…" });
      neg.value = get("params.negative") || "";
      neg.addEventListener("input", function () { set("params.negative", neg.value); });
      bus.on("change:params.negative", function (v) { if (document.activeElement !== neg && v != null) neg.value = v; });

      var genBtn = elx("button", { id: "btn-generate", class: "btn btn-primary", type: "button", text: "Generate" });

      // Ctrl/Cmd+Enter generates from either box
      function genKey(e) { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); genBtn.click(); } }
      pos.addEventListener("keydown", genKey);
      neg.addEventListener("keydown", genKey);

      root.appendChild(addBtn);
      root.appendChild(elx("div", { class: "pb-cols" }, [
        elx("div", { class: "pb-pos-wrap" }, [pos, count]),
        neg,
      ]));
      root.appendChild(elx("div", { class: "pb-gen" }, [genBtn]));
      console.info("[promptBar] ready");
    },
  });
})();
