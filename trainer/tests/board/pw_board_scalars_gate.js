// SerenityBoard SCALARS + TensorBoard-parity gate.
//
// Oracle chain (all measured, 2026-07-06):
//   1. Raw values: SerenityTrainer tfevents (EventAccumulator) == board API, f32-exact
//      — gated separately in python before this script runs.
//   2. Smoothing: TensorBoard 2.20's bundled frontend algorithm, transcribed
//      VERBATIM from webfiles.zip index.js:
//        s = a.every(l => l == a[0])
//        if (s || !Number.isFinite(u)) smoothed = u
//        else { i = i*n + (1-n)*u; o++; h = (n!==1) ? 1 - Math.pow(n,o) : 1; smoothed = i/h }
//      This script runs the page's REAL smoothEMA in-browser on the REAL
//      766-point SerenityTrainer loss series and requires BIT-EQUALITY vs the
//      reference at several smoothing weights.
//   3. Render: both a Mojo run and the imported SerenityTrainer run chart together
//      (cross-trainer overlay), dynamic tags (lr/transformer) get charts,
//      console/network clean.
const { chromium } = require('playwright');
const fs = require('fs');

const URL = 'http://localhost:8188/board';
const OUT = '/home/alex/mojodiffusion/output/konva_wire';
const MOJO_RUN = 'krea2_boxjana_lora_adamw';
const SERENITY_RUN = 'serenity_zimage_alina';

// TensorBoard 2.20 bundle algorithm, verbatim semantics (reference).
const TB_BODY = `
  const isConst = ys.every((v) => v == ys[0]);
  let i = 0, o = 0;
  return ys.map((u) => {
    if (isConst || !Number.isFinite(u)) return u;
    i = i * n + (1 - n) * u;
    o++;
    const h = n !== 1 ? 1 - Math.pow(n, o) : 1;
    return i / h;
  });`;
function tbSmooth(ys, n) {
  const isConst = ys.every((v) => v == ys[0]);
  let i = 0, o = 0;
  return ys.map((u) => {
    if (isConst || !Number.isFinite(u)) return u;
    i = i * n + (1 - n) * u;
    o++;
    const h = n !== 1 ? 1 - Math.pow(n, o) : 1;
    return i / h;
  });
}

(async () => {
  const logs = [];
  const results = {};
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[console.${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  page.on('requestfailed', (r) => logs.push(`[requestfailed] ${r.method()} ${r.url()} :: ${r.failure() && r.failure().errorText}`));
  page.on('response', (r) => { if (r.status() >= 400) logs.push(`[http ${r.status()}] ${r.request().method()} ${r.url()}`); });

  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForSelector(`#run-list input[type="checkbox"][value="${MOJO_RUN}"]`, { timeout: 20000 });

  // ── 1. select BOTH runs (Mojo + imported SerenityTrainer) on the Scalars tab ──
  await page.check(`#run-list input[type="checkbox"][value="${MOJO_RUN}"]`);
  await page.waitForSelector(`#run-list input[type="checkbox"][value="${SERENITY_RUN}"]`, { timeout: 10000 });
  await page.check(`#run-list input[type="checkbox"][value="${SERENITY_RUN}"]`);
  // tags are explicit in SerenityBoard (not auto-shown like TB): select the
  // shared tag (cross-trainer overlay) + an reference trainer-only tag
  await page.waitForSelector('input[type="checkbox"][value="loss/train_step"]', { timeout: 10000 });
  await page.check('input[type="checkbox"][value="loss/train_step"]');
  await page.check('input[type="checkbox"][value="lr/transformer"]');
  await page.waitForTimeout(3000); // charts fetch + draw

  const chartInfo = await page.evaluate(() => {
    const out = { panels: [], canvases: document.querySelectorAll('canvas').length, series: {} };
    document.querySelectorAll('.chart-panel').forEach((el) => {
      out.panels.push(el.id);
      const inst = echarts.getInstanceByDom(el);
      if (inst) {
        const opt = inst.getOption();
        out.series[el.id] = (opt.series || []).map((sr) => ({
          name: sr.name, n: (sr.data || []).length,
        }));
      }
    });
    return out;
  });
  results.panels = chartInfo.panels;
  results.canvases = chartInfo.canvases;
  results.series = chartInfo.series;
  const lossPanel = chartInfo.panels.find((id) => id.startsWith('chart-loss_train_step'));
  const lrPanel = chartInfo.panels.find((id) => id.startsWith('chart-lr_transformer'));
  results.has_loss_chart = !!lossPanel && chartInfo.series[lossPanel] && chartInfo.series[lossPanel].length >= 2;
  results.has_serenity_only_tag = !!lrPanel && chartInfo.series[lrPanel] && chartInfo.series[lrPanel].length >= 1;
  // cross-trainer overlay: the loss chart must carry series for BOTH runs
  if (lossPanel) {
    const names = chartInfo.series[lossPanel].map((sr) => String(sr.name));
    results.loss_overlay_both_runs =
      names.some((n) => n.includes('krea2_boxjana')) && names.some((n) => n.includes('serenity_zimage_alina'));
  }

  // ── 2. smoothing BIT-parity vs the TB 2.20 bundle algorithm ──
  const loss = await new Promise((resolve, reject) => {
    require('http').get(`http://localhost:8188/api/board/runs/${SERENITY_RUN}/scalars?tag=loss/train_step`, (res) => {
      let b = '';
      res.on('data', (c) => (b += c));
      res.on('end', () => resolve(JSON.parse(b).map((row) => ({ x: row[0], y: row[2] }))));
    }).on('error', reject);
  });
  results.serenity_loss_points = loss.length;

  results.smoothing = {};
  for (const w of [0.0, 0.3, 0.6, 0.9, 0.97, 1.0]) {
    // run BOTH the page's smoothEMA and the TB-bundle reference IN THE SAME
    // ENGINE (Chromium) so the bar can be bit-equality (Math.pow libm varies
    // between node and chromium builds by 1-2 ulp otherwise — measured).
    const [got, ref] = await page.evaluate(
      ([pts, wt, tbSrc]) => {
        const tb = new Function('ys', 'n', tbSrc);
        const ys = pts.map((p) => p.y);
        return [smoothEMA(pts, wt).map((p) => p.y), wt <= 0 ? ys : tb(ys, Math.min(wt, 1))];
      },
      [loss, w, TB_BODY]
    );
    let maxAbs = 0, bad = 0;
    for (let i = 0; i < ref.length; i++) {
      const d = Math.abs(got[i] - ref[i]);
      if (d > maxAbs) maxAbs = d;
      if (got[i] !== ref[i]) bad++;
    }
    results.smoothing[w] = { max_abs: maxAbs, non_bitequal: bad, n: ref.length };
  }

  // ── 3. NaN / constant-series semantics (TB edge cases) ──
  results.edge = await page.evaluate(() => {
    const nanSeries = [{ x: 1, y: 1 }, { x: 2, y: NaN }, { x: 3, y: 3 }];
    const sm = smoothEMA(nanSeries, 0.6).map((p) => p.y);
    const constSeries = [{ x: 1, y: 5 }, { x: 2, y: 5 }];
    const smc = smoothEMA(constSeries, 0.9).map((p) => p.y);
    return { nan_passthrough: Number.isNaN(sm[1]), nan_skips_accum: sm[2], const_passthrough: smc[0] === 5 && smc[1] === 5 };
  });
  // reference for nan_skips_accum via tbSmooth
  results.edge.nan_skips_accum_ref = tbSmooth([1, NaN, 3], 0.6)[2];
  results.edge.nan_ok = results.edge.nan_skips_accum === results.edge.nan_skips_accum_ref;

  await page.screenshot({ path: `${OUT}/board_scalars_gate.png`, fullPage: true });
  await browser.close();

  results.console_issues = logs;
  const smoothPass = Object.values(results.smoothing).every((s) => s.non_bitequal === 0);
  const pass =
    results.has_loss_chart &&
    results.has_serenity_only_tag &&
    results.loss_overlay_both_runs === true &&
    results.serenity_loss_points === 766 &&
    smoothPass &&
    results.edge.nan_passthrough && results.edge.nan_ok && results.edge.const_passthrough &&
    logs.filter((l) => !l.includes('favicon')).length === 0;
  results.VERDICT = pass ? 'PASS' : 'FAIL';
  console.log(JSON.stringify(results, null, 2));
  process.exit(pass ? 0 : 1);
})();
