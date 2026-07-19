// SerenityBoard completeness gate: Artifacts + HParams + LoRA tabs against the
// live boxjana run. Fresh chromium, domcontentloaded. Captures every console
// error / failed request / >=400 response and asserts zero at the end.
const { chromium } = require('playwright');
const fs = require('fs');

const URL = 'http://localhost:8188/board';
const OUT = '/home/alex/mojodiffusion/output/konva_wire';
const RUN = 'krea2_boxjana_lora_adamw';
const DEMO = 'hparams_demo_run';
const CK = '/home/alex/mojodiffusion/output/krea2_boxjana_lora_adamw';
const CK_A = `${CK}/boxjana_krea2_500.safetensors`;
const CK_B = `${CK}/boxjana_krea2_2000.safetensors`;

const results = {};

(async () => {
  const logs = [];
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await ctx.newPage();
  page.on('console', m => logs.push(`[console.${m.type()}] ${m.text()}`));
  page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
  page.on('requestfailed', r => logs.push(`[requestfailed] ${r.method()} ${r.url()} :: ${r.failure() && r.failure().errorText}`));
  page.on('response', r => { if (r.status() >= 400) logs.push(`[http ${r.status()}] ${r.request().method()} ${r.url()}`); });

  console.log('goto', URL);
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // Run list populates from /api/board/runs
  await page.waitForSelector(`#run-list input[type="checkbox"][value="${RUN}"]`, { timeout: 20000 });

  // ── 1. ARTIFACTS ──────────────────────────────────────────────────────────
  await page.check(`#run-list input[type="checkbox"][value="${RUN}"]`);
  await page.click('.tab-btn[data-tab="artifacts"]');
  await page.waitForSelector('.artifact-filmstrip-thumb', { timeout: 20000 });
  // count thumbs (8 turbo PNGs = 4 slot sections x 2 checkpoints)
  const thumbs = await page.locator('.artifact-filmstrip-thumb').count();
  const sections = await page.locator('.artifact-section').count();
  // at least one preview image must have actually decoded
  await page.waitForFunction(() => {
    const imgs = Array.from(document.querySelectorAll('.artifact-preview-img'));
    return imgs.some(i => i.complete && i.naturalWidth > 0);
  }, { timeout: 20000 });
  const previewLoaded = await page.evaluate(() => {
    const imgs = Array.from(document.querySelectorAll('.artifact-preview-img'));
    return imgs.filter(i => i.complete && i.naturalWidth > 0).map(i => ({ w: i.naturalWidth, h: i.naturalHeight, src: i.getAttribute('src') }));
  });
  results.artifacts = { thumbs, sections, preview_loaded: previewLoaded.length, sample: previewLoaded[0] };
  await page.screenshot({ path: `${OUT}/board_artifacts.png`, fullPage: false });
  console.log('ARTIFACTS thumbs=%d sections=%d previewsLoaded=%d', thumbs, sections, previewLoaded.length);

  // ── 2. HPARAMS ────────────────────────────────────────────────────────────
  await page.check(`#run-list input[type="checkbox"][value="${DEMO}"]`);
  await page.click('.tab-btn[data-tab="hparams"]');
  // parcoords renders a canvas inside #hparams-container; empty-state must be gone
  await page.waitForFunction(() => {
    const c = document.getElementById('hparams-container');
    if (!c) return false;
    const hasCanvas = c.querySelector('canvas') !== null;
    const empty = c.querySelector('.empty-state') !== null;
    return hasCanvas && !empty;
  }, { timeout: 20000 });
  // pull the real values the endpoint returned. NOTE: board_boot.js shims
  // window.fetch to prepend /api/board, so request the UNPREFIXED path here.
  const hp = await page.evaluate(async () => {
    const r = await fetch('/api/compare/hparams?runs=hparams_demo_run');
    return (await r.json())[0];
  });
  results.hparams = { canvas: true, demo_hparams: hp && hp.hparams, demo_metrics: hp && hp.metrics };
  await page.screenshot({ path: `${OUT}/board_hparams.png`, fullPage: false });
  console.log('HPARAMS demo run recipe:', JSON.stringify(hp && hp.hparams));

  // ── 3. LoRA (compare two of the four checkpoints) ───────────────────────────
  await page.click('.tab-btn[data-tab="lora"]');
  await page.fill('#lora-path-1', CK_A);
  await page.fill('#lora-path-2', CK_B);
  await page.click('#lora-analyze-btn');
  await page.waitForSelector('.lora-table tbody tr', { timeout: 60000 });
  const loraRows = await page.locator('.lora-table tbody tr').count();
  const summaryShown = await page.locator('#lora-summary').isVisible();
  const chartsShown = await page.locator('#lora-charts').isVisible();
  // read the first row's rendered spectral cells to prove real numbers landed
  const firstRow = await page.evaluate(() => {
    const tr = document.querySelector('.lora-table tbody tr');
    if (!tr) return null;
    return Array.from(tr.querySelectorAll('td')).slice(0, 7).map(td => td.textContent.trim());
  });
  results.lora = { rows: loraRows, summary_bar: summaryShown, charts: chartsShown, first_row_cells: firstRow };
  await page.screenshot({ path: `${OUT}/board_lora.png`, fullPage: false });
  console.log('LoRA rows=%d summaryBar=%s charts=%s', loraRows, summaryShown, chartsShown);

  // ── console verdict ─────────────────────────────────────────────────────────
  fs.writeFileSync(`${OUT}/board_gate_console.log`, logs.join('\n') + '\n');
  const errs = logs.filter(l => l.startsWith('[pageerror]') || l.startsWith('[requestfailed]') ||
                                 l.startsWith('[http ') || l.startsWith('[console.error]'));
  results.console = { total: logs.length, errors: errs.length, error_lines: errs.slice(0, 20) };

  console.log('\n=== GATE RESULT ===');
  console.log(JSON.stringify(results, null, 1));
  const pass = results.artifacts.thumbs === 8 && results.artifacts.preview_loaded >= 1 &&
               results.hparams.canvas && results.hparams.demo_hparams &&
               results.lora.rows > 0 && errs.length === 0;
  console.log(pass ? '\nGATE PASS' : '\nGATE FAIL');
  fs.writeFileSync(`${OUT}/board_gate_result.json`, JSON.stringify(results, null, 2));
  await browser.close();
  process.exit(pass ? 0 : 1);
})().catch(e => { console.error('PW_FAIL', e); process.exit(2); });
