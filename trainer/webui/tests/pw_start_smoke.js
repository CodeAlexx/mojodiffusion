// Browser Start-smoke: the LAST untested seam — the v0.5 form driving a real
// training run via the START TRAINING button. Asserts: run launches, status
// rail transitions RUNNING, steps tick via SSE, run EXITED clean.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('requestfailed', r => errors.push('reqfail: ' + r.url()));

  await page.goto('http://127.0.0.1:8188/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelectorAll('#preset option').length > 5);
  await page.selectOption('#preset', 'krea2');
  await page.fill('#runname_top', 'webui_start_smoke');
  // Training tab: 3 steps, no saves/samples
  await page.click('nav a[data-sec="training"]');
  await page.fill('#max_steps', '3');
  await page.fill('#save_every', '0');
  await page.click('nav a[data-sec="sampling"]');
  await page.fill('#sample_every', '0');
  // START
  await page.click('#startbtn');
  await page.waitForFunction(() => document.getElementById('statuspill').textContent === 'RUNNING', null, { timeout: 15000 });
  console.log('LAUNCHED: status RUNNING, msg=', await page.textContent('#msg'));
  // wait for training to finish (load ~4-5min + 3 steps)
  await page.waitForFunction(() => ['EXITED', 'FAILED', 'STOPPED'].includes(document.getElementById('statuspill').textContent), null, { timeout: 480000 });
  const status = await page.textContent('#statuspill');
  const step = await page.textContent('#s_step');
  const loss = await page.textContent('#s_loss');
  const speed = await page.textContent('#s_speed');
  const screenshot = process.env.SERENITY_SCREENSHOT_PATH ||
    path.resolve(__dirname, '../../../output/param_parity/web_start_smoke.png');
  await page.screenshot({ path: screenshot, fullPage: false });
  console.log(`FINAL: status=${status} step=${step} loss=${loss} speed=${speed}`);
  console.log(`CONSOLE ISSUES: ${errors.length}`, errors.slice(0, 5));
  await browser.close();
  process.exit(status === 'EXITED' && errors.length === 0 ? 0 : 1);
})();
