// Browser verification of the explorer: loads the page in the local Microsoft Edge (puppeteer-core) and checks
// that it renders, that picking a pixel opens the inspector with a consistent grid, that the phone layout has no
// horizontal scroll, and that the data pack's layers are all reachable.
//
//   python3 -m http.server 8772 --directory site &      (from the project root)
//   node tools/verify_site.mjs http://127.0.0.1:8772/index.html?data=data [outdir]
//
// Exit code 0 = all checks passed. Screenshots (desktop, phone, inspector) are written to outdir.
import puppeteer from "puppeteer-core";
import fs from "node:fs";

const url = process.argv[2] || "http://127.0.0.1:8772/index.html?data=data";
const out = process.argv[3] || "/tmp/verify_site";
fs.mkdirSync(out, { recursive: true });
const EDGE = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge";
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok, detail }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`); };

const browser = await puppeteer.launch({ executablePath: EDGE, headless: "new", args: ["--no-sandbox", "--disable-gpu", "--hide-scrollbars"] });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  const bad = [];
  page.on("response", (r) => { if (r.status() >= 400) bad.push(`${r.status()} ${r.url()}`); });

  // ---- desktop
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => document.body.dataset.ready === "1" || document.body.dataset.error, { timeout: 60000 });
  check("page loaded the data pack", await page.evaluate(() => !document.body.dataset.error), await page.evaluate(() => document.body.dataset.error || ""));
  check("no 4xx/5xx responses", bad.length === 0, bad.join(" | "));

  // canvas is not blank: count distinct colours in a sample of pixels
  const distinct = await page.evaluate(() => {
    const c = document.getElementById("stage"), g = c.getContext("2d");
    const d = g.getImageData(0, 0, c.width, c.height).data, seen = new Set();
    for (let i = 0; i < d.length; i += 4 * 977) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return seen.size;
  });
  check("canvas shows an image (not blank)", distinct > 40, `${distinct} distinct colours sampled`);
  await page.screenshot({ path: `${out}/desktop.png` });

  const layers = await page.evaluate(() => fetch((new URLSearchParams(location.search).get("data") || "data").replace(/\/?$/, "/") + "manifest.json").then((r) => r.json()).then((m) => m.layers.length));
  check("manifest lists layers", layers >= 1, `${layers} layers`);

  // ---- pick a pixel through the real pointer path
  await page.mouse.move(720, 450);
  await page.mouse.click(720, 450);
  await new Promise((r) => setTimeout(r, 600));
  const panel = await page.evaluate(() => {
    const ins = document.getElementById("inspector");
    const cells = [...document.querySelectorAll("#iGrid .cell")];
    const given = cells.filter((c) => c.classList.contains("given")).length;
    return { hidden: ins.hidden, nCells: cells.length, given, badge: document.getElementById("iBadge").textContent,
             loops: document.getElementById("iLoops").textContent, coords: document.getElementById("iCoords").textContent,
             digits: cells.filter((c) => /[1-9]/.test(c.textContent)).length };
  });
  check("click opens the inspector", panel.hidden === false, JSON.stringify({ badge: panel.badge, loops: panel.loops, coords: panel.coords }));
  check("inspector shows 81 cells", panel.nCells === 81);
  check("givens are styled (some, not all)", panel.given > 10 && panel.given < 60, `${panel.given} givens`);
  check("badge is one of SOLVED/STUCK/UNSETTLED", ["SOLVED", "STUCK", "UNSETTLED"].includes(panel.badge));
  check("loops is a number 0..24", /^\d+$/.test(panel.loops) && +panel.loops <= 24);
  await page.screenshot({ path: `${out}/inspector.png` });

  // ---- featured starts: each chip selects a pixel with a 24-loop replay
  const nChips = await page.evaluate(() => document.querySelectorAll("#tryRow .chip").length);
  if (nChips > 0) {
    for (let i = 0; i < nChips; i++) {
      await page.evaluate((i) => document.querySelectorAll("#tryRow .chip")[i].click(), i);
      await new Promise((r) => setTimeout(r, 900));
      const st = await page.evaluate(() => ({ replay: !document.getElementById("iReplay").hidden, cells: document.querySelectorAll("#iGrid .cell").length,
        label: document.querySelectorAll("#tryRow .chip")[0] && document.getElementById("iBadge").textContent, loops: document.getElementById("iLoops").textContent }));
      check(`featured start ${i + 1}: inspector with replay controls and 81 cells`, st.replay && st.cells === 81, JSON.stringify(st));
    }
    await page.click("#btnPlay");
    await new Promise((r) => setTimeout(r, 800));
    const loopNo = await page.evaluate(() => document.getElementById("iLoopNo").textContent);
    check("replay advances through the loops", /^loop \d+$/.test(loopNo), loopNo);
    await page.screenshot({ path: `${out}/featured.png` });
  } else check("featured starts present", false, "manifest.meta.featured is empty");

  // ---- explainer drawer
  await page.click("#btnExplain");
  const drawerOpen = await page.evaluate(() => !document.getElementById("drawer").hidden);
  check("drawer opens", drawerOpen);
  await page.screenshot({ path: `${out}/drawer.png` });
  await page.keyboard.press("Escape");

  // ---- phone width: no horizontal scroll
  await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => document.body.dataset.ready === "1", { timeout: 60000 });
  const scroll = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth, bw: document.body.scrollWidth }));
  check("phone: no horizontal scroll", scroll.sw <= scroll.iw && scroll.bw <= scroll.iw, JSON.stringify(scroll));
  await page.screenshot({ path: `${out}/phone.png` });
  await page.touchscreen.tap(190, 420);
  await new Promise((r) => setTimeout(r, 600));
  const sheet = await page.evaluate(() => { const i = document.getElementById("inspector"); const r = i.getBoundingClientRect(); return { hidden: i.hidden, top: r.top, bottom: r.bottom, ih: innerHeight, peek: i.classList.contains("peek") }; });
  check("phone: tap opens a bottom sheet", sheet.hidden === false && Math.abs(sheet.bottom - sheet.ih) < 2, JSON.stringify(sheet));
  await page.screenshot({ path: `${out}/phone_sheet.png` });

  check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
} finally {
  await browser.close();
}
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(`${out}/results.json`, JSON.stringify(results, null, 1));
console.log(failed.length ? `\n${failed.length} check(s) FAILED` : "\nall checks passed");
process.exit(failed.length ? 1 : 0);
