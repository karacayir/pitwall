// Capture UI screenshots against the replay backend -> reports/ui/
// Usage: node scripts/screenshots.mjs [baseUrl]
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const base = process.argv[2] ?? "http://localhost:3000";
const outDir = fileURLToPath(new URL("../../reports/ui/", import.meta.url));
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

async function shot(path, name, extra = null) {
  await page.goto(`${base}${path}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(4000); // let the WS stream a few laps in
  if (extra) await extra();
  await page.screenshot({ path: `${outDir}${name}.png`, fullPage: false });
  console.log(`saved ${name}.png`);
}

await shot("/", "live-board");

// find a driver number from the tower to visit the driver page
const driver = await page.getAttribute("li[data-driver]", "data-driver");
await shot(`/driver/${driver ?? 1}`, "driver-view");

await shot("/strategy", "strategy-lab", async () => {
  // run a simulation for the first available driver so the page shows results
  await page.selectOption("select", { index: 1 });
  await page.click("button:has-text('Run 2000 sims')");
  await page.waitForSelector("text=Recommended", { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);
});

await browser.close();
console.log("done");
