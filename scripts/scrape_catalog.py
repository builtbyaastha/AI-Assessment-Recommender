"""Scrapes the SHL Individual Test Solutions catalog.

Two passes:
  1. Walk the paginated listing at /products/product-catalog/?type=1
     (type=1 is Individual Test Solutions, type=2 is Pre-packaged Job
     Solutions which is out of scope for this assignment) and collect
     every (name, url, test_type) row.
  2. Visit each detail page for description / job levels / remote &
     adaptive flags / duration where SHL actually publishes them.

Needs real internet access to shl.com - won't run inside a network-locked
sandbox. Output feeds straight into build_catalog.py.

    playwright install chromium
    python scripts/scrape_catalog.py --out scripts/scraped_raw.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "https://www.shl.com/products/product-catalog/"
INDIVIDUAL_TEST_SOLUTIONS_TYPE = 1
PAGE_SIZE = 12


async def collect_listing_rows(page) -> list[dict]:
    rows: list[dict] = []
    start = 0
    while True:
        url = f"{BASE}?start={start}&type={INDIVIDUAL_TEST_SOLUTIONS_TYPE}"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_selector("table", timeout=15000)

        # there can be two tables on page 1 (job solutions + individual
        # test solutions) - grab the one with the right header
        tables = await page.query_selector_all("table")
        target_table = None
        for t in tables:
            header_text = await t.inner_text()
            if "Individual Test Solutions" in header_text:
                target_table = t
                break
        if target_table is None:
            break

        trs = await target_table.query_selector_all("tbody tr")
        if not trs:
            break

        for tr in trs:
            link = await tr.query_selector("a")
            if not link:
                continue
            name = (await link.inner_text()).strip()
            href = await link.get_attribute("href")
            full_url = href if href.startswith("http") else f"https://www.shl.com{href}"

            cells = await tr.query_selector_all("td")
            test_type_cell_text = (await cells[-1].inner_text()).strip() if cells else ""
            test_types = test_type_cell_text.split()

            # remote testing / adaptive columns are check-icons, not text
            remote = False
            adaptive = False
            if len(cells) >= 3:
                remote = bool(await cells[-3].query_selector(".catalogue__circle, .-yes, svg"))
                adaptive = bool(await cells[-2].query_selector(".catalogue__circle, .-yes, svg"))

            rows.append({
                "name": name,
                "url": full_url,
                "test_type": test_types,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
            })

        if len(trs) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    seen = set()
    deduped = []
    for r in rows:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)
    return deduped


async def enrich_detail_page(page, row: dict) -> dict:
    try:
        await page.goto(row["url"], wait_until="domcontentloaded", timeout=20000)
        body_text = await page.inner_text("body")

        desc_match = re.search(
            r"(?:Description|Overview)\s*\n(.+?)(?:\n\n|Job levels|Assessment length)",
            body_text,
            re.DOTALL,
        )
        if desc_match:
            row["description"] = " ".join(desc_match.group(1).split())[:600]

        levels_match = re.search(r"Job levels?\s*\n(.+)", body_text)
        if levels_match:
            row["job_levels"] = [
                lvl.strip() for lvl in levels_match.group(1).split(",") if lvl.strip()
            ][:10]

        duration_match = re.search(r"Assessment length[^\d]*(\d+)", body_text)
        if duration_match:
            row["duration_minutes"] = int(duration_match.group(1))
    except Exception as exc:  # noqa: BLE001 - best-effort, don't kill the whole run over one page
        row["_enrich_error"] = str(exc)
    return row


async def main(out_path: str, enrich: bool, limit: int | None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("collecting catalog listing (Individual Test Solutions)...")
        rows = await collect_listing_rows(page)
        print(f"found {len(rows)} assessments")

        if limit:
            rows = rows[:limit]

        if enrich:
            print("visiting detail pages for description/levels/duration...")
            for i, row in enumerate(rows):
                rows[i] = await enrich_detail_page(page, row)
                if (i + 1) % 20 == 0:
                    print(f"  enriched {i + 1}/{len(rows)}")

        await browser.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scripts/scraped_raw.json")
    parser.add_argument("--no-enrich", action="store_true", help="skip detail-page visits, listing data only")
    parser.add_argument("--limit", type=int, default=None, help="cap number of items, useful for testing")
    args = parser.parse_args()
    asyncio.run(main(args.out, enrich=not args.no_enrich, limit=args.limit))
