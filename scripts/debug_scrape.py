"""
Tries several URL variants to find which one actually serves the real
catalog table from your network. Prints title + table count for each.

    uv run python scripts/debug_scrape_v3.py
"""
import asyncio

from playwright.async_api import async_playwright

CANDIDATE_URLS = [
    "https://www.shl.com/solutions/products/product-catalog/",
    "https://www.shl.com/solutions/products/product-catalog/?start=0&type=1",
    "https://www.shl.com/products/product-catalog/",
    "https://www.shl.com/products/product-catalog/?start=0&type=1",
    "https://www.shl.com/en/solutions/products/product-catalog/",
]


async def try_url(page, url):
    print(f"\n=== {url} ===")
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"  goto failed: {e}")
        return

    await page.wait_for_timeout(1500)

    # accept cookies if the banner shows up (only needs doing once per browser context really)
    for selector in ["text=Allow all cookies", "button:has-text('Allow all cookies')", "#onetrust-accept-btn-handler"]:
        try:
            btn = page.locator(selector).first
            if await btn.count() > 0:
                await btn.click(timeout=3000)
                await page.wait_for_timeout(1000)
                break
        except Exception:
            pass

    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass

    title = await page.title()
    final_url = page.url
    table_count = len(await page.query_selector_all("table"))
    print(f"  title: {title}")
    print(f"  final url after any redirects: {final_url}")
    print(f"  tables found: {table_count}")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        for url in CANDIDATE_URLS:
            await try_url(page, url)

        await page.wait_for_timeout(3000)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())