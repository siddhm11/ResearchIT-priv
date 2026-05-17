"""Drive a real Chromium browser to verify the search UI shows results once."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
QUERY = "attention is all you need"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 1800},
        )
        # Pre-seed cookie of a user that has saves so has_recs=True
        ctx.add_cookies([{
            "name": "arxiv_user_id",
            "value": "browser-test-user",
            "url": URL,
        }])
        page = ctx.new_page()

        # 1) Land on the homepage and search from there.
        page.goto(URL + "/", wait_until="networkidle")
        page.fill("input[name='q']", QUERY)
        page.screenshot(path="scripts/screenshot_before_submit.png", full_page=True)

        page.click("button[type='submit']")
        page.wait_for_url("**/search?q=*", timeout=10_000)
        # search.html does not auto-load anything heavy when q is set, but give it a beat
        page.wait_for_load_state("networkidle", timeout=15_000)

        page.screenshot(path="scripts/screenshot_after_search.png", full_page=True)

        # 2) Inspect the DOM
        url = page.url
        paper_cards = page.locator(".paper-card").count()
        recs_visible = page.locator("#rec-section").count()
        recs_heading = page.get_by_role("heading", name="Recommended for You").count()
        results_heading_count = page.locator("text=results for").count()

        print(f"URL after search: {url}")
        print(f".paper-card count: {paper_cards}")
        print(f"#rec-section count: {recs_visible}")
        print(f"'Recommended for You' heading count: {recs_heading}")
        print(f"'results for' phrase count: {results_heading_count}")

        # 3) Check for duplicate paper IDs (the original 'twice' complaint)
        ids = page.locator("[id^='paper-']").evaluate_all(
            "els => els.map(e => e.id)"
        )
        unique = set(ids)
        print(f"paper element ids: {len(ids)} total, {len(unique)} unique")
        if len(ids) != len(unique):
            from collections import Counter
            dups = [k for k, v in Counter(ids).items() if v > 1]
            print(f"DUPLICATE IDS: {dups}")

        # Phase: title-match boost — Vaswani's "Attention Is All You Need"
        # (1706.03762) must be the #1 result for this exact-title query.
        first_paper_id = page.locator("[id^='paper-']").first.get_attribute("id")
        print(f"first paper id: {first_paper_id}")

        ok = (
            recs_visible == 0
            and recs_heading == 0
            and results_heading_count == 1
            and paper_cards == len(unique)
            and paper_cards > 0
            and first_paper_id == "paper-1706.03762"
        )
        print("\nRESULT:", "PASS" if ok else "FAIL")

        browser.close()


if __name__ == "__main__":
    run()
