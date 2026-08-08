"""Verify the onboarding seed-search step does not duplicate the panel."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
QUERY = "attention is all you need"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 1800})
        # Use a fresh, unonboarded user so we land on /onboarding
        ctx.add_cookies([{
            "name": "arxiv_user_id",
            "value": "onboarding-test-user-fresh",
            "url": URL,
        }])
        page = ctx.new_page()

        page.goto(URL + "/onboarding", wait_until="networkidle")

        # Step 1: pick a category, click Continue
        page.click("[data-key='nlp']")
        page.click("#continue-btn")

        # Step 2 should appear (rendered by submitCategories() via fetch + innerHTML)
        page.wait_for_selector("#seed-results", timeout=10_000)

        # Snapshot before search
        page.screenshot(path="scripts/screenshot_onboard_step2_before.png", full_page=True)

        # Now search — this is what triggered the duplication bug
        page.fill("input[name='q']", QUERY)
        page.click("button:has-text('Search')")
        # wait for results to swap in
        page.wait_for_function(
            "document.querySelectorAll('.seed-card').length > 0",
            timeout=15_000,
        )
        page.wait_for_load_state("networkidle", timeout=15_000)

        page.screenshot(path="scripts/screenshot_onboard_step2_after.png", full_page=True)

        # ── Inspect the DOM
        save_panels = page.locator("h2:has-text('Save a few papers you like')").count()
        quick_imports = page.locator("text=Quick import:").count()
        search_inputs = page.locator("input[name='q']").count()
        seed_counters = page.locator("#seed-counter").count()
        done_buttons = page.locator("button:has-text('Done — start exploring')").count()
        seed_cards = page.locator(".seed-card").count()
        seed_card_ids = page.locator(".seed-card").evaluate_all("els => els.map(e => e.id)")

        print(f"'Save a few papers you like' headings: {save_panels} (expected 1)")
        print(f"'Quick import:' blocks: {quick_imports} (expected 1)")
        print(f"search inputs: {search_inputs} (expected 1)")
        print(f"#seed-counter elements: {seed_counters} (expected 1)")
        print(f"'Done — start exploring' buttons: {done_buttons} (expected 1)")
        print(f"seed-cards: {seed_cards}, unique ids: {len(set(seed_card_ids))}")

        ok = (
            save_panels == 1
            and quick_imports == 1
            and search_inputs == 1
            and seed_counters == 1
            and done_buttons == 1
            and seed_cards > 0
            and seed_cards == len(set(seed_card_ids))
        )
        print("\nRESULT:", "PASS" if ok else "FAIL")

        browser.close()


if __name__ == "__main__":
    run()
