# Release and rollback

Code reaches production switched off, then turns on in stages. Undoing a feature is a switch flip, never a hurried code change.

## Rollout stages for each switch

Each feature climbs these stages one at a time, and only the owner moves it up. Changing a Space variable restarts the Space (T12), so group switch changes into one restart at a quiet hour.

1. **Off.** The code is merged and deployed but dormant. Check that health endpoints are green and that page timings haven't moved.
2. **Shadow** (PR 4, PR 7). The feature does its work and shows nothing. Run it for 3 days and watch errors, arXiv soft blocks, model spend and timings.
3. **Preview** (PR 6–10). Only the owner sees it, through the preview cookie. Run it for 3 days on at least 20 papers: figure-allowed with figures, arXiv licence, no HTML page, a 2008 paper, maths-heavy, and phone and laptop widths.
4. **On.** Everyone sees it. Watch the After launch signals below for 2 weeks against the PR 1 baseline before turning on the next feature.

## Deploy routine

The owner deploys. An agent runs these steps only when the owner says so in the session.

1. The PR is merged on GitHub without squashing, and CI is green on `main`.
2. On a clean checkout of `main`: run `make check`.
3. For PRs that add replicated tables (3, 4, 7, 8, 10), run a fresh-container restore drill before the first deploy, as DISCOVERY-PLAN.md's release gates require. Use a scratch Turso database the owner creates: start the app with an empty `DB_PATH`, the scratch credentials and replication on; write rows; restart with a new empty `DB_PATH`; confirm every new table restores and the next sync pushes nothing twice.
4. `git push hf main`. Hugging Face keeps the old container serving until the new one is ready. Its edge returned intermittent 502s for about five minutes after the switch on 2026-09-28, so wait before judging.
5. Smoke checks, all read-only:
    1. `/healthz/deep`: sidecar, Turso, model chain and Turso sync status all ok.
    2. `/healthz/runtime` and, from PR 4, `/healthz/sources`.
    3. Five paper pages: 2609.31507 (CC BY, figures), 2609.23863 (arXiv licence, so no figures), 1706.03762 (2017), 0704.0001 (the oldest id), and one paper from yesterday.
    4. One search, and the feed with the owner's own cookie.
6. Record the deploy: the commit, the date and the smoke results in CURRENT-STATE's header, and the Build order status in this doc.

## Rolling back

The first move is always the switch. Code changes come after the bleeding stops.

| What went wrong | First move | Then |
| --- | --- | --- |
| A feature misbehaves | Set its switch to `off` | Fix on a branch, test, and climb the stages again |
| A figure shows under the wrong licence | `FIGURES=off` at once | Fix `licences.py` and add the case to its test. The restart clears the image cache in `/tmp` |
| arXiv blocks us or complains | `PAPER_SOURCES=off`, `FIGURE_PREWARM=off` | Halve the pacer rate and confirm the `User-Agent` is sent |
| Model spend jumps | Lower `LLM_DAILY_USD_CAP`, or `BRIEFING=off` and `ASK=off` | Read `llm_calls` to find the cause |
| Bad cached model output | Raise the prompt version, which gives every paper a new cache key | For one paper, the owner deletes its rows with a script that has a dry-run mode |
| The deploy breaks start-up or a page | Revert the merge on a branch (`git revert -m 1 <merge>`), open a PR, merge, `git push hf main` | Never force-push `hf` |

**Schema changes are additive only.** No PR in this plan drops or renames a column the running code uses; T7's rebuild of the explanations table keeps all of them. So the previous release always runs on the newer database and a code rollback never needs a data rollback.

## After launch: what to watch

Check these daily for the first week after a switch goes `on`, then weekly. The normal levels come from PR 1's baseline and from the shadow stages.

| Signal | Where | Act when |
| --- | --- | --- |
| Paper page p95 | `/healthz/runtime`, `paper_page.total_ms` | 30% above the baseline for a day |
| Feed p95 | `/healthz/runtime`, `feed.build_ms` | 10% above the baseline for a day |
| arXiv soft blocks | `/healthz/sources` | The breaker opens twice in a day |
| Fetch failures | `/healthz/sources` | Over 5% of fetches |
| Model spend | `llm_calls`, and the provider's dashboard | 80% of the daily cap is used by noon UTC |
| Briefings not shown | `paper_briefings` status | The share marked insufficient or failed doubles week on week |
| Ask quotes dropped | `ask_log` | Over 20% of quotes dropped in a week |
| Turso backup | `/healthz/deep`, Turso sync status | Any table listed as not replicating, or a sync error |
| Reader reports | `content_reports` | Any report: review it within 2 days |
| Saves, Pass share, seven-day return | PR 1's report, rerun weekly | Worse than the normal weekly range two weeks in a row |

With few readers these numbers move slowly. The five-reader sessions in milestone M5 will surface problems sooner than the charts.
