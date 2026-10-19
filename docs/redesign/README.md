# ResearchIT redesign

The redesign of the feed and the paper page: why it exists, the approved design, and the plan
for building it into the app. These files are copies of two artifacts made on 2026-10-17 and
2026-10-18, archived here on 2026-10-19 so they stay available after the originals go private.

## The design

`preview/index.html` is the approved clickable preview (v3). Open it in a browser from this
folder; its images load from the `redesign-images` branch. It uses the app's real stylesheet and real
September 2026 papers, and switches between **Today** and **Proposed** for the feed and the
paper page. Widen or narrow the window to see the laptop and phone layouts. Its answers and
data are hard-coded; it is a design reference, not the product.

The screenshots and diagrams the documents show were taken from it with Playwright on
2026-10-18. They live on the `redesign-images` branch, not here: this folder is deployed to
the Hugging Face Space, which refuses binary files in git, so the documents link to them by
raw URL.

## The documents

| File | What it holds |
| --- | --- |
| [01-vision-history-next-steps.md](01-vision-history-next-steps.md) | Why: the use case, history, where "In plain terms" comes from, PaperCode, the richer paper page, figures and Ask, less clutter, future paths, next steps |
| [02-implementation-plan.md](02-implementation-plan.md) | The summary: architecture, ten pull requests in five milestones, data model, decisions, guardrails, risks, metrics, evidence |
| [03-start-here-agent-handoff.md](03-start-here-agent-handoff.md) | For whoever builds it: hand-off prompt, target screens, reading list, traps T1–T18, rules, routine |
| [04-pr-1-3-foundations.md](04-pr-1-3-foundations.md) | PR 1 click logging and baseline, PR 2 daily harvest and licence, PR 3 full abstracts |
| [05-pr-4-5-paper-sources-figures.md](05-pr-4-5-paper-sources-figures.md) | arXiv HTML layout, PR 4 paper source service, PR 5 figure thumbnails |
| [06-pr-6-7-paper-page-briefing.md](06-pr-6-7-paper-page-briefing.md) | Target layout, PR 6 paper page v2, PR 7 briefing |
| [07-pr-8-ask-this-paper.md](07-pr-8-ask-this-paper.md) | PR 8 Ask this paper: request path, build steps, checks, tests |
| [08-pr-9-10-feed.md](08-pr-9-10-feed.md) | PR 9 feed card v2, PR 10 wide feed, filter and figure warm-up |
| [09-testing-playbook.md](09-testing-playbook.md) | Test layers, fixtures and fakes, required tests per PR, commands |
| [10-release-and-rollback.md](10-release-and-rollback.md) | Rollout stages, deploy routine, rollback, what to watch |

The Markdown was rebuilt from the session that wrote the original Claude Doc. Edits made
directly in that doc by anyone else are not included. Diagrams are the last screenshot taken
of each one, so two of them (`diagram-4.png`, `diagram-5.png`) may miss small label edits made
after the screenshot.
