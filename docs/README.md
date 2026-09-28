# ResearchIT documentation

Updated: 2026-09-24.

## Start here

1. [Project README](../README.md): product, setup, architecture, and limits.
2. [Current technical contract](CURRENT-STATE.md): what the code actually does.
3. [Discovery plan](DISCOVERY-PLAN.md): refresh behavior, interest capture,
   reading history, validation gates, and deferred work.
4. [Doc 06](research/06-Deep-Research-Verdict.md): architectural rationale;
   later dated amendments supersede earlier decisions.
5. [CLAUDE.md](../CLAUDE.md): contributor/agent invariants.

See also [Emerging discovery quality evaluation](DISCOVERY-QUALITY-EVALUATION.md)
for the September 25 market review, proposed freshness/video experiment, and
time-aware human evaluation protocol. This is a proposal, not a completed benchmark.

The recommendation audit dashboard is generated locally, not committed:
[reproduction instructions](../reports/recommendations/README.md) show how to
build it, and it reports executed tests, source observations, and blocked live
checks separately.

[HF discovery operations](HF-DISCOVERY-OPERATIONS.md) documents the implemented
local worker, shadow readiness, baseline capture and comparison commands.

## Historical references

`TASK-TRACKER.md`, phase plans, and walkthroughs record the implementation at
specific points in time. Their checkboxes and latency numbers are not current
release status. The current technical contract takes precedence for shipped
code/defaults; the deployment must be verified separately.

- `research/01`–`05`: product vision and superseded architectural proposals.
- `research/07`: research into summaries, distillation, and scaling.
- `PHASE6-HANDOFF.md`: citation-trained model provenance and feature schema.
- `phases/PHASE7-Data-Freshness-And-Capacity.md`: historical capacity/freshness analysis.
- `phases/PHASE8-Search-And-Recommendation-Design.md`: pipeline reference, updated
  by its current-status addendum and the current technical contract.
- `walkthroughs/`: phase-specific code tours and earlier roadmap.

## How to keep this accurate

For each behavior change, update CURRENT-STATE.md and relevant README sections.
Record an architectural decision in Doc 06 when policy changes. Keep historical
measurements dated and retain their experiment conditions. Record test commands
and limitations in a dated validation entry. Mark deployment only after it is
actually checked. Do not use a phase number as a substitute for these distinctions.
