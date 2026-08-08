---
name: researchit-testing-eval
description: "Guide testing and evaluation for ResearchIT. Use for test planning, running tests, and explaining evaluation metrics. Triggers: testing plan, run tests, evaluation metrics, offline eval." 
argument-hint: "Specify scope (unit/integration/e2e) and whether to include metrics." 
---

# Testing and Evaluation Guidance

## When to Use
- The user wants to run or plan tests.
- The user asks about evaluation metrics or offline evaluation.
- You need to explain test coverage or risks.

## Required Sources
1. docs/walkthroughs/03-Code-Summary-and-Test-Plan.md
2. tests/ (overview)
3. pytest.ini
4. test_e2e_recs.py

## Procedure
1. Identify test scope (unit, integration, live, e2e).
2. Provide the correct test command(s) and file locations.
3. Call out live tests that hit external services.
4. Provide evaluation metrics and how they map to system goals.
5. Note any missing coverage or potential regressions.

## Output Format
- Test scope summary.
- Commands and expected outputs.
- Evaluation metric checklist.
