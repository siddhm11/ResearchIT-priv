# AI-judged recommendation test

Answers "of what a model would put on the screen, how much is worth opening?" on two
surfaces, the way an Up next rail and a home feed are evaluated:

| Surface | Context | Each model shows |
|---|---|---|
| `upnext` | a paper the reader just finished | the 10 papers it would suggest next |
| `home` | a real reader's recent saves and dislikes (Scholar Inbox) | the 10 papers it would put in their feed |

Two LLM judges grade every suggested paper 0-3 **blind**: picks from all models are
pooled, de-duplicated, shuffled and relabelled `c1..cN` before the judge sees them.

## Run

```bash
cd scripts/embedding_eval/judge
PY=~/.cache/researchit-embedding-eval/venv-main/bin/python
$PY cases.py                                  # build cases.json (deterministic, seed 42)
$PY judge.py openai/gpt-oss-120b &            # primary judge   } run in parallel:
$PY judge.py qwen/qwen3.8-27b &               # second judge    } separate Groq limits
$PY judge.py openai/gpt-oss-120b --retest     # 10% of cases again, reshuffled
$PY score.py                                  # metrics + judge checks -> results.json
```

### Free-tier budget and extra keys

Groq's free tier allows each model ~200k tokens **per day** (and 8k per minute) per
account, roughly 1,300 graded papers. Extra keys go one per line in
`~/.cache/researchit-embedding-eval/judge/.groq_keys` (chmod 600, never in the repo).
Run one worker per key and model, each on its own slice of the cases:

```bash
for i in 0 1 2; do for m in openai/gpt-oss-120b qwen/qwen3.8-27b; do
  $PY judge.py $m --k 5 --shard $i/3 --key $i &      # first screen, then --k 10
done; done
```

Other options: `--only home|upnext`, `--short` (shorter snippets, stored as prompt
version `v1-short`), `--retest`. A worker stops cleanly at the daily cap; re-run later.
`score.py 5` / `score.py 10` reports the first screen / top 10, using the
gpt-oss-120b + Qwen3.8 panel where both graded a whole case.

Everything lands in `~/.cache/researchit-embedding-eval/judge/`. Grades are cached in
`judgments.sqlite` keyed by (case, paper, judge, prompt version), so re-runs, crashes
and **new models only pay for papers not graded yet**. To add a model: give it vectors
in the bake-off cache (`vec_<name>.npy` + `ids_<name>.json`), add it to `MODELS` in
`cases.py`, rebuild cases, re-run both judges, re-score.

Cost: Groq free tier. A full run is ~2,600 graded papers per judge for the first
screen and ~5,100 for the top 10, so a single account needs two to four days per
judge; three accounts in parallel finished both judges in about 35 minutes. Paper text sent to Groq is
public arXiv metadata; home cases also send titles of papers the reader saved (from a
public dataset).

## Metrics (per case, top 10)

- `good10` papers graded >= 2 (worth opening); `great10` >= 2.5; `off10` <= 0.5
- `first5` mean grade of the first five
- `ndcg10` gain 2^g - 1, ideal = best 10 grades among the pooled papers
- `cited10` citation-linked papers (up next), `cats10` distinct categories, `fresh10`
  share published within a year before the case date (home)

A paper's grade is the mean of the two judges. Intervals are paired bootstraps over
cases; "top tier" is not significantly worse than the best after Holm correction.

## Judge checks (in `results.json` -> `checks`)

- cited vs uncited AUC: do linked papers get higher grades? (judge never sees links)
- liked vs disliked AUC: on papers the reader actually rated later, mixed in blind
- between judges and retest: exact / within-1 agreement, quadratic weighted kappa

Change the prompts in `judge.py` only together with `PROMPT_VERSION`, or cached grades
from the old prompt will be mixed with new ones.
