# Qwen key rotation and model fallback plan

Date: 2026-10-26. Status: first opt-in implementation validated locally;
disabled by default, live access/quality validation and deployment pending.
Scope confirmed by the user: Qwen text generation through Groq, not embeddings.

Release integration: prepared against cached `origin/main` at `cf44d2646e` in
an isolated checkout. The initial working branch was older than main. Main
already has a single-key model chain and an October 4 model comparison;
this integration preserves them and adds credential rotation. The first two
`GROQ_MODELS` supply the new router's default order, currently Qwen then
GPT-OSS 20B. The earlier 120B-first proposal below has been superseded.
A fresh remote fetch/push was blocked by GitHub DNS resolution in this session.

## Implementation checkpoint and immediate actions

Implemented: async model-first routing, scoped credential/quota/permission
failures, cooldowns and recovery probes, one total deadline, cancellation,
bounded in-flight requests, model-specific parameters, all three callers,
actual-model cache attribution, shared concurrent explanation generation,
output escaping, and sanitized diagnostics. No dependency was added.

One distinct Groq credential was found locally. Organization/project membership
is unknown. A live catalog probe failed with `APIConnectionError`, without an
HTTP status; no completion was sent. This does not prove the key is invalid or
that either model is available. Local credentials and deployment settings were
not modified. The new flag remains off.

The next release steps are (the code is prepared; publication is still blocked):

1. Add backup secrets through the deployment's secret store, using names such
   as `GROQ_API_KEY_2`. Configure their non-secret references and real quota
   membership in `GROQ_KEY_POOL_JSON` using `.env.example`. Do not share key
   values in a chat or commit them. Additional keys in the same organization
   help with key failures, but do not add organization quota.
2. On a network-enabled machine run the commands below. Resolve access errors
   before turning on failover. The completion probe checks exact parameters as
   well as authentication; its five-second smoke budget is not a measured
   production latency guarantee.
3. Run the held-out task comparison described below; choose whether the current
   preview Qwen is acceptable and establish the fallback's quality. This
   comparison has not been performed by the implementation tests.
4. Canary with `GROQ_FAILOVER_ENABLED=1` and `GROQ_FAILOVER_TASKS=explain`.
   Then add `summary`, then `rewrite` only after quality/cost/latency review.
   Restart after configuration or credential changes. Tasks outside the list
   retain the legacy path, whose model access also needs verification.

```bash
# Configuration inventory only; never prints secret values.
.venv/bin/python scripts/check_groq_routes.py
# Live catalog check, followed by optional short route completions.
.venv/bin/python scripts/check_groq_routes.py --probe
.venv/bin/python scripts/check_groq_routes.py --probe --complete
```

Deliberately still pending: proactive token-quota reservations based on measured
usage/limits, router-specific aggregated health counters and distributed quota coordination.
Existing `ai_usage` logging and `scripts/eval_groq_models.py` are preserved;
a refreshed quality evaluation has not been run. Current v1 enforces concurrency
(four requests total, one per credential) and reacts to provider quota errors;
it does not claim to predict token consumption or guarantee no 429s. No paid
provider fallback, account creation, or quota expansion was added.

The design below remains the full target; this checkpoint is the source of
truth for which parts have been implemented.

## Decision

Prefer Qwen across all eligible configured credentials before moving to the
next ranked model. Keep model preference separate from credential health:

```text
Qwen: key A -> key B -> key C
  all eligible routes failed or are already known unavailable
Fallback model: key A -> key B -> key C
  all eligible routes failed or are already known unavailable
Existing graceful degradation
```

Stop at the first valid result. A bad key must not make us abandon a good model.
An exhausted shared quota, retired model, invalid request, or provider outage
must not cause a pointless sweep through every key. A request deadline can
terminate this sequence early; “all keys first” cannot mean unbounded waiting.

This is request failover among existing authorized credentials. Credential
replacement/revocation remains a separate operational activity.

## Legacy behavior preserved while failover is disabled

- The legacy path uses only `GROQ_API_KEY`.
- `app/groq_svc.py` holds one lazy synchronous client and traverses
  `GROQ_MODELS` for rewriting, search summaries, and explanations.
- Calls run in executor threads with 2, 4, and 8 second budgets. Main already
  disables SDK retries. The new async path additionally enforces an outer
  cancellation deadline including queuing and transport.
- Failures return the original search query or omit the generated section.
  Preserve these caller contracts and the existing academic-query skip.
- The previous explanation cache used a fixed model identity. The new cache
  includes title and actual model; old entries are left intact but do not
  collide with new entries.

## Model choice and availability gate

Proposed evaluation order:

| Priority | Model | Decision before enabling |
|---|---|---|
| 1 | `qwen/qwen3.8-27b` | User's preferred family; verify exact ID, access, output behavior, latency, and preview suitability |
| 2 | `openai/gpt-oss-20b` | Preserve main's recorded October 4 ranking; repeat access/quality validation before rollout |
| Alternative | `openai/gpt-oss-120b` | Supported task override; the new router uses at most two models per task |

The [current catalog](https://console.groq.com/docs/models) lists this Qwen as
preview and both GPT-OSS candidates as production models. Groq describes preview
models as evaluation-only, with possible discontinuation at short notice. Qwen
therefore stays opt-in during evaluation; production enablement needs an explicit
decision about that availability risk or a supported Qwen serving option.

Do not use stale IDs from old examples: Groq's
[deprecation record](https://console.groq.com/docs/deprecations) lists
`qwen/qwen3-32b` as shut down on July 17, 2026, Llama 3.3 70B on August 16,
and Qwen 3.6 27B superseded by Qwen 3.8 in September. The stated retirements have
enterprise exceptions; verify this account's access. These are documentation
findings, not live tests of our credentials.

“Second best” means the best remaining model that meets our quality, latency,
access, and cost requirements. Main records an October 4 text-generation comparison favoring GPT-OSS 20B
as fallback. That is historical evidence, not a new measurement from this
session. Keep an explicit ordered allowlist per task, defaulting to the first
two `GROQ_MODELS`. Do not auto-promote a model from a catalog.

## Credentials, quotas, and state

Use one shared router (`app/llm_router.py`) behind the existing service functions.
Configure credentials as records, not an opaque comma-separated secret list:

```json
[
  {"id": "primary", "secret_env": "GROQ_API_KEY", "org": "org-a", "project": "app"},
  {"id": "backup", "secret_env": "GROQ_API_KEY_2", "org": "org-a", "project": "app"}
]
```

Implemented settings: `GROQ_KEY_POOL_JSON` for these records,
`GROQ_MODEL_ORDER_JSON` for task-specific model lists, and `GROQ_FAILOVER_ENABLED`
with `GROQ_FAILOVER_TASKS` for staged rollout. Only the three model IDs in the
table above have adapters; configuration permits one or two models per task.
When no pool is supplied, wrap the existing `GROQ_API_KEY` as one credential.
Ignore missing/blank secrets, deduplicate identical secrets internally, validate
unique aliases, and report configuration errors without printing secret values.
Unknown organization/project membership must not be treated as independent quota.

Groq documents organization-level rate limits and model-specific limits;
projects can impose additional restrictions. Two keys in the same organization
do not create fresh organization quota. Track the actual limiting scope, and
only use independently entitled routes when available. See
[rate limits](https://console.groq.com/docs/rate-limits) and
[projects](https://console.groq.com/docs/projects).

Keep separate state for:

- **Credential:** invalid/revoked authentication; disable across all models.
- **Organization/project + model:** quota cooldown or explicit model restriction.
- **Organization/project:** spend/account restrictions that affect all models.
- **Credential + model:** a route-specific transient failure or unknown access.
- **Provider + model:** confirmed retirement or model-wide unavailability.
- **Provider:** correlated connection/service failures affecting all routes.

Use in-memory state for the existing single-process deployment. Restart loses
cooldowns; begin conservatively and respect new rate-limit responses. Multiple
workers/replicas require shared state or partitioned quotas before scaling.

## Failure decisions

Classify structured provider error codes and headers as well as HTTP status.
Never infer a universal model outage from one ambiguous 403/404 response.

| Result | State/action | What happens next |
|---|---|---|
| Valid output | Clear transient route failure state | Return immediately |
| 401 invalid/revoked credential | Disable this credential until secret/config replacement | Next key, same model; omit this key for fallback too |
| 403 explicit model restriction | Mark the documented org/project + model scope unavailable | Next eligible Qwen route; then fallback |
| 403 unknown cause | Quarantine affected route and report sanitized diagnostic | Next eligible route; do not globally disable the model |
| 429 | Honor `Retry-After` and applicable reset headers at the identified quota scope | Skip siblings sharing the exhausted scope; next eligible Qwen route, then fallback |
| Spend/account exhaustion | Disable affected account scope until reset or operator repair | Only routes outside that scope; otherwise degrade |
| Confirmed model retirement | Disable provider + model until configuration/catalog revalidation | Next model immediately |
| Ambiguous model-not-found | Mark only known affected route/scope unavailable | Try another eligible route; diagnostic for ID/access validation |
| Timeout, connection error, 408/409, 5xx | Short transient cooldown; count toward provider/model breakers | Next eligible route within the same deadline |
| Invalid payload, unsupported parameter, context overflow (400/413/422) | Request/configuration error; no key sweep | Degrade and fix task adapter; no blind replay to another model |
| Empty/malformed output or truncation (`finish_reason=length`) | Task output failure, not proof of a bad credential | Mark model unsuitable for this request; try fallback once per eligible route policy |
| `INSUFFICIENT`, explicit refusal, or inadequate source abstract | Valid terminal abstention | Omit generated section; do not shop for a different answer |
| Caller cancellation | Cancel outstanding async transport and release reservation | No retry and no false health penalty |

For malformed output, further keys on the same model do not repair a prompt or
format problem. This is an explicit exception to transport failover ordering.
Never quarantine a key globally because one model produced a bad answer.

If a 429's exact scope is unknown, conservatively cool the organization/model
group (all keys in the pool if membership is unknown); broaden to all models
when the provider identifies an account-wide constraint. No sleeping through a
long reset inside a user request. Preserve provider minimum wait times; add
jitter after, not before, that minimum. Without reset information, start with a
30-second cooldown and increase on repeated failures, capped at five minutes.
These defaults must be tuned against measured traffic.

## Bounded execution and recovery

1. Check input eligibility and valid cache entries before reserving any route.
2. Create one monotonic deadline for the entire generation operation, including
   queueing and all attempts: initially 2s rewrite, 4s summary, 8s explanation.
   Metadata retrieval remains outside this generation budget.
3. Iterate models in preference order. Within each model, choose healthy keys
   with available capacity, preferring lower in-flight load and rotating ties.
   The last successful fallback model must not permanently displace Qwen.
4. Atomically reserve quota/concurrency, then release the state lock before I/O.
   An exhausted shared scope can exclude several keys without calling them.
5. Use `AsyncGroq` with `max_retries=0`; the router alone owns retries. Clamp
   per-attempt timeout to remaining time and enforce the outer async deadline.
   Cancelling an executor future does not stop its synchronous network call,
   which is why the current thread wrapper is insufficient here.
6. Attempt each `(credential, model)` at most once per request. With N unique
   keys and two models there are at most 2N attempts, often far fewer. Bound
   concurrency and token reservations across tasks as well as per credential.
7. Stop for valid output, abstention, request error, cancellation, deadline, or
   exhaustion. Return typed status internally; preserve public graceful results.

Do not promise every key or the fallback will be contacted in two seconds.
Fast authentication/quota failures permit rotation; slow timeouts may consume
the budget. Record `deadline_exhausted` separately from `all_routes_unavailable`.
Per-attempt slices should be tuned from measured latency, not made so short
that healthy Qwen responses repeatedly time out.

Initial transient breaker policy: cool a failed route for 30 seconds; after
three consecutive transient failures, open the model breaker. Failures across
different models with transport or service symptoms can open the provider
breaker; do not sweep 2N routes
during an obvious provider outage. Require evidence across models before making
a model-specific failure provider-wide. After expiry allow one atomic probe;
other requests use an eligible route or degrade. Success closes the breaker;
failure increases its cooldown with jitter up to five minutes. Successful
requests reset consecutive failure counts for their scope. Rate-limit resets
and permanent credential errors use their own policies above.

## Output correctness, cache, and observability

- Give each model a tested parameter adapter. Do not copy Qwen reasoning
  options into another model's request. For these short tasks use a supported
  low-latency mode and expose only final answer content; include reasoning
  token usage in budgets when the API reports it. Validate against the exact
  model API before selecting parameters.
- Keep source-only prompts and abstention behavior. Validate rewrite length,
  nonempty summaries, and explanation output; do not render reasoning blocks.
  Escape generated text before applying the allowed bold formatting in search
  summaries. Output shape checks do not establish factual accuracy.
- Return an internal result containing text, requested/actual model, credential
  alias, outcome, attempts, latency, and usage. Public wrappers can retain their
  existing return types, but the explanation route must receive provenance.
- Key explanation cache entries by input content (including title), prompt
  version, and **actual** producing model. At lookup check accepted models in
  priority order; a valid cached fallback may be reused without another API call.
  Store fallback output under the fallback identity, never under Qwen. Changing
  allowed models or prompt version must invalidate incompatible entries.
- Coalesce identical concurrent explanation requests into one generation. Keep
  failures out of the content cache; routing cooldowns prevent repeated outage
  calls. Define waiter cancellation so one departing reader does not cancel work
  still awaited by another reader.
- Log sanitized alias, task, model, attempt, error category, cooldown scope,
  latency, and final reason. Never log keys, authorization headers, raw provider
  exception bodies, prompts, or abstracts. Record cache hits, fallback share,
  deadlines, successful recoveries, request/token usage, and in-flight counts.
- Keep `/healthz` cheap: summarize cached availability counts, not live paid
  completions per health probe. Detailed account diagnostics belong in operator
  tooling, not a public health response.

## Next steps, in order

1. **Verify access and establish the model pair.** Inventory credential aliases
   and org/project membership without exposing values. Query model availability
   once per relevant scope and run one short completion per intended route;
   catalog presence alone does not prove completion permission. Record exact
   IDs, access, limits, and date. Keep this outside startup and the request path.
2. **Evaluate task quality.** Use at least 30 held-out cases per task: casual
   queries/technical terms; paper sets requiring synthesis; full/truncated/
   ambiguous abstracts. Compare Qwen and GPT-OSS 20B on factual support, intent
   preservation, abstention, formatting, p50/p95 latency, and total token cost.
   Human-review unsupported claims. Freeze task-specific ranking and adapters
   before rollout; treat this small set as a release gate, not a universal ranking.
3. **Implement the shared router and configuration.** Add `app/llm_router.py`,
   settings in `app/config.py`, documented examples in `.env.example`, and
   mocked router tests. Use the existing Groq dependency. Preserve the legacy
   single-key configuration and keep failover opt-in initially.
4. **Integrate all three callers and cache provenance.** Update `groq_svc.py`,
   the paper explanation route, and DB helpers if required; manage async client
   shutdown in application lifespan. Test output escaping and cancellation.
5. **Validate failure behavior and canary.** Run targeted tests, then the offline
   suite against scratch storage. Inject failures locally; do not deliberately
   exhaust live quotas. Run a bounded smoke test on intended accounts. Enable
   explanations first, then summaries, then latency-sensitive rewrites. Compare
   fallback frequency, completion quality, cost, and latency with the baseline.
6. **Record and release.** Update `CURRENT-STATE.md`, README configuration, and
   Doc 06 when behavior ships. Record tests and deployment separately. Rollback
   disables failover or uses the last verified available single model; it must
   not blindly restore a retired Llama ID. Core search works with generation off.

## Acceptance cases

- Qwen/A fails, Qwen/B succeeds: no fallback call occurs.
- All eligible Qwen routes fail: fallback/A is attempted before fallback/B;
  selection within a model may rotate across requests, never interleave models.
- A has invalid credentials: skip A for every remaining model and later request.
- A and B share an exhausted org/model quota: B is skipped without an API call;
  C in an independently available scope can still serve Qwen.
- Model-specific permission failure leaves other models usable; account-wide
  failure excludes all affected model routes. Ambiguous errors stay scoped.
- A confirmed retired Qwen ID never causes a sweep through every credential.
- Fake-clock tests prove deadlines include queueing, SDK retries are disabled,
  attempts do not repeat, cooldowns expire correctly, and cancellation stops work.
- Concurrent calls cannot all take the recovery probe or oversubscribe the local
  concurrency limit; unrelated keys are not blocked behind a network-held lock.
- Invalid payloads and `INSUFFICIENT` are terminal; malformed output follows the
  explicit quality-failure policy without marking credentials invalid.
- Cached fallback output records its real model, reuses correctly, and cannot
  collide with Qwen output or a changed title/abstract/prompt.
- Missing keys/SDK, complete provider outage, and exhausted time budgets preserve
  original search and empty optional sections without request errors or secret logs.

## Local validation record — 2026-10-26

Final focused router/service/cache regression tests passed (86 passed, one
skipped), including provider reset timing. Compilation and `git diff --check`
passed. Before that final quota-reset refinement, the full non-live/non-browser
suite passed 767 tests, with four skips and 27
deselections, using:

```bash
METADATA_SIDECAR_PATH=/tmp/researchit-no-sidecar.sqlite \
  .venv/bin/python -m pytest tests/ -q -m 'not live and not browser' \
  --deselect tests/test_net_deadline.py::test_deadline_interrupts_a_read_the_socket_timeout_would_allow
```

The socket test was attempted in the first full run and failed during setup
because localhost binding is prohibited here. Four existing hybrid-search
tests initially picked up real FTS results from the local sidecar despite
mocking retrieval; the isolated run above excludes that corpus. User storage
is isolated by existing test fixtures and Turso replication is disabled.

Live catalog probing failed with `APIConnectionError` and no HTTP status.
No completion, quality evaluation, production enablement, or deployment was
performed. This is local correctness evidence, not a provider availability
or model-quality result.


## Release integration validation

The release candidate is based on cached main `cf44d2646e`, not the older
`feat/oai-daily-harvest` working branch. Main's model ranking, single-key chain,
AI usage logging, rate-limit windows, full-abstract repair, and redesigned
paper routes are preserved. The new router defaults to the first two
`GROQ_MODELS` (Qwen followed by GPT-OSS 20B).

Against that integrated tree, the non-live/non-browser suite passed **1,427
tests**, with four skips and 64 deselections including the localhost socket
test prohibited by this environment. After the final usage-counter adjustment,
the focused router/model-chain/usage/cache/health suites passed **131 tests**,
with one skip. The real Groq SDK was exercised through an HTTP mock transport
to verify request options, async response parsing, and rate-limit headers.
No live Groq availability or deployment success is claimed.

GitHub DNS lookup failed during the remote check. Publication and production
activation remain pending. The default-off flag is retained; after publication
and successful live route validation, set `GROQ_FAILOVER_ENABLED=1` and start
with `GROQ_FAILOVER_TASKS=explain` before expanding to summary and rewrite.
