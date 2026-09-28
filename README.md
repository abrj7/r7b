# r7b

**A learned router that decides, per query, whether Command R7B is good enough or whether it needs Command A — and an honest accounting of what that saves and what it costs.**

Site: [abdullahr.com/r7b](https://abdullahr.com/r7b/) · everything below reproduces with `uv run r7b run`.

> **The committed results are from mock mode.** No Cohere key was available when this was built, so every number in `runs/mock/` and on the site comes from a deterministic simulator of the two models (see [Mock mode](#mock-mode)). The pipeline, splits, metrics, and site are the real thing; the numbers are placeholders until someone runs [it for real](#running-it-for-real). The site says so up front.

## Why

Command A output costs **$10.00 / 1M tokens**. Command R7B output costs **$0.15 / 1M**. That is a ~67× gap. For many queries — "what's the capital of Australia", "convert 3 km to miles", most multiple-choice trivia — the 7B model gets it right, and paying for Command A buys nothing. For others (multi-step word problems, careful constrained writing), R7B falls over.

A router sits in front of both models and predicts, from the query text alone and before either model runs, whether R7B will do as well as Command A. If it's confident, send it to R7B; otherwise escalate. The question this repo answers is not "can you build one" (you can) but **"how much does it actually save, at what quality cost, compared with the dumb alternatives?"** — random routing at the same spend, and a perfect oracle.

## Results (mock)

Test split, n = 76, held out from everything, threshold picked before test was touched:

> **18.4% cost saved at 94.9% of Command A quality**, escalating 74% of queries to Command A.
> Bootstrap 95% CI: cost saved 9.8–28.7%, quality retained 88.1–100%.

And the part that matters:

| policy (test) | cost saved | quality retained |
|---|---:|---:|
| always Command A | 0% | 100% |
| **router, ≥95% target** | **18.4%** | **94.9%** |
| random routing, same spend | 18.4% | 95.2% (CI 90.7–99.1%) |
| router, ≥90% target | 44.7% | 84.5% |
| random routing, same spend | 44.7% | 88.3% |
| always R7B | 99.0% | 74.1% |
| oracle (perfect labels) | 65.5% | 99.2% |

**On the simulated data the router is not beating random routing.** Its average performance gap recovered (APGR, the RouteLLM metric; random = 0.5) is 0.49 on test; the best cross-validated classifier has an AUC of 0.64. That's not a bug in the pipeline — the simulator's difficulty model has a hidden-noise term that no text feature can see, which caps any router's AUC at ≈0.79 even with perfect features, and 228 training queries is not enough to get close to that ceiling. I left the simulator honest rather than tuning it until the router looked good.

The oracle row is the real finding: **if** you could tell which queries R7B handles, you'd save 65% at 99% quality, because R7B matches Command A on ~74% of this mix. The headroom is large; the open question is whether real Embed v4 embeddings of real queries carry the signal the mock can't. That is what the live run answers.

What the mock numbers are calibrated to (per-source accuracy, R7B / Command A): GSM8K 77% / 94%, MMLU 62% / 85%, open-domain QA 36% / 52% (strict exact match), open-ended judge score 6.3 / 8.0 out of 10.

All nine classifier × feature-set combinations, with CV AUC, val/test AUC, APGR and Brier score, are in `runs/mock/results.json` and on the site under "Models".

## Method

### Queries (380)

| source | n | how it's graded |
|---|---:|---|
| GSM8K (test) | 100 | final number, exact numeric match |
| MMLU (test, all subjects) | 120 | extracted letter A–D |
| Natural Questions open (val) | 80 | normalized whole-token match against any alias |
| hand-written open-ended | 80 | Command A as judge, 1–10 rubric |

The open-ended prompts (`data/open_prompts.jsonl`) cover writing with hard constraints, explanation, code, reasoning, planning, analysis and knowledge, at three difficulty levels. Everything is sampled with a fixed seed and committed as `data/queries.jsonl`, so the set is stable even if the HF datasets move.

### Answers and labels

Both models answer every query at temperature 0 with a fixed seed, with the same system prompt asking for a final `Answer:` line. Each response gets a quality in [0, 1]: 0/1 for the verifiable sources, `(score − 1) / 9` for judged ones.

A query is labeled **"R7B good enough"** if `q_R7B ≥ q_A`. For judged prompts there's a tolerance of one rubric point (1/9), because the judge's score is noisy at that resolution and treating a 7 vs. 8 as a routing failure would mostly be training on judge noise.

### The judge

Command A grades both responses **in one blinded pairwise call** — labeled "Response 1/2" with order set by a hash of the query, so neither model is always first — against a rubric that caps scores for material errors (≤5) and missed constraints (≤6), and must reply in JSON.

Before trusting it, it's checked against `data/judge_validation.jsonl`: 16 response pairs I scored by hand, chosen to include traps (a "50-word summary" that is exactly 50 words, a haiku that's 5-8-6 vs. one that's 5-7-5, confident wrong answers, verbose right ones). Reported: MAE, within-1 rate, Spearman ρ, pairwise-preference agreement and Cohen's κ, and — the one that matters for routing — how often the judge's scores produce the same good-enough label as mine. (Mock: MAE 0.38, Spearman 0.97, label agreement 100%. The mock judge is built to be close to the truth, so this is only a pipeline check until it runs live.)

### Features

- **Embed v4** (`embed-v4.0`, `input_type=classification`, 256 dims), batched 96 texts per call.
- **17 hand-crafted features** that cost nothing: length, line/number/sentence counts, answer-choice lines, math and code character ratios, reasoning words ("prove", "step", "why"), constraint words ("exactly", "at most", "must"), simple-lookup cues, and so on. They're implemented identically in Python and TypeScript so the site can run the router in the browser, and a shared fixture keeps the two in lockstep.

Three feature sets are compared: hand-crafted only, embeddings only, and both.

### Classifiers and protocol

Logistic regression, histogram gradient boosting and a small MLP, each with a small grid.

1. Split 60/20/20 (228/76/76), stratified by source × label.
2. 5-fold grid search on train. **Model and feature set are chosen by cross-validated AUC**, not by val — with 76 val queries, picking the best of nine by val AUC is mostly picking noise.
3. The routing **threshold** is chosen on out-of-fold predictions over train + val: the lowest-cost threshold whose quality retained is ≥ the target (90/95/98%).
4. The chosen model is refit on train + val, and **test is scored once**.
5. Test numbers get bootstrap CIs (1,000 resamples).

### Baselines

- **always R7B / always Command A** — the two ends.
- **random at matched cost** — escalate each query with probability *p*, with *p* solved so expected spend equals the router's; computed analytically, with a bootstrap CI. This is the baseline a router has to beat. A router that can't beat a coin flip at the same budget isn't routing.
- **oracle** — escalate exactly the queries labeled "needs Command A". The ceiling.

### Cost

Costs use measured token counts (`usage.billed_units` from each response) × list price, and include the router's own embedding call per query.

| model | input / 1M | output / 1M | source |
|---|---:|---:|---|
| Command R7B (`command-r7b-12-2024`) | $0.0375 | $0.15 | cohere.com/pricing |
| Command A (`command-a-03-2025`) | $2.50 | $10.00 | OpenRouter `cohere` provider listing |
| Embed v4 | $0.12 (text) | — | cohere.com/pricing |

Checked 2026-09-27. Command A is no longer listed on cohere.com/pricing (the page now leads with Command A+), so its price comes from Cohere's own listing on OpenRouter. Prices live in `src/r7b/costs.py`; `r7b budget` prints them.

## Fitting in a free trial key

Trial keys are free but capped: **1,000 API calls per month** across all endpoints, **20 requests/minute** per chat model, 2,000 embed inputs/minute. The dataset is sized to fit in one month with room to spare:

| calls | count |
|---|---:|
| R7B answers | 380 |
| Command A answers | 380 |
| judge calls (80 open-ended + 16 validation) | 96 |
| embed calls (380 texts / 96 per batch) | 4 |
| **total** | **860** |
| monthly cap | 1,000 |
| held in reserve | 40 |
| **headroom** | **100** |

The guardrails:

- **Every response is cached to disk**, keyed by a hash of the exact request (`cache/live/`). Re-running the pipeline, changing a classifier, or changing the site costs zero calls.
- **A call ledger** (`cache/live/ledger.jsonl`) records every call that leaves the machine and **refuses to send** one that would dip into the reserve (`R7B_MONTHLY_CALL_CAP`, `R7B_CALL_RESERVE` in `.env`).
- **A client-side rate limiter** keeps each model at 90% of its per-minute limit, with backoff on 429s.
- Embeds are cached per text and only misses are sent, in batches of 96.

Running this on a production key instead would cost about **$0.88** in total (answers $0.74, judge $0.14, embeddings $0.003).

## Mock mode

With no `COHERE_API_KEY`, everything runs against `src/r7b/mock.py`, a deterministic simulator:

- Each query gets a latent difficulty from visible cues (length, numbers, reasoning and constraint words, hard MMLU subjects) plus a hidden per-query term that no feature can see.
- Each model has a per-source ability; P(correct) = σ(2.2 × (ability − difficulty)), with the two models' luck correlated (ρ = 0.65) because hard questions are hard for both.
- Token counts are drawn per model and source; responses are placeholder text marked `[simulated response: mock mode, no API call was made]`.
- The mock judge recovers the true score and adds noise; embeddings are LSA (TF-IDF → SVD) on the query set, a stand-in for Embed v4.

Everything is a function of hashes of the query and model, so the same code gives the same numbers on any machine. The mock cache is namespaced by a hash of the simulator's source, so editing the simulator never serves stale answers.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Node 22+.

```bash
uv sync
uv run r7b budget          # the call plan against the monthly cap
uv run r7b run --mock      # full pipeline on the simulator (~20 s), exports site data
uv run pytest              # metrics, routing, cost, budget, grading, features, pipeline

cd site
npm install
npm run dev                # http://127.0.0.1:4719/r7b/
npm test                   # TS/Python parity checks for features and headline numbers
npm run build              # static export to site/dist
```

`uv run r7b data` rebuilds `data/queries.jsonl` from the source datasets (only needed if you change the sample).

### Running it for real

```bash
cp .env.example .env       # then set COHERE_API_KEY=...
uv run r7b budget          # confirm the plan fits what's left this month
uv run r7b check           # 3 calls: proves the key and all three model names work
uv run r7b run --live      # ~860 calls; resumable, everything is cached
```

`r7b check` exists so a bad key or a retired model name costs 3 calls instead of a failed run. Its two chat calls are the same requests the full run makes first, so they're reused from cache.

At 20 requests/minute per model with two models in parallel, the answer pass takes roughly 20 minutes and judging another 5. If it's interrupted, re-run the same command: cached calls are free and the ledger picks up where it left off. The run writes `runs/live/` and re-exports `site/public/data/`, and the site's mock banner disappears on its own. Commit `runs/live/`, `site/public/data/` and `cache/live/` (the cached responses, so nobody has to spend the calls again) and push to `main` to redeploy.

After the first live run, read `runs/live/judge_validation.json` before trusting the open-ended labels. If the judge's label agreement with the hand scores is poor, the open-ended half of the results isn't meaningful.

## Deploying the site

`.github/workflows/deploy.yml` builds `site/` and publishes `site/dist` to GitHub Pages on every push to `main`. The site is built with base path `/r7b/`, so as a project page of `abrj7/r7b` it is served at `abdullahr.com/r7b` under the existing `abdullahr.com` Pages domain. No DNS changes needed. To go live, enable Pages with "GitHub Actions" as the source in the repo settings.

## Layout

```
src/r7b/
  data.py       query set: sampling from GSM8K / MMLU / NQ + the open prompts
  prompts.py    task instructions, token limits, judge rubric and template
  clients.py    cached, budgeted, rate-limited Cohere client (live or mock)
  mock.py       the deterministic simulator
  cache.py      content-addressed disk cache
  budget.py     monthly call ledger, rate limiter, call plan
  grading.py    answer extraction, graders, judge parsing, the labeling rule
  judge.py      blinded pairwise judging and judge validation
  collect.py    answers + judgments for every query
  features.py   hand-crafted features and embedding matrix
  train.py      splits, model search, threshold selection, test evaluation
  routing.py    threshold sweeps, Pareto frontier, baselines, APGR
  metrics.py    AUC, Brier, Spearman, kappa, bootstrap CIs
  costs.py      pricing and cost helpers
  export.py     JSON for the site
  cli.py        `r7b data | budget | run`
data/           committed query set, open prompts, hand-labeled judge pairs
runs/mock/      committed results the site is built from
site/           Vite + React + Tailwind site
tests/          pytest suite
```

## Limitations

- **The committed numbers are simulated.** They show the pipeline works and what the analysis looks like, not how well a router does on Cohere's models.
- **Small.** 380 queries, 76 in test. Confidence intervals are wide (cost saved 10–29% at the headline point), and differences between classifiers are mostly inside the noise. The trial key's monthly cap is the binding constraint; a production key would let this scale 10×.
- **The query mix sets the answer.** R7B is "good enough" on ~74% of this set, which is what makes the oracle look so good. A mix that's mostly hard reasoning or mostly lookup would move every number.
- **Binary labels throw away information.** "R7B scored 7, A scored 9" and "both scored 9" are different; a regression on the quality gap, or a cost-weighted loss, might route better.
- **LLM-as-judge.** Command A judges its own outputs against R7B's. Blinding helps, and it's validated on 16 hand-scored pairs, but self-preference bias is possible and 16 pairs is a sanity check, not a calibration.
- **Exact match undercounts NQ.** Strict alias matching marks some correct paraphrases wrong for both models, which mostly adds noise to the labels.
- **One sample per query.** Temperature 0 with a fixed seed makes answers reproducible, not representative of sampling variance.
- **Prices move.** Command A's price is taken from a third-party listing of Cohere's own endpoint because Cohere's page no longer lists it.
