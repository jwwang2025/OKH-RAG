# Why the reproduction's Table 1 is far below the paper's

Paper, Table 1, `OKH-RAG (ours)`:

| TF | MC | SA | TD | Overall |
|---|---|---|---|---|
| 0.833 | 0.652 | 0.452 | 0.441 | 0.534 |

Reproduced at the time this report was written (`reports/generation_eval.md`):

| TF | MC | SA | TD | Overall (micro) |
|---|---|---|---|---|
| 0.583 | 0.550–0.583 | 0.082 | 0.196 | 0.314–0.326 |

**Update.** Cause 0 below was found after this report was written and has since been fixed; it is the
reason SA was 5.5× low, not 2× low like the other columns. With it fixed the same run gives
TF 0.583 / MC 0.733 / SA 0.633 / TD 0.157 / micro 0.523. See `reports/sa_gap.md`. Causes 1–5 were
measured **before** that fix and are kept as written; their absolute numbers are superseded, their
mechanisms are not.

Six causes, measured rather than guessed. Causes 0 and 2 account for most of the gap; cause 3 makes the
remaining columns untrustworthy; causes 4 and 5 are fidelity issues that would matter even if the rest
were fixed.

All numbers below come from `scripts/diagnose_generation.py` (per-question dump in
`reports/generation_diagnosis*.json`) and from `evaluation/oracle.py` scored against the reference graph.

---

## 0. Retrieval was not scoped to the question's horizon block — this is why SA was 0.082

CyPortQA asks the *same question text* once per horizon block (`T-12`, `T-24`, `T-36`, `T-48` …) with a
block-dependent gold. `QARecord.group_id` is the scenario alone and `retrieve()` was called without a
horizon, so all N blocks of a question shared one candidate universe — and since the query embedding is
byte-identical, they received the **same trajectory**. All 76 SA questions in the sample sit in such a
group of 4, so SA was capped near 1/4 before generation even started.

Measured: oracle gold-hyperedge recall 0.083 → **0.507** when `C` is drawn from the question's own
block. Fixed by `retrieval.horizon_scoping: "own_block"` plus `horizon_hours=` on `retrieve()`;
SA 0.082 → 0.633, micro 0.331 → 0.523. Full write-up in `reports/sa_gap.md`.

---

## 1. The order term swamps the relevance term — retrieval is at chance

This is the main cause, and it moves every column.

The gold hyperedges are reaching the candidate pool; they are just not surviving into the trajectory:

| variant | pool recall | trajectory recall | chance recall |
|---|---:|---:|---:|
| `learned` (the default, §4.1's config) | 0.983 | **0.073** | 0.052 |
| `heuristic` (§A.3 rules) | 0.983 | 0.248 | 0.052 |
| `none` (pure top-k relevance) | 0.983 | **0.466** | 0.052 |

`pool_recall` 0.983 and `anchor_recall` 0.923 say the evidence is *in* `C`. `trajectory_recall` 0.073
against a chance floor of 0.052 says the beam then picks it **barely more often than random**. The order
metric is inverted relative to §4.5's Table 2 (0.501 → 0.509 → 0.534): here learned < heuristic < none.

**Why.** Eq. 5 adds a bounded cosine sum to an unbounded log-probability sum, and the two are not on
comparable scales. Measured over a 150-element pool for one real query:

| quantity | p95 − p5 spread | std |
|---|---:|---:|
| `Rel(e, q)` (the only query-dependent term) | 0.061 | 0.024 |
| `λ · log P_θ(e′ \| e)` (query-*independent*) | **1.640** | 0.505 |

`text-embedding-3-small` cosines on this corpus live in a narrow band (0.431–0.570), so relevance can
move a candidate by ~0.06. `log P_θ` has mean −3.87 per step and std 0.421; with λ = 1.2 that is ~±0.5
per step, **roughly 27× the discriminative range of the relevance term**. The winning trajectory's
breakdown makes it explicit: `relevance=3.82`, `order_coherence=−27.08`.

So Algorithm 1 is largely maximising `log P_θ` — "which 8-step chain does the transition model find
natural" — and the question barely enters the argmax.

**What it looks like in the answers.** Numeric questions are answered from a *neighbouring* horizon
block, which is exactly the failure order-aware retrieval is supposed to prevent:

```
gold '36' → predicted 12      gold '48' → predicted 12      gold '4' → predicted 2
gold '18' → predicted 12      gold '30' → predicted 12      gold '6 mph' → predicted 7
```

and 22 of 200 questions came back with `answer: null` — the model following the instruction "if the
evidence does not state the quantity, use null" — because the needed field never made it into the
trajectory.

**Measured effect on accuracy.** Rerunning the same 200 questions with `transition_variant=none`
(pure relevance, everything else identical):

| | TF | MC | SA | TD | Overall |
|---|---:|---:|---:|---:|---:|
| paper | 0.833 | 0.652 | 0.452 | 0.441 | 0.534 |
| reproduced, `learned` | 0.583 | 0.550 | 0.082 | 0.196 | 0.314 |
| reproduced, `none` | 0.750 | **0.650** | **0.388** | 0.196 | **0.448** |

MC lands on the paper's 0.652 almost exactly, SA recovers from 0.082 to 0.388, and micro goes
0.314 → 0.448. TD does not move at all — it is capped by cause 2, not by retrieval.

This is a faithful transcription of Eq. 5 as printed, not a coding slip. `scoring.py` and `beam.py`
match the paper, and the telescoping invariants are tested. The problem is that Eq. 5's two halves are
dimensionally incompatible at §4.1's stated `(λ, µ, ν, ρ) = (1.2, 0.3, 0.2, 0.5)`. Either the paper's
`P_θ` is not a raw per-pool log-probability, or its `Rel` is not a raw cosine, or the reported 0.534
was produced under a scaling this reproduction does not have. The `cosine_relevance` docstring shows
the mismatch was noticed ("the relevance term is added to log-probabilities of order `-log|C|`") but
it was never reconciled.

**Candidate fixes**, in order of how little they change the paper's intent:

1. Normalise the transition term to a comparable range — per-row softmax over `C` with a floor, or
   `log P_θ / log(1/|C|)`, or the bounded `P_θ` instead of `log P_θ`.
2. Make relevance the gate and the order terms the tie-break: restrict the beam's step-5 expansion
   `C \ β` to the top-*m* by `Rel`, then let λ, µ, ν, ρ order within it.
3. Lower λ. Note this is *not* the Table 2 `−λ` row (which removes order coherence entirely and costs
   only −0.002 in the paper); a λ that keeps the signal but not the scale is a different change.

---

## 2. TD is scored by exact match, not by §4.1's GPT-4o judge

§4.1: TD is scored by "an LLM-judged semantic score in [0, 1] using GPT-4o as the evaluator".
`configs/base.yaml` defaults to `td_scoring: "exact_match"`, and the run used the default.

That costs the TD column most of its value. Measured: **0.196**. The golds are verbatim NOAA phrases,
and the model answers with a shorter paraphrase that is frequently semantically right:

```
gold: 'South-central and southeast Florida coasts, including Brevard County (Port Canaveral),
       the Keys, and northwest Bahamas.'
pred: 'Central and south Florida coast'
```

Exact match scores that 0. A judge would score it 0.6–0.8, which is how the paper's 0.441 is reachable
at all. Two further facts pin the blame on the scorer rather than the retriever:

* With `transition_variant=none` the TD column does **not** move (0.196 → 0.196) while every other
  column rises sharply — TD is not retrieval-limited.
* Among the 51 scorable DE questions, all 10 hits are questions whose gold string is literally in the
  retrieved context, and all 41 misses split between "gold not retrieved" (27) and "gold retrieved but
  paraphrased" (14). The second group is pure scorer loss.

The justification in `base.yaml` (`[SIMPLIFICATION]`) is about the *gold*, not about the model's
answer: it observes that all 24,094 description golds are byte-identical to the field their template
declares, which is true, and concludes that a judge is unnecessary. That does not follow — it says
the gold is copyable, not that the model copies it. §5.2 of `data/CONTRACT.md` repeats the same
inference.

**Fix.** Set `evaluation.td_scoring: "llm_judge"` and pass a `judge` callable to `score_record`. The
protocol (`JudgeFn`) and the temperature config already exist; only the caller is missing. Expect TD
to land near 0.44 and micro to rise accordingly.

---

## 3. The sample is 200 questions from **2 scenarios** — one storm

Not 24. `select_records` expands `--scenario-limit` scenarios and then truncates the flattened list at
`--question-limit`, and scenarios hold 39–506 questions (mean 248), so 200 questions stop inside the
second one:

```
ALEX_2022_Canaveral_Port_District,_FL   160
ALEX_2022_Jacksonville,_FL               40
```

One storm, two ports, 644 hyperedges. Consequences:

* Denominators are tiny: TF **n=12**, NU n=49, DE n=51, MC n=60. TF's 0.583 is 7/12 — a single
  question moves the column by 8.3 points, and its 95% CI is roughly ±28 points. The paper's 0.833 is
  inside that interval, so TF tells you nothing either way.
* `pool_recall` 0.983 is an artifact of the small graph: `|C| = 150` is 23% of all 644 hyperedges, so
  almost everything is in the pool. On the full 248,652-hyperedge graph, `data/CONTRACT.md` §5.3
  measured pool recall at **0.578**. The reproduced run is not measuring the same retrieval problem the
  paper faced.
* The whole benchmark is 117,178 questions over 472 scenarios. 200 from 1 storm cannot support a
  five-column table.

**Fix.** Sample questions *per scenario* rather than truncating a flat list — cap `k` questions per
scenario and take all `scenario_limit` scenarios, or shuffle with a seeded RNG before truncating. Then
raise `n` until the columns stabilise; a few thousand stratified questions is the minimum for a table
whose SA column rests on ~50 scorable items.

---

## 4. The generator is a third-party relay with no structured output

`.env` points at `https://api.openlux.ai/v1` and pins `OKH_RAG_RESPONSE_FORMAT=object`. §A.8 requires
"Output is constrained to a strict JSON schema"; because the relay does not implement `json_schema`,
every call falls back to `response_format: json_object` with the schema injected as a prompt hint.

Measured cost: **27/200 = 13.5% of answers violated §A.8's contract** (22 `answer: null` on numeric
questions, 5 non-numeric). Those NU answers scored 0/10, against 4/39 for conforming ones. Under
`transition_variant=none` violations fell to 16/200, but the mechanism is the same.

Whether the relay's `gpt-4o` is GPT-4o is not verifiable from here, and §4.1's numbers are GPT-4o
numbers.

**Fix.** Point `OPENAI_BASE_URL` at an endpoint that supports `json_schema` and unset
`OKH_RAG_RESPONSE_FORMAT`, so the decoder enforces the contract instead of the prompt requesting it.

---

## 5. Secondary issues

* **Baselines are gone.** Commit `fabe893` removed them, so Table 1's other three rows
  (Text-RAG 0.287, GraphRAG 0.414, HyperGraphRAG 0.511) cannot be produced. The paper's headline is
  the +0.023 over HyperGraphRAG; without the baseline the reproduction cannot confirm or refute it.
* **SA's denominator.** 27 of 76 NU questions leave it (26 non-numeric gold, 1 unanswerable), matching
  the release-wide 39.6% documented in `data/CONTRACT.md` §5.4. Reported, but the SA column is computed
  on 49 questions.
* **Order-model leakage.** `P_θ` was trained on 377 of 472 scenarios (`artifacts/order_model.train_report.json`,
  seed 42); the evaluation scenarios may sit in the training split. Not a cause of the gap, but the
  reported number is not a held-out number.

---

## Bottom line

| fix | TF | MC | SA | TD | Overall |
|---|---:|---:|---:|---:|---:|
| paper | 0.833 | 0.652 | 0.452 | 0.441 | 0.534 |
| as run | 0.583 | 0.550 | 0.082 | 0.196 | 0.314 |
| **+ horizon-scoped retrieval** (shipped) | 0.583 | 0.733 | **0.633** | 0.157 | **0.523** |
| + `td_scoring: llm_judge` (expected) | ~0.58 | ~0.73 | ~0.63 | ~0.44 | ~0.60 |

The gap is not noise and not one bug. It is **(0) a retriever that could not tell horizon blocks
apart**, **(1) an order objective whose query-independent term is 27× stronger than its query-dependent
one**, and **(2) a TD scorer that measures verbatim copying where the paper measured semantics**.
Cause 0 is fixed and moves SA from 0.082 to 0.633; TD is now the only column still far off, and it is
the one column that is not retrieval-limited. The remaining shortfall is sample size — cause 3.

One caveat worth stating plainly: the fix in (1) is a departure from Eq. 5 as printed. That the paper's
own Table 2 hierarchy cannot be reproduced under the literal reading — while `none` beats `learned` by
6× on trajectory recall — is evidence that the published objective is underspecified, not that this
implementation misread it.
