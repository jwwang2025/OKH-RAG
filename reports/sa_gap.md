# Why the SA column was 0.082 instead of 0.452 — and the fix

**Short version.** CyPortQA asks the *same question text* once per horizon block of a scenario
(`T-12`, `T-24`, `T-36`, `T-48` …), and the gold answer is a function of the block. Retrieval was
scoped to the *scenario* only. Because the query text is byte-identical across blocks, every block
received the **same trajectory**, so at most one of the N blocks could be answered correctly. SA was
the column that paid for it, because SA is the only column whose gold is a number that differs block
to block.

Scoping the candidate universe to the question's own block takes SA from **0.082 to 0.633** and micro
from **0.331 to 0.523** (paper: 0.452 / 0.534).

---

## 1. Symptom

`reports/generation_eval.md` before the fix:

| | TF | MC | SA | TD | Overall |
|---|---:|---:|---:|---:|---:|
| paper | 0.833 | 0.652 | **0.452** | 0.441 | 0.534 |
| reproduced | 0.583 | 0.550 | **0.082** | 0.196 | 0.331 |

SA was off by 5.5× while the other three columns were off by 1.2–2.2×. A column-specific gap that
large is not explained by the global causes (order-term scaling, sample size) — those move all four
columns together.

## 2. The question set is block-duplicated

Over the 200-question sample (`scripts/diagnose_sa.py`):

```
questions: 200   distinct texts: 56   duplicated texts: 48
questions sitting in a duplicated group: 192
group sizes: {4: 48}
duplicated questions per format: {'DE': 52, 'MC': 52, 'NU': 76, 'TF': 12}
```

**All 76 SA questions sit in a group of 4 with byte-identical question text.** Example, `Q11`:

```
ALEX_2022_Canaveral|Q11|-|T-12   "…cumulative probability of 34-kt wind arrival … within the next 5 days?"   gold '24%'
ALEX_2022_Canaveral|Q11|-|T-24   (identical text)                                                            gold '18%'
ALEX_2022_Canaveral|Q11|-|T-36   (identical text)                                                            gold '15%'
ALEX_2022_Canaveral|Q11|-|T-48   (identical text)                                                            gold '15%'
```

The block is part of the question's *identity* (`QARecord.qa_id` includes `T-{horizon}`), and the
gold is `{senario.time.<key>.…}` — resolved at that block — but it is not part of the *text*.

## 3. The retriever never saw the block

`QARecord.group_id` is the scenario alone:

```python
@property
def group_id(self) -> str:
    return self.scenario_id
```

and `run_generation.py` called

```python
retriever.retrieve(record.question, group_id=record.group_id, query_vector=query_vector)
```

— no horizon. So the candidate universe for all four blocks was the whole 184-hyperedge scenario, and
with an identical query embedding the beam produced an identical result. Measured, for `Q11`:

```
gold= 24%  block=T-12   own-block steps=10/24  horizon mix={12: 10, 24: 6, 36: 4, 48: 4}
gold= 18%  block=T-24   own-block steps= 6/24  horizon mix={12: 10, 24: 6, 36: 4, 48: 4}
gold= 15%  block=T-36   own-block steps= 4/24  horizon mix={12: 10, 24: 6, 36: 4, 48: 4}
gold= 15%  block=T-48   own-block steps= 4/24  horizon mix={12: 10, 24: 6, 36: 4, 48: 4}
```

The horizon mix is **literally the same list** for all four. The model therefore answered all four with
one number — `18` for `Q11`/`Q15`, `12` for `Q21`/`Q22`/`Q28`, `7` for `Q8` — and got credit only on
the block that happened to match the retrieved evidence:

```
gold '24%' → pred 18   gold '18%' → pred 18 ✓   gold '15%' → pred 18   gold '15%' → pred 18
gold '18'  → pred 12   gold '30'  → pred 12    gold '42'  → pred 12    gold '36'  → pred 12
```

**Note the internal inconsistency this exposes:** `gold_for_record` in
`src/okh_rag/evaluation/oracle.py` already resolves evidence at `record.context.horizon_hours`. The
*evaluator* knew the block; the *retriever* did not. The two halves of the pipeline were answering
different questions.

## 4. Measured effect of scoping

Oracle gold-hyperedge recall over the 200-question sample, three candidate universes
(`reports/sa_horizon_diagnosis.json`):

| fmt | n | baseline (whole group) | `own_block` | `own_block_plus_cross` |
|---|---:|---:|---:|---:|
| NU (SA) | 76 | **0.083** | **0.507** | 0.314 |
| MC | 52 | 0.142 | 0.482 | 0.362 |
| DE (TD) | 52 | 0.204 | 0.617 | 0.288 |
| TF | 12 | 0.083 | 0.306 | 0.250 |
| ALL | 192 | 0.132 | **0.518** | 0.316 |

`own_block_plus_cross` is worse: the cross-horizon change edges carry values from *every* block, so
they reintroduce the ambiguity the scoping removes. `own_block` is the universe the gold definition
already assumes.

## 5. The fix

* `configs/base.yaml` — new key `retrieval.horizon_scoping`, default `"own_block"`
  (alternatives `"off"`, `"own_block_plus_cross"`).
* `src/okh_rag/config.py` — `HORIZON_SCOPING_MODES` + `RetrievalConfig.horizon_scoping` + validation.
* `src/okh_rag/retrieval/retriever.py` — `_HorizonIndex` (built once, `(group, horizon) -> ids`, plus a
  `cross` bucket for the block-less §A.2 step-4 edges) and a new `horizon_hours` argument on
  `OrderAwareRetriever.retrieve`, intersected with any caller-supplied `allowed_ids`.
* `scripts/run_generation.py`, `scripts/diagnose_generation.py` — pass
  `horizon_hours=record.context.horizon_hours`.

Scoping falls back to the whole group when the block is unknown or empty, so it never silently yields
an empty pool. Full test suite passes (326 tests).

## 6. After the fix

`scripts/run_generation.py --backend openai --retrieve 200`:

| | TF | MC | SA | TD | Overall |
|---|---:|---:|---:|---:|---:|
| paper | 0.833 | 0.652 | 0.452 | 0.441 | 0.534 |
| before | 0.583 | 0.550 | **0.082** | 0.196 | 0.331 |
| after | 0.583 | 0.733 | **0.633** | 0.157 | **0.523** |

and the mechanism is visible in the per-question split: SA accuracy is now **87.5% (28/32) when the
gold value is in the retrieved evidence** and 11.8% when it is not — before the fix the evidence itself
was wrong, so no amount of reading could have helped.

Two ablations, same 200 questions (`scripts/diagnose_generation.py`):

| variant | TF | MC | SA | TD | micro |
|---|---:|---:|---:|---:|---:|
| `learned` (default) + scoping | 0.583 | 0.700 | 0.612 | 0.137 | 0.500 |
| `none` (pure relevance) + scoping | 1.000 | 0.850 | 0.837 | 0.314 | 0.698 |

## 7. What is left, and why SA no longer "sits far off"

SA is now **above** the paper's 0.452 on the tolerance-scorable subset. Two honest caveats:

* **Denominator convention.** 27 of the 76 SA questions still leave the denominator (26 non-numeric
  golds — 12 dates, 8 coordinate pairs, 4 per-horizon mappings, 2 `N/A` — plus 1 unanswerable). On the
  full 76 the same run scores **31/76 = 0.408**, and the paper's 0.452 lies between 0.408 and 0.633.
  Which convention the paper used is not stated; `src/okh_rag/evaluation/answers.py` documents the
  choice it makes and why.
* **Sample.** 200 questions, 2 scenarios, 1 storm, 644 hyperedges, SA resting on 49 scorable items.
  Its 95% CI is roughly ±14 points — the paper's 0.452 is inside it under either convention.

The remaining gap in the table is **TD (0.157 vs 0.441)**, which is cause 2 of
`reports/reproduction_gap.md`: `td_scoring: exact_match` scores verbatim copying where §4.1 specifies
a GPT-4o semantic judge. TD is the one column scoping did not fix, and it is the one column that is
not retrieval-limited.

---

## 8. Measured on §4.1's own population

Section 6's numbers came from 200 questions that — because `select_records` sorted the scenario keys and
truncated — all sat inside **2 scenarios of one storm**. That is a different population from §4.1's, and
`reports/reproduction_gap.md` cause 3 says so. The sampler now draws a seeded random sample of the real
population, so the same 200 questions span **156 scenarios, 48 storms, 42 templates and all 10 horizon
blocks**, with a type mix that tracks the release:

```
TF 8 (4.00%)  vs paper 5812 (4.96%)
MC 62 (31.00%) vs paper 37813 (32.27%)
NU 84 (42.00%) vs paper 49459 (42.21%)
DE 46 (23.00%) vs paper 24094 (20.56%)
```

and Table 1 reads:

| | TF | MC | SA | TD | Overall |
|---|---:|---:|---:|---:|---:|
| paper | 0.833 | 0.652 | 0.452 | 0.441 | 0.534 |
| §4.1 population, n=200 | 1.000 (n=8) | **0.645** | **0.492** | 0.140 | 0.483 |

**SA 0.492 against the paper's 0.452, MC 0.645 against 0.652** — both inside the sampling interval at
n=63 and n=62. TF's n=8 carries no information. TD is still the outlier, and it is still cause 2.

So the SA column was never far off because of scoring or tolerance. It was 0.082 because retrieval could
not tell horizon blocks apart (§1–§5) *and* because the 200 questions it was measured on came from one
storm, where every block-duplicated question collapses onto the same four golds (§7).
