# §4.1 generation evaluation

> **These are not §4.1's numbers.** The backend is `lexical`: a deterministic token-overlap embedder and a stub generator that never reads the evidence. The run checks that the pipeline is wired — four methods, one generator, one question set, one scorer — and nothing about accuracy. Re-run with `--backend openai` to reproduce §4.1.

## Run

| field | value |
|---|---|
| `backend` | lexical |
| `answer_mode` | single_call |
| `sources` | okh_rag, hypergraph_rag, text_rag, graph_rag |
| `shuffle_trajectory` | False |
| `scenario_limit` | 8 |
| `question_limit` | 200 |
| `seed` | 42 |
| `generator_model` | `gpt-4o` |
| `td_scoring` | `exact_match` |
| `exclude_unanswerable` | True |

## Accuracy by answer type

| method | TF | MC | SA (NU) | TD (DE) | overall (micro) | overall (macro) |
|---|---|---|---|---|---|---|
| `okh_rag` | 75.0% | 31.7% | 4.1% | 0.0% | 17.4% | 27.7% |
| `hypergraph_rag` | 75.0% | 31.7% | 4.1% | 0.0% | 17.4% | 27.7% |
| `text_rag` | 75.0% | 31.7% | 4.1% | 0.0% | 17.4% | 27.7% |
| `graph_rag` | 75.0% | 31.7% | 4.1% | 0.0% | 17.4% | 27.7% |

## Denominators and exclusions

| method | questions | scorable | unanswerable | SA gold not numeric | empty context | non-conforming |
|---|---|---|---|---|---|---|
| `okh_rag` | 200 | 172 | 2 | 26 | 0 | 0 |
| `hypergraph_rag` | 200 | 172 | 2 | 26 | 0 | 0 |
| `text_rag` | 200 | 172 | 2 | 26 | 0 | 0 |
| `graph_rag` | 200 | 172 | 2 | 26 | 0 | 0 |

## Per-format denominators

| method | DE (scorable/total) | MC (scorable/total) | NU (scorable/total) | TF (scorable/total) |
|---|---|---|---|---|
| `okh_rag` | 51/52 | 60/60 | 49/76 | 12/12 |
| `hypergraph_rag` | 51/52 | 60/60 | 49/76 | 12/12 |
| `text_rag` | 51/52 | 60/60 | 49/76 | 12/12 |
| `graph_rag` | 51/52 | 60/60 | 49/76 | 12/12 |

## How to read this

* **SA is the only column with a large unscorable share.** 39.58% of the release's NU-tagged golds are not a single numeric value (a coordinate pair, a date, a per-horizon mapping, or a placeholder), so no tolerance test exists for them. They leave the denominator and are counted above rather than scored as wrong — see `src/okh_rag/evaluation/answers.py`.
* **The four methods share one question set and one generator.** A column that is high because it was scored on fewer questions is visible in the denominators table, not hidden in the mean.
* **`overall (micro)` is Table 1's "Overall".** It is not the macro-average of the four columns; both are reported because §4.1's own table is micro while its column layout suggests macro.
