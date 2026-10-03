# §4.1 generation evaluation

## Run

| field | value |
|---|---|
| `backend` | openai |
| `answer_mode` | single_call |
| `shuffle_trajectory` | False |
| `scenario_limit` | 0 |
| `question_limit` | 200 |
| `seed` | 42 |
| `dump` | reports/generation_eval.jsonl |
| `progress_every` | 50 |
| `generator_model` | `gpt-4o` |
| `td_scoring` | `exact_match` |
| `exclude_unanswerable` | True |

## Accuracy by answer type

| method | TF | MC | SA (NU) | TD (DE) | overall (micro) | overall (macro) |
|---|---|---|---|---|---|---|
| `okh_rag` | 100.0% | 64.5% | 49.2% | 14.0% | 48.3% | 56.9% |

## Denominators and exclusions

| method | questions | scorable | unanswerable | SA gold not numeric | empty context | non-conforming |
|---|---|---|---|---|---|---|
| `okh_rag` | 200 | 176 | 13 | 11 | 0 | 26 |

## Per-format denominators

| method | DE (scorable/total) | MC (scorable/total) | NU (scorable/total) | TF (scorable/total) |
|---|---|---|---|---|
| `okh_rag` | 43/46 | 62/62 | 63/84 | 8/8 |

## How to read this

* **SA is the only column with a large unscorable share.** 39.58% of the release's NU-tagged golds are not a single numeric value (a coordinate pair, a date, a per-horizon mapping, or a placeholder), so no tolerance test exists for them. They leave the denominator and are counted above rather than scored as wrong — see `src/okh_rag/evaluation/answers.py`.
* **The four methods share one question set and one generator.** A column that is high because it was scored on fewer questions is visible in the denominators table, not hidden in the mean.
* **`overall (micro)` is Table 1's "Overall".** It is not the macro-average of the four columns; both are reported because §4.1's own table is micro while its column layout suggests macro.
