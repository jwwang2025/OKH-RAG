# CyPortQA release audit

Data root: `data\raw\cyportqa`

Recomputed from the files on disk by `scripts/audit_cyportqa.py`. Every number below is
derived at run time; nothing is copied from a previous run.

## Corpus

| quantity | value | paper (§4.1) |
|---|---:|---|
| scenario keys in the release | 472 | — |
| horizon blocks | 2910 | 2,917 scenarios |
| knowledge groups | 472 | — |
| ports | 45 | 145 |
| storms | 69 | 90 |
| questions | 117178 | 117,178 |
| questions with an answer | 108841 | — |
| questions with a placeholder answer | 8337 | — |
| questions whose context omits the port | 31945 | — |

Answers by format: `{'DE': 24094, 'MC': 37813, 'NU': 49459, 'TF': 5812}`

Questions by stage: `{'S1': 84244, 'S2': 31502, 'S3': 1432}`

Horizons present: `[12, 24, 36, 48, 60, 72, 84, 96, 108, 120]` (T-120 .. T-12)

## Template audit

For each template, the shipped answers are compared against the value at the path the
template itself declares. Two rates are reported, and they answer different questions:

- `agreement` — of the records where a comparison was possible offline, how many agree with
  the declared field.
- `determinacy` — of those same records, how many have an answer the encoding determines
  *uniquely*. This is low for the ordinal-bucket templates, whose options overlap, so a
  value can satisfy two buckets at once. High agreement with low determinacy means the
  template is scored on a question the data does not settle.

The `unverifiable` column is the count of records whose score needs the §4.1 LLM judge. It is
zero everywhere, and `td-answers-are-verbatim-field-copies` explains why: every description
answer is a byte-identical copy of its source field, so the judge has nothing left to judge.

| template | fmt | semantics | stage | declared path | records | match | mismatch | indeterminate | missing | placeholder | unverifiable | agreement | determinacy |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | TF | lookup | S1 | `weather_observation.within_uncertainty_cone` | 2908 | 2908 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q2 | MC | lookup | S1 | `weather_observation.under_watch` | 2873 | 2870 | 3 | 0 | 0 | 0 | 0 | 0.999 | 1.000 |
| Q3 | MC | lookup | S1 | `weather_observation.under_warning` | 2902 | 2899 | 3 | 0 | 0 | 0 | 0 | 0.999 | 1.000 |
| Q4 | DE | lookup | S1 | `weather_observation.watch_coast` | 2909 | 2562 | 0 | 0 | 0 | 347 | 0 | 1.000 | 1.000 |
| Q5 | DE | lookup | S1 | `weather_observation.warning_coast` | 2909 | 2539 | 0 | 0 | 0 | 370 | 0 | 1.000 | 1.000 |
| Q6 | NU | lookup | S1 | `cyclone_condition_and_forecast.coordinates` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q7 | DE | lookup | S1 | `cyclone_condition_and_forecast.motion.direction` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q8 | NU | lookup | S1 | `cyclone_condition_and_forecast.motion.speed` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q9 | DE | lookup | S1 | `weather_observation.impacted_coast` | 2909 | 2909 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q10 | DE | lookup | S1 | `weather_observation.closest_wind_data_location` | 2909 | 2772 | 0 | 0 | 0 | 137 | 0 | 1.000 | 1.000 |
| Q11 | NU | lookup | S1 | `weather_observation.culmulative_wind_34kt` | 2909 | 2908 | 0 | 0 | 0 | 1 | 0 | 1.000 | 1.000 |
| Q12 | NU | lookup | S1 | `weather_observation.culmulative_wind_50kt` | 2908 | 2907 | 0 | 0 | 0 | 1 | 0 | 1.000 | 1.000 |
| Q13 | NU | lookup | S1 | `weather_observation.time_to_34kt_max` | 2908 | 2843 | 0 | 0 | 0 | 65 | 0 | 1.000 | 1.000 |
| Q14 | NU | lookup | S1 | `weather_observation.time_to_50kt_max` | 2908 | 2586 | 0 | 0 | 0 | 322 | 0 | 1.000 | 1.000 |
| Q15 | NU | lookup | S1 | `weather_observation.34kt_max` | 2908 | 2907 | 0 | 0 | 0 | 1 | 0 | 1.000 | 1.000 |
| Q16 | MC | lookup | S1 | `port.name` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q17 | DE | lookup | S1 | `weather_observation.impacted_coast` | 2909 | 2909 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q18 | MC | lookup | S1 | `cyclone_condition_and_forecast.category` | 2851 | 2789 | 62 | 0 | 0 | 0 | 0 | 0.978 | 1.000 |
| Q19 | MC | lookup | S1 | `port.name` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q20 | DE | lookup | S1 | `weather_observation.impacted_coast` | 2909 | 2909 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q21 | NU | lookup | S1 | `cyclone_condition_and_forecast.hours_to_landfall` | 2910 | 2872 | 0 | 0 | 0 | 38 | 0 | 1.000 | 1.000 |
| Q22 | NU | lookup | S1 | `cyclone_condition_and_forecast.hours_to_offshore` | 2910 | 2836 | 0 | 0 | 0 | 74 | 0 | 1.000 | 1.000 |
| Q23 | MC | ordinal_bucket | S1 | `weather_observation.forecast_window` | 2909 | 2909 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q24 | MC | ordinal_bucket | S1 | `cyclone_condition_and_forecast.hours_of_strike` | 2910 | 2765 | 26 | 119 | 0 | 0 | 0 | 0.991 | 0.866 |
| Q25 | MC | ordinal_bucket | S1 | `weather_observation.time_to_34kt_max` | 2908 | 2402 | 133 | 373 | 0 | 0 | 0 | 0.948 | 0.860 |
| Q26 | TF | threshold_predicate | S1 | `weather_observation.34kt_max` | 2904 | 2903 | 0 | 1 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q27 | MC | ordinal_bucket | S1 | `weather_observation.time_to_34kt_non_zero` | 2908 | 2504 | 31 | 373 | 0 | 0 | 0 | 0.988 | 0.604 |
| Q28 | NU | lookup | S1 | `cyclone_condition_and_forecast.hours_to_landfall` | 2910 | 2872 | 0 | 0 | 0 | 38 | 0 | 1.000 | 1.000 |
| Q29 | MC | lookup | S1 | `weather_observation.first_hazard` | 2896 | 2896 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q30 | MC | lookup | S2 | `weather_observation.storm_surge_above_3_feet` | 2749 | 2731 | 18 | 0 | 0 | 0 | 0 | 0.993 | 1.000 |
| Q31 | MC | lookup | S2 | `weather_observation.rain_fall_above_4_inch` | 2567 | 2536 | 31 | 0 | 0 | 0 | 0 | 0.988 | 1.000 |
| Q32 | DE | lookup | S2 | `weather_observation.other_hazards` | 2908 | 2635 | 0 | 0 | 0 | 273 | 0 | 1.000 | 1.000 |
| Q33 | NU | lookup | S2 | `weather_observation.gale_probability` | 2908 | 2908 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q34 | NU | lookup | S2 | `impact_prediction.start_impact_date` | 2910 | 1576 | 0 | 0 | 0 | 1334 | 0 | 1.000 | 1.000 |
| Q35 | NU | lookup | S2 | `impact_prediction.max_impact_date` | 2910 | 1576 | 0 | 0 | 0 | 1334 | 0 | 1.000 | 1.000 |
| Q36 | NU | lookup | S2 | `impact_prediction.day_of_recover` | 2910 | 5 | 1571 | 0 | 0 | 1334 | 0 | 0.003 | 1.000 |
| Q37 | NU | lookup | S2 | `impact_prediction.day_of_impact` | 2910 | 2910 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q38 | NU | lookup | S2 | `impact_prediction.day_of_recover` | 2910 | 1576 | 0 | 0 | 0 | 1334 | 0 | 1.000 | 1.000 |
| Q39 | NU | lookup | S2 | `impact_prediction.max_impact` | 2910 | 1576 | 0 | 0 | 0 | 1334 | 0 | 1.000 | 1.000 |
| Q40 | MC | ordinal_bucket | S2 | `impact_prediction.total_impact_value` | 2910 | 1576 | 0 | 1334 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q41 | MC | lookup | S3 | `operation.now.Condition` | 117 | 117 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q42 | MC | lookup | S3 | `operation.<key>.Condition` | 493 | 493 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q43 | DE | lookup | S3 | `operation.now.VesselOps` | 53 | 53 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q44 | DE | lookup | S3 | `operation.now.OceanVesselOps` | 53 | 53 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q45 | DE | lookup | S3 | `operation.now.FacilityOps` | 53 | 53 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q46 | DE | lookup | S3 | `operation.<key>.VesselOps` | 221 | 221 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q47 | DE | lookup | S3 | `operation.<key>.OceanVesselOps` | 221 | 221 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |
| Q48 | DE | lookup | S3 | `operation.<key>.FacilityOps` | 221 | 221 | 0 | 0 | 0 | 0 | 0 | 1.000 | 1.000 |

Across all templates with decidable comparisons: **104763/106641 = 0.9824** of shipped answers agree with the field their own template names.

### Templates whose shipped answers disagree with their declared field

- **Q2** (lookup) — agreement 0.999, determinacy 1.000 over 2873 records, declared path `weather_observation.under_watch`
  - `BARRY_2019_New_Orleans,_LA|Q2|-|T-60`: shipped `Tropical Storm` vs expected `Hurricane Watch, Tropical Storm Watch, Storm Surge Watch` (option text differs from ground truth)
  - `BARRY_2019_Plaquemines_Port_District,_LA|Q2|-|T-60`: shipped `Tropical Storm` vs expected `Hurricane Watch, Tropical Storm Watch, Storm Surge Watch` (option text differs from ground truth)
- **Q3** (lookup) — agreement 0.999, determinacy 1.000 over 2902 records, declared path `weather_observation.under_warning`
  - `ISAIAS_2020_Wilmington,_NC|Q3|-|T-12`: shipped `Tropical Storm` vs expected `Hurricane Warning (also Storm Surge Warning and nested Tropical Storm Warning for broader area)` (option text differs from ground truth)
  - `MARCO_2020_South_Louisiana,_LA,_Port_of|Q3|-|T-36`: shipped `Tropical Storm` vs expected `Hurricane Warning (also Storm Surge Warning, Tropical Storm Warning for Lakes Pontchartrain & Maurepas/Metro New Orleans)` (option text differs from ground truth)
- **Q18** (lookup) — agreement 0.978, determinacy 1.000 over 2851 records, declared path `cyclone_condition_and_forecast.category`
  - `ALEX_2022_Jacksonville,_FL|Q18|-|T-48`: shipped `Tropical Storm` vs expected `Potential Tropical Cyclone / Sub-Tropical Storm strength (≤39 mph)` (option text differs from ground truth)
  - `ALEX_2022_Port_Everglades,_FL|Q18|-|T-36`: shipped `Tropical Storm` vs expected `Potential Tropical Storm / below Category 1` (option text differs from ground truth)
- **Q24** (ordinal_bucket) — agreement 0.991, determinacy 0.866 over 2910 records, declared path `cyclone_condition_and_forecast.hours_of_strike`
  - `BARRY_2019_Beaumont,_TX|Q24|-|T-72`: shipped `0 hours` vs expected `≈24` (value 24 falls in buckets ['C', 'D'], not in A)
  - `BETA_2020_Calhoun_Port_Authority,_TX|Q24|-|T-72`: shipped `0 hours` vs expected `30-36` (value 30 falls in buckets ['D'], not in A)
- **Q25** (ordinal_bucket) — agreement 0.948, determinacy 0.860 over 2908 records, declared path `weather_observation.time_to_34kt_max`
  - `ANA_2015_Port_of_Charleston,_SC|Q25|-|T-12`: shipped `12-24 Hours` vs expected `6` (value 6 falls in buckets ['A'], not in B)
  - `ANA_2015_Port_of_Savannah,_GA|Q25|-|T-60`: shipped `12-24 Hours` vs expected `6` (value 6 falls in buckets ['A'], not in B)
- **Q27** (ordinal_bucket) — agreement 0.988, determinacy 0.604 over 2908 records, declared path `weather_observation.time_to_34kt_non_zero`
  - `BARRY_2019_Kansas_City_Port_Authority,_MO|Q27|-|T-12`: shipped `0 hours` vs expected `>120` (value 120 falls in buckets ['D'], not in A)
  - `BARRY_2019_Kansas_City_Port_Authority,_MO|Q27|-|T-60`: shipped `0 hours` vs expected `≈72` (value 72 falls in buckets ['D'], not in A)
- **Q30** (lookup) — agreement 0.993, determinacy 1.000 over 2749 records, declared path `weather_observation.storm_surge_above_3_feet`
  - `BARRY_2019_Port_Arthur,_TX|Q30|-|T-48`: shipped `Unlikely` vs expected `No, unlikely for Port Arthur` (option text differs from ground truth)
  - `CINDY_2017_Galveston,_TX|Q30|-|T-12`: shipped `No` vs expected `Possible but not likely to exceed 3 ft` (option text differs from ground truth)
- **Q31** (lookup) — agreement 0.988, determinacy 1.000 over 2567 records, declared path `weather_observation.rain_fall_above_4_inch`
  - `CLAUDETTE_2021_Lake_Charles_Harbor_District,_LA|Q31|-|T-24`: shipped `Possible` vs expected `Likely (4–6 inches possible)` (option text differs from ground truth)
  - `COLIN_2016_Jacksonville,_FL|Q31|-|T-24`: shipped `Possible` vs expected `Likely (3–5 inches expected, isolated 8 inches possible)` (option text differs from ground truth)
- **Q36** (lookup) — agreement 0.003, determinacy 1.000 over 2910 records, declared path `impact_prediction.day_of_recover`
  - `ALEX_2022_Canaveral_Port_District,_FL|Q36|-|T-12`: shipped `6/3/2022` vs expected `4.0` (|6.0 - 4.0| > 0.04)
  - `ALEX_2022_Canaveral_Port_District,_FL|Q36|-|T-24`: shipped `6/3/2022` vs expected `4.0` (|6.0 - 4.0| > 0.04)

## Declared defects

| id | severity | verdict | measured |
|---|---|---|---|
| `qa-answer-placeholder-none` | label | reproduced | `{"literal_none": 7660, "empty_string": 677, "share_unanswerable": 0.0711, "horizon_blocks_with_empty_operation": 2560, "horizon_blocks_whose_impact_prediction_is_the_string_None": 1334, "top_templates": [["Q34", 1334], ["Q35", 1334], ["Q36", 1334], ["Q38", 1334], ["Q39", 1334]]}` |
| `template-q36-answer-field-mismatch` | label | reproduced | `{"q36_records": 2910, "q36_equals_start_impact_date": 2910, "q36_equals_end_impact_date": 1334, "share_answering_from_start_field": 1.0, "q34_q35_q36_groups": 2910, "groups_where_all_three_agree": 2079}` |
| `qa-q33-answer-is-python-repr` | format | reproduced | `{"q33_records": 2908, "parseable_as_json": 0, "recovered_by_literal_eval": 2908}` |
| `imbalanced-fragments-unterminated` | blocking | reproduced | `{"concat_parses_as_json": false, "brace_depth_at_eof": 1, "ended_inside_string": false, "scenarios_recovered_by_scanner": 700, "scanner_note": "recovered 700 scenario objects from the concatenated Imbalanced fragments (700 keys are present in the raw text); the stream never returns to brace depth 0, so records after...` |
| `qa-context-arity-is-variable` | format | reproduced | `{"arity_3": 31945, "arity_4": 85233, "share_arity_3": 0.2726, "three_element_only_templates": ["Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q17", "Q18", "Q20", "Q21", "Q23"]}` |
| `mc-options-duplicated-and-inconsistent-with-prompt` | format | reproduced | `{"mc_records": 37813, "records_with_choice_array": 37813, "records_with_options_recovered": 37813, "inline_matches_choice_length": 37813, "records_where_prompt_clause_disagrees_with_choice_array": 0, "records_where_question_clause_disagrees_with_choice_array": 2851, "examples": [{"qa_id": "ALEX_2022_Canaveral_Port_D...` |
| `raw-categorical-fields-not-normalised` | format | reproduced | `{"distinct_category_strings": 123, "categories_not_mapped_to_a_rung": 8, "distinct_within_uncertainty_cone_strings": 21, "distinct_under_watch_strings": 36, "condition_spellings": {"WHISKEY": 85, "ZULU": 209, "X-RAY": 105, "YANKEE": 208, "XRAY": 3}, "condition_spellings_off_the_ladder": []}` |
| `ordinal-buckets-overlap` | label | reproduced | `{"templates": {"Q23": {"checked": 2909, "ambiguous": 0, "determinacy": 1.0, "option_sets": [["2 days", "3 days", "4 days", "5 days"]], "example_value": 5.0}, "Q24": {"checked": 2783, "ambiguous": 374, "determinacy": 0.8656, "option_sets": [["0 hours", "1-12 Hours", "13-24 hours", "24 hours or more"]], "example_value...` |
| `td-answers-are-verbatim-field-copies` | simplification | reproduced | `{"description_records": 24094, "byte_identical_to_the_declared_field": 24094, "share_verbatim": 1.0, "templates": ["Q4", "Q5", "Q7", "Q9", "Q10", "Q17", "Q20", "Q32", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48"]}` |
| `paper-counts-exceed-release` | format | reproduced | `{"release_scenario_keys": 472, "release_horizon_blocks": 2910, "release_ports": 45, "release_storms": 69, "paper_ports": 145, "paper_storms": 90, "paper_scenarios": 2917, "questions": 117178, "paper_questions": 117178}` |

### Details

#### `qa-answer-placeholder-none` — reproduced

*Severity:* label  
*Affects:* dataset/CyPortQA.json (8,337 records), Encoded_senario.json (4,002 impact_prediction blocks)

7,660 QA records ship the literal string "None" as their answer and 677 ship an empty string — together 7.1% of the benchmark. Importantly, most of these are faithful copies rather than generation failures: the encoding itself stores the *string* "None" (not JSON null) in `impact_prediction` for 4,002 (scenario, horizon) pairs, and the QA generator copied it through. The underlying gap is in the source data — the AIS-derived impact predictions are unavailable for those scenarios — so these questions are unanswerable from the encoding by construction, not merely by accident. The rest correlate with `operation`, which is empty in 2,560 of 2,910 horizon blocks.

*Workaround:* Treat them as unanswerable and exclude them from accuracy denominators — they cannot be answered from the released data, so scoring them measures nothing. `QARecord.is_answerable` marks them; `is_placeholder_answer` is the predicate.

```json
{
  "literal_none": 7660,
  "empty_string": 677,
  "share_unanswerable": 0.0711,
  "horizon_blocks_with_empty_operation": 2560,
  "horizon_blocks_whose_impact_prediction_is_the_string_None": 1334,
  "top_templates": [
    [
      "Q34",
      1334
    ],
    [
      "Q35",
      1334
    ],
    [
      "Q36",
      1334
    ],
    [
      "Q38",
      1334
    ],
    [
      "Q39",
      1334
    ]
  ]
}
```

#### `template-q36-answer-field-mismatch` — reproduced

*Severity:* label  
*Affects:* Q34, Q35, Q36 (S2/IR/NU — 8,730 records)

Q36 asks 'When is the expected end of impact on freight?' and its template names `impact_prediction.day_of_recover` as the answer source, but the shipped answer is neither that field nor `end_impact_date`: it equals `start_impact_date` in 100% of the 2,910 cases (sample: ANA_2015 Savannah at every horizon answers 5/4/2015 while `end_impact_date` is 5/6/2015). Q34 and Q35 answer from the same value in 2,079 of 2,910 groups, so all three date questions collapse onto one field.

*Workaround:* Score Q34 against `start_impact_date` only, and report Q35/Q36 as label-defective: `Template.answer_path` returns the template's declared path, and `audit_templates` flags the discrepancy rather than hiding it. Excluding them entirely is the conservative option for a headline number.

```json
{
  "q36_records": 2910,
  "q36_equals_start_impact_date": 2910,
  "q36_equals_end_impact_date": 1334,
  "share_answering_from_start_field": 1.0,
  "q34_q35_q36_groups": 2910,
  "groups_where_all_three_agree": 2079
}
```

#### `qa-q33-answer-is-python-repr` — reproduced

*Severity:* format  
*Affects:* Q33 (S2/IR/NU — 2,908 records)

Q33 says 'Answer with ONLY a JSON' but its ground truth is a Python `repr` of a dict, with single quotes and an unquoted-looking percentage: "{'in 12h': '1%', 'in 24h': '3%', ...}". `json.loads` rejects it.

*Workaround:* `QARecord.answer_structured` parses it with `ast.literal_eval` after `json.loads` fails.

```json
{
  "q33_records": 2908,
  "parseable_as_json": 0,
  "recovered_by_literal_eval": 2908
}
```

#### `imbalanced-fragments-unterminated` — reproduced

*Severity:* blocking  
*Affects:* source_data/CyPortQA_Imbalanced/CyPortQA_Imbalanced_part{1,2}.json (122 MB)

`CyPortQA_Imbalanced_part1.json` and `_part2.json` are two fragments of one file cut mid-stream — part1 ends with `},` and part2 begins with a bare key — but concatenating them still does not yield valid JSON: a brace-depth scan that is string-aware shows the concatenation never returns to depth 0, so a container opened inside part2 is never closed. Per-part brace counts are 61,559/61,558 and 88,560/88,560; the second part should close one more brace than it opens.

*Workaround:* `load_imbalanced` repairs the stream with a string-aware record scanner and reports how many scenarios it recovered, refusing to claim a clean parse. The split also uses a cleaner schema than `dataset/CyPortQA.json` — MCQ options are a separate `choice` array instead of being inlined into the question string — so it is worth recovering.

```json
{
  "concat_parses_as_json": false,
  "brace_depth_at_eof": 1,
  "ended_inside_string": false,
  "scenarios_recovered_by_scanner": 700,
  "scanner_note": "recovered 700 scenario objects from the concatenated Imbalanced fragments (700 keys are present in the raw text); the stream never returns to brace depth 0, so records after the unterminated region are unrecoverable. final depth was 1 (expected 0)."
}
```

#### `qa-context-arity-is-variable` — reproduced

*Severity:* format  
*Affects:* all 117,178 QA records

`context` is a positional array whose arity is not fixed: 85,233 records carry [port, storm, year, horizon] and 31,945 carry [storm, year, horizon] with the port omitted. The 3-element form is used by exactly the port-independent templates (Q4-Q9, Q17, Q18, Q20, Q21, Q23). Indexing `context[3]` positionally, as the array invites, silently reads the wrong field on 27% of the benchmark.

*Workaround:* `QAContext.parse` keys on arity and validates the horizon against `HORIZON_HOURS`.

```json
{
  "arity_3": 31945,
  "arity_4": 85233,
  "share_arity_3": 0.2726,
  "three_element_only_templates": [
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q17",
    "Q18",
    "Q20",
    "Q21",
    "Q23"
  ]
}
```

#### `mc-options-duplicated-and-inconsistent-with-prompt` — reproduced

*Severity:* format  
*Affects:* all MC records (37,813); stale clause on 2,851 (Q18)

All 37,813 multiple-choice records carry their options twice — as a structured `choice` array and inlined into the `question` string — and the two agree in length on every record. The inconsistency is in the *boilerplate*: both the `prompt` and the `question` open with a clause stating the option count in words, and the question's copy is stale on 2,851 records (all of Q18). Q18 asks for the SSHS category, offers A-G, and the prompt correctly says '(A, B, C, D, E, F or G)', but the question itself still says 'Answer with ONLY a choice (A, B, C, or D)'. The shipped answer is a bare letter with a trailing period ('B.'), so a consumer that trusts the question's own clause would truncate the option list to four and misread answers E-G.

*Workaround:* `load_qa` prefers the `choice` array and falls back to `parse_inline_options`, keeping `extra['choice']` for inspection. `resolve_mc_letter` maps the letter to its text. Do not derive the option list from the question's clause.

```json
{
  "mc_records": 37813,
  "records_with_choice_array": 37813,
  "records_with_options_recovered": 37813,
  "inline_matches_choice_length": 37813,
  "records_where_prompt_clause_disagrees_with_choice_array": 0,
  "records_where_question_clause_disagrees_with_choice_array": 2851,
  "examples": [
    {
      "qa_id": "ALEX_2022_Canaveral_Port_District,_FL|Q18|-|T-12",
      "source": "question",
      "clause_states": 4,
      "choice_array_has": 7,
      "inline_offers": 7
    },
    {
      "qa_id": "ALEX_2022_Canaveral_Port_District,_FL|Q18|-|T-24",
      "source": "question",
      "clause_states": 4,
      "choice_array_has": 7,
      "inline_offers": 7
    },
    {
      "qa_id": "ALEX_2022_Canaveral_Port_District,_FL|Q18|-|T-36",
      "source": "question",
      "clause_states": 4,
      "choice_array_has": 7,
      "inline_offers": 7
    }
  ]
}
```

#### `raw-categorical-fields-not-normalised` — reproduced

*Severity:* format  
*Affects:* cyclone_condition_and_forecast.category, weather_observation.within_uncertainty_cone, weather_observation.under_watch

Fields that §A.2 expects to normalise are shipped in raw natural-language form with hundreds of surface variants. Measured on `Encoded_senario.json`: `cyclone_condition_and_forecast.category` has 130+ distinct strings for what is really the SSHS ladder plus a pre-storm ladder (e.g. 'Category 2', 'Category 2 Hurricane', 'Category 2 (105-110 mph, forecast to reach major-hurricane strength)', 'Hurricane (Category 1 now - forecast to strengthen)'); `within_uncertainty_cone` has ~22 spellings of Yes/No; `under_watch` has ~40. `operation.Condition` is the one clean field, with the single exception of the `XRAY`/`X-RAY` split.

*Workaround:* This is the workload §A.2 step 1's alias table and §A.2 step 2's entity-id canonicalisation exist to absorb. It is evidence *for* the paper's design rather than a defect to route around, but it means the alias table has to be built against these real strings — `vocabulary.RELATION_ALIASES` alone is not enough, and `data/CONTRACT.md` records the expected normalisation for each field.

```json
{
  "distinct_category_strings": 123,
  "categories_not_mapped_to_a_rung": 8,
  "distinct_within_uncertainty_cone_strings": 21,
  "distinct_under_watch_strings": 36,
  "condition_spellings": {
    "WHISKEY": 85,
    "ZULU": 209,
    "X-RAY": 105,
    "YANKEE": 208,
    "XRAY": 3
  },
  "condition_spellings_off_the_ladder": []
}
```

#### `ordinal-buckets-overlap` — reproduced

*Severity:* label  
*Affects:* Q23, Q24, Q25, Q27, Q40 (ordinal buckets); Q26 (threshold predicate)

Six templates (Q23, Q24, Q25, Q27, Q40 and Q26's predicate sibling) do not answer from their declared field directly: the field is a scalar that has to be placed in an ordinal bucket supplied by the question's options. Several of those bucket sets overlap, so one value satisfies two options and the question is not decidable from the encoding at all. Q25 offers '0-12 hours', '12-24 Hours', '24-48 hours', '48 hours or more', making 12, 24 and 48 each admissible under two options; Q27's '0 hours' and '0-24 Hours' both admit 0; Q24's '13-24 hours' and '24 hours or more' both admit 24. §4.1 scores these by exact letter match, so part of CyPortQA's headline accuracy is decided by an ambiguity the field cannot resolve.

*Workaround:* Classify the templates in `cyportqa.TEMPLATE_SEMANTICS` and score them through `groundtruth.bucket_letters`, which accepts any bucket containing the value and reports whether the value was ambiguous. `TemplateAudit.determinacy` then quantifies the unavoidable share. Report agreement and determinacy side by side; reporting agreement alone overstates how well the benchmark measures what it claims to.

```json
{
  "templates": {
    "Q23": {
      "checked": 2909,
      "ambiguous": 0,
      "determinacy": 1.0,
      "option_sets": [
        [
          "2 days",
          "3 days",
          "4 days",
          "5 days"
        ]
      ],
      "example_value": 5.0
    },
    "Q24": {
      "checked": 2783,
      "ambiguous": 374,
      "determinacy": 0.8656,
      "option_sets": [
        [
          "0 hours",
          "1-12 Hours",
          "13-24 hours",
          "24 hours or more"
        ]
      ],
      "example_value": 12.0
    },
    "Q25": {
      "checked": 2522,
      "ambiguous": 350,
      "determinacy": 0.8612,
      "option_sets": [
        [
          "0-12 hours",
          "12-24 Hours",
          "24-48 hours",
          "48 hours or more"
        ]
      ],
      "example_value": 96.0
    },
    "Q27": {
      "checked": 2532,
      "ambiguous": 1001,
      "determinacy": 0.6047,
      "option_sets": [
        [
          "0 hours",
          "0-24 Hours",
          "24-48 hours",
          "48 hours or more"
        ]
      ],
      "example_value": 96.0
    },
    "Q40": {
      "checked": 2910,
      "ambiguous": 0,
      "determinacy": 1.0,
      "option_sets": [
        [
          "None",
          "1-10 vessels",
          "10-20 vessels",
          "20 vessels or more"
        ]
      ],
      "example_value": 0.0
    }
  }
}
```

#### `td-answers-are-verbatim-field-copies` — reproduced

*Severity:* simplification  
*Affects:* Q4, Q5, Q7, Q9, Q10, Q17, Q20, Q32, Q43-Q48 (24,094 records)

All 24,094 text-description answers are byte-identical to the value at the field their template declares — not merely equivalent, identical. §4.1 scores this type with "an LLM-judged semantic score in [0,1] using GPT-4o as the evaluator". On these labels that judge has nothing to do: the reference answer and the source field are the same string, so exact match measures the generator's fidelity to the retrieved evidence, which is what the metric is for, while the judge can only add sampling variance to a number that needs none.

*Workaround:* `compare` scores a description by normalised exact match, which the early equality check routes to before the format dispatch. Keep the §4.1 GPT-4o judge available as an ablation — it demonstrates the judge does not change the ranking — but do not make it the primary metric, or a reproduction imports stochasticity the benchmark does not have.

```json
{
  "description_records": 24094,
  "byte_identical_to_the_declared_field": 24094,
  "share_verbatim": 1.0,
  "templates": [
    "Q4",
    "Q5",
    "Q7",
    "Q9",
    "Q10",
    "Q17",
    "Q20",
    "Q32",
    "Q43",
    "Q44",
    "Q45",
    "Q46",
    "Q47",
    "Q48"
  ]
}
```

#### `paper-counts-exceed-release` — reproduced

*Severity:* format  
*Affects:* §4.1 vs dataset/CyPortQA.json

§4.1 states '2,917 real-world disruption scenarios ... spanning 145 U.S. principal ports and 90 named storms'. The release has 472 storm-year-port scenario keys, 45 distinct ports and 69 distinct storms. The question count (117,178) and the horizon range (T-120..T-12) match exactly. Reading §4.1's 'scenario' as a (storm, port, year, horizon) quadruple reconciles the scenario number — `Encoded_senario.json` holds 2,910 horizon blocks — but not the port and storm counts.

*Workaround:* Report the counts the release actually contains. Do not write 145 ports / 90 storms into any table this reproduction emits.

```json
{
  "release_scenario_keys": 472,
  "release_horizon_blocks": 2910,
  "release_ports": 45,
  "release_storms": 69,
  "paper_ports": 145,
  "paper_storms": 90,
  "paper_scenarios": 2917,
  "questions": 117178,
  "paper_questions": 117178
}
```

## Blocking issues

- `imbalanced-fragments-unterminated`: `CyPortQA_Imbalanced_part1.json` and `_part2.json` are two fragments of one file cut mid-stream — part1 ends with `},` and part2 begins with a bare key — but concatenating them still does not yield valid JSON: a brace-depth scan that is string-aware shows the concatenation never returns to depth 0, so a container opened inside part2 is never closed. Per-part brace counts are 61,559/61,558 and 88,560/88,560; the second part should close one more brace than it opens.
