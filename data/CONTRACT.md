# Data contract — CyPortQA

**Status: the benchmark is public and this project consumes it directly.**

This file replaces an earlier version that specified a *synthesised* schema for user-supplied
data. That version was written from the OKH-RAG PDF alone, before anyone checked whether the
benchmark §4.1 names had actually been released. It had. The contract below is derived from
the released bytes, and every claim in it is re-measured by
`scripts/audit_cyportqa.py`, which writes `reports/cyportqa_audit.md`.

---

## 1. Provenance

| | |
|---|---|
| Repository | <https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA> |
| Benchmark paper | Kuai et al., *CyPortQA: Benchmarking Multimodal Large Language Models for Cyclone Preparedness in Port Operation*, [arXiv:2508.15846](https://arxiv.org/abs/2508.15846) |
| Licence | MIT (the repository's `LICENSE` is vendored alongside the data) |
| Tool that builds it | `scripts/audit_cyportqa.py` consumes `data/raw/cyportqa/` |

There is no provenance ambiguity between this dataset and the paper being reproduced. OKH-RAG
is by Keshu Wu, **Chenchen Kuai**, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang, Chan-Wei Hu,
**Zhengzhong Tu** and **Yang Zhou**; CyPortQA is by **Chenchen Kuai**, Chenhao Wu, **Yang Zhou**,
Xiubin Bruce Wang, Tianbao Yang, **Zhengzhong Tu**, Zihao Li and Yunlong Zhang (arXiv:2508.15846v2
listing; the release's own bibTeX entry agrees). **Four authors appear on both** — Kuai, Zhou, Tu
and Li. An earlier revision of this file said five, having counted OKH-RAG's *Keshu* Wu against
CyPortQA's *Chenhao* Wu; those are two different people, and no author of either paper is listed
under both spellings.
§4.1's description also matches the release exactly on the numbers it states:

| §4.1 | release | |
|---|---|---|
| "117,178 questions" | **117,178** | exact |
| "multi-horizon descriptions from T-120 to T-12" | **10 horizon keys, 12 … 120, at 12-hour spacing** | exact |
| "four types: True/False (TF), Multiple Choice (MC), Short Answer / Numeric (SA), and Text Description (TD)" | answer-format tags `TF` / `MC` / `NU` / `DE` | see §4 below — the paper's four names are a coarsening |
| "2,917 real-world disruption scenarios" | 2,910 horizon blocks over 472 scenario keys | reconciled in §6 |
| "145 U.S. principal ports and 90 named storms" | **45 ports, 69 storms** | **does not reconcile** |

## 2. What to vendor, and what not to

```
data/raw/cyportqa/
├── LICENSE                                             # MIT, from the upstream repo
├── README.md                                           # upstream README, verbatim
├── dataset/
│   └── CyPortQA.json                   101.4 MB        # 117,178 QA records
├── raw_text/                             5.87 MB       # rescued NOAA text products — see §2.1
│   ├── NOAA NHC Cyclone Products/
│   │   ├── Cyclone Text Archive Advisory/    475 files # NOAA TCP advisories, per storm + lead time
│   │   └── Cyclone Text Archive Wind/        449 files # NOAA wind-speed probability products
│   └── Port_Condition_Bullitens/
│       └── uscg_port_conditions_agg.csv     0.09 MB    # USCG port-condition bulletins, aggregated
└── source_data/
    ├── Encoded_senario.json              9.6 MB        # structured knowledge, the hypergraph source
    ├── CyPortQA_template.json            0.04 MB       # the 48 templates and their answer paths
    └── CyPortQA_Imbalanced/
        ├── CyPortQA_Imbalanced_part1.json  51.9 MB     # corrupted — see §7
        └── CyPortQA_Imbalanced_part2.json  70.6 MB     # corrupted — see §7
```

**Not vendored: the two graphics trees.** `dataset/MultiModalInput/Cyclone Graphics Archive *`
and `source_data/NOAA NHC Cyclone Products/Cyclone Graphics Archive *` hold **2,031 PNG/JPEG
files, 157.1 MB** — the pictorial half of CyPortQA. OKH-RAG is text-only: §4.1 gives it and every
baseline the same `gpt-4o` generator and the same `text-embedding-3-small` embedder, neither of
which is a vision model. A reader comparing §4.1 to this vendoring should know the graphics exist
and were deliberately excluded rather than silently dropped, so §2.2 records where they sit in the
released input protocol.

**The `.txt` trees are the opposite case, and are vendored.** They are text; they are precisely the
`text_advisory` and `Table_wind` modalities the QA records request (§2.2); and 5.9 MB of NOAA
advisories is the only prose corpus this benchmark offers, which the project's own prose branch
(`configs/base.yaml` → `extraction.header_pattern`) would otherwise have nothing to read. They were
copied out of the clone before it was removed. The clone is not kept — `raw_text/` plus the commands
below regenerate it — and `raw_text/` is under `data/raw/`, so it stays out of git like the rest of
the bytes.

To reproduce the vendoring from a fresh clone:

```bash
git clone --depth 1 https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA.git
mkdir -p data/raw/cyportqa/{dataset,source_data/CyPortQA_Imbalanced,raw_text}
cp MLLM-Bench-CyPortQA/dataset/CyPortQA.json                       data/raw/cyportqa/dataset/
cp MLLM-Bench-CyPortQA/source_data/Encoded_senario.json            data/raw/cyportqa/source_data/
cp MLLM-Bench-CyPortQA/source_data/CyPortQA_template.json          data/raw/cyportqa/source_data/
cp MLLM-Bench-CyPortQA/source_data/CyPortQA_Imbalanced/*.json      data/raw/cyportqa/source_data/CyPortQA_Imbalanced/
cp MLLM-Bench-CyPortQA/{LICENSE,README.md}                         data/raw/cyportqa/

# text products only; the graphics trees are skipped on purpose
cp -r "MLLM-Bench-CyPortQA/source_data/NOAA NHC Cyclone Products"  data/raw/cyportqa/raw_text/
find data/raw/cyportqa/raw_text -path "*Cyclone Graphics Archive*" -delete
mkdir -p data/raw/cyportqa/raw_text/Port_Condition_Bullitens
cp "MLLM-Bench-CyPortQA/source_data/Port_Condition_Bullitens/uscg_port_conditions_agg.csv" \
   data/raw/cyportqa/raw_text/Port_Condition_Bullitens/
```

The upstream clone stores each text tree **twice** — byte-identical copies under
`dataset/MultiModalInput/` and under `source_data/NOAA NHC Cyclone Products/`. `raw_text/` keeps one
copy of each: 960 `.txt` files collapse to 924 distinct contents, 36 duplicates dropped.

### 2.1 Rescued upstream code — `third_party/cyportqa_official/`

The clone also held 13 non-image files of the benchmark's own harness, which would have been lost
with it. They are MIT, 36 KB in total, and they are vendored **tracked** at
`third_party/cyportqa_official/` with a `PROVENANCE.md` recording the source commit. Two of the
three groups matter to this reproduction:

| file(s) | why it is kept |
|---|---|
| `evaluators/prompt_*.txt` (3) | the actual §4.1 judge prompts — an ordinal 0–1 rubric per ability category (`0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0`) plus a "hallucination rule" that forces `score = 0.0` when invented data appears. This is the strongest available evidence for how §4.1's LLM-judged score is computed, and it is needed by the evaluation layer. |
| `run.py`, `config.json`, `Requirements.txt` | the official multi-model harness: `config.json` selects which `models/run_*.py` to launch and with which `--start/--end/--dataset/--output` arguments. |
| `models/run_*.py` (7) | the seven released zero-shot runners (GPT-4o, Gemini Flash, Gemma 3, LLaMA 3.2, LLaVA 1.6, Mistral, Qwen2.5-VL). They fix the prompt and input-file protocol described in §2.2. |

### 2.2 What the released runners establish about the input protocol

`models/run_gpt4o.py` is the reference implementation of "how a question is posed", and it settles
three things this project had been inferring:

1. **The `context` layout.** The runner does `storm, year, leadtime = Q["context"][-3:]` — it reads
   from the *end*, never positionally from the front. This is the upstream's own answer to the
   arity problem (§4.2) and it independently confirms `src/okh_rag/data/cyportqa.py:QAContext.parse`:
   over all **117,178** records the two readings agree **exactly (0 disagreements)**. The arity-4
   form is `[port, storm, year, leadtime]`; the prefix is a port name, of which there are exactly
   **45 distinct** values in 85,233 arity-4 records — matching §1's port count.
2. **What the two string fields are.** `Q["prompt"]` is sent as the **system** message and
   `Q["question"]` as the user turn. So the template's `prompt` (its static prefix) is not part of
   the question text a model is scored on; it is framing.
3. **How input files are named.** `f"{storm}_{year}_{leadtime}h"` with extension `.txt` for text
   modalities and `.png` for graphics — i.e. `ALEX_2022_12h.txt`, exactly the filenames in
   `raw_text/`.

It also exposes a defect in the release (§2.3).

### 2.3 `modalities` values are not directory names

The four values that appear in `CyPortQA.json`'s `modalities` arrays do **not** name the directories
the files actually live in:

| `modalities` value | records | actual tree under `dataset/MultiModalInput/` | ext |
|---|---|---|---|
| `Graphic_Uncertainty_cone` | 67,805 | `Cyclone Graphics Archive Uncertainty Cone` | `.png` |
| `Table_wind` | 50,880 | `Cyclone Text Archive Wind` | `.txt` |
| `text_advisory` | 47,410 | `Cyclone Text Archive Advisory` | `.txt` |
| `Graphic_Wind` | 27,621 | `Cyclone Graphics Archive Wind` | `.png` |

`run_gpt4o.py` builds `Path(input_folder)/modality/f"{filename}.txt"`, so against the shipped tree
it raises `FileNotFoundError` on **every** text modality — 98,290 of the 193,716 modality
references. Reproducing §4.1's baseline therefore requires the rename map above; the released code
only works if the caller has renamed the directories first. Two corollaries worth keeping:

* `Table_wind` is **not** a table. It resolves to `Cyclone Text Archive Wind`, a `.txt` tree, and
  the runner takes its `else` branch accordingly. The name invites the opposite reading.
* The released code draws a distinction the data does not: `run_gpt4o.py` branches on
  `modality in ["Graphic_Uncertainty_cone", "Graphic_Wind"]` to pick `.png` vs `.txt`, but both
  text modalities and both graphics modalities are spelled differently again, so the branch is
  correct while the path is not.

This is a documentation-and-naming defect, not a data one; it is recorded here rather than in §7's
measured table because the graphics trees are not vendored, so part of it cannot be re-measured
offline. `raw_text/` lets the text half be checked: no directory in it is named after any
`modalities` value.


## 3. `Encoded_senario.json` — the file that matters

This is the single most important file in the release, and neither OKH-RAG's paper nor
CyPortQA's spells out what it is for. It is a per-scenario *structured* knowledge encoding,
and it is exactly the raw material §A.2 extracts a hypergraph from.

```
{STORM}_{YEAR}_{PORT}                  # e.g. "ANA_2015_Port_of_Savannah,_GA"   — 472 keys
└── senario
    ├── port      { name, location: {lon, lat}, daily_vessel_count }
    ├── cyclone   { name, year }
    └── time                             # keyed by lead time in hours, as strings
        ├── "12"                         # one of 12 … 120; scenarios are ragged (1–10 blocks)
        │   ├── cyclone_condition_and_forecast
        │   ├── weather_observation
        │   ├── operation                # EMPTY in 2,560 of 2,910 blocks
        │   └── impact_prediction
        ├── "24"
        └── …
```

All four blocks are present on all 2,910 horizon blocks — the loader asserts this, because a
missing block would change what §A.2 has to extract.

**`operation` is keyed by a *relative* offset, not an absolute lead time** — `now`, `in_12h`,
`in_24h`, … `in_108h`. That nesting is the clearest order-sensitivity in the benchmark: one
scenario at one absolute horizon predicts a *sequence* of future port conditions.

### 3.1 Entity categories (§A.1)

The three categories §A.1 defines fall out of the encoding without any interpretation:

| §A.1 category | fields |
|---|---|
| **persistent objects** | `port.name`, `port.location.{lon,lat}`, `port.daily_vessel_count`, `cyclone.name`, `cyclone.year` |
| **transient states** | every leaf under the four horizon blocks **except** the horizon key itself |
| **temporal anchors** | the `time` key (`T-12` … `T-120`) and each `operation` offset (`now`, `in_12h`, …) |

This is what makes the §A.2 step-2 identity rule checkable: a *persistent* object must get one
entity id for the whole scenario (no horizon in its id), while a *transient* state must get one
per horizon — otherwise §A.6 Eq. 12's entity-overlap term and §A.4's overlap signal have
nothing to match across horizons. The loader's `Entity` construction must branch on category
rather than assigning every entity a horizon.

### 3.2 Field → relation family (Table 3)

§A.2 says an LLM assigns "relation labels from R with quantitative attributes". With the
encoding in hand that assignment is not free-form: each field has a canonical family. The
mapping below is the specification the hypergraph builder must implement, and it is what makes
LLM extraction scoreable rather than taken on faith — the extraction output can be compared
against the family each field implies.

| family | relation(s) | source fields |
|---|---|---|
| 1 `cyclone_state` | `has_cyclone_state` | `cyclone_condition_and_forecast.current_location`, `.coordinates` |
| | `has_category_state` | `cyclone_condition_and_forecast.category` |
| | `has_motion` | `cyclone_condition_and_forecast.motion.direction`, `.motion.speed` |
| 2 `track_landfall` | `forecasts_track` | `cyclone_condition_and_forecast.trajectory`, `weather_observation.within_uncertainty_cone` |
| | `forecasts_landfall` | `cyclone_condition_and_forecast.expected_landfall_location`, `.expected_landfall_date`, `weather_observation.impacted_coast`, `.expected_closest_date_to_port`, `.expected_leaving_port_date` |
| 3 `timing` | `has_hours_to_landfall` | `cyclone_condition_and_forecast.hours_to_landfall`, `.hours_to_offshore`, `.hours_of_strike` |
| | `has_forecast_window` | `weather_observation.forecast_window` |
| 4 `advisory` | `has_watch_status` | `weather_observation.under_watch`, `.watch_coast` |
| | `has_warning_status` | `weather_observation.under_warning`, `.warning_coast` |
| 5 `probability` | `has_leadtime_probability` | `weather_observation.gale_probability.*`, `.time_to_34kt_max`, `.time_to_34kt_non_zero`, `.time_to_50kt_max` |
| | `has_cumulative_probability` | `weather_observation.culmulative_wind_34kt`, `.culmulative_wind_50kt`, `.34kt_max` |
| 6 `hazard_forecast` | `forecasts_hazard_at_horizon` | `weather_observation.wind`, `.storm_surge`, `.rainfall`, `.first_hazard` |
| 7 `hazard_observation` | `observes_hazard_at_horizon` | `weather_observation.closest_wind_data_location` |
| 8 `threshold` | `has_threshold_status` | `weather_observation.rain_fall_above_4_inch`, `.storm_surge_above_3_feet` |
| 9 `additional_hazards` | `has_additional_hazard` | `weather_observation.other_hazards` |
| 10 `operations` | `has_operation_status` | `operation.<offset>.Condition` |
| | `affects_vessel_handling` | `operation.<offset>.VesselOps`, `.OceanVesselOps`, `.FacilityOps` |
| 11 `impact` | `has_impact_prediction` | `impact_prediction.start_impact_date`, `.max_impact_date`, `.end_impact_date`, `.max_impact` |
| | `causes_operational_disruption` | `impact_prediction.day_of_impact`, `.total_impact_value` |
| 12 `recovery` | `has_recovery_status` | `impact_prediction.day_of_recover` |
| 13 `cross_horizon` | `forecast_updates_to`, `changes_probability_to` | *no source field* — synthesised by §A.2 step 4 from families 2, 5 and 6 |

Two notes that matter downstream:

* **Family 7 has exactly one source field** (`closest_wind_data_location`), so the
  `hazard_observation` phase is thin. §A.6 Eq. 13's phase-coverage term is computed over a
  6-member phase set that includes both `hazard_forecast` and `hazard_observation`; on this
  data one of the six is nearly always absent, which caps the achievable coverage.
* **`operation` is empty in 88% of horizon blocks.** Family 10 therefore contributes nothing
  in most scenarios, and §4.1's own observation that "S3-type decision reasoning questions
  receive sparser coverage" is the same fact seen from the QA side: S3 is 1,432 of 117,178
  questions (1.2%).

**Three shapes a declared field can take**, and the table above covers all three because §5's
templates declare all three. A field is either a **leaf** (`weather_observation.under_watch`), an
**offset wildcard** (`operation.<offset>.Condition`, matching all ten relative offsets), or a
**sub-dict wildcard / container** (Q33 names `weather_observation.gale_probability` rather than a
key inside it, so it resolves to the union of the mapping's children). Only the two wildcard
forms are in doubt — a typo must fall through rather than fuzzy-match onto a neighbour, or a
misspelled path would manufacture a gold set from the wrong relation.

Two fields are deliberately **not** in the table, and adding them would be wrong:

* `cyclone_condition_and_forecast.name` is a real field of every block and appears in no row.
  §3.1 lists the storm under *persistent objects* — it is an anchor in every ``V_e``, not the
  subject of a relation — so it has no relation and is reported as uncovered rather than guessed
  at. It is the only systematic gap, and `reference_graph` counts it.
* `port.name` and `cyclone.name` (§3.1's other persistent objects) are likewise no relation.
  Q16 and Q19 answer `{senario.port.name}`, so their requirement is an *entity*, not a
  hyperedge.

The table is implemented, once, in `src/okh_rag/evaluation/field_map.py` — 46 rows, the five
wildcard keys, and an entity-path set. Nothing else in the project re-states it.

### 3.3 The categorical fields that need normalising

§A.2 step 1's alias table exists to absorb exactly this, and the released data shows the size
of the job. Measured on `Encoded_senario.json`:

| field | distinct raw strings | note |
|---|---:|---|
| `cyclone_condition_and_forecast.category` | **130+** | one SSHS rung or pre-storm rung, written a hundred ways |
| `weather_observation.within_uncertainty_cone` | ~22 | "Yes", "Yes (southern fringe)", "Yes – on the southwestern edge of the cone", "Borderline/No", … |
| `weather_observation.under_watch` | ~40 | composite lists: "Hurricane Watch (also under Storm Surge Watch)", "None (watches expected soon)", … |
| `operation.<offset>.Condition` | 5 | the one clean field: WHISKEY / X-RAY / XRAY / YANKEE / ZULU |

Samples of the `category` field:

```
"Category 2"
"Category 2 Hurricane"
"Category 2 (105-110 mph, forecast to reach major-hurricane strength)"
"Hurricane (Category 1 now – forecast to strengthen)"
"Tropical Storm (Potential Tropical Cyclone Three – 45 mph)"
"Tropical Depression / Remnants"
"Potential Tropical Cyclone (not yet on SSHS scale)"
```

`groundtruth.normalise_ssh_category` reduces these to a rung where a rule suffices. Rungs it
cannot place are counted in the audit rather than silently dropped — they need hand-written
aliases, and the count tells us how many.

The port-condition ladder is `WHISKEY ≺ X-RAY ≺ YANKEE ≺ ZULU` (escalating), which is the real
USCG order under 33 CFR § 165.781 and is what the `operation` phase's precedence rests on.
`X-RAY` and `XRAY` both occur (105 vs 3) and are the same condition.

## 4. `CyPortQA.json` — the QA side

```
{STORM}_{YEAR}_{PORT}
└── qa  [ 117,178 records in total ]
    {
      "question_type": ["Q18","S1","EI","MC"],   # tags; see below
      "context":       ["HERMINE","2016","12"],  # POSITIONAL AND VARIABLE-ARITY
      "modalities":    ["text_advisory"],
      "prompt":        "Act as a cyclone and port operations analyst. Based on …",
      "question":      "What is the current Saffir Simpson … A. Tropical Depression B. …",
      "choice":        ["Tropical Depression","Tropical Storm", …],   # MC only
      "answer":        "B."
    }
```

### 4.1 `question_type` is a tag list, not a single type

Every record carries 3–6 tags drawn from four disjoint vocabularies:

| role | values | counts |
|---|---|---|
| template id | `Q1` … `Q48` | 48 templates, ~2,910 records each |
| preparedness stage | `S1`, `S2`, `S3` | 84,244 / 31,502 / 1,432 |
| ability dimension | `SA`, `TU`, `EI`, `UQ`, `IR`, `WP` | — |
| **answer format** | **`TF`, `MC`, `NU`, `DE`** | 5,812 / 37,813 / 49,459 / 24,094 |
| annotation | `AUTO`, `CC`, `FC`, `CO`, `FO`, `GV`, `OV`, `PF` | — |
| relative horizon | `in_12h` … `in_108h` | only on Q42, Q46–Q48 |

**§4.1's four type names do not match these tag values.** The paper says "True/False (TF),
Multiple Choice (MC), Short Answer / Numeric (SA), and Text Description (TD)". The release tags
the answer format as `TF` / `MC` / `NU` / `DE`, and `SA` is a *separate* "short answer"
dimension tag that co-occurs with `TF`, `NU` and `DE` — a record can carry both `SA` and `TF`.
Mapping the paper's names onto the data: TF→`TF`, MC→`MC`, SA→`NU`, TD→`DE`.

This matters for the evaluation harness: scoring needs the format tag, and taking the paper's
word for it would mean reading `SA` as a format and mis-scoring all 5,812 true/false items.

### 4.2 `context` arity varies — do not index it positionally

| arity | layout | records | templates |
|---|---|---:|---|
| 4 | `[port, storm, year, horizon]` | 85,233 | 37 templates |
| 3 | `[storm, year, horizon]` — no port | 31,945 | Q4–Q9, Q17, Q18, Q20, Q21, Q23 |

`context[3]` is the natural-looking way to read the horizon and is wrong on **27% of the
benchmark**. `QAContext.parse` keys on arity and validates the horizon against the ten known
values.

### 4.3 Multiple choice

The options appear **twice** — as a structured `choice` array and inlined in the `question`
string — on all 37,813 MC records, and the two agree on length everywhere. The shipped answer
is a bare letter with a trailing period (`"B."`). Use the `choice` array; the inlined copy has
to be recovered by pattern.

## 5. `CyPortQA_template.json` — the answer oracle

48 entries, one per template. Each names the exact field that produces its answer:

```json
{
  "question_type": ["Q1","S1","SA","TU","TF"],
  "context": ["{senario.port.name}","{senario.cyclone.name}","{senario.cyclone.year}","{senario.time.<key>}"],
  "modalities": ["Graphic_Uncertainty_cone"],
  "prompt": "Act as a cyclone and port operations analyst. Based on the Track Forecast Cone …",
  "question": "Is {senario.port.name} positioned inside the TC's projected path cone? …",
  "answer": "{senario.time.<key>.weather_observation.within_uncertainty_cone}"
}
```

Read `answer` as a path: substitute `<key>` with the record's horizon and you have
`weather_observation.within_uncertainty_cone` at that horizon. `groundtruth.resolve_path` does
the substitution and the lookup.

**This is the retrieval oracle §3.2 needs.** Because a question's answer is a named field, the
hyperedges a question requires are the hyperedges built from that field. That makes retrieval
scoreable *independently of generation* — which §4.1's end-to-end accuracy cannot do, and
which is the single biggest methodological gain available from the release.

### 5.1 Three kinds of template

The declared field is not always the literal answer. `TemplateSemantics` records which is
which; comparing a predicate template as a string makes a correct answer look wrong.

| semantics | templates | what the field is |
|---|---|---|
| `lookup` | the other 42 | the answer, up to normalisation |
| `ordinal_bucket` | Q23, Q24, Q25, Q27, Q40 | a scalar to be placed in an ordinal bucket supplied by the options |
| `threshold_predicate` | Q26 | an operand of a threshold stated in the *question* ("reach or exceed 20%") |

Worked examples:

* **Q23** offers `2 days / 3 days / 4 days / 5 days` and the field `forecast_window` holds
  `"5"`. The option text carries a unit the field omits, so string equality always fails.
* **Q26** asks whether 34-kt wind probability "reach[es] or exceed[s] 20%" and the field
  `34kt_max` holds `"11%"`. The answer is `11 >= 20 → "No"`; recovering it needs the threshold
  from the question text.

### 5.2 The text-description templates do not need §4.1's judge

§4.1 scores text description with "an LLM-judged semantic score in [0,1] using GPT-4o as the
evaluator". Measured on the release: **all 24,094 description answers are byte-identical to the
value at the field their template declares** — not equivalent, identical. The judge therefore has
nothing to judge, and making it the primary metric would import GPT-4o sampling variance into a
number the labels already determine. This project scores TD by normalised exact match and keeps
the judge as an ablation, which is also the cleaner test: it shows the judge does not change the
ranking.

The `unverifiable` column of the audit is the count of records needing the judge. It is zero for
every template.

### 5.3 The oracle is wired

The oracle §5 describes is implemented, and it runs **with no API key** — which is the point. §A.2
extraction is the only non-deterministic step before retrieval, and `Encoded_senario.json` already
holds the same scenario in structured form, so "what would a perfect extractor have produced?"
has a closed-form answer that calls no model:

> one hyperedge per (scenario, horizon, populated field), with the relation §3.2 gives that field.

| module | what it is |
|---|---|
| `evaluation/field_map.py` | §3.2 as data — 46 rows, 5 wildcard keys, 2 entity paths |
| `evaluation/reference_graph.py` | the yardstick above, running §A.2 steps 2–5 for real via `normalize_hypergraph` |
| `evaluation/oracle.py` | the gold-set definition and the metrics — works on **any** hypergraph, so it scores an extracted one too |
| `scripts/run_retrieval_oracle.py` | driver → `reports/retrieval_oracle.{md,json}` |

**Gold is keyed on the relation, not the field**, and §3.2's own wording is why ("compared against
the family each field implies"). On the reference graph the two readings coincide, because one
hyperedge is one field — but on an *extracted* hypergraph §A.2 asks for evidence sentences and
never for source offsets, so there is no field to key on and a field-keyed oracle would score zero
records on the graph the pipeline actually produces. `oracle.gold_matches_field_provenance` checks
the relation-keyed sets against the field-keyed ones wherever both exist, so the looser key is
verified rather than assumed: **0 mismatches over all 111,358 scorable questions**.

**Template coverage.** All 48 templates resolve: 46 to an §A.1 Table 3 relation, 2 (Q16, Q19) to
a persistent object, **0 uncovered**. `template_coverage` is a static property of the spec —
no corpus, no graph — so an uncovered template fails before any run does work.

**The reference hypergraph over the whole release** (472 scenarios, 2,910 horizon blocks):

| quantity | value |
|---|---:|
| fields seen | 138,196 |
| fields kept | 135,248 (97.87%) |
| hyperedges (within-horizon, one per kept field) | 135,248 |
| hyperedges (family 13, §A.2 step 4) | 113,404 — **45.6%**, §4.2's "nearly half" |
| hyperedges total | 248,652 |
| entities | 139,102 |
| hyperedge ids aliased across groups | **0** (see `entity-id-has-no-year`) |

**The gold inventory.** Of the 117,178 questions, **111,358 (95.0%)** are scorable against a
relation and **5,820** are the entity questions; **0 are unmapped** and **0 are empty**, so the
oracle scores 95.0% of the benchmark with 100% classification and no coverage hole.

**The metrics**, per §A.7 stage, so a failure can be localised where §4.1's accuracy cannot:

`pool_recall` (was the evidence even in `C`?) · `anchor_recall` · `trajectory_recall` ·
`trajectory_precision` · `chance_recall` (= `pool_recall × L/|C|`, the floor the previous number
must be read against) · `first_gold_step` · `canonical_concordance` and `gold_order_concordance`
(agreement with §A.3's canonical order).

The last two exist for §4.5 Table 2's shuffle row, which holds retrieved content *fixed* while
permuting it: every content metric is invariant there by construction, so a harness that measured
only recall would report that ablation as a no-op. `canonical_concordance` is defined without
reference to gold — a message-passing metric restricted to gold is undefined whenever fewer than
two gold hyperedges are retrieved, which on this corpus is almost always (measured: 2 of 120
records).

**Instrument check.** With a token-overlap stand-in for §A.4's embedder (diagnostic only — its
absolute values say nothing about OKH-RAG), the three §A.7 configurations behave as required:

| configuration | pool recall | trajectory recall | chance recall | first gold step | canonical concordance |
|---|---:|---:|---:|---:|---:|
| `heuristic` (§A.3 order) | 0.5782 | 0.0319 | 0.0308 | 3.10 | 0.678 |
| `none` (order-free) | 0.5782 | 0.0312 | 0.0308 | 3.38 | 0.541 |
| `heuristic` + shuffle | 0.5782 | **0.0319** | 0.0308 | **5.70** | **0.438** |

Recall is flat across the shuffle row (identical content, as §4.5 states) while the order metrics
move — which is exactly the separation the oracle has to be able to make. The headline numbers
require `--backend openai`.

### 5.4 The scoring side, and the one regime the release only partly supports

The oracle above scores **what was retrieved**; §4.1's accuracy scores **what was said**. The two
are deliberately separate — §4.1's end-to-end number cannot tell a retrieval failure from a
generation failure, which is the whole reason §5.3 exists — and the second one lives in
`src/okh_rag/evaluation/answers.py`.

§4.1 gives four regimes: exact match for TF and MC, tolerance-based accuracy for SA, an LLM-judged
semantic score for TD. Three of the four can be applied to this release as written. **The fourth
cannot, for 39.6% of its questions.**

| | count | share of NU |
|---|---:|---:|
| SA (`NU`) questions | 49,459 | — |
| tolerance-scorable | 29,885 | 60.4% |
| not a single numeric value | **19,574** | **39.6%** |

The unscorable 39.6% is not noise; it breaks down into causes that mean different things:

| cause | count | example | blame |
|---|---:|---|---|
| `absent` | 8,896 | Q14 `N/A`, `Not expected` | unanswerable — the encoding has no value |
| `date` | 4,728 | Q34 `5/4/2015`, `9/29/2016 0:00` | the question asks "what date", tagged `NU` |
| `coordinates` | 2,910 | Q6 `32.9N, 78.3W` | 100% of Q6; tagged `NU`, answered as a pair |
| `multi_horizon` | 2,908 | Q33 `{'in 12h': '1%', 'in 24h': '3%', ...}` | 100% of Q33; a mapping, shipped as a Python `repr` (defect `qa-q33-answer-is-python-repr`) |
| `range` | 62 | Q21 `8-12`, Q13 `0-6` | genuinely ambiguous |
| `prose` | 51 | Q8 `0 mph (will increase to ~10 mph)` | genuinely ambiguous |
| `other` | 19 | Q11 `Data not provided for St. Croix` | unanswerable |

Only the `range` and `prose` rows are quantities a tolerance test *could* compare if a convention
were chosen, and together they are **0.2%** of SA questions. That is why the scorer reports these
questions as unscorable rather than inventing a rule for them: the ambiguity is not where the mass
is, and the four large rows are not ambiguity at all but the release's own format tags disagreeing
with its own answers.

**Consequence for any reported SA number.** §4.1's SA column, applied literally, is computable on
29,885 of 49,459 questions. A reproduction that scores the other 19,574 as wrong reports an SA
accuracy that is mostly a measure of the release's tagging. The number is still reportable — but it
has to travel with its denominator, which is why every accuracy in
`src/okh_rag/evaluation/answers.py` is returned beside the count it divides by, and why
`reports/generation_eval.md` prints a denominators table under the accuracy table.

TF, MC and DE are unaffected: 0 unscorable out of 5,812 / 37,813 / 24,094.

## 6. What is answerable, and what is not

Re-measured figures, from `reports/cyportqa_audit.md`:

| | count | share |
|---|---:|---:|
| questions | 117,178 | — |
| with a non-placeholder answer | 108,841 | 92.9% |
| shipped answer is `"None"` or `""` | 8,337 | 7.1% |
| whose scenario has an empty `operation` block (so family 10 is absent) | — | 88% of horizon blocks |

Reconciling §4.1's "2,917 scenarios": the release holds **472** scenario keys
(`STORM_YEAR_PORT`), each with 1–10 horizon blocks, for **2,910** blocks in total. §4.1's
number matches the *block* count, not the key count, so §4.1 is using "scenario" for a
(storm, port, year, horizon) quadruple. The port and storm counts do **not** reconcile — §4.1
says 145 and 90, the release has 45 and 69 — and the release's numbers are the ones this
project reports.

## 7. Known defects

The released files are internally inconsistent. The registry lives in
`cyportqa.KNOWN_DEFECTS` and `scripts/audit_cyportqa.py` re-measures every entry, so the
numbers below can be checked rather than believed. Full detail in
`reports/cyportqa_audit.md`.

| id | severity | measured |
|---|---|---|
| `qa-answer-placeholder-none` | label | 7,660 answers are the string `"None"`, 677 are `""` — and 4,002 come from the encoding's own `"None"` value, so they are a source-data gap, not a generation failure |
| `template-q36-answer-field-mismatch` | label | Q36's shipped answer equals `start_impact_date` in **2,910/2,910** cases, not the `day_of_recover` its template declares |
| `ordinal-buckets-overlap` | label | Q27's options make only **60.5%** of its questions decidable; Q24 86.6%, Q25 86.1% |
| `imbalanced-fragments-unterminated` | **blocking** | the two fragments concatenate to a stream that never returns to brace depth 0 |
| `qa-context-arity-is-variable` | format | 31,945 records are arity-3 |
| `mc-options-duplicated-and-inconsistent-with-prompt` | format | 2,851 questions state a stale option count of 4 while offering 7 |
| `raw-categorical-fields-not-normalised` | format | 130+ distinct `category` strings |
| `qa-q33-answer-is-python-repr` | format | Q33's "JSON" answer is a Python `repr` with single quotes |
| `paper-counts-exceed-release` | format | §4.1 says 145 ports / 90 storms; the release has 45 / 69 |
| `td-answers-are-verbatim-field-copies` | *simplification* | all 24,094 description answers are byte-identical to their source field |
| `entity-id-has-no-year` | **blocking** | §A.2 step 2's id template `{family}:{storm}:{port}:{horizon}` omits the year, and 4 (storm, port) pairs repeat across years — `CLAUDETTE` at each of four ports in both 2015 and 2021 |
| `stray-commentary-keys-in-blocks` | format | 10 model comments stored as *keys* of a block, across 7 scenarios — `_comment`, `_note`, `_comment2`, and prose sentences beginning `//` or `--` |
| `displaced-weather-block` | format | `FLORENCE_2018_New_York,_NY_&_NJ` T-84 nests the whole `weather_observation` mapping inside `cyclone_condition_and_forecast` while the real block is empty, so 24 weather fields arrive under the wrong prefix |
| `nu-format-tags-disagree-with-answers` | format | **19,574 of 49,459** SA (`NU`) golds are not a single numeric value, so §4.1's tolerance test cannot be applied to 39.6% of the SA column — see §5.4 for the seven-way breakdown |

Three of these change how results must be reported:

**`nu-format-tags-disagree-with-answers` bounds the SA column the way
`ordinal-buckets-overlap` bounds the MC one.** §4.1 scores SA by tolerance against the shipped
answer, and for 39.6% of SA questions that answer is a coordinate pair, a date, a per-horizon
mapping or a placeholder — none of which a tolerance can be applied to. The full breakdown is in
§5.4. An SA accuracy is only comparable to another run's if both name the denominator they used.

**Q36 is the one that will silently poison a reproduction.** §4.1 scores SA questions by
tolerance-based accuracy against the shipped answer. For Q36 that answer is a date taken from
the wrong field, so a correct model scores 0 and an incorrect one can score 1. The
`template-q36-answer-field-mismatch` entry documents it; a headline number should either drop
Q36's 2,910 records or report them separately.

**`ordinal-buckets-overlap` bounds what accuracy can mean.** `TemplateAudit.determinacy` is
reported alongside `agreement` for exactly this reason: for Q27, 39.5% of questions have no
determinate answer, so a model's error on them is not evidence of a reasoning failure. §4.1
scores these by exact letter match, so part of CyPortQA's headline accuracy is decided by an
ambiguity the field cannot resolve — and a reproduction that reports only accuracy inherits
that.

The last three are structural, are not in `KNOWN_DEFECTS`, and are re-measured by
`evaluation/reference_graph.py` and `tests/test_oracle.py` rather than by the audit script —
they concern the *hypergraph* built from the encoding, which the audit does not construct.

**`entity-id-has-no-year` is the dangerous one, and it is silent.** §A.2's own example is
`wind_fcst:IRMA:port_arthur:T-48` — a bare storm name. That is unambiguous only while storm names
are unique, and CyPortQA's key space is `STORM_YEAR_PORT`: 69 storms over 9 years, in which names
repeat. When two scenarios collide, their entity ids collide, so their §A.2 step-5 hyperedge
hashes collide whenever the encoded values also coincide, and `add_hyperedge` lets the second
overwrite the first — while *both* groups keep listing the id. The corpus loses a fact **and** the
survivors get attributed to the wrong knowledge group, which §A.7's group-aware expansion and
§A.6 Eq. 12 then match across. The reference builder qualifies the storm with its year (not the
port — `HypergraphIndex.neighbours` needs a port shared by every storm that affected it), which
drives the count to **0 aliased ids over all 248,652 hyperedges**; a non-zero reading means the
convention and the corpus have drifted apart again. 293 hyperedges and 114 whose gold set came
back empty were the pre-fix symptom.

**The other two cost facts, not correctness.** Stray keys and the displaced block are each a field
that becomes no hyperedge, and they are counted separately from a genuinely absent field because
"the question's gold set is empty" is otherwise indistinguishable between the release's ragged
scenarios and a misencoding. The displaced block is currently *unreachable*: no T-84 QA record at
`FLORENCE_2018` uses a weather template, so it has no effect on any reported number — a
constructed Q2 at that horizon does return `EMPTY`, which is how the defect is pinned.

## 8. Validating your copy

```bash
D:/Anaconda/python.exe scripts/audit_cyportqa.py --data data/raw/cyportqa --out reports
```

This reads all four files, joins them, audits all 48 templates against their declared fields,
re-measures all nine defects, and writes `reports/cyportqa_audit.md` +
`reports/cyportqa_audit.json`. It calls no model and needs no API key.

Expected output: every defect `reproduced`, and every template at agreement ≥ 0.99 except Q36
(0.003), Q25 (0.948) and Q18 (0.978) — the three whose templates the dataset itself
mis-specifies. Any other template below 0.99 means the loader or the comparator has drifted,
not the data.

The retrieval oracle of §5.3 is validated by a second command, which also needs no API key:

```bash
D:/Anaconda/python.exe scripts/run_retrieval_oracle.py --retrieve 0
```

This builds the reference hypergraph, resolves every template against §3.2, and classifies all
117,178 questions by whether the oracle can score them — writing `reports/retrieval_oracle.md` +
`.json`. `--retrieve 0` runs only the model-free stages. Expected: 48/48 templates covered,
0 aliased hyperedge ids, 0 provenance mismatches, and a 95.0% scorable rate; the numbers in §5.3
are the ones it prints. Drop `--retrieve 0` (with `--backend openai`) to add §A.7 retrieval and
the §4.5 shuffle ablation.

The scoring side of §5.4 is validated by a third command:

```bash
D:/Anaconda/python.exe scripts/run_generation.py \
  --sources okh_rag hypergraph_rag text_rag graph_rag \
  --transition-variant heuristic --scenario-limit 8 --retrieve 200
```

This runs all four §4.1 methods over one question set with one generator, writing
`reports/generation_eval.md` + `.json`. `--backend lexical` (the default) substitutes a
token-overlap embedder and a stub generator that never reads the evidence, so **the accuracies it
prints are not §4.1's** — the run checks the wiring, and the report says so in its first line.
Expected from it: four identical accuracy rows (the stub cannot distinguish the methods), 0
non-conforming answers, and a denominators table whose SA row shows roughly 60% scorable. Replace
`--backend lexical` with `--backend openai` and drop `--transition-variant heuristic` once a
trained `P_θ` exists to get §4.1's table; add `--shuffle` for §4.5 Table 2's Shuffle row.

## 9. Citation

```bibtex
@misc{kuai2025cyportqa,
  author        = {Chenchen Kuai and Chenhao Wu and Yang Zhou and Xiubin Bruce Wang and
                   Tianbao Yang and Zhengzhong Tu and Zihao Li and Yunlong Zhang},
  title         = {CyPortQA: Benchmarking Multimodal Large Language Models for Cyclone
                   Preparedness in Port Operation},
  year          = {2025},
  eprint        = {2508.15846},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2508.15846}
}
```

Released under the MIT Licence; the upstream `LICENSE` is vendored at
`data/raw/cyportqa/LICENSE`. Under §4.1 the benchmark is used as-is — no filtering, no
subsampling, no re-splitting — and the split this project trains on is defined by
`configs/base.yaml:training.split`, grouped by scenario so that no entity is shared across the
train/test boundary.
