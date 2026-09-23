# Provenance — `third_party/cyportqa_official/`

Verbatim copies of the CyPortQA benchmark's own harness, kept so that the reproduction's
evaluation layer and baseline protocol rest on the released code rather than on inference
from prose. Nothing here is modified; nothing here is ours.

| | |
|---|---|
| Source | <https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA> |
| Commit | `1c38abf339d1471d710f6cb719a5e3874b547077` (`main`, 2025-11-12) |
| Licence | MIT — Copyright (c) 2025 CyPortQA (`LICENSE`, included) |
| Vendored | 2026-09-23, copied from a `--depth 1` clone before that clone was deleted |
| Size | 15 files, 35,967 bytes |
| Verified | every file's sha256 matches the source clone at copy time |

## Why these files, and not others

The upstream repository is ~392 MB. Almost all of it is data that this project vendors elsewhere
(`data/raw/cyportqa/`, see `data/CONTRACT.md` §2) or excludes on scope grounds (the graphics
trees). What is *not* anywhere else is the harness — 13 non-image files that would have been lost
when the clone was removed. They are kept here because they answer questions the paper leaves open:

* **`evaluators/prompt_*.txt`** — three prompt templates, one per ability category
  (`CycloneStatusAndForecastArea` = S1, `Port-LevelHazardandForecastContext` = S2,
  `PortOperationalGuidance` = S3). Together they are the concrete form of §4.1's "LLM-judged
  semantic score in [0, 1] using GPT-4o": an **ordinal** rubric
  (`1.0 / 0.8 / 0.6 / 0.4 / 0.2 / 0.0`) applied per `Answer_LLM*` field against
  `Answer_Ground_Truth`, plus a **hallucination rule** that forces `score = 0.0` when invented
  data appears (`0.2` if trivial). Output is JSON-Lines, one minified object per input row.
  Two consequences for this reproduction: the score is *not* a smooth similarity, and a single
  invented fact can zero an otherwise good answer. The project's
  `evaluation.judge_model` / `td_scoring` settings are calibrated against these files.
* **`run.py` + `config.json` + `Requirements.txt`** — the released multi-model harness.
  `config.json` selects which runners to launch and passes through `--start/--end/--dataset/--output`;
  `run.py` shells out per model and tees logs into `logs/`.
* **`models/run_*.py`** — the seven released zero-shot runners: GPT-4o, Gemini Flash, Gemma 3,
  LLaMA 3.2, LLaVA 1.6, Mistral, Qwen2.5-VL. These fix the released protocol, which matters more
  than the numbers they produce. See `data/CONTRACT.md` §2.2 for the three facts they establish:

  1. `storm, year, leadtime = Q["context"][-3:]` — the context array is read from the end, which
     independently confirms `QAContext.parse` over all 117,178 records (0 disagreements).
  2. `Q["prompt"]` is the **system** message and `Q["question"]` the user turn.
  3. input files are `f"{storm}_{year}_{leadtime}h.{txt|png}"`.

   They also expose §2.3's defect: the `modalities` values in `CyPortQA.json` are not the directory
   names the files sit in, so `run_gpt4o.py` raises `FileNotFoundError` on every text modality
   against the shipped tree. Reproducing the baseline needs the rename map in §2.3.

## Known upstream nits (not ours to fix, recorded so nobody re-debugs them)

* `models/run_gpt4o.py` imports `langchain_core.messages` and never uses it, so the script fails at
  import time in an environment built only from `Requirements.txt` (`langchain-core` is not in it).
* The README documents `pip install -r requirements.txt` but the file is tracked as
  `Requirements.txt` — case-sensitive on Linux.
* `models/run_gpt4o.py` calls `random.shuffle(qa)` before iterating, so the released runners are
  not deterministic across runs and their per-event CSVs are not reproducible without seeding.
* The README's repository tree lists `models/run.py`; `run.py` actually sits at the repository root
  and expects to be invoked from there (`model_dir = Path("models")`).

## Scope

Do not edit files in this directory. They are upstream snapshots; a local fix would silently
diverge from the released benchmark and destroy the reason for keeping them. If a change is needed,
copy the file out, adjust the copy, and record the delta.
