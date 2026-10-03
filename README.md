# OKH-RAG 复现：如何使用论文中的完整数据集做实验

本仓库复现论文 **《Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models》**
(arXiv:2604.12185)。本文件回答一个具体问题：**论文 §4.1 提到的那个完整数据集是什么、怎么拿到、
怎么用它把实验从头跑到尾。**

一句话概括：论文用的数据集是公开基准 **CyPortQA**（117,178 个问题 / 472 个场景），已经 vendored 在
`data/raw/cyportqa/`；跑完整实验只需要三条命令：**体检 → 训练 P_θ → 全量生成评测**。

---

## 1. 数据集：论文 §4.1 用的是 CyPortQA

论文 §4.1 没有自己造数据，它直接在 CyPortQA 上评测。两者的作者高度重叠（Kuai、Zhou、Tu、Li 同时在两篇
作者列表里），所以不存在"论文说的"和"发布出来的"是两套东西的问题。

| 项 | 值 |
|---|---|
| 数据集 | CyPortQA（MIT 许可） |
| 上游仓库 | <https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA> |
| 基准论文 | Kuai et al., arXiv:2508.15846 |
| 问题总数 | **117,178**（与 §4.1 "117,178 questions" 完全一致） |
| 场景（storm_year_port） | **472** 个场景键，**2,910** 个 horizon block（T-12 … T-120，12 小时一格） |
| 风暴 / 港口 | 69 个风暴 / 45 个港口（§4.1 写 90 / 145，与发布版不符，以发布版为准） |
| 答案格式 | `TF` 5,812 / `MC` 37,813 / `NU` 49,459 / `DE` 24,094 |
| 构建出的参考超图 | **248,652** 条超边、**139,102** 个实体 |

> **一个容易踩的坑**：论文把答案格式写成 TF / MC / SA / TD，而数据里的标签是 `TF` / `MC` / `NU` / `DE`。
> `SA` 在数据里是"能力维度"标签，不是格式标签。映射关系是 TF→`TF`、MC→`MC`、SA→`NU`、TD→`DE`。
> 完整的 schema、字段含义和缺陷清单见 **`data/CONTRACT.md`**（本项目的权威数据契约）。

### 目录里有什么

```
data/raw/cyportqa/
├── dataset/CyPortQA.json              101 MB   117,178 条 QA 记录
├── source_data/Encoded_senario.json   9.6 MB   结构化场景编码 —— 超图的事实来源
├── source_data/CyPortQA_template.json  41 KB   48 个模板 + 每个模板的答案字段路径（答案 oracle）
├── raw_text/                          5.9 MB   NOAA 文本公告 + USCG 港口状态（可选，文本模态）
└── LICENSE / README.md                         上游原样保留
```

**没有 vendored 的是图形树**（2,031 张 PNG/JPEG，157 MB）。OKH-RAG 是纯文本方法，§4.1 给所有方法配的都是
`gpt-4o` + `text-embedding-3-small`，都不是视觉模型，所以图像数据是**有意**排除的，不是漏掉的。

关键点：`Encoded_senario.json` 已经把场景写成结构化形式，因此本项目的 §A.2 提取步骤**不需要调用 LLM**——
"完美的提取器会产出什么"有闭式解。这是完整数据集能离线跑通大部分流程的原因。

---

## 2. 环境准备

```bash
pip install -r requirements.txt        # torch / numpy / pyyaml / openai / python-dotenv
```

脚本自己会把 `src/` 加入 `sys.path`，所以**不需要** `pip install -e .`（想装也可以）。
Windows + Anaconda 下 torch 与 numpy 的 `libiomp5md.dll` 冲突（"OMP: Error #15"）已由脚本在导入前设置
`KMP_DUPLICATE_LIB_OK` 处理，无需手动干预。

配置 LLM 凭据（脚本用 `python-dotenv` 自动加载仓库根目录的 `.env`）：

```bash
cp .env.example .env                   # Windows: Copy-Item .env.example .env
```

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://your-relay.example.com/v1     # 直连官方时可留空/删除
```

你指向的端点**必须同时提供**两个模型：`gpt-4o`（生成 / 判定）和 `text-embedding-3-small`（嵌入，1536 维）。

---

## 3. 获取并校验完整数据集

`data/raw/` 被 `.gitignore` 排除（233 MB，一条命令可重建），但当前工作副本里**已经备齐了**。
若你的副本是空的，或换机器重来，执行（`data/CONTRACT.md` §2 原文）：

```bash
git clone --depth 1 https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA.git
mkdir -p data/raw/cyportqa/{dataset,source_data/CyPortQA_Imbalanced,raw_text}
cp MLLM-Bench-CyPortQA/dataset/CyPortQA.json                    data/raw/cyportqa/dataset/
cp MLLM-Bench-CyPortQA/source_data/Encoded_senario.json          data/raw/cyportqa/source_data/
cp MLLM-Bench-CyPortQA/source_data/CyPortQA_template.json        data/raw/cyportqa/source_data/
cp MLLM-Bench-CyPortQA/source_data/CyPortQA_Imbalanced/*.json    data/raw/cyportqa/source_data/CyPortQA_Imbalanced/
cp MLLM-Bench-CyPortQA/{LICENSE,README.md}                       data/raw/cyportqa/

cp -r "MLLM-Bench-CyPortQA/source_data/NOAA NHC Cyclone Products" data/raw/cyportqa/raw_text/
find data/raw/cyportqa/raw_text -path "*Cyclone Graphics Archive*" -delete
mkdir -p data/raw/cyportqa/raw_text/Port_Condition_Bullitens
cp "MLLM-Bench-CyPortQA/source_data/Port_Condition_Bullitens/uscg_port_conditions_agg.csv" \
   data/raw/cyportqa/raw_text/Port_Condition_Bullitens/
```

**校验你的副本**（不调用任何模型，不需要 API key，约 100 秒）：

```bash
python scripts/audit_cyportqa.py --data data/raw/cyportqa --out reports
```

应看到：`scenarios=472` `horizon_blocks=2910` `questions=117178`、格式分布
`{'DE': 24094, 'MC': 37813, 'NU': 49459, 'TF': 5812}`、10 个已知缺陷全部 `reproduced`，
模板一致率除 Q36(0.003) / Q25(0.948) / Q18(0.978) 外均 ≥ 0.99。写出的报告在 `reports/cyportqa_audit.md`。
**任何一项对不上，说明数据没拿全或拿错了版本，先别往下跑。**

---

## 4. 三档运行规模

| 档位 | 命令 | API key | 用途 |
|---|---|---|---|
| 体检 | `scripts/audit_cyportqa.py` | 不需要 | 验证数据、重测缺陷、审计 48 个模板 |
| 离线接线检查 | `--backend lexical` | 不需要 | 用 token-overlap 替身嵌入 + 桩生成器，验证管线通断 |
| §4.1 完整复现 | `--backend openai` | 需要 | 论文 Table 1 / Table 2 的真实数字 |

`--backend lexical` 打印的准确率**不是**论文数字，报告首行也会写明；它只用来确认管线没断。

---

## 5. 完整实验流水线（§4.1 全量）

下面所有命令都从仓库根目录执行。`0` 在 `--scenario-limit` / `--retrieve` 里一律表示"**全量**"。

### Stage 0 · 预检（强烈建议，约 2.5 分钟，不花 API 钱）

```bash
python scripts/preflight_full.py --scenario-limit 0 --retrieve 0
```

在建图、调 API 之前先告诉你三件事：抽样的组成、超图规模、嵌入缓存命中数。当前实测输出：

```
questions : 117178        scenarios : 472        templates : 48        storms : 69
formats   : {'DE': 24094, 'MC': 37813, 'NU': 49459, 'TF': 5812}   # 与论文一致
REFERENCE GRAPH built in 21.8s
  hyperedges : 248652     entities : 139102     groups : 472
  matrix     : 1.53 GB float32
  cached     : 248652/248652            # 嵌入已全量缓存，热跑 0 次嵌入调用
GENERATION  projected for 117178 questions: 81.4 h
```

### Stage 1 · 参考超图 + 嵌入（自动完成，无需单独命令）

`Encoded_senario.json` → 每条"已填值字段"一条超边 + §A.2 step 4 合成的 cross-horizon 边（113,404 条，
占 45.6%，对应 §4.2 的"近一半"）。嵌入走内容寻址缓存：

- 超边向量：`artifacts/llm_cache/embeddings`（256 分片，1.6 GB，**当前已覆盖全部 248,652 条**）
- 问题向量：`artifacts/llm_cache/questions`

所以全量热跑**不产生任何嵌入调用**。千万别把 `store=None` 传进去——那会把整个语料重新嵌入一遍。

### Stage 2 · 训练顺序模型 P_θ（§3.1 Eq. 3–4 / §A.4）

```bash
# 48 个场景的试点（便宜，先验证 learned 路径能跑通）
python scripts/train_order_model.py --scenario-limit 48

# 全量：472 个场景 → artifacts/order_model.pt（run_generation.py 默认加载它）
python scripts/train_order_model.py --scenario-limit 0
```

当前 `artifacts/order_model.pt` 是全量训练产物（seed 42，train/val/test = 377/47/48，5 个 epoch 早停，
304,856 个正样本对；`P_doc` 196,016 + `P_ent` 108,840，`P_ret` 未训练——它需要"正确答案的检索轨迹"，
即先跑一遍生成）。训练明细见 `artifacts/order_model.train_report.json`。

### Stage 3 · 全量生成评测（§4.1 Table 1）——真正的花销所在

```bash
python scripts/run_generation.py --backend openai \
  --scenario-limit 0 --retrieve 0 \
  --dump reports/table1.jsonl --progress-every 1000
```

产出 `reports/generation_eval.md` + `.json`，以及逐题的 `reports/table1.jsonl`。

### Stage 4 · 检索 oracle（无需生成的检索评测）

```bash
python scripts/run_retrieval_oracle.py --backend openai --scenario-limit 0 --retrieve 0
# 只想离线验证接线：python scripts/run_retrieval_oracle.py --retrieve 0
```

产出 `reports/retrieval_oracle.md` + `.json`。因为每个模板都声明了答案字段，检索质量可以**独立于生成**
被打分——这是发布版数据带来的最大方法学收益，也是 §4.1 的端到端准确率做不到的。离线档期望值：
48/48 模板覆盖、0 个 id 冲突、0 处 gold 不匹配、95.0% 可评分。

### Stage 5 · 消融（§4.5 Table 2）

```bash
python scripts/run_generation.py --backend openai --scenario-limit 0 --retrieve 0 --shuffle   # Shuffle 行
python scripts/run_generation.py --backend openai --scenario-limit 0 --retrieve 0 --transition-variant heuristic
python scripts/run_generation.py --backend openai --scenario-limit 0 --retrieve 0 --transition-variant none
```

`--shuffle` 走的是 Table 2 harness 自己的代码路径（改 `ablation.shuffle_trajectory` 后重新打分），
不是驱动层自己洗牌——否则同样的洗牌动作会跑出不同的跌幅，无法说清哪个才是 Table 2。

### 小规模试跑（推荐先做）

```bash
python scripts/run_generation.py --backend openai --scenario-limit 48 --retrieve 200
```

抽样是**带种子的随机抽样，不是截断前缀**（旧实现把 200 题全落在 ALEX_2022 的两个港口里，一行表格是在
单一风暴上估出来的）。同样的种子下，小 `n` 是同一总体的低精度测量，不是另一个总体。

---

## 6. 参数速查

| 参数 | 默认 | 含义 |
|---|---|---|
| `--scenario-limit` | 0 | 建图用的场景数，**0 = 全部 472**；非 0 时按 `--seed` 随机抽样 |
| `--retrieve` | 120（oracle 120 / 生成 120） | 回答多少问题，**0 = 全部 117,178** |
| `--backend` | `lexical` | `lexical` 离线替身 / `openai` 论文真实模型 |
| `--transition-variant` | 配置里的 `learned` | `learned`（需 checkpoint）/ `heuristic`（§A.3 规则）/ `none`（纯相关性） |
| `--order-model` | `artifacts/order_model.pt` | 指定 P_θ checkpoint 路径 |
| `--shuffle` | 关 | Table 2 的 Shuffle 行 |
| `--seed` | 42 | 抽样与训练种子 |
| `--dump` | 无 | **逐题追加写 JSONL**。全量跑务必加上：中断不会丢掉已付费的答案 |
| `--progress-every` | 100 | 每 N 题打印进度与 ETA（0 = 静默） |
| `--cache-root` | `artifacts/llm_cache` | 嵌入缓存根目录（超边/问题各一个子目录） |
| `--mode` | `single_call` | §A.8 的生成模式，备选 `per_trajectory` |

超参数一律在 `configs/base.yaml`（λ=1.2, µ=0.3, ν=0.2, ρ=0.5；K=80；\|C\|=150；B=8；L=8；N=3）。
每一条都带 `§` 引用、`[UNSPECIFIED]`（论文没写）或 `[CONTRADICTION]`（论文自相矛盾）标记，
`src/` 里不放魔法数字。

---

## 7. 成本与时间预算

| 项 | 量级 |
|---|---|
| 语料加载 | ~5 s |
| 参考超图构建 | ~22 s（248,652 条超边） |
| 嵌入矩阵 | 248,652 × 1536 float32 = **1.53 GB** 常驻内存 |
| 嵌入调用 | **全量已缓存 → 0 次**；冷启动需重新嵌入 248,652 条 |
| 生成 | gpt-4o 上 2.5–3.0 s/题 → 全量 **≈ 81 小时**（驱动是单线程） |

内存跟着样本走，不跟着数据集走：40 道题跨 36 个场景只建 22,776 条超边；全量才建满 248,652 条。
全量生成跑不完是常态，所以**一定要 `--dump`**：它是逐题 flush 的追加写，第 70 小时中断也保住前 70 小时。

---

## 8. 产物在哪里

| 文件 | 由谁产生 | 内容 |
|---|---|---|
| `reports/cyportqa_audit.md` / `.json` | Stage 0 体检 | schema、48 模板审计、10 个缺陷重测 |
| `reports/generation_eval.md` / `.json` | Stage 3 | Table 1 四列准确率 + 分母表 + micro/macro |
| `reports/*.jsonl` | `--dump` | 逐题答案与得分（可断点续算） |
| `reports/retrieval_oracle.md` / `.json` | Stage 4 | §A.7 各阶段检索指标 + §4.5 shuffle 对照 |
| `artifacts/order_model.pt` | Stage 2 | P_θ checkpoint |
| `artifacts/llm_cache/` | 自动 | 嵌入缓存（1.6 GB） |

回归测试：`python -m pytest tests -q`。

---

## 9. 读结果前必读：数据集自带的坑

这些不是本项目的 bug，是发布版数据的性质；**不处理会让你的数字不可比**。细节在 `data/CONTRACT.md`
§5.4 / §7 和 `reports/reproduction_gap.md`。

1. **SA 列只有 60.4% 可评分。** 49,459 道 `NU` 题里有 **19,574 道**（39.6%）的 gold 不是单个数值
   （坐标对、日期、per-horizon 映射、占位符），容差检验无从施加。分母随准确率一起报告，
   `generation_eval.md` 里的分母表就是干这个的。TF / MC / DE 三列无此问题（0 不可评分）。
2. **Q36 会静默污染结果。** Q36 的 gold 取自 `start_impact_date`，而它模板声明的是 `day_of_recover`
   （2,910/2,910 都对不上）。正确的模型拿 0 分。报告 headline 时要么剔除 Q36，要么单列。
3. **TD 默认按归一化精确匹配打分**，论文用的是 GPT-4o 语义判定。全部 24,094 条 DE gold 都与其源字段
   逐字节相同，所以判定器对 gold 无事可做；但对**模型的回答**它有用（模型给的是改写后的简短表述）。
   想要论文的 0.441 量级，设 `evaluation.td_scoring: "llm_judge"`。
4. **MC 有一部分本质上不可判定。** Q27 只有 60.5% 的题有确定答案（选项区间重叠），Q24 86.6%、Q25 86.1%。
   这部分错误不是推理失败。
5. **基线已被移除**（commit `fabe893`），Table 1 的 Text-RAG 0.287 / GraphRAG 0.414 / HyperGraphRAG 0.511
   三行目前**无法**复现，因此论文"+0.023"的核心主张暂不可验证。
6. **顺序模型存在训练/评测泄漏**：当前 P_θ 在 377/472 个场景上训练，评测场景可能落在训练集里，
   所以现有数字不是 held-out 数字。若要严格，按 `configs/base.yaml:training.split` 的分组拆分重训并只评测 test 组。
7. **超边 id 必须带年份**：§A.2 的 id 模板漏了年份，而 (storm, port) 组合会跨年重复（如 CLAUDETTE 在 2015
   与 2021 各出现于 4 个港口）。参考构建器补上了年份，实测 **0 个 id 冲突**；非 0 说明约定与语料又漂移了。

---

## 10. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `OPENAI_API_KEY is not set` | 训练脚本直接退出。填 `.env` 或导出环境变量 |
| 400 `'response_format' of type 'json_schema' is not supported` | 中转站不支持 Structured Outputs。客户端会自动降级；也可显式设 `OKH_RAG_RESPONSE_FORMAT=object`（见 `.env.example`）。代价是约 13.5% 的回答违反 §A.8 契约 |
| `transition_variant='learned'` 但报缺 checkpoint | 先跑 Stage 2，或改用 `--transition-variant heuristic` |
| `OMP: Error #15` | 脚本已设 `KMP_DUPLICATE_LIB_OK`；若自己写入口，需在导入 numpy/torch 前设置 |
| 内存爆掉 | 全量矩阵 1.53 GB 常驻。先用 `--scenario-limit 48` 试跑，确认峰值后再上全量 |
| 全量跑到一半断了 | 这就是 `--dump` 存在的理由。重跑时嵌入命中缓存，只重做生成 |

---

## 11. 引用与许可

```bibtex
@misc{kuai2025cyportqa,
  author        = {Chenchen Kuai and Chenhao Wu and Yang Zhou and Xiubin Bruce Wang and
                   Tianbao Yang and Zhengzhong Tu and Zihao Li and Yunlong Zhang},
  title         = {CyPortQA: Benchmarking Multimodal Large Language Models for Cyclone
                   Preparedness in Port Operation},
  year          = {2025},
  eprint        = {2508.15846},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

CyPortQA 为 MIT 许可，上游 `LICENSE` 存在于 `data/raw/cyportqa/LICENSE`；官方评测代码保留在
`third_party/cyportqa_official/`（含 §4.1 的判定 prompt 原文）。§4.1 下数据集**原样使用**：不过滤、
不二次抽样、不重新切分；训练切分由 `configs/base.yaml:training.split` 定义，按场景分组，
保证没有实体跨越训练/测试边界。
