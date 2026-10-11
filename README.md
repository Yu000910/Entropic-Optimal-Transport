# Entropic-Optimal-Transport

Reference implementation and data release for the manuscript

> **Zero-Shot Adversarial Technique Extraction from Cyber Threat Intelligence via Entropic Optimal Transport and Logical Verification**

The pipeline identifies MITRE ATT&CK techniques in unstructured cyber threat
intelligence (CTI) reports **without any task-specific parameter training**. It
combines four stages:

| Stage | Module | Role |
|---|---|---|
| 1 | **CASD** — Context-Aware Semantic Decoupling | decomposes a report into atomic adversarial actions via a constrained LLM prompt |
| 2 | **FSD** — Feature-Space Denoising | projects action embeddings away from a generic-vocabulary subspace estimated by PCA |
| 3 | **EOT** — Entropic Optimal Transport | aligns the set of actions to a cosine-screened candidate pool through a Sinkhorn transport plan |
| 4 | **SLV** — Structured Logical Verification | scores the surviving candidates against an explicit rubric and fuses the scores |

---

## 1. Data

| Artefact | Location | Description |
|---|---|---|
| **BEDR** reference knowledge base | `BEDR_dataset.csv` | 21,453 annotated technique–sentence pairs over 679 technique and sub-technique classes |
| **CTI-1002** evaluation corpus | `CTI_reports/` | 1,002 pre-cleaned CTI reports with technique-level ground-truth annotations (`*_ground_truth.json`) |
| **Fixed partition** | `partition_index.json` | the 802 training / 200 test split (random seed 42) used in every experiment reported in the paper |

The BEDR knowledge base is also archived separately at
<https://github.com/Yu000910/BEDR>.

### Ground truth

Each `*_ground_truth.json` file contains the report identifier, its cleaned text
and the list of ATT&CK technique identifiers it exercises
(`actual_found_ids`).

---

## 2. Installation

```bash
git clone https://github.com/Yu000910/Entropic-Optimal-Transport.git
cd Entropic-Optimal-Transport
python -m pip install numpy pandas scipy scikit-learn tqdm openai
```

### Credentials

API keys are read **only** from environment variables; no key is stored in this
repository.

```bash
export DASHSCOPE_API_KEY=sk-...   # Alibaba Cloud DashScope, for text-embedding-v4
export DEEPSEEK_API_KEY=sk-...    # DeepSeek, for the CASD and SLV prompts
```

### Paths

All paths resolve relative to this repository, so a fresh clone works without
editing any file. Override with environment variables if your data lives
elsewhere:

```bash
python config.py     # prints a path self-check
```

| Variable | Purpose |
|---|---|
| `EOT_DATA_ROOT` | root directory of the data files (defaults to the repository root) |
| `BEDR_CSV`, `CTI_REPORTS_DIR` | override the two data paths individually |
| `BEDR_EMB_CACHE` | path to the cached BEDR prototype embeddings (`.npy`) |
| `PARTITION_INDEX` | path to `partition_index.json` |
| `ATTACK_JSON` | ATT&CK STIX definition, needed only by the auxiliary ablation scripts |

---

## 3. Reproducing the reported results

```bash
# 0) build the prototype-embedding cache (once; requires DASHSCOPE_API_KEY)
python build_embeddings.py

# 1) the full pipeline on the held-out test partition (200 reports)
python run_full_pipeline.py --limit 200 --tag full200

# 2) the component ablation (single-module / cumulative / complementary)
python run_ablation.py

# 3) the supervised and retrieval-augmented baselines
python run_baseline_tfidf_svm.py
python run_baseline_ttpfshot.py
python run_baseline_techniquerag.py
python run_baseline_tief.py

# 4) the statistical validation (bootstrap CIs, paired Wilcoxon test, Friedman test)
python bootstrap_ci.py --a results/full_pipeline_full200.json

# 5) the external validation on the TRAM benchmark
python run_tram_eval.py
```

Results are written to `results/` as JSON.

### Partition handling

`splits.py` supports three modes:

| mode | meaning |
|---|---|
| `random` | draw *n* reports with a fixed random seed (the protocol of §4.1.3) |
| `first_n` | take the first *n* reports by sorted file name |
| `index`  | read the fixed partition from `partition_index.json` |

---

## 4. Repository layout

```
config.py                     paths, hyper-parameters, credential loading
splits.py                     train / test partitioning
bedr_data.py                  BEDR loading and prototype alignment
metrics.py                    Micro-F1, Macro-F1, Recall@K
llm_clients.py                LLM back-end wrappers
build_embeddings.py           prototype-embedding cache construction
run_full_pipeline.py          the four-stage pipeline (CASD → FSD → EOT → SLV)
run_ablation.py               component-wise ablation
run_baseline_*.py             supervised and retrieval-augmented baselines
bootstrap_ci.py               statistical validation
run_tram_eval.py              external validation on TRAM
partition_index.json          the fixed 802 / 200 partition
BEDR_dataset.csv              reference knowledge base
CTI_reports/                  evaluation corpus and ground truth
```

---

## 5. Notes on the implementation

* **Prototype alignment.** The cached prototype matrix has 21,454 rows while the
  CSV holds 21,453 records: row 0 is the vector of the CSV header line.
  `BEDR_EMB_OFFSET = 1` in `config.py` applies the correct alignment and is
  verified by re-embedding (`sim = 1.0000`).
* **Hyper-parameters.** `N_cand = 30` candidates per atomic action,
  `K = 5` candidates passed to logical verification, EOT regularisation
  `γ = 50` with at most 100 Sinkhorn iterations, fusion weight `α = 1.0`, and
  FSD noise-subspace dimension `k = 1`. These correspond to the configuration
  reported in §4.2.4 of the manuscript.
* **Prompt templates.** The two instruction protocols used in Stages 1 and 4
  (system role, task definition, rubric weights, JSON output schema) are given in
  the Supplementary Material of the manuscript.

---

## 6. Citation

If you use this code or data, please cite the manuscript above.

## 7. Contact

Yu Fengrui — People's Public Security University of China
