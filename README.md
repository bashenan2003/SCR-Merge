# SCR-Merge (SkillSCR)

**Verification-Gated Merging of Semantically Conflicting Agent Skill Documents across Multiple Optimization Trajectories**

SCR-Merge is a framework for merging multiple independently-optimized agent skill
documents that share a common base but diverge in semantically conflicting
directions. Given a base document `S0` and `K` optimization trajectories
`S1 … SK` produced by a skill-optimization loop (e.g., SkillOpt), SCR-Merge
detects semantic conflicts, searches a principled compromise, and gates every
candidate through a verification stage that guarantees no trajectory's
validation score degrades below the base document.

## Highlights

- **Semantic conflict detection (Types I–IV)** using RoBERTa-large-mnli
  entailment and Sentence-BERT embeddings — no text-level diff heuristics.
- **Compromise search** as a greedy maximin over a binary edit vector with
  distance-based regularization (Rawlsian objective, no predefined inter-team
  weights).
- **Verification gating** with three independent gates: performance
  non-degradation, semantic consistency, and approximate Pareto optimality.
- **Infeasibility detection**: when trajectories cannot be productively merged,
  the framework reports infeasibility and returns `S0` instead of emitting a
  degraded compromise.
- **Fully offline mode** (template-driven rendering, CPU-only) plus an optional
  **online mode** accelerated by a GPyTorch Bayesian surrogate.

## Pipeline

The system is organized as six sequential modules:

| Module | Name | Description |
|---|---|---|
| 1 | Semantic Parser & Rule Alignment | Extracts 5-tuple rules via T5-small, aligns rules across versions with Hungarian matching, builds the semantic dependency graph |
| 2 | Efficient Conflict Detector | Classifies edit pairs into conflict Types I–IV using NLI + cosine similarity |
| 3 | Compromise Search | Greedy maximin search over the binary edit vector with field-interaction scoring and distance regularization |
| 4 | Verification Gating | Three gates: performance, semantic consistency, approximate Pareto |
| 5 | Incremental Update Engine | Updates the dependency graph and conflict matrix within graph distance 2 of each modified rule |
| 6 | NL Renderer | Renders the merged edit vector into a natural-language skill document (template / LLM / human-in-the-loop) |

## Repository layout

```
SkillSCR/
├── run_merge.py            # CLI entry point
├── requirements.txt
├── config/                 # default / offline / online configurations
├── src/
│   ├── core/               # graph, skill_document, types
│   ├── models/             # T5 / SBERT / NLI / GPyTorch wrappers
│   ├── modules/            # module1_parser … module6_renderer
│   ├── pipeline/           # orchestrator, evaluator, checkpointer
│   └── utils/              # config, hungarian, logging, math
├── demo/
│   ├── run_demo.py         # end-to-end demo
│   ├── sample_docs/        # sample S0 + trajectories
│   └── validation_set/     # sample validation cases
├── scripts/                # benchmark generation, ablation, reproduction
├── gfigures/               # paper figure scripts
├── data/                   # sample datasets (optional)
└── outputs/                # merge results
```

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd SkillSCR
pip install -r requirements.txt
```

Dependencies: `torch`, `transformers`, `sentence-transformers`, `gpytorch`,
`numpy`, `scipy`, `scikit-learn`, `networkx`, `pyyaml`, `tqdm`, `joblib`.

> **Note on model weights.** The pipeline uses `t5-small`,
> `roberta-large-mnli`, and a Sentence-BERT encoder. The first run downloads
> these from the Hugging Face Hub automatically (an internet connection is
> required once). Offline mode performs *no LLM API calls*.

## Quick start

Run the end-to-end demo on bundled sample documents:

```bash
python demo/run_demo.py                 # offline, CPU
python demo/run_demo.py --mode online   # with GPyTorch surrogate
python demo/run_demo.py --output result.txt
```

## Usage

Merge skill documents with `run_merge.py`.

**Method 1 — scan a directory** (auto-discovers `S0` and `S1 … SK` by file name):

```bash
python run_merge.py -d data/dataset2/dataset_md
```

**Method 2 — specify files explicitly:**

```bash
python run_merge.py --s0 data/dataset2/dataset_md/S0.md \
                    --s1 data/dataset2/dataset_md/S1.md \
                    --s2 data/dataset2/dataset_md/S2.md
```

Mixed formats (Markdown / JSON) and mixed languages are supported across
trajectories.

**Method 3 — custom output path:**

```bash
python run_merge.py -d data/dataset2/dataset_md -o outputs/my_result.md
```

**Method 4 — online mode (GPyTorch surrogate):**

```bash
python run_merge.py -d data/dataset1/dataset_md --mode online
```

See all options:

```bash
python run_merge.py --help
```

## Configuration

Configuration lives in `config/*.yaml` (`default.yaml`, `offline.yaml`,
`online.yaml`). Key sections:

- `thresholds` — conflict-detection thresholds (`theta_seq`, `theta_trig`,
  `theta_opp`, `theta_conflict`, `theta_cons`, …).
- `compromise` — greedy search settings (`lambda`, `max_iterations`,
  `patience`, `init_strategy`).
- `verification` — gate settings (`gate3_method`, `gate3_mc_samples`,
  `max_rollback_attempts`).
- `priority` — conflict-ordering weights (`w1`, `w2`).
- `surrogate` — GPyTorch surrogate settings for online mode.

> **Important.** The default config references absolute local paths under
> `models_base` / `sentence_transformers_src` / `gpytorch_src`. Replace these
> with paths valid on your machine (or install the libraries via `pip` and
> point the wrappers at the installed packages) before running.

## Benchmark dataset

The benchmark scenarios used for evaluation are released separately in
**[TrajMergeBench](https://github.com/<your-org>/TrajMergeBench)** — 4,579
merge scenarios (development / validation / held-out / cross-domain) with
conflict gold annotations across conflict Types I–IV. See that repository for
the scenario format and download.

## Reproducing the paper results

- Benchmark generation: `scripts/generate_mergebench_v3.py`
- Ablation study: `scripts/run_ablation.py`
- Real-agent experiment: `scripts/run_real_experiment.py`
- Table 6 reproduction: `scripts/reproduce_table6.py`
- Figure scripts: `gfigures/fig2_mergescore.py` … `fig8_efficiency.py`

# TrajMergeBench

Benchmark input scenarios for SCR-Merge. These files define the
merge tasks; they are not experimental result measurements.

## Counts

- Held-out test: 135 base + 1220 high-conflict = 1355 scenarios
- Development: 225 base + 2035 high-conflict = 2260 scenarios
- Validation: 90 base + 814 high-conflict = 904 scenarios
- Cross-domain test: 30 customer service + 30 code generation = 60 scenarios
- Total: 4579 scenarios

## Scenario format

Each scenario directory contains:

- `dataset_json/manifest.json` with scenario metadata and trajectory map
- `dataset_json/S0_base.json` plus `S1_optimized.json` through `SK_optimized.json`
- `dataset_md/` with the same documents rendered as structured Markdown

Rules use the 5-tuple format `rule_id / trigger / precondition / action /
postcondition / tools`. Trajectory JSON files also include `edit_set`, which
records every field-level edit from the base document.

## Splits

- `held_out/` contains the 1415 test scenarios
- `development/` contains the 2260 development scenarios
- `validation/` contains the 904 validation scenarios
- `cross_domain/` contains the 60 cross-domain test scenarios

## Conflict counts

- Type I: 6910 pairs
- Type II: 4050 pairs
- Type III: 4890 pairs
- Type IV: 2580 pairs


## Citation

If you use SCR-Merge or TrajMergeBench in your research, please cite:

```bibtex
@article{scrmerge2026,
  title     = {SCR-Merge: Verification-Gated Merging of Semantically
               Conflicting Agent Skill Documents across Multiple
               Optimization Trajectories},
  author    = {},
  year      = {2026},
  note      = {Under review}
}
```

## License

[Choose a license — e.g., MIT.]
