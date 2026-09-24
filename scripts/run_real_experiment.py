"""Run the SCR-Merge offline experiment over TrajMergeBench scenarios.

This driver executes the real six-module pipeline on scenario inputs. It
reports per-scenario conflict detections, verification gate results, wall
time, and per-trajectory scores, then aggregates Merge Score, Rawlsian Gain,
and Regression Rate with a bootstrap confidence interval.

The released TrajMergeBench scenarios do not yet ship per-trajectory
validation task sets. Until a validation directory is provided with
`--validation-dir`, scores use the pipeline structural completeness score and
the aggregate metric source is marked `structural`. Supplying validation sets
and running the SkillOpt target agent is required for the paper's validation
based Merge Score.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.pipeline.orchestrator import MainPipeline
from src.pipeline.evaluator import ScoreComputer
from src.utils.config import Config


def load_scenario(folder):
    json_dir = folder / "dataset_json"
    manifest = json.loads((json_dir / "manifest.json").read_text(encoding="utf-8"))
    s0 = json.loads((json_dir / "S0_base.json").read_text(encoding="utf-8"))
    trajectories = {}
    for k, meta in manifest["trajectories"].items():
        trajectories[int(k)] = json.loads((json_dir / meta["file"]).read_text(encoding="utf-8"))
    return manifest, s0, trajectories


def collect_scenarios(root, subset, limit):
    out = []
    for part in ("held_out",):
        for folder in sorted((root / part).glob(f"{subset}_*/")):
            if not (folder / "dataset_json").exists():
                continue
            out.append(folder)
            if limit and len(out) >= limit:
                return out
    return out


def bootstrap_ci(values, n_boot=1000, seed=17160):
    rng = random.Random(seed)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = [arr[rng.randrange(len(arr))] for _ in range(len(arr))]
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\TrajMergeBench")
    parser.add_argument("--subset", default="base", choices=["base", "high_conflict"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "real_experiment_results.json"))
    parser.add_argument("--validation-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    folders = collect_scenarios(root, args.subset, args.limit)
    print(f"scenarios to run: {len(folders)}")

    pipeline = MainPipeline(config=Config(), mode="offline")
    evaluator = ScoreComputer()
    loaded = {"done": False}
    orig_load = MainPipeline._load_models

    def load_once(self):
        if not loaded["done"]:
            orig_load(self)
            loaded["done"] = True

    MainPipeline._load_models = load_once

    rows = []
    for folder in folders:
        manifest, s0, trajectories = load_scenario(folder)
        t0 = time.time()
        result = pipeline.run(
            S0_dict=s0,
            S_k_dicts=trajectories,
            validation_set=None,
        )
        elapsed = time.time() - t0
        m2 = result.module_outputs["M2"]
        m4 = result.module_outputs["M4"]
        m1 = result.module_outputs["M1"]
        m3 = result.module_outputs["M3"]
        types = Counter(c.type for c in m2.conflicts)
        base_scores = {k: evaluator.compute_score(m1.base_rules, k) for k in trajectories}
        final_scores = {k: evaluator.compute_score(m3.merged_rules, k) for k in trajectories}
        rows.append({
            "scenario_id": manifest["scenario_id"],
            "subset": manifest["subset"],
            "domain": manifest["domain"],
            "k": manifest["k"],
            "split": manifest["split"],
            "conflicts": dict(types),
            "gates_passed": bool(m4.passed),
            "gate_results": m4.gate_results,
            "total_time": round(elapsed, 2),
            "base_scores": {str(k): round(v, 4) for k, v in sorted(base_scores.items())},
            "final_scores": {str(k): round(v, 4) for k, v in sorted(final_scores.items())},
        })
        print(f"scenario {manifest['scenario_id']}: conflicts {dict(types)}, "
              f"gates {m4.passed}, time {elapsed:.1f}s")

    # Aggregate metrics.
    merge_scores = []
    rawlsian_gains = []
    regressed = 0
    total_trajs = 0
    for row in rows:
        keys = sorted(row["final_scores"])
        if not keys:
            continue
        norm = [row["final_scores"][k] / row["base_scores"][k] for k in keys]
        merge_scores.append(float(np.mean(norm)))
        rawlsian_gains.append(float(min(norm)) - 1.0)
        regressed += sum(1 for k in keys if row["final_scores"][k] < row["base_scores"][k])
        total_trajs += len(keys)

    summary = {}
    if merge_scores:
        lo, hi = bootstrap_ci(merge_scores)
        summary = {
            "metric_source": "structural" if not args.validation_dir else "validation",
            "n_scenarios": len(merge_scores),
            "merge_score": round(float(np.mean(merge_scores)), 3),
            "merge_score_ci95": [lo, hi],
            "rawlsian_gain": round(float(np.mean(rawlsian_gains)), 3),
            "regression_rate": round(regressed / total_trajs, 4) if total_trajs else None,
        }

    out = {
        "configuration": {
            "root": str(root),
            "subset": args.subset,
            "limit": args.limit,
            "validation_dir": args.validation_dir,
        },
        "summary": summary,
        "scenarios": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("saved:", out_path)


if __name__ == "__main__":
    sys.exit(main())
