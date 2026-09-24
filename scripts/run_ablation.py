"""Real ablation runs for SCR-Merge without editing core pipeline files.

Each variant replaces one module method at runtime on a fresh pipeline
instance. Scores use the pipeline structural completeness metric because
TrajMergeBench does not yet ship per-trajectory validation task sets.
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.pipeline.orchestrator import MainPipeline
from src.pipeline.evaluator import ScoreComputer
from src.utils.config import Config
from src.core.types import (
    CompromiseVector, Conflict, Field, Module2Output, Module3Output, Module4Output, Module5Output,
)

SCENARIO = Path(r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\TrajMergeBench\held_out\base_0001\dataset_json")
OUT = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR\outputs\ablation_pilot.json")
SEED = 20260821
RULE_IDS = ["r1", "r2", "r3", "r4", "r5"]


def load_scenario():
    manifest = json.loads((SCENARIO / "manifest.json").read_text(encoding="utf-8"))
    s0 = json.loads((SCENARIO / "S0_base.json").read_text(encoding="utf-8"))
    trajectories = {}
    for k, meta in manifest["trajectories"].items():
        trajectories[int(k)] = json.loads((SCENARIO / meta["file"]).read_text(encoding="utf-8"))
    return manifest, s0, trajectories


def apply_variant(pipeline, variant):
    if variant == "no_gates":
        def pass_gates(self, m3_output, m2_output, base_rules, base_doc_score=None):
            return Module4Output(
                passed=True,
                gate_results={"gate1": True, "gate2": True, "gate3": True},
                degradation={},
                diagnostics="",
            )
        pipeline._m4.run = pass_gates.__get__(pipeline._m4)

    elif variant == "no_field":
        def identity_weights(self, rules):
            return np.eye(len(rules) * 5, dtype=np.float64)
        pipeline._m3.stage2_precompute_weights = identity_weights.__get__(pipeline._m3)

    elif variant == "random_priority":
        def random_priority_greedy(self, v_comp, vectors, weights, disputed, n_fields, D0,
                                   m1_output, edit_sets):
            rng = random.Random(SEED)
            v = v_comp.copy()
            D = list(disputed)
            rng.shuffle(D)
            for c in D:
                best_action = v[c]
                best_delta = 0.0
                for action in (1.0, 0.0):
                    if v[c] == action:
                        continue
                    vc = v.copy()
                    vc[c] = action
                    delta = self._compute_delta(v, vc, vectors, edit_sets, m1_output, len(edit_sets))
                    if delta > best_delta:
                        best_delta = delta
                        best_action = action
                v[c] = best_action
            return v, []
        pipeline._m3.stage3_greedy = random_priority_greedy.__get__(pipeline._m3)

    elif variant == "random_subset":
        def random_subset_m3(self, m1_output, m2_output, edit_sets):
            rng = random.Random(SEED)
            merged = copy.deepcopy(m1_output.base_rules)
            for es in edit_sets.values():
                for e in es.edits:
                    if rng.random() < 0.5:
                        rule = next(r for r in merged if r.rule_id == e.rule_id)
                        if e.field == Field.TOOLS:
                            vals = e.v_new if isinstance(e.v_new, (list, tuple, set)) else [e.v_new]
                            rule.tools = set(vals)
                        else:
                            setattr(rule, e.field.value, e.v_new)
            n_fields = len(merged) * 5
            cv = CompromiseVector(
                vector=np.zeros(n_fields, dtype=np.float64),
                n_rules=len(merged),
                n_fields=n_fields,
                disputed_fields=set(),
                per_trajectory_score={},
            )
            return Module3Output(compromise_vector=cv, merged_rules=merged, intermediate_states=[])
        pipeline._m3.run = random_subset_m3.__get__(pipeline._m3)

    elif variant == "text_diff":
        def diff_detector(self, m1_output, edit_sets):
            buckets = {}
            for es in edit_sets.values():
                for e in es.edits:
                    buckets.setdefault((e.rule_id, e.field), set()).add(
                        tuple(e.v_new) if isinstance(e.v_new, (list, tuple, set)) else str(e.v_new)
                    )
            conflicts = []
            for i, ((rid, f), vals) in enumerate(buckets.items()):
                if len(vals) > 1:
                    conflicts.append(Conflict(
                        conflict_id=f"C{i:04d}", type="I",
                        involved_rules=[rid], involved_fields=[f], score=0.5,
                    ))
            return Module2Output(conflicts=conflicts, influence_sets={}, conflict_matrix=None)
        pipeline._m2.run = diff_detector.__get__(pipeline._m2)

    elif variant == "full_recompute":
        def all_rules(self, resolved_rule_ids, graph):
            return set(RULE_IDS)
        pipeline._m5.affected_rules = all_rules.__get__(pipeline._m5)


def main():
    manifest, s0, trajectories = load_scenario()
    variants = ["full", "no_gates", "no_field", "random_priority",
                "random_subset", "text_diff", "full_recompute"]

    evaluator = ScoreComputer()
    pipeline = MainPipeline(config=Config(), mode="offline")
    loaded = {"done": False}
    inited = {"done": False}
    orig_load = MainPipeline._load_models
    orig_init = MainPipeline._init_modules

    def load_once(self):
        if not loaded["done"]:
            orig_load(self)
            loaded["done"] = True

    def init_once(self):
        if not inited["done"]:
            orig_init(self)
            inited["done"] = True

    MainPipeline._load_models = load_once
    MainPipeline._init_modules = init_once
    pipeline._load_models()
    pipeline._init_modules()

    originals = {
        "m4_run": pipeline._m4.run,
        "m3_weights": pipeline._m3.stage2_precompute_weights,
        "m3_greedy": pipeline._m3.stage3_greedy,
        "m3_run": pipeline._m3.run,
        "m2_run": pipeline._m2.run,
        "m5_affected": pipeline._m5.affected_rules,
    }

    rows = []
    for variant in variants:
        pipeline._m4.run = originals["m4_run"]
        pipeline._m3.stage2_precompute_weights = originals["m3_weights"]
        pipeline._m3.stage3_greedy = originals["m3_greedy"]
        pipeline._m3.run = originals["m3_run"]
        pipeline._m2.run = originals["m2_run"]
        pipeline._m5.affected_rules = originals["m5_affected"]
        apply_variant(pipeline, variant)
        t0 = time.time()
        result = pipeline.run(S0_dict=s0, S_k_dicts=trajectories, validation_set=None)
        elapsed = time.time() - t0
        m2 = result.module_outputs["M2"]
        m4 = result.module_outputs["M4"]
        m1 = result.module_outputs["M1"]
        m3 = result.module_outputs["M3"]
        types = {}
        for c in m2.conflicts:
            types[c.type] = types.get(c.type, 0) + 1
        base_scores = {k: evaluator.compute_score(m1.base_rules, k) for k in trajectories}
        final_scores = {k: evaluator.compute_score(m3.merged_rules, k) for k in trajectories}
        norm = [final_scores[k] / base_scores[k] for k in sorted(final_scores)]
        rows.append({
            "variant": variant,
            "scenario_id": manifest["scenario_id"],
            "merge_score": round(float(np.mean(norm)), 4),
            "rawlsian_gain": round(float(min(norm)) - 1.0, 4),
            "regression": round(sum(1 for k in final_scores if final_scores[k] < base_scores[k]) / len(final_scores), 4),
            "conflicts": types,
            "gates_passed": bool(m4.passed),
            "total_time": round(elapsed, 2),
            "final_scores": {str(k): round(v, 4) for k, v in sorted(final_scores.items())},
        })
        print(variant, rows[-1])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scenario": str(SCENARIO), "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    sys.exit(main())
