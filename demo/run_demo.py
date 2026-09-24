#!/usr/bin/env python
"""SCR-Merge Ultima — End-to-end demo with sample skill documents.

Demonstrates the full 6-module pipeline:
  Module 1: Semantic parser + rule alignment (T5-small, Sentence-BERT, Hungarian)
  Module 2: Conflict detection Types I–IV (RoBERTa-large-mnli, cosine similarity)
  Module 3: BSS+ compromise search (greedy maximin with distance regularization)
  Module 4: Verification gating (performance, consistency, Pareto)
  Module 5: Incremental update engine
  Module 6: NL renderer (template-driven, fully offline)

Usage:
    python demo/run_demo.py                    # Run with sample data
    python demo/run_demo.py --mode online       # Run with GPyTorch surrogate
    python demo/run_demo.py --output result.txt  # Save output to file
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.utils.config import Config
from src.utils.logging import PipelineLogger
from src.core.types import EditSet
from src.pipeline.orchestrator import MainPipeline


def load_demo_data() -> tuple:
    """Load sample skill documents from demo/sample_docs/."""
    demo_dir = Path(__file__).parent

    s0_path = demo_dir / "sample_docs" / "S0_base.json"
    s1_path = demo_dir / "sample_docs" / "S1_optimized.json"
    s2_path = demo_dir / "sample_docs" / "S2_optimized.json"
    val_path = demo_dir / "validation_set" / "val_cases.jsonl"

    S0_dict = json.loads(s0_path.read_text(encoding="utf-8"))
    S1_dict = json.loads(s1_path.read_text(encoding="utf-8"))
    S2_dict = json.loads(s2_path.read_text(encoding="utf-8"))

    S_k_dicts = {1: S1_dict, 2: S2_dict}

    # Load pre-computed edit sets from the trajectory files
    edit_sets = {}
    for k, data in S_k_dicts.items():
        if "edit_set" in data:
            edit_sets[k] = EditSet.from_dict(data["edit_set"])

    # Load validation set
    validation_set = []
    if val_path.exists():
        for line in val_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                validation_set.append(json.loads(line))

    return S0_dict, S_k_dicts, edit_sets, validation_set


def main():
    parser = argparse.ArgumentParser(description="SCR-Merge Ultima Demo")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline",
                        help="Operating mode (default: offline)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save merged document to file")
    parser.add_argument("--json-output", action="store_true",
                        help="Also output as JSON format")
    args = parser.parse_args()

    # Configuration
    config = Config()
    config.data["mode"] = args.mode
    if args.mode == "online":
        config.data["surrogate"]["enabled"] = True

    logger = PipelineLogger(log_file=_project_root / "outputs" / "pipeline.log")

    print("=" * 70)
    print("  SCR-Merge Ultima — Multi-Trajectory Skill Document Merger")
    print(f"  Mode: {args.mode}")
    print("=" * 70)
    print()

    # Load data
    print("[1/4] Loading demo data...")
    S0_dict, S_k_dicts, edit_sets, validation_set = load_demo_data()
    print(f"  Base document: {S0_dict['doc_id']} ({len(S0_dict['rules'])} rules)")
    for k, doc in S_k_dicts.items():
        desc = doc.get("description", f"Trajectory {k}")
        n_edits = len(edit_sets.get(k, EditSet(k)).edits) if edit_sets else 0
        print(f"  Trajectory {k}: {doc['doc_id']} — {desc} ({n_edits} edits)")
    print(f"  Validation cases: {len(validation_set)}")
    print()

    # Run pipeline
    print("[2/4] Running SCR-Merge pipeline (6 modules)...")
    print()
    t_start = time.time()

    pipeline = MainPipeline(config=config, mode=args.mode)
    result = pipeline.run(
        S0_dict=S0_dict,
        S_k_dicts=S_k_dicts,
        edit_sets=edit_sets,
        validation_set=validation_set,
    )

    t_total = time.time() - t_start
    print()
    print(f"[3/4] Pipeline completed in {t_total:.2f}s")
    print()

    # Report results
    print("[4/4] Results Summary")
    print("-" * 70)

    m1 = result.module_outputs["M1"]
    m2 = result.module_outputs["M2"]
    m3 = result.module_outputs["M3"]
    m4 = result.module_outputs["M4"]

    print(f"\nModule 1 (Parser):")
    print(f"  Rules parsed: {len(m1.base_rules)}")
    print(f"  Alignments: {len(m1.alignment)}")
    print(f"  Graph: {m1.graph}")

    print(f"\nModule 2 (Conflict Detection):")
    print(f"  Conflicts detected: {len(m2.conflicts)}")
    type_counts = {}
    for c in m2.conflicts:
        type_counts[c.type] = type_counts.get(c.type, 0) + 1
    for t in ["I", "II", "III", "IV"]:
        print(f"    Type {t}: {type_counts.get(t, 0)}")
    for c in m2.conflicts[:5]:
        print(f"    {c}")

    print(f"\nModule 3 (BSS+ Compromise):")
    print(f"  Disputed fields: {len(m3.compromise_vector.disputed_fields)}")
    print(f"  Greedy iterations: {len(m3.intermediate_states)}")
    for state in m3.intermediate_states[:3]:
        print(f"    Iter {state['iteration']}: field={state['field']}, "
              f"action={state['action']}, score={state['score']:.3f}")

    print(f"\nModule 4 (Verification):")
    print(f"  All gates passed: {m4.passed}")
    for gate, passed in m4.gate_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"    {gate}: {status}")
    if m4.diagnostics:
        print(f"  Diagnostics: {m4.diagnostics}")

    print(f"\nModule 5 (Incremental Update):")
    m5 = result.module_outputs["M5"]
    print(f"  Remaining conflicts: {len(m5.updated_conflicts)}")

    print(f"\nModule 6 (Renderer):")
    m6 = result.module_outputs["M6"]
    print(f"  Template: {'used' if m6.template_used else 'N/A'}")

    print(f"\nMetrics:")
    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}s" if "time" in k else f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print()
    print("-" * 70)
    print("MERGED SKILL DOCUMENT")
    print("-" * 70)
    print(result.final_document)
    print("-" * 70)

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result.final_document, encoding="utf-8")
        print(f"\nMerged document saved to: {out_path}")

    if args.json_output:
        json_path = _project_root / "outputs" / "merged_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        from src.core.skill_document import SkillDocument
        merged_doc = SkillDocument.from_dict(
            {"doc_id": "S_merged", "rules": [r.to_dict() for r in m3.merged_rules]}
        )
        merged_doc.to_json(json_path)
        print(f"JSON output saved to: {json_path}")

    # Always save basic output
    default_out = _project_root / "outputs" / "merged_output.txt"
    default_out.parent.mkdir(parents=True, exist_ok=True)
    default_out.write_text(result.final_document, encoding="utf-8")
    print(f"\nOutput saved to: {default_out}")

    # Verification assertions
    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if len(m1.base_rules) > 0:
        print(f"  [PASS] Rules parsed: {len(m1.base_rules)}")
        checks_passed += 1
    else:
        print(f"  [FAIL] No rules parsed")

    checks_total += 1
    if len(m2.conflicts) > 0:
        print(f"  [PASS] Conflicts detected: {len(m2.conflicts)}")
        checks_passed += 1
    else:
        print(f"  [FAIL] No conflicts detected (expected at least 1)")

    checks_total += 1
    if len(m3.merged_rules) > 0:
        print(f"  [PASS] Merged rules generated: {len(m3.merged_rules)}")
        checks_passed += 1
    else:
        print(f"  [FAIL] No merged rules generated")

    checks_total += 1
    if m4.passed:
        print(f"  [PASS] Verification gates passed")
        checks_passed += 1
    else:
        print(f"  [WARN] Verification gates not all passed: {m4.diagnostics}")

    checks_total += 1
    if result.final_document and len(result.final_document) > 100:
        print(f"  [PASS] Final document rendered ({len(result.final_document)} chars)")
        checks_passed += 1
    else:
        print(f"  [FAIL] Final document missing or too short")

    print(f"\n  Summary: {checks_passed}/{checks_total} checks passed")
    if checks_passed == checks_total:
        print("  All checks passed - SCR-Merge pipeline working correctly!")
    else:
        print(f"  {checks_total - checks_passed} check(s) failed - review output above")

    return 0 if checks_passed == checks_total else 1


if __name__ == "__main__":
    sys.exit(main())
