"""Reproduce and verify Table 6 of the SCR-Merge manuscript.

The Merge Score, Rawlsian Gain, Regression Rate, and LLM API call columns are
measurements produced by running the SCR-Merge pipeline and its baselines.
This script reproduces everything that can be derived from the released
TrajMergeBench inputs and cross-checks the manuscript table against its source
data table and the surrounding prose. It cannot re-measure scores without
running the full experiment pipeline.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

SKILLSCR_ROOT = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR")
DATA_ROOT = Path(r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\TrajMergeBench")
DOCX_PATH = Path(r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\codex\6.2.docx")
CSV_PATH = Path(r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_06\实验+outputs\data\MergeBench_v3_0_results.csv")

OUT_REPORT = SKILLSCR_ROOT / "outputs" / "table6_verification_report.md"
OUT_CSV = SKILLSCR_ROOT / "outputs" / "table6_verified.csv"


def norm_name(name):
    name = (name.replace("SCR Merge Ultima", "SCR-Merge")
               .replace("SCR Merge", "SCR-Merge")
               .strip())
    return re.sub(r"\s*\[\d+\]\s*$", "", name)


def csv_ci_to_interval(score, half_width):
    hw = float(half_width)
    return f"[{score - hw:.3f}, {score + hw:.3f}]"


def read_csv_rows():
    rows = {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("block") != "main":
                continue
            name = norm_name(rec["method"])
            rows[name] = {
                "merge_score": float(rec["merge_score"]),
                "ci95": rec["ci95"],
                "rawlsian_gain": float(rec["rawlsian_gain"]),
                "regression_rate": float(rec["regression_rate"]),
                "llm_api_calls": float(rec["llm_api_calls"]),
            }
    return rows


def read_docx_table():
    import docx
    d = docx.Document(str(DOCX_PATH))
    table = d.tables[0]
    rows = {}
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        if len(cells) < 6:
            continue
        name = norm_name(cells[0])
        rows[name] = {
            "merge_score": float(cells[1]),
            "ci95": cells[2],
            "rawlsian_gain": float(cells[3]),
            "regression_rate": float(cells[4]),
            "llm_api_calls": float(cells[5]),
        }
    return rows


def ci_center(ci):
    lo, hi = ci.strip("[]").split(",")
    return (float(lo) + float(hi)) / 2.0


def structural_stats():
    counts = Counter()
    conflicts = Counter()
    same_rf = 0
    for part in ("held_out", "development", "validation", "cross_domain"):
        for mp in sorted((DATA_ROOT / part).rglob("dataset_json/manifest.json")):
            m = json.loads(mp.read_text(encoding="utf-8"))
            counts[(m["subset"], m["split"])] += 1
            for cp in m.get("conflict_pairs", []):
                conflicts[cp["type"]] += 1
    test_base_high = counts[("base", "test")] + counts[("high_conflict", "test")]
    return {
        "test_base_high": test_base_high,
        "test_base": counts[("base", "test")],
        "test_high": counts[("high_conflict", "test")],
        "total": sum(counts.values()),
        "conflicts": dict(conflicts),
    }


def check_ci_symmetry(rows):
    issues = []
    for name, r in rows.items():
        center = ci_center(r["ci95"])
        if abs(center - r["merge_score"]) > 1e-9:
            issues.append(f"{name}: CI center {center} != Merge Score {r['merge_score']}")
    return issues


TEXT_ASSERTIONS = {
    "SCR-Merge offline": {"merge_score": 1.083, "rawlsian_gain": 0.061, "regression_rate": 0.0},
    "LLM Debate": {"merge_score": 1.048, "rawlsian_gain": 0.038},
    "Multi Objective SkillOpt": {"rawlsian_gain": 0.032},
    "SCR-Merge online": {"merge_score": 1.081},
    "Random Subset": {"merge_score": 1.019},
}

API_ASSERTIONS = {
    "LLM Debate": 8.4,
    "AutoPrompt Merge": 26.7,
    "SkillOpt RoundRobin": 142.6,
    "Multi Objective SkillOpt": 131.2,
}


def main():
    csv_rows = read_csv_rows()
    docx_rows = read_docx_table()
    stats = structural_stats()

    report = []
    report.append("# Table 6 Verification Report")
    report.append("")
    report.append("Source data table: `MergeBench_v3_0_results.csv`")
    report.append("Manuscript table: `6.2.docx` Table 6")
    report.append("Benchmark inputs: `TrajMergeBench`")
    report.append("")
    report.append("## Benchmark structure (derived from TrajMergeBench)")
    report.append(f"- Total scenarios: {stats['total']}")
    report.append(f"- Held-out test base + high conflict: {stats['test_base_high']} "
                  f"({stats['test_base']} base + {stats['test_high']} high conflict)")
    report.append(f"- Injected conflict pairs: {json.dumps(stats['conflicts'])}")
    report.append("")

    issues = check_ci_symmetry(docx_rows)
    report.append("## CI symmetry check")
    report.append(f"- Issues: {len(issues)}")
    for issue in issues:
        report.append(f"  - {issue}")
    report.append("")

    mismatches = []
    ci_mismatches = []
    report.append("## CSV vs 6.2.docx Table 6")
    report.append("| Method | Merge Score | CI | Rawlsian Gain | Regression Rate | LLM API Calls | Match |")
    report.append("|---|---|---|---|---|---|---|")
    for name in docx_rows:
        d = docx_rows[name]
        c = csv_rows.get(name)
        if c is None:
            mismatches.append(f"{name}: missing in CSV")
            report.append(f"| {name} | {d['merge_score']} | {d['ci95']} | {d['rawlsian_gain']} "
                          f"| {d['regression_rate']} | {d['llm_api_calls']} | CSV missing |")
            continue
        csv_ci = csv_ci_to_interval(c["merge_score"], c["ci95"])
        ci_ok = d["ci95"] == csv_ci
        if not ci_ok:
            ci_mismatches.append(f"{name}: CSV half-width {c['ci95']} gives {csv_ci}, Table 6 has {d['ci95']}")
        match = (
            abs(d["merge_score"] - c["merge_score"]) < 1e-9
            and ci_ok
            and abs(d["rawlsian_gain"] - c["rawlsian_gain"]) < 1e-9
            and abs(d["regression_rate"] - c["regression_rate"]) < 1e-9
            and abs(d["llm_api_calls"] - c["llm_api_calls"]) < 1e-9
        )
        if not match:
            mismatches.append(f"{name}: CSV {c} != docx {d}")
        report.append(f"| {name} | {d['merge_score']} | {d['ci95']} | {d['rawlsian_gain']} "
                      f"| {d['regression_rate']} | {d['llm_api_calls']} | {'OK' if match else 'DIFF'} |")
    report.append("")
    report.append(f"## CSV ci95 mismatch")
    report.append(f"- Rows where the CSV ci95 column does not reproduce the Table 6 interval: {len(ci_mismatches)}")
    for issue in ci_mismatches:
        report.append(f"  - {issue}")
    report.append("")

    report.append("## Prose cross-checks")
    for name, expected in TEXT_ASSERTIONS.items():
        row = docx_rows.get(name)
        if row is None:
            report.append(f"- {name}: row missing")
            continue
        diffs = []
        for key, val in expected.items():
            if abs(row[key] - val) > 1e-9:
                diffs.append(f"{key} {row[key]} != {val}")
        report.append(f"- {name}: {'OK' if not diffs else 'DIFF ' + '; '.join(diffs)}")
    for name, val in API_ASSERTIONS.items():
        row = docx_rows.get(name)
        status = "OK" if row and abs(row["llm_api_calls"] - val) < 1e-9 else "DIFF"
        report.append(f"- {name} LLM API calls: {status}")
    report.append("")

    report.append("## Measurement status")
    report.append(
        "- The Merge Score, Rawlsian Gain, Regression Rate, and LLM API call columns "
        "are experimental measurements produced by the SCR-Merge pipeline and its "
        "baselines. The delivery notes in `SCR-Merge_06/实验+outputs/README_Experiments.md` "
        "state that the current values are internally consistent placeholders and are "
        "not measurements from an executed run of the pipeline."
    )
    report.append(
        "- This script reproduces the structural statistics and every cross-check that "
        "can be derived from released inputs. Re-measuring the score columns requires "
        "running the full pipeline, including LLM-driven baselines and Type IV agent "
        "execution, on the current 1,355-scenario held-out test set."
    )
    report.append("")

    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Merge Score", "95% CI", "Rawlsian Gain", "Regression Rate (%)", "LLM API Calls"])
        for name, r in docx_rows.items():
            writer.writerow([name, r["merge_score"], r["ci95"], r["rawlsian_gain"],
                             r["regression_rate"], r["llm_api_calls"]])

    print("benchmark structure:", stats)
    print("CI symmetry issues:", len(issues))
    print("CSV vs docx mismatches:", len(mismatches))
    for m in mismatches[:20]:
        print(" ", m)
    print("report:", OUT_REPORT)
    print("verified table:", OUT_CSV)


if __name__ == "__main__":
    sys.exit(main())
