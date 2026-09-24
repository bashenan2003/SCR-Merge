import json
from collections import Counter, defaultdict
from pathlib import Path

import docx

DOCX = r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\codex\2.docx"
DATA = Path(r"E:\资料备份\论文写作\第3篇\我的方案\Manscirpt\SCR-Merge_07\TrajMergeBench")
PILOT_BASE = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR\outputs\real_experiment_base3.json")
PILOT_HIGH = Path(r"E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR\outputs\real_experiment_high3.json")

N_PAIRS = {"I": 6910, "II": 4050, "III": 4890, "IV": 2580}

print("=== 1. Table 7 arithmetic checks ===")
d = docx.Document(DOCX)
t = d.tables[3]
rows = {}
for r in t.rows[1:]:
    c = [x.text.strip() for x in r.cells]
    rows[c[1]] = (float(c[2]), float(c[3]), float(c[4]))
for name, (p, rec, f1) in rows.items():
    calc = 2 * p * rec / (p + rec) if p + rec else 0.0
    print(f"{name:22s} F1 {f1:.3f} calc {calc:.3f} {'OK' if abs(calc-f1) < 0.001 else 'DIFF'}")

pool = rows["Pooled Types I to III"]
macro = rows["Macro average"]
w_p = sum(N_PAIRS[k] * rows[f"Type {k}"][0] for k in ("I", "II", "III")) / 15850
w_r = sum(N_PAIRS[k] * rows[f"Type {k}"][1] for k in ("I", "II", "III")) / 15850
w_f1 = 2 * w_p * w_r / (w_p + w_r)
print(f"pooled expected {w_p:.3f}/{w_r:.3f}/{w_f1:.3f} vs table {pool[0]:.3f}/{pool[1]:.3f}/{pool[2]:.3f}")
m_p = sum(rows[f"Type {k}"][0] for k in ("I", "II", "III", "IV")) / 4
m_r = sum(rows[f"Type {k}"][1] for k in ("I", "II", "III", "IV")) / 4
m_f1 = 2 * m_p * m_r / (m_p + m_r)
print(f"macro expected {m_p:.3f}/{m_r:.3f}/{m_f1:.3f} vs table {macro[0]:.3f}/{macro[1]:.3f}/{macro[2]:.3f}")

print("\n=== 2. Candidate edit pair count ===")
total = 0
for mp in (DATA / "held_out").rglob("dataset_json/manifest.json"):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m["subset"] not in ("base", "high_conflict") or m["split"] != "test":
        continue
    buckets = defaultdict(set)
    for k, meta in m["trajectories"].items():
        doc = json.loads((mp.parent / meta["file"]).read_text(encoding="utf-8"))
        for e in doc.get("edit_set", {}).get("edits", []):
            buckets[(e["rule_id"], e["field"])].add(int(k))
    total += sum(len(v) * (len(v) - 1) // 2 for v in buckets.values())
print("computed test base+high candidate pairs:", total)

print("\n=== 3. Pilot-based rough detection estimate ===")
gt = Counter()
for mp in (DATA / "held_out").glob("high_conflict_00*/dataset_json/manifest.json"):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m["scenario_id"] <= 3:
        for cp in m.get("conflict_pairs", []):
            gt[cp["type"]] += 1

det = Counter()
for p in (PILOT_BASE, PILOT_HIGH):
    data = json.loads(p.read_text(encoding="utf-8"))
    for row in data["scenarios"]:
        for k, v in row["conflicts"].items():
            det[k] += v

print("injected ground truth (3 high scenarios):", dict(gt))
print("detector output (6 pilot scenarios):", dict(det))
for typ in ("I", "II", "III", "IV"):
    g = gt.get(typ, 0)
    e = det.get(typ, 0)
    tp = min(g, e)
    p = tp / e if e else 0.0
    r = tp / g if g else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"Type {typ}: gt {g} detected {e} rough P {p:.3f} R {r:.3f} F1 {f1:.3f}")
