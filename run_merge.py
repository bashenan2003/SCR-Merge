#!/usr/bin/env python3
"""SCR-Merge Ultima - Multi-Trajectory Skill Document Merger CLI.

正确运行的方法与指令

cd "E:\资料备份\论文写作\第3篇\我的方案\代码\SkillSCR"
方式 1：扫描目录（自动发现 S0 + S1..SN）

# 英文 6 域 (SearchQA, Spreadsheet, OfficeQA, ALFWorld, DocVQA, LiveMath)
python run_merge.py -d data/dataset1/dataset_md

# 英文 3 域 (Claude API, Web Testing, Frontend Design)
python run_merge.py -d data/dataset2/dataset_md

# 中文 3 域 (代码审查, 邮件发送, PDF处理)
python run_merge.py -d data/dataset3/dataset_md

# 英文 JSON 格式
python run_merge.py -d data/dataset1/dataset_json
方式 2：分别指定文件

# 两条轨迹
python run_merge.py --s0 data/dataset2/dataset_md/S0.md \
                    --s1 data/dataset2/dataset_md/S1.md \
                    --s2 data/dataset2/dataset_md/S2.md

# 混合中英文 + 混合 .md/.json
python run_merge.py --s0 data/dataset2/dataset_md/S0.md \
                    --s1 data/dataset1/dataset_json/S1_optimized.json \
                    --s2 data/dataset3/dataset_md/S1.md
方式 3：指定输出路径

python run_merge.py -d data/dataset2/dataset_md -o outputs/my_result.md
方式 4：在线模式（GPyTorch）

python run_merge.py -d data/dataset1/dataset_md --mode online
查看帮助

python run_merge.py --help
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.pipeline.orchestrator import MainPipeline


def scan_input_dir(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    s0_path = None
    sk_paths = {}
    s0_pat = re.compile(r"^S0(?:_base)?\.(json|md)$", re.IGNORECASE)
    sk_pat = re.compile(r"^S(\d+)(?:_optimized)?\.(json|md)$", re.IGNORECASE)
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if s0_pat.match(f.name):
            s0_path = f
        m = sk_pat.match(f.name)
        if m:
            k = int(m.group(1))
            if k >= 1:
                sk_paths[k] = f
    if s0_path is None:
        raise FileNotFoundError(f"No S0 file found in {directory}")
    if not sk_paths:
        raise FileNotFoundError(f"No trajectory files in {directory}")
    return s0_path, sk_paths


def main():
    parser = argparse.ArgumentParser(description="SCR-Merge Ultima")
    parser.add_argument("--input-dir", "-d", type=str, default=None)
    parser.add_argument("--s0", type=str, default=None)
    parser.add_argument("--s1", type=str, default=None)
    parser.add_argument("--s2", type=str, default=None)
    parser.add_argument("--s3", type=str, default=None)
    parser.add_argument("--s4", type=str, default=None)
    parser.add_argument("--s5", type=str, default=None)
    parser.add_argument("--s6", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--mode", "-m", choices=["offline", "online"], default="offline")
    args = parser.parse_args()

    if args.input_dir:
        s0_path, sk_paths = scan_input_dir(args.input_dir)
    else:
        if args.s0 is None:
            parser.error("Must specify --input-dir or --s0")
        s0_path = Path(args.s0)
        sk_paths = {}
        for k in range(1, 7):
            val = getattr(args, f"s{k}", None)
            if val:
                sk_paths[k] = Path(val)
        if not sk_paths:
            parser.error("At least one trajectory required")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "S_merged.md"

    print("=" * 60)
    print("  SCR-Merge Ultima")
    print("=" * 60)
    print(f"  S0:       {s0_path.name}")
    for k in sorted(sk_paths):
        print(f"  S{k}:       {sk_paths[k].name}")
    print(f"  Output:   {output_path}")
    print(f"  Trajs:    {len(sk_paths)}")
    print("=" * 60)
    print()

    config = Config()
    config.data["mode"] = args.mode
    pipeline = MainPipeline(config=config, mode=args.mode)
    result = pipeline.run_from_paths(
        s0_path=s0_path, sk_paths=sk_paths, output_path=output_path,
    )

    print()
    print("=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    m2 = result.module_outputs.get("M2")
    m4 = result.module_outputs.get("M4")
    if m2:
        tc = {}
        for c in m2.conflicts:
            tc[c.type] = tc.get(c.type, 0) + 1
        print(f"  Conflicts: {len(m2.conflicts)} "
              f"(I:{tc.get('I',0)} II:{tc.get('II',0)} "
              f"III:{tc.get('III',0)} IV:{tc.get('IV',0)})")
    if m4:
        print(f"  Gates:     {'PASSED' if m4.passed else 'FAILED'}")
        for gate, passed in m4.gate_results.items():
            print(f"    {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"  Time:      {result.metrics.get('total_time', 0):.1f}s")
    print(f"  Output:    {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
