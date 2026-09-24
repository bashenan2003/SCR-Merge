#!/usr/bin/env python3
"""Convert .md skill files to standard Agent Skills Spec format.

Adds complete YAML frontmatter (name, description, license, metadata)
and a "When to Apply" section without modifying existing rule content.
"""
import re
import sys
from pathlib import Path


# ── Per-file metadata ────────────────────────────────────────────────

FILE_META: dict[str, dict] = {
    # ── dataset1 ──
    "dataset1/dataset_md/S0.md": {
        "name": "cross-domain-agent-skill",
        "description": (
            "Cross-domain base skill covering code generation, error handling, "
            "output formatting, testing, and security. Use when building or "
            "debugging agent pipelines, data export tools, or report generators."
        ),
        "triggers": [
            "用户请求数据导出、格式转换或报告生成",
            "Agent encounters errors during pipeline execution",
            "User requests data validation or schema checking",
        ],
    },
    "dataset1/dataset_md/S1.md": {
        "name": "search-qa-agent",
        "description": (
            "Search engine question-answering optimization. Use when the agent "
            "needs to search document corpora, extract answers from passages, "
            "or cross-validate information across multiple sources."
        ),
        "triggers": [
            "用户提出基于搜索的问答请求",
            "Agent needs to find answers in a document corpus",
            "User asks for information retrieval or fact-checking",
        ],
    },
    "dataset1/dataset_md/S2.md": {
        "name": "spreadsheet-data-processing",
        "description": (
            "Spreadsheet manipulation with Python for data processing tasks. "
            "Use when working with Excel files, CSV data, or performing "
            "bulk data transformations and aggregations."
        ),
        "triggers": [
            "处理 Excel 或 CSV 文件",
            "用户要求数据转换、聚合或清洗",
            "Agent needs to manipulate spreadsheets programmatically",
        ],
    },
    "dataset1/dataset_md/S3.md": {
        "name": "office-document-qa",
        "description": (
            "Multi-format office document question answering. Use when extracting "
            "information from PDF, DOCX, PPTX, or spreadsheet files. Handles "
            "structured data extraction from tables and charts."
        ),
        "triggers": [
            "用户上传办公文档并要求提取信息",
            "Agent needs to search across multiple document formats",
            "User asks questions about document content",
        ],
    },
    "dataset1/dataset_md/S4.md": {
        "name": "embodied-task-execution",
        "description": (
            "Physical task planning and execution in simulated environments. "
            "Use when the agent operates in ALFWorld-style environments requiring "
            "sequential action planning, object manipulation, and state verification."
        ),
        "triggers": [
            "Agent needs to plan physical actions in a simulated world",
            "用户要求执行具身智能任务",
            "Sequential sub-goal planning with state verification",
        ],
    },
    "dataset1/dataset_md/S5.md": {
        "name": "document-visual-qa",
        "description": (
            "Visual document question answering from scanned images. Use when "
            "extracting text via OCR, reading form fields, or answering questions "
            "about document images with spatial layout awareness."
        ),
        "triggers": [
            "用户上传扫描文档或图片",
            "Agent needs to extract text from document images",
            "User asks about content visible in a document screenshot",
        ],
    },
    "dataset1/dataset_md/S6.md": {
        "name": "mathematical-reasoning",
        "description": (
            "Mathematical reasoning with theorem-level precision. Use when "
            "evaluating mathematical claims, comparing multiple-choice options, "
            "or verifying logical statements with quantifiers and domain constraints."
        ),
        "triggers": [
            "用户提出数学推理或定理验证请求",
            "Agent needs to evaluate multiple-choice math problems",
            "User asks for logical verification of mathematical claims",
        ],
    },

    # ── dataset2 ──
    "dataset2/dataset_md/S0.md": {
        "name": "cross-domain-engineering-skill",
        "description": (
            "Cross-domain base engineering skill for code generation, error handling, "
            "output generation, testing, and security. Use as the foundation skill "
            "for domain-specific optimization in API, testing, or frontend contexts."
        ),
        "triggers": [
            "User asks to generate, review, or refactor code",
            "Agent needs to handle errors in software development tasks",
            "User requests testing, security audit, or output generation",
        ],
    },
    "dataset2/dataset_md/S1.md": {
        "name": "claude-api-integration",
        "description": (
            "Build and debug Claude API and Anthropic SDK integrations. Use when "
            "creating API clients, implementing prompt caching, handling streaming "
            "responses, or debugging API errors with retry logic."
        ),
        "triggers": [
            "User asks to integrate Claude API or Anthropic SDK",
            "Agent needs to implement prompt caching or streaming",
            "User encounters Claude API rate limits or authentication errors",
        ],
    },
    "dataset2/dataset_md/S2.md": {
        "name": "webapp-browser-testing",
        "description": (
            "Web application testing with Playwright and browser automation. "
            "Use when writing end-to-end tests, debugging UI failures with "
            "screenshots, or verifying responsive behavior across viewports."
        ),
        "triggers": [
            "User asks to test a web application end-to-end",
            "Agent needs to debug browser-based UI failures",
            "User requests Playwright test scripts or visual regression tests",
        ],
    },
    "dataset2/dataset_md/S3.md": {
        "name": "frontend-ui-engineering",
        "description": (
            "Frontend UI design and implementation with accessibility and "
            "responsive design. Use when building UI components, ensuring "
            "WCAG compliance, or implementing responsive layouts."
        ),
        "triggers": [
            "User asks to build or redesign UI components",
            "Agent needs to ensure accessibility compliance",
            "User requests responsive layout or cross-browser compatibility",
        ],
    },

    # ── dataset3 ──
    "dataset3/dataset_md/S0.md": {
        "name": "kua-ling-yu-ji-chu-gui-ze",
        "description": (
            "跨领域基础技能文档，覆盖代码生成、错误处理、输出生成、测试验证、安全防护。"
            "作为领域特化优化的起点，适用于各类软件工程和数据处理任务。"
        ),
        "triggers": [
            "用户请求生成、审查或重构代码",
            "智能体在开发过程中遇到错误需要处理",
            "用户要求输出格式化结果或执行安全审查",
        ],
    },
    "dataset3/dataset_md/S1.md": {
        "name": "dai-ma-shen-cha",
        "description": (
            "代码质量审查技能，覆盖安全漏洞检测、性能分析和代码规范检查。"
            "当用户请求 Code Review、PR Review 或代码安全检查时使用。"
        ),
        "triggers": [
            "用户请求代码审查或 PR 审查",
            '用户问"这段代码安全吗"或"帮我找 Bug"',
            "用户提到代码质量、安全审计或风格检查",
        ],
    },
    "dataset3/dataset_md/S2.md": {
        "name": "you-jian-fa-song-tong-zhi",
        "description": (
            "邮件构建、模板渲染、批量发送与投递状态追踪技能。"
            "当用户请求发送邮件、构建邮件模板或追踪邮件投递状态时使用。"
        ),
        "triggers": [
            "用户请求发送邮件或构建邮件模板",
            "用户需要邮件批量发送或投递状态追踪",
            "用户提到邮件通知、告警或营销邮件",
        ],
    },
    "dataset3/dataset_md/S3.md": {
        "name": "pdf-wen-dang-chu-li",
        "description": (
            "PDF 文档处理技能，覆盖文本提取、表单填写、文档合并拆分和 OCR 识别。"
            "当用户需要处理 PDF 文件、提取 PDF 内容或填写 PDF 表单时使用。"
        ),
        "triggers": [
            "用户上传或提到 PDF 文件处理",
            "用户需要提取 PDF 中的文本、表格或表单数据",
            "用户要求合并、拆分或填写 PDF 文档",
        ],
    },
}


def build_frontmatter(name: str, description: str) -> str:
    """Build YAML frontmatter with name, description, license, metadata."""
    lines = [
        "---",
        f"name: {name}",
    ]
    # Multi-line description
    if len(description) > 80:
        lines.append("description: >")
        remaining = description.strip()
        while len(remaining) > 80:
            cut = remaining.rfind(' ', 0, 80)
            if cut < 20:
                cut = 80
            lines.append(f"  {remaining[:cut]}")
            remaining = remaining[cut:].strip()
        if remaining:
            lines.append(f"  {remaining}")
    else:
        lines.append(f"description: {description}")
    lines.append("license: MIT")
    lines.append("metadata:")
    lines.append("  version: \"1.0.0\"")
    lines.append("  type: skill")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_trigger_section(triggers: list) -> str:
    """Build 'When to Apply' section."""
    if not triggers:
        return ""
    lines = ["## When to Apply"]
    for t in triggers:
        lines.append(f"- {t}")
    lines.extend(["", ""])
    return "\n".join(lines)


def convert_file(filepath: Path, meta: dict) -> None:
    """Convert a single .md file to standard Agent Skills Spec format."""
    text = filepath.read_text(encoding="utf-8")

    # Strip any existing YAML frontmatter (to rebuild cleanly)
    if text.strip().startswith("---"):
        parts = text.split("---", 2)
        text = parts[2].strip() if len(parts) >= 3 else text

    name = meta["name"]
    description = meta["description"]
    triggers = meta.get("triggers", [])

    frontmatter = build_frontmatter(name, description)
    trigger_section = build_trigger_section(triggers)

    # Find the position after the first heading to insert trigger section
    heading_match = re.search(r'^#\s+.+$', text, re.MULTILINE)
    if heading_match:
        before = text[:heading_match.end()]
        after = text[heading_match.end():].strip()
        body = before + "\n\n" + trigger_section + after
    else:
        body = trigger_section + text

    new_text = frontmatter + body.strip() + "\n"
    filepath.write_text(new_text, encoding="utf-8")
    n_rules = len(re.findall(r'^##\s+(?:Rule|规则)\s+', new_text, re.MULTILINE))
    print(f"  OK  {filepath.name}  name={name} rules={n_rules} triggers={len(triggers)}")


def main():
    data_dir = Path("E:/资料备份/论文写作/第3篇/我的方案/代码/SkillSCR/data")

    for ds_name in ["dataset1", "dataset2", "dataset3"]:
        md_dir = data_dir / ds_name / "dataset_md"
        if not md_dir.exists():
            continue
        print(f"\n{'='*55}")
        print(f"  {ds_name}/dataset_md")
        print(f"{'='*55}")

        for md_file in sorted(md_dir.glob("S*.md")):
            key = f"{ds_name}/dataset_md/{md_file.name}"
            meta = FILE_META.get(key)
            if meta is None:
                print(f"  WARN  {md_file.name}  no metadata defined, skipping")
                continue
            convert_file(md_file, meta)

    print(f"\nDone. All .md files converted to standard Agent Skills Spec format.")


if __name__ == "__main__":
    main()
