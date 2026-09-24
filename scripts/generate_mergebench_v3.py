#!/usr/bin/env python3
"""Generate MergeBench v3.0 scenario folders for SCR-Merge Ultima.

Layout (one scenario per directory):
    <root>/held_out/base_%04d/dataset_json/{manifest.json,S0_base.json,S1_optimized.json,...}
    <root>/held_out/base_%04d/dataset_md/{S0.md,S1.md,...}
    <root>/held_out/high_conflict_%04d/...
    <root>/cross_domain/customer_service_%04d/...
    <root>/cross_domain/code_generation_%04d/...

Counts: 135 base + 36 high-conflict held-out scenarios, plus 30 customer
service and 30 code-generation cross-domain scenarios.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
from pathlib import Path


SEED = 17160
DEFAULT_SEED = SEED
FIELD_ORDER = ["trigger", "precondition", "action", "postcondition", "tools"]
RULE_IDS = ["r1", "r2", "r3", "r4", "r5"]


def TH(trigger=None, precondition=None, action=None, postcondition=None, tools=None):
    """Build a compact theme override dict."""
    out = {}
    if trigger is not None:
        out["trigger"] = trigger
    if precondition is not None:
        out["precondition"] = precondition
    if action is not None:
        out["action"] = action
    if postcondition is not None:
        out["postcondition"] = postcondition
    if tools is not None:
        out["tools"] = tools
    return out


BASE_SKELETONS = {
    "retrieval": {
        "r1": {
            "trigger": "the agent receives a question that needs evidence",
            "precondition": "the answer is not yet known and multiple sources may contain it",
            "action": "locate the most relevant source, extract candidate evidence, and verify it against the question",
            "postcondition": "verified evidence that directly answers the question",
            "tools": ["locate_source", "extract_candidate", "verify_evidence"],
        },
        "r2": {
            "trigger": "the agent has extracted candidate evidence",
            "precondition": "the evidence is partial or inconsistent across sources",
            "action": "merge related evidence, resolve inconsistencies, and record confidence for every claim",
            "postcondition": "a reconciled evidence set with confidence labels",
            "tools": ["merge_evidence", "resolve_inconsistency", "label_confidence"],
        },
        "r3": {
            "trigger": "the agent receives a multi-step task",
            "precondition": "the task can be split into ordered sub-goals",
            "action": "decompose the task, execute each sub-goal in order, and verify intermediate results before continuing",
            "postcondition": "all sub-goals completed and verified in order",
            "tools": ["decompose_task", "verify_subgoal", "continue_next"],
        },
        "r4": {
            "trigger": "the agent encounters an execution error",
            "precondition": "the error includes diagnostic information",
            "action": "identify the root cause, apply the minimal correction, and retry only after confirming the fix",
            "postcondition": "the error resolved with no regressions",
            "tools": ["read_error", "analyze_root_cause", "apply_fix", "retry_checked"],
        },
        "r5": {
            "trigger": "the agent has completed all processing",
            "precondition": "the task expects a final answer",
            "action": "generate the answer in the required format and perform a final consistency check against the evidence",
            "postcondition": "a complete answer consistent with all verified evidence",
            "tools": ["generate_answer", "consistency_check", "format_result"],
        },
    },
    "tabular": {
        "r1": {
            "trigger": "the agent receives a tabular data task",
            "precondition": "the data is stored in a spreadsheet or table",
            "action": "load the table, identify the requested columns and rows, and extract the relevant values",
            "postcondition": "the relevant values extracted with their cell locations",
            "tools": ["load_table", "locate_columns", "extract_values"],
        },
        "r2": {
            "trigger": "the agent has extracted values from the table",
            "precondition": "the values need cleaning or type conversion",
            "action": "normalize the values, record the original cell, and apply consistent types",
            "postcondition": "a normalized value set traceable to source cells",
            "tools": ["normalize_value", "record_cell", "apply_type"],
        },
        "r3": {
            "trigger": "the agent receives a multi-step data task",
            "precondition": "the task can be split into ordered sub-goals",
            "action": "decompose the data task, execute each sub-goal in order, and verify intermediate tables before continuing",
            "postcondition": "all data sub-goals completed and verified",
            "tools": ["decompose_task", "verify_table", "continue_next"],
        },
        "r4": {
            "trigger": "the agent encounters a data processing error",
            "precondition": "the error includes diagnostic information",
            "action": "identify the failing row or cell, apply the minimal correction, and retry only after confirming the fix",
            "postcondition": "the data error resolved with no regressions",
            "tools": ["read_error", "locate_cell", "apply_fix", "retry_checked"],
        },
        "r5": {
            "trigger": "the agent has completed the data task",
            "precondition": "the task expects a deliverable table or report",
            "action": "generate the output table or report and verify row and column totals",
            "postcondition": "a complete output with verified totals",
            "tools": ["generate_output", "verify_totals", "format_table"],
        },
    },
    "embodied": {
        "r1": {
            "trigger": "the agent must complete a physical task in a simulated environment",
            "precondition": "the environment contains the objects and receptacles needed",
            "action": "inspect available objects, plan an ordered sequence of actions, and execute each action with state checks",
            "postcondition": "the task completed with verified environment state",
            "tools": ["inspect_objects", "plan_sequence", "execute_action", "check_state"],
        },
        "r2": {
            "trigger": "the agent has performed an action",
            "precondition": "the environment state may differ from the expected state",
            "action": "compare the observed state with the expected state and update the plan",
            "postcondition": "an updated plan grounded in the observed state",
            "tools": ["observe_state", "compare_state", "update_plan"],
        },
        "r3": {
            "trigger": "the agent receives a multi-step physical task",
            "precondition": "the task can be split into ordered sub-goals",
            "action": "decompose the task into sub-goals, execute each sub-goal, and verify the state after each one",
            "postcondition": "all sub-goals completed with verified states",
            "tools": ["decompose_task", "execute_subgoal", "verify_state"],
        },
        "r4": {
            "trigger": "the agent encounters an execution error",
            "precondition": "the error includes diagnostic information",
            "action": "identify the root cause, apply the minimal correction, and retry only after confirming the fix",
            "postcondition": "the error resolved with no regressions",
            "tools": ["read_error", "analyze_root_cause", "apply_fix", "retry_checked"],
        },
        "r5": {
            "trigger": "the agent has completed all sub-goals",
            "precondition": "the task requires a final status",
            "action": "report the completed status, the executed actions, and the final environment state",
            "postcondition": "a final status report with executed actions and state",
            "tools": ["report_status", "list_actions", "final_state"],
        },
    },
    "conversational": {
        "r1": {
            "trigger": "the agent receives a customer request",
            "precondition": "the request concerns an account, order, or service issue",
            "action": "identify the customer need, retrieve the relevant account or order context, and determine the applicable policy",
            "postcondition": "the customer need identified with the applicable policy",
            "tools": ["identify_need", "retrieve_context", "lookup_policy"],
        },
        "r2": {
            "trigger": "the agent has identified the customer need",
            "precondition": "the resolution depends on missing details",
            "action": "ask for the missing details and confirm the customer context before acting",
            "postcondition": "a confirmed customer context ready for resolution",
            "tools": ["ask_details", "confirm_context", "prepare_resolution"],
        },
        "r3": {
            "trigger": "the agent receives a multi-step customer request",
            "precondition": "the request can be split into ordered sub-goals",
            "action": "decompose the request, complete each sub-goal in order, and verify the customer outcome before continuing",
            "postcondition": "all customer sub-goals completed and verified",
            "tools": ["decompose_request", "verify_outcome", "continue_next"],
        },
        "r4": {
            "trigger": "the agent encounters a service error",
            "precondition": "the error includes diagnostic information",
            "action": "identify the root cause, apply the minimal correction, and retry only after confirming the fix",
            "postcondition": "the service error resolved with no regressions",
            "tools": ["read_error", "analyze_root_cause", "apply_fix", "retry_checked"],
        },
        "r5": {
            "trigger": "the agent has resolved the customer request",
            "precondition": "the customer expects a clear outcome",
            "action": "state the resolution, the next step, and the reference number in plain language",
            "postcondition": "a clear resolution message with next steps",
            "tools": ["state_resolution", "next_steps", "reference_number"],
        },
    },
    "engineering": {
        "r1": {
            "trigger": "the agent receives a code generation task",
            "precondition": "the task requires creating or modifying source files",
            "action": "understand the existing codebase, design the minimal implementation, and write code that follows project conventions",
            "postcondition": "working code that integrates with the existing codebase",
            "tools": ["read_codebase", "search_project", "write_code"],
        },
        "r2": {
            "trigger": "the agent has written or modified code",
            "precondition": "the code may introduce behavior changes",
            "action": "run the relevant tests, review the diff, and fix regressions before delivering",
            "postcondition": "tested code with a reviewed diff",
            "tools": ["run_tests", "review_diff", "fix_regression"],
        },
        "r3": {
            "trigger": "the agent receives a multi-step code task",
            "precondition": "the task can be split into ordered sub-goals",
            "action": "decompose the code task, implement each sub-goal in order, and verify the build after each one",
            "postcondition": "all code sub-goals completed and verified",
            "tools": ["decompose_task", "verify_build", "continue_next"],
        },
        "r4": {
            "trigger": "the agent encounters a code error",
            "precondition": "the error includes diagnostic information",
            "action": "identify the root cause, apply the minimal correction, and retry only after confirming the fix",
            "postcondition": "the code error resolved with no regressions",
            "tools": ["read_error", "analyze_root_cause", "apply_fix", "retry_checked"],
        },
        "r5": {
            "trigger": "the agent has completed the code change",
            "precondition": "the task expects a deliverable implementation",
            "action": "generate the final code, usage notes, and a summary of behavior changes",
            "postcondition": "a complete implementation with usage notes",
            "tools": ["generate_output", "usage_notes", "change_summary"],
        },
    },
}


def P(name, family, topic_trigger, topics, themes, pairs):
    return {
        "family": family,
        "topic_trigger": topic_trigger,
        "topics": topics,
        "themes": themes,
        "pairs": pairs,
    }


DOMAIN_PROFILES = {
    "SearchQA": P(
        "SearchQA",
        "retrieval",
        "the agent must answer a question about {topic}",
        [
            "financial filings", "medical guidelines", "product documentation",
            "legal judgments", "academic abstracts", "news archives",
            "government policies", "technical manuals", "travel regulations",
            "consumer reviews", "patent records", "encyclopedia entries",
            "engineering standards", "market reports", "safety bulletins",
        ],
        [
            {
                "r1": TH(action="issue targeted queries and rank passages by exact phrase overlap with the question", tools=["issue_query", "rank_phrase_overlap", "extract_span"]),
                "r2": TH(action="cross-check the span across independent passages and keep only consensus values"),
                "r4": TH(action="narrow the query, drop noisy results, and retry with quoted exact terms"),
                "r5": TH(postcondition="a concise answer with inline passage citations"),
            },
            {
                "r1": TH(action="retrieve multiple independent passages and group them by supporting claim", tools=["retrieve_many", "group_claims", "rank_support"]),
                "r2": TH(trigger="the agent has collected passages with different answers", precondition="the passages disagree on the value", action="count sources per answer, inspect source quality, and select the answer with the strongest independent support", postcondition="a consensus answer with source counts"),
                "r4": TH(action="add a source quality filter and rerun the aggregation"),
            },
            {
                "r1": TH(trigger="the agent receives a question containing entities or aliases", action="resolve entities and aliases before searching", tools=["extract_entities", "resolve_alias", "search_entity"]),
                "r2": TH(action="map every candidate value to its canonical entity and reject ambiguous matches"),
                "r4": TH(action="expand the alias table and retry the entity resolution"),
            },
            {
                "r1": TH(action="decompose the question into sub-queries and merge results by evidence chain", tools=["decompose_query", "run_subquery", "merge_chain"]),
                "r3": TH(action="execute each sub-query, record the evidence chain, and verify the chain before composing the answer"),
                "r5": TH(action="compose the answer only from verified evidence chains"),
            },
            {
                "r1": TH(action="score passages by recency, source authority, and exact span match", tools=["score_passage", "authority_rank", "span_match"]),
                "r2": TH(action="attach citations to every claim and keep claims with authoritative sources"),
                "r5": TH(postcondition="an answer with ranked citations per claim"),
            },
            {
                "r1": TH(action="issue broad queries, collect all candidate spans, and filter by semantic relevance afterward", tools=["broad_query", "collect_spans", "semantic_filter"]),
                "r2": TH(action="keep borderline candidates until verification and rank by precision and recall balance afterward"),
                "r4": TH(action="on a low-result first pass, expand synonyms and related terms"),
            },
        ],
        [
            ("r1", "action", "rank passages by exact phrase overlap only", "rank passages by semantic similarity only"),
            ("r1", "tools", ["search_phrase"], ["search_semantic"]),
            ("r2", "action", "keep the answer with the highest source count", "keep the answer from the newest source"),
            ("r4", "action", "retry immediately with a broader query", "stop and ask the user for clarification"),
            ("r2", "action", "trust only passages from official sources", "trust only passages from peer reviewed sources"),
        ],
    ),
    "SpreadsheetBench": P(
        "SpreadsheetBench",
        "tabular",
        "the agent must process a {topic} spreadsheet",
        [
            "sales ledger cleaning", "employee timesheets", "inventory reconciliation",
            "budget variance reports", "customer feedback scoring", "payment aging tables",
            "marketing campaign metrics", "expense categorization", "supply chain forecasts",
            "survey response coding", "invoice register audits", "attendance consolidation",
            "commission calculations", "stock movement logs", "project cost tracking",
        ],
        [
            {
                "r1": TH(action="parse every column with its declared type and report coercion failures", tools=["read_sheet", "infer_column_type", "coerce_value"]),
                "r2": TH(action="fix coercion failures with the nearest valid typed value and log every correction"),
                "r4": TH(action="fall back to the raw text value on coercion failure and flag the cell"),
                "r5": TH(postcondition="a normalized table with typed columns and a correction log"),
            },
            {
                "r1": TH(action="copy the source table to a staging sheet before any transformation", tools=["copy_staging", "transform_rows", "keep_source"]),
                "r2": TH(action="apply transformations in ordered stages and validate the row count after each stage"),
                "r4": TH(action="restore the staging copy and reapply the failed stage with guards"),
            },
            {
                "r1": TH(action="identify the grouping keys and aggregate measures before touching the table", tools=["group_rows", "aggregate_measure", "verify_total"]),
                "r2": TH(action="verify each aggregate against a hand-checked sample before accepting it"),
                "r5": TH(postcondition="a verified aggregate table with sample checks documented"),
            },
            {
                "r1": TH(action="trace formulas cell by cell and record dependencies", tools=["trace_formula", "record_dependency", "inspect_cell"]),
                "r2": TH(action="recompute changed formulas from their source cells and compare results"),
                "r4": TH(action="mark circular references and block them from the computation"),
            },
            {
                "r1": TH(action="validate every row against the required schema before processing", tools=["row_schema", "validate_row", "collect_errors"]),
                "r2": TH(action="group validation errors by pattern and repair the most frequent pattern first"),
                "r5": TH(postcondition="a validated table with a pattern-level error report"),
            },
            {
                "r1": TH(action="apply cleaning rules in a fixed order so reruns produce identical output", tools=["fixed_rule_order", "clean_cell", "hash_output"]),
                "r2": TH(action="skip already cleaned cells using stable cell fingerprints"),
                "r4": TH(action="detect accidental double cleaning and restore the canonical value"),
            },
        ],
        [
            ("r1", "action", "parse every column with strict types", "parse every column as text first"),
            ("r1", "tools", ["strict_type_parser"], ["text_first_parser"]),
            ("r2", "action", "transform data in-place on the original sheet", "copy data to a staging sheet before transformation"),
            ("r4", "action", "revert to the last saved version", "apply a targeted patch and revalidate"),
            ("r2", "action", "keep the first non-empty value", "keep the most recently updated value"),
        ],
    ),
    "OfficeQA": P(
        "OfficeQA",
        "retrieval",
        "the agent must answer a question from an office document about {topic}",
        [
            "board meeting minutes", "annual reports", "employee handbooks",
            "procurement contracts", "project charters", "audit findings",
            "tender documents", "training manuals", "quarterly reviews",
            "compliance notices", "budget memos", "policy updates",
            "vendor agreements", "HR procedures", "risk assessments",
        ],
        [
            {
                "r1": TH(action="detect the document format and route the query to the format-specific extractor", tools=["detect_format", "route_query", "format_extractor"]),
                "r2": TH(action="merge format-specific answers and resolve format conflicts"),
                "r4": TH(action="on an empty extraction result, re-detect the format"),
            },
            {
                "r1": TH(action="locate tables near the query context and extract their headers", tools=["locate_table", "extract_headers", "bind_query"]),
                "r2": TH(action="bind every answer value to a table cell and keep the cell address"),
                "r5": TH(postcondition="an answer with table cell references"),
            },
            {
                "r1": TH(action="search related files and link passages that mention the same entities", tools=["search_files", "link_entities", "merge_passages"]),
                "r2": TH(action="resolve cross-file references and prefer the most recent authoritative file"),
                "r4": TH(action="on missing linked documents, refresh file metadata"),
            },
            {
                "r1": TH(action="identify the requested section and search only that section first", tools=["section_index", "target_section", "search_within"]),
                "r2": TH(action="on an incomplete target section, expand to adjacent sections"),
                "r5": TH(postcondition="an answer with section and paragraph anchors"),
            },
            {
                "r1": TH(action="collect candidate spans and score them by citation count and recency", tools=["span_candidates", "citation_score", "recency_rank"]),
                "r2": TH(action="keep only claims with an explicit citation and record the citation span"),
                "r5": TH(postcondition="an answer with explicit citations for every claim"),
            },
            {
                "r1": TH(action="use the document layout to distinguish headers, footers, and body text", tools=["layout_model", "block_type", "extract_body"]),
                "r2": TH(action="reassemble split sentences across layout blocks before answering"),
                "r4": TH(action="on incomplete body extraction, retry with a lower confidence threshold"),
            },
        ],
        [
            ("r1", "action", "search only the document body text", "search headings and tables first"),
            ("r1", "tools", ["body_text_search"], ["heading_table_search"]),
            ("r2", "action", "prefer values in the newest revision", "prefer values in the longest section"),
            ("r4", "action", "return the closest partial match", "return no answer and request clarification"),
            ("r2", "action", "merge answers from all formats", "keep the answer from the first matching format"),
        ],
    ),
    "DocVQA": P(
        "DocVQA",
        "retrieval",
        "the agent must answer a visual document question about {topic}",
        [
            "scanned invoices", "signed contracts", "medical forms",
            "bank statements", "shipping labels", "tax returns",
            "handwritten notes", "registration cards", "certificate scans",
            "receipt images", "insurance claims", "property deeds",
            "school transcripts", "laboratory reports", "customs declarations",
        ],
        [
            {
                "r1": TH(action="run OCR and correct obvious character errors using the question vocabulary", tools=["run_ocr", "correct_ocr", "vocab_check"]),
                "r2": TH(action="rank OCR variants by vocabulary likelihood and keep the top variant"),
                "r4": TH(action="on low OCR confidence, re-run OCR at higher resolution"),
            },
            {
                "r1": TH(action="locate the question fields by their spatial anchors on the page", tools=["spatial_anchor", "field_region", "crop_region"]),
                "r2": TH(action="verify the extracted value appears inside the anchored region"),
                "r5": TH(postcondition="an answer with region coordinates"),
            },
            {
                "r1": TH(action="match the page against known form templates and extract mapped fields", tools=["template_match", "field_map", "extract_mapped"]),
                "r2": TH(action="in the absence of a template match, fall back to text search"),
                "r4": TH(action="add the unknown layout to the template pool after user confirmation"),
            },
            {
                "r1": TH(action="verify each answer by re-reading the corresponding image region", tools=["answer_region", "re_read_region", "verify_pixels"]),
                "r2": TH(action="on ambiguous text, combine OCR text with visual features"),
                "r5": TH(postcondition="an answer verified against the image region"),
            },
            {
                "r1": TH(action="aggregate fields across all pages before answering", tools=["page_aggregate", "field_consolidate", "page_order"]),
                "r2": TH(action="resolve conflicting values across pages using page order and document type"),
                "r4": TH(action="flag missing pages and answer only from complete fields"),
            },
            {
                "r1": TH(action="normalize layout variations so the same field is found on any template", tools=["layout_normalize", "field_find", "normalized_extract"]),
                "r2": TH(action="map normalized field values to canonical units and formats"),
                "r5": TH(postcondition="a normalized answer independent of layout variation"),
            },
        ],
        [
            ("r1", "action", "use OCR text only for extraction", "use image layout analysis only for extraction"),
            ("r1", "tools", ["ocr_text_extract"], ["layout_region_extract"]),
            ("r2", "action", "prefer values in the top-left region", "prefer values in the largest text block"),
            ("r4", "action", "re-run OCR at higher resolution", "send the crop to a vision model"),
            ("r2", "action", "prefer typed text over handwritten text", "prefer handwritten annotations over typed text"),
        ],
    ),
    "LiveMathematicianBench": P(
        "LiveMathematicianBench",
        "retrieval",
        "the agent must verify a mathematical claim about {topic}",
        [
            "calculus limits", "linear algebra proofs", "probability bounds",
            "number theory lemmas", "combinatorial identities", "optimization problems",
            "logic equivalences", "differential equations", "geometric inequalities",
            "series convergence", "graph theory properties", "group theory results",
            "functional analysis bounds", "set theory cardinalities", "game theory equilibria",
        ],
        [
            {
                "r1": TH(action="verify every transformation step with symbolic algebra", tools=["symbolic_verify", "step_check", "rewrite_rule"]),
                "r2": TH(action="record each verified step and reject steps without a matching rewrite rule"),
                "r4": TH(action="backtrack to the last verified step on a step failure"),
                "r5": TH(postcondition="a proof whose every step is symbolically verified"),
            },
            {
                "r1": TH(action="enumerate candidate proof strategies and test them in parallel", tools=["enumerate_proof", "strategy_test", "parallel_check"]),
                "r2": TH(action="keep the shortest verified proof among the candidates"),
                "r4": TH(action="on zero verified candidates, increase the enumeration budget"),
            },
            {
                "r1": TH(action="probe boundary cases and edge parameters before accepting a result", tools=["boundary_probe", "edge_case", "limit_check"]),
                "r2": TH(action="test zero, infinity, and undefined regions explicitly"),
                "r5": TH(postcondition="a result with boundary case notes"),
            },
            {
                "r1": TH(action="instantiate quantifiers with witness candidates from the goal", tools=["quantifier_witness", "instantiate", "goal_check"]),
                "r2": TH(action="choose witnesses that satisfy the strongest constraints first"),
                "r4": TH(action="on a witness constraint violation, reject the proof"),
            },
            {
                "r1": TH(action="expand definitions first and derive from first principles afterward", tools=["expand_definition", "derive_step", "principle_check"]),
                "r2": TH(action="keep derivations aligned with the expanded definitions"),
                "r5": TH(postcondition="a derivation traceable to definitions"),
            },
            {
                "r1": TH(action="search for counterexamples before committing to a proof", tools=["counterexample_search", "candidate_find", "falsify_check"]),
                "r2": TH(action="on finding a counterexample, report it and stop the proof attempt"),
                "r4": TH(action="on an overly broad search, restrict the search to the problem domain"),
            },
        ],
        [
            ("r1", "action", "verify every step with symbolic algebra", "verify only the final result numerically"),
            ("r1", "tools", ["symbolic_algebra"], ["numeric_check"]),
            ("r2", "action", "enumerate candidate proofs exhaustively", "search for a single elegant proof"),
            ("r4", "action", "try a larger search budget", "switch to a heuristic and report uncertainty"),
            ("r2", "action", "prefer the longest derivation", "prefer the shortest derivation"),
        ],
    ),
    "ALFWorld": P(
        "ALFWorld",
        "embodied",
        "the agent must complete a {topic} task in the simulated home",
        [
            "kitchen cleaning", "bathroom tidying", "bedroom organization",
            "living room staging", "office supply setup", "garage inventory",
            "laundry sorting", "dining table arrangement", "storage room restock",
            "utility closet inspection", "hallway decluttering", "patio arrangement",
            "study room organization", "pantry restocking", "closet consolidation",
        ],
        [
            {
                "r1": TH(action="inspect the current environment state and plan actions from that state", tools=["inspect_state", "plan_actions", "state_check"]),
                "r2": TH(action="verify the state after each action before scheduling the next one"),
                "r4": TH(action="replan from the observed state after any failed action"),
            },
            {
                "r1": TH(action="prefer reversible actions and record how to undo each one", tools=["prefer_reversible", "record_undo", "execute_action"]),
                "r2": TH(action="undo the last action on state divergence from expectation"),
                "r4": TH(action="restore the recorded state before retrying"),
            },
            {
                "r1": TH(action="search receptacles systematically and keep an object location map", tools=["receptacle_search", "location_map", "mark_checked"]),
                "r2": TH(action="check all open receptacles before moving to closed ones"),
                "r5": TH(postcondition="a location map with every found object"),
            },
            {
                "r1": TH(action="split the task into sub-goals and checkpoint after each one", tools=["subgoal_split", "checkpoint", "verify_subgoal"]),
                "r2": TH(action="on a sub-goal failure, reload the last checkpoint"),
                "r4": TH(action="on a single sub-goal failure, keep the task progress and retry that sub-goal"),
            },
            {
                "r1": TH(action="classify failures by action type and apply a recovery plan per type", tools=["failure_type", "recovery_plan", "retry_action"]),
                "r2": TH(action="try the most likely recovery action before a full reset"),
                "r4": TH(action="on a repeated identical failure, reset the episode"),
            },
            {
                "r1": TH(action="inventory every visible object before selecting task targets", tools=["inventory_objects", "visible_scan", "target_select"]),
                "r2": TH(action="confirm the target object still exists before acting on it"),
                "r5": TH(postcondition="a task plan grounded in the full object inventory"),
            },
        ],
        [
            ("r1", "action", "plan the full sequence before acting", "act greedily and replan after each step"),
            ("r1", "tools", ["full_plan"], ["greedy_replan"]),
            ("r2", "action", "move objects one at a time and verify each move", "move several objects together to save steps"),
            ("r4", "action", "reset the task and restart from scratch", "inspect the failed state and continue where possible"),
            ("r2", "action", "check only goal-relevant receptacles", "check every receptacle in the room"),
        ],
    ),
    "CustomerService": P(
        "CustomerService",
        "conversational",
        "the customer asks about {topic}",
        [
            "order status inquiry", "refund request", "account recovery",
            "billing dispute", "subscription cancellation", "shipping delay",
            "damaged item claim", "payment failure", "loyalty points correction",
            "warranty claim", "product return label", "delivery address change",
            "invoice correction", "gift card balance", "promotional code issue",
            "membership upgrade", "data deletion request", "privacy consent update",
            "technical setup help", "billing cycle change", "overdue balance notice",
            "fraud alert review", "recurring charge dispute", "exchange request",
            "partial shipment missing", "delivery signature issue", "prepaid balance refund",
            "account merging request", "service outage update", "satisfaction follow-up",
        ],
        [
            {
                "r1": TH(action="look up the applicable policy and apply it exactly", tools=["policy_lookup", "eligibility_check", "apply_policy"]),
                "r2": TH(action="quote the policy section that supports the decision"),
                "r4": TH(action="verify the customer context before applying any policy exception"),
                "r5": TH(postcondition="a policy-grounded resolution with the cited policy"),
            },
            {
                "r1": TH(action="acknowledge the concern and ask one clarifying question as needed", tools=["acknowledge", "clarify_once", "confirm_need"]),
                "r2": TH(action="restate the customer need in the response before resolving"),
                "r4": TH(action="on a repeated customer request, keep the tone calm"),
            },
            {
                "r1": TH(action="check escalation criteria before promising a resolution", tools=["escalation_check", "triage", "route_case"]),
                "r2": TH(action="escalate cases with fraud, safety, or legal implications"),
                "r4": TH(action="transfer the case with a complete context summary"),
                "r5": TH(postcondition="a case routed to the correct team with full context"),
            },
            {
                "r1": TH(action="create a case record with the request, timestamps, and owner", tools=["create_case", "track_status", "set_owner"]),
                "r2": TH(action="update the case after every customer contact"),
                "r5": TH(postcondition="a case timeline ready for follow-up"),
            },
            {
                "r1": TH(action="verify identity before sharing account details", tools=["identity_verify", "pii_guard", "consent_check"]),
                "r2": TH(action="share only the minimum information required for the request"),
                "r4": TH(action="refuse unsafe data access and log the attempt"),
            },
            {
                "r1": TH(action="investigate the root cause before proposing a workaround", tools=["root_cause", "history_review", "impact_assessment"]),
                "r2": TH(action="distinguish account, system, and policy causes"),
                "r4": TH(action="propose a temporary workaround only on an unclear root cause"),
                "r5": TH(postcondition="a resolution with the root cause documented"),
            },
        ],
        [
            ("r1", "action", "resolve the issue with the minimum policy action", "escalate every unresolved request immediately"),
            ("r1", "tools", ["minimum_policy_action"], ["immediate_escalation"]),
            ("r2", "action", "offer a refund first", "offer a replacement first"),
            ("r4", "action", "repeat the policy verbatim", "paraphrase the policy in plain language"),
            ("r2", "action", "resolve the request in one reply", "split the resolution into multiple follow-up messages"),
        ],
    ),
    "CodeGeneration": P(
        "CodeGeneration",
        "engineering",
        "the user requests a {topic} implementation",
        [
            "REST API client", "data cleaning pipeline", "SQL reporting query",
            "React form component", "pytest test suite", "CLI configuration tool",
            "web scraping script", "CSV parser", "ETL job",
            "CI workflow", "logging middleware", "cache invalidation service",
            "authentication helper", "file watcher", "markdown generator",
            "queue consumer", "rate limiting proxy", "feature flag loader",
            "database migration script", "email templating function", "search index builder",
            "scheduled backup job", "error tracking adapter", "metrics exporter",
            "command dispatcher", "config validator", "template renderer",
            "dependency updater", "API pagination helper", "report generator",
        ],
        [
            {
                "r1": TH(action="define typed contracts for inputs and outputs before writing code", tools=["type_contract", "signature_design", "code_write"]),
                "r2": TH(action="enforce the contracts at runtime boundaries"),
                "r4": TH(action="fix contract violations instead of bypassing them"),
                "r5": TH(postcondition="code with explicit typed contracts"),
            },
            {
                "r1": TH(action="write minimal idiomatic code using the project style guide", tools=["style_guide", "minimal_impl", "idiom_check"]),
                "r2": TH(action="remove unused imports and redundant branches before delivery"),
                "r4": TH(action="prefer the standard library over new dependencies"),
            },
            {
                "r1": TH(action="write failing tests first and implement until they pass", tools=["test_first", "run_tests", "implement_until_pass"]),
                "r2": TH(action="cover happy path, edge cases, and error paths in the test suite"),
                "r4": TH(action="fix implementation regressions and rerun the full suite"),
                "r5": TH(postcondition="implemented code with a passing test suite"),
            },
            {
                "r1": TH(action="apply security checks for injection, credentials, and permissions", tools=["injection_check", "credential_guard", "permission_check"]),
                "r2": TH(action="use parameterized operations and least privilege by default"),
                "r4": TH(action="on an unsatisfied security check, fail closed"),
                "r5": TH(postcondition="code hardened against common security risks"),
            },
            {
                "r1": TH(action="document the API and usage examples before implementation", tools=["doc_contract", "usage_example", "code_write"]),
                "r2": TH(action="keep examples runnable and in sync with the implementation"),
                "r4": TH(action="on behavior changes, update documentation"),
                "r5": TH(postcondition="code with runnable usage examples"),
            },
            {
                "r1": TH(action="refactor only after tests cover the changed behavior", tools=["coverage_check", "small_refactor", "test_after"]),
                "r2": TH(action="unless the task explicitly changes behavior, keep behavior identical"),
                "r4": TH(action="on a test failure exposing behavior change, revert the refactor"),
            },
        ],
        [
            ("r1", "action", "generate the minimal implementation without dependencies", "generate a robust implementation with extra libraries"),
            ("r1", "tools", ["minimal_stdlib"], ["robust_extra_libs"]),
            ("r2", "action", "write tests before implementation", "write implementation before tests"),
            ("r4", "action", "suppress exceptions and log warnings", "propagate exceptions for caller handling"),
            ("r2", "action", "reuse existing utility functions", "write self-contained helpers for every step"),
        ],
    ),
}


TYPE_III_IV_TEMPLATES = {
    "SearchQA": [
        ("r1", "precondition", "the answer is not yet known and multiple sources may contain conflicting values", "the answer is not yet known and every candidate source must be verified independently", "III"),
        ("r2", "postcondition", "a reconciled evidence set with confidence labels and verified sources", "a reconciled evidence set with confidence labels and source quality rankings", "III"),
        ("r5", "postcondition", "a complete answer consistent with verified evidence and format checks", "a complete answer with inline passage citations", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "candidate spans ranked by phrase overlap")], "b": [("r2", "precondition", "evidence with confidence labels from cross checked sources")]},
        {"type": "IV", "a": [("r2", "tools", ["merge_evidence", "label_confidence"])], "b": [("r5", "action", "generate the answer in the required format using only reconciled evidence")]},
    ],
    "SpreadsheetBench": [
        ("r1", "precondition", "the data is stored in a spreadsheet with merged cells", "the data is stored in a spreadsheet with multiple sheets and named ranges", "III"),
        ("r2", "postcondition", "a normalized value set with source cell references", "a normalized value set with original cell addresses", "III"),
        ("r5", "postcondition", "a complete output with totals and a correction log", "a formatted output with verified totals and column provenance", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "raw cell values extracted without type conversion")], "b": [("r2", "precondition", "the values are typed and ready for aggregation")]},
        {"type": "IV", "a": [("r2", "tools", ["normalize_value", "record_cell"])], "b": [("r5", "action", "generate the output table from normalized values and verify row and column totals")]},
    ],
    "OfficeQA": [
        ("r1", "precondition", "the answer is not yet known and the target documents may be incomplete", "the answer is not yet known and the document set must be fully indexed", "III"),
        ("r2", "postcondition", "a reconciled evidence set with confidence labels and format tags", "a reconciled evidence set with confidence labels and document provenance", "III"),
        ("r5", "postcondition", "a complete answer consistent with verified evidence and section anchors", "a complete answer with section and paragraph citations", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "document spans ranked by format detection")], "b": [("r2", "precondition", "evidence from the target document format only")]},
        {"type": "IV", "a": [("r2", "tools", ["merge_evidence", "resolve_inconsistency"])], "b": [("r5", "action", "generate the answer using only format reconciled evidence")]},
    ],
    "DocVQA": [
        ("r1", "precondition", "the answer is not yet known and the scanned pages may be incomplete", "the answer is not yet known and every page must be OCR verified", "III"),
        ("r2", "postcondition", "a reconciled evidence set with confidence labels and page order", "a reconciled evidence set with confidence labels and spatial layout", "III"),
        ("r5", "postcondition", "a complete answer with extracted text spans", "a complete answer with region coordinates", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "raw OCR text regions without layout verification")], "b": [("r2", "precondition", "extracted regions with verified layout order")]},
        {"type": "IV", "a": [("r2", "tools", ["merge_evidence", "label_confidence"])], "b": [("r5", "action", "generate the answer using only layout verified regions")]},
    ],
    "LiveMathematicianBench": [
        ("r1", "precondition", "the answer is not yet known and the derivation may require multiple lemmas", "the answer is not yet known and every lemma must be independently verified", "III"),
        ("r2", "postcondition", "a reconciled proof chain with step checks", "a reconciled proof chain with confidence labels", "III"),
        ("r5", "postcondition", "a complete result with the main derivation", "a complete result with verified derivation steps", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "candidate derivations ranked by symbolic similarity")], "b": [("r2", "precondition", "a complete derivation with all steps verified")]},
        {"type": "IV", "a": [("r2", "tools", ["merge_evidence", "label_confidence"])], "b": [("r5", "action", "generate the result using only verified derivation chains")]},
    ],
    "ALFWorld": [
        ("r1", "precondition", "the environment contains visible target objects", "the environment contains the target objects in known receptacles", "III"),
        ("r2", "postcondition", "an updated plan grounded in observed state changes", "an updated plan grounded in the observed state with object locations", "III"),
        ("r5", "postcondition", "a final status report with executed actions", "a final status report with executed actions and object locations", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "a grounded task plan without object location verification")], "b": [("r2", "precondition", "the environment state matches the planned object locations")]},
        {"type": "IV", "a": [("r2", "tools", ["observe_state", "compare_state"])], "b": [("r5", "action", "report the completed status using observed state comparisons")]},
    ],
    "CustomerService": [
        ("r1", "precondition", "the request concerns an account or service issue", "the request concerns a verified account or order", "III"),
        ("r2", "postcondition", "a confirmed customer context with the missing details", "a confirmed customer context with policy citations ready for resolution", "III"),
        ("r5", "postcondition", "a clear resolution message with the next step", "a clear resolution message with next steps and a reference number", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "the customer need identified without policy lookup")], "b": [("r2", "precondition", "the resolution depends on the applicable policy")]},
        {"type": "IV", "a": [("r2", "tools", ["ask_details", "confirm_context"])], "b": [("r5", "action", "state the resolution using confirmed context only")]},
    ],
    "CodeGeneration": [
        ("r1", "precondition", "the task requires creating new source files", "the task requires modifying existing source files without changing public interfaces", "III"),
        ("r2", "postcondition", "tested code with a reviewed change list", "tested code with a reviewed diff and coverage report", "III"),
        ("r5", "postcondition", "a complete implementation with basic usage notes", "a complete implementation with usage notes and migration guide", "III"),
        {"type": "IV", "a": [("r1", "postcondition", "standalone code with no integration tests")], "b": [("r2", "precondition", "the code must pass the full integration suite")]},
        {"type": "IV", "a": [("r2", "tools", ["run_tests", "review_diff"])], "b": [("r5", "action", "generate the final code from reviewed diffs only")]},
    ],
}


def add_iii_iv_templates():
    """Append Type III and Type IV conflict templates to every domain pool."""
    for domain, templates in TYPE_III_IV_TEMPLATES.items():
        profile = DOMAIN_PROFILES.get(domain)
        if profile is None:
            continue
        existing_keys = {(p[0], p[1]) for p in profile["pairs"] if isinstance(p, (tuple, list))}
        for t in templates:
            if isinstance(t, dict):
                profile["pairs"].append(t)
            else:
                key = (t[0], t[1])
                if key not in existing_keys:
                    profile["pairs"].append(t)
                    existing_keys.add(key)


POLISH_EDITS = {
    "retrieval": [
        ("r1", "precondition", "the question has multiple constraints that must be satisfied together"),
        ("r1", "postcondition", "a grounded result with explicit source links"),
        ("r1", "tools", ["gather_context", "verify_scope", "source_link"]),
        ("r2", "postcondition", "a traceable evidence set with provenance for every value"),
        ("r2", "tools", ["provenance_trace", "audit_log"]),
        ("r3", "action", "execute each step with a checkpoint and compare the observed outcome with the expected outcome"),
        ("r3", "tools", ["checkpoint_state", "compare_outcome"]),
        ("r4", "postcondition", "a clean recovery with the original data preserved"),
        ("r4", "tools", ["preserve_state", "recovery_log"]),
        ("r5", "action", "assemble the final output from verified intermediate results and mark its provenance"),
        ("r5", "postcondition", "a final output with provenance and consistency checks"),
        ("r5", "tools", ["assemble_output", "mark_provenance"]),
    ],
    "tabular": [
        ("r1", "precondition", "the requested values span multiple sheets or ranges"),
        ("r1", "postcondition", "the values extracted with row and column provenance"),
        ("r1", "tools", ["sheet_provenance", "range_verify"]),
        ("r2", "postcondition", "a normalized value set with original cell provenance"),
        ("r2", "tools", ["provenance_trace", "audit_log"]),
        ("r3", "action", "execute each data stage with a checkpoint and compare row counts and totals"),
        ("r3", "tools", ["stage_checkpoint", "total_compare"]),
        ("r4", "postcondition", "a clean recovery with the source table preserved"),
        ("r4", "tools", ["preserve_table", "recovery_log"]),
        ("r5", "action", "assemble the output table from verified intermediate results and mark its provenance"),
        ("r5", "postcondition", "a final table with provenance and verified totals"),
        ("r5", "tools", ["assemble_table", "mark_provenance"]),
    ],
    "embodied": [
        ("r1", "precondition", "the target objects may be hidden inside multiple receptacles"),
        ("r1", "postcondition", "a grounded task plan with explicit object locations"),
        ("r1", "tools", ["object_provenance", "plan_verify"]),
        ("r2", "postcondition", "a state trace with the expected and observed values for every step"),
        ("r2", "tools", ["state_trace", "compare_log"]),
        ("r3", "action", "execute each sub-goal with a state checkpoint and compare the observed state with the expected state"),
        ("r3", "tools", ["state_checkpoint", "outcome_compare"]),
        ("r4", "postcondition", "a clean recovery with the environment state preserved"),
        ("r4", "tools", ["preserve_state", "recovery_log"]),
        ("r5", "action", "assemble the final status from verified sub-goal results and mark their provenance"),
        ("r5", "postcondition", "a final status with verified sub-goal provenance"),
        ("r5", "tools", ["assemble_status", "mark_provenance"]),
    ],
    "conversational": [
        ("r1", "precondition", "the customer context spans multiple accounts or orders"),
        ("r1", "postcondition", "a grounded customer context with source records"),
        ("r1", "tools", ["record_provenance", "context_verify"]),
        ("r2", "postcondition", "a confirmed context with the details needed for resolution"),
        ("r2", "tools", ["detail_trace", "audit_log"]),
        ("r3", "action", "execute each customer sub-goal with a checkpoint and verify the customer outcome before continuing"),
        ("r3", "tools", ["outcome_checkpoint", "compare_outcome"]),
        ("r4", "postcondition", "a clean recovery with the customer context preserved"),
        ("r4", "tools", ["preserve_context", "recovery_log"]),
        ("r5", "action", "assemble the final resolution from verified case records and mark their provenance"),
        ("r5", "postcondition", "a final resolution with verified case provenance"),
        ("r5", "tools", ["assemble_resolution", "mark_provenance"]),
    ],
    "engineering": [
        ("r1", "precondition", "the change must satisfy existing interfaces and build constraints"),
        ("r1", "postcondition", "a grounded implementation with explicit dependency notes"),
        ("r1", "tools", ["interface_check", "dependency_note"]),
        ("r2", "postcondition", "a reviewed diff with test evidence for every changed path"),
        ("r2", "tools", ["test_evidence", "diff_audit"]),
        ("r3", "action", "execute each code sub-goal with a build checkpoint and compare the observed behavior with the expected behavior"),
        ("r3", "tools", ["build_checkpoint", "behavior_compare"]),
        ("r4", "postcondition", "a clean recovery with the previous code state preserved"),
        ("r4", "tools", ["preserve_code", "recovery_log"]),
        ("r5", "action", "assemble the final implementation from verified sub-goal results and mark their provenance"),
        ("r5", "postcondition", "a final implementation with verified change provenance"),
        ("r5", "tools", ["assemble_change", "mark_provenance"]),
    ],
}


def field_value(rule, field):
    if field == "tools":
        return tuple(sorted(rule.get("tools", [])))
    return str(rule.get(field, "")).strip()


def compute_edit_set(base_rules, traj_rules, k):
    """Replicate SkillDocument.compute_edit_set output as plain dicts."""
    edits = []
    base_by_id = {r["rule_id"]: r for r in base_rules}
    cur_by_id = {r["rule_id"]: r for r in traj_rules}
    for rid in sorted(set(base_by_id) | set(cur_by_id)):
        br = base_by_id.get(rid)
        cr = cur_by_id.get(rid)
        if br is None:
            for f in FIELD_ORDER:
                val = field_value(cr, f)
                edits.append({
                    "rule_id": rid,
                    "field": f,
                    "v_old": None,
                    "v_new": ",".join(val) if f == "tools" else val,
                    "trajectory_k": k,
                    "category": "action_addition",
                })
            continue
        if cr is None:
            for f in FIELD_ORDER:
                val = field_value(br, f)
                edits.append({
                    "rule_id": rid,
                    "field": f,
                    "v_old": ",".join(val) if f == "tools" else val,
                    "v_new": None,
                    "trajectory_k": k,
                    "category": "action_removal",
                })
            continue
        for f in FIELD_ORDER:
            ov = field_value(br, f)
            nv = field_value(cr, f)
            if ov == nv:
                continue
            category = "value_replacement"
            if f == "trigger":
                if ov and nv and ov in nv:
                    category = "trigger_expansion"
                elif ov and nv and nv in ov:
                    category = "trigger_narrowing"
            elif f == "action":
                if not ov and nv:
                    category = "action_addition"
                elif ov and not nv:
                    category = "action_removal"
            elif f == "tools":
                category = "tool_set_modification"
            edits.append({
                "rule_id": rid,
                "field": f,
                "v_old": ",".join(ov) if f == "tools" else ov,
                "v_new": ",".join(nv) if f == "tools" else nv,
                "trajectory_k": k,
                "category": category,
            })
    return edits


def render_md(doc_id, rules):
    lines = [f"# Skill Document: {doc_id}", ""]
    for r in rules:
        tools = ", ".join(sorted(r.get("tools", [])))
        lines.append(f"## Rule {r['rule_id']}")
        lines.append(
            f"WHEN {r['trigger']}, IF {r['precondition']}, "
            f"THEN {r['action']}, RESULTING IN {r['postcondition']}. "
            f"TOOLS: {tools}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clone_rules(rules):
    return [
        {
            "rule_id": r["rule_id"],
            "trigger": r["trigger"],
            "precondition": r["precondition"],
            "action": r["action"],
            "postcondition": r["postcondition"],
            "tools": list(r["tools"]),
        }
        for r in rules
    ]


def skeleton_rules(family):
    return [
        {"rule_id": rid, **fields}
        for rid, fields in BASE_SKELETONS[family].items()
    ]


def build_trajectory(profile, base_rules, topic, k, theme_index, forced, rng):
    rules = clone_rules(base_rules)
    theme = profile["themes"][theme_index % len(profile["themes"])]
    candidates = []

    for rid, overrides in theme.items():
        if rid == "_label":
            continue
        base_rule = next(r for r in base_rules if r["rule_id"] == rid)
        for f, val in overrides.items():
            if field_value(base_rule, f) != field_value({f: val}, f):
                candidates.append((rid, f, val))

    topic_value = profile["topic_trigger"].format(topic=topic)
    if field_value(base_rules[0], "trigger") != topic_value:
        candidates.append(("r1", "trigger", topic_value))

    forced_edits = []
    for (rid, f), val in forced.items():
        candidates = [c for c in candidates if not (c[0] == rid and c[1] == f)]
        forced_edits.append((rid, f, val))
        candidates.append((rid, f, val))

    selected = []
    for rid in RULE_IDS:
        opts = [c for c in candidates if c[0] == rid]
        if opts:
            selected.append(rng.choice(opts))
    for c in forced_edits:
        if c not in selected:
            selected.append(c)

    remaining = [c for c in candidates if c not in selected]
    rng.shuffle(remaining)
    n_target = max(len(selected), rng.randint(8, 15))
    selected += remaining[: max(0, n_target - len(selected))]

    for rid, f, val in selected:
        rule = next(r for r in rules if r["rule_id"] == rid)
        if f == "tools":
            rule[f] = sorted(set(val))
        else:
            rule[f] = val

    edit_set = compute_edit_set(base_rules, rules, k)
    pool = list(POLISH_EDITS[profile["family"]])
    min_edits = rng.randint(8, 12)
    while len(edit_set) < min_edits and pool:
        changed = {(e["rule_id"], e["field"]) for e in edit_set}
        available = [c for c in pool if (c[0], c[1]) not in changed]
        if not available:
            break
        rid, f, val = available[rng.randrange(len(available))]
        rule = next(r for r in rules if r["rule_id"] == rid)
        if f == "tools":
            rule[f] = sorted(set(val))
        else:
            rule[f] = val
        pool.remove((rid, f, val))
        edit_set = compute_edit_set(base_rules, rules, k)
    return rules, edit_set


def make_scenario(root, rng, scenario_id, subset, domain, topic, k, conflict_pairs):
    profile = DOMAIN_PROFILES[domain]
    conflict_pairs = conflict_pairs or []
    base_rules = skeleton_rules(profile["family"])
    theme_indexes = rng.sample(range(len(profile["themes"])), k)

    forced_by_traj = {kk: {} for kk in range(1, k + 1)}
    pair_meta = []
    if conflict_pairs:
        used_keys = set()
        for pair in conflict_pairs:
            if isinstance(pair, dict):
                edits_a = pair.get("a", [])
                edits_b = pair.get("b", [])
                pair_keys = [(r, f) for r, f, _ in edits_a] + [(r, f) for r, f, _ in edits_b]
                if any(k in used_keys for k in pair_keys):
                    continue
                traj_ids = rng.sample(range(1, k + 1), 2)
                ta, tb = traj_ids
                for rid, f, val in edits_a:
                    forced_by_traj[ta][(rid, f)] = val
                    used_keys.add((rid, f))
                for rid, f, val in edits_b:
                    forced_by_traj[tb][(rid, f)] = val
                    used_keys.add((rid, f))
                pair_meta.append({
                    "rule_id": f"{edits_a[0][0]}+{edits_b[0][0]}",
                    "field": f"{edits_a[0][1]}+{edits_b[0][1]}",
                    "trajectories": [ta, tb],
                    "type": "IV",
                    "edits": {"trajectory_a": edits_a, "trajectory_b": edits_b},
                })
                continue
            rid, f, val_a, val_b = pair[0], pair[1], pair[2], pair[3]
            key = (rid, f)
            if key in used_keys:
                continue
            traj_ids = rng.sample(range(1, k + 1), 2)
            ta, tb = traj_ids
            forced_by_traj[ta][key] = val_a
            forced_by_traj[tb][key] = val_b
            used_keys.add(key)
            conflict_type = pair[4] if len(pair) >= 5 else "I"
            if len(pair) < 5:
                if f == "tools":
                    conflict_type = "II"
                elif f in ("trigger", "precondition", "postcondition"):
                    conflict_type = "III"
            pair_meta.append({
                "rule_id": rid,
                "field": f,
                "trajectories": [ta, tb],
                "type": conflict_type,
            })

    trajectories = {}
    total_edits = 0
    for kk in range(1, k + 1):
        theme_label = profile["themes"][theme_indexes[kk - 1]]["_label"]
        rules, edit_set = build_trajectory(
            profile, base_rules, topic, kk, theme_indexes[kk - 1],
            forced_by_traj[kk], rng,
        )
        total_edits += len(edit_set)
        doc = {
            "doc_id": f"S{kk}",
            "trajectory_k": kk,
            "description": f"{domain} | {topic} | {theme_label}",
            "domain": domain,
            "rules": rules,
            "edit_set": {"trajectory_k": kk, "edits": edit_set},
        }
        trajectories[kk] = doc

    manifest = {
        "scenario_id": scenario_id,
        "subset": subset,
        "split": "test",
        "domain": domain,
        "topic": topic,
        "k": k,
        "conflict_density": round(len(conflict_pairs) / total_edits, 4) if total_edits else 0.0,
        "semantic_noise": 0.0,
        "s0": "S0_base.json",
        "trajectories": {
            str(kk): {
                "domain": domain,
                "file": f"S{kk}_optimized.json",
                "description": trajectories[kk]["description"],
                "n_edits": len(trajectories[kk]["edit_set"]["edits"]),
            }
            for kk in sorted(trajectories)
        },
    }
    if conflict_pairs:
        manifest["conflict_pairs"] = pair_meta

    return base_rules, trajectories, manifest


def write_scenario(root, rng, scenario_id, subset, domain, topic, k, conflict_pairs):
    if subset == "base":
        folder = root / "held_out" / f"base_{scenario_id:04d}"
    elif subset == "high_conflict":
        folder = root / "held_out" / f"high_conflict_{scenario_id:04d}"
    elif subset == "customer_service":
        folder = root / "cross_domain" / f"customer_service_{scenario_id:04d}"
    else:
        folder = root / "cross_domain" / f"code_generation_{scenario_id:04d}"

    json_dir = folder / "dataset_json"
    md_dir = folder / "dataset_md"
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    base_rules, trajectories, manifest = make_scenario(
        root, rng, scenario_id, subset, domain, topic, k, conflict_pairs,
    )
    write_json(json_dir / "manifest.json", manifest)
    write_json(json_dir / "S0_base.json", {"doc_id": "S0", "rules": base_rules})
    (md_dir / "S0.md").write_text(render_md("S0", base_rules), encoding="utf-8")
    for kk, doc in trajectories.items():
        write_json(json_dir / f"S{kk}_optimized.json", doc)
        (md_dir / f"S{kk}.md").write_text(render_md(f"S{kk}", doc["rules"]), encoding="utf-8")
    return manifest


def add_theme_labels():
    for profile in DOMAIN_PROFILES.values():
        for i, theme in enumerate(profile["themes"]):
            theme["_label"] = f"theme_{i + 1}"


def generate(root, seed):
    add_theme_labels()
    add_iii_iv_templates()
    rng = random.Random(seed)
    scenarios = []
    base_domains = ["SearchQA", "SpreadsheetBench", "OfficeQA", "DocVQA", "LiveMathematicianBench", "ALFWorld"]

    for i in range(135):
        domain = base_domains[i % len(base_domains)]
        profile = DOMAIN_PROFILES[domain]
        topic = profile["topics"][i % len(profile["topics"])]
        k = 2 + (i % 5)
        scenario_id = i + 1
        manifest = write_scenario(
            root, rng, scenario_id, "base", domain, topic, k, None,
        )
        scenarios.append({
            "id": manifest["scenario_id"],
            "subset": "base",
            "domain": domain,
            "k": k,
            "path": f"held_out/base_{scenario_id:04d}",
        })

    for i in range(36):
        domain = base_domains[i % len(base_domains)]
        profile = DOMAIN_PROFILES[domain]
        topic = profile["topics"][(i + 7) % len(profile["topics"])]
        k = 3 + (i % 4)
        n_pairs = min(3 + (i % 4), len(profile["pairs"]))
        conflict_pairs = rng.sample(profile["pairs"], n_pairs)
        scenario_id = i + 1
        manifest = write_scenario(
            root, rng, scenario_id, "high_conflict", domain, topic, k, conflict_pairs,
        )
        scenarios.append({
            "id": f"high_conflict_{scenario_id:04d}",
            "subset": "high_conflict",
            "domain": domain,
            "k": k,
            "conflict_pairs": len(conflict_pairs),
            "path": f"held_out/high_conflict_{scenario_id:04d}",
        })

    for i in range(30):
        topic = DOMAIN_PROFILES["CustomerService"]["topics"][i]
        k = 3 + (i % 3)
        scenario_id = i + 1
        manifest = write_scenario(
            root, rng, scenario_id, "customer_service", "CustomerService", topic, k, None,
        )
        scenarios.append({
            "id": f"customer_service_{scenario_id:04d}",
            "subset": "customer_service",
            "domain": "CustomerService",
            "k": k,
            "path": f"cross_domain/customer_service_{scenario_id:04d}",
        })

    for i in range(30):
        topic = DOMAIN_PROFILES["CodeGeneration"]["topics"][i]
        k = 3 + (i % 3)
        scenario_id = i + 1
        manifest = write_scenario(
            root, rng, scenario_id, "code_generation", "CodeGeneration", topic, k, None,
        )
        scenarios.append({
            "id": f"code_generation_{scenario_id:04d}",
            "subset": "code_generation",
            "domain": "CodeGeneration",
            "k": k,
            "path": f"cross_domain/code_generation_{scenario_id:04d}",
        })

    index = {
        "benchmark": "MergeBench v3.0",
        "version": "3.0",
        "seed": seed,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "layout": "one directory per scenario with dataset_json and dataset_md",
        "counts": {
            "held_out_base": 135,
            "held_out_high_conflict": 36,
            "held_out_total": 171,
            "cross_domain_customer_service": 30,
            "cross_domain_code_generation": 30,
            "cross_domain_total": 60,
            "total": 231,
        },
        "scenarios": scenarios,
    }
    write_json(root / "index.json", index)

    readme = (
        "# MergeBench v3.0\n\n"
        "Benchmark input scenarios for SCR-Merge Ultima. These files define the\n"
        "merge tasks; they are not experimental result measurements.\n\n"
        "## Counts\n\n"
        "- Held-out test: 135 base + 36 high-conflict = 171 scenarios\n"
        "- Cross-domain: 30 customer service + 30 code generation = 60 scenarios\n\n"
        "## Scenario format\n\n"
        "Each scenario directory contains:\n\n"
        "- `dataset_json/manifest.json` with scenario metadata and trajectory map\n"
        "- `dataset_json/S0_base.json` plus `S1_optimized.json` through `SK_optimized.json`\n"
        "- `dataset_md/` with the same documents rendered as structured Markdown\n\n"
        "Rules use the 5-tuple format `rule_id / trigger / precondition / action /\n"
        "postcondition / tools`. Trajectory JSON files also include `edit_set`, which\n"
        "records every field-level edit from the base document.\n\n"
        "## Regeneration\n\n"
        "```powershell\n"
        "python generate_mergebench_v3.py --root \"E:\\资料备份\\论文写作\\第3篇\\我的方案\\代码\\SkillSCR\\data\\MergeBench_v3\"\n"
        "```\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")


def validate(root):
    json_dirs = list((root / "held_out").rglob("dataset_json"))
    json_dirs += list((root / "cross_domain").rglob("dataset_json"))
    if len(json_dirs) != 231:
        raise SystemExit(f"expected 231 scenario dirs, found {len(json_dirs)}")

    n_base = n_high = n_cs = n_cg = 0
    bad = []
    for jd in json_dirs:
        manifest = json.loads((jd / "manifest.json").read_text(encoding="utf-8"))
        s0 = json.loads((jd / "S0_base.json").read_text(encoding="utf-8"))
        md_dir = jd.parent / "dataset_md"
        if not md_dir.exists():
            bad.append(f"{jd} missing dataset_md")
            continue
        subset = manifest["subset"]
        if subset == "base":
            n_base += 1
        elif subset == "high_conflict":
            n_high += 1
        elif subset == "customer_service":
            n_cs += 1
        elif subset == "code_generation":
            n_cg += 1
        else:
            bad.append(f"{jd} unknown subset {subset}")
        if len(s0.get("rules", [])) != 5:
            bad.append(f"{jd} S0 does not have 5 rules")
        for kk, meta in manifest["trajectories"].items():
            doc_path = jd / meta["file"]
            if not doc_path.exists():
                bad.append(f"{doc_path} missing")
                continue
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            if len(doc.get("rules", [])) != 5:
                bad.append(f"{doc_path} does not have 5 rules")
            if len(doc.get("edit_set", {}).get("edits", [])) != meta["n_edits"]:
                bad.append(f"{doc_path} n_edits mismatch")
            recomputed = compute_edit_set(s0["rules"], doc["rules"], int(kk))
            stored = doc["edit_set"]["edits"]
            norm = lambda e: (e["rule_id"], e["field"], e["v_old"], e["v_new"], e["category"])
            if [norm(e) for e in recomputed] != [norm(e) for e in stored]:
                bad.append(f"{doc_path} edit_set inconsistent with rules")
            md_file = md_dir / f"S{kk}.md"
            if not md_file.exists():
                bad.append(f"{md_file} missing")
            elif len(re.findall(r"^## Rule ", md_file.read_text(encoding="utf-8"), re.MULTILINE)) != 5:
                bad.append(f"{md_file} rule count mismatch")
        if not (md_dir / "S0.md").exists():
            bad.append(f"{md_dir / 'S0.md'} missing")

    expected = (135, 36, 30, 30)
    actual = (n_base, n_high, n_cs, n_cg)
    if actual != expected:
        bad.append(f"count mismatch: {actual} != {expected}")
    if bad:
        for line in bad[:80]:
            print("BAD:", line)
        raise SystemExit(f"validation failed with {len(bad)} issues")
    print(f"validation OK: base={n_base} high_conflict={n_high} customer_service={n_cs} code_generation={n_cg}")


def main():
    parser = argparse.ArgumentParser(description="Generate MergeBench v3.0 scenario folders")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    generate(root, args.seed)

    script_target = root.parent.parent / "scripts" / "generate_mergebench_v3.py"
    try:
        script_target.parent.mkdir(parents=True, exist_ok=True)
        script_target.write_text(Path(__file__).resolve().read_text(encoding="utf-8"), encoding="utf-8")
        print(f"generator copied to {script_target}")
    except Exception as exc:
        print(f"WARN could not copy generator: {exc}")

    validate(root)
    print(f"generated {root}")


if __name__ == "__main__":
    main()
