"""
MASCoD Judge Agent (Phase 4) - single-program LangGraph node.

This node intentionally does not load benchmark JSON/JSONL files. The root
agent is expected to pass one large JSON-like state object containing the
program pair, execution ground truth, prediction, and any mutation metadata.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import re
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

judge_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    model_kwargs={"response_format": {"type": "json_object"}},
)


# ---------------------------------------------------------------------------
# Ground-truth and prediction normalization
# ---------------------------------------------------------------------------

ALLOWED_LABELS = {"EQUIV", "DIVERGE", "ERROR"}


def _normalize_label(label: Any) -> str:
    normalized = str(label or "ERROR").strip().upper()
    return normalized if normalized in ALLOWED_LABELS else "ERROR"


def _label_from_outputs_diverge(outputs_diverge: Any) -> str:
    if isinstance(outputs_diverge, str):
        normalized = outputs_diverge.strip().lower()
        if normalized in {"error", "err", "failed", "failure", "none", "null"}:
            return "ERROR"
        if normalized in {"true", "diverge", "divergent", "different"}:
            return "DIVERGE"
        if normalized in {"false", "equiv", "equivalent", "same"}:
            return "EQUIV"

    if outputs_diverge is True:
        return "DIVERGE"
    if outputs_diverge is False:
        return "EQUIV"
    return "ERROR"


def _derive_ground_truth_label(ground_truth: Mapping[str, Any]) -> str:
    status_a = ground_truth.get("compilation_status_A", "")
    status_b = ground_truth.get("compilation_status_B", "")

    if status_a != "success" or status_b != "success":
        return "ERROR"

    return _label_from_outputs_diverge(ground_truth.get("outputs_diverge"))


def _compute_objective_score(
    predicted_label: str,
    ground_truth_label: str,
) -> Dict[str, Any]:
    label_match = predicted_label == ground_truth_label
    return {
        "label_match": label_match,
        "correctness": 1.0 if label_match else 0.0,
        "predicted_label": predicted_label,
        "ground_truth_label": ground_truth_label,
    }


def _normalize_prediction_payload(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(prediction.get("evaluation_result"), dict):
        merged = {**prediction["evaluation_result"]}
        merged.setdefault("predicted_label", merged.get("label"))
    else:
        merged = {**prediction}

    if "predicted_output" not in merged and (
        "predicted_output_A" in merged or "predicted_output_B" in merged
    ):
        merged["predicted_output"] = {
            "programA": merged.get("predicted_output_A"),
            "programB": merged.get("predicted_output_B"),
        }

    merged.setdefault("predicted_label", merged.get("label"))
    return merged


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _prediction_claims_error(
    predicted_label: str,
    predicted_output: Any,
    explanation: str,
) -> bool:
    if _normalize_label(predicted_label) == "ERROR":
        return True

    text = f"{_stringify(predicted_output)} {explanation or ''}".lower()
    negated_error = re.search(
        r"\b(no|not|without|does not|doesn't|do not|don't)\b.{0,24}\b(error|fail|failure|failed)\b",
        text,
    )
    positive_error = re.search(
        r"\b(error|failure|fails?|failed|cannot compile|does not compile|"
        r"compilation failure|compilation error|runtime failure|runtime error|"
        r"verification failure|verification error|invalid|unsupported|exception)\b",
        text,
    )
    return bool(positive_error and not negated_error)


# ---------------------------------------------------------------------------
# Recursive JSON extraction from the incoming state
# ---------------------------------------------------------------------------

def _iter_dicts(root: Any) -> Iterable[Dict[str, Any]]:
    """Breadth-first traversal over dictionaries inside the root-passed JSON."""
    queue: deque[Any] = deque([root])
    seen: set[int] = set()

    while queue:
        value = queue.popleft()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)

        if isinstance(value, dict):
            yield value
            queue.extend(value.values())
        elif isinstance(value, list):
            queue.extend(value)


def _first_mapping_with_keys(
    root: Any,
    required_any: Sequence[str],
    required_all: Sequence[str] = (),
) -> Dict[str, Any]:
    any_keys = set(required_any)
    all_keys = set(required_all)
    for item in _iter_dicts(root):
        if all_keys and not all_keys.issubset(item.keys()):
            continue
        if any_keys and not any_keys.intersection(item.keys()):
            continue
        return item
    return {}


def _first_value(root: Any, keys: Sequence[str]) -> Any:
    key_set = set(keys)
    for item in _iter_dicts(root):
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        for key, value in item.items():
            if key.lower() in {candidate.lower() for candidate in key_set} and value not in (None, ""):
                return value
    return None


def _normalize_root_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Choose the root-passed JSON object as this node's only data source."""
    for key in ("judge_input", "judge_payload_input", "root_payload", "payload", "input"):
        value = state.get(key)
        if isinstance(value, dict):
            return value
    return dict(state)


def _extract_program_pair(root: Mapping[str, Any]) -> Dict[str, Any]:
    source = _first_mapping_with_keys(
        root,
        required_any=("programA", "program_a", "input_mlir", "inputMLIR", "original_mlir"),
        required_all=(),
    )
    program_a = (
        source.get("programA")
        or source.get("program_a")
        or source.get("original_mlir")
        or source.get("input_mlir")
        or source.get("inputMLIR")
    )
    program_b = (
        source.get("programB")
        or source.get("program_b")
        or source.get("mutated_mlir")
        or source.get("mutatedMLIR")
        or source.get("output_mlir")
    )

    return {
        "program_id": source.get("program_id") or _first_value(root, ("program_id",)),
        "pair_id": source.get("pair_id") or _first_value(root, ("pair_id",)),
        "programA": program_a,
        "programB": program_b,
        "dialect": source.get("dialect")
        or source.get("selected_dialect")
        or _first_value(root, ("dialect", "selected_dialect", "user_selected_dialect")),
        "tool_used": source.get("tool_used") or _first_value(root, ("tool_used",)),
        "mutation_id": source.get("mutation_id") or _first_value(root, ("mutation_id",)),
        "mutation_kind": source.get("mutation_kind")
        or source.get("requirement_description")
        or _first_value(root, ("mutation_kind", "requirement_description")),
        "source": source.get("source") or _first_value(root, ("source",)),
        "metadata": source.get("metadata", {}),
    }


def _extract_execution_ground_truth(
    root: Mapping[str, Any],
    program_pair: Mapping[str, Any],
) -> Dict[str, Any]:
    source = _first_mapping_with_keys(
        root,
        required_any=(
            "programA_output",
            "programB_output",
            "compilation_status_A",
            "compilation_status_B",
            "outputs_diverge",
        ),
    )
    return {
        "program_id": source.get("program_id") or program_pair.get("program_id"),
        "embedded_inputs": source.get("embedded_inputs"),
        "compilation_status_A": source.get("compilation_status_A"),
        "compilation_status_B": source.get("compilation_status_B"),
        "programA_output": source.get("programA_output"),
        "programB_output": source.get("programB_output"),
        "outputs_diverge": source.get("outputs_diverge"),
        "divergence_details": source.get("divergence_details"),
        "semantic_explanation": source.get("semantic_explanation"),
    }


def _extract_prediction(root: Mapping[str, Any]) -> Dict[str, Any]:
    source = _first_mapping_with_keys(
        root,
        required_any=(
            "predicted_label",
            "label",
            "predicted_output",
            "predicted_output_A",
            "explanation",
            "evaluation_result",
        ),
    )
    return _normalize_prediction_payload(source)


def _validate_required_inputs(
    program_pair: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> List[str]:
    missing: List[str] = []

    if not program_pair.get("programA"):
        missing.append("programA")
    if not program_pair.get("programB"):
        missing.append("programB")
    if not program_pair.get("dialect"):
        missing.append("dialect")
    if not ground_truth.get("compilation_status_A"):
        missing.append("compilation_status_A")
    if not ground_truth.get("compilation_status_B"):
        missing.append("compilation_status_B")
    if "outputs_diverge" not in ground_truth:
        missing.append("outputs_diverge")
    if not prediction:
        missing.append("prediction/evaluation_result")

    return missing


# ---------------------------------------------------------------------------
# Mutation-aware result-structure rubric
# ---------------------------------------------------------------------------

DEFAULT_STRUCTURE_RUBRIC = {
    "structure_points_max": 2.0,
    "exactness_points_max": 5.0,
    "severity": "default",
    "prompt": (
        "Use the default structure rubric. Award structure credit for correctly "
        "describing return arity, scalar/tensor/list categories, ranks, shapes, "
        "and types for both programs."
    ),
}

STRUCTURE_RUBRIC_PROFILES = {
    "low": {
        "structure_points_max": 1.0,
        "severity": "low",
        "prompt": (
            "The mutation rule is syntactic or explicitly semantics-preserving "
            "(for example SSA renaming, formatting, canonical ordering, or "
            "independent-op reordering). Output structure is expected to be "
            "unchanged, so structure carries low weight. Award structure credit "
            "only for correctly preserving returned arity, categories, ranks, "
            "shapes, and types; do not reward discussion of names or formatting "
            "as output structure."
        ),
    },
    "default": DEFAULT_STRUCTURE_RUBRIC,
    "medium": {
        "structure_points_max": 3.0,
        "severity": "medium",
        "prompt": (
            "The mutation rule can change semantic behavior while often "
            "preserving broad output shape. Structure and exact values both "
            "matter. Award structure credit for correct return arity, categories, "
            "ranks, shapes, and types, especially where the changed operation, "
            "parameter, or data-flow stage affects returned components."
        ),
    },
    "high": {
        "structure_points_max": 4.0,
        "severity": "high",
        "prompt": (
            "The mutation rule directly targets shape, rank, broadcasting, "
            "indexing maps, iteration space, or other output-structure-critical "
            "semantics. Structure carries high weight. Award full structure "
            "credit only when both programs' returned arity, scalar/tensor/list "
            "categories, ranks, shapes, element types, and layout-relevant "
            "structure are correct."
        ),
    },
    "error": {
        "structure_points_max": 2.0,
        "severity": "error-oriented",
        "prompt": (
            "The mutation rule is intended to introduce a semantic, typing, "
            "operator-constraint, dependency, or structural failure. If the "
            "sandbox ground truth is ERROR, use the binary ERROR result mode; "
            "judge the failed side, failure category, diagnostic cause, and "
            "failing operation only under explanation_rating."
        ),
    },
}


def _profile_name_for_rule(rule: Mapping[str, Any], mutation_id: str) -> str:
    description = str(rule.get("description", "")).lower()
    rule_text = " ".join(str(item).lower() for item in rule.get("rules", []))
    text = f"{mutation_id.lower()} {description}"
    fallback_text = f"{text} {rule_text}"

    if re.search(r"\b[lt]5_r\d\b", text) or any(
        token in text
        for token in [
            "semantic failure",
            "type mismatch",
            "operator constraints",
            "dependency chain",
            "structural violation",
        ]
    ):
        return "error"

    if any(
        token in text
        for token in [
            "tensor dimensions",
            "tensor shapes",
            "broadcasting",
            "indexing maps",
            "iteration space",
            "iteration bounds",
        ]
    ):
        return "high"

    if any(token in text for token in ["constants", "constant tensors"]):
        return "low"

    if any(
        token in text
        for token in [
            "element types",
            "dtypes",
            "device/layout",
            "layout attributes",
            "operator parameters",
            "transform linalg operations",
            "replace torch ops",
            "decompose",
            "fuse",
            "semantic transformation",
            "restructure computation",
            "reorganize",
            "split computation",
            "merge computation",
        ]
    ):
        return "medium"

    if any(
        token in text
        for token in [
            "rename",
            "renaming",
            "formatting",
            "normalize",
            "canonicalize",
            "canonical ordering",
            "reorder independent",
            "preserve full structure",
            "reconstruct program",
        ]
    ):
        return "low"

    if not description and any(
        token in fallback_text
        for token in ["shape", "rank", "broadcast", "indexing map", "iteration"]
    ):
        return "high"

    return "default"


def _rubric_from_rule(rule: Mapping[str, Any], mutation_id: str) -> Dict[str, Any]:
    profile_name = _profile_name_for_rule(rule, mutation_id)
    profile = STRUCTURE_RUBRIC_PROFILES.get(profile_name, DEFAULT_STRUCTURE_RUBRIC)
    structure_max = float(profile.get("structure_points_max", 2.0))
    exactness_max = 10.0 - structure_max - 3.0
    return {
        **DEFAULT_STRUCTURE_RUBRIC,
        **profile,
        "exactness_points_max": exactness_max,
        "rule_description": rule.get("description", "unknown mutation rule"),
        "rule_profile": profile_name,
    }


def _find_rule_in_json(root: Any, mutation_id: str) -> Dict[str, Any]:
    if not mutation_id:
        return {}
    for item in _iter_dicts(root):
        if item.get("id") == mutation_id:
            return {**item}
        if item.get("mutation_id") == mutation_id and "description" in item:
            return {**item}
    return {}


def _extract_mutation_context(
    root: Mapping[str, Any],
    program_pair: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    mutation_id = (
        program_pair.get("mutation_id")
        or prediction.get("mutation_id")
        or _first_value(root, ("mutation_id",))
        or ""
    )
    mutation_kind = (
        program_pair.get("mutation_kind")
        or prediction.get("mutation_kind")
        or _first_value(root, ("mutation_kind", "requirement_description"))
        or ""
    )
    pair_id = program_pair.get("pair_id") or prediction.get("pair_id") or _first_value(root, ("pair_id",))
    program_id = (
        program_pair.get("program_id")
        or prediction.get("program_id")
        or _first_value(root, ("program_id",))
    )
    source = program_pair.get("source") or prediction.get("source") or _first_value(root, ("source",))
    metadata = (
        program_pair.get("metadata")
        or prediction.get("metadata")
        or _first_value(root, ("metadata",))
        or {}
    )

    if not mutation_id:
        searchable = " ".join(
            str(value)
            for value in [
                program_pair.get("tool_used"),
                mutation_kind,
                source,
                pair_id,
                program_id,
            ]
            if value
        )
        match = re.search(r"(?<![A-Z0-9])[LT]\d_R\d(?![A-Z0-9])", searchable)
        if match:
            mutation_id = match.group(0)

    explicit_rule = _first_value(root, ("mutation_rule", "rule"))
    rule = explicit_rule if isinstance(explicit_rule, dict) else _find_rule_in_json(root, str(mutation_id))

    explicit_rubric = _first_value(root, ("dynamic_structure_rubric", "structure_rubric"))
    if isinstance(explicit_rubric, dict):
        structure_rubric = {**DEFAULT_STRUCTURE_RUBRIC, **explicit_rubric}
    elif rule:
        structure_rubric = _rubric_from_rule(rule, str(mutation_id))
    else:
        structure_rubric = {**DEFAULT_STRUCTURE_RUBRIC}

    if not mutation_kind and rule:
        mutation_kind = str(rule.get("description", "") or "")

    return {
        "mutation_id": mutation_id or "unknown",
        "mutation_kind": mutation_kind or "unknown",
        "pair_id": pair_id or "unknown",
        "program_id": program_id or "unknown",
        "source": source or "root_payload",
        "metadata": metadata if isinstance(metadata, dict) else {},
        "rule": rule if isinstance(rule, dict) else {},
        "structure_rubric": structure_rubric,
    }


def _build_structure_rubric_prompt(mutation_context: Mapping[str, Any]) -> str:
    rubric = mutation_context.get("structure_rubric", DEFAULT_STRUCTURE_RUBRIC)
    rule = mutation_context.get("rule", {})
    structure_max = float(rubric.get("structure_points_max", 2.0))
    exactness_max = float(rubric.get("exactness_points_max", 5.0))
    value_max = 3.0
    rule_lines = rule.get("rules", []) if isinstance(rule, dict) else []
    if isinstance(rule_lines, list):
        rule_text = "; ".join(str(item) for item in rule_lines)
    else:
        rule_text = _stringify(rule_lines)

    return f"""
--- Mutation Metadata and Dynamic Result-Structure Rubric ---
mutation_id: {mutation_context.get('mutation_id', 'unknown')}
mutation_kind: {mutation_context.get('mutation_kind', 'unknown')}
mutation_source: {mutation_context.get('source', 'unknown')}
mutation_metadata: {_stringify(mutation_context.get('metadata', {}))}
mutation_rule_file: {rule.get('rule_file', 'root_payload') if isinstance(rule, dict) else 'root_payload'}
mutation_rule_level: {rule.get('level', 'unknown') if isinstance(rule, dict) else 'unknown'}
mutation_rule_description: {rubric.get('rule_description', rule.get('description', 'unknown') if isinstance(rule, dict) else 'unknown')}
mutation_rule_constraints: {rule_text or 'unknown'}

For Mode A result_accuracy, use this mutation-specific structure rubric:
- shape_structure_points range: 0-{structure_max:g}
- exactness_points range: 0 or {exactness_max:g} only
- value_precision_points range: 0-{value_max:g}
- total remains 10 because {structure_max:g} + {exactness_max:g} + {value_max:g} = 10
- mutation_severity: {rubric.get('severity', 'default')}
- rubric_profile: {rubric.get('rule_profile', 'default')}
- structure_focus: {rubric.get('prompt', DEFAULT_STRUCTURE_RUBRIC['prompt'])}

When assigning partial structure credit, scale the old 0/1/2 anchors to this mutation-specific maximum:
- full structure credit = {structure_max:g}
- partial structure credit = {structure_max / 2:g}
- no structure credit = 0
""".strip()


# ---------------------------------------------------------------------------
# Prompt copied from kimi_batch_judge_agent.py
# ---------------------------------------------------------------------------

LINALG_KNOWLEDGE_PROMPT = """
=== DIALECT KNOWLEDGE: MLIR LINALG ===
Use this knowledge when judging Linalg MLIR:
- Linalg is structured tensor/memref computation IR. Many ops implement DestinationStyleOpInterface: inputs are read, outputs/init tensors or memrefs define destination/result shape and accumulator state.
- `linalg.generic` semantics are defined by `indexing_maps`, `iterator_types`, operand/result types, and the region body ending in `linalg.yield`. Equivalent-looking regions can diverge if maps, iterators, yielded values, or output shapes differ.
- `iterator_types = ["parallel", ...]` means independent elementwise iteration; `reduction` iterators accumulate across a dimension. Changing parallel/reduction roles can change semantics even when op names stay the same.
- Named ops encode common structured computations. For example, `linalg.matmul`, `linalg.batch_matmul`, `linalg.dot`, and convolution ops perform contractions and usually accumulate into the output/init operand.
- Elementwise named ops such as `linalg.add`, `linalg.mul`, `linalg.sub`, `linalg.div_*`, `linalg.exp`, `linalg.sqrt`, etc. require compatible shapes; broadcasts, casts, transposes, and reductions are explicit rather than implicit.
- `linalg.broadcast`, `linalg.transpose`, `tensor.empty`, `tensor.extract_slice`, `tensor.insert_slice`, `tensor.collapse_shape`, and `tensor.expand_shape` can change shape/layout semantics without changing the arithmetic operation.
- For semantic equivalence, check operation names, operand order, affine/indexing maps, iterator types, region arithmetic, yielded values, result tensor/memref shapes, element types, and initialization values.
""".strip()


TORCH_KNOWLEDGE_PROMPT = """
=== DIALECT KNOWLEDGE: TORCH-MLIR TORCH DIALECT ===
Use this knowledge when judging Torch MLIR:
- The Torch dialect models PyTorch programs inside MLIR. Its ops commonly mirror ATen operators such as `torch.aten.mm`, `torch.aten.matmul`, `torch.aten.add.Tensor`, `torch.aten.mul.Tensor`, `torch.aten.view`, `torch.aten.reshape`, and `torch.aten.transpose.int`.
- Torch types model PyTorch/Python concepts, including `!torch.tensor` and value-semantic `!torch.vtensor<shape,dtype>`, plus scalar/list/optional-like Torch types. Unknown shape or dtype can appear and should not be invented.
- Correctness often depends on PyTorch/ATen operator contracts: matrix multiplication rank rules, broadcasting, dtype promotion, scalar-vs-tensor overloads, reshape/view element-count preservation, transpose dimension swaps, slicing/indexing, and reduction dimensions.
- Prefer semantic reasoning from the Torch op name and operands over surface text similarity. A small change in operand order, dimension argument, list constant, dtype, or shape annotation can change the result.
- In-place or aliasing-like PyTorch behavior may be represented differently after value-semantics conversion. For MASCoD judging, treat the provided Torch MLIR as the source of truth and verify whether values flowing to `return` differ.
- When Torch programs lower toward Linalg or backend IR, shape and dtype inference are central. If a mutation changes inferred shape, dtype, broadcastability, or legality, it can cause DIVERGE or ERROR even if the high-level op family is unchanged.
- For semantic equivalence, check ATen op variants, operand order, constants/list operands, rank and shape annotations, dtype annotations, broadcasting/reduction dimensions, reshape/view sizes, data-flow through SSA values, and returned values.
""".strip()


GENERIC_MLIR_KNOWLEDGE_PROMPT = """
=== DIALECT KNOWLEDGE: GENERIC MLIR ===
Use this knowledge when the dialect is unknown:
- Judge the provided MLIR text directly. Track SSA data flow from inputs/constants through operations to returned values.
- Check operation names, operands, attributes, regions, yields, types, shapes, and return values.
- Treat execution ground truth as authoritative for hard labels, and use code-level evidence to judge whether the explanation understood why that result occurred.
""".strip()


def _select_dialect_knowledge(dialect: str) -> str:
    normalized = (dialect or "").strip().lower()
    if "linalg" in normalized:
        return LINALG_KNOWLEDGE_PROMPT
    if "torch" in normalized:
        return TORCH_KNOWLEDGE_PROMPT
    return GENERIC_MLIR_KNOWLEDGE_PROMPT


JUDGE_SYSTEM_PROMPT = """
You are the MASCoD Judge Agent — an expert MLIR compiler engineer acting as a benchmark scorer.

Benchmark alignment: La-MIR / LLM4IR (IR equivalence reasoning), CodeJudgeBench (LLM-as-a-judge protocol), G-Eval (rubric-form-filling + CoT).
Semantic reasoning alignment: CodeSense-style fine-grained code semantics. Treat this as an execution-oriented semantic reasoning task, not a text similarity or style judgment task.

You are NOT a conversational assistant. You do NOT add commentary outside the required JSON.

=== INPUTS YOU RECEIVE ===
1. Two MLIR programs (programA = original, programB = mutated) from the execution embedded-program source of truth.
2. A dialect-specific knowledge section for either Linalg or Torch MLIR.
3. Execution ground truth from the sandbox: embedded inputs, compilation statuses, actual outputs, raw outputs_diverge, normalized ground-truth label, and semantic ground-truth explanation when present.
4. The evaluation LLM's prediction: predicted outputs for Program A and Program B, its label (EQUIV/DIVERGE/ERROR), and explanation.

=== YOUR TASK ===
Score how well the evaluation LLM understood the semantic relationship between the two MLIR programs, judged strictly against the execution ground truth and MLIR code.

The hard label score is computed outside this prompt. You must score the other two dimensions:
1. result_accuracy: a 0-10 score for the predicted outputs/result behavior.
2. explanation_rating: a MASCoD-ICE 0-4 score for the explanation.

=== CODESENSE-STYLE SEMANTIC AUDIT (MANDATORY) ===
Before assigning scores, perform a fine-grained semantic audit of the MLIR pair:
1. Block/function semantics: determine the actual behavior of programA and programB from the sandbox result before judging the prediction.
2. Statement/op semantics: identify the MLIR operations most responsible for equivalence or divergence. Prioritize arithmetic ops, API/dialect calls, region bodies, yield values, and tensor/memref shape-changing ops.
3. Code properties: check control-flow, branch-like behavior, loop/iterator behavior, indexing maps, reduction vs parallel iterators, broadcasting, and data-flow through SSA values when present.
4. Reverse reasoning: when the predicted explanation infers a cause from an observed output, verify that the cause is actually supported by the MLIR text and not merely plausible.
5. Approximate semantics: if exact numeric values are unavailable or too large, judge whether the explanation correctly captures the direction/category of behavior, such as same shape but different values, changed rank, compilation failure, or unsupported op.

Do NOT score an explanation highly for matching the label alone. A correct EQUIV/DIVERGE/ERROR label with shallow or unsupported semantic reasoning must receive a low explanation_rating score.

=== ANTI-BIAS RULES (MANDATORY — CodeJudgeBench) ===
- Do NOT reward verbosity. A short, precise explanation that identifies the key semantic difference scores HIGHER than a long, vague one.
- Do NOT reward sophisticated vocabulary or formal tone. Score ONLY technical correctness.
- Do NOT let the predicted label alone determine result_accuracy or explanation_rating. Evaluate each dimension on its own merits.
- Do NOT give benefit of the doubt. If a claim is not verifiable from the MLIR code or ground truth, treat it as unsupported.
- You MUST cite specific MLIR operations, attributes, or output values from the provided data when justifying each score. Generic reasoning without concrete references should receive a low explanation_rating score.

=== SCORING RUBRIC (La-MIR dimensions + CodeJudgeBench anchors) ===

Overall score weights:
- hard_label_accuracy: 1/3, computed deterministically outside your JSON from predicted_label vs ground_truth_label.
- result_accuracy: 1/3, from your rule-guided score below.
- explanation_rating: 1/3, from your MASCoD-ICE score below.

--- Dimension 1: result_accuracy (weight: 1/3, 0-10 score) ---
Score how close the evaluation LLM's predicted outputs/result behavior are to the sandbox ground truth.

First extract result claims from BOTH `predicted_output` and `explanation`. Accept exact literals, tuple/list outputs, tensor shape/type descriptions, symbolic formulas, value patterns, natural-language descriptions, compilation-failure claims, and error claims. If `predicted_output` is missing, infer result claims from the explanation only.

You MUST use exactly one of the two calculation modes below.

Mode A: both programs compiled and ran successfully
Use this mode when compilation_status_A == "success", compilation_status_B == "success", and ground_truth_label is EQUIV or DIVERGE.

Compute:
result_accuracy = shape_structure_points + exactness_points + value_precision_points

This dimension must NOT award points merely for saying EQUIV or DIVERGE. That is already measured by hard_label_accuracy. In result_accuracy, only score the predicted concrete outputs, symbolic output formulas, output shapes/types, and output value patterns.

Normalize the sandbox result into one combined output object:
combined_truth = (programA_output, programB_output)
If each program returns a tuple of tensors/scalars, preserve the tuple arity, then flatten each returned tensor/list only for value-precision counting.

shape_structure_points: dynamic range from the "Mutation Metadata and Dynamic Result-Structure Rubric" section in the user prompt.
- Use the mutation-specific maximum given there. Default is 0-2 only if no dynamic section is provided.
- Full credit: The answer gives the correct output structure for both programs: correct number of returned values, scalar/vector/tensor category, and tensor/list shape, rank, and type when relevant. Exact literal lists imply their own shape.
- Half credit: The answer is partially correct: it gets broad structure right but misses one program's shape, one tuple component, scalar-vs-vector detail, type detail when relevant, or one dimension.
- 0.0: The answer gives no output structure, or the stated structure contradicts the sandbox output.

exactness_points: 0 or the mutation-specific exactness maximum only.
- Use the exactness maximum from the "Mutation Metadata and Dynamic Result-Structure Rubric" section. Default is 5.0 only if no dynamic section is provided.
- Full credit: The answer gives exact literal outputs OR exact symbolic formulas/patterns that uniquely determine every sandbox output value for both programs. This includes exact tuple outputs and exact formulas for every returned tensor/scalar.
- 0.0: Anything less than fully exact. If even one returned value, tensor element, tuple component, boundary case, constant, or error-free output is missing or wrong, exactness_points = 0.

value_precision_points: 0-3 total.
Compute the fraction of correctly predicted comparable output values, then multiply by 3:
value_precision_points = 3 * (# correctly predicted comparable scalar leaves / # sandbox scalar leaves)

Counting rules:
- A scalar leaf is one scalar output value, or one scalar element inside a returned tensor/list.
- Include leaves from both Program A and Program B in the denominator.
- A predicted scalar is correct if it equals the sandbox scalar. For floats, tolerate tiny formatting differences such as 1 vs 1.0, but not a different numeric value.
- If the answer gives an exact symbolic formula instead of literal values, count each sandbox leaf as correct only when the formula unambiguously produces that leaf.
- If the answer predicts only some leaves/components, count the missing leaves as incorrect.
- If the answer only gives vague categories such as "different values", "some zeros", "a min-like tensor", or "same shape", value_precision_points must be at most 1.0.
- If no comparable output values or formulas are predicted, value_precision_points = 0.

Mode A examples:
- Truth [5, 4, 3], prediction [5, 4, 2]:
  shape_structure_points = the dynamic maximum because shape [3] is correct.
  exactness_points = 0.0 because one value is wrong.
  value_precision_points = 3 * (2/3) = 2.0.
  result_accuracy = dynamic structure maximum + 0 + 2.
- Truth Program A [1, 1, 4], Program B [2, 1, 4]; prediction gives both exact lists:
  shape_structure_points = dynamic maximum, exactness_points = dynamic maximum, value_precision_points = 3.0, result_accuracy = 10/10.
- Truth Program A tensor<8xi32> values [1,1,4,1,5,5,0,0], Program B tensor<8xi32> values [2,1,4,1,0,0,0,0]; prediction says only "both return tensor<8xi32> and outputs diverge":
  shape_structure_points = dynamic maximum, exactness_points = 0.0, value_precision_points = 0.0, result_accuracy = dynamic maximum/10.
- Truth is a tuple (scalar, tensor<3xi1>, scalar). Prediction describes only the tensor component exactly and omits the two scalars:
  shape_structure_points is at most half of the dynamic maximum, exactness_points = 0.0, value_precision_points counts only the correctly predicted tensor leaves; omitted scalar leaves are incorrect.

Mode B: at least one program failed to compile or run
Use this mode when ground_truth_label is ERROR, or either compilation status is not "success".

Compute:
result_accuracy = error_prediction_points

error_prediction_points: 0 or 10 only.
- 10.0: The prediction identifies that execution results in ERROR/failure. This can be via predicted_label == ERROR, predicted_output explicitly saying ERROR/failure, or the explanation clearly saying one or both programs fail to compile/run/verify.
- 0.0: The prediction claims normal successful outputs, EQUIV, or DIVERGE without acknowledging error/failure behavior.

Do NOT grade error category, failed side, diagnostic type, failing operation, type/shape mismatch, SSA dominance, unsupported op, or stderr detail under result_accuracy. Those are explanation-quality issues and must be assessed only in explanation_rating.

Mode B examples:
- Program B fails mlir-opt because of a type mismatch; prediction says "Program B fails verification due to a result type mismatch":
  error_prediction_points = 10.0 because it predicts ERROR/failure. The type mismatch and Program B details affect explanation_rating, not result_accuracy.
- Program B fails, prediction says only "there will be a compilation error":
  error_prediction_points = 10.0 because it predicts ERROR/failure. The lack of failed-side/diagnostic detail affects explanation_rating, not result_accuracy.
- Program B fails, prediction says "Program A has a runtime error":
  error_prediction_points = 10.0 because it predicts ERROR/failure. The wrong failed side must lower explanation_rating.

Strict result_accuracy rules:
- Do not use the predicted label by itself as a substitute for predicted values. A correct EQUIV/DIVERGE label with no output/value/shape/error claims gets 0 result_accuracy unless the explanation contains scoreable output claims.
- Do not award any result_accuracy points just because the answer correctly says outputs match or diverge.
- Do not require exact numeric literals when the answer gives a precise symbolic description that is checkable against the MLIR and sandbox output.
- Do not reward fabricated exact values. Exact-looking values that contradict sandbox output score 0 for those values.
- Use the sandbox outputs and diagnostics as ground truth. The MLIR code can help interpret whether symbolic formulas match the outputs.
- Return the component arithmetic in `evidence.result_accuracy_components.calculation`.
- For ground_truth_label ERROR, result_accuracy must be exactly 10 if error/failure was predicted, otherwise exactly 0.

--- Dimension 2: explanation_rating (weight: 1/3, MASCoD-ICE 0-4 rating) ---
Rate the evaluation LLM's explanation as an MLIR code-understanding explanation.
This dimension adapts ICE-Score's aspect-specific 0-4 LLM rating protocol to MLIR semantic-equivalence explanations. You must first assess the explanation using the five MASCoD-ICE subcriteria below, then assign one final discrete explanation_rating.
When `semantic_ground_truth_explanation` is non-null, treat it as authoritative explanation ground truth alongside the sandbox outputs and diagnostics. Use it to verify or refute the predicted explanation's causal claims.

For ERROR ground truth, explanation_rating is where you must judge error quality:
- Whether it identifies the actual error category, such as compilation failure, mlir-opt verification failure, runtime failure, unsupported op, invalid type/shape, SSA dominance, or execution sandbox failure.
- Whether it identifies all and only the failing program(s), using Program A, Program B, original, mutated, both, or equivalent wording.
- Whether it cites the key diagnostic cause or failing operation/type/shape/attribute consistently with sandbox stderr/details.
- A prediction that only says "ERROR" can receive full result_accuracy in Mode B, but should receive a low explanation_rating if it lacks concrete, correct diagnostic reasoning.

MASCoD-ICE subcriteria:
1. Usefulness: Does the explanation help a reader understand whether the pair is EQUIV, DIVERGE, or ERROR?
2. Factual consistency: Are the explanation's claims consistent with the MLIR text and execution ground truth?
3. Semantic causality: Does it explain why the observed behavior happens, not just restate the label?
4. Evidence specificity: Does it cite concrete operations, attributes, shapes, outputs, or diagnostics?
5. Dialect grounding: Does it use relevant Linalg/Torch semantics from the knowledge section when those semantics matter?

MASCoD-ICE Explanation Rating (0-4):
- 4: Excellent. Satisfies all or nearly all subcriteria. The explanation is useful, factually correct, causally grounded, evidence-specific, and dialect-aware. Minor omissions are allowed only if they do not affect the core semantic judgment.
- 3: Good. Mostly satisfies the subcriteria and reaches the right semantic conclusion, but misses one important detail such as a shape/output detail, exact op attribute, or dialect-specific consequence.
- 2: Partial. Somewhat useful and broadly related to the correct behavior, but weak on causality, concrete evidence, or dialect grounding. It may identify the right category without explaining the mechanism.
- 1: Poor. Contains one small relevant observation, but major claims are unsupported, incomplete, or disconnected from the actual MLIR pair.
- 0: Useless or misleading. Empty, generic, fabricated, or contradicted by the MLIR text or execution ground truth.

=== EVALUATION PROCEDURE (G-Eval CoT ordering) ===
You MUST follow this exact sequence:

Step 1: Read both MLIR programs first.
Step 2: Read the dialect knowledge section and identify which rules are relevant to the pair.
Step 3: Read the execution ground truth (embedded inputs, outputs, compilation statuses, raw divergence flag, normalized label, and semantic ground-truth explanation). Independently determine the actual computational result. Do NOT read the LLM's prediction yet.
Step 4: Build a semantic audit: list the key MLIR ops/properties that explain the ground truth label.
Step 5: Read the LLM's predicted_label, predicted_output, and explanation. Extract result claims and score result_accuracy using the exact Mode A or Mode B arithmetic above.
Step 6: Extract structural and semantic explanation claims about ops, attributes, outputs, shapes, diagnostics, and causality.
Step 7: Score explanation_rating using the MASCoD-ICE 0-4 criteria above. Check each claim against the MLIR text, dialect knowledge, and execution ground truth. Include a brief subcriteria assessment in the JSON.
Step 8: Determine verdict:
  - CORRECT: result_accuracy >= 8 AND explanation_rating >= 3
  - PARTIALLY_CORRECT: result_accuracy >= 4 OR explanation_rating >= 2
  - INCORRECT: all other cases

=== OUTPUT FORMAT ===
Return ONLY valid JSON. Do NOT wrap in markdown code fences.

{
    "scores": {
        "result_accuracy": <number 0-10>,
        "explanation_rating": <int 0-4>
    },
    "evidence": {
        "correct_claims": ["<claim from explanation verified against code/ground truth>", ...],
        "incorrect_claims": ["<claim contradicted by code/ground truth>", ...],
        "unverifiable_claims": ["<claim that cannot be checked from provided data>", ...],
        "semantic_audit": ["<MLIR op/property/output fact used as judging evidence>", ...],
        "result_accuracy_components": {
            "mode": "<success | error>",
            "shape_structure_points": <number or null>,
            "shape_structure_max": <number or null>,
            "exactness_points": <number or null>,
            "exactness_max": <number or null>,
            "value_precision_points": <number or null>,
            "error_prediction_points": <number or null>,
            "calculation": "<brief arithmetic, e.g. shape 2 + exactness 0 + precision 2 = 4>"
        },
        "mascod_ice_subcriteria": {
            "usefulness": "<brief assessment>",
            "factual_consistency": "<brief assessment>",
            "semantic_causality": "<brief assessment>",
            "evidence_specificity": "<brief assessment>",
            "dialect_grounding": "<brief assessment>"
        }
    },
    "reasoning": "<step-by-step reasoning following the evaluation procedure above>",
    "verdict": "<CORRECT | PARTIALLY_CORRECT | INCORRECT>"
}
""".strip()


DIMENSION_WEIGHTS = {
    "result_accuracy": {"weight": 1.0 / 3.0, "max_score": 10.0},
    "explanation_rating": {"weight": 1.0 / 3.0, "max_score": 4.0},
}

HARD_LABEL_WEIGHT = 1.0 / 3.0


def _build_judge_prompt(
    program_a: str,
    program_b: str,
    dialect: str,
    mutation_context: Mapping[str, Any],
    predicted_label: str,
    predicted_output: Any,
    explanation: str,
    ground_truth_label: str,
    ground_truth: Mapping[str, Any],
) -> str:
    dialect_knowledge = _select_dialect_knowledge(dialect)
    structure_rubric_prompt = _build_structure_rubric_prompt(mutation_context)
    return f"""dialect: {dialect}
pair_id: {mutation_context.get('pair_id', 'unknown')}
execution_program_id: {ground_truth.get('program_id', mutation_context.get('program_id', 'unknown'))}

--- Programs ---
programA:
{program_a}

programB:
{program_b}

--- Knowledge Base ---
{dialect_knowledge}

{structure_rubric_prompt}

--- Execution Ground Truth ---
embedded_inputs: {_stringify(ground_truth.get('embedded_inputs'))}
compilation_status_A: {ground_truth.get('compilation_status_A')}
compilation_status_B: {ground_truth.get('compilation_status_B')}
programA_output: {ground_truth.get('programA_output')}
programB_output: {ground_truth.get('programB_output')}
outputs_diverge: {ground_truth.get('outputs_diverge')}
divergence_details: {ground_truth.get('divergence_details')}
ground_truth_label: {ground_truth_label}
semantic_ground_truth_explanation: {_stringify(ground_truth.get('semantic_explanation'))}

--- Evaluation LLM Prediction ---
predicted_label: {predicted_label}
predicted_output: {_stringify(predicted_output)}
explanation: {explanation}

Score the evaluation LLM's result prediction and explanation following the rubric and return JSON only."""


def _strip_json_markdown(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    return raw


def _invoke_llm_judge(
    program_a: str,
    program_b: str,
    dialect: str,
    mutation_context: Mapping[str, Any],
    predicted_label: str,
    predicted_output: Any,
    explanation: str,
    ground_truth_label: str,
    ground_truth: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    user_content = _build_judge_prompt(
        program_a,
        program_b,
        dialect,
        mutation_context,
        predicted_label,
        predicted_output,
        explanation,
        ground_truth_label,
        ground_truth,
    )

    response = judge_llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    raw = _strip_json_markdown(str(response.content))

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Judge LLM returned unparseable JSON: %s", raw[:500])
        return None


def _build_fallback_judge_result(reason: str) -> Dict[str, Any]:
    return {
        "scores": {"result_accuracy": 0, "explanation_rating": 0},
        "evidence": {
            "correct_claims": [],
            "incorrect_claims": [],
            "unverifiable_claims": [],
            "semantic_audit": [],
            "result_accuracy_components": {
                "mode": "not assessed",
                "shape_structure_points": None,
                "shape_structure_max": None,
                "exactness_points": None,
                "exactness_max": None,
                "value_precision_points": None,
                "error_prediction_points": None,
                "calculation": "not assessed",
            },
            "mascod_ice_subcriteria": {
                "usefulness": "not assessed",
                "factual_consistency": "not assessed",
                "semantic_causality": "not assessed",
                "evidence_specificity": "not assessed",
                "dialect_grounding": "not assessed",
            },
        },
        "reasoning": reason,
        "verdict": "INCORRECT",
    }


def _clamp_score(value: Any, low: float, high: float) -> float:
    if not isinstance(value, (int, float)):
        return low
    return max(low, min(float(value), high))


def _score_record_from_details(details: Mapping[str, Any]) -> Dict[str, Any]:
    scores = details.get("llm_judge_scores", {})
    return {
        "program_id": details.get("execution_program_id"),
        "pair_id": details.get("pair_id"),
        "evaluated_model": (details.get("evaluated_model") or {}).get("model_key"),
        "ground_truth_label": details.get("ground_truth_label"),
        "predicted_label": details.get("predicted_label"),
        "subscores": {
            "hard_label_accuracy": 1.0 if details.get("label_match") else 0.0,
            "result_accuracy": scores.get("result_accuracy"),
            "explanation_rating": scores.get("explanation_rating"),
        },
        "weighted_scores": {
            "hard_label": details.get("hard_label_weighted"),
            "result_accuracy": details.get("result_accuracy_weighted"),
            "explanation_rating": details.get("explanation_rating_weighted"),
        },
        "total_score": details.get("final_score"),
        "verdict": details.get("verdict"),
    }


def _error_score_record(
    state: Mapping[str, Any],
    program_pair: Mapping[str, Any],
    prediction: Mapping[str, Any],
    error_msg: str,
) -> Dict[str, Any]:
    return {
        "program_id": program_pair.get("program_id") or prediction.get("program_id"),
        "pair_id": program_pair.get("pair_id") or prediction.get("pair_id"),
        "evaluated_model": prediction.get("model_key") or state.get("eval_model_name"),
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main LangGraph node
# ---------------------------------------------------------------------------

def judge_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Judge one program pair from the JSON already present in LangGraph state.

    The root agent should pass one large JSON object in root_payload/judge_input
    (or as the state itself in tests). This node extracts only from that JSON.
    """
    logs = list(state.get("logs", []))
    logs.append("Judge Agent: Starting single-pair benchmark scoring")

    root = _normalize_root_payload(state)
    program_pair = _extract_program_pair(root)
    ground_truth = _extract_execution_ground_truth(root, program_pair)
    prediction = _extract_prediction(root)
    eval_model_name = (
        state.get("eval_model_name")
        or prediction.get("model_key")
        or prediction.get("llm_name")
        or "unknown"
    )

    missing = _validate_required_inputs(program_pair, ground_truth, prediction)
    if missing:
        error_msg = f"Judge Agent: Missing required input fields: {', '.join(missing)}"
        logs.append(error_msg)
        judge_payload = _error_score_record(state, program_pair, prediction, error_msg)
        return {**state, "judge_payload": judge_payload, "logs": logs}

    program_a = str(program_pair.get("programA", ""))
    program_b = str(program_pair.get("programB", ""))
    dialect = str(program_pair.get("dialect", "unknown"))
    predicted_label = _normalize_label(prediction.get("predicted_label", prediction.get("label")))
    predicted_output = prediction.get("predicted_output")
    explanation = str(prediction.get("explanation", "") or "")
    confidence = prediction.get("confidence", 0.0)

    ground_truth_label = _derive_ground_truth_label(ground_truth)
    objective = _compute_objective_score(predicted_label, ground_truth_label)
    mutation_context = _extract_mutation_context(root, program_pair, prediction)
    rubric = mutation_context.get("structure_rubric", DEFAULT_STRUCTURE_RUBRIC)

    logs.append(
        "Judge Agent: "
        f"ground_truth={ground_truth_label}, predicted={predicted_label}, "
        f"mutation={mutation_context.get('mutation_id')}, "
        f"structure_max={rubric.get('structure_points_max', 2.0)}"
    )

    try:
        llm_judge_result = _invoke_llm_judge(
            program_a,
            program_b,
            dialect,
            mutation_context,
            predicted_label,
            predicted_output,
            explanation,
            ground_truth_label,
            ground_truth,
        )
    except Exception as exc:
        logs.append(f"Judge Agent: LLM judge request failed - {exc}")
        llm_judge_result = None

    if llm_judge_result is None:
        llm_judge_result = _build_fallback_judge_result(
            "Judge LLM failed to return valid JSON or content; result and explanation scores set to 0."
        )

    raw_scores = llm_judge_result.get("scores", {})
    result_accuracy = _clamp_score(raw_scores.get("result_accuracy", 0), 0.0, 10.0)

    if ground_truth_label == "ERROR":
        error_predicted = _prediction_claims_error(
            predicted_label,
            predicted_output,
            explanation,
        )
        result_accuracy = 10.0 if error_predicted else 0.0
        llm_judge_result.setdefault("scores", {})["result_accuracy"] = result_accuracy
        llm_judge_result.setdefault("evidence", {})["result_accuracy_components"] = {
            "mode": "error",
            "shape_structure_points": None,
            "shape_structure_max": None,
            "exactness_points": None,
            "exactness_max": None,
            "value_precision_points": None,
            "error_prediction_points": result_accuracy,
            "calculation": (
                "ground_truth_label is ERROR and prediction "
                f"{'claims' if error_predicted else 'does not claim'} ERROR/failure; "
                f"result_accuracy = {result_accuracy:g}"
            ),
        }

    explanation_rating = _clamp_score(raw_scores.get("explanation_rating", 0), 0.0, 4.0)
    scores = {
        "result_accuracy": result_accuracy,
        "explanation_rating": explanation_rating,
    }

    objective_score = objective["correctness"] * 100.0
    hard_label_weighted = HARD_LABEL_WEIGHT * objective_score
    result_weighted = (
        result_accuracy / DIMENSION_WEIGHTS["result_accuracy"]["max_score"]
    ) * DIMENSION_WEIGHTS["result_accuracy"]["weight"] * 100.0
    explanation_weighted = (
        explanation_rating / DIMENSION_WEIGHTS["explanation_rating"]["max_score"]
    ) * DIMENSION_WEIGHTS["explanation_rating"]["weight"] * 100.0
    final_score = hard_label_weighted + result_weighted + explanation_weighted

    if objective["label_match"] and result_accuracy >= 8 and explanation_rating >= 3:
        verdict = "CORRECT"
    elif final_score >= 50 or objective["label_match"] or result_accuracy >= 4 or explanation_rating >= 2:
        verdict = "PARTIALLY_CORRECT"
    else:
        verdict = "INCORRECT"

    details = {
        "dialect": dialect,
        "tool_used": program_pair.get("tool_used", "root_payload"),
        "mutation_id": mutation_context.get("mutation_id"),
        "mutation_kind": mutation_context.get("mutation_kind"),
        "pair_id": mutation_context.get("pair_id"),
        "execution_program_id": ground_truth.get("program_id", mutation_context.get("program_id")),
        "mutation_rule": mutation_context.get("rule", {}),
        "dynamic_structure_rubric": mutation_context.get("structure_rubric", {}),
        "evaluated_model": {
            "model_key": prediction.get("model_key", eval_model_name),
            "model_id": prediction.get("model_id"),
            "llm_name": prediction.get("llm_name"),
            "provider": prediction.get("provider"),
        },
        "judge_model": {"model": OPENAI_MODEL},
        "ground_truth_label": ground_truth_label,
        "predicted_label": predicted_label,
        "label_match": objective["label_match"],
        "confidence": confidence,
        "explanation": explanation,
        "predicted_output": predicted_output,
        "objective_score": objective_score,
        "hard_label_weighted": round(hard_label_weighted, 2),
        "llm_judge_scores": scores,
        "result_accuracy_weighted": round(result_weighted, 2),
        "explanation_rating_weighted": round(explanation_weighted, 2),
        "llm_judge_weighted": round(result_weighted + explanation_weighted, 2),
        "llm_judge_evidence": llm_judge_result.get("evidence", {}),
        "llm_judge_reasoning": llm_judge_result.get("reasoning", ""),
        "final_score": round(final_score, 2),
        "verdict": verdict,
        "scoring_weights": {
            "hard_label_accuracy": HARD_LABEL_WEIGHT,
            "result_accuracy": DIMENSION_WEIGHTS["result_accuracy"]["weight"],
            "explanation_rating": DIMENSION_WEIGHTS["explanation_rating"]["weight"],
        },
        "execution_details": {
            "embedded_inputs": ground_truth.get("embedded_inputs"),
            "compilation_status_A": ground_truth.get("compilation_status_A"),
            "compilation_status_B": ground_truth.get("compilation_status_B"),
            "programA_output": ground_truth.get("programA_output"),
            "programB_output": ground_truth.get("programB_output"),
            "outputs_diverge": ground_truth.get("outputs_diverge"),
            "divergence_details": ground_truth.get("divergence_details"),
            "semantic_explanation": ground_truth.get("semantic_explanation"),
        },
    }

    judge_payload = _score_record_from_details(details)
    logs.append(
        f"Judge Agent: Final score={final_score:.1f}/100, verdict={verdict}"
    )

    return {
        **state,
        "judge_payload": judge_payload,
        "judge_details": details,
        "logs": logs,
    }
