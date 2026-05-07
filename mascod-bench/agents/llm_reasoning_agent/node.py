from dotenv import load_dotenv
load_dotenv()

import json
import time
from langchain_core.messages import SystemMessage, HumanMessage

# Hybrid registry — every model (open-source cloud + closed-source) is defined
# in nodes/model_registry.py. The legacy "openai|gemini|claude" aliases still work.
try:
    # When run as a package (python -m mascod_graph.main, batch runner, etc.)
    from .model_registry import get_model_client, get_spec, ModelSpec
except ImportError:
    # When run as a script from inside mascod_graph/ (python reasoning_agent.py ...)
    from model_registry import get_model_client, get_spec, ModelSpec  # type: ignore


def get_llm(provider: str):
    """
    Backwards-compatible shim.
    `provider` may be a legacy alias ("openai" | "gemini" | "claude") or any
    registry key (e.g. "qwen3-coder-480b", "glm-4.6", "gpt-oss-120b").
    """
    return get_model_client(provider)

def _extract_response_text(response) -> str:
    """
    Pull a single text string out of a LangChain chat response.

    LangChain ChatModel results vary across providers and across reasoning
    vs. non-reasoning models:
      - .content may be str (most common)
      - .content may be a list of dicts/strs (Anthropic-style content blocks)
      - .content may be empty when the model wrote everything to a non-standard
        field like additional_kwargs['reasoning'] or
        additional_kwargs['reasoning_content'] (some OpenRouter reasoning models)

    We try them in order and return the first non-empty hit. If everything is
    empty, returns ''. Never raises.
    """
    content = getattr(response, "content", "")

    # Common case: list of content blocks → flatten.
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Standard shapes: {"type":"text","text":"..."} or {"text":"..."}
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
        content = "".join(parts)

    if isinstance(content, str) and content.strip():
        return content

    # Fallback: some reasoning models stash output in additional_kwargs.
    extra = getattr(response, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = extra.get(key)
        if isinstance(v, str) and v.strip():
            return v

    return content if isinstance(content, str) else ""


def parse_json_response(content: str | None):
    """
    Robust JSON parser for chat model output.

    Returns (parsed_obj_or_None, raw_head, error_msg_or_None).

    Handles:
      - None / empty content
      - markdown fences (```json ... ```, ``` ... ```)
      - leading/trailing prose around a JSON object
      - extracts the first top-level {...} via balanced-brace scan
        (string-aware: ignores braces inside JSON strings, including escapes)

    Never raises. Callers branch on whether parsed is None.
    """
    raw_head = (content or "")[:500]

    if content is None:
        return None, raw_head, "empty_content (None)"

    text = content.strip()
    if not text:
        return None, raw_head, "empty_content"

    # Strip markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()

        # Remove opening fence line
        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        # Remove closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Fast path — whole thing is JSON.
    try:
        return json.loads(text), raw_head, None
    except json.JSONDecodeError:
        pass

    # Slow path — try every top-level balanced {...} block in order.
    # Stops at the first one that parses as a JSON OBJECT (dict).
    last_decode_err: str | None = None
    for candidate in _iter_top_level_json_objects(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_decode_err = str(e)

            # Final fallback: some models emit literal control characters
            # (raw newlines / tabs) inside JSON string values, which is
            # forbidden by RFC 8259. Sanitize and retry.
            sanitized = _strip_unescaped_control_chars(candidate)

            if sanitized != candidate:
                try:
                    obj = json.loads(sanitized)
                    if isinstance(obj, dict):
                        return obj, raw_head, None
                except json.JSONDecodeError:
                    pass

            continue

        if isinstance(obj, dict):
            return obj, raw_head, None

    if last_decode_err is None:
        return None, raw_head, "no_json_object_found"

    return None, raw_head, f"json_decode_error: {last_decode_err}"


def _strip_unescaped_control_chars(s: str) -> str:
    """
    Replace bare \\n, \\r, \\t INSIDE JSON string literals with their escaped
    forms. JSON spec forbids unescaped controls in strings; some models emit
    them. Conservative implementation: only touches characters between
    matched quotes, not other JSON whitespace.
    """
    out = []
    in_string = False
    escape = False
    for ch in s:
        if in_string:
            if escape:
                escape = False
                out.append(ch)
            elif ch == "\\":
                escape = True
                out.append(ch)
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _extract_first_json_object(text: str) -> str | None:
    """Backwards-compatible single-shot extractor — returns the FIRST balanced
    {...} block. Most callers should use _iter_top_level_json_objects instead,
    which is forgiving toward prose containing stray braces."""
    for c in _iter_top_level_json_objects(text):
        return c
    return None


def _iter_top_level_json_objects(text: str):
    """
    Yield every top-level balanced { ... } block in `text`, left-to-right,
    ignoring braces inside string literals (with escape handling).

    Top-level means depth-0 in the surrounding text, not inside another object.
    Allows the parser to skip a prose-stray "{bracket}" and still find the
    real JSON object that follows.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Found a candidate opener. Scan until balanced.
        depth = 0
        in_string = False
        escape = False
        start = i
        for j in range(start, n):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:j + 1]
                    i = j + 1
                    break
        else:
            # Reached end of string without closing — no more candidates.
            return


# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert in MLIR compiler infrastructure with deep specialization in the linalg dialect,
affine maps, tensor/memref type systems, and SSA-based intermediate representations.
You have extensive experience in static program analysis, symbolic execution, and semantic
equivalence verification of compiler IR programs.

Your role in the MASCoD (Multi-Agent Semantic Code Divergence) benchmark is to serve as the
authoritative Reasoning Agent. You determine whether two MLIR programs are semantically
equivalent, divergent, or contain structural errors — using only static reasoning over the
visible source. You do not compile, execute, or simulate.

═══════════════════════════════════════════════════════
PHASE 1 — STATIC PROGRAM ANALYSIS (do this internally)
═══════════════════════════════════════════════════════

For EACH program independently:

1. SIGNATURE ANALYSIS
   - Parse the func.func signature
   - Extract argument names, shapes (static/dynamic dims), element types, and encoding attributes
   - Identify return types and their shapes

2. SYMBOLIC SSA TRACE
   - Walk each SSA definition in order
   - For tensor operations: track shape transformations (collapse, expand, broadcast, transpose)
   - For linalg generics: extract the indexing maps, iterator types (parallel/reduction), and region body semantics
   - For arith ops inside regions: evaluate what computation is expressed (multiply, add, max, etc.)
   - Record the symbolic value each %result represents

3. OUTPUT CHARACTERIZATION
   - Describe the shape and semantic content of each returned value
   - Use notation like: "rank-2 float32 tensor [M×N], each element = sum of row-wise products of A[i,k] and B[k,j]"
   - Flag any ambiguity caused by dynamic dimensions

═══════════════════════════════════════════════════════
PHASE 2 — SEMANTIC COMPARISON
═══════════════════════════════════════════════════════

Compare the characterized outputs of programA and programB:

EQUIV   — The programs produce identical results for ALL valid inputs under the declared type constraints.
          This includes: reordered operations with same net effect, algebraically equivalent expressions,
          loops with same iteration semantics but different tile sizes, SSA value renaming.

DIVERGE — There exists at least one valid input for which the outputs differ.
          This includes: transposed index maps, swapped operands in non-commutative ops,
          different reduction axes (sum vs. max, AND vs. OR — note: linalg.generic with arith.andi over
          i1 is ALL-true, not ANY-true; do not confuse the two), off-by-one bounds, incorrect broadcasting.

ERROR   — One or both programs are STRUCTURALLY INVALID MLIR — the defect prevents output prediction:
          invalid SSA use-before-def, type mismatches, malformed affine maps, undefined operations,
          missing required attributes, dialect violations.
          DO NOT use ERROR to mean "I am unsure" or "I could not parse" — only use it when you can
          identify a specific structural defect in the source. When uncertain, pick EQUIV or DIVERGE
          based on best-effort reasoning and lower the confidence accordingly.

═══════════════════════════════════════════════════════
CALIBRATION GUIDELINES
═══════════════════════════════════════════════════════

- Confidence is a strict probability in the OPEN interval [0.0, 1.0). NEVER output 1.0 — cap at 0.99.
- Assign confidence 0.85-0.99 only when the symbolic trace is fully deterministic and unambiguous.
- Assign confidence 0.65-0.84 when dynamic shapes or complex affine maps introduce uncertainty.
- Assign confidence 0.40-0.64 when the distinction requires assumptions about runtime values.
- Assign confidence below 0.40 only if genuinely undecidable from static analysis alone.
- For DIVERGE at high confidence (>0.80) you MUST provide a concrete witness input in the 'witness' field.
- The label "UNCERTAIN" is NOT allowed; choose EQUIV / DIVERGE / ERROR and lower the confidence instead.

═══════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON ONLY
═══════════════════════════════════════════════════════

Return ONLY valid JSON. No markdown fences. No preamble. No trailing commentary.
Output MUST begin with '{' and end with '}'.

For Gemini compatibility:
- Keep predicted_output_A under 300 characters.
- Keep predicted_output_B under 300 characters.
- Keep explanation under 2 sentences.
- Keep key_differences to at most 3 short strings.
- Do not use markdown fences.
- Do not include long tensor-by-tensor derivations.
- Return compact JSON only.

{
    "predicted_output_A": "<symbolic description of programA's returned values — concise, type+shape+meaning>",
    "predicted_output_B": "<symbolic description of programB's returned values — concise, type+shape+meaning>",
    "label": "EQUIV | DIVERGE | ERROR",
    "confidence": <float in [0.0, 0.99]; never 1.0>,
    "key_differences": [
        "<bullet listing each specific SSA-value, op, indexing_map, type, shape, or return-tuple difference>",
        "<empty list [] if EQUIV>"
    ],
    "witness": "<concrete input value that triggers divergence (DIVERGE only); null otherwise>",
    "explanation": "<See EXPLANATION REQUIREMENTS below>"
}

predicted_output_A / predicted_output_B must be SYMBOLIC summaries of what each program returns
(e.g., "tensor<3xf32> = row-wise sum of arg0 where arg0[i,j] > 0"), NOT fabricated concrete numeric
arrays. Do not hallucinate sample values.

If classification is genuinely impossible (e.g., the input is not MLIR at all), return EXACTLY:
{
    "error": "<description of what prevented classification>"
}

═══════════════════════════════════════════════════════
EXPLANATION REQUIREMENTS — MUST FOLLOW ALL RULES
═══════════════════════════════════════════════════════

The 'explanation' field MUST satisfy ALL of the following rules:

1. START WITH THE VERDICT:
   Begin with a one-sentence verdict that names the label and its root cause.
   Example: "Classified as DIVERGE because the second indexing map differs between the two programs."
   Example: "Classified as EQUIV because both linalg.generic bodies apply identical indexing maps and arithmetic."
   Example: "Classified as ERROR because programB contains an out-of-bounds affine map for its declared tensor shape."

2. CITE SPECIFIC OPERATIONS BY NAME:
   Reference the exact SSA values, op names, affine maps, or iterator types that drove your decision.
   Do NOT write vague summaries — name the line-level construct.

   For DIVERGE:
     State the exact op or map that differs and explain the semantic consequence. Example:
     "In programA, the second indexing map is affine_map<(i) -> (i+1)>, which reads input[i+1].
      In programB, it is affine_map<(i) -> (i+2)>, which reads input[i+2]. These access different
      elements of the input tensor, so %b receives a different value in each program's linalg.generic
      region body, causing the arith.select and linalg.yield to produce different outputs."

   For EQUIV:
     Name the operations that are structurally or algebraically identical and explain why any
     surface-level differences (e.g., reordered ops, renamed SSA values) do not affect the result.

   For ERROR:
     Name the exact malformed construct (e.g., "affine_map<(i) -> (i+10)> accesses index 10–19
     on a tensor<10xi32>, which is out of bounds") and which program contains it.

3. PROVIDE A CONCRETE WITNESS (DIVERGE only):
   Give a specific numeric input example that proves divergence. Trace through BOTH programs
   for that input to show the outputs differ. Example:
   "For input tensor [5, 1, 3, 2, 4, ...], at loop index i=0:
    - programA: %a=input[0]=5, %b=input[1]=1 → %cond=(5>1)=true → %min=1 → output[0]=1
    - programB: %a=input[0]=5, %b=input[2]=3 → %cond=(5>3)=true → %min=3 → output[0]=3
    output[0] differs (1 vs 3), confirming divergence."

4. DO NOT just restate the predicted outputs:
   The explanation must reason about WHY the programs behave the way they do by tracing back
   to specific source constructs — not just restate that output A and output B are different.
"""


# ──────────────────────────────────────────────
# CORE EVALUATION FUNCTION (model-agnostic)
# ──────────────────────────────────────────────

ALLOWED_LABELS = {"EQUIV", "DIVERGE", "ERROR"}


def _build_user_prompt(
    program_a: str,
    program_b: str,
    dialect: str,
    embedded_inputs: dict | None = None,
) -> str:
    parts = [
        f"dialect: {dialect}",
        "",
        f"programA:\n{program_a}",
        "",
        f"programB:\n{program_b}",
    ]
    if embedded_inputs:
        # Concrete inputs from the execution agent. Telling the model to USE
        # them grounds predicted_output_A / predicted_output_B in actual values
        # so they can be compared against the execution agent's ground truth.
        parts.extend([
            "",
            "CONCRETE INPUT VALUES (use these to ground your output predictions):",
            json.dumps(embedded_inputs, indent=2),
            "",
            "When predicting outputs, evaluate the programs symbolically over the "
            "concrete inputs above. Your `predicted_output_A` and `predicted_output_B` "
            "should describe what the programs actually return for THESE specific inputs "
            "(include concrete values where you can derive them, e.g. \"tensor<3xf32> "
            "= [12.0, -1.0, 0.0]\"). Still classify EQUIV / DIVERGE / ERROR based on "
            "whether the two programs produce the same output for these inputs.",
        ])
    parts.extend([
        "",
        "Return compact JSON only. Do not use markdown fences. Keep predicted_output_A and predicted_output_B under 300 characters each. Keep explanation under 2 sentences.",
    ])
    return "\n".join(parts)


CONFIDENCE_CLAMP = 0.99  # >=1.0 is clamped to this; bare ints/floats only


def _base_record(spec, dialect: str) -> dict:
    """Skeleton row — every exit path mutates and returns this shape."""
    return {
        "provider":            spec.provider_label,
        "llm_name":            spec.llm_name,
        "model_key":           spec.key,
        "model_id":            spec.model_id,
        "dialect":             dialect,
        "status":              "MODEL_FAILURE",   # overwritten before return
        "predicted_output_A":  None,
        "predicted_output_B":  None,
        "predicted_label":     None,
        "confidence":          0.0,
        "original_confidence": None,
        "key_differences":     [],
        "witness":             None,
        "explanation":         None,
        "latency_ms":          0,
        "attempts":            1,
        "error":               None,
        "raw_response_head":   None,
    }


_FORCE_VERDICT_RIDER = (
    "\n\nIMPORTANT: Abstention is forbidden. You MUST choose exactly one of "
    "EQUIV, DIVERGE, or ERROR. If genuinely uncertain, pick the most likely "
    "label and lower the `confidence` field accordingly. Do NOT return an "
    "{\"error\": ...} payload. Do NOT use any label other than EQUIV / "
    "DIVERGE / ERROR. Return only the strict JSON object specified."
)


def reason_about_pair(
    program_a: str,
    program_b: str,
    dialect: str,
    model: str,
    max_retries: int = 0,
    embedded_inputs: dict | None = None,
) -> dict:
    """
    Evaluate ONE program pair with ONE model. Pure function. No file writes.

    `max_retries`: If > 0, on a non-SUCCESS response we retry up to this many
    times. The FINAL retry appends an "abstention forbidden" rider to the
    prompt so MODEL_FAILUREs convert to forced verdicts.

    The returned row records `attempts: N` (1-indexed) so callers can audit
    which pairs needed multiple tries.

    Returns a row with explicit `status`:
        SUCCESS       — model produced valid JSON + valid semantic label
        API_ERROR     — OpenRouter / network / client init failure
        PARSE_ERROR   — empty content or JSON could not be extracted
        SCHEMA_ERROR  — JSON parsed but missing/invalid required fields
        MODEL_FAILURE — model returned an explicit {"error": "..."} payload

    On non-SUCCESS: predicted_label / predicted_outputs / explanation are null,
    confidence = 0.0, error explains the failure, raw_response_head is populated
    when there was any response at all (for debugging).
    """
    spec = get_spec(model)
    last_rec = None
    total_attempts = max(1, max_retries + 1)

    for attempt in range(1, total_attempts + 1):
        # Apply the no-abstention rider on the FINAL attempt only, so the
        # earlier attempts capture the model's natural behavior.
        force = (attempt == total_attempts and total_attempts > 1)
        rec = _evaluate_once(program_a, program_b, dialect, model,
                             force_verdict=force, embedded_inputs=embedded_inputs)
        rec["attempts"] = attempt
        last_rec = rec
        if rec["status"] == "SUCCESS":
            return rec
    return last_rec  # exhausted retries — return the most recent non-SUCCESS


def _evaluate_once(
    program_a: str,
    program_b: str,
    dialect: str,
    model: str,
    force_verdict: bool = False,
    embedded_inputs: dict | None = None,
) -> dict:
    """One reasoning attempt. reason_about_pair() wraps this with retry logic."""
    spec = get_spec(model)
    rec = _base_record(spec, dialect)

    # ── Build client (may fail if API key missing or langchain integration absent) ──
    try:
        llm = get_model_client(model)
    except Exception as e:
        rec.update(status="API_ERROR", error=f"client_init_failed: {e}")
        return rec

    # ── Invoke model ──
    user_prompt = _build_user_prompt(program_a, program_b, dialect, embedded_inputs)

    # if "gemini" in model.lower():
    #     user_prompt += """

    # IMPORTANT FOR GEMINI:
    # - Return ultra-compact JSON only.
    # - No markdown fences.
    # - No prose outside JSON.
    # - Keep predicted_output_A under 120 characters.
    # - Keep predicted_output_B under 120 characters.
    # - Keep explanation under 1 sentence.
    # - Round floats to at most 4 decimal places.
    # """

    

    if force_verdict:
        user_prompt = user_prompt + _FORCE_VERDICT_RIDER
    t0 = time.perf_counter()
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        rec.update(
            status="API_ERROR",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=str(e),
        )
        return rec
    rec["latency_ms"] = int((time.perf_counter() - t0) * 1000)

    # ── Pull text out (handles content lists, reasoning fields) ──
    raw_content = _extract_response_text(response)
    rec["raw_response_head"] = (raw_content or "")[:500] or None

    # ── Parse JSON ──
    parsed, raw_head, parse_err = parse_json_response(raw_content)
    if parsed is None:
        rec.update(
            status="PARSE_ERROR",
            error=parse_err or "unknown_parse_error",
            raw_response_head=raw_head or rec["raw_response_head"],
        )
        return rec

    # ── Model returned an explicit error payload ──
    if isinstance(parsed, dict) and "error" in parsed and "label" not in parsed:
        rec.update(
            status="MODEL_FAILURE",
            error=str(parsed.get("error")),
        )
        return rec

    # ── Schema validation ──
    label = parsed.get("label")
    if label not in ALLOWED_LABELS:
        rec.update(status="SCHEMA_ERROR",
                   error=f"invalid_label: {label!r} (allowed: {sorted(ALLOWED_LABELS)})")
        return rec

    raw_conf = parsed.get("confidence")
    if not isinstance(raw_conf, (int, float)) or isinstance(raw_conf, bool):
        rec.update(status="SCHEMA_ERROR",
                   error=f"invalid_confidence_type: {raw_conf!r}")
        return rec
    if raw_conf < 0.0:
        rec.update(status="SCHEMA_ERROR",
                   error=f"confidence_out_of_range: {raw_conf}")
        return rec
    if raw_conf >= 1.0:
        confidence = CONFIDENCE_CLAMP
        original_confidence = float(raw_conf)
    else:
        confidence = float(raw_conf)
        original_confidence = None

    explanation = parsed.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        rec.update(status="SCHEMA_ERROR", error="missing_or_empty_explanation")
        return rec

    pred_a = parsed.get("predicted_output_A")
    pred_b = parsed.get("predicted_output_B")

    # When input grounding is used, models often emit predicted outputs as
    # actual JSON arrays (e.g. [true, 12.0, [true, true]]) instead of as a
    # string. Accept either shape and normalize to a JSON-string so downstream
    # consumers always see strings, matching the original symbolic-mode
    # behavior. None and primitives also get coerced.
    def _coerce(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(v)

    pred_a = _coerce(pred_a)
    pred_b = _coerce(pred_b)

    if not isinstance(pred_a, str) or not isinstance(pred_b, str):
        rec.update(status="SCHEMA_ERROR",
                   error="missing_or_non_string_predicted_outputs")
        return rec

    # Optional fields — accept if well-typed, else default.
    key_diffs = parsed.get("key_differences", [])
    if not (isinstance(key_diffs, list) and all(isinstance(s, str) for s in key_diffs)):
        key_diffs = []
    witness = parsed.get("witness")
    if witness is not None and not isinstance(witness, str):
        witness = None

    rec.update(
        status="SUCCESS",
        predicted_output_A=pred_a,
        predicted_output_B=pred_b,
        predicted_label=label,
        confidence=confidence,
        original_confidence=original_confidence,
        key_differences=key_diffs,
        witness=witness,
        explanation=explanation,
        error=None,
    )
    return rec


# ──────────────────────────────────────────────
# INTEGRATION ENTRY POINT — JSON object in, JSON object out
# ──────────────────────────────────────────────
#
# This is the function the integrated MASCoD pipeline calls. It is pure:
# no file I/O, no global state, no side effects. The Mutation Agent and the
# Execution Agent each produce JSON objects; this function consumes them as
# one combined object and returns one combined JSON object that the Judge
# Agent reads downstream.
#
#     mutation agent  ─┐
#                      ├──>  reasoning_agent.run({...})  ──>  {Agents: {LLM_Reasoning_Agent: {...}}}
#     execution agent ─┘                                         ↓
#                                                             judge agent
#
# Reproducibility note: every external call goes through OpenRouter (5 OSS
# models) or the model's native API (3 closed-source). The 8 model keys are
# defined in `nodes/model_registry.py`; see REASONING_MODELS below for the
# canonical fan-out order.

# Canonical fan-out: 5 open-source + 3 closed-source = 8 models, one per lab.
REASONING_MODELS: list[str] = [
    "qwen3-coder-next",
    "llama-4",
    "glm-5.1",
    "kimi-k2.6",
    "gpt-oss-120b",
    "gpt-4.1-mini",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]


# Pair-level metadata fields copied from the input `pair` onto every per-model
# row. Matches the existing `runs.jsonl` schema produced by batch_reasoning_agent
# so downstream consumers (Judge Agent, summarize.py, leaderboard.py) keep
# working unchanged.
_PAIR_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "dialect",
)


def _validate_input(payload: dict) -> tuple[dict, dict]:
    """Validate the integration payload and return (pair, embedded_inputs).

    Raises ValueError with an actionable message on malformed input.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"reasoning_agent.run() expected a JSON object (dict), "
            f"got {type(payload).__name__}"
        )

    for required in ("programA", "programB", "dialect"):
        if not payload.get(required):
            raise ValueError(
                f"reasoning_agent.run(): payload is missing required field '{required}'."
            )

    if payload["dialect"] not in ("linalg", "torch"):
        raise ValueError(
            f"reasoning_agent.run(): payload['dialect'] must be "
            f"'linalg' or 'torch', got {payload['dialect']!r}."
        )

    # Build a pair dict compatible with _row_for_model.
    pair = {
        "programA":       payload["programA"],
        "programB":       payload["programB"],
        "dialect":        payload["dialect"],
    }

    embedded_inputs = payload.get("embedded_inputs") or {}
    if not isinstance(embedded_inputs, dict):
        raise ValueError(
            f"reasoning_agent.run(): payload['embedded_inputs'] must be a JSON "
            f"object (dict), got {type(embedded_inputs).__name__}"
        )

    return pair, embedded_inputs


def _row_for_model(
    pair: dict,
    embedded_inputs: dict,
    model_key: str,
) -> dict:
    """Run ONE model on the pair and return one row dict matching the existing
    runs.jsonl schema. Pure function. Retry-once-then-ERROR on failure.

    On any non-SUCCESS status, `predicted_label` is forced to "ERROR" so the
    Judge Agent never has to handle a null label. The original `status` is
    preserved on the row (API_ERROR / PARSE_ERROR / SCHEMA_ERROR / MODEL_FAILURE)
    for downstream debugging.
    """
    record = reason_about_pair(
        program_a=pair["programA"],
        program_b=pair["programB"],
        dialect=pair["dialect"],
        model=model_key,
        max_retries=1,                          # one retry, then surface ERROR
        embedded_inputs=embedded_inputs or None,
    )

    # If retry budget exhausted with a non-SUCCESS status, lock the verdict to
    # ERROR. The granular failure mode is still on `record["status"]`.
    if record.get("status") != "SUCCESS" and record.get("predicted_label") is None:
        record["predicted_label"] = "ERROR"

    # Merge pair-level metadata onto the row. Schema matches the existing
    # runs.jsonl format the project already writes for benchmarking.
    row = {field: pair.get(field) for field in _PAIR_PASSTHROUGH_FIELDS}
    row["embedded_inputs"] = embedded_inputs or {}
    row.update(record)
    # Keep the canonical dialect that came from `pair`, in case
    # _base_record set a different dialect (it shouldn't, but be explicit).
    row["dialect"] = pair["dialect"]
    return row


def run(payload: dict) -> dict:
    """Pure JSON-object-in / JSON-object-out reasoning over one program pair.

    Parameters
    ----------
    payload : dict
        {
          "programA":       str,                 # required, original MLIR
          "programB":       str,                 # required, mutated MLIR
          "dialect":        "linalg" | "torch",  # required
          "embedded_inputs": {                   # optional, from execution agent
            "%arg0": <value>,
            ...
          }
        }

    Returns
    -------
    dict
        {
          "Agents": {
            "LLM_Reasoning_Agent": {
              "results": [<row>, <row>, ...]    # one entry per model, in order
            }
          }
        }

    Each row matches the existing runs.jsonl schema produced by
    batch_reasoning_agent.py — pair-level metadata, per-model verdict
    (predicted_label / confidence / predicted_output_A/B / explanation /
    key_differences / witness), runtime metadata (status / latency_ms /
    attempts / error / raw_response_head), and the embedded_inputs the
    reasoning was conditioned on.

    Side effects
    ------------
    None. This function does not read or write files.
    """
    pair, embedded_inputs = _validate_input(payload)

    results = [
        _row_for_model(pair, embedded_inputs, model_key)
        for model_key in REASONING_MODELS
    ]

    return {
        "Agents": {
            "LLM_Reasoning_Agent": {
                "results": results,
            }
        }
    }
