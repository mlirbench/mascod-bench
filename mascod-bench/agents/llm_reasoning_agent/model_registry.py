"""
mascod_graph.nodes.model_registry
─────────────────────────────────

Hybrid LLM registry for the MASCoD Reasoning Agent.

Every entry maps a short, leaderboard-friendly key to:
    - provider_label  (shown in JSON output and on leaderboard)
    - llm_name        (shown in JSON output and on leaderboard)
    - via             ("openrouter" | "native")
    - base_url        (OpenAI-compatible chat-completions endpoint)
    - model_id        (the exact id sent in the `model` field of the request)
    - api_key_env     (env var that holds the API key)
    - default_temp    (default temperature; eval pipeline uses 0)
    - max_tokens      (cap on completion tokens)

Most modern open-source cloud APIs (OpenRouter, DashScope, Z.ai, Moonshot,
Mistral La Plateforme, Groq, Together, Fireworks) all expose an
OpenAI-compatible /v1/chat/completions endpoint, so we route them through
langchain_openai.ChatOpenAI with a custom base_url + api_key.

Closed-source models (OpenAI, Anthropic, Google) keep their native LangChain
clients because their SDKs use non-OpenAI request shapes.

To add a model: append a new ModelSpec to MODELS. To swap a model from
OpenRouter to its native provider, change `via` and `base_url`/`model_id`/
`api_key_env`. Nothing else changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# LangChain clients. These are imported lazily inside get_model_client() so
# that simply importing model_registry doesn't fail when a particular
# integration isn't installed.


# ──────────────────────────────────────────────────────────────────
# Spec
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    key: str                    # short, stable id used in CLI + result file paths
    provider_label: str         # human-friendly provider name for the leaderboard
    llm_name: str               # human-friendly model name for the leaderboard
    via: str                    # "openrouter" | "native_openai_compat" | "anthropic" | "google" | "openai"
    base_url: Optional[str]     # OpenAI-compatible base URL (None for native SDKs)
    model_id: str               # exact id passed in the "model" field
    api_key_env: str            # env var name holding the API key
    default_temp: float = 0.0
    max_tokens: int = 2048
    family: str = "open-source" # "open-source" | "closed-source"
    notes: str = ""
    # extra_body: arbitrary key/value pairs merged into the request body. Used
    # primarily for OpenRouter's reasoning controls — e.g. for reasoning-mode
    # models we set {"reasoning": {"exclude": true}} so the model emits only
    # the final JSON answer in `content` instead of burning the token budget
    # on hidden reasoning that never makes it into the response.
    extra_body: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Filesystem-safe folder name used for results/{slug}/runs.jsonl.
        We use just the registry key (e.g. 'qwen3-coder-480b') — short,
        stable, and Judge-Agent-friendly. The full provider/llm metadata
        is preserved on every JSONL row, so this rename loses no info."""
        return self.key


# ──────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────
# Convention:
#   - All open-source cloud models default to OpenRouter (one API key).
#   - Native overrides are listed below the OpenRouter entry, commented in.
#     Flip `via`, `base_url`, `model_id`, `api_key_env` to switch.
#
# OpenRouter base URL is constant for every model:
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


MODELS: Dict[str, ModelSpec] = {

    # ───────────────────────────────────────────────────────────────
    # Open-source leaderboard set — 5 models, one per lab
    #   1. Alibaba   →  qwen3-coder-next       (Qwen3-Coder-Next, code-specialized)
    #   2. Meta      →  llama-4                 (Llama-4-Maverick MoE flagship)
    #   3. Zhipu AI  →  glm-5.1                 (GLM-5.1, slug TBV on OpenRouter)
    #   4. Moonshot  →  kimi-k2.6               (Kimi K2.6, slug TBV on OpenRouter)
    #   5. OpenAI OSS → gpt-oss-120b            (gpt-oss-120b, reasoning-effort=low)
    #
    # Closed-source baselines (paid by professor): gpt-4o, claude-sonnet-4-5,
    # gemini-2.5-flash. With the OSS+closed split, every leaderboard row
    # comes from a distinct lab — Alibaba / Meta / Zhipu / Moonshot / OpenAI-OSS
    # for OSS, OpenAI / Anthropic / Google for closed = 8 unique labs.
    # ───────────────────────────────────────────────────────────────

    "qwen3-coder-next": ModelSpec(
        key="qwen3-coder-next",
        provider_label="Alibaba (via OpenRouter)",
        llm_name="Qwen3-Next-80B-A3B-Instruct",
        via="openrouter",
        base_url=OPENROUTER_BASE,
        model_id="qwen/qwen3-next-80b-a3b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        family="open-source",
        # Bumped from 2048 default — torch input-grounded prompts produce longer
        # JSON answers (concrete output tensors), and 2048 was truncating qwen
        # mid-JSON on ~10 pairs. 4096 gives comfortable headroom.
        max_tokens=4096,
        notes="If OpenRouter rejects this id, try qwen/qwen3-next or fall back to DashScope native.",
    ),

    "llama-4": ModelSpec(
        # "Llama 4" → using the Maverick MoE flagship (Scout is the smaller MoE).
        # If you specifically wanted Scout, change model_id to meta-llama/llama-4-scout.
        key="llama-4",
        provider_label="Meta (via OpenRouter)",
        llm_name="Llama-4-Maverick",
        via="openrouter",
        base_url=OPENROUTER_BASE,
        model_id="meta-llama/llama-4-maverick",
        api_key_env="OPENROUTER_API_KEY",
        family="open-source",
        notes=(
            "Maverick is Meta's Llama-4 MoE flagship (~400B params, ~17B active). "
            "To switch to Scout (smaller MoE), set model_id=meta-llama/llama-4-scout."
        ),
    ),

    "glm-5.1": ModelSpec(
        key="glm-5.1",
        provider_label="Zhipu AI (via OpenRouter)",
        llm_name="GLM-5.1",
        via="openrouter",
        base_url=OPENROUTER_BASE,
        model_id="z-ai/glm-5.1",
        api_key_env="OPENROUTER_API_KEY",
        family="open-source",
        # Kept at 4096 (down from 8192) so OpenRouter's pre-auth credit
        # reservation is half. With ~$0.30-1.20 / M output, 4096 max_tokens
        # ≈ $0.005 reserved per call, which fits even on a low-balance key.
        max_tokens=4096,
        extra_body={"reasoning": {"exclude": True}},
        notes=(
            "Slug 'z-ai/glm-5.1' is best-guess. If 404, try 'z-ai/glm-5' "
            "or fall back to 'z-ai/glm-4.6'."
        ),
    ),

    "kimi-k2.6": ModelSpec(
        key="kimi-k2.6",
        provider_label="Moonshot (via OpenRouter)",
        llm_name="Kimi-K2.6",
        via="openrouter",
        base_url=OPENROUTER_BASE,
        model_id="moonshotai/kimi-k2.6",
        api_key_env="OPENROUTER_API_KEY",
        family="open-source",
        # Empirically: without reasoning controls Kimi-K2.6 burns 200+ s on
        # internal thinking and returns empty content. Excluding reasoning
        # collapses latency to a normal ~10-20 s and pushes the JSON answer
        # into `content`. max_tokens=4096 keeps credit pre-auth modest.
        max_tokens=4096,
        extra_body={"reasoning": {"exclude": True}},
        notes=(
            "Slug 'moonshotai/kimi-k2.6' is best-guess. Fallbacks: "
            "'moonshotai/kimi-k2-0905-preview' or 'moonshotai/kimi-k2'. "
            "If reasoning.exclude is rejected with 'mandatory', swap to "
            "{'reasoning': {'effort': 'low'}} (matches gpt-oss config)."
        ),
    ),

    "gpt-oss-120b": ModelSpec(
        key="gpt-oss-120b",
        provider_label="OpenAI OSS (via OpenRouter)",
        llm_name="gpt-oss-120b",
        via="openrouter",
        base_url=OPENROUTER_BASE,
        model_id="openai/gpt-oss-120b",
        api_key_env="OPENROUTER_API_KEY",
        family="open-source",
        # 4096 (down from 8192) to halve OpenRouter's credit pre-auth.
        max_tokens=4096,
        # gpt-oss requires reasoning enabled; cap at lowest effort tier.
        extra_body={"reasoning": {"effort": "low"}},
        notes="Native alt: Groq base_url=https://api.groq.com/openai/v1, model_id=openai/gpt-oss-120b, api_key_env=GROQ_API_KEY",
    ),

    # ───────── Closed-source baselines (paid by professor) ─────────
    # Best-guess API model IDs — verify against each provider's current docs
    # before launch:
    #   OpenAI:    https://platform.openai.com/docs/models
    #   Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
    #   Google:    https://ai.google.dev/gemini-api/docs/models

    "gpt-4.1-mini": ModelSpec(
        key="gpt-4.1-mini",
        provider_label="OpenAI",
        llm_name="GPT-4.1-Mini",
        via="openai",
        base_url=None,
        model_id="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        family="closed-source",
        # 4096 so input-grounded predictions fit cleanly (matches the qwen fix).
        max_tokens=4096,
        notes=(
            "Slug 'gpt-4.1-mini' is the standard OpenAI ID for GPT-4.1 Mini. "
            "If the API returns 'invalid model', try 'gpt-4.1-mini-2025-04-14' "
            "(dated variant) or check platform.openai.com/docs/models."
        ),
    ),

        "claude-sonnet-4-5": ModelSpec(
        key="claude-sonnet-4-5",
        provider_label="Anthropic",
        llm_name="Claude Sonnet 4.5",
        via="anthropic",
        base_url=None,
        model_id="claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        family="closed-source",
        max_tokens=4096,
        notes=(
            "Claude Sonnet 4.5 stable release."
        ),
    ),


        "claude-haiku-4-5": ModelSpec(
        key="claude-haiku-4-5",
        provider_label="Anthropic",
        llm_name="Claude Haiku 4.5",
        via="anthropic",
        base_url=None,
        model_id="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        family="closed-source",
        max_tokens=4096,
        notes="Fast and cost-efficient Claude model.",
    ),

    "gemini-3-flash": ModelSpec(
        key="gemini-3-flash",
        provider_label="Google",
        llm_name="Gemini 3 Flash (preview)",
        via="google",
        base_url=None,
        # Confirmed accessible via the user's key. The bare 'gemini-3-flash'
        # 404s; the working slug is 'gemini-3-flash-preview'.
        model_id="gemini-3-flash-preview",
        api_key_env="GOOGLE_API_KEY",
        family="closed-source",
        max_tokens=4096,
        notes=(
            "Verified slug as of 2026-05. Other Gemini-3 variants visible "
            "to the same key: gemini-3-pro-preview, gemini-3.1-pro-preview, "
            "gemini-3.1-flash-lite-preview."
        ),
    ),

        "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash",
        provider_label="Google",
        llm_name="Gemini 2.5 Flash",
        via="google",
        base_url=None,
        model_id="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        family="closed-source",
        max_tokens=4096,
        notes="Stable Gemini 2.5 Flash model.",
    ),

    "gemini-2.5-pro": ModelSpec(
        key="gemini-2.5-pro",
        provider_label="Google",
        llm_name="Gemini 2.5 Pro",
        via="google",
        base_url=None,
        model_id="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        family="closed-source",
        max_tokens=4096,
        notes="Gemini 2.5 Pro for stronger reasoning/code evaluation.",
    ),
}


# Legacy aliases — keep the old "openai" / "claude" / "gemini" CLI flags working
# by routing them to the new closed-source entries.
LEGACY_ALIASES: Dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "claude": "claude-sonnet-4-5",
    "gemini": "gemini-3-flash",
}


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def resolve_key(name: str) -> str:
    """Map any user-facing identifier to a canonical registry key."""
    if name in MODELS:
        return name
    if name in LEGACY_ALIASES:
        return LEGACY_ALIASES[name]
    # Allow case-insensitive match
    lower = name.lower()
    for k in MODELS:
        if k.lower() == lower:
            return k
    raise KeyError(
        f"Unknown model '{name}'. "
        f"Known: {sorted(list(MODELS.keys()) + list(LEGACY_ALIASES.keys()))}"
    )


def get_spec(name: str) -> ModelSpec:
    return MODELS[resolve_key(name)]


def list_models(family: Optional[str] = None) -> list[ModelSpec]:
    """Return all registered specs, optionally filtered by family."""
    if family is None:
        return list(MODELS.values())
    return [m for m in MODELS.values() if m.family == family]


def get_model_client(name: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """
    Return a LangChain chat client configured for the given registered model.

    Raises a clear RuntimeError if the API key env var is missing, so the
    batch runner can skip-and-record-error rather than crashing the run.
    """
    spec = get_spec(name)

    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for {spec.llm_name}: set env var {spec.api_key_env}"
        )

    temp = spec.default_temp if temperature is None else temperature
    mt = spec.max_tokens if max_tokens is None else max_tokens

    if spec.via in ("openrouter", "native_openai_compat"):
        from langchain_openai import ChatOpenAI
        # model_kwargs are merged into the chat-completions request body, which
        # is how OpenRouter's `reasoning` parameter gets passed through.
        return ChatOpenAI(
            model=spec.model_id,
            temperature=temp,
            max_tokens=mt,
            base_url=spec.base_url,
            api_key=api_key,
            timeout=180,
            model_kwargs=dict(spec.extra_body) if spec.extra_body else {},
        )

    if spec.via == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=spec.model_id,
            temperature=temp,
            max_tokens=mt,
            api_key=api_key,
            timeout=180,
            model_kwargs=dict(spec.extra_body) if spec.extra_body else {},
        )

    if spec.via == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=spec.model_id,
            temperature=temp,
            max_tokens=mt,
            api_key=api_key,
            timeout=120,
        )

    if spec.via == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Note: Gemini's parameter is `max_output_tokens`, not `max_tokens`.
        return ChatGoogleGenerativeAI(
            model=spec.model_id,
            temperature=temp,
            max_output_tokens=mt,
            google_api_key=api_key,
            timeout=180,
        )

    raise ValueError(f"Unknown 'via' value for model {spec.key}: {spec.via}")


# ──────────────────────────────────────────────────────────────────
# Tiny CLI for inspecting the registry
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for spec in MODELS.values():
            key_status = "✓" if os.environ.get(spec.api_key_env) else "✗"
            print(f"  [{key_status}] {spec.key:25s}  {spec.family:13s}  {spec.provider_label}  ::  {spec.llm_name}")
        print()
        print("Legend: [✓] API key found in env, [✗] missing")
    else:
        print("Usage: python -m nodes.model_registry list")
