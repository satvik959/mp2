from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


# Load project .env so API keys are available when launched from IDE terminals.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_PRIORITY = {"low", "medium", "high", "critical"}
DEFAULT_GROQ_MODEL = "groq/llama3-8b-8192"
GROQ_MODEL_CANDIDATES = [
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.1-70b-versatile",
]


def _validate_provider_key(llm_model: str) -> None:
    model_name = str(llm_model or "").strip().lower()

    # Groq hosted models (e.g., groq/llama-3.1-8b-instant)
    if model_name.startswith("groq"):
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Groq API key not found. Set GROQ_API_KEY before running agents.")
        return

    # Gemini path intentionally disabled for Groq-only operation.
    # if model_name.startswith("gemini") or model_name.startswith("google/"):
    #     if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    #         raise RuntimeError(
    #             "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY before running agents."
    #         )
    #     return

    # xAI Grok style model aliases. We allow multiple env var names to reduce setup friction.
    if model_name.startswith("xai") or model_name.startswith("grok") or model_name.startswith("x-ai"):
        if not os.getenv("XAI_API_KEY") and not os.getenv("GROK_API_KEY") and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "Grok/xAI API key not found. Set XAI_API_KEY (or GROK_API_KEY)."
            )
        return

    if model_name.startswith("gpt") or model_name.startswith("openai/"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY before running agents.")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_json_payload(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _sanitize_input(inp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction": str(inp.get("prediction", "normal")).lower(),
        "protocol": str(inp.get("protocol", "UNKNOWN")).upper(),
        "flags_pattern": [str(f).upper() for f in inp.get("flags_pattern", [])][:3],
        "packet_rate": _coerce_int(inp.get("packet_rate", 0), 0),
        "connection_count": _coerce_int(inp.get("connection_count", 0), 0),
        "batch_summary": str(inp.get("batch_summary", ""))[:100],
        "avg_packet_size": _coerce_float(inp.get("avg_packet_size", 0.0), 0.0),
    }


def _sanitize_analyzer_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    confidence = str(raw.get("confidence", "low")).lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"

    evidence_raw = raw.get("evidence", [])
    if not isinstance(evidence_raw, list):
        evidence_raw = [evidence_raw]

    return {
        "anomaly_type": str(raw.get("anomaly_type", "unknown"))[:60],
        "cause": str(raw.get("cause", "insufficient evidence"))[:140],
        "confidence": confidence,
        "evidence": [str(item)[:80] for item in evidence_raw if str(item).strip()][:4],
    }


def _sanitize_remediation_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    priority = str(raw.get("priority", "medium")).lower()
    if priority not in ALLOWED_PRIORITY:
        priority = "medium"

    actions_raw = raw.get("recommended_actions", [])
    if not isinstance(actions_raw, list):
        actions_raw = [actions_raw]

    return {
        "recommended_actions": [str(item)[:90] for item in actions_raw if str(item).strip()][:4],
        "priority": priority,
        "notes": str(raw.get("notes", ""))[:140],
    }


def _fallback_response(
    clean_input: Dict[str, Any],
    error: str = "LLM failed",
    attempted_models: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "analysis_input": clean_input,
        "analyzer_output": {
            "anomaly_type": clean_input.get("prediction", "unknown"),
            "cause": str(error)[:140],
            "confidence": "low",
            "evidence": ["Model response unavailable"],
        },
        "remediation_output": {
            "recommended_actions": ["Retry later", "Collect additional packet evidence"],
            "priority": "medium",
            "notes": "Fallback response generated.",
        },
        "model_used": "none",
        "fallback_used": True,
        "attempted_models": attempted_models or [],
    }


def _normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    low = raw.lower()
    if low == "groq/llama-3.1-8b-instant":
        return "groq/llama3-8b-8192"
    if low == "groq/llama-3.1-70b-versatile":
        return "groq/llama3-70b-8192"
    return raw


def _provider_family(model_name: str) -> str:
    model_name = str(model_name or "").strip().lower()
    # Gemini family intentionally disabled.
    # if model_name.startswith("gemini") or model_name.startswith("google/"):
    #     return "gemini"
    if model_name.startswith("groq"):
        return "groq"
    if model_name.startswith("xai") or model_name.startswith("grok") or model_name.startswith("x-ai"):
        return "grok"
    if model_name.startswith("gpt") or model_name.startswith("openai/"):
        return "openai"
    return "unknown"


def _is_rate_limit_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota exceeded" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def _candidate_models(primary_model: str) -> list[str]:
    primary_normalized = _normalize_model_name(primary_model)
    candidates = [primary_normalized]
    family = _provider_family(primary_normalized)

    if family == "groq":
        candidates.extend(GROQ_MODEL_CANDIDATES)

    # If selected provider is throttled, try the other provider automatically.
    if family in {"unknown"} and os.getenv("GROQ_API_KEY"):
        candidates.extend(GROQ_MODEL_CANDIDATES)
    # Gemini fallback intentionally disabled for Groq-only operation.
    # if family in {"groq", "grok", "unknown"} and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    #     candidates.append(DEFAULT_GEMINI_MODEL)

    unique: list[str] = []
    seen = set()
    for model_name in candidates:
        key = str(model_name).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(model_name)
    return unique


def _run_single_model(clean_input: Dict[str, Any], llm_model: str) -> Dict[str, Any]:
    _validate_provider_key(llm_model)

    llm = LLM(model=llm_model, max_tokens=220, temperature=0.1)

    agent = Agent(
        role="SOC analyst",
        goal="Analyze anomaly and suggest remediation",
        backstory="Expert network intrusion analyst",
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    task = Task(
        description=(
            f"Data:{json.dumps(clean_input, separators=(',', ':'))} "
            "Reply JSON only:{\"analyzer_output\":{\"anomaly_type\":\"str\","
            "\"cause\":\"str\",\"confidence\":\"low|medium|high\",\"evidence\":[\"str\"]},"
            "\"remediation_output\":{\"recommended_actions\":[\"str\"],"
            "\"priority\":\"low|medium|high|critical\",\"notes\":\"str\"}}"
        ),
        expected_output="JSON",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    parsed = _extract_json_payload(str(crew.kickoff()))

    if "analyzer_output" not in parsed or "remediation_output" not in parsed:
        raise RuntimeError("Model response missing required JSON keys.")

    return {
        "analysis_input": clean_input,
        "analyzer_output": _sanitize_analyzer_output(parsed.get("analyzer_output", {})),
        "remediation_output": _sanitize_remediation_output(parsed.get("remediation_output", {})),
    }


def run_agents(payload: Dict[str, Any], llm_model: str = DEFAULT_GROQ_MODEL) -> Dict[str, Any]:
    clean_input = _sanitize_input(payload)
    attempted_models: list[str] = []
    last_error = "LLM failed"

    for candidate_model in _candidate_models(llm_model):
        attempted_models.append(candidate_model)

        for attempt in range(2):
            try:
                result = _run_single_model(clean_input, candidate_model)
                result["model_used"] = candidate_model
                result["fallback_used"] = candidate_model != _normalize_model_name(llm_model)
                result["attempted_models"] = attempted_models.copy()
                return result
            except Exception as exc:
                last_error = f"{candidate_model}: {exc}"

                if _is_rate_limit_error(str(exc)) and attempt < 1:
                    # Small backoff helps with per-minute/provider burst throttling.
                    time.sleep(1.2)
                    continue

                break

    return _fallback_response(
        clean_input,
        error=f"{last_error} | tried={attempted_models}",
        attempted_models=attempted_models,
    )


if __name__ == "__main__":
    test_payload = {
        "prediction": "brute_force",
        "packet_rate": 45.0,
        "protocol": "TCP",
        "flags_pattern": ["PA", "PA", "PA"],
        "connection_count": 15,
        "batch_summary": "Suspicious PA flags to destination",
    }

    print(json.dumps(run_agents(test_payload), indent=2))
