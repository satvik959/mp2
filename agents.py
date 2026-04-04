from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


# Load project .env so API keys are available when launched from IDE terminals.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_PRIORITY = {"low", "medium", "high", "critical"}


def _validate_provider_key(llm_model: str) -> None:
    model_name = str(llm_model or "").strip().lower()

    if model_name.startswith("groq"):
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Groq API key not found. Set GROQ_API_KEY before running agents.")
        return

    if model_name.startswith("gemini") or model_name.startswith("google/"):
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY before running agents."
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


def _fallback_response(clean_input: Dict[str, Any], error: str = "LLM failed") -> Dict[str, Any]:
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
    }


def run_agents(payload: Dict[str, Any], llm_model: str = "groq/llama-3.1-8b-instant") -> Dict[str, Any]:
    clean_input = _sanitize_input(payload)

    try:
        _validate_provider_key(llm_model)

        llm = LLM(model=llm_model, max_tokens=300, temperature=0.1)

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
        analyzer_output = _sanitize_analyzer_output(parsed.get("analyzer_output", {}))
        remediation_output = _sanitize_remediation_output(parsed.get("remediation_output", {}))

        return {
            "analysis_input": clean_input,
            "analyzer_output": analyzer_output,
            "remediation_output": remediation_output,
        }

    except Exception as exc:
        return _fallback_response(clean_input, error=str(exc))


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
