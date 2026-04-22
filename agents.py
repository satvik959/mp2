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
    text_str = str(text)
    
    try:
        parsed = json.loads(text_str)
        if isinstance(parsed, dict) and "analyzer_output" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text_str, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "analyzer_output" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
            
    # Aggressive Regex Fallback if LLM generated broken JSON (e.g. missing commas/unescaped quotes)
    def get_val(key, default=""):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text_str)
        return m.group(1) if m else default

    anomaly_type = get_val("anomaly_type", "Network Anomaly Detected")
    cause = get_val("cause", "Suspicious traffic patterns detected based on payload metrics.")
    confidence = get_val("confidence", "high")
    priority = get_val("priority", "high")
    
    actions = []
    actions_match = re.search(r'"recommended_actions"\s*:\s*\[(.*?)\]', text_str, re.DOTALL)
    if actions_match:
        for m in re.finditer(r'"([^"]+)"', actions_match.group(1)):
            actions.append(m.group(1))
            
    if not actions:
        actions = ["Implement rate limiting and block suspicious IPs", "Monitor firewall logs"]

    return {
        "analyzer_output": {
            "anomaly_type": anomaly_type,
            "cause": cause,
            "confidence": confidence,
            "evidence": []
        },
        "remediation_output": {
            "recommended_actions": actions,
            "priority": priority,
            "notes": ""
        }
    }


def _sanitize_input(inp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction": str(inp.get("prediction", "normal")).lower(),
        "protocol": str(inp.get("protocol", "UNKNOWN")).upper(),
        "flags_pattern": [str(f).upper() for f in inp.get("flags_pattern", [])][:3],
        "packet_rate": _coerce_int(inp.get("packet_rate", 0), 0),
        "connection_count": _coerce_int(inp.get("connection_count", 0), 0),
        "avg_packet_size": _coerce_float(inp.get("avg_packet_size", 0.0), 0.0),
        "packet_info": str(inp.get("info", ""))[:100],
    }


def _sanitize_analyzer_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    confidence = str(raw.get("confidence", "low")).lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"

    evidence_raw = raw.get("evidence", [])
    if not isinstance(evidence_raw, list):
        evidence_raw = [evidence_raw]

    return {
        "anomaly_type": str(raw.get("anomaly_type", "unknown")),
        "cause": str(raw.get("cause", "insufficient evidence")),
        "confidence": confidence,
        "evidence": [str(item) for item in evidence_raw if str(item).strip()][:4],
    }


def _sanitize_remediation_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    priority = str(raw.get("priority", "medium")).lower()
    if priority not in ALLOWED_PRIORITY:
        priority = "medium"

    actions_raw = raw.get("recommended_actions", [])
    if not isinstance(actions_raw, list):
        actions_raw = [actions_raw]

    return {
        "recommended_actions": [str(item) for item in actions_raw if str(item).strip()][:4],
        "priority": priority,
        "notes": str(raw.get("notes", "")),
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

        llm = LLM(model=llm_model, max_tokens=300, temperature=0.3)

        agent = Agent(
            role="Senior SOC Analyst",
            goal="Provide highly creative, diverse, and extremely specific remediation strategies for network anomalies.",
            backstory="Elite threat hunter known for thinking outside the box. Never gives the same generic advice twice.",
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=1,
        )

        task = Task(
            description=(
                f"PAYLOAD:{json.dumps(clean_input, separators=(',', ':'))}\n"
                "CRITICAL INSTRUCTION: You MUST provide highly unique, creative, and distinct recommended_actions tailored explicitly to the IPs and Packet Info provided! DO NOT repeat standard generic advice (like 'increase SYN cookie timeout') for every anomaly. Think of unique, advanced network defense strategies! Ensure the 'cause' is also uniquely phrased based on the specific packet info.\n"
                "CRITICAL JSON INSTRUCTION: You must output ONLY valid, raw JSON. Do NOT wrap in markdown backticks. Do NOT include ANY conversational text.\n"
                "Reply STRICTLY with this exact JSON schema:{\"analyzer_output\":{\"anomaly_type\":\"str\","
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
    import sys
    if len(sys.argv) > 1:
        payload = json.loads(sys.argv[1])
        print(json.dumps(run_agents(payload)))
    else:
        test_payload = {
            "prediction": "brute_force",
            "packet_rate": 45.0,
            "protocol": "TCP",
            "flags_pattern": ["PA", "PA", "PA"],
            "connection_count": 15,
            "batch_summary": "Suspicious PA flags to destination",
        }
        print(json.dumps(run_agents(test_payload), indent=2))
