"""CrewAI analyzer and remediation orchestration for network anomaly reasoning."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from crewai import Agent, Crew, Process, Task


ALLOWED_PREDICTIONS = {"flood", "probe", "brute_force", "exploit", "malware", "normal"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_PRIORITY = {"low", "medium", "high", "critical"}


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
	"""Extract a JSON object from model text output."""
	stripped = text.strip()
	try:
		parsed = json.loads(stripped)
		return parsed if isinstance(parsed, dict) else {}
	except json.JSONDecodeError:
		pass

	# Recover if LLM returns prose before/after JSON.
	match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
	if not match:
		return {}

	try:
		parsed = json.loads(match.group(0))
		return parsed if isinstance(parsed, dict) else {}
	except json.JSONDecodeError:
		return {}


def _sanitize_analysis_input(analysis_input: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize analysis input to expected schema and safe types."""
	prediction = str(analysis_input.get("prediction", "normal")).strip().lower()
	if prediction not in ALLOWED_PREDICTIONS:
		prediction = "normal"

	flags = analysis_input.get("flags_pattern", [])
	if not isinstance(flags, list):
		flags = [str(flags)]
	flags_pattern = [str(flag).strip().upper() for flag in flags if str(flag).strip()]

	protocol = str(analysis_input.get("protocol", "UNKNOWN")).strip().upper() or "UNKNOWN"
	batch_summary = str(analysis_input.get("batch_summary", "")).strip()

	return {
		"prediction": prediction,
		"packet_rate": _coerce_float(analysis_input.get("packet_rate", 0.0), 0.0),
		"protocol": protocol,
		"flags_pattern": flags_pattern,
		"connection_count": _coerce_int(analysis_input.get("connection_count", 0), 0),
		"batch_summary": batch_summary,
	}


def _sanitize_analyzer_output(raw: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate analyzer output against required schema and value sets."""
	anomaly_type = str(raw.get("anomaly_type", "unknown")).strip() or "unknown"
	cause = str(raw.get("cause", "insufficient evidence")).strip() or "insufficient evidence"

	confidence = str(raw.get("confidence", "low")).strip().lower()
	if confidence not in ALLOWED_CONFIDENCE:
		confidence = "low"

	evidence_raw = raw.get("evidence", [])
	if not isinstance(evidence_raw, list):
		evidence_raw = [evidence_raw]
	evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]

	return {
		"anomaly_type": anomaly_type,
		"cause": cause,
		"confidence": confidence,
		"evidence": evidence,
	}


def _sanitize_remediation_output(raw: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate remediation output against required schema and value sets."""
	actions_raw = raw.get("recommended_actions", [])
	if not isinstance(actions_raw, list):
		actions_raw = [actions_raw]
	recommended_actions = [str(item).strip() for item in actions_raw if str(item).strip()]

	priority = str(raw.get("priority", "medium")).strip().lower()
	if priority not in ALLOWED_PRIORITY:
		priority = "medium"

	notes = str(raw.get("notes", "")).strip()

	return {
		"recommended_actions": recommended_actions,
		"priority": priority,
		"notes": notes,
	}


def build_agents(llm_model: str) -> Tuple[Agent, Agent]:
	"""Create analyzer and remediation agents using the same LLM model."""
	analyzer_agent = Agent(
		role="Network Anomaly Analyzer",
		goal="Identify the likely root cause of anomalous network behavior from ML signals and evidence.",
		backstory=(
			"You are a SOC analyst focused on evidence-based root-cause analysis. "
			"You infer causes only from provided network signals and context."
		),
		llm=llm_model,
		verbose=False,
		allow_delegation=False,
		tools=[],
	)

	remediation_agent = Agent(
		role="Network Remediation Advisor",
		goal="Recommend practical and prioritized mitigation actions based on analyzed anomaly cause.",
		backstory=(
			"You are an incident responder who provides safe, prioritized containment and remediation actions."
		),
		llm=llm_model,
		verbose=False,
		allow_delegation=False,
		tools=[],
	)

	return analyzer_agent, remediation_agent


def run_agents(analysis_input: Dict[str, Any], llm_model: str = "gpt-4o-mini") -> Dict[str, Any]:
	"""Run Analyzer -> Remediation sequentially and return structured outputs."""
	clean_input = _sanitize_analysis_input(analysis_input)
	analyzer_agent, remediation_agent = build_agents(llm_model)

	analyzer_task = Task(
		description=(
			"You are given network anomaly evidence as JSON:\n"
			f"{json.dumps(clean_input, indent=2)}\n\n"
			"Return STRICT JSON only with this exact schema:\n"
			"{\n"
			'  "anomaly_type": "string",\n'
			'  "cause": "string",\n'
			'  "confidence": "low|medium|high",\n'
			'  "evidence": ["string", "..."]\n'
			"}\n"
			"No markdown. No extra keys. No prose outside JSON."
		),
		expected_output="JSON object with anomaly_type, cause, confidence, evidence",
		agent=analyzer_agent,
	)

	analyzer_crew = Crew(
		agents=[analyzer_agent],
		tasks=[analyzer_task],
		process=Process.sequential,
		verbose=False,
	)
	analyzer_raw_text = str(analyzer_crew.kickoff())
	analyzer_parsed = _extract_json_payload(analyzer_raw_text)
	analyzer_output = _sanitize_analyzer_output(analyzer_parsed)

	remediation_task = Task(
		description=(
			"You are given analyzer output and original evidence.\n"
			f"Analyzer Output:\n{json.dumps(analyzer_output, indent=2)}\n\n"
			f"Original Evidence:\n{json.dumps(clean_input, indent=2)}\n\n"
			"Return STRICT JSON only with this exact schema:\n"
			"{\n"
			'  "recommended_actions": ["string", "..."],\n'
			'  "priority": "low|medium|high|critical",\n'
			'  "notes": "string"\n'
			"}\n"
			"No markdown. No extra keys. No prose outside JSON."
		),
		expected_output="JSON object with recommended_actions, priority, notes",
		agent=remediation_agent,
	)

	remediation_crew = Crew(
		agents=[remediation_agent],
		tasks=[remediation_task],
		process=Process.sequential,
		verbose=False,
	)
	remediation_raw_text = str(remediation_crew.kickoff())
	remediation_parsed = _extract_json_payload(remediation_raw_text)
	remediation_output = _sanitize_remediation_output(remediation_parsed)

	# Fallback minimums if LLM returns unusable payloads.
	if not analyzer_output["cause"]:
		analyzer_output = {
			"anomaly_type": "unknown",
			"cause": "insufficient evidence",
			"confidence": "low",
			"evidence": [],
		}

	if not remediation_output["recommended_actions"]:
		remediation_output = {
			"recommended_actions": ["Collect additional packet evidence", "Escalate to SOC triage"],
			"priority": "medium",
			"notes": "No reliable remediation suggestions were generated.",
		}

	return {
		"analysis_input": clean_input,
		"analyzer_output": analyzer_output,
		"remediation_output": remediation_output,
	}


if __name__ == "__main__":
	# Minimal usage example
	minimal_input = {
		"prediction": "probe",
		"packet_rate": 12.5,
		"protocol": "TCP",
		"flags_pattern": ["SYN", "ACK"],
		"connection_count": 18,
		"batch_summary": "3 anomalies in last batch",
	}

	# SYN flood scenario example
	syn_flood_input = {
		"prediction": "flood",
		"packet_rate": 240.0,
		"protocol": "TCP",
		"flags_pattern": ["SYN", "SYN", "SYN", "SYN", "SYN"],
		"connection_count": 320,
		"batch_summary": "8 anomalies in last batch with sharp SYN spike",
	}

	# Optional local smoke run (requires CrewAI provider credentials in environment).
	_ = minimal_input, syn_flood_input
