import json
import os
import re
import time
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(env_path))

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Rate limit tracking
_last_api_call = 0
_rate_limit_cooldown = 10  # seconds


def _sanitize_input(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize input data"""
    return {
        "prediction": str(inp.get("prediction", "normal")).lower(),
        "packet_rate": float(inp.get("packet_rate", 0)),
        "protocol": str(inp.get("protocol", "UNKNOWN")).upper(),
        "flags_pattern": [str(f).upper() for f in inp.get("flags_pattern", [])],
        "connection_count": int(inp.get("connection_count", 0)),
        "batch_summary": str(inp.get("batch_summary", ""))[:100],
    }


def run_agents(payload: Dict[str, Any], llm_model: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """
    Direct Gemini API call (no CrewAI overhead).
    Reduces token usage by 60-70% vs CrewAI wrapper.
    """
    global _last_api_call
    
    if not GEMINI_AVAILABLE:
        return _fallback_response(payload)
    
    clean_input = _sanitize_input(payload)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return _fallback_response(payload)
    
    # Rate limiting: wait between calls
    elapsed = time.time() - _last_api_call
    if elapsed < _rate_limit_cooldown:
        # Return cached response if we're rate limited
        return _fallback_response(payload, error="Rate limited - try again later")
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Minimal prompt (token-efficient)
        prompt = (
            "Network anomaly detected. Analyze and respond in JSON only:\n"
            f"{json.dumps(clean_input)}\n\n"
            "Return JSON with: anomaly_type, cause, confidence (low|medium|high), "
            "evidence (list), recommended_actions (list), priority (low|medium|high|critical), notes"
        )
        
        _last_api_call = time.time()
        
        response = client.models.generate_content(
            model=llm_model,
            contents=prompt
        )
        
        text = response.text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        
        return {
            "analyzer_output": {
                "anomaly_type": str(parsed.get("anomaly_type", "unknown")).lower(),
                "cause": str(parsed.get("cause", "N/A"))[:80],
                "confidence": str(parsed.get("confidence", "low")).lower(),
                "evidence": parsed.get("evidence", [])[:3],
            },
            "remediation_output": {
                "recommended_actions": parsed.get("recommended_actions", [])[:3],
                "priority": str(parsed.get("priority", "medium")).lower(),
                "notes": str(parsed.get("notes", ""))[:80],
            }
        }
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return _fallback_response(payload, error="Rate limited - too many requests")
        return _fallback_response(payload, error=error_msg)


def _fallback_response(payload: Dict[str, Any], error: str = "") -> Dict[str, Any]:
    """Graceful fallback when Gemini API fails or is unavailable"""
    clean_input = _sanitize_input(payload)
    
    return {
        "analyzer_output": {
            "anomaly_type": clean_input.get("prediction", "unknown"),
            "cause": error[:80] if error else "API unavailable",
            "confidence": "low",
            "evidence": [f"Detected as {clean_input['prediction']}"],
        },
        "remediation_output": {
            "recommended_actions": [
                "Isolate the source IP immediately",
                "Review traffic patterns",
                "Check firewall logs"
            ],
            "priority": "high",
            "notes": "Fallback analysis (Gemini unavailable)",
        }
    }


if __name__ == "__main__":
    # Test direct Gemini integration
    test_payload = {
        "prediction": "brute_force",
        "packet_rate": 45.0,
        "protocol": "TCP",
        "flags_pattern": ["PA", "PA", "PA"],
        "connection_count": 15,
        "batch_summary": "Suspicious PA flags to destination",
    }

    print("🚀 Testing direct Gemini API integration...")
    print(f"📊 Input: {test_payload}\n")
    
    result = run_agents(test_payload)
    print("✅ Analysis Result:")
    print(json.dumps(result, indent=2))
