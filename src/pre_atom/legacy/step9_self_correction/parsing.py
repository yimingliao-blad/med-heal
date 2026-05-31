"""
Shared Qwen3-32B parsing module for detection, correction, and verdict.

Extracts structured JSON from free-form LLM output.
Validates all fields. Reports parse quality.
"""
import json
import re
import requests

QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"
QWEN32B_MODEL = "Qwen/Qwen3-32B-MLX-bf16"


# ============================================================
# DETECTION EXTRACTION
# ============================================================

DETECT_EXTRACT_PROMPT = """/nothink
Read this self-critique output from a medical AI checking its own answer against discharge notes.

SELF-CRITIQUE:
{raw_output}

Extract the following:

1. VERDICT: Did the AI conclude the answer was CORRECT or INCORRECT?
   - INCORRECT if: the AI found specific factual errors, contradictions, critical omissions, or question misalignment
   - CORRECT if: the AI confirmed all claims are supported and the answer addresses the question
   - UNCLEAR if: cannot determine

2. ERROR_TYPE: Based on what the AI ACTUALLY FOUND (not just what it labeled), classify as:
   - CONTRADICTION: the answer states a fact that CONFLICTS with the notes (wrong medication, wrong value, wrong procedure, fabricated detail)
   - OMISSION: critical information is MISSING that would change the answer's conclusion
   - QUESTION_MISALIGNMENT: the answer addresses the wrong visit, time period, or clinical focus
   - NONE: no error found

3. ERROR_STATEMENT: The specific wrong claim or missing info, stated as ONE factual sentence.
   For CONTRADICTION: state what the answer claims that is wrong.
   For OMISSION: state what information is missing.
   For QUESTION_MISALIGNMENT: state what the answer addresses vs what the question asks.

4. CORRECT_STATEMENT: What the discharge notes actually say about this topic.
   Must be grounded in the notes — not inferred.

5. CONFIDENCE: How clear was the AI's conclusion?
   - HIGH: the AI explicitly stated the answer is wrong with specific evidence
   - MEDIUM: the AI found issues but hedged ("mostly correct but...")
   - LOW: the AI was unclear or contradicted itself

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT" or "UNCLEAR", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "...", "correct_statement": "...", "confidence": "HIGH" or "MEDIUM" or "LOW"}}"""


# ============================================================
# VERDICT EXTRACTION
# ============================================================

VERDICT_EXTRACT_PROMPT = """/nothink
Read this comparison output where a medical AI compared two answers.

COMPARISON:
{raw_output}

Which answer was chosen as better? Extract:

{{"pick": "A" or "B" or "ORIGINAL" or "CORRECTED" or "UNCLEAR", "reason": "brief reason"}}"""


# ============================================================
# CORE FUNCTIONS
# ============================================================

def _call_qwen32b(system, user, max_tokens=400):
    """Call Qwen3-32B. Returns raw text."""
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": QWEN32B_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"</think>", "", text).strip()
        return text
    except Exception as e:
        return f'{{"error": "{e}"}}'


def _extract_json(text):
    """Extract JSON object from text. Returns (dict, method) or (None, 'failed')."""
    # Direct parse
    try:
        return json.loads(text), "direct"
    except:
        pass
    # Find JSON in text
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), "extracted"
        except:
            pass
    # Strip markdown
    cleaned = re.sub(r'^```\w*\n?', '', text.strip())
    cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try:
        return json.loads(cleaned), "cleaned"
    except:
        pass
    return None, "failed"


# ============================================================
# DETECTION PARSING
# ============================================================

VALID_VERDICTS = {"CORRECT", "INCORRECT", "UNCLEAR"}
VALID_TYPES = {"CONTRADICTION", "OMISSION", "QUESTION_MISALIGNMENT", "NONE"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def parse_detection(raw_output):
    """
    Parse free-form detection output via Qwen3-32B.

    Returns dict with:
      verdict: CORRECT/INCORRECT/UNCLEAR/PARSE_FAIL
      error_type: CONTRADICTION/OMISSION/QUESTION_MISALIGNMENT/NONE
      error_statement: str
      correct_statement: str
      confidence: HIGH/MEDIUM/LOW
      parse_method: str
      validation: dict of field-level checks
    """
    q32_raw = _call_qwen32b(
        "Extract structured information. Output ONLY valid JSON.",
        DETECT_EXTRACT_PROMPT.format(raw_output=raw_output[:3000]),
    )
    obj, method = _extract_json(q32_raw)

    result = {
        "verdict": "PARSE_FAIL",
        "error_type": "NONE",
        "error_statement": "",
        "correct_statement": "",
        "confidence": "LOW",
        "parse_method": method,
        "q32b_raw": q32_raw[:300],
        "validation": {},
    }

    if not obj or not isinstance(obj, dict):
        result["validation"]["json_parsed"] = False
        return result

    result["validation"]["json_parsed"] = True

    # Validate each field
    v = str(obj.get("verdict", "")).upper()
    result["validation"]["verdict_valid"] = v in VALID_VERDICTS
    result["verdict"] = v if v in VALID_VERDICTS else "UNCLEAR"

    et = str(obj.get("error_type", "NONE")).upper()
    # Normalize common variants
    if "MISREAD" in et: et = "CONTRADICTION"
    if "FABRICAT" in et: et = "CONTRADICTION"
    result["validation"]["error_type_valid"] = et in VALID_TYPES
    result["error_type"] = et if et in VALID_TYPES else "NONE"

    es = str(obj.get("error_statement", ""))
    result["error_statement"] = es[:300]
    result["validation"]["has_error_statement"] = bool(es) and es.lower() not in ("", "none", "n/a", "empty")

    cs = str(obj.get("correct_statement", ""))
    result["correct_statement"] = cs[:300]
    result["validation"]["has_correct_statement"] = bool(cs) and cs.lower() not in ("", "none", "n/a", "empty")

    conf = str(obj.get("confidence", "LOW")).upper()
    result["validation"]["confidence_valid"] = conf in VALID_CONFIDENCE
    result["confidence"] = conf if conf in VALID_CONFIDENCE else "LOW"

    # Cross-validation: if verdict=INCORRECT, should have error details
    if result["verdict"] == "INCORRECT":
        result["validation"]["has_error_for_incorrect"] = result["validation"]["has_error_statement"]
        result["validation"]["has_notes_for_incorrect"] = result["validation"]["has_correct_statement"]
    elif result["verdict"] == "CORRECT":
        result["validation"]["correct_has_no_error"] = result["error_type"] == "NONE"

    # Infer error type from content if label seems wrong
    if result["verdict"] == "INCORRECT" and result["error_type"] == "NONE":
        # Detected error but no type — infer from content
        es_lower = es.lower()
        if any(w in es_lower for w in ["incorrectly", "contradicts", "states", "wrong", "conflicts"]):
            result["error_type"] = "CONTRADICTION"
            result["validation"]["type_inferred"] = True
        elif any(w in es_lower for w in ["omit", "missing", "absent", "lacks", "not mention"]):
            result["error_type"] = "OMISSION"
            result["validation"]["type_inferred"] = True
        elif any(w in es_lower for w in ["wrong visit", "wrong aspect", "misalign", "different question"]):
            result["error_type"] = "QUESTION_MISALIGNMENT"
            result["validation"]["type_inferred"] = True

    return result


def infer_type_from_content(error_statement):
    """
    Infer error type from error_statement content.
    Returns (type, confidence) tuple.
    """
    es = error_statement.lower()
    contra_words = ["incorrectly", "contradicts", "states", "claims", "wrong", "conflicts",
                    "not match", "says", "but the notes", "not mentioned in the notes"]
    omis_words = ["omit", "missing", "absent", "lacks", "not mention", "not include",
                  "does not address", "not provided"]
    qmis_words = ["wrong visit", "wrong aspect", "wrong admission", "wrong time",
                  "different question", "misalign", "addresses the first instead",
                  "addresses the second instead"]

    contra_score = sum(1 for w in contra_words if w in es)
    omis_score = sum(1 for w in omis_words if w in es)
    qmis_score = sum(1 for w in qmis_words if w in es)

    if qmis_score > 0 and qmis_score >= contra_score:
        return "QUESTION_MISALIGNMENT", "HIGH" if qmis_score >= 2 else "MEDIUM"
    if contra_score > omis_score:
        return "CONTRADICTION", "HIGH" if contra_score >= 2 else "MEDIUM"
    if omis_score > contra_score:
        return "OMISSION", "HIGH" if omis_score >= 2 else "MEDIUM"
    if contra_score > 0:
        return "CONTRADICTION", "LOW"
    if omis_score > 0:
        return "OMISSION", "LOW"
    return "NONE", "LOW"


# ============================================================
# VERDICT PARSING
# ============================================================

def parse_verdict(raw_output):
    """
    Parse comparison/verdict output via Qwen3-32B.

    Returns dict with:
      pick: A/B/ORIGINAL/CORRECTED/UNCLEAR
      reason: str
      parse_method: str
      validation: dict
    """
    q32_raw = _call_qwen32b(
        "Extract verdict. Output ONLY valid JSON.",
        VERDICT_EXTRACT_PROMPT.format(raw_output=raw_output[:2000]),
    )
    obj, method = _extract_json(q32_raw)

    result = {
        "pick": "UNCLEAR",
        "reason": "",
        "parse_method": method,
        "validation": {},
    }

    if not obj or not isinstance(obj, dict):
        result["validation"]["json_parsed"] = False
        return result

    result["validation"]["json_parsed"] = True

    pick = str(obj.get("pick", "UNCLEAR")).upper()
    valid_picks = {"A", "B", "ORIGINAL", "CORRECTED", "UNCLEAR"}
    result["validation"]["pick_valid"] = pick in valid_picks
    result["pick"] = pick if pick in valid_picks else "UNCLEAR"

    result["reason"] = str(obj.get("reason", ""))[:200]

    return result


# ============================================================
# BATCH VALIDATION REPORT
# ============================================================

def validation_report(parsed_items):
    """
    Generate a validation report for a list of parsed results.

    Args:
        parsed_items: list of dicts from parse_detection()

    Returns: dict with stats
    """
    n = len(parsed_items)
    if n == 0:
        return {"n": 0}

    report = {"n": n}

    # Parse success
    report["json_parsed"] = sum(1 for r in parsed_items if r["validation"].get("json_parsed", False))
    report["json_parsed_pct"] = report["json_parsed"] / n

    # Field validity
    for field in ["verdict_valid", "error_type_valid", "has_error_statement",
                   "has_correct_statement", "confidence_valid"]:
        count = sum(1 for r in parsed_items if r["validation"].get(field, False))
        report[field] = count
        report[f"{field}_pct"] = count / n

    # Parse methods
    methods = {}
    for r in parsed_items:
        m = r.get("parse_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    report["parse_methods"] = methods

    # Type inference
    report["type_inferred"] = sum(1 for r in parsed_items if r["validation"].get("type_inferred", False))

    # Cross-validation issues
    incorrect = [r for r in parsed_items if r["verdict"] == "INCORRECT"]
    if incorrect:
        report["incorrect_with_error"] = sum(1 for r in incorrect if r["validation"].get("has_error_for_incorrect", False))
        report["incorrect_with_notes"] = sum(1 for r in incorrect if r["validation"].get("has_notes_for_incorrect", False))
        report["incorrect_total"] = len(incorrect)

    return report
