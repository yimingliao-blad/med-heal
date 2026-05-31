from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."


def load_api_key() -> str:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        env = SOURCE_REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found")


_client: OpenAI | None = None


def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_api_key())
    return _client


def build_judge_user(note: str, question: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        "Respond with ONLY a single digit:\n"
        "1 = Correct\n"
        "0 = Incorrect"
    )


def parse_binary(text: str | None) -> int | None:
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


def gpt4o_stage1_judge(note: str, question: str, ground_truth: str, model_answer: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_user(note, question, ground_truth, model_answer)},
    ]
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=10,
            )
            raw = (r.choices[0].message.content or "").strip()
            return {"label": parse_binary(raw), "raw": raw, "model": "gpt-4o", "temperature": 0.1}
        except Exception as e:
            if attempt == 4:
                return {"label": None, "raw": "", "error": str(e), "model": "gpt-4o", "temperature": 0.1}
            time.sleep(2 * (attempt + 1))
    return {"label": None}


def gpt4omini_extract_json(system: str, user: str, max_tokens: int = 500) -> dict[str, Any]:
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads((r.choices[0].message.content or "{}").strip())
        except Exception as e:
            if attempt == 4:
                return {"error": str(e)}
            time.sleep(2 * (attempt + 1))
    return {}


def served_model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.S | re.I).strip()
    return text


def vllm_chat(system: str, user: str, *, port: int, max_tokens: int, temperature: float) -> str:
    model = served_model_id(port)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    body = r.json()
    if "choices" not in body:
        payload["messages"] = [{"role": "user", "content": f"{system}\n\n{user}"}]
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
        body = r.json()
    if "choices" not in body:
        raise RuntimeError(str(body))
    return strip_think((body["choices"][0]["message"]["content"] or "").strip())
