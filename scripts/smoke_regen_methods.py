#!/usr/bin/env python3
"""Smoke-test regen method wiring without external model/API calls.

This intentionally monkeypatches vLLM, Qwen parser, and GPT judge calls. The
goal is not to measure quality; it checks that each regen method imports, builds
prompts, parses expected outputs, and writes audit-shaped records.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "src" / "pre_atom" / "legacy"
V2_ROOT = LEGACY_ROOT / "step9_self_correction" / "v2"
os.environ.setdefault("PRE_ATOM_SOURCE_REPO_ROOT", str(LEGACY_ROOT))
os.environ.setdefault("PRE_ATOM_PROJECT_ROOT", str(ROOT))
sys.path.insert(0, str(LEGACY_ROOT))
sys.path.insert(0, str(V2_ROOT))


def _fake_item() -> dict:
    return {
        "fold": 0,
        "idx": 1,
        "patient_id": 123,
        "question": "What medication was changed at discharge?",
        "ground_truth": "Aspirin was stopped.",
        "model_answer": "Aspirin was continued.",
        "label": 0,
    }


def _fake_note() -> str:
    return 'The discharge medication plan states, "Aspirin was stopped."'


def smoke_step9_regen_count() -> dict:
    import regen_pilot
    from audit_log import AuditLog

    def fake_vllm_chat(system: str, user: str, port: int, **kwargs) -> str:
        if "ANSWER A:" in user:
            return "A_ERRORS: 0\nB_ERRORS: 1"
        return "Aspirin was stopped."

    regen_pilot.vllm_chat = fake_vllm_chat
    regen_pilot.qwen3_parse_decision = lambda analysis: ("A", "A has fewer contradictions")
    regen_pilot.judge_call = lambda *a, **k: {"label": 1, "raws": ["1"]}

    with tempfile.TemporaryDirectory(prefix="regen_count_") as td:
        log = AuditLog(Path(td) / "audit.jsonl")
        args = SimpleNamespace(force=True)
        regen_pilot.run_one(_fake_item(), {"123": _fake_note()}, port=8003, args=args, log=log)
        rec = log.get(0, 1)
    assert rec is not None
    assert rec["correction"]["method"] == "regen_zeroshot"
    assert rec["verdict"]["variant"] == "count_compare_qwen3parse"
    assert rec["outcome"]["action"] == "corrected"
    return {"method": rec["correction"]["method"], "action": rec["outcome"]["action"]}


def smoke_step9_regen_v3() -> dict:
    import regen_v3_pilot
    from audit_log import AuditLog

    calls = {"regen": 0, "critique": 0, "verdict": 0}

    def fake_vllm_chat(system: str, user: str, port: int, **kwargs) -> str:
        if "ANSWER A:" in user and "A_CONTRADICTIONS" in user:
            calls["verdict"] += 1
            return (
                "A_CONTRADICTIONS: 0\n"
                "A_UNADDRESSED: 0\n"
                "A_UNSUPPORTED: 0\n"
                "B_CONTRADICTIONS: 1\n"
                "B_UNADDRESSED: 0\n"
                "B_UNSUPPORTED: 0\n"
                "WINNER: A\n"
            )
        if "UNSUPPORTED_CLAIMS" in user:
            calls["critique"] += 1
            return "UNSUPPORTED_CLAIMS:\nNONE"
        calls["regen"] += 1
        return (
            "EVIDENCE:\n"
            '- "Aspirin was stopped."\n\n'
            "ANSWER:\n"
            "Aspirin was stopped."
        )

    regen_v3_pilot.vllm_chat = fake_vllm_chat
    regen_v3_pilot._q32_fallback_winner = lambda analysis: "A"
    regen_v3_pilot.judge_call = lambda *a, **k: {"label": 0, "raws": ["0"]}

    with tempfile.TemporaryDirectory(prefix="regen_v3_") as td:
        log = AuditLog(Path(td) / "audit.jsonl")
        args = SimpleNamespace(force=True)
        regen_v3_pilot.run_one(_fake_item(), {"123": _fake_note()}, port=8003, args=args, log=log)
        rec = log.get(0, 1)
    assert rec is not None
    assert rec["correction"]["method"] == "cove_2round_regen_v3"
    assert rec["verdict"]["variant"] == "verdict_v3_3count_dualparse"
    assert rec["verdict"]["regex_full_parse"] is True
    assert calls["regen"] >= 1 and calls["critique"] == 1 and calls["verdict"] == 1
    return {
        "method": rec["correction"]["method"],
        "verdict_source": rec["verdict"]["final_source"],
        "action": rec["outcome"]["action"],
    }


def smoke_legacy_subvariants() -> dict:
    from ichl.prompt_engineering.correction.runner import run_correction_one_item
    from ichl.prompt_engineering.correction.sub_variants import SUB_VARIANTS

    class FakeResp:
        success = True
        error = None
        raw_text = "Corrected answer: Aspirin was stopped."
        text = "Corrected answer: Aspirin was stopped."
        finish_reason = "stop"
        usage = {"completion_tokens": 8, "prompt_tokens": 50, "total_tokens": 58}
        latency = 0.001
        client = "fake"

    class FakeClient:
        def call(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeResp()

    item = {
        "pilot_item_id": "smoke-0-1",
        "patient_id": 123,
        "fold": 0,
        "note": _fake_note(),
        "question": "What medication was changed at discharge?",
        "A0": "Aspirin was continued.",
        "A0_binary_correct": 0,
    }
    seen = []
    for sv in sorted(SUB_VARIANTS):
        rec = run_correction_one_item(
            client=FakeClient(),
            target="qwen2.5-7b-instruct",
            sub_variant_id=sv,
            item=item,
            max_tokens=128,
            temperature=0.0,
            raw_log_dir=None,
        )
        assert rec["sub_variant_id"] == sv
        assert rec["text"]
        assert rec["truncation_report"]["is_truncated_certain"] is False
        seen.append(f"{sv}:{rec['sub_variant_name']}")

    qwen3_rec = run_correction_one_item(
        client=FakeClient(),
        target="qwen3-8b",
        sub_variant_id="a",
        item=item,
        max_tokens=128,
        temperature=0.0,
        raw_log_dir=None,
    )
    assert qwen3_rec["enable_thinking"] is False
    return {"subvariants": seen, "qwen3_enable_thinking": qwen3_rec["enable_thinking"]}


def main() -> int:
    results = {
        "step9_regen_count": smoke_step9_regen_count(),
        "step9_regen_v3": smoke_step9_regen_v3(),
        "legacy_t0_subvariants": smoke_legacy_subvariants(),
    }
    print(json.dumps({"ok": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
