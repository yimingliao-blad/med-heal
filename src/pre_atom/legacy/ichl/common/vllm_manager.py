"""vLLM server lifecycle helpers — stop / start / wait-ready.

Authorised per MEMORY 'vLLM Model Swap Permission' (2026-04-22): Claude may
stop and start vLLM with a different model when the experiment needs it.

All models use port 8003. One model at a time (24GB GPU).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PORT = 8003
BASE_URL = f"http://localhost:{PORT}"


@dataclass
class VLLMLaunchSpec:
    internal_key: str              # 'qwen3-8b' etc. (used as log-file tag)
    model_dir: str                 # absolute dir under models/
    served_name: str               # HF id to report to clients
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85   # 0.9 \u2192 0.85 on 2026-04-27 to leave headroom for residual
                                           # GPU memory from sentence-transformers (gte-large) loaded earlier in the pipeline
    dtype: str = "bfloat16"


# Table matches MEMORY 'vLLM Model Swap Permission'.
TARGETS: dict[str, VLLMLaunchSpec] = {
    "qwen3-8b": VLLMLaunchSpec(
        internal_key="qwen3-8b",
        model_dir=str(PROJECT_ROOT / "models" / "qwen3-8b"),
        # Bumped to 32768 (native max) on 2026-04-28 for Error Location locator
        # which needs >16K thinking budget on tail items (Cell B Mode D / locator
        # v2 hit 11% truncation at 14000 max_tokens because thinking exceeded budget).
        # Required gpu_memory_utilization 0.85 → 0.90 (vLLM hint: at 0.85 only 3.99
        # GiB KV cache available, needs 4.5 GiB for 32K context per single sequence).
        served_name="Qwen/Qwen3-8B", max_model_len=32768,
        gpu_memory_utilization=0.90,
    ),
    "qwen2.5-7b-instruct": VLLMLaunchSpec(
        internal_key="qwen2.5-7b-instruct",
        model_dir=str(PROJECT_ROOT / "models" / "qwen2.5-7b-instruct"),
        # Bumped from 8192 \u2192 16384 on 2026-04-27 to fit RA-ICL prompts with
        # FULL discharge notes (test note + ref note ~ 11-12K tokens), per
        # [Workflow] No Silent Truncation. Native limit is 32768 via YaRN.
        # Bumped 16384 → 32768 on 2026-04-30 (preemptive vLLM context audit):
        # native YaRN-extended limit is 32768; larger=safer per user directive.
        served_name="Qwen/Qwen2.5-7B-Instruct", max_model_len=32768,
    ),
    "biomistral-7b": VLLMLaunchSpec(
        internal_key="biomistral-7b",
        model_dir=str(PROJECT_ROOT / "models" / "biomistral-7b"),
        # Bumped 8192 → 32768 on 2026-04-30 (preemptive vLLM context audit):
        # config.json has max_position_embeddings=32768; 4K sliding window
        # (effective attention capped at 4K but positional supports 32K).
        # Larger=safer per user directive — avoids 400 rejections at high max_gen.
        served_name="BioMistral/BioMistral-7B", max_model_len=32768,
    ),
    "llama-3.1-8b-instruct": VLLMLaunchSpec(
        internal_key="llama-3.1-8b-instruct",
        model_dir=str(PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct"),
        # Bumped 8192 \u2192 16384 on 2026-04-27 (truncation audit) to fit RA-ICL
        # Mode A (full test note + full ref note ~ 14K tokens). Native is 131072 (128K).
        # 16384 → 32768 on 2026-04-30 (preemptive context audit, GPU-bounded):
        # native is 131072 but 24GB KV cache caps practical at ~36K; 32K matches Qwen3.
        served_name="meta-llama/Llama-3.1-8B-Instruct", max_model_len=32768,
    ),
    "magistral-small-2509-awq": VLLMLaunchSpec(
        internal_key="magistral-small-2509-awq",
        model_dir=str(PROJECT_ROOT / "models" / "magistral-small-2509-awq"),
        # Reduced 32768 → 24576 on 2026-04-27: at gpu_memory_utilization=0.85 the
        # estimated max KV cache fits ~26624 tokens; leaving margin for safety.
        served_name="Magistral-Small-2509-AWQ", max_model_len=24576,
    ),
    "deepseek-r1-distill-llama-8b": VLLMLaunchSpec(
        internal_key="deepseek-r1-distill-llama-8b",
        model_dir=str(PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B"),
        # 32768 stays on 2026-04-30 (GPU-bounded): tried 65536 but vLLM
        # required 8 GiB KV cache vs 4.36 available. Estimated max_model_len ≈ 35728.
        # 32K is GPU-safe ceiling; per-V max_gen sized within this budget.
        served_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", max_model_len=32768,
    ),
    "mistral-7b-instruct-v0.3": VLLMLaunchSpec(
        internal_key="mistral-7b-instruct-v0.3",
        model_dir=str(PROJECT_ROOT / "models" / "Mistral-7B-Instruct-v0.3"),
        # Native 32768; matches the existing 7B model conventions.
        served_name="mistralai/Mistral-7B-Instruct-v0.3", max_model_len=32768,
    ),
    "llama-3.2-3b-instruct": VLLMLaunchSpec(
        internal_key="llama-3.2-3b-instruct",
        model_dir=str(PROJECT_ROOT / "models" / "Llama-3.2-3B-Instruct"),
        # Native 131072 (128K); cap at 32K for parity with other targets.
        served_name="meta-llama/Llama-3.2-3B-Instruct", max_model_len=32768,
    ),
}


# ─────────────────────── queries ───────────────────────

def is_ready(timeout: float = 3.0) -> tuple[bool, str | None]:
    """Return (ready, served_model_name). served_model_name is None if not ready."""
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=timeout)
        if r.status_code == 200:
            body = r.json()
            data = body.get("data") or []
            if data:
                return True, data[0].get("id")
            return True, None
        return False, None
    except Exception:
        return False, None


def currently_served() -> str | None:
    ok, name = is_ready()
    return name if ok else None


# ─────────────────────── lifecycle ───────────────────────

def _compute_apps_pids() -> list[int]:
    """Return list of PIDs currently holding the GPU (via nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, timeout=10,
        )
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def stop() -> None:
    """Stop the running vLLM server cleanly — pkill then force-kill remaining EngineCore.

    Pattern is `vllm.entrypoints.openai.api_server` (the full module path) to avoid
    matching shells whose command line happens to contain the substring "vllm"
    (e.g., when launched from a Bash tool wrapper that embeds the script args).
    Origin: 2026-04-27 incident where `pkill -f vllm` killed the calling shell.
    """
    # 1. pkill parent process — match the FULL vLLM module path, not just "vllm"
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(3)
    # 2. force-kill any remaining compute-apps PIDs (EngineCore) — but NEVER kill our own PID,
    # since the calling script may itself hold GPU memory (e.g., from sentence-transformers / nomic).
    # Origin: 2026-04-27 incident where stop() force-killed its own caller.
    self_pid = os.getpid()
    parent_pid = os.getppid()
    for _ in range(4):
        pids = [p for p in _compute_apps_pids() if p not in (self_pid, parent_pid)]
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        time.sleep(2)
    # 3. final confirm
    remaining = _compute_apps_pids()
    if remaining:
        print(f"[vllm_manager] WARNING: GPU still held by PIDs {remaining}")


def start(spec: VLLMLaunchSpec, log_dir: Path | None = None) -> int:
    """Launch vLLM under nohup and return the PID of the parent Python process."""
    log_file = (log_dir or Path("/tmp")) / f"vllm_{spec.internal_key}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec.model_dir,
        "--served-model-name", spec.served_name,
        "--port", str(PORT),
        "--max-model-len", str(spec.max_model_len),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--dtype", spec.dtype,
    ]
    # Launch detached.
    with open(log_file, "a") as lf:
        lf.write(f"\n\n=== vllm_manager.start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        lf.write(f"cmd: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
    return proc.pid


def wait_ready(expected_name: str | None = None, max_wait_s: int = 600, poll_s: float = 2.0) -> bool:
    """Poll /v1/models until ready. Returns True on success, False on timeout."""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        ok, name = is_ready()
        if ok:
            if expected_name is None or (name or "").lower() == expected_name.lower():
                return True
        time.sleep(poll_s)
    return False


def ensure_model(key: str, log_dir: Path | None = None) -> VLLMLaunchSpec:
    """Ensure vLLM on port 8003 serves model `key`. Swap if necessary.

    Raises on unknown key or startup failure. Returns the spec that's now serving.
    """
    if key not in TARGETS:
        raise KeyError(f"Unknown target key '{key}'. Known: {list(TARGETS)}")
    spec = TARGETS[key]

    current = currently_served()
    if current and current.lower() == spec.served_name.lower():
        return spec    # already serving

    if current:
        print(f"[vllm_manager] swapping: {current} → {spec.served_name}")
        stop()
    else:
        print(f"[vllm_manager] starting fresh: → {spec.served_name}")

    pid = start(spec, log_dir=log_dir)
    print(f"[vllm_manager] launched PID {pid} (log: /tmp/vllm_{spec.internal_key}.log)")
    if not wait_ready(expected_name=spec.served_name, max_wait_s=300):
        raise RuntimeError(
            f"vLLM did not become ready within 600s for {spec.served_name}. "
            f"Check /tmp/vllm_{spec.internal_key}.log"
        )
    print(f"[vllm_manager] ready: {spec.served_name}")
    return spec
