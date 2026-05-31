#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import requests


def served_model(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--served-model-contains", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    deadline = time.time() + args.timeout
    needle = args.served_model_contains.lower()
    while True:
        try:
            model = served_model(args.port)
            if not needle or needle in model.lower():
                print(model, flush=True)
                return 0
        except Exception:
            pass
        if args.once or time.time() >= deadline:
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
