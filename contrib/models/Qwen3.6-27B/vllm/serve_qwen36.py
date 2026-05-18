#!/usr/bin/env python3
"""vLLM CLI wrapper that registers Qwen3.6 aliases before validation."""

from __future__ import annotations

import sys

from hf_qwen35_config import register_qwen35_config


def main() -> int:
    register_qwen35_config()

    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    return int(vllm_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
