#!/usr/bin/env python3
"""
Test script for Qwen3-Coder-30B-A3B-Instruct
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeuroborosFoundations" / "model_validation"))

from validate_model import validate_model


def test_qwen3_coder_30b():
    """Test Qwen3-Coder-30B-A3B-Instruct model"""
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False

    print("Testing Qwen3-Coder-30B-A3B-Instruct...")
    result = validate_model(str(config_path))

    if result:
        print("✓ Qwen3-Coder-30B-A3B-Instruct validation passed")
        return True
    else:
        print("✗ Qwen3-Coder-30B-A3B-Instruct validation failed")
        return False


if __name__ == "__main__":
    success = test_qwen3_coder_30b()
    sys.exit(0 if success else 1)
