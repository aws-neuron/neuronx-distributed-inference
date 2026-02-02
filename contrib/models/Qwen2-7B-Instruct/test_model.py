#!/usr/bin/env python3
"""
Test script for Qwen2-7B-Instruct
"""

import sys
from pathlib import Path

# Add validation framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "NeuroborosFoundations" / "model_validation"))

from validate_model import validate_model

def test_qwen2_7b_instruct():
    """Test Qwen2-7B-Instruct model"""
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False
    
    print(f"Testing Qwen2-7B-Instruct...")
    result = validate_model(str(config_path))
    
    if result:
        print(f"✓ Qwen2-7B-Instruct validation passed")
        return True
    else:
        print(f"✗ Qwen2-7B-Instruct validation failed")
        return False

if __name__ == "__main__":
    success = test_qwen2_7b_instruct()
    sys.exit(0 if success else 1)
