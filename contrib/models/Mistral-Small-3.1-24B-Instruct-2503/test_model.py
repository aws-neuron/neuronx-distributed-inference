#!/usr/bin/env python3
"""
Test script for Mistral-Small-3.1-24B-Instruct-2503
"""

import sys
from pathlib import Path

# Add validation framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "NeuroborosFoundations" / "model_validation"))

from validate_model import validate_model

def test_mistral_small_3_1_24b_instruct_2503():
    """Test Mistral-Small-3.1-24B-Instruct-2503 model"""
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False
    
    print(f"Testing Mistral-Small-3.1-24B-Instruct-2503...")
    result = validate_model(str(config_path))
    
    if result:
        print(f"✓ Mistral-Small-3.1-24B-Instruct-2503 validation passed")
        return True
    else:
        print(f"✗ Mistral-Small-3.1-24B-Instruct-2503 validation failed")
        return False

if __name__ == "__main__":
    success = test_mistral_small_3_1_24b_instruct_2503()
    sys.exit(0 if success else 1)
