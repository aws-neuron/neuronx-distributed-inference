#!/usr/bin/env python3
"""
Test script for Phi-3.5-mini-instruct
"""

import sys
from pathlib import Path

# Add validation framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "NeuroborosFoundations" / "model_validation"))

from validate_model import validate_model

def test_phi_3_5_mini_instruct():
    """Test Phi-3.5-mini-instruct model"""
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False
    
    print(f"Testing Phi-3.5-mini-instruct...")
    result = validate_model(str(config_path))
    
    if result:
        print(f"✓ Phi-3.5-mini-instruct validation passed")
        return True
    else:
        print(f"✗ Phi-3.5-mini-instruct validation failed")
        return False

if __name__ == "__main__":
    success = test_phi_3_5_mini_instruct()
    sys.exit(0 if success else 1)
