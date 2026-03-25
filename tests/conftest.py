"""
Pytest configuration for SentinelLM tests.

Adds the gateway directory to sys.path so that `from app.xxx import yyy`
works in tests.
"""

import sys
from pathlib import Path

# Add gateway/ to the Python path
gateway_dir = Path(__file__).parent.parent / "gateway"
sys.path.insert(0, str(gateway_dir))
