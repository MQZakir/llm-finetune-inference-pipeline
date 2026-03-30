"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work without install
sys.path.insert(0, str(Path(__file__).parent))
