"""Make the repository root importable so `pytest` works from a clean checkout.

The package is not installed (there is no build backend); the scripts inject the
repo root into sys.path themselves, and this does the same for the test suite.
Without it, `import materialgen` fails under a bare `pytest` run, because pytest
only puts the test file's own directory on sys.path.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
