"""
Runner script for MCTS unit tests.

Mocks the C++ engine module before importing alphazero,
so tests can run without building the compiled engine extension.
"""
import sys
import os
import types
from unittest.mock import MagicMock

# Ensure alphazero package is findable
_src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
# Also add the current directory so relative imports within alphazero work
os.chdir(_src_root)

# 1. Mock engine module BEFORE anything imports alphazero
engine = types.ModuleType("engine")
engine.vec4 = MagicMock()
sys.modules["engine"] = engine

# 2. Now run the tests
import unittest

test_module = "alphazero.test_mcts_unit"
suite = unittest.defaultTestLoader.loadTestsFromModule(
    __import__(test_module, fromlist=["test_mcts_unit"])
)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)