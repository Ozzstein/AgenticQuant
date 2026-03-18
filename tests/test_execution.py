"""Umbrella test module for TASK-05 Execution Layer.

Imports and re-exports all WU test modules so that ``pytest tests/test_execution.py``
runs the full execution-layer test suite in one command.
"""

from __future__ import annotations

# WU-1: Paper Trader
from tests.test_paper_trader import *  # noqa: F401, F403

# WU-2: Risk Controls
from tests.test_risk_controls import *  # noqa: F401, F403

# WU-3: Signal Bridge
from tests.test_signal_bridge import *  # noqa: F401, F403

# WU-4: Portfolio Optimizer
from tests.test_portfolio_optimizer import *  # noqa: F401, F403

# WU-5: Crypto Pipeline
from tests.test_crypto_pipeline import *  # noqa: F401, F403

# WU-6: Crypto Factors
from tests.test_crypto_factors import *  # noqa: F401, F403

# WU-7: Unified Data Layer
from tests.test_unified_data import *  # noqa: F401, F403
