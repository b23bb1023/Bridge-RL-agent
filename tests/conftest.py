import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridgepulse.config import load_env_config  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402


@pytest.fixture
def env_factory():
    def make(**overrides):
        cfg = load_env_config()
        if cfg["budget"]["annual"] is None:
            overrides.setdefault("budget", {"annual": 3.0e6})
        return BridgeNetworkEnv(cfg, overrides=overrides)

    return make
