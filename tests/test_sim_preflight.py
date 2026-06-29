"""Local-only guard tests (pure, offline)."""

from __future__ import annotations

import pytest

from sentisense.sim.miro_client import MiroError
from sentisense.sim.preflight import assert_local, is_loopback


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:5001", True),
    ("http://127.0.0.1:5001", True),
    ("http://[::1]:5001", True),
    ("http://0.0.0.0:5001", True),
    ("https://api.getzep.com", False),
    ("http://10.0.0.5:5001", False),
])
def test_is_loopback(url, expected):
    assert is_loopback(url) is expected


def test_assert_local_allows_loopback():
    assert_local("http://localhost:5001", allow_remote=False)   # no raise


def test_assert_local_blocks_remote_by_default():
    with pytest.raises(MiroError, match="not loopback"):
        assert_local("http://10.0.0.5:5001", allow_remote=False)


def test_assert_local_remote_opt_in():
    assert_local("http://10.0.0.5:5001", allow_remote=True)   # explicit opt-in → no raise
