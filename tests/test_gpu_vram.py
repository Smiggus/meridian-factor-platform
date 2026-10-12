"""Tests for the #296 VRAM instrumentation.

The point of these is narrow but load-bearing: `log_vram_peak` runs inside the
CUDA teardown of two nightly assets, so a bug in it would fail scoring runs to
report a number nobody is currently reading. It must never raise.
"""

import sys
import types

import pytest
from meridian_pipeline.gpu_vram import LOW_HEADROOM_BYTES, log_vram_peak

_GIB = 1024**3


class _Log:
    def __init__(self):
        self.info_msgs = []
        self.warning_msgs = []

    def info(self, m):
        self.info_msgs.append(m)

    def warning(self, m):
        self.warning_msgs.append(m)


def _fake_torch(*, available=True, peak=1.0, reserved=1.5, free=4.0, total=6.0, boom=None):
    """A torch stand-in; figures in GiB. `boom` makes mem_get_info raise."""
    mod = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        max_memory_allocated=lambda: int(peak * _GIB),
        max_memory_reserved=lambda: int(reserved * _GIB),
        mem_get_info=(
            (lambda: (_ for _ in ()).throw(boom))
            if boom
            else (lambda: (int(free * _GIB), int(total * _GIB)))
        ),
        get_device_name=lambda i: "Example GPU",
    )
    mod.cuda = cuda
    return mod


@pytest.fixture
def install_torch(monkeypatch):
    def _install(mod):
        monkeypatch.setitem(sys.modules, "torch", mod)

    return _install


def test_returns_none_when_no_cuda(install_torch):
    install_torch(_fake_torch(available=False))
    log = _Log()
    assert log_vram_peak(log, "x") is None
    assert log.info_msgs == []


def test_reports_peak_and_headroom(install_torch):
    install_torch(_fake_torch(peak=1.25, reserved=1.75, free=4.0, total=6.0))
    log = _Log()
    stats = log_vram_peak(log, "gpu_asset")
    assert stats["peak_allocated_gib"] == pytest.approx(1.25)
    assert stats["free_gib"] == pytest.approx(4.0)
    assert "gpu_asset" in log.info_msgs[0]
    # Comfortable headroom must not cry wolf, or the warning stops being read.
    assert log.warning_msgs == []


def test_warns_when_headroom_is_thin(install_torch):
    """The thin-free-VRAM case the pin has to stay clear of."""
    thin = (LOW_HEADROOM_BYTES / _GIB) * 0.5
    install_torch(_fake_torch(peak=1.8, free=thin, total=6.0))
    log = _Log()
    log_vram_peak(log, "risk_metrics")
    assert len(log.warning_msgs) == 1
    assert "296" in log.warning_msgs[0]


def test_never_raises_when_torch_misbehaves(install_torch):
    install_torch(_fake_torch(boom=RuntimeError("CUDA driver shutting down")))
    log = _Log()
    assert log_vram_peak(log, "x") is None
    assert log.warning_msgs, "a swallowed failure must still be visible in the log"


def test_absent_torch_is_silent(monkeypatch):
    """CI runs a partial dependency set; that is not a fault worth warning on."""
    monkeypatch.setitem(sys.modules, "torch", None)
    log = _Log()
    # `import torch` with a None entry raises ImportError, the silent path.
    assert log_vram_peak(log, "x") is None
    assert log.warning_msgs == []
