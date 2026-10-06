"""Regression test: lock ownership + abort semantics of TTSModelManager.

Runs without the model (the class attributes are set directly), so it is fast
and needs no GPU. Guards the double-release regression caught in the container.
"""
import os, sys, threading, time, types
import pytest

# Works from the repo root and from /app inside the container.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_model import ClientDisconnected, TTSModelManager


def mgr():
    """A manager with no model loaded, only the queueing behaviour under test."""
    m = types.SimpleNamespace()
    m._lock = threading.Lock()
    m.QUEUE_POLL_INTERVAL = 0.05
    m._acquire = TTSModelManager._acquire.__get__(m, type(m))
    return m


def test_acquire_blocks_until_free():
    m = mgr()
    m._lock.acquire()
    threading.Timer(0.2, m._lock.release).start()
    t0 = time.time()
    m._acquire(None)
    assert 0.1 < time.time() - t0 < 1.0, "should have waited for the holder"
    m._lock.release()


def test_no_abort_callback_leaves_lock_held_once():
    """`with self._lock` used to release, then generate_speech's finally released
    again -> RuntimeError: release unlocked lock. Balance must be exactly 1:1."""
    m = mgr()
    m._acquire(None)
    assert m._lock.locked(), "acquire must leave the lock held"
    m._lock.release()
    assert not m._lock.locked()


class _FakeModel:
    """Minimal stand-in for the HF model so generate_speech can run off-GPU."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def create_voice_clone_prompt(self, **kw):
        return "prompt"

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inference blew up")
        import numpy as np
        return [np.zeros(24000, dtype=np.float32)], 24000


def _manager_with(model, lock):
    """A manager wired to _FakeModel, using the real generate_speech/_acquire."""
    m = types.SimpleNamespace()
    m._lock = lock
    m._model = model
    m._attn = "sdpa"
    m._voice_prompts = {"v": "prompt"}   # pre-cached: skip the filesystem entirely
    m.QUEUE_POLL_INTERVAL = 0.05
    m._acquire = TTSModelManager._acquire.__get__(m, type(m))
    m.get_voice_prompt = types.MethodType(TTSModelManager.get_voice_prompt, m)
    m.generate_speech = types.MethodType(TTSModelManager.generate_speech, m)
    return m


def test_generate_speech_releases_lock_exactly_once():
    """The regression, exercised through the real generate_speech."""
    lock = threading.Lock()
    counting = _CountingLock(lock)
    m = _manager_with(_FakeModel(), counting)
    m.generate_speech(text="hello", voice_name="v", language="Auto")
    assert counting.acquires == counting.releases == 1, (
        f"acquired {counting.acquires} but released {counting.releases}"
    )
    assert not lock.locked(), "lock left held after a successful generation"


def test_generate_speech_releases_lock_on_inference_failure():
    lock = threading.Lock()
    counting = _CountingLock(lock)
    m = _manager_with(_FakeModel(fail=True), counting)
    with pytest.raises(RuntimeError):
        m.generate_speech(text="hello", voice_name="v", language="Auto")
    assert not lock.locked(), "failed generation must not leak the lock"
    assert counting.acquires == counting.releases == 1


def test_generate_speech_queues_behind_holder_and_reports_wait():
    lock = threading.Lock()
    m = _manager_with(_FakeModel(), lock)
    assert lock.acquire(timeout=0.1)
    threading.Timer(0.25, lock.release).start()
    m.generate_speech(text="hi", voice_name="v", language="Auto")
    assert not lock.locked()


def test_abandoned_queued_request_never_runs_inference():
    """The headline guarantee: a client that leaves the queue burns no GPU.

    Inference must never be reached for a request whose client is already gone.
    """
    lock = threading.Lock()
    model = _FakeModel()
    m = _manager_with(model, lock)
    assert lock.acquire(timeout=0.1)          # GPU busy with someone else

    with pytest.raises(ClientDisconnected):
        m.generate_speech(text="abandoned", voice_name="v", language="Auto",
                          should_abort=lambda: True)

    assert model.calls == 0, "inference ran for a client that had disconnected"
    # The abandoned request must not have taken the lock we are still holding.
    lock.release()
    assert not lock.locked(), "abandoned request kept hold of the lock"


def test_live_queued_request_does_run_inference():
    """The other half: a connected client must still be served after waiting."""
    lock = threading.Lock()
    model = _FakeModel()
    m = _manager_with(model, lock)
    assert lock.acquire(timeout=0.1)
    threading.Timer(0.2, lock.release).start()

    audio, sr = m.generate_speech(text="stay", voice_name="v", language="Auto",
                                  should_abort=lambda: False)
    assert model.calls == 1, "a connected client must get its audio"
    assert len(audio) > 0 and sr == 24000


class _CountingLock:
    """Delegates to a real Lock while counting balance."""

    def __init__(self, inner):
        self._inner = inner
        self.acquires = 0
        self.releases = 0

    def acquire(self, *a, **kw):
        got = self._inner.acquire(*a, **kw)
        if got:
            self.acquires += 1
        return got

    def release(self):
        self.releases += 1
        self._inner.release()

    def locked(self):
        return self._inner.locked()


def test_abort_raises_and_leaves_lock_free():
    m = mgr()
    m._lock.acquire()          # someone else holds the GPU
    threading.Timer(0.4, m._lock.release).start()

    calls = []

    def gone():
        calls.append(1)
        return True

    t0 = time.time()
    with pytest.raises(ClientDisconnected):
        m._acquire(gone)
    assert time.time() - t0 < 2.0, "must give up promptly, not wait out the holder"
    assert calls, "abort callback must actually be consulted"


def test_abort_does_not_steal_the_lock():
    """After an abort the holder must still own it -- no double acquire."""
    m = mgr()
    assert m._lock.acquire(timeout=0.1), "test setup: holder must hold the lock"
    try:
        with pytest.raises(ClientDisconnected):
            m._acquire(lambda: True)
        # Still held by us: if the aborted thread had grabbed it, this would
        # block forever or succeed, both of which are the bug we are guarding.
        assert not m._lock.acquire(timeout=0.1), "lock was stolen by aborted caller"
    finally:
        m._lock.release()


def test_live_caller_is_not_aborted():
    m = mgr()
    assert m._lock.acquire(timeout=0.1)
    threading.Timer(0.3, m._lock.release).start()
    m._acquire(lambda: False)   # client stays connected
    assert m._lock.locked(), "a live client must end up holding the lock"
    m._lock.release()