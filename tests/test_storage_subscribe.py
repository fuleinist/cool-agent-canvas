"""Tests for InMemoryStore.subscribe() sid allocation under concurrency.

Before the fix, subscribe() incremented _sub_counter outside the lock,
so two concurrent calls could read the same value and return duplicate
sids — corrupting the (run_id, sid) -> queue map.
"""

from __future__ import annotations

import threading

from agent_canvas.storage import InMemoryStore


def test_subscribe_returns_unique_sids_under_concurrency() -> None:
    """N concurrent subscribe() calls must produce N distinct sids.

    On the fixed code, the increment-and-append is a single critical
    section, so this holds by construction. On the unfixed code, two
    threads that both read ``_sub_counter`` before either writes back
    can collide; the barrier maximises the chance of catching that.
    """
    store = InMemoryStore()
    n_threads = 200
    sids: list[int] = [0] * n_threads
    barrier = threading.Barrier(n_threads)

    def worker(i: int) -> None:
        barrier.wait()
        sid, _q = store.subscribe("run_x")
        sids[i] = sid

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(sids)) == n_threads, (
        f"expected {n_threads} distinct sids, got {len(set(sids))} (collisions in {sorted(sids)})"
    )


def test_subscribe_then_unsubscribe_removes_only_target_subscriber() -> None:
    """A sid returned by subscribe() must map to exactly one queue."""
    store = InMemoryStore()
    sid_a, _ = store.subscribe("run_y")
    sid_b, _ = store.subscribe("run_y")

    assert sid_a != sid_b

    store.unsubscribe("run_y", sid_a)

    # After unsubscribing sid_a, sid_b's queue should still be reachable.
    _, q_b = store._subscribe_refs("run_y", sid_b)
    assert q_b is not None

    # sid_a should now be gone.
    try:
        store._subscribe_refs("run_y", sid_a)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unsubscribed sid_a")