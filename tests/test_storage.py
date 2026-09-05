"""Append-only run log and resume behaviour."""

from agent_bluff.records import MatchRecord
from agent_bluff.storage import MatchLog


def _match(seed, error=None):
    return MatchRecord(
        seed=seed, informant_model="a", chooser_model="b", match_index=0, rounds=[],
        config={}, started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:01:00Z",
        error=error,
    )


async def test_roundtrip(tmp_path):
    log = MatchLog(tmp_path)
    await log.append(_match("a|b|0"))
    await log.append(_match("a|b|1"))
    assert [m.seed for m in log.read()] == ["a|b|0", "a|b|1"]


async def test_errored_matches_are_retried_on_resume(tmp_path):
    """A partial match should be replayed, not silently accepted as done."""
    log = MatchLog(tmp_path)
    await log.append(_match("a|b|0"))
    await log.append(_match("a|b|1", error="PlayerRefused: no"))
    assert log.completed_seeds() == {"a|b|0"}


def test_reading_a_fresh_directory_yields_nothing(tmp_path):
    assert list(MatchLog(tmp_path / "new").read()) == []


async def test_concurrent_appends_do_not_interleave(tmp_path):
    import asyncio

    log = MatchLog(tmp_path)
    await asyncio.gather(*(log.append(_match(f"a|b|{i}")) for i in range(50)))
    assert len(list(log.read())) == 50
