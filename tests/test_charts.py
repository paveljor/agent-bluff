"""Figure rendering.

A smoke test only: it asserts the figure builds and writes both themes without
raising. Whether a chart *reads* well is checked by looking at it, not by an
assertion, so this guards against crashes and regressions in the data path.
"""

import pytest

from agent_bluff.game import Claim, Role
from agent_bluff.records import CallRecord, MatchRecord, RoundRecord

matplotlib = pytest.importorskip("matplotlib")


def _match(rounds_n=10):
    rounds = [
        RoundRecord(
            index=i, informant_has_reward=i % 2 == 0, informant_open="o",
            open_claim=Claim.EMPTY, chooser_reply="r", informant_close="c",
            close_claim=Claim.EMPTY, switched=i % 3 == 0,
            confidence_informant_has_reward=0.5, chooser_reasoning="why",
            informant_lied=i % 2 == 0, informant_won=True,
            calls=[CallRecord(role=Role.INFORMANT, model="m", kind="informant_open",
                              prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
                              latency_s=0.1)],
        )
        for i in range(rounds_n)
    ]
    return MatchRecord(
        seed="a|b|0", informant_model="a", chooser_model="b", match_index=0,
        rounds=rounds, config={}, started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
    )


def test_writes_both_themes(tmp_path):
    from agent_bluff.charts import write_lie_rate

    paths = write_lie_rate([_match()], tmp_path)

    assert {p.name for p in paths} == {
        "lie-rate-by-round-light.png", "lie-rate-by-round-dark.png"
    }
    assert all(p.stat().st_size > 5000 for p in paths)


def test_dark_theme_is_its_own_steps_not_an_inverted_light_theme():
    """The skill requires dark to be selected against the dark surface."""
    from agent_bluff.charts import THEMES

    assert THEMES["dark"]["surface"] != THEMES["light"]["surface"]
    assert THEMES["dark"]["accent"] != THEMES["light"]["accent"]
