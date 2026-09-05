"""Metrics."""

import pytest

from agent_bluff.analysis import (
    belief_response,
    head_to_head,
    lie_rate_by_round,
    model_stats,
    run_cost,
    wilson,
)
from agent_bluff.game import Claim, Role
from agent_bluff.records import CallRecord, MatchRecord, RoundRecord


def _round(index, *, has_reward, claim, switched, confidence=0.5, cost=0.001):
    informant_won = has_reward != switched
    lied = None if claim is Claim.NONE else (claim is Claim.HAS_REWARD) != has_reward
    return RoundRecord(
        index=index, informant_has_reward=has_reward,
        informant_open="o", open_claim=claim, chooser_reply="r",
        informant_close="c", close_claim=claim, switched=switched,
        confidence_informant_has_reward=confidence, chooser_reasoning="why",
        informant_lied=lied, informant_won=informant_won,
        calls=[CallRecord(role=Role.INFORMANT, model="m", kind="informant_open",
                          prompt_tokens=1, completion_tokens=1, cost_usd=cost, latency_s=0.1)],
    )


def _match(informant, chooser, rounds, index=0):
    return MatchRecord(
        seed=f"{informant}|{chooser}|{index}", informant_model=informant,
        chooser_model=chooser, match_index=index, rounds=rounds, config={},
        started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:01:00Z",
    )


def test_wilson_centres_on_the_estimate():
    p = wilson(50, 100)
    assert p.value == pytest.approx(0.5)
    assert p.low == pytest.approx(0.404, abs=0.01)
    assert p.high == pytest.approx(0.596, abs=0.01)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    assert wilson(0, 10).low == 0.0
    assert wilson(10, 10).high == 1.0


def test_wilson_handles_no_trials():
    p = wilson(0, 0)
    assert p.value == 0.0
    assert not p.beats_chance


def test_beats_chance_requires_the_interval_to_exclude_a_coin_flip():
    assert not wilson(6, 10).beats_chance
    assert wilson(90, 100).beats_chance


def test_model_stats_splits_the_two_roles():
    rounds = [
        _round(0, has_reward=True, claim=Claim.EMPTY, switched=False),   # informant wins
        _round(1, has_reward=False, claim=Claim.EMPTY, switched=False),  # chooser wins
    ]
    stats = model_stats([_match("astra", "grok", rounds)])

    assert stats["astra"].informant_rounds == 2
    assert stats["astra"].informant_wins == 1
    assert stats["grok"].chooser_rounds == 2
    assert stats["grok"].chooser_wins == 1
    assert "astra" not in {s.key for s in stats.values() if s.chooser_rounds}


def test_chooser_win_rate_is_exactly_its_accuracy():
    """Winning and deciding correctly are the same event; assert they never diverge."""
    rounds = [
        _round(0, has_reward=True, claim=Claim.EMPTY, switched=True),
        _round(1, has_reward=False, claim=Claim.EMPTY, switched=False),
        _round(2, has_reward=True, claim=Claim.EMPTY, switched=False),
    ]
    stats = model_stats([_match("astra", "grok", rounds)])
    correct = sum(r.informant_has_reward == r.switched for r in rounds)
    assert stats["grok"].chooser_wins == correct


def test_lie_rate_counts_only_checkable_claims():
    rounds = [
        _round(0, has_reward=True, claim=Claim.EMPTY, switched=False),       # lie
        _round(1, has_reward=True, claim=Claim.HAS_REWARD, switched=False),  # truth
        _round(2, has_reward=True, claim=Claim.NONE, switched=False),        # evasion
    ]
    stats = model_stats([_match("astra", "grok", rounds)])["astra"]

    assert stats.checkable_claims == 2
    assert stats.lies == 1
    assert stats.evasions == 1
    assert stats.lie_rate.value == pytest.approx(0.5)


def test_brier_rewards_calibration():
    confident_and_right = [_round(0, has_reward=True, claim=Claim.NONE, switched=True,
                                  confidence=1.0)]
    confident_and_wrong = [_round(0, has_reward=False, claim=Claim.NONE, switched=True,
                                  confidence=1.0)]
    assert model_stats([_match("a", "b", confident_and_right)])["b"].brier == pytest.approx(0.0)
    assert model_stats([_match("a", "b", confident_and_wrong)])["b"].brier == pytest.approx(1.0)


def test_switch_rate_exposes_position_bias():
    rounds = [_round(i, has_reward=True, claim=Claim.NONE, switched=True) for i in range(4)]
    assert model_stats([_match("a", "b", rounds)])["b"].switch_rate.value == pytest.approx(1.0)


def test_lie_rate_by_round_is_ordered_by_round_index():
    matches = [
        _match("a", "b", [
            _round(0, has_reward=True, claim=Claim.HAS_REWARD, switched=False),  # truth
            _round(1, has_reward=True, claim=Claim.EMPTY, switched=False),       # lie
        ]),
    ]
    series = lie_rate_by_round(matches)
    assert [p.value for p in series] == [0.0, 1.0]


def test_head_to_head_is_directional():
    rounds = [_round(0, has_reward=True, claim=Claim.NONE, switched=False)]
    h2h = head_to_head([_match("a", "b", rounds), _match("b", "a", rounds)])
    assert set(h2h) == {("a", "b"), ("b", "a")}


def test_belief_response_tabulates_lies_against_decisions():
    rounds = [
        _round(0, has_reward=True, claim=Claim.EMPTY, switched=False),   # lied, kept
        _round(1, has_reward=True, claim=Claim.EMPTY, switched=True),    # lied, switched
        _round(2, has_reward=True, claim=Claim.HAS_REWARD, switched=True),  # truth, switched
    ]
    table = belief_response([_match("a", "b", rounds)])
    assert table[True, False] == 1
    assert table[True, True] == 1
    assert table[False, True] == 1


def test_run_cost_sums_every_call():
    rounds = [_round(i, has_reward=True, claim=Claim.NONE, switched=False, cost=0.01)
              for i in range(3)]
    assert run_cost([_match("a", "b", rounds)]) == pytest.approx(0.03)


def test_reasoning_tokens_are_attributed_to_the_model_that_spent_them():
    """The confound has to be visible per model, not pooled."""
    from agent_bluff.game import Role

    rnd = _round(0, has_reward=True, claim=Claim.NONE, switched=False)
    rnd.calls = [
        CallRecord(role=Role.INFORMANT, model="i", kind="informant_open",
                   prompt_tokens=1, completion_tokens=1, reasoning_tokens=900,
                   cost_usd=0.0, latency_s=0.1),
        CallRecord(role=Role.CHOOSER, model="c", kind="chooser_decide",
                   prompt_tokens=1, completion_tokens=1, reasoning_tokens=100,
                   cost_usd=0.0, latency_s=0.1),
    ]
    stats = model_stats([_match("grok", "astra", [rnd])])

    assert stats["grok"].reasoning_per_call == 900
    assert stats["astra"].reasoning_per_call == 100
