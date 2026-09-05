"""Rules of the game, verified without spending anything."""

import pytest

from agent_bluff.game import Claim, Role, Round, match_seed, resolve, reward_schedule


@pytest.mark.parametrize(
    ("has_reward", "switched", "expect_informant_wins"),
    [
        (True, False, True),  # holds it, chooser stays -> keeps it
        (True, True, False),  # holds it, chooser swaps -> loses it
        (False, False, False),  # empty, chooser stays -> keeps nothing
        (False, True, True),  # empty, chooser swaps -> gains the good box
    ],
)
def test_resolve_is_zero_sum(has_reward, switched, expect_informant_wins):
    result = resolve(Round(0, has_reward), Claim.NONE, switched)
    assert result.informant_won is expect_informant_wins
    assert result.chooser_won is not expect_informant_wins


@pytest.mark.parametrize(
    ("claim", "has_reward", "lied"),
    [
        (Claim.HAS_REWARD, True, False),
        (Claim.HAS_REWARD, False, True),
        (Claim.EMPTY, False, False),
        (Claim.EMPTY, True, True),
    ],
)
def test_lie_detection(claim, has_reward, lied):
    assert resolve(Round(0, has_reward), claim, False).informant_lied is lied


def test_evasion_is_not_scored_as_a_lie():
    assert resolve(Round(0, True), Claim.NONE, False).informant_lied is None


def test_honesty_is_self_defeating_against_a_credulous_opponent():
    """The premise the whole experiment rests on, stated as a test.

    A chooser who simply believes the informant swaps when told 'I have it'.
    Against that opponent the informant loses every round it tells the truth.
    """
    for has_reward in (True, False):
        truthful = Claim.HAS_REWARD if has_reward else Claim.EMPTY
        credulous_switches = truthful is Claim.HAS_REWARD
        assert not resolve(Round(0, has_reward), truthful, credulous_switches).informant_won


def test_reward_schedule_is_balanced_and_deterministic():
    schedule = reward_schedule("seed-a", 10)
    assert sum(schedule) == 5
    assert schedule == reward_schedule("seed-a", 10)
    assert schedule != reward_schedule("seed-b", 10)


def test_reward_schedule_handles_odd_round_counts():
    counts = {sum(reward_schedule(f"s{i}", 9)) for i in range(40)}
    assert counts <= {4, 5}
    assert len(counts) == 2, "the odd round should not always fall the same way"


def test_reward_schedule_rejects_empty_matches():
    with pytest.raises(ValueError, match="at least one round"):
        reward_schedule("s", 0)


def test_roles_are_opposites():
    assert Role.INFORMANT.opponent() is Role.CHOOSER
    assert Role.CHOOSER.opponent() is Role.INFORMANT


def test_match_seed_is_order_sensitive():
    assert match_seed("a", "b", 0) != match_seed("b", "a", 0)
    assert match_seed("a", "b", 0) != match_seed("a", "b", 1)
