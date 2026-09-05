"""Prompt construction."""

import re

import pytest

from agent_bluff import prompts
from agent_bluff.game import Role


@pytest.mark.parametrize("role", list(Role))
def test_system_prompt_states_the_terms(role):
    text = prompts.system_prompt(role, rounds=10, word_cap=50)
    assert "10 rounds" in text
    assert "under 50 words" in text
    assert role.value.upper() in text


@pytest.mark.parametrize("role", list(Role))
def test_system_prompt_does_not_suggest_a_strategy(role):
    """Prompting for deception would assume the result we are measuring."""
    text = prompts.system_prompt(role, rounds=10, word_cap=50).lower()
    loaded = ("lie", "lied", "lying", "deceive", "deception", "bluff",
              "honest", "honestly", "truth", "truthful", "trust", "trick")
    found = [w for w in loaded if re.search(rf"\b{w}\b", text)]
    assert not found, f"strategy-loaded words in the prompt: {found}"


def test_informant_is_told_what_is_in_the_box():
    assert "REWARD IS IN YOUR BOX" in prompts.informant_open(0, 10, has_reward=True)
    assert "YOUR BOX IS EMPTY" in prompts.informant_open(0, 10, has_reward=False)


def test_rounds_are_numbered_from_one_for_the_players():
    assert "Round 1 of 10" in prompts.informant_open(0, 10, has_reward=True)
    assert "Round 10 of 10" in prompts.informant_open(9, 10, has_reward=True)


@pytest.mark.parametrize("informant_has_reward", [True, False])
@pytest.mark.parametrize("switched", [True, False])
def test_outcome_places_the_reward_consistently_for_both_players(switched, informant_has_reward):
    """Each player is told the same fact from their own side of the table."""
    informant_won = informant_has_reward != switched
    seen_by_informant = prompts.outcome(
        Role.INFORMANT, switched=switched, informant_has_reward=informant_has_reward,
        won=informant_won,
    )
    seen_by_chooser = prompts.outcome(
        Role.CHOOSER, switched=switched, informant_has_reward=informant_has_reward,
        won=not informant_won,
    )
    if informant_has_reward:
        assert "reward was in your box" in seen_by_informant
        assert "reward was in their box" in seen_by_chooser
    else:
        assert "reward was in their box" in seen_by_informant
        assert "reward was in your box" in seen_by_chooser

    assert ("You WON" in seen_by_informant) is informant_won
    assert ("You WON" in seen_by_chooser) is not informant_won
