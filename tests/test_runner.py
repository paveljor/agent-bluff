"""Match orchestration, driven by a stub client so no network is touched."""

import itertools

import pytest

from agent_bluff.client import PlayerRefused
from agent_bluff.config import MODELS, RunConfig
from agent_bluff.game import Claim
from agent_bluff.records import CallRecord, ChooserDecision, ChooserTurn, InformantTurn
from agent_bluff.runner import Conversation, play_match


class StubClient:
    """Returns scripted turns and records what it was asked."""

    def __init__(self, *, claim="empty", switch=False, fail_on=None):
        self.claim = claim
        self.switch = switch
        self.fail_on = fail_on
        self.calls = []
        self._counter = itertools.count()

    async def complete(self, spec, messages, schema, *, role, kind,
                       reasoning_effort, max_output_tokens):
        index = next(self._counter)
        self.calls.append((spec.key, kind, len(messages)))
        if self.fail_on is not None and index == self.fail_on:
            raise PlayerRefused("scripted refusal")

        parsed: InformantTurn | ChooserTurn | ChooserDecision
        if schema is InformantTurn:
            parsed = InformantTurn(message=f"informant {kind}", claim=self.claim)
        elif schema is ChooserTurn:
            parsed = ChooserTurn(message="chooser reply")
        else:
            parsed = ChooserDecision(
                switch=self.switch, confidence_informant_has_reward=0.5, reasoning="because"
            )
        record = CallRecord(
            role=role, model=spec.id, kind=kind, prompt_tokens=10, completion_tokens=5,
            cost_usd=0.001, latency_s=0.01,
        )
        return parsed, record


@pytest.fixture
def config():
    return RunConfig(models=("astra", "grok"), rounds=3, matches_per_pair=1)


async def test_conversation_merges_adjacent_user_messages():
    """A round's outcome and the next round's opening are both user turns."""
    convo = Conversation()
    convo.system("rules")
    convo.user("result of round 1")
    convo.user("round 2 begins")
    assert len(convo.messages) == 2
    assert convo.messages[1]["content"] == "result of round 1\n\nround 2 begins"


async def test_conversation_keeps_turns_separated_by_an_assistant_reply():
    convo = Conversation()
    convo.user("a")
    convo.assistant("{}")
    convo.user("b")
    assert [m["role"] for m in convo.messages] == ["user", "assistant", "user"]


async def test_play_match_runs_every_round(config):
    client = StubClient()
    match = await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)

    assert match.error is None
    assert len(match.rounds) == config.rounds
    assert [r.index for r in match.rounds] == [0, 1, 2]
    assert match.informant_model == "astra"
    assert match.chooser_model == "grok"


async def test_play_match_makes_four_calls_per_round(config):
    client = StubClient()
    await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)

    assert len(client.calls) == 4 * config.rounds
    kinds = [kind for _, kind, _ in client.calls[:4]]
    assert kinds == ["informant_open", "chooser_reply", "informant_close", "chooser_decide"]


async def test_context_grows_monotonically_so_the_cache_prefix_holds(config):
    """Append-only history is what makes prompt caching pay; assert it stays so."""
    client = StubClient()
    await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)

    informant_lengths = [n for _, kind, n in client.calls if kind.startswith("informant")]
    chooser_lengths = [n for _, kind, n in client.calls if kind.startswith("chooser")]

    assert informant_lengths == sorted(informant_lengths)
    assert chooser_lengths == sorted(chooser_lengths)
    assert informant_lengths[0] < informant_lengths[-1], "history should accumulate"
    assert chooser_lengths[0] < chooser_lengths[-1]


async def test_play_match_scores_rounds_against_the_hidden_schedule(config):
    """A constant 'empty' claim is a lie exactly when the informant holds the reward."""
    client = StubClient(claim="empty", switch=False)
    match = await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)

    for rnd in match.rounds:
        assert rnd.close_claim is Claim.EMPTY
        assert rnd.informant_lied is rnd.informant_has_reward
        # The chooser always keeps, so the informant wins exactly when holding it.
        assert rnd.informant_won is rnd.informant_has_reward


async def test_partial_match_is_kept_when_a_player_refuses(config):
    """Rounds already paid for must survive a mid-match failure."""
    client = StubClient(fail_on=6)  # second round, informant's closing turn
    match = await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)

    assert match.error is not None
    assert "PlayerRefused" in match.error
    assert len(match.rounds) == 1


async def test_match_records_its_own_cost(config):
    client = StubClient()
    match = await play_match(client, config, MODELS["astra"], MODELS["grok"], 0)
    assert match.cost_usd == pytest.approx(0.001 * 4 * config.rounds)
