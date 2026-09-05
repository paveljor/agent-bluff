"""Ledger, response parsing, and the failure modes we name explicitly."""

from typing import Any

import pytest

from agent_bluff.client import (
    BudgetExceeded,
    MalformedResponse,
    PlayerRefused,
    SpendLedger,
    _extract_json,
    _with_cache_breakpoints,
)
from agent_bluff.config import MODELS
from agent_bluff.game import Role
from agent_bluff.records import ChooserTurn


async def test_ledger_accumulates():
    ledger = SpendLedger(budget_usd=1.0)
    await ledger.add(0.25)
    await ledger.add(0.25)
    assert ledger.spent_usd == pytest.approx(0.5)
    assert ledger.remaining_usd == pytest.approx(0.5)
    assert ledger.calls == 2


async def test_ledger_stops_the_run_at_the_ceiling():
    ledger = SpendLedger(budget_usd=0.10)
    await ledger.add(0.05)
    with pytest.raises(BudgetExceeded, match="stopping"):
        await ledger.add(0.06)


async def test_ledger_books_the_overspending_call_before_raising():
    """The call was made and billed; the ledger must not under-report it."""
    ledger = SpendLedger(budget_usd=0.10)
    with pytest.raises(BudgetExceeded):
        await ledger.add(0.50)
    assert ledger.spent_usd == pytest.approx(0.50)


def test_extract_json_handles_a_bare_object():
    assert _extract_json('{"message": "hi"}') == {"message": "hi"}


def test_extract_json_recovers_from_prose_and_fences():
    text = 'Sure!\n```json\n{"message": "hi"}\n```\nHope that helps.'
    assert _extract_json(text) == {"message": "hi"}


def test_extract_json_rejects_output_with_no_object():
    with pytest.raises(MalformedResponse, match="no JSON object"):
        _extract_json("I would rather not play this game.")


def test_cache_breakpoints_mark_the_stable_prefix_and_the_newest_message():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "round 1"},
        {"role": "assistant", "content": "{}"},
        {"role": "user", "content": "round 2"},
    ]
    marked = _with_cache_breakpoints(messages)

    assert marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert marked[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert marked[1]["content"] == "round 1"
    assert messages[0]["content"] == "rules", "input must not be mutated"


def test_cache_breakpoints_on_a_single_message_do_not_duplicate():
    marked = _with_cache_breakpoints([{"role": "system", "content": "rules"}])
    assert len(marked) == 1
    assert len(marked[0]["content"]) == 1


class _Interpreter:
    """Exposes the client's private payload interpretation for testing."""

    @staticmethod
    def call(payload):
        from agent_bluff.client import OpenRouterClient

        client = OpenRouterClient.__new__(OpenRouterClient)
        return client._interpret(
            payload, ChooserTurn, spec=MODELS["grok"], role=Role.CHOOSER,
            kind="chooser_reply", attempts=1, used_schema=True, latency=0.5,
        )


def _payload(content, **message_extra):
    return {
        "id": "gen-1",
        "provider": "xai",
        "choices": [{"message": {"content": content, **message_extra}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
    }


def test_interpret_parses_and_accounts():
    turn, record = _Interpreter.call(_payload('{"message": "your box looks light"}'))
    assert turn.message == "your box looks light"
    assert record.cost_usd == pytest.approx(0.001)
    assert record.prompt_tokens == 100
    assert record.provider == "xai"


def test_refusal_is_raised_not_retried():
    """A model that will not play is a finding, not a transient error."""
    payload = _payload("", refusal="I won't deceive another party.")
    with pytest.raises(PlayerRefused, match="won't deceive"):
        _Interpreter.call(payload)


def test_empty_content_counts_as_a_refusal():
    with pytest.raises(PlayerRefused, match="empty content"):
        _Interpreter.call(_payload("   "))


def test_content_filter_counts_as_a_refusal():
    payload = _payload('{"message": "x"}')
    payload["choices"][0]["finish_reason"] = "content_filter"
    with pytest.raises(PlayerRefused, match="content-filtered"):
        _Interpreter.call(payload)


def test_schema_violation_is_reported():
    with pytest.raises(MalformedResponse, match="failed validation"):
        _Interpreter.call(_payload('{"wrong_field": 1}'))


def test_truncation_is_not_mistaken_for_a_refusal():
    """Reasoning tokens count against the ceiling, so a cut-off looks like silence."""
    from agent_bluff.client import Truncated

    payload = _payload("")
    payload["choices"][0]["finish_reason"] = "length"
    with pytest.raises(Truncated, match="reasoning budget"):
        _Interpreter.call(payload)


class _ScriptedClient:
    """Drives complete() over a fixed sequence of payloads, without HTTP."""

    def __init__(self, payloads):
        from agent_bluff.client import OpenRouterClient, SpendLedger

        self.payloads = list(payloads)
        self.bodies: list[dict[str, Any]] = []
        self.client = OpenRouterClient.__new__(OpenRouterClient)
        self.ledger = SpendLedger(budget_usd=10.0)
        self.client._ledger = self.ledger

        async def fake_post(body):
            self.bodies.append(body)
            return self.payloads.pop(0), 1, True

        self.client._post_with_retries = fake_post  # type: ignore[method-assign]

    async def run(self, schema=ChooserTurn):
        from agent_bluff.game import Role

        return await self.client.complete(
            MODELS["grok"], [{"role": "user", "content": "go"}], schema,
            role=Role.CHOOSER, kind="chooser_reply",
            reasoning_effort='medium', max_output_tokens=1000,
        )


async def test_truncated_call_is_retried_with_a_larger_ceiling():
    cut_off = _payload("")
    cut_off["choices"][0]["finish_reason"] = "length"
    scripted = _ScriptedClient([cut_off, _payload('{"message": "second try"}')])

    turn, record = await scripted.run()

    assert turn.message == "second try"
    assert record.attempts == 2
    assert scripted.bodies[1]["max_tokens"] == 2 * scripted.bodies[0]["max_tokens"]


async def test_unparseable_output_is_retried_with_a_correction():
    scripted = _ScriptedClient([_payload("Sure, here you go!"),
                                _payload('{"message": "ok"}')])

    turn, _ = await scripted.run()

    assert turn.message == "ok"
    nudge = scripted.bodies[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "not valid JSON" in nudge["content"]


async def test_failed_attempts_are_still_billed_to_the_ledger():
    """A rejected generation was produced and charged; the ceiling must know."""
    scripted = _ScriptedClient([_payload("prose"), _payload('{"message": "ok"}')])

    await scripted.run()

    assert scripted.ledger.spent_usd == pytest.approx(0.002)  # both attempts
    assert scripted.ledger.calls == 2


async def test_a_refusal_is_never_retried():
    scripted = _ScriptedClient([_payload("", refusal="no thanks"),
                                _payload('{"message": "unused"}')])

    with pytest.raises(PlayerRefused):
        await scripted.run()

    assert len(scripted.bodies) == 1, "refusals are data, not transient errors"


async def test_persistent_malformed_output_eventually_raises():
    scripted = _ScriptedClient([_payload("prose"), _payload("still prose")])
    with pytest.raises(MalformedResponse):
        await scripted.run()


async def test_every_model_is_sent_the_same_reasoning_instruction():
    """A token budget is only enforced by Anthropic, so effort is the one
    parameter that means 'the same request' to all three providers."""
    scripted = _ScriptedClient([_payload('{"message": "ok"}')])
    await scripted.run()
    assert scripted.bodies[0]["reasoning"] == {"effort": "medium"}
