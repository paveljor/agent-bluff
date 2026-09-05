"""Match and tournament orchestration."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent_bluff import prompts
from agent_bluff.client import BluffError, BudgetExceeded, SpendLedger, TurnTaker
from agent_bluff.config import ModelSpec, RunConfig
from agent_bluff.game import Claim, Role, Round, match_seed, resolve, reward_schedule
from agent_bluff.records import (
    CallKind,
    CallRecord,
    ChooserDecision,
    ChooserTurn,
    InformantTurn,
    MatchRecord,
    RoundRecord,
)
from agent_bluff.storage import MatchLog


@dataclass(slots=True)
class Conversation:
    """One player's append-only view of a match.

    Consecutive user messages are merged rather than appended separately: a
    round's outcome report and the next round's opening are both user turns,
    and some providers reject two in a row. Merging keeps the wire format valid
    without distorting what the player is shown.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)

    def system(self, content: str) -> None:
        """Seed the conversation with its system prompt.

        Args:
            content: The system prompt.
        """
        self.messages.append({"role": "system", "content": content})

    def user(self, content: str) -> None:
        """Append a user message, merging into the previous one if adjacent.

        Args:
            content: Message text.
        """
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += f"\n\n{content}"
        else:
            self.messages.append({"role": "user", "content": content})

    def assistant(self, content: str) -> None:
        """Append the player's own turn verbatim.

        Args:
            content: The raw JSON the model emitted, kept as-is so the format
                is reinforced by the player's own history.
        """
        self.messages.append({"role": "assistant", "content": content})


async def play_match(
    client: TurnTaker,
    config: RunConfig,
    informant: ModelSpec,
    chooser: ModelSpec,
    match_index: int,
) -> MatchRecord:
    """Play one match of ``config.rounds`` rounds between an ordered pair.

    A match that fails partway -- a refusal, an unparseable turn, a provider
    error -- is returned with the rounds it completed and the error recorded,
    rather than discarded. Budget exhaustion propagates, since the whole run
    must stop.

    Args:
        client: Shared API client.
        config: Run configuration.
        informant: Model playing the informant.
        chooser: Model playing the chooser.
        match_index: Zero-based repeat number for this ordered pair.

    Returns:
        The match record, possibly partial.

    Raises:
        BudgetExceeded: The run hit its spending ceiling.
    """
    seed = match_seed(informant.key, chooser.key, match_index)
    started = datetime.now(UTC).isoformat()

    inf = Conversation()
    inf.system(prompts.system_prompt(Role.INFORMANT, config.rounds, config.word_cap))
    cho = Conversation()
    cho.system(prompts.system_prompt(Role.CHOOSER, config.rounds, config.word_cap))

    rounds: list[RoundRecord] = []
    error: str | None = None

    try:
        for index, has_reward in enumerate(reward_schedule(seed, config.rounds)):
            rounds.append(
                await _play_round(client, config, informant, chooser, inf, cho, index, has_reward)
            )
    except BudgetExceeded:
        raise
    except BluffError as exc:
        error = f"{type(exc).__name__}: {exc}"

    return MatchRecord(
        seed=seed,
        informant_model=informant.key,
        chooser_model=chooser.key,
        match_index=match_index,
        rounds=rounds,
        config={
            "rounds": config.rounds,
            "reasoning_effort": config.reasoning_effort,
            "word_cap": config.word_cap,
            "informant_id": informant.id,
            "chooser_id": chooser.id,
        },
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        error=error,
    )


async def _play_round(
    client: TurnTaker,
    config: RunConfig,
    informant: ModelSpec,
    chooser: ModelSpec,
    inf: Conversation,
    cho: Conversation,
    index: int,
    has_reward: bool,
) -> RoundRecord:
    """Run the four calls of a single round and score it.

    Args:
        client: Shared API client.
        config: Run configuration.
        informant: Model playing the informant.
        chooser: Model playing the chooser.
        inf: The informant's conversation, extended in place.
        cho: The chooser's conversation, extended in place.
        index: Zero-based round number.
        has_reward: Whether the reward sits in the informant's box.

    Returns:
        The scored round, including every message and call record.
    """
    calls: list[CallRecord] = []

    async def turn(
        spec: ModelSpec, convo: Conversation, schema: type[Any], role: Role, kind: CallKind
    ) -> Any:
        """Take one turn and fold the accounting into ``calls``."""
        parsed, record = await client.complete(
            spec,
            convo.messages,
            schema,
            role=role,
            kind=kind,
            reasoning_effort=config.reasoning_effort,
            max_output_tokens=config.max_output_tokens,
        )
        convo.assistant(parsed.model_dump_json())
        calls.append(record)
        return parsed

    inf.user(prompts.informant_open(index, config.rounds, has_reward))
    opening = await turn(informant, inf, InformantTurn, Role.INFORMANT, "informant_open")

    cho.user(prompts.chooser_reply(index, config.rounds, opening.message))
    reply = await turn(chooser, cho, ChooserTurn, Role.CHOOSER, "chooser_reply")

    inf.user(prompts.informant_close(reply.message))
    closing = await turn(informant, inf, InformantTurn, Role.INFORMANT, "informant_close")

    cho.user(prompts.chooser_decide(closing.message))
    decision = await turn(chooser, cho, ChooserDecision, Role.CHOOSER, "chooser_decide")

    close_claim = Claim(closing.claim)
    resolution = resolve(Round(index, has_reward), close_claim, decision.switch)

    inf.user(
        prompts.outcome(
            Role.INFORMANT,
            switched=decision.switch,
            informant_has_reward=has_reward,
            won=resolution.informant_won,
        )
    )
    cho.user(
        prompts.outcome(
            Role.CHOOSER,
            switched=decision.switch,
            informant_has_reward=has_reward,
            won=resolution.chooser_won,
        )
    )

    return RoundRecord(
        index=index,
        informant_has_reward=has_reward,
        informant_open=opening.message,
        open_claim=Claim(opening.claim),
        chooser_reply=reply.message,
        informant_close=closing.message,
        close_claim=close_claim,
        switched=decision.switch,
        confidence_informant_has_reward=decision.confidence_informant_has_reward,
        chooser_reasoning=decision.reasoning,
        informant_lied=resolution.informant_lied,
        informant_won=resolution.informant_won,
        calls=calls,
    )


async def run_tournament(
    client: TurnTaker,
    config: RunConfig,
    ledger: SpendLedger,
    log: MatchLog,
) -> None:
    """Play every scheduled match, skipping any already in the log.

    Args:
        client: Shared API client.
        config: Run configuration.
        ledger: Spend ledger, shared across all matches.
        log: Destination for completed matches.
    """
    specs = {spec.key: spec for spec in config.specs()}
    done = log.completed_seeds()

    todo = [
        (informant, chooser, index)
        for informant, chooser in config.ordered_pairs()
        for index in range(config.matches_per_pair)
        if match_seed(informant, chooser, index) not in done
    ]

    if done:
        print(f"resuming: {len(done)} matches already played", file=sys.stderr)
    print(f"playing {len(todo)} matches, budget ${ledger.budget_usd:.2f}", file=sys.stderr)

    semaphore = asyncio.Semaphore(config.max_concurrency)
    completed = 0
    halted = False

    async def worker(informant: str, chooser: str, index: int) -> None:
        nonlocal completed, halted
        async with semaphore:
            if halted:
                return
            try:
                record = await play_match(
                    client, config, specs[informant], specs[chooser], index
                )
            except BudgetExceeded as exc:
                halted = True
                print(f"\nBUDGET STOP: {exc}", file=sys.stderr)
                return
            await log.append(record)
            completed += 1
            flag = " ERROR" if record.error else ""
            print(
                f"[{completed}/{len(todo)}] {informant} vs {chooser} #{index} "
                f"${ledger.spent_usd:.2f} spent{flag}",
                file=sys.stderr,
            )

    await asyncio.gather(*(worker(*job) for job in todo))
    print(
        f"\ndone: {completed} matches, ${ledger.spent_usd:.2f} over {ledger.calls} calls",
        file=sys.stderr,
    )
