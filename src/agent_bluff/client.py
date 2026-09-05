"""OpenRouter client, spend ledger, and the failure modes worth naming.

Every call returns both the parsed structured output and a :class:`CallRecord`
of what it cost, so accounting is a by-product of playing rather than a
separate estimation step.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from types import TracebackType
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agent_bluff.config import Effort, ModelSpec
from agent_bluff.game import Role
from agent_bluff.records import CallKind, CallRecord, json_schema

T = TypeVar("T", bound=BaseModel)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4
_SEMANTIC_ATTEMPTS = 2
_JSON_NUDGE = (
    "Your last reply was not valid JSON. Reply with the JSON object only -- "
    "no prose, no code fences, no explanation."
)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class BluffError(Exception):
    """Base class for harness failures."""


class BudgetExceeded(BluffError):
    """The run hit its spending ceiling and stopped."""


class PlayerRefused(BluffError):
    """A model declined to take its turn.

    Worth its own type rather than a retry: a model that will not deceive is a
    finding about that model, and silently retrying until it complies would
    both distort the result and hide it.
    """


class MalformedResponse(BluffError):
    """A model's output could not be parsed into the requested schema."""


class Truncated(BluffError):
    """Output hit the token ceiling before the model finished.

    Distinct from :class:`PlayerRefused` on purpose. Reasoning tokens are
    billed as output and count against the ceiling, so a model that thinks
    hard can return empty content that looks exactly like a refusal. Counting
    those as refusals would corrupt the one metric where a refusal is the
    finding.
    """


class TurnTaker(Protocol):
    """Anything that can take one turn on behalf of a player.

    Orchestration depends on this rather than on :class:`OpenRouterClient` so a
    match can be played against a stub, which is what lets the runner be tested
    without a network or a budget.
    """

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        role: Role,
        kind: CallKind,
        reasoning_effort: Effort,
        max_output_tokens: int,
    ) -> tuple[T, CallRecord]:
        """Take one turn and report what it cost."""
        ...


class SpendLedger:
    """Running total of spend, with a hard ceiling.

    OpenRouter reports the exact charge for every generation, so this tracks
    real money rather than an estimate from token counts.
    """

    def __init__(self, budget_usd: float) -> None:
        """Initialise an empty ledger.

        Args:
            budget_usd: Ceiling above which :meth:`add` refuses to continue.
        """
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.calls = 0
        self._lock = asyncio.Lock()

    async def add(self, cost_usd: float) -> None:
        """Record a charge and stop the run if the budget is now exhausted.

        Args:
            cost_usd: What the call actually cost.

        Raises:
            BudgetExceeded: If cumulative spend has reached the ceiling.
        """
        async with self._lock:
            self.spent_usd += cost_usd
            self.calls += 1
            if self.spent_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"spent ${self.spent_usd:.2f} of ${self.budget_usd:.2f} budget "
                    f"over {self.calls} calls; stopping"
                )

    @property
    def remaining_usd(self) -> float:
        """Headroom left before the ceiling."""
        return max(0.0, self.budget_usd - self.spent_usd)


def _with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the cacheable prefix for providers that require explicit breakpoints.

    The conversation is append-only, so the whole history up to the latest
    message is a stable prefix. Two breakpoints -- the system prompt and the
    newest message -- give an incrementally extending cache that hits on every
    call after the first.

    Args:
        messages: Chat messages with plain string content.

    Returns:
        A copy with ``cache_control`` markers attached.
    """
    marked = [dict(m) for m in messages]
    for index in {0, len(marked) - 1}:
        message = marked[index]
        message["content"] = [
            {
                "type": "text",
                "text": message["content"],
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return marked


def _cost_of(payload: dict[str, Any]) -> float:
    """Read what a generation cost, including one that failed to parse.

    A rejected attempt was still generated and billed, so it has to reach the
    ledger or the budget ceiling drifts above real spend.

    Args:
        payload: Decoded response body.

    Returns:
        Dollars charged for the generation.
    """
    return float((payload.get("usage") or {}).get("cost", 0.0))


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Args:
        text: Raw assistant content.

    Returns:
        The decoded object.

    Raises:
        MalformedResponse: If no JSON object could be recovered.
    """
    for candidate in (text, *(m.group(0) for m in [_JSON_BLOCK.search(text)] if m)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise MalformedResponse(f"no JSON object in response: {text[:400]!r}")


class OpenRouterClient:
    """Async client for the one endpoint this project needs."""

    def __init__(self, api_key: str, ledger: SpendLedger, timeout_s: float = 300.0) -> None:
        """Open a client.

        Args:
            api_key: OpenRouter API key.
            ledger: Shared spend ledger; every call is booked against it.
            timeout_s: Per-request timeout. Reasoning calls can be slow.
        """
        self._ledger = ledger
        self._http = httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "agent-bluff",
                "HTTP-Referer": "https://github.com/paveljor/agent-bluff",
            },
        )

    async def __aenter__(self) -> OpenRouterClient:
        """Enter the async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        role: Role,
        kind: CallKind,
        reasoning_effort: Effort,
        max_output_tokens: int,
    ) -> tuple[T, CallRecord]:
        """Take one turn.

        Args:
            spec: Which model to call and how.
            messages: Append-only conversation for this player.
            schema: Structured output to constrain and validate against.
            role: Player role, recorded for accounting.
            kind: Which of the four per-round calls this is.
            reasoning_effort: Reasoning effort, identical for every model.
            max_output_tokens: Per-call output ceiling.

        Returns:
            The validated response and the call's accounting record.

        Raises:
            PlayerRefused: The model declined to answer.
            MalformedResponse: Output could not be parsed after a retry.
            BudgetExceeded: The run hit its ceiling.
        """
        started = time.monotonic()
        turn_messages, ceiling, attempts = messages, max_output_tokens, 0
        failure: BluffError | None = None

        # Two recoverable failures deserve one more try each, because both are
        # artefacts of how the request was framed rather than facts about the
        # model: output truncated by the reasoning budget, and a provider that
        # ignored the JSON schema. A refusal is never retried -- that is data.
        for _ in range(_SEMANTIC_ATTEMPTS):
            attempts += 1
            body: dict[str, Any] = {
                "model": spec.id,
                "messages": _with_cache_breakpoints(turn_messages)
                if spec.explicit_cache_control
                else turn_messages,
                "max_tokens": ceiling,
                "reasoning": {"effort": reasoning_effort},
                "response_format": json_schema(schema, schema.__name__),
                "usage": {"include": True},
            }
            if (routing := spec.routing()) is not None:
                body["provider"] = routing

            payload, _, used_schema = await self._post_with_retries(body)
            try:
                parsed, record = self._interpret(
                    payload, schema, spec=spec, role=role, kind=kind,
                    attempts=attempts, used_schema=used_schema,
                    latency=time.monotonic() - started,
                )
            except Truncated as exc:
                # The attempt was still generated and billed.
                await self._ledger.add(_cost_of(payload))
                failure, ceiling = exc, ceiling * 2
                continue
            except MalformedResponse as exc:
                await self._ledger.add(_cost_of(payload))
                failure = exc
                turn_messages = [*messages, {"role": "user", "content": _JSON_NUDGE}]
                continue

            await self._ledger.add(record.cost_usd)
            return parsed, record

        assert failure is not None
        raise failure

    async def _post_with_retries(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], int, bool]:
        """POST the request, retrying transient failures.

        Falls back to unconstrained generation if the provider rejects the JSON
        schema, so one provider's limitation degrades quality rather than
        aborting the tournament -- the fallback is recorded per call.

        Args:
            body: Request body.

        Returns:
            The response payload, the number of attempts made, and whether the
            structured-output schema survived.

        Raises:
            httpx.HTTPStatusError: On a non-retryable error, or after the last
                attempt.
        """
        used_schema = True
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._http.post(API_URL, json=body)
            except httpx.RequestError as exc:  # network flake
                last_error = exc
                await self._backoff(attempt)
                continue

            if response.status_code == httpx.codes.OK:
                return response.json(), attempt, used_schema

            if response.status_code == httpx.codes.BAD_REQUEST and used_schema:
                detail = response.text.lower()
                if "response_format" in detail or "json_schema" in detail:
                    body = {k: v for k, v in body.items() if k != "response_format"}
                    used_schema = False
                    continue

            if response.status_code not in _RETRY_STATUS or attempt == _MAX_ATTEMPTS:
                response.raise_for_status()

            await self._backoff(attempt)

        raise BluffError(f"exhausted retries: {last_error}")

    @staticmethod
    async def _backoff(attempt: int) -> None:
        """Sleep before the next attempt, with jitter to avoid thundering herds.

        Args:
            attempt: One-based attempt number that just failed.
        """
        await asyncio.sleep(min(2.0**attempt, 30.0) * (0.5 + random.random()))

    def _interpret(
        self,
        payload: dict[str, Any],
        schema: type[T],
        *,
        spec: ModelSpec,
        role: Role,
        kind: CallKind,
        attempts: int,
        used_schema: bool,
        latency: float,
    ) -> tuple[T, CallRecord]:
        """Turn a raw API payload into a validated turn plus its accounting.

        Args:
            payload: Decoded response body.
            schema: Expected response model.
            spec: The model that was called.
            role: Player role.
            kind: Which per-round call this is.
            attempts: How many HTTP attempts it took.
            used_schema: Whether structured output was in force.
            latency: Wall-clock seconds for the call.

        Returns:
            The validated response and its call record.

        Raises:
            PlayerRefused: The model declined.
            MalformedResponse: Output did not satisfy the schema.
        """
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""

        if choice.get("finish_reason") == "length":
            raise Truncated(
                f"{spec.key} hit the {payload.get('usage', {}).get('completion_tokens', '?')}"
                f"-token ceiling; reasoning budget likely too tight"
            )
        if message.get("refusal"):
            raise PlayerRefused(f"{spec.key} refused: {message['refusal']}")
        if choice.get("finish_reason") == "content_filter":
            raise PlayerRefused(f"{spec.key} was content-filtered")
        if not content.strip():
            raise PlayerRefused(f"{spec.key} returned empty content")

        try:
            parsed = schema.model_validate(_extract_json(content))
        except ValidationError as exc:
            raise MalformedResponse(f"{spec.key} output failed validation: {exc}") from exc

        usage = payload.get("usage") or {}
        record = CallRecord(
            role=role,
            model=spec.id,
            kind=kind,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            ),
            cost_usd=float(usage.get("cost", 0.0)),
            latency_s=round(latency, 3),
            attempts=attempts,
            used_json_schema=used_schema,
            provider=payload.get("provider"),
            generation_id=payload.get("id"),
        )
        return parsed, record
