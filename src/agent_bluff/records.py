"""Schemas for model responses and for everything written to disk.

Two families live here. The ``*Turn`` / ``*Decision`` models are the structured
outputs we ask the players for; their JSON schemas are sent to the API. The
``*Record`` models are the append-only run log -- one ``MatchRecord`` per line
of ``matches.jsonl``, carrying enough detail to recompute every published
metric and to quote any exchange verbatim.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_bluff.game import Claim, Role

# --------------------------------------------------------------------------
# Structured outputs requested from the players
# --------------------------------------------------------------------------


class InformantTurn(BaseModel):
    """A message from the informant, with their own account of what they claimed."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="What you say to the other player.")
    claim: Literal["has_reward", "empty", "none"] = Field(
        description=(
            "Your own summary of the position you just took: 'has_reward' if you "
            "asserted your box holds the reward, 'empty' if you asserted it is "
            "empty, 'none' if you deliberately committed to neither."
        )
    )


class ChooserTurn(BaseModel):
    """A message from the chooser during the exchange."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="What you say to the other player.")


class ChooserDecision(BaseModel):
    """The chooser's final call for the round."""

    model_config = ConfigDict(extra="forbid")

    switch: bool = Field(description="True to swap boxes, false to keep your own.")
    confidence_informant_has_reward: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your probability, from 0 to 1, that the reward is in THEIR box. "
            "Answer honestly; this is scored for calibration, not for winning."
        ),
    )
    reasoning: str = Field(description="One sentence on why.")


def json_schema(model: type[BaseModel], name: str) -> dict[str, Any]:
    """Render a Pydantic model as an OpenRouter ``response_format`` payload.

    Args:
        model: The response model to constrain generation to.
        name: Schema name reported to the provider.

    Returns:
        A ``response_format`` dict suitable for the chat-completions body.
    """
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


# --------------------------------------------------------------------------
# Run log
# --------------------------------------------------------------------------


CallKind = Literal["informant_open", "chooser_reply", "informant_close", "chooser_decide"]
"""Which of the four calls that make up a round this is."""


class CallRecord(BaseModel):
    """Accounting for a single model call."""

    role: Role
    model: str
    kind: CallKind
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float
    latency_s: float
    attempts: int = 1
    used_json_schema: bool = True
    provider: str | None = None
    generation_id: str | None = None


class RoundRecord(BaseModel):
    """Everything that happened in one round, including the full dialogue."""

    index: int
    informant_has_reward: bool
    informant_open: str
    open_claim: Claim
    chooser_reply: str
    informant_close: str
    close_claim: Claim
    """The informant's final position, and the one the chooser acts on.

    Scoring uses this rather than ``open_claim``; keeping both makes a
    mid-round reversal visible instead of silently collapsing it.
    """

    switched: bool
    confidence_informant_has_reward: float
    chooser_reasoning: str
    informant_lied: bool | None
    informant_won: bool
    calls: list[CallRecord]


class MatchRecord(BaseModel):
    """One complete match between an ordered pair of models."""

    seed: str
    informant_model: str
    chooser_model: str
    match_index: int
    rounds: list[RoundRecord]
    config: dict[str, Any]
    started_at: str
    finished_at: str
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        """Total spend attributable to this match."""
        return sum(call.cost_usd for rnd in self.rounds for call in rnd.calls)
