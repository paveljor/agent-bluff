"""Model registry and run configuration.

Model ids were verified against ``https://openrouter.ai/api/v1/models``; run
``bluff validate`` to re-check them before a run, since slugs change when
providers ship new versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Effort = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """How to call one player, and what that provider needs handled specially."""

    key: str
    """Short label used in filenames, tables and plots."""

    id: str
    """OpenRouter model slug."""

    explicit_cache_control: bool = False
    """Whether cache breakpoints must be marked by hand.

    Anthropic bills cached reads only where a ``cache_control`` breakpoint was
    placed; OpenAI and xAI cache automatically above a size threshold. Marking
    breakpoints for a provider that does not use them is harmless but noisy, so
    we only do it where it pays.
    """

    provider_order: tuple[str, ...] = ()
    """Providers to pin, in order. Empty means let OpenRouter route.

    Closed models have exactly one upstream, so routing cannot vary. Open-weight
    models are served by many hosts at differing quantisations, which would make
    a run irreproducible -- pin those.
    """

    def routing(self) -> dict[str, object] | None:
        """Return the ``provider`` routing block for the request body, if any."""
        if not self.provider_order:
            return None
        return {"order": list(self.provider_order), "allow_fallbacks": False}


#: The tournament field, plus a cheap stand-in for each provider.
#:
#: The ``poc`` entries mirror the provider mix of the real field rather than
#: being the three cheapest models available, so a dry run exercises the same
#: code paths: Anthropic's explicit cache breakpoints, OpenAI's schema dialect
#: and xAI's parameter handling. Open-weight models hosted by third parties
#: would be cheaper but would test none of that.
MODELS: dict[str, ModelSpec] = {
    "astra": ModelSpec(key="astra", id="openai/gpt-6-astra"),
    "fable": ModelSpec(key="fable", id="anthropic/claude-fable-5.1", explicit_cache_control=True),
    "grok": ModelSpec(key="grok", id="x-ai/grok-4.6"),
    "deepseek": ModelSpec(
        key="deepseek",
        id="deepseek/deepseek-v4-pro-0813",
        provider_order=("deepseek",),
    ),
    "qwen": ModelSpec(key="qwen", id="qwen/qwen3.8-max-0902", provider_order=("alibaba",)),
    # Cheap stand-ins, one per provider, for end-to-end dry runs.
    "haiku": ModelSpec(
        key="haiku", id="anthropic/claude-haiku-4.5", explicit_cache_control=True
    ),
    "nano": ModelSpec(key="nano", id="openai/gpt-5-nano"),
    "grokmini": ModelSpec(key="grokmini", id="x-ai/grok-4.20"),
}

DEFAULT_FIELD: tuple[str, ...] = ("astra", "fable", "grok")
"""The models the results are actually about."""

POC_FIELD: tuple[str, ...] = ("haiku", "nano", "grokmini")
"""Cheap equivalents for validating the harness before spending on the real run.

A full-fidelity dry run over this field costs well under a dollar, so it can
mirror the real tournament exactly instead of being a cut-down version. It
proves the plumbing, not the cost: per-match spend is model-specific, so the
projection still has to come from a pilot over ``DEFAULT_FIELD``.
"""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything that defines a tournament run.

    Serialised into every match record so a result can always be traced back to
    the settings that produced it.
    """

    models: tuple[str, ...] = DEFAULT_FIELD
    rounds: int = 10
    """Rounds per match. Memory persists across rounds; it resets between matches."""

    matches_per_pair: int = 10
    """Independent repeats of each ordered pair."""

    word_cap: int = 50
    """Soft cap on message length, stated in the prompt."""

    max_concurrency: int = 8
    """Matches in flight at once."""

    budget_usd: float = 30.0
    """Hard ceiling. The run aborts rather than exceed it."""

    reasoning_effort: Effort = "medium"
    """Reasoning effort, the single knob sent identically to every model.

    Effort rather than a token budget, even though a budget would equalise the
    actual resource, because only Anthropic enforces a budget: OpenRouter
    converts it to an effort level for OpenAI, and xAI overshot a 512-token cap
    by 4x. Mixing the two would mean sending different parameters to different
    models for no gain. One instruction to everyone is the defensible protocol;
    the differing outcome is measured and reported rather than assumed away.

    ``medium`` rather than ``low`` because at ``low`` this field barely
    deliberates -- measured on real game prompts, Claude Fable 5.1 reported
    zero reasoning tokens and GPT-6 Astra about 17. A deception leaderboard
    where two of three models answered reflexively would not mean much.
    Measured reasoning per call at ``medium``: astra 152, fable 269, grok 1589.
    """

    max_output_tokens: int = 8000
    """Ceiling per call.

    A ceiling, not a target -- unused headroom costs nothing, while hitting it
    wastes the whole generation. Sized for a model that reasons freely despite
    a low budget, which is the observed failure mode.
    """

    run_dir: Path = field(default=Path("runs"))

    def ordered_pairs(self) -> list[tuple[str, str]]:
        """Every (informant, chooser) pairing of distinct models."""
        return [(a, b) for a in self.models for b in self.models if a != b]

    def total_matches(self) -> int:
        """Number of matches the full tournament will play."""
        return len(self.ordered_pairs()) * self.matches_per_pair

    def specs(self) -> list[ModelSpec]:
        """Resolve the configured model keys to their specs."""
        unknown = sorted(set(self.models) - set(MODELS))
        if unknown:
            raise KeyError(f"unknown model keys {unknown}; known: {sorted(MODELS)}")
        return [MODELS[k] for k in self.models]
