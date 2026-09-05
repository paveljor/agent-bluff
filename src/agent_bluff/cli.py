"""Command line interface.

bluff validate    check the configured model ids against OpenRouter
bluff pilot       play one match per ordered pair, then extrapolate cost
bluff run         play the full tournament, resuming if interrupted
bluff analyze     print the leaderboards and the round-by-round lie rate
bluff transcripts dump readable dialogue for quoting
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agent_bluff.analysis import (
    belief_response,
    head_to_head,
    lie_rate_by_round,
    model_stats,
    run_cost,
)
from agent_bluff.client import OpenRouterClient, SpendLedger
from agent_bluff.config import DEFAULT_FIELD, MODELS, RunConfig
from agent_bluff.records import MatchRecord
from agent_bluff.runner import run_tournament
from agent_bluff.storage import MatchLog

MODELS_URL = "https://openrouter.ai/api/v1/models"


def _api_key() -> str:
    """Read the OpenRouter key from the environment.

    Returns:
        The API key.

    Raises:
        SystemExit: If no key is configured.
    """
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


def _config(args: argparse.Namespace) -> RunConfig:
    """Build a run configuration from parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        The configuration for this invocation.
    """
    return RunConfig(
        models=tuple(args.models),
        rounds=args.rounds,
        matches_per_pair=args.matches_per_pair,
        reasoning_effort=args.effort,
        word_cap=args.word_cap,
        max_concurrency=args.concurrency,
        budget_usd=args.budget,
        run_dir=Path(args.run_dir),
    )


def cmd_validate(args: argparse.Namespace) -> int:
    """Check every configured model id against the live OpenRouter catalogue.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    response = httpx.get(MODELS_URL, timeout=30.0)
    response.raise_for_status()
    available = {entry["id"] for entry in response.json()["data"]}

    ok = True
    for key in args.models:
        spec = MODELS[key]
        mark = "ok " if spec.id in available else "MISSING"
        ok &= spec.id in available
        print(f"  {mark}  {key:10s} {spec.id}")
    if not ok:
        print("\nSlugs change when providers ship new versions; fix config.py.", file=sys.stderr)
    return 0 if ok else 1


async def _play(config: RunConfig) -> None:
    """Run a tournament to completion.

    Args:
        config: The run configuration.
    """
    ledger = SpendLedger(config.budget_usd)
    log = MatchLog(config.run_dir)
    async with OpenRouterClient(_api_key(), ledger) as client:
        await run_tournament(client, config, ledger, log)


def cmd_run(args: argparse.Namespace) -> int:
    """Play the full tournament.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config = _config(args)
    print(
        f"{len(config.models)} models, {len(config.ordered_pairs())} ordered pairs, "
        f"{config.total_matches()} matches, {config.total_matches() * config.rounds} rounds",
        file=sys.stderr,
    )
    asyncio.run(_play(config))
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """Play one match per ordered pair, then extrapolate the full run's cost.

    Every model and both role assignments get exercised once, so this shakes
    out provider quirks and produces a cost estimate from measured tokens
    rather than assumed ones.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    full = _config(args)
    pilot = RunConfig(
        models=full.models,
        rounds=full.rounds,
        matches_per_pair=1,
        reasoning_effort=full.reasoning_effort,
        word_cap=full.word_cap,
        max_concurrency=full.max_concurrency,
        budget_usd=min(full.budget_usd, args.pilot_budget),
        run_dir=full.run_dir / "pilot",
    )
    asyncio.run(_play(pilot))

    matches = list(MatchLog(pilot.run_dir).read())
    if not matches:
        print("no matches completed", file=sys.stderr)
        return 1

    played = [m for m in matches if m.rounds]
    per_match = run_cost(played) / len(played)
    projected = per_match * full.total_matches()
    failures = [m for m in matches if m.error]

    print(f"\n  measured   ${per_match:.3f} per match over {len(played)} matches")
    if set(full.models) == set(DEFAULT_FIELD):
        print(f"  projected  ${projected:.2f} for the full {full.total_matches()}-match run")
    else:
        # Per-match cost is model-specific; projecting stand-in prices onto the
        # real field would produce a confident and badly wrong number.
        print(f"  projection SKIPPED: {', '.join(full.models)} are not the tournament field.")
        print("             This run proves the plumbing, not the cost.")
    if failures:
        print(f"  {len(failures)} match(es) errored:", file=sys.stderr)
        for match in failures:
            print(f"    {match.seed}: {match.error}", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Print the leaderboards and the headline chart data.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    matches = list(MatchLog(Path(args.run_dir)).read())
    if not matches:
        sys.exit(f"no matches in {args.run_dir}")

    stats = model_stats(matches)
    rounds = sum(len(m.rounds) for m in matches)
    print(f"{len(matches)} matches, {rounds} rounds, ${run_cost(matches):.2f} spent\n")

    print("DECEPTION -- win rate as informant (* = interval excludes chance)")
    for s in sorted(stats.values(), key=lambda s: -s.deception.value):
        star = "*" if s.deception.beats_chance else " "
        print(f"  {s.key:10s} {s.deception} {star}   lie rate {s.lie_rate}")

    print("\nDETECTION -- win rate as chooser, which is its decision accuracy")
    for s in sorted(stats.values(), key=lambda s: -s.detection.value):
        star = "*" if s.detection.beats_chance else " "
        print(
            f"  {s.key:10s} {s.detection} {star}   switch {s.switch_rate}  "
            f"brier {s.brier:.3f}"
        )

    print("\nREASONING -- asked of every model identically; honoured differently")
    for s_ in sorted(stats.values(), key=lambda s: -s.reasoning_per_call):
        print(f"  {s_.key:10s} {s_.reasoning_per_call:8.0f} tokens/call over {s_.calls} calls")
    print("  A ranking that tracks this column may be measuring thinking budget.")

    print("\nLIE RATE BY ROUND -- does trust get built early and spent late?")
    for index, prop in enumerate(lie_rate_by_round(matches), start=1):
        bar = "#" * round(prop.value * 40)
        print(f"  r{index:<3d} {prop}  {bar}")

    print("\nHEAD TO HEAD -- informant win rate, row bluffs column")
    keys = sorted(stats)
    h2h = head_to_head(matches)
    print("            " + "".join(f"{k:>12s}" for k in keys))
    for row in keys:
        cells = "".join(
            f"{h2h[row, col].value:>12.3f}" if (row, col) in h2h else f"{'--':>12s}"
            for col in keys
        )
        print(f"  {row:10s}{cells}")

    print("\nCLAIM vs RESPONSE")
    table = belief_response(matches)
    for lied in (False, True):
        label = "lied      " if lied else "told truth"
        kept, switched = table.get((lied, False), 0), table.get((lied, True), 0)
        print(f"  {label}  chooser kept {kept:4d}   switched {switched:4d}")

    evasions = sum(s.evasions for s in stats.values())
    if evasions:
        print(f"\n  {evasions} round(s) ended with no checkable claim (evasive informant)")
    errored = [m for m in matches if m.error]
    if errored:
        print(f"  {len(errored)} match(es) ended early; see 'error' in matches.jsonl")
    return 0


def cmd_transcripts(args: argparse.Namespace) -> int:
    """Print readable dialogue, for finding the quotable exchanges.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    matches: list[MatchRecord] = list(MatchLog(Path(args.run_dir)).read())
    if args.pair:
        informant, chooser = args.pair.split(":")
        matches = [
            m for m in matches if m.informant_model == informant and m.chooser_model == chooser
        ]

    for match in matches[: args.limit]:
        print(f"\n{'=' * 78}\n{match.informant_model} bluffs {match.chooser_model} "
              f"(match {match.match_index}, seed {match.seed})\n{'=' * 78}")
        for rnd in match.rounds:
            box = "REWARD" if rnd.informant_has_reward else "EMPTY "
            verdict = "informant" if rnd.informant_won else "chooser"
            print(f"\n-- round {rnd.index + 1}  [informant box: {box}]  winner: {verdict}")
            print(f"  I: {rnd.informant_open}")
            print(f"  C: {rnd.chooser_reply}")
            print(f"  I: {rnd.informant_close}   (claims: {rnd.close_claim.value})")
            action = "SWITCH" if rnd.switched else "KEEP"
            print(
                f"  C: {action} "
                f"(p={rnd.confidence_informant_has_reward:.2f}) {rnd.chooser_reasoning}"
            )
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    """Render the figures for a run.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    try:
        from agent_bluff.charts import write_lie_rate
    except ImportError:
        sys.exit("matplotlib is not installed. Install the extra: pip install -e '.[viz]'")

    matches = list(MatchLog(Path(args.run_dir)).read())
    if not matches:
        sys.exit(f"no matches in {args.run_dir}")
    for path in write_lie_rate(matches, Path(args.out)):
        print(f"  wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="bluff", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        """Attach the arguments common to every command that plays matches."""
        p.add_argument("--models", nargs="+", default=list(RunConfig().models),
                       choices=list(MODELS))
        p.add_argument("--rounds", type=int, default=RunConfig().rounds)
        p.add_argument("--matches-per-pair", type=int, default=RunConfig().matches_per_pair)
        p.add_argument("--effort", default=RunConfig().reasoning_effort,
                       choices=["low", "medium", "high"],
                       help="reasoning effort, sent identically to every model; "
                            "the dominant cost driver")
        p.add_argument("--word-cap", type=int, default=RunConfig().word_cap)
        p.add_argument("--concurrency", type=int, default=RunConfig().max_concurrency)
        p.add_argument("--budget", type=float, default=RunConfig().budget_usd,
                       help="hard ceiling in USD; the run stops rather than exceed it")
        p.add_argument("--run-dir", default="runs/main")

    validate = sub.add_parser("validate", help="check model ids against OpenRouter")
    validate.add_argument("--models", nargs="+", default=list(RunConfig().models),
                          choices=list(MODELS))
    validate.set_defaults(func=cmd_validate)

    pilot = sub.add_parser("pilot", help="one match per pair, then project the full cost")
    shared(pilot)
    pilot.add_argument("--pilot-budget", type=float, default=5.0)
    pilot.set_defaults(func=cmd_pilot)

    run = sub.add_parser("run", help="play the full tournament")
    shared(run)
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze", help="print leaderboards and metrics")
    analyze.add_argument("--run-dir", default="runs/main")
    analyze.set_defaults(func=cmd_analyze)

    chart = sub.add_parser("chart", help="render figures from a run")
    chart.add_argument("--run-dir", default="runs/main")
    chart.add_argument("--out", default="figures")
    chart.set_defaults(func=cmd_chart)

    transcripts = sub.add_parser("transcripts", help="dump readable dialogue")
    transcripts.add_argument("--run-dir", default="runs/main")
    transcripts.add_argument("--pair", help="filter to 'informant:chooser', e.g. astra:grok")
    transcripts.add_argument("--limit", type=int, default=3)
    transcripts.set_defaults(func=cmd_transcripts)

    return parser


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
