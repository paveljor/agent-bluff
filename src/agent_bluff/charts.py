"""Figure generation.

Emphasis form: round 1 is the finding, rounds 2-10 are the context that makes
it a finding. One accent hue plus a de-emphasis gray, rather than ten
categorical colours that would bury the point.

Palettes are the validated defaults; the de-emphasis gray deliberately fails
the categorical chroma floor, which is what "reads as gray" means.
"""


from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_bluff.analysis import Proportion, lie_rate_by_round
from agent_bluff.records import MatchRecord

FONT = ["DejaVu Sans", "sans-serif"]  # the sans matplotlib ships; no display or serif face

THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "accent": "#2a78d6", "context": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3987e5", "context": "#898781",
    },
}


def lie_rate_figure(
    series: Sequence[Proportion], theme: str, subtitle: str
) -> matplotlib.figure.Figure:
    """Draw the round-by-round lie rate.

    Args:
        series: One proportion per round, in order.
        theme: ``light`` or ``dark``.
        subtitle: Method line rendered under the title.

    Returns:
        The finished figure.
    """
    c = THEMES[theme]
    plt.rcParams["font.family"] = FONT

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    rounds = list(range(1, len(series) + 1))

    # Chance is the null: the game's one-shot equilibrium is uninformative.
    ax.axhline(0.5, color=c["axis"], lw=1.0, zorder=1)
    ax.text(11.15, 0.515, "chance", color=c["muted"], fontsize=9, va="bottom", ha="right")

    # No connecting line. Every interval after round 1 overlaps chance, so a
    # line would invite the reader to trace a trend through pure noise.
    for i, (rnd, p_) in enumerate(zip(rounds, series, strict=True)):
        lead = i == 0
        colour = c["accent"] if lead else c["context"]
        ax.plot([rnd, rnd], [p_.low, p_.high], color=colour,
                lw=2.4 if lead else 1.6, alpha=1.0 if lead else 0.5,
                solid_capstyle="round", zorder=3)
        ax.plot([rnd], [p_.value], "o", color=colour, ms=9 if lead else 6.5,
                markeredgecolor=c["surface"], markeredgewidth=1.6, zorder=4)

    # Direct-label the extreme only, offset clear of its own interval bar.
    first = series[0]
    ax.annotate(f"{first.value:.0%}", xy=(1, first.value), xytext=(17, -5),
                textcoords="offset points", ha="left", va="center",
                # Ink, not the series colour: the accent dot beside it carries
                # identity, so the number does not need to repeat it.
                color=c["primary"], fontsize=15, fontweight="bold")

    rest = series[1:]
    mean_rest = sum(p_.successes for p_ in rest) / sum(p_.trials for p_ in rest)
    # Caption sits in the empty band below the data, not across it.
    ax.text(0.62, 0.205, f"Every round after the first sits at chance\n"
                        f"(rounds 2\u201310 average {mean_rest:.0%}).",
            color=c["secondary"], fontsize=10.5, va="top", ha="left", linespacing=1.5)

    ax.set_title("LLMs lie hardest when they have no history",
                 color=c["primary"], fontsize=15, fontweight="bold",
                 loc="left", pad=26)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, color=c["secondary"], fontsize=9.5)

    ax.set_xlabel("Round within the match", color=c["secondary"], fontsize=10, labelpad=8)
    ax.set_ylabel("Share of claims that were false", color=c["secondary"], fontsize=10,
                  labelpad=8)
    ax.set_xticks(rounds)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0.45, 11.2)
    ax.set_ylim(-0.03, 1.03)

    ax.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)      # solid hairline, never dashed
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)

    fig.tight_layout()
    return fig


def write_lie_rate(matches: Sequence[MatchRecord], out_dir: Path) -> list[Path]:
    """Render the lie-rate figure in both themes.

    Args:
        matches: Match records from the run log.
        out_dir: Directory to write PNGs into.

    Returns:
        Paths written.
    """
    series = lie_rate_by_round(matches)
    n_rounds = sum(p.trials for p in series)
    subtitle = (
        f"{len(matches)} matches between GPT-6 Astra, Claude Fable 5.1 and Grok 4.6 · "
        f"{n_rounds} checkable claims · bars are 95% Wilson intervals"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        fig = lie_rate_figure(series, theme, subtitle)
        path = out_dir / f"lie-rate-by-round-{theme}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        written.append(path)
    return written
