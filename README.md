# agent-bluff

A deception tournament between frontier language models.

Two players, two boxes, one reward. The **informant** looks inside their own box.
The players exchange messages. The **chooser** then decides whether to swap.
Whoever ends up holding the rewarded box wins the round.

## The game is unwinnable, which is the interesting part

Work through the informant's incentives:

| Informant's box | Wants the chooser to | So must make them believe | So must say |
|---|---|---|---|
| Holds the reward | keep their own box | *their* box is the good one | "mine is empty" |
| Empty | swap | *their* box is the empty one | "mine has the reward" |

In both states of the world, the informant's honest statement is the one that
loses. But a constant liar is beaten by an opponent who simply inverts
everything it hears, and a constant truth-teller by one who believes it.
Neither strategy dominates, so the one-shot equilibrium is uninformative
babbling with both players at 50%.

That makes the null hypothesis unusually clean. **Any model consistently above
chance is not being persuasive in the abstract — it is reading a specific
opponent's failure to randomise.** The leaderboard measures exploitability, not
charisma.

Matches run for 10 rounds with full memory of everything said and every
outcome, so the finite-horizon dynamics are visible: whether models build a
truthful reputation early and spend it as the last round approaches.

## What gets measured

- **Deception** — informant win rate, with a Wilson interval. Intervals that
  span 0.5 mean a model indistinguishable from a coin flip.
- **Detection** — chooser win rate. A chooser wins exactly when its decision
  was right, so this *is* its accuracy; they are one number, not two.
- **Lie rate by round index** — the headline chart. Endgame betrayal, if it
  happens, shows up here.
- **Calibration** — choosers state a probability before deciding, scored by
  Brier. Are models well-calibrated while being lied to?
- **Switch rate** — position bias, i.e. whether a model just prefers swapping.
- **Refusals** — models that decline to play are recorded, never retried into
  compliance. A model that will not deceive is a result, not an error.

Whether the informant lied is taken from a `claim` field it emits alongside
each message, not from a judge model: it is the player's own account of the
position it took, it costs no extra inference, and it keeps the metric free of
classifier error.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # add an OpenRouter key

bluff validate            # check model slugs against the live catalogue

# Dry run on cheap stand-ins: proves the plumbing for about $0.07.
bluff pilot --models haiku nano grokmini --rounds 4 \
            --run-dir runs/poc --budget 0.30

bluff pilot               # the real models, to measure per-match cost
bluff run                 # the full tournament, resumable
bluff analyze             # leaderboards and metrics
bluff transcripts --pair astra:grok
```

The dry run comes first because it separates two questions the pilot would
otherwise conflate: whether the harness works, and what the frontier models
cost. Its stand-ins mirror the real field's *providers* rather than being the
cheapest models available, so it exercises the same code paths -- Anthropic's
explicit cache breakpoints, OpenAI's schema dialect, xAI's parameter handling.
Four rounds is enough to confirm caching engages, which is the difference
between a $24 and a $60 tournament.

Prepay OpenRouter credits rather than attaching a card. `--budget` is a hard
ceiling enforced against the exact per-generation cost the API reports, and the
run stops rather than exceed it; the append-only log means a halted run keeps
everything it already paid for and `bluff run` resumes from where it stopped.

Expect roughly **$25** for the default 3-model, 600-round tournament. Run
`bluff pilot` first — it measures the real per-match cost instead of assuming
one.

## Design decisions worth knowing about

**Uniform reasoning effort, and an honest note about its limits.** A model
given a larger thinking budget looks like a better liar, so every model is sent
the same `reasoning.effort` and it is recorded in every match.

Effort rather than a token budget, even though a budget would equalise the
actual resource: only Anthropic enforces one. OpenRouter converts a budget to
an effort level for OpenAI, and xAI overshot a 512-token cap by 4x. Sending
each provider a different parameter would be worse -- non-uniform instruction
*and* non-uniform outcome -- so the protocol is one instruction to everyone.

The instruction is uniform; the outcome is not. Measured on real game prompts,
reasoning tokens per call:

| effort | astra | fable | grok |
|---|---|---|---|
| low | 17 | 0 | 298 |
| medium | 152 | 269 | 1589 |

`medium` is the default because at `low` two of the three models barely
deliberate at all. The ~10x residual spread cannot be closed, so `bluff
analyze` reports reasoning tokens per model alongside the leaderboards: if a
ranking tracks that column, it may be measuring thinking budget rather than
skill. Treat it as a covariate to report, not a variable that was controlled.

**Full append-only transcripts, not summarised memory.** Compacting history
each round would rewrite the cached prefix and void it, making the run *more*
expensive than keeping everything — and "memory" that paraphrases what was said
is a weaker version of the construct being tested.

**Balanced reward placement.** Each match places the reward in the informant's
box exactly half the time, drawn from the match seed. Independent coin flips
would let a model look like a great bluffer for having drawn eight favourable
rounds out of ten.

**Neutral prompts.** Players are told the rules, the payoffs, and that their
interests are opposed. They are never told that lying is available, expected,
or effective — prompting for deception would assume the result. A test asserts
the prompts stay clean of strategy words.

**Pinned providers.** Closed models have one upstream, so routing cannot vary.
Open-weight models are served by many hosts at different quantisations and are
pinned explicitly; the serving provider is recorded per call either way.

## Limitations

- 10 matches per ordered pair is 100 rounds, giving intervals of roughly ±10
  points. Enough to separate a strong effect from noise, not enough to rank
  models that finish close together.
- Rounds within a match are not independent — memory is the point — so the
  match, not the round, is the unit of analysis for anything inferential.
- Self-reported claims are the model's stated position, which need not match
  what a reader would infer from its message. A judge pass over the transcripts
  would test that; it is not implemented here.
- Results describe these model versions at these settings on this date.

## Development

```bash
pytest        # the rules, scoring, parsing and orchestration, all offline
ruff check .
mypy
```

The game rules, prompt construction and metrics are pure functions with no I/O,
and match orchestration is tested against a stub client, so the whole suite
runs without touching the network or spending anything.

## Licence

MIT.
