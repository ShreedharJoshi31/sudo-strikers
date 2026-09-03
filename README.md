# afc-contender

A competition team for the Agentic Football Cup, assembled from the two source
repos you had:

- **`aws-agentinc-football/agentic-football-sample-agents`** — per-position
  specialisation, staged deploy with a single shared `lib/`, layered fallback.
- **`agentic-football-arena`** — the platform contract, structured output,
  stateless per-tick inference, normalised coordinates, and the baseline AI
  whose thresholds seed `lib/policy.py`.

It speaks the real platform contract, handled in `lib/wire.py`:

```
POST /invocations
body: {"gameState": {...}, "teamId": 0, "myPlayers": [3]}
resp: [{"commandType": "PASS", "playerId": 3,
        "parameters": {"target_player_id": 4, "type": "GROUND"}, "duration": 0.0}]
```

## The one idea

A decision that misses the platform's budget does **not** idle the player — it
leaves their LAST command running. A bad or late answer is therefore not a
wasted tick, it is a wasted tick that keeps going.

So the model is never on the critical path:

```
handle(payload)
  │
  ├─ 1. policy.decide(...)      microseconds, always a valid command
  ├─ 2. ask the model           hard deadline at AFC_LLM_DEADLINE (1.8 s)
  └─ 3. use the model's answer only if it arrived in time and passed checks
```

A late model answer costs nothing, because a valid command already exists.
Without step 1 the player carries on running a stale order.

## Layout

```
afc-contender/
├── agents/                  five deployable agents, one per position
│   ├── ai-gk/src/main.py        player 0  GK
│   ├── ai-def1/src/main.py      player 1  DEFENDER
│   ├── ai-def2/src/main.py      player 2  DEFENDER
│   ├── ai-mid/src/main.py       player 3  MIDFIELDER
│   └── ai-fwd/src/main.py       player 4  FORWARD
├── lib/                     shared by all five — single source of truth
│   ├── command.py     AgentCommand — the platform contract
│   ├── wire.py        payload in, wire commands out
│   ├── policy.py      pure-Python policy; thresholds live in Params
│   ├── passing.py     interception model
│   ├── scouting.py    per-opponent profiling
│   ├── analysis.py    derived features pushed into the prompt
│   ├── memory.py      per-player short-term memory
│   ├── gateway.py     MCP tools the model can pull
│   ├── prompts.py     system prompt builder
│   ├── runtime_app.py wiring shared by the five entrypoints
│   └── brain.py       Squad: deadline-guarded LLM over the policy
├── gateway_tools/           four Lambda handlers behind AgentCore Gateway
├── serve.py                 local HTTP server (serves all five in one process)
├── team.yaml                team definition
├── test_local.py            no AWS, no LLM, no server
├── bench.py                 deterministic matches + parameter sweeps
├── deploy-all.sh            stage + deploy the five agents
├── manage_gateway.py        create the Gateway, register the tools
└── deploy_gateway.sh        package the Lambdas, then the above
```

Each agent file declares only three things — `MY_PLAYER_INDEX`, `ROLE` and
`ROLE_PROMPT`. Everything else is in `lib/`, so a fix lands once rather than in
four files out of five. `deploy-all.sh` copies `lib/` into each bundle at deploy
time; it is never duplicated in the tree.

The index is **pinned** in the agent file rather than read from `myPlayers`, so
a runtime deployed as the keeper stays the keeper even if routing sends it
someone else. A disagreement is recorded in `/stats` rather than silently
followed.

## Run it

Everything below works with no AWS account and no LLM.

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

**1. Local tests** — 16 game situations, GK behaviour, latency:

```bash
./.venv/bin/python test_local.py
```

**2. Serve the team and play a match.** In one terminal:

```bash
./.venv/bin/python serve.py
```

In another (from the arena directory):

```bash
./.venv/bin/python -m afc.cli.main check --team ../afc-contender/team.yaml
./.venv/bin/python -m afc.cli.main match --home ../afc-contender/team.yaml --away teams/crimson-rovers.yaml
```

**3. Watch it in 3D:**

```bash
./.venv/bin/python -m afc.cli.main serve --port 8000   # then open http://127.0.0.1:8000
```

## Tune it

`bench.py` runs the engine in-process, which matters: matches driven over HTTP
are **not** reproducible, because whether a response lands inside the decision
window depends on real network timing. The same seed gives different scores.
In-process runs are deterministic, so a parameter change is the only thing that
moved the result.

```bash
# run it with the arena's venv, which has the engine installed
../agentic-football-arena/.venv/bin/python bench.py --matches 10
../agentic-football-arena/.venv/bin/python bench.py --matches 8 --sweep gk_depth_base=1.0,2.2,3.5
```

Both default to playing **every** opponent, because tuning against one
formation overfits to it — measured below. The sweep prints a per-opponent
spread; a value that wins on average but collapses against one formation is a
bad bet.

Every threshold in `lib/policy.py` is a field on `Params`, so anything is
sweepable by name. Read the record, but read `timeouts` first — timeouts are
lost decisions and no amount of tactical tuning compensates for them.

## What the model is given

Three things, on top of the raw observation:

**Tools — as local functions, not remote calls.** [analysis.py](lib/analysis.py)
computes the same four things the workshop exposes through Gateway — pass
options, shot evaluation, open space, defensive assignment — in 0.02 ms with no
MCP round trip. They arrive as an `analysis` block in the model's input. The
division of labour is deliberate: Python does geometry exactly in microseconds,
a small model does it badly in a second. Give it the measurements, let it spend
its budget on judgement.

**State — in process, not AgentCore Memory.** [memory.py](lib/memory.py) keeps
the last four decisions per player and flags a player repeating itself, which a
single-tick observation cannot show. AgentCore Memory would put a retrieval
inside the decision budget; this costs nothing. Note that the platform already
supplies `last_command`, `recent_events`, score and stamina every tick — that is
not duplicated here.

**A prompt shaped around command diversity.** The organisers report it is "the
single biggest lever" and that teams leaning on `PASS` lose, so
[prompts.py](lib/prompts.py) names which command each situation calls for
instead of listing them neutrally.

Both extras are additive — if the model ignores them it still has everything it
had before — and both are switchable, because **neither has been measured
against a real model**:

```bash
AFC_ANALYSIS=0 ...   # drop the derived features
AFC_MEMORY=0 ...     # drop the per-player history
```

Measure before believing either helps.

## Add the LLM

```bash
./.venv/bin/pip install -r requirements-llm.txt
AFC_USE_LLM=1 AFC_MODEL_ID=amazon.nova-micro-v1:0 ./.venv/bin/python serve.py
curl -s localhost:8081/stats     # llm_share, llm_late, p95_ms
```

`llm_late` is the number the deadline is protecting you from. If it is high,
lower `AFC_LLM_DEADLINE` or use a faster model — the policy absorbs the misses
either way, so the team keeps playing.

Start on Nova Micro everywhere. Promote a position to a bigger model only after
`/stats` shows it has budget headroom to spare.

## Deploy

```bash
pip install bedrock-agentcore-starter-toolkit
AWS_DEFAULT_REGION=us-east-1 ./deploy-all.sh          # all five
AWS_DEFAULT_REGION=us-east-1 ./deploy-all.sh ai-gk    # just one
```

Five separate AgentCore runtimes, one per position. That buys per-position model
choice (`AFC_MODEL_MIDFIELDER=...` only affects the midfielder) and independent
failure — a broken forward does not take the defence down with it. The script
prints the ARN for each player index at the end.

Locally, `serve.py` still serves all five from one process; there is nothing to
deploy to test.

## Where it stands

48 matches, 12 against each baseline team, home and away alternating,
formation **2-1-1**:

| Opponent | Formation | Record | ppg |
|---|---|---|---|
| Azure Arrows | 1-3 | 9W 1D 2L | 2.33 |
| Crimson Rovers | 2-2 | 10W 1D 1L | 2.58 |
| Solar Storm | 3-1 | 9W 2D 1L | 2.42 |
| Verdant Vipers | 1-2-1 | 9W 2D 1L | 2.42 |
| **total** | | **37W 6D 5L** | **2.44** (81%) |

118 goals for, 51 against. **Zero timeouts.** Winning record against all four.

## Things measured here, not assumed

- The policy answers in **0.0025 ms**, about five orders of magnitude inside the
  1-second budget. The arena's `check` reports 1–11 ms end-to-end over HTTP.
- **GK depth was worth more than everything else tried.** Moving the keeper from
  2.2 m off its line to 1.0 m took the record against Crimson Rovers from
  6W 4D 10L to 19W 4D 7L over 30 matches. At a 2-second interval a keeper that
  steps out cannot get back before the next decision.
- **`GK_DIVE` is close to unreachable.** A shot from 12 m covers the distance in
  0.46 s while decisions are 2 s apart, so a dive opportunity lands on a
  decision point about once a match before any other condition applies. Sweeping
  `gk_dive_range` over 6–18 changed nothing at all. GK *positioning* is what
  saves goals; this also means an agent that never emits `GK_DIVE` is not
  meaningfully handicapped.
- **Single-opponent tuning overfits.** The GK change was worth 2.12 ppg against
  Crimson Rovers but only 1.20 against Azure Arrows. `bench.py` therefore runs
  the whole field by default and prints the per-opponent spread.
- **Formation mattered more than any tactical parameter.** 2-2 — picked from the
  arena's description rather than measured — scored 54% over 48 matches. 2-1-1
  scored 71% with the same policy. Do not take a formation on description.
- **The midfielder should hold, not push.** `mid_push=-2.0` (sitting just behind
  the ball when a teammate has it) beat `+2.0` by 2.59 to 2.03 ppg, evenly
  across all four opponents.
- Matches over HTTP are non-reproducible; in-process matches are. Tune
  in-process, then verify over the transport you will compete on.
- The LLM path is covered by tests using a fake model — deadline miss, model
  exception, invalid command, and feature injection all verified without AWS.
  What is **not** verified is whether a real model beats the policy.

## Credit

`lib/policy.py` is adapted from the Agentic Football Arena baseline AI (MIT).
The arena is an unofficial fan reconstruction of the Cup — its physics, pitch
dimensions and scoring are its own design, so treat tuned *numbers* as
disposable and the *structure* (stateless, structured output, sub-budget
fallback) as the part that transfers.
