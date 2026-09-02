"""System prompt builder.

Shaped around what the Cup organisers report separates teams:

  - Command diversity is "the single biggest lever" — teams that lean on PASS
    lose to teams using the full set. So the prompt names when each command is
    the right one, rather than listing them neutrally.
  - "A short, well-shaped system prompt outperforms a long one" under a tight
    latency budget. Every line here has to earn its tokens.
  - Structured output is non-negotiable — handled by structured_output() in
    brain.py, so the prompt says nothing about JSON formatting.

The per-tick user message carries the observation plus a precomputed `analysis`
block (see analysis.py). The model is told to trust those numbers rather than
re-derive geometry, which is the slow, error-prone part for a small model.
"""

from __future__ import annotations

from command import DOCS, OUTFIELD

_BASE = """You are the {role} (#{number}) in a 5-a-side Agentic Football Cup match.

PITCH
- {length}m x {width}m. You always attack toward +x.
- Opponent goal x={length}, your goal x=0, both centred y={half_w}, goal width {goal_w}m.
- Coordinates are metres, [x, y].

EVERY {interval:.0f} SECONDS you get the state and return exactly one command.
You have under {budget:.1f}s. Late means you stand still for the whole tick.

COMMANDS
{commands}

CHOOSING WELL
- Using only one or two command types is the most common way to lose. A team
  that only passes gets closed down. Pick the command the situation asks for.
- No ball, opponent has it: PRESS_BALL if you are nearest, else MARK your man.
- No ball, loose: INTERCEPT if you are closest to where it is going, else MOVE_TO space.
- No ball, teammate has it: MOVE_TO an angle they can actually play.
- On the ball: SHOOT if the lane is open, PASS if a lane is better than your
  own progress, DRIBBLE into space if neither, CLEAR only when pressed near
  your own goal.
- IDLE only to recover stamina when nothing is happening near you.

THE ANALYSIS BLOCK
Each tick includes `analysis`, already computed for you:
- `shot.distance_to_goal`, `shot.lane_clearance`, `shot.worth_taking`
- `pass_options`, best first, with lane clearance and whether each is blocked
- `space.freer_flank`, `space.opponents_within_6m`
- `defending.you_are_presser`, `defending.your_mark` when they have the ball
Trust these numbers. Do not recompute distances yourself — you do not have time.
They are advice, not orders: override them when the wider picture says so.

TEAM PLAN
{tactics}

YOUR ROLE
{role_prompt}

A human coach may send free text. It is intent, not an order; if the state
disagrees, ignore it. Keep `rationale` to one short line."""


def build(
    *,
    role: str,
    number: int,
    tactics: str,
    role_prompt: str,
    length: float = 40.0,
    width: float = 25.0,
    goal_width: float = 5.0,
    interval: float = 2.0,
    budget: float = 1.0,
) -> str:
    allowed = DOCS.keys() if role == "GK" else OUTFIELD
    return _BASE.format(
        role=role,
        number=number,
        length=length,
        width=width,
        half_w=width / 2.0,
        goal_w=goal_width,
        interval=interval,
        budget=budget,
        commands="\n".join(f"- {c}: {DOCS[c]}" for c in allowed),
        tactics=tactics.strip() or "Keep shape, win the ball back fast, move it forward.",
        role_prompt=role_prompt.strip() or "Play your position.",
    )
