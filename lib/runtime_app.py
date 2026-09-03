"""Shared wiring for the five per-position AgentCore runtimes.

Each `agents/ai-*/src/main.py` is its own deployable agent, matching the
workshop's layout. What differs between them is exactly three things — which
player they control, what role that is, and the tactical brief for that role —
so those three live in the agent file and everything else lives here.

The alternative would be five near-identical 80-line entrypoints. Those drift:
a fix goes into four of them, the fifth quietly keeps the bug, and the symptom
is one player behaving oddly for a whole match. Keeping the mechanism in one
place means the agent files stay short enough to actually read and edit, which
is the point of splitting them up.
"""

from __future__ import annotations

import json
import os
import sys

from brain import DEFAULT_MODELS, Squad

#: Team-wide plan. Identical for all five on purpose: it is the shared doctrine,
#: and the per-role brief in each agent file is what specialises it.
DEFAULT_TACTICS = """
Win the ball back inside three seconds of losing it.
Move it forward early; short and clean beats long and hopeful.
Shoot when the lane is open rather than working for a better angle.
"""


def use_llm() -> bool:
    return os.environ.get("AFC_USE_LLM", "1").lower() in ("1", "true", "yes")


def build(app, *, player_index: int, role: str, role_prompt: str,
          tactics: str | None = None):
    """Register the entrypoint for one position and return its Squad.

    `player_index` is pinned rather than read from the payload, so a runtime
    deployed as the keeper stays the keeper even if routing sends it something
    else. See Squad.handle for what happens when they disagree.
    """
    tactics = tactics or os.environ.get("AFC_TACTICS", DEFAULT_TACTICS)
    enabled = use_llm()

    squad = Squad(
        tactics=tactics,
        # Only this position's brief is ever needed: this runtime resolves one
        # role and never sees the other four.
        role_prompts={role: role_prompt},
        use_llm=enabled,
    )

    @app.entrypoint
    async def invoke(payload: dict, context=None):
        """One decision, in the exact shape the platform parses.

        This MUST be a generator that yields a JSON STRING, not a function that
        returns a list. The runtime replies over SSE and reads the commands off
        the last `data:` line, so a returned object arrives as something it
        cannot parse and the whole tick comes back as NO_PARSE - which looks
        like a broken agent, not a formatting mistake.

        Nothing may be written to stdout from here on, for the same reason: any
        stray print lands in the stream after the payload and corrupts it. The
        cold-start banner below goes to stderr, which CloudWatch still captures.

        `context` is optional so the entrypoint stays callable from a test;
        AgentCore passes it, local callers do not.
        """
        yield json.dumps(squad.handle(payload, my_index=player_index))

    # Printed at cold start so the deployed runtime's identity and model show up
    # in CloudWatch. Otherwise the only way to find out which model a position
    # actually ran is to lose a match and go digging.
    print(
        f"[afc] player={player_index} role={role} llm={enabled} "
        f"model={DEFAULT_MODELS.get(role)}",
        file=sys.stderr,          # never stdout: stdout is the response stream
        flush=True,
    )
    return squad
