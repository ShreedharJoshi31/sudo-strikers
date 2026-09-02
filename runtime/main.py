"""Amazon Bedrock AgentCore Runtime entrypoint.

    POST /invocations
    body: {"player_id": "H3", "observation": {...}}
    resp: AgentCommand JSON

One runtime serves all five players — the platform sends `player_id` with every
call, and role and number come from the observation. Deploy five copies only if
the competition requires a separate endpoint per player (`routing: per_player`);
nothing in the code needs to change for that.

Local check:
    python runtime/main.py     # listens on :8080
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Deployed layout puts lib/ next to this file; local dev has it one level up.
for candidate in (os.path.join(_HERE, "lib"), os.path.join(_HERE, "..", "lib")):
    if os.path.isdir(candidate):
        sys.path.insert(0, os.path.normpath(candidate))
        break

# Settings baked in by deploy.sh. A deployed runtime does NOT inherit the shell
# that deployed it, so anything exported at deploy time is otherwise lost and
# every AFC_* setting silently reverts to its built-in default. This must run
# BEFORE brain is imported, because brain resolves the per-role model map at
# import time. setdefault, not assignment: a variable genuinely set on the
# runtime still wins over the baked-in value.
try:
    from afc_env import SETTINGS as _BAKED  # generated at deploy time
except ImportError:  # local dev, or a deploy that baked nothing
    _BAKED = {}
for _key, _value in _BAKED.items():
    os.environ.setdefault(_key, str(_value))

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402

from brain import DEFAULT_MODELS, Squad  # noqa: E402

app = BedrockAgentCoreApp()

USE_LLM = os.environ.get("AFC_USE_LLM", "1").lower() in ("1", "true", "yes")

TACTICS = os.environ.get("AFC_TACTICS", """
Win the ball back inside three seconds of losing it.
Move it forward early; short and clean beats long and hopeful.
Shoot when the lane is open rather than working for a better angle.
""")

ROLE_PROMPTS = {
    "GK": "Hold the angle between ball and goal. Claim loose balls in the area. Restart fast to the most advanced open teammate.",
    "DEFENDER": "Hold the back line. Do not both step up at once - read your partner.",
    "MIDFIELDER": "Link the lines. Offer an angle when a teammate has it, screen the middle when they do not.",
    "FORWARD": "Attack the space behind the last defender. Shoot early when the lane opens.",
}

squad = Squad(tactics=TACTICS, role_prompts=ROLE_PROMPTS, use_llm=USE_LLM)

# Printed once at cold start so a deployed runtime's model map shows up in
# CloudWatch. Otherwise the only way to find out which model a position is
# actually running is to lose a match and go digging.
print(f"[afc] llm={USE_LLM} models={DEFAULT_MODELS}", flush=True)


@app.entrypoint
def invoke(payload: dict) -> list[dict]:
    """The platform sends {"gameState": ..., "teamId": N, "myPlayers": [i]} and
    expects a JSON ARRAY of commands back."""
    return squad.handle(payload)


if __name__ == "__main__":
    app.run()
