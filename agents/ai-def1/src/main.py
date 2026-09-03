"""AI Defender agent (left of the pair).

Controls player 1 only. One AgentCore runtime per position, so each can run
its own model and fail independently of the other four.

    POST /invocations
    body: {"gameState": ..., "teamId": N, "myPlayers": [1]}
    resp: JSON array of commands

Everything shared lives in lib/. The three things below are what make this
agent different from its four teammates - edit them, not the plumbing.

Local check:
    python src/main.py      # listens on :8080
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Deployed layout puts lib/ beside src/; local dev has it three levels up.
for _candidate in (os.path.join(_HERE, "..", "lib"),
                   os.path.join(_HERE, "..", "..", "..", "lib")):
    if os.path.isdir(_candidate):
        sys.path.insert(0, os.path.normpath(_candidate))
        break

# Settings baked in by deploy-all.sh. A deployed runtime does NOT inherit the
# shell that deployed it, so every AFC_* setting would otherwise revert to its
# built-in default. Must run BEFORE brain is imported, because brain resolves
# the per-role model map at import time. setdefault, not assignment: a variable
# genuinely set on the runtime still wins.
try:
    from afc_env import SETTINGS as _BAKED
except ImportError:                                  # local dev, or nothing baked
    _BAKED = {}
for _key, _value in _BAKED.items():
    os.environ.setdefault(_key, str(_value))

from bedrock_agentcore.runtime import BedrockAgentCoreApp   # noqa: E402

from runtime_app import build                               # noqa: E402

app = BedrockAgentCoreApp()

# --- what makes this agent this agent ------------------------------------
MY_PLAYER_INDEX = 1
ROLE = "DEFENDER"
ROLE_PROMPT = """You are the left of two centre-backs. Hold the line and read your partner: never
step up at the same time as they do. Mark the runner the shared plan gives you and
trust the other four to take theirs. Clear only when genuinely pressed near goal."""
# -------------------------------------------------------------------------

squad = build(
    app,
    player_index=MY_PLAYER_INDEX,
    role=ROLE,
    role_prompt=ROLE_PROMPT,
)

if __name__ == "__main__":
    app.run()
