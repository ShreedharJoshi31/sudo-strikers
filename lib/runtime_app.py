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

import brain
import inbound
from brain import DEFAULT_MODELS, Squad

#: How much of each inbound payload to print. Enough to see the shape and the
#: first players; not so much that a match floods CloudWatch.
PAYLOAD_LOG_CHARS = int(os.environ.get("AFC_PAYLOAD_LOG_CHARS", "4000"))

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
    def invoke(payload, context=None) -> list[dict]:
        """One platform invocation.

        The second parameter MUST be named `context`: the runtime decides
        whether to pass a RequestContext by looking at that name, and the
        context is where the session id lives. The session id is how we know
        which slot we are being played in, which the pinned index cannot tell
        us when one runtime is registered in two positions.

        The payload is logged raw and unconditionally. The failure this exists
        to prevent is silent: an unreadable payload does not raise, it returns
        the safe default in microseconds and looks like a healthy 200.

        This MUST be a generator. The platform reads the last `data:` line of
        an SSE stream; a plain return is served as application/json and is
        rejected with NO_PARSE.

        YIELD THE LIST, NOT A JSON STRING. The runtime already serialises
        whatever you yield (`_convert_to_sse` -> `json.dumps(obj)`), so
        `yield json.dumps(cmds)` double-encodes into
        `data: "[{\\"commandType\\": ...}]"` - a JSON *string* where the
        platform wants a JSON *array* - and that is a NO_PARSE too. The
        platform's own error text recommends the double-encoding; it is wrong
        for this SDK version. Yield the list and let the runtime encode once.

        Nothing may be written to stdout after the yield.
        """
        session_id = getattr(context, "session_id", "") or ""
        try:
            raw = json.dumps(payload, default=str)
        except Exception:
            raw = repr(payload)
        print(f"[afc:in] player={player_index} session={session_id!r} "
              f"bytes={len(raw)} payload={raw[:PAYLOAD_LOG_CHARS]}", flush=True)

        try:
            canonical, notes = inbound.normalise(
                payload, default_index=player_index, session_id=session_id)
            index = canonical["myPlayers"][0]
            if notes:
                print(f"[afc:in] index={index} team={canonical['teamId']} "
                      f"notes={'; '.join(notes)}", flush=True)
            commands = list(squad.handle(canonical, my_index=index))
        except Exception as exc:
            print(f"[afc:in] UNREADABLE ({type(exc).__name__}: {exc}) "
                  f"-> safe default", flush=True)
            commands = list(squad.handle(payload, my_index=player_index))

        print(f"[afc:out] {json.dumps(commands)}", flush=True)
        yield commands

    # Printed at cold start so the deployed runtime's identity and model show up
    # in CloudWatch. Otherwise the only way to find out which model a position
    # actually ran is to lose a match and go digging.
    # Every feature flag, not just the model. These default to ON and are baked
    # at deploy time, so the only way to know what a running agent actually has
    # enabled is to print it — and finding out from a lost match is too late.
    gw = squad.gateway.status()
    print(
        f"[afc] player={player_index} role={role} llm={enabled} "
        f"model={DEFAULT_MODELS.get(role)} deadline={squad.deadline}s "
        f"analysis={brain.USE_ANALYSIS} memory={brain.USE_MEMORY} "
        f"scouting={brain.USE_SCOUTING} "
        f"gateway={'configured' if gw['configured'] else 'off'} "
        # The build stamp matters as much as the model here: when five runtimes
        # disagree, the first question is always "are they even running the
        # same code?", and answering that from CloudWatch alone beats
        # reconstructing it from deploy logs afterwards.
        f"build={os.environ.get('AFC_BUILD_SHA', 'unstamped')} "
        f"lib={os.environ.get('AFC_LIB_HASH', 'unknown')}",
        flush=True,
    )
    return squad
