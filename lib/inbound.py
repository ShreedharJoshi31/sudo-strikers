"""Normalise whatever the platform actually sends into the shape we expect.

Why this exists
---------------
`runtime_app.py` referenced this module but it was never committed, so the
branch could not import. It is reconstructed here from its call site, which
needs exactly one function returning `(canonical, notes)`.

The job is narrow on purpose: get `gameState`, `teamId` and `myPlayers` into
the shape `wire.to_observation` reads, and REPORT anything that had to be
guessed. It does not repair game state or invent players — a payload we cannot
read should surface as a note and fall back to the pinned index, not be quietly
patched into something plausible.

Every inference is recorded in `notes` and logged by the caller. That matters
because the failure this guards against is silent: a payload whose shape drifts
does not raise, it produces a valid-looking command for the wrong player.
"""

from __future__ import annotations

from typing import Any

#: Keys the platform has been observed to use for the controlled-player list.
_MY_PLAYER_KEYS = ("myPlayers", "my_players", "myPlayerIds", "controlledPlayers")

#: Keys for the team id.
_TEAM_KEYS = ("teamId", "team_id", "team")

#: Keys the game state has been seen under.
_STATE_KEYS = ("gameState", "game_state", "state")


def _first_key(payload: dict, keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for k in keys:
        if k in payload:
            return k, payload[k]
    return None, None


def _to_index(value: Any) -> int | None:
    """Parse a player index out of any spelling the platform might use.

    Accepts 3, "3", "agentId_3" and "home_3". Returns None rather than a
    default, so the caller decides what a missing index means — guessing 0 here
    would silently make every unreadable payload the goalkeeper's problem.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 4 else None
    if isinstance(value, str):
        tail = value.rsplit("_", 1)[-1].strip()
        if tail.isdigit():
            idx = int(tail)
            return idx if 0 <= idx <= 4 else None
    return None


def _team_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("0", "home"):
            return 0
        if v in ("1", "away"):
            return 1
    return None


def normalise(
    payload: dict,
    *,
    default_index: int,
    session_id: str = "",
) -> tuple[dict, list[str]]:
    """Return `(canonical, notes)`.

    `canonical` always has `gameState`, `teamId` and `myPlayers`, so the caller
    can hand it straight to `Squad.handle`. `notes` lists everything that was
    renamed, coerced or defaulted, and is empty when the payload arrived in the
    expected shape — so a quiet log means nothing needed guessing.

    `session_id` is accepted because the caller has it, and is recorded when a
    guess had to be made. It is deliberately NOT used to choose a slot: how the
    platform maps sessions to positions is not documented anywhere we have, and
    inventing that mapping would be a wrong player rather than a missing one.
    """
    notes: list[str] = []
    if not isinstance(payload, dict):
        return (
            {"gameState": {}, "teamId": 0, "myPlayers": [default_index]},
            [f"payload was {type(payload).__name__}, not an object"],
        )

    state_key, state = _first_key(payload, _STATE_KEYS)
    if state_key is None:
        # A payload with no state at all is unusable. Say so; the caller falls
        # back to the safe default, which is still a current, valid command.
        notes.append("no gameState found")
        state = {}
    elif state_key != "gameState":
        notes.append(f"gameState found under {state_key!r}")
    if not isinstance(state, dict):
        notes.append(f"gameState was {type(state).__name__}, not an object")
        state = {}

    team_key, team_raw = _first_key(payload, _TEAM_KEYS)
    team = _team_id(team_raw)
    if team is None:
        notes.append(f"teamId unreadable ({team_raw!r}); assuming 0/home")
        team = 0
    elif team_key != "teamId":
        notes.append(f"teamId found under {team_key!r}")

    mine_key, mine_raw = _first_key(payload, _MY_PLAYER_KEYS)
    index: int | None = None
    if isinstance(mine_raw, (list, tuple)) and mine_raw:
        index = _to_index(mine_raw[0])
    elif mine_raw is not None:
        index = _to_index(mine_raw)          # a bare value, not a list
        if index is not None:
            notes.append(f"{mine_key} was a scalar, not a list")

    if index is None:
        # The pin is the answer here. This runtime was deployed as one position
        # and its prompt is written for that position; playing someone else on
        # a malformed payload is worse than playing ourselves.
        notes.append(
            f"no usable player index (key={mine_key!r} value={mine_raw!r}"
            f"{f' session={session_id!r}' if session_id else ''}); "
            f"using pinned {default_index}"
        )
        index = default_index
    elif mine_key != "myPlayers":
        notes.append(f"player index found under {mine_key!r}")

    return {"gameState": state, "teamId": team, "myPlayers": [index]}, notes
