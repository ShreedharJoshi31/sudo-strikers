"""Inbound payload normalisation: whatever the platform sends -> the shape wire.py reads.

Why this module exists
----------------------
`wire.to_observation` needs exactly `{"gameState": {...}, "teamId": N,
"myPlayers": [i]}`. The competition platform sends something else, and when it
does, `brain.handle` falls through BOTH of its rescue paths and returns the
99-byte last-ditch constant - INTERCEPT as playerId 0 - in 0.000s, for every
player. That reads as a healthy 200 to the platform and as a failed fitness
check to us, with no error anywhere to find.

So this module refuses to care what the keys are called. It hunts the payload
for the two things that actually matter - a squad list and a ball - and rebuilds
the canonical shape around them. Unknown wrappers are unwrapped, unknown key
spellings are tried in turn, and a plain-text scout briefing (the format in
AGENT_PROTOCOL.md section 3) is parsed as a last resort.

Everything it had to guess is reported in `notes`, which the entrypoint logs.
Guessing silently is how we got here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

MAX_DEPTH = 6

#: Key spellings seen or plausible for each thing we need.
STATE_KEYS = ("gameState", "game_state", "gamestate", "state", "observation",
              "snapshot", "world", "game")
TEAM_KEYS = ("teamId", "team_id", "teamID", "team", "myTeam", "my_team",
             "teamIndex", "team_index", "side")
INDEX_KEYS = ("myPlayers", "my_players", "playerId", "player_id", "playerIndex",
              "player_index", "agentId", "agent_id", "position", "pos",
              "playerNumber", "player_number", "me", "controlledPlayers")
PLAYER_LIST_KEYS = ("players", "allPlayers", "all_players", "agents", "squad",
                    "playerStates", "player_states", "entities")
BALL_KEYS = ("ball", "ballState", "ball_state")
TEXT_KEYS = ("prompt", "input", "text", "message", "briefing", "inputText",
             "body", "query", "question")


# ----------------------------------------------------------------- helpers

def _maybe_json(value):
    """A JSON document arriving as a string is still a JSON document."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s[:1] in ("{", "[") and s[-1:] in ("}", "]"):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _get(obj, keys):
    if not isinstance(obj, Mapping):
        return None
    for k in keys:
        if k in obj and obj[k] is not None:
            return _maybe_json(obj[k])
    return None


def _xy(value):
    """Pull an (x, y) out of {'x':..,'y':..}, [x, y] or 'x,y'. None if absent."""
    value = _maybe_json(value)
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
        if x is None:
            x, y = value.get("X"), value.get("Y")
        if x is None:
            return None
        try:
            return float(x), float(y or 0.0)
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
    return None


def _find_position(record):
    """The position of a player record, whatever the field is called."""
    if not isinstance(record, Mapping):
        return None
    for k in ("position", "pos", "location", "loc", "coordinates", "coords", "xy"):
        got = _xy(record.get(k))
        if got is not None:
            return got
    return _xy(record)          # the record may itself be {"x":.., "y":..}


def _looks_like_players(value):
    """A squad is a list of >=4 records that each carry a position."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) < 4:
        return False
    positioned = sum(1 for r in value if _find_position(r) is not None)
    return positioned >= max(4, len(value) - 2)


def _walk(node, depth=0):
    """Every Mapping in the payload, outermost first."""
    if depth > MAX_DEPTH:
        return
    node = _maybe_json(node)
    if isinstance(node, Mapping):
        yield node
        for v in node.values():
            yield from _walk(v, depth + 1)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for v in node:
            yield from _walk(v, depth + 1)


def _find_players(payload):
    """The squad list, by key name first, then by shape."""
    for node in _walk(payload):
        for k in PLAYER_LIST_KEYS:
            cand = _maybe_json(node.get(k)) if isinstance(node, Mapping) else None
            if _looks_like_players(cand):
                return cand, f"players from key {k!r}"
    for node in _walk(payload):
        if not isinstance(node, Mapping):
            continue
        for k, v in node.items():
            if _looks_like_players(_maybe_json(v)):
                return _maybe_json(v), f"players inferred from shape at key {k!r}"
    return None, "no player list found"


def _find_ball(payload):
    for node in _walk(payload):
        cand = _get(node, BALL_KEYS)
        if isinstance(cand, Mapping) and _find_position(cand) is not None:
            return cand, "ball from key"
    return None, "no ball found"


def _team_token(record):
    """Whatever this record says about which side it is on."""
    if not isinstance(record, Mapping):
        return None
    for k in ("teamCode", "team_code", "teamId", "team_id", "team", "side",
              "teamName", "team_name", "color", "colour"):
        if k in record and record[k] is not None:
            return str(record[k]).strip().lower()
    for k in ("isHome", "is_home", "home"):
        if k in record and record[k] is not None:
            return "home" if bool(record[k]) else "away"
    return None


def _home_away(token, home_token):
    """Map an arbitrary side token onto 'home'/'away'."""
    if token is None:
        return None
    if token in ("home", "0", "h", "true"):
        return "home"
    if token in ("away", "1", "a", "false"):
        return "away"
    return "home" if token == home_token else "away"


def _index_of(record, fallback):
    """The shirt number of a player record."""
    if isinstance(record, Mapping):
        for k in ("agentId", "agent_id", "playerId", "player_id", "id",
                  "number", "index", "shirt"):
            if k in record and record[k] is not None:
                m = re.search(r"(\d+)", str(record[k]))
                if m:
                    return int(m.group(1))
    return fallback


def _stamina(record):
    if isinstance(record, Mapping):
        for k in ("stamina", "energy", "fitness", "stam"):
            if k in record and record[k] is not None:
                try:
                    return float(record[k])
                except (TypeError, ValueError):
                    pass
    return 1.0


def index_from_session(session_id):
    """The platform names sessions '<something>-pos2'. That is the slot we are in."""
    if not session_id:
        return None
    m = re.search(r"pos(?:ition)?[-_]?(\d+)\s*$", str(session_id).strip(), re.I)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------- briefing

BRIEF_TEAM = re.compile(r"Team:\s*(\d)\s*\((HOME|AWAY)\)", re.I)
BRIEF_BALL = re.compile(
    r"Ball:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*(?:held by\s*(.+))?", re.I)
BRIEF_ME = re.compile(
    r">>>\s*YOUR PLAYER\s*\(\s*\w+\s*,\s*id\s*=\s*(\d+)\s*\)\s*:\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", re.I)
BRIEF_MATE = re.compile(
    r"^\s*(?:GK|P(\d+))\s*\(\s*id\s*=\s*(\d+)\s*\)\s*:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", re.M)
BRIEF_OPP = re.compile(
    r"^\s*P(\d+)\s*:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", re.M)


def parse_briefing(text):
    """The plain-text scout briefing (AGENT_PROTOCOL.md section 3) -> canonical payload.

    Velocity is absent from the briefing by design, so every player is written
    with zero velocity. That is a real loss of information, not a formatting
    detail: `passing.py` leads its passes off velocity. If this path ever fires
    in a real match, the raw snapshot is what we should be asking for instead.
    """
    if not isinstance(text, str) or ">>>" not in text:
        return None, "not a briefing"

    me = BRIEF_ME.search(text)
    if not me:
        return None, "briefing has no YOUR PLAYER line"
    my_index = int(me.group(1))

    team = BRIEF_TEAM.search(text)
    team_id = int(team.group(1)) if team else 0
    my_code = "home" if team_id == 0 else "away"
    opp_code = "away" if team_id == 0 else "home"

    mates, opps = {}, {}
    mate_block = text.split("Teammates:", 1)
    opp_split = re.split(r"Opponents[^\n:]*:", text, maxsplit=1)

    if len(mate_block) > 1:
        segment = re.split(r"Opponents[^\n:]*:", mate_block[1], maxsplit=1)[0]
        for m in BRIEF_MATE.finditer(segment):
            mates[int(m.group(2))] = (float(m.group(3)), float(m.group(4)))
    if len(opp_split) > 1:
        for m in BRIEF_OPP.finditer(opp_split[1]):
            opps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

    mates[my_index] = (float(me.group(2)), float(me.group(3)))

    players = []
    for idx, (x, y) in sorted(mates.items()):
        players.append({"agentId": f"agentId_{idx}", "teamCode": my_code,
                        "position": {"x": x, "y": y},
                        "velocity": {"x": 0.0, "y": 0.0},
                        "stamina": 1.0, "isSprinting": False})
    for idx, (x, y) in sorted(opps.items()):
        players.append({"agentId": f"agentId_{idx}", "teamCode": opp_code,
                        "position": {"x": x, "y": y},
                        "velocity": {"x": 0.0, "y": 0.0},
                        "stamina": 1.0, "isSprinting": False})

    ball_xy, holder = (0.0, 0.0), None
    b = BRIEF_BALL.search(text)
    if b:
        ball_xy = (float(b.group(1)), float(b.group(2)))
        who = (b.group(3) or "").strip()
        m = re.search(r"(MY|OPP)\s+player\s+(\d+)", who, re.I)
        if m:
            holder = f"agentId_{m.group(2)}"

    ball = {"position": {"x": ball_xy[0], "y": ball_xy[1]},
            "velocity": {"x": 0.0, "y": 0.0}}
    if holder:
        ball["possessionAgentId"] = holder
    else:
        ball["isFree"] = True

    t = re.search(r"Time:\s*([\d.]+)", text)
    s = re.search(r"Score:\s*(\d+)\s*-\s*(\d+)", text)
    state = {"gameTime": float(t.group(1)) if t else 0.0,
             "score": {"home": int(s.group(1)) if s else 0,
                       "away": int(s.group(2)) if s else 0},
             "ball": ball, "players": players}
    return ({"gameState": state, "teamId": team_id, "myPlayers": [my_index]},
            f"parsed text briefing ({len(players)} players)")


# ------------------------------------------------------------- normalise

def normalise(payload, default_index=None, session_id=""):
    """Return (canonical_payload, notes). Raises ValueError if nothing is usable.

    `default_index` is the runtime's pinned player. It is the LAST resort: the
    session slot and the payload both win over it, because one runtime may be
    registered in two positions and the pin cannot tell those apart.
    """
    notes = []
    payload = _maybe_json(payload)

    # A text briefing may arrive bare or under any of the usual text keys.
    if isinstance(payload, str):
        parsed, why = parse_briefing(payload)
        if parsed:
            return parsed, [why]
        raise ValueError("payload is a string that is not a briefing")

    if not isinstance(payload, Mapping):
        raise ValueError(f"payload is {type(payload).__name__}, not an object")

    for key in TEXT_KEYS:
        val = _maybe_json(payload.get(key))
        if isinstance(val, str):
            parsed, why = parse_briefing(val)
            if parsed:
                return parsed, [f"{why} from key {key!r}"]

    # Already canonical?
    state = _get(payload, STATE_KEYS)
    if isinstance(state, Mapping) and _looks_like_players(_maybe_json(state.get("players"))) \
            and _get(payload, TEAM_KEYS) is not None:
        idx = _resolve_index(payload, state, default_index, session_id, notes)
        return ({"gameState": state,
                 "teamId": _resolve_team(payload, state, notes),
                 "myPlayers": [idx]}, notes or ["canonical payload"])

    players, why = _find_players(payload)
    notes.append(why)
    if not players:
        raise ValueError("no player list anywhere in payload")
    ball, why_ball = _find_ball(payload)
    notes.append(why_ball)

    team_id = _resolve_team(payload, state if isinstance(state, Mapping) else payload, notes)
    my_code = "home" if team_id == 0 else "away"

    # Split the squad in two. A side token on the records is authoritative;
    # without one, fall back to first-half/second-half and say so loudly.
    tokens = [_team_token(r) for r in players]
    if any(t is not None for t in tokens):
        home_token = next((t for t in tokens if t is not None), "home")
        codes = [_home_away(t, home_token) or "home" for t in tokens]
    else:
        half = len(players) // 2
        codes = ["home"] * half + ["away"] * (len(players) - half)
        notes.append("WARNING: no team field on players; split by list order")

    normalised_players = []
    for i, (record, code) in enumerate(zip(players, codes)):
        pos = _find_position(record) or (0.0, 0.0)
        vel = None
        for k in ("velocity", "vel", "speedVector", "movement"):
            vel = _xy(record.get(k)) if isinstance(record, Mapping) else None
            if vel is not None:
                break
        idx = _index_of(record, i % 5)
        entry = {"agentId": f"agentId_{idx}", "teamCode": code,
                 "position": {"x": pos[0], "y": pos[1]},
                 "velocity": {"x": (vel or (0.0, 0.0))[0], "y": (vel or (0.0, 0.0))[1]},
                 "stamina": _stamina(record)}
        if isinstance(record, Mapping):
            for src, dst in (("isSprinting", "isSprinting"), ("is_sprinting", "isSprinting"),
                             ("sprinting", "isSprinting"), ("speed", "speed")):
                if src in record and record[src] is not None:
                    entry[dst] = record[src]
        normalised_players.append(entry)
    if vel is None:
        notes.append("no velocity on players (passing model will run blind)")

    ball_out = {"position": {"x": 0.0, "y": 0.0}, "velocity": {"x": 0.0, "y": 0.0},
                "isFree": True}
    if isinstance(ball, Mapping):
        bpos = _find_position(ball) or (0.0, 0.0)
        bvel = None
        for k in ("velocity", "vel", "direction"):
            bvel = _xy(ball.get(k))
            if bvel is not None:
                break
        ball_out = {"position": {"x": bpos[0], "y": bpos[1]},
                    "velocity": {"x": (bvel or (0.0, 0.0))[0],
                                 "y": (bvel or (0.0, 0.0))[1]}}
        holder = None
        for k in ("possessionAgentId", "possession_agent_id", "possession",
                  "owner", "heldBy", "held_by", "carrier", "ownerId"):
            if isinstance(ball, Mapping) and ball.get(k) is not None:
                holder = ball[k]
                break
        if holder is not None and str(holder).lower() not in ("none", "null", "-1", ""):
            m = re.search(r"(\d+)", str(holder))
            ball_out["possessionAgentId"] = f"agentId_{m.group(1)}" if m else str(holder)
        else:
            ball_out["isFree"] = True

    src = state if isinstance(state, Mapping) else payload
    score = _get(src, ("score", "scores", "goals")) or {}
    if not isinstance(score, Mapping):
        score = {}
    game_time = _get(src, ("gameTime", "game_time", "time", "matchTime", "tick")) or 0.0
    try:
        game_time = float(game_time)
    except (TypeError, ValueError):
        game_time = 0.0

    state_out = {"gameTime": game_time,
                 "score": {"home": int(score.get("home", 0) or 0),
                           "away": int(score.get("away", 0) or 0)},
                 "ball": ball_out, "players": normalised_players}

    idx = _resolve_index(payload, src, default_index, session_id, notes)
    mine = [p for p in normalised_players if p["teamCode"] == my_code]
    if not any(_index_of(p, -1) == idx for p in mine) and mine:
        fallback = _index_of(mine[0], 0)
        notes.append(f"WARNING: player {idx} not in {my_code} squad; using {fallback}")
        idx = fallback

    return {"gameState": state_out, "teamId": team_id, "myPlayers": [idx]}, notes


def _resolve_team(payload, state, notes):
    """Which side we are on. Searched at EVERY depth, not just the top two.

    The platform nests the whole state as a JSON string under `prompt`, so a
    top-level lookup finds nothing and silently assumes HOME - which points
    every coordinate at our own goal when we are drawn AWAY. That is the most
    expensive mistake in the competition (AGENT_PROTOCOL.md section 2), so it
    must never be reached by a shallow lookup missing a nested key.
    """
    raw = _get(payload, TEAM_KEYS)
    if raw is None:
        raw = _get(state, TEAM_KEYS) if isinstance(state, Mapping) else None
    if raw is None:
        for node in _walk(payload):
            raw = _get(node, TEAM_KEYS)
            if raw is not None:
                break
    if raw is None:
        notes.append("WARNING: no teamId in payload; assuming HOME")
        return 0
    token = str(raw).strip().lower()
    if token in ("1", "away", "a"):
        return 1
    if token in ("0", "home", "h"):
        return 0
    notes.append(f"WARNING: unrecognised team {raw!r}; assuming HOME")
    return 0


def _resolve_index(payload, state, default_index, session_id, notes):
    """Which player this call is about. Session slot > pin > payload.

    The session slot is the platform's own routing and beats everything. After
    that the PIN wins, not the payload: `runtime_app` deploys one runtime per
    position, and the whole point of pinning is that a runtime deployed as the
    keeper keeps playing the keeper even if routing sends it someone else. The
    payload is consulted only when neither is available.
    """
    slot = index_from_session(session_id)
    if slot is not None:
        return slot

    if default_index is not None:
        return default_index

    for key in INDEX_KEYS:
        for src in list(_walk(payload)) + ([state] if isinstance(state, Mapping) else []):
            if not isinstance(src, Mapping) or key not in src or src[key] is None:
                continue
            val = _maybe_json(src[key])
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                if not val:
                    continue
                val = val[0]
            m = re.search(r"(\d+)", str(val))
            if m:
                return int(m.group(1))

    notes.append("WARNING: no index in session, pin or payload; defaulting to 0")
    return 0
