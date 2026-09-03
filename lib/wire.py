"""Boundary layer: the real platform payload in, the real command array out.

Why this module exists
----------------------
Everything else in `lib/` is written against one convenient fiction: you are
always at x=0 attacking toward +x, players have roles and home positions, and
possession is a fact rather than a guess. The actual AWS Agentic Football Cup
payload provides none of that. It gives absolute coordinates that mean opposite
things to the two teams, no roles, no formation, no pressure, and a possession
field that is genuinely ambiguous.

Keeping that translation in one file is what lets `policy.py` stay a football
brain instead of half a football brain and half a parser. It is also the file
where the workshop spec is vague, so the vagueness is handled here — visibly,
in both directions — rather than being guessed at five call sites.

Three known traps in the payload, all handled below:

  - `agentId` is NOT unique across teams. Both sides number their players
    `agentId_0`..`agentId_4`, and `ball.possessionAgentId` is a bare
    `"agentId_N"` with no team on it. Scanning a home-first player list for the
    first matching index therefore misattributes EVERY away possession, which
    is the kind of bug that never raises and just quietly loses matches.
    `_resolve_possession` disambiguates by distance to the ball and reports
    whether it had to infer.
  - The stamina scale is contradictory: the workshop docs say 0-100, the AWS
    sample fixture ships 0-1. `normalise_stamina` handles both and says which
    branch it took, so a caller can log it once rather than discovering it from
    a team that never gets tired.
  - `speed` is not the magnitude of `velocity` in the sample data: velocity
    {2,0} is paired with speed 1.5, velocity {1,0} with speed 1.0, and 1.5 is
    also what `isSprinting` implies. `velocity` is treated as truth and `speed`
    is carried through as `speed_hint` so that nothing downstream can mistake
    it for physics.

Pure functions only. No I/O, no network, stdlib and pydantic.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from command import (
    COMMAND_TYPES,
    GK_ONLY,
    AgentCommand,
    AimLocation,
    DistributionMethod,
    PassType,
    TeamSide,
    Tightness,
    player_index as parse_index,
)

# --------------------------------------------------------------- geometry

# THE PLATFORM DOES NOT USE METRES, AND ITS BALL DOES NOT USE THE PLAYERS' AXES.
#
# The workshop docs describe a 110 x 70 m field with x in [-55, +55]. The LIVE
# match payload does not look like that at all. Measured over 386 real match
# ticks (3,860 player samples) on 2026-09-03:
#
#     player.x   min  -6.470   median  -1.414   max   6.400
#     player.y   min  -3.500   median   0.000   max   3.559
#     ball.x     min  -6.978   median  -1.200   max   6.816
#     ball.y     min   0.139   median   0.143   max   0.986   <- HEIGHT
#     ball.z     min  -1.618   median   0.000   max   3.579   <- LATERAL
#
# Two separate bugs fell out of taking the docs at face value:
#
#  1. SCALE. The pitch is about +/-6.9 by +/-3.6 platform units, not +/-55 by
#     +/-35. Treating units as metres put the opponent goal roughly 48 "metres"
#     beyond where it really is, so every distance gate in policy.py was ~8x out.
#     Consequence, measured: SHOOT was chosen 0 times in 373 on-ball ticks, the
#     forward was ordered to x=23 (off the pitch) and pinned on the goal line by
#     the platform's clamp, and PRESSURE_RADIUS=8 covered the entire field so
#     every player read as permanently under maximum pressure.
#
#  2. BALL AXES. For PLAYERS, y is lateral. For the BALL, y is HEIGHT and z is
#     lateral - note ball.y sits at 0.143 (the resting radius) in most ticks
#     while ball.z spans the same range as player.y. Reading ball.y as lateral
#     meant the ball's across-pitch position was invisible: the keeper computed
#     the identical hold-the-angle target to 14 decimal places on 75 ticks.
#
# The fix keeps policy.py and every Params threshold in metres by scaling at
# this boundary, which is the only place that knows about platform units.
# Override from the environment if the organisers change the field.

#: Half-extent of the real pitch, in PLATFORM units.
PLATFORM_X_LIMIT = float(os.environ.get("AFC_PLATFORM_X_LIMIT", "6.9"))
PLATFORM_Y_LIMIT = float(os.environ.get("AFC_PLATFORM_Y_LIMIT", "3.6"))

#: Metres per platform unit. Uniform on both axes, so distances and angles stay
#: true; it only sets what "one metre" means to the thresholds. 8.0 keeps the
#: pitch about 110 m long, which is the scale Params was written against.
PLATFORM_SCALE = float(os.environ.get("AFC_PLATFORM_SCALE", "8.0"))

GOAL_HALF_WIDTH = 5.0          # the goal mouth, in metres, in the local frame

# The normalised frame handed to policy.py: own goal at x=0, attacking toward
# +x, y increasing across the pitch. Same pitch, shifted so both teams can run
# identical code. policy.py's `_cx`/`_cy` clamp to [0, length] / [0, width],
# which is why this frame starts at zero rather than staying centred.
PITCH_LENGTH = PLATFORM_X_LIMIT * 2.0 * PLATFORM_SCALE   # ~110.4 m
PITCH_WIDTH = PLATFORM_Y_LIMIT * 2.0 * PLATFORM_SCALE    # ~57.6 m
GOAL_WIDTH = GOAL_HALF_WIDTH * 2.0        # 10.0
HALF_LENGTH = PITCH_LENGTH / 2.0          # 55.0, the x offset between frames
HALF_WIDTH = PITCH_WIDTH / 2.0            # 35.0, the y offset, and the centre line

OWN_GOAL_CENTER = (0.0, HALF_WIDTH)
OPPONENT_GOAL_CENTER = (PITCH_LENGTH, HALF_WIDTH)

#: Whether to put `playerId` on each emitted command.
#:
#: AGENT_PROTOCOL.md section 4 ("How to say it") shows the normative form
#: WITHOUT it - `[{"commandType":"PASS","parameters":{...},"duration":0}]` - and
#: states plainly: "You never need to name your own player or your own team.
#: Both are stamped on for you." The platform's own NO_PARSE error text shows
#: the same shape. Only the worked examples in section 6 include `playerId`,
#: and those are illustrative rather than normative.
#:
#: So the default is to omit it. Set AFC_EMIT_PLAYER_ID=1 to put it back
#: without a redeploy of this logic if the platform turns out to want it.
EMIT_PLAYER_ID = os.environ.get("AFC_EMIT_PLAYER_ID", "0").lower() in ("1", "true", "yes")

#: Duration stamped on commands that do not set their own. See
#: policy.Params.command_duration - this is the env override for a live test.
DURATION_DEFAULT = int(os.environ.get("AFC_COMMAND_DURATION", "0"))

TEAM_HOME = 0
TEAM_AWAY = 1
TEAM_CODE: dict[int, str] = {TEAM_HOME: "home", TEAM_AWAY: "away"}
#: teamCode -> the spelling FOLLOW_PLAYER's `target_team` parameter wants.
TEAM_SIDE: dict[str, str] = {"home": "HOME", "away": "AWAY"}

#: Player index 0 is always the goalkeeper. The platform sends no role field,
#: so this is the only thing about the roster that is guaranteed.
GK_INDEX = 0

#: Roles for the other four. The platform does not send these, so they are a
#: convention this team picks and both `policy.py` and `prompts.py` read. Maps
#: the 2-1-1 in team.yaml: two at the back, one linking, one ahead.
DEFAULT_ROLES: dict[int, str] = {
    0: "GK",
    1: "DEFENDER",
    2: "DEFENDER",
    3: "MIDFIELDER",
    4: "FORWARD",
}

#: Formation base positions, in the NORMALISED frame, so one table serves both
#: teams. Spread across a 110 x 70 pitch: keeper just off his line, defenders
#: split either side of centre, midfielder on the halfway line, forward high.
DEFAULT_FORMATION: dict[int, tuple[float, float]] = {
    0: (5.0, HALF_WIDTH),
    1: (25.0, 21.0),
    2: (25.0, 49.0),
    3: (52.0, HALF_WIDTH),
    4: (78.0, HALF_WIDTH),
}

#: Radius over which an opponent counts as applying pressure, and the distance
#: at which their contribution falls to zero. Chosen so the thresholds already
#: tuned into policy.Params keep their meaning: with linear falloff, one
#: opponent at 6 m gives 0.25 (policy.pass_when_pressed) and two inside about
#: 3.5 m give 1.1 (policy.clear_pressure).
PRESSURE_RADIUS = 8.0

#: How close a player must be to the ball to be credited with it outright. A
#: controlled ball sits within about a stride; 2 m is loose enough to survive a
#: 2-second-old snapshot but tight enough that normally only one player of a
#: given index qualifies.
POSSESSION_RADIUS = 2.0

#: Stamina at or below this is read as a 0-1 fraction and scaled to 0-100.
#: Exactly 1.0 is genuinely ambiguous (1% or 100%?) and is read as full, which
#: matches the AWS sample fixture where every player starts at 1.0.
STAMINA_FRACTION_MAX = 1.0

AIM_LOCATIONS: tuple[str, ...] = get_args(AimLocation)
PASS_TYPES: tuple[str, ...] = get_args(PassType)
DISTRIBUTION_METHODS: tuple[str, ...] = get_args(DistributionMethod)
TIGHTNESS_VALUES: tuple[str, ...] = get_args(Tightness)
TEAM_SIDES: tuple[str, ...] = get_args(TeamSide)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finite(*values: object) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _other(team_code: str) -> str:
    """The opposing teamCode."""
    return "away" if team_code == "home" else "home"


# ------------------------------------------------------------------- frame

@dataclass(frozen=True)
class Frame:
    """Invertible map between platform coordinates and the attack-relative frame.

    Platform coordinates are centred and fixed for both teams: x in [-55, +55]
    with HOME attacking +x and AWAY attacking -x. Downstream code instead wants
    a frame where its own goal is at (0, 35) and it always attacks toward +x,
    so that one policy plays both sides with no `if team == away` anywhere.

    HOME needs only a translation. AWAY needs a 180-degree rotation about the
    centre spot first, which is the whole content of `sign = -1`.

    Because `sign` is exactly +/-1 and the offsets are exact binary fractions,
    `to_platform` is a true inverse of `to_local` up to float representation
    error. That property is the one most likely to be silently wrong, so it is
    isolated here where a two-line test can pin it down.
    """

    team_id: int
    sign: float

    @classmethod
    def for_team(cls, team_id: int) -> Frame:
        if team_id not in (TEAM_HOME, TEAM_AWAY):
            raise ValueError(f"teamId must be 0 (home) or 1 (away), got {team_id!r}")
        return cls(team_id=team_id, sign=1.0 if team_id == TEAM_HOME else -1.0)

    @property
    def team_code(self) -> str:
        return TEAM_CODE[self.team_id]

    # Points carry the translation; vectors do not.
    def to_local(self, x: float, y: float) -> tuple[float, float]:
        return (self.sign * x * PLATFORM_SCALE + HALF_LENGTH,
                self.sign * y * PLATFORM_SCALE + HALF_WIDTH)

    def to_platform(self, x: float, y: float) -> tuple[float, float]:
        return (self.sign * (x - HALF_LENGTH) / PLATFORM_SCALE,
                self.sign * (y - HALF_WIDTH) / PLATFORM_SCALE)

    def vector_to_local(self, vx: float, vy: float) -> tuple[float, float]:
        return (self.sign * vx * PLATFORM_SCALE, self.sign * vy * PLATFORM_SCALE)

    def vector_to_platform(self, vx: float, vy: float) -> tuple[float, float]:
        return (self.sign * vx / PLATFORM_SCALE, self.sign * vy / PLATFORM_SCALE)

    def heading_to_local(self, degrees: float) -> float:
        """Rotate an orientation into the local frame. AWAY is flipped 180."""
        return (degrees + (0.0 if self.sign > 0 else 180.0)) % 360.0


def frame_for(payload: Mapping) -> Frame:
    """The transform for this payload, so `to_wire` can undo `to_observation`."""
    return Frame.for_team(int(payload["teamId"]))


# ------------------------------------------------------------ ambiguity (a)

def normalise_stamina(raw: object) -> tuple[float, str]:
    """Return (stamina on a 0-100 scale, which scale the input was on).

    The workshop docs say 0-100; the AWS sample fixture ships 0-1. Rather than
    pick one and be wrong half the time, read the value: anything at or below
    1.0 is treated as a fraction. The returned tag exists so a caller can log
    the branch once instead of guessing from behaviour.
    """
    if not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return 100.0, "missing"
    if raw <= STAMINA_FRACTION_MAX:
        return float(raw) * 100.0, "fraction"
    return float(raw), "percent"


@dataclass(frozen=True)
class Possession:
    """Who has the ball, and whether we are sure."""

    owner_id: str | None       # normalised id, e.g. "away_2"
    relation: str | None       # "you" | "teammate" | "opponent" | None
    inferred: bool             # True when the index alone did not settle it


def _resolve_possession(
    everyone: Sequence[dict],
    ball_xy: tuple[float, float],
    agent_id: object,
    is_free: bool,
    my_id: str,
    my_code: str,
    radius: float,
) -> Possession:
    """Attribute `ball.possessionAgentId` to an actual player.

    The payload's id is a bare "agentId_N" and BOTH teams have an N. So the
    index narrows it to two candidates and geometry picks between them:

      1. If exactly one of the two is within `radius` of the ball, it is theirs
         and we are certain.
      2. Otherwise take whichever is nearest the ball and flag it as inferred,
         so a caller can log it and a policy can discount it.

    Ties break on id purely for determinism: five runtimes decide independently
    and must reach the same answer from the same payload.
    """
    if is_free or agent_id in (None, ""):
        return Possession(None, None, False)

    try:
        index = parse_index(agent_id)
    except ValueError:
        return Possession(None, None, True)

    candidates = [p for p in everyone if p["number"] == index]
    if not candidates:
        return Possession(None, None, True)

    close = [p for p in candidates if _distance(p["position"], ball_xy) <= radius]
    if len(close) == 1:
        owner, inferred = close[0], False
    else:
        owner = min(candidates, key=lambda p: (_distance(p["position"], ball_xy), p["id"]))
        inferred = True

    if owner["id"] == my_id:
        relation = "you"
    elif owner["team_code"] == my_code:
        relation = "teammate"
    else:
        relation = "opponent"
    return Possession(owner["id"], relation, inferred)


# ---------------------------------------------------------------- inbound

def normalised_id(team_code: str, index: int) -> str:
    """A stable id that carries the team, e.g. "home_3".

    The platform's own ids collide across teams. Everything downstream keys on
    this instead, so no amount of index arithmetic can confuse the two sides.
    """
    return f"{team_code}_{index}"


def _pressure(position: tuple[float, float], opponents: Sequence[dict], radius: float) -> float:
    """Weighted count of opponents crowding a point.

    Linear falloff to zero at `radius`, summed. Weighted rather than a plain
    count because "one man at 7 m" and "two men at 1 m" are not the same
    situation, and the policy's thresholds only mean something on a scale that
    distinguishes them.
    """
    total = 0.0
    for o in opponents:
        d = _distance(position, o["position"])
        if d < radius:
            total += 1.0 - d / radius
    return total


def _player_record(
    raw: Mapping,
    frame: Frame,
    team_code: str,
    roles: Mapping[int, str],
    formation: Mapping[int, tuple[float, float]],
) -> tuple[dict, str]:
    """One platform player entry, in the normalised frame.

    Returns the record and which stamina scale the payload turned out to use,
    so `to_observation` can surface the ambiguity without re-reading the value.
    Pressure is filled in afterwards, once both squads are known.
    """
    index = parse_index(raw["agentId"])
    pos = raw.get("position") or {}
    vel = raw.get("velocity") or {}
    stamina, stamina_scale = normalise_stamina(raw.get("stamina"))

    role = "GK" if index == GK_INDEX else roles.get(index, "MIDFIELDER")
    return {
        "id": normalised_id(team_code, index),
        "number": index,               # what goes on the wire as playerId
        "team_code": team_code,
        "role": role,
        # z is dropped: the policy is two-dimensional, and the one place height
        # would matter (SHOOT's TL/TR/BL/BR) takes an enum, not a coordinate.
        "position": frame.to_local(float(pos.get("x", 0.0)), float(pos.get("y", 0.0))),
        "velocity": frame.vector_to_local(float(vel.get("x", 0.0)), float(vel.get("y", 0.0))),
        "stamina": stamina,
        "home_position": tuple(formation.get(index, (HALF_LENGTH, HALF_WIDTH))),
        "orientation": frame.heading_to_local(float(raw.get("orientation", 0.0) or 0.0)),
        "is_sprinting": bool(raw.get("isSprinting", False)),
        "last_action": raw.get("lastAction"),
        # Named `speed_hint`, never `speed`: it does not agree with |velocity|
        # in the sample data, so it is a gait label, not a measurement.
        "speed_hint": raw.get("speed"),
    }, stamina_scale


def _previous_command(payload: Mapping, frame: "Frame") -> dict | None:
    """`previousCommand` from the platform, normalised into the local frame.

    Returned as {"type": str, "target": (x, y) | None}. The target is converted
    with the same transform as everything else, so a caller can compare it to a
    position it computed itself without knowing which way the team is kicking.
    """
    raw = payload.get("previousCommand") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("commandType")
    if not kind:
        return None
    params = raw.get("parameters") or {}
    target = None
    if isinstance(params, Mapping) and params.get("target_x") is not None:
        try:
            target = frame.to_local(float(params["target_x"]), float(params["target_y"]))
        except (TypeError, ValueError, KeyError):
            target = None
    return {"type": str(kind), "target": target}


def to_observation(
    payload: Mapping,
    *,
    my_index: int | None = None,
    roles: Mapping[int, str] = DEFAULT_ROLES,
    formation: Mapping[int, tuple[float, float]] = DEFAULT_FORMATION,
    pressure_radius: float = PRESSURE_RADIUS,
    possession_radius: float = POSSESSION_RADIUS,
) -> dict:
    """Real platform payload -> the normalised observation `policy.py` consumes.

    `my_index` overrides `payload["myPlayers"][0]`; the platform sends exactly
    one controlled player per call, so the override is only for tests and for a
    server that routes by its own player id.

    Raises ValueError on a payload that cannot be read at all — callers should
    treat that like a late decision and fall back to `command.SAFE_DEFAULT`.
    """
    state = payload.get("gameState")
    if not isinstance(state, Mapping):
        raise ValueError("payload has no gameState object")

    frame = frame_for(payload)
    my_code = frame.team_code

    if my_index is None:
        controlled = payload.get("myPlayers") or []
        if not controlled:
            raise ValueError("payload has no myPlayers entry")
        my_index = parse_index(controlled[0])
    my_id = normalised_id(my_code, my_index)

    # Split on teamCode, never on agentId: the ids collide across teams.
    mine: list[dict] = []
    theirs: list[dict] = []
    stamina_scale = "missing"
    for raw in state.get("players") or []:
        code = str(raw.get("teamCode", "")).lower()
        is_mine = code == my_code
        record, stamina_scale = _player_record(
            raw, frame, my_code if is_mine else _other(my_code), roles, formation
        )
        (mine if is_mine else theirs).append(record)

    for player in mine:
        player["pressure"] = _pressure(player["position"], theirs, pressure_radius)
    for player in theirs:
        player["pressure"] = _pressure(player["position"], mine, pressure_radius)

    me = next((p for p in mine if p["number"] == my_index), None)
    if me is None:
        raise ValueError(f"controlled player {my_index} is not in the {my_code} squad")

    ball = state.get("ball") or {}
    ball_pos = ball.get("position") or {}
    ball_vel = ball.get("velocity") or {}
    # The ball is 3D and does NOT share the players' axis order: when a `z` is
    # present, `y` is HEIGHT and `z` is the across-pitch axis that `y` means for
    # every player. Detect rather than assume, because the scrimmage/fitness
    # fixtures send a flat 2D ball where `y` really is lateral.
    ball_is_3d = ball_pos.get("z") is not None
    ball_lat = float(ball_pos.get("z" if ball_is_3d else "y", 0.0) or 0.0)
    ball_height = float(ball_pos.get("y", 0.0) or 0.0) if ball_is_3d else 0.0
    ball_xy = frame.to_local(float(ball_pos.get("x", 0.0)), ball_lat)
    ball_v = frame.vector_to_local(
        float(ball_vel.get("x", 0.0) or 0.0),
        float(ball_vel.get("z" if ball_is_3d else "y", 0.0) or 0.0))

    possession = _resolve_possession(
        mine + theirs,
        ball_xy,
        ball.get("possessionAgentId"),
        bool(ball.get("isFree", False)),
        my_id,
        my_code,
        possession_radius,
    )

    score = state.get("score") or {}

    return {
        "tick": int(state.get("tick", 0)),
        "game_time": float(state.get("gameTime", 0.0)),
        "play_mode": state.get("playMode", "OPEN_PLAY"),
        "score": {
            "you": int(score.get(my_code, 0)),
            "opponent": int(score.get(_other(my_code), 0)),
        },
        "possession": possession.relation,
        "you": me,
        "teammates": [p for p in mine if p["id"] != my_id],
        "opponents": theirs,
        "ball": {
            "position": ball_xy,
            "velocity": ball_v,
            # Height above the turf, in local-frame metres. Every distance
            # check in policy.py is 2D, so without this a ball sailing
            # overhead reads as "at my feet". A 2D ball reports 0.0, which is
            # exactly the pre-3D behaviour.
            "height": ball_height * PLATFORM_SCALE,
            "owner_id": possession.owner_id,
            "owner_inferred": possession.inferred,
            "is_free": bool(ball.get("isFree", False)),
        },
        # What we played last tick, with any target mapped into THIS team's
        # frame so policy.py can compare it against its own coordinates.
        "previous_command": _previous_command(payload, frame),
        "pitch": {
            "length": PITCH_LENGTH,
            "width": PITCH_WIDTH,
            "you_attack_toward": "+x",
            "your_goal": {"center": OWN_GOAL_CENTER, "width": GOAL_WIDTH},
            "opponent_goal": {"center": OPPONENT_GOAL_CENTER, "width": GOAL_WIDTH},
        },
        "team_chat": list(state.get("teamChat") or []),
        # Everything the spec left ambiguous, in one JSON-safe block so it can
        # be logged. Deliberately not the Frame object itself: this dict is
        # json.dumps'd into the model prompt, and the frame is recoverable from
        # team_id via `Frame.for_team`.
        "platform": {
            "team_id": frame.team_id,
            "stamina_scale": stamina_scale,
            "possession_inferred": possession.inferred,
        },
    }


# --------------------------------------------------------------- outbound

def to_wire(command: AgentCommand, transform: Frame, player_index: int) -> list[dict]:
    """AgentCommand -> the JSON array the platform expects.

    Coordinates go back through the inverse transform and are clamped to the
    real field, because a MOVE_TO a metre outside the touchline is exactly the
    sort of thing the platform drops without saying so.

    Relies on `AgentCommand`'s own per-type validation having already filled in
    or rejected missing parameters, so there is no defensive `or 0.0` here to
    turn a malformed command into a plausible-looking one.
    """
    t = command.type
    params: dict[str, object] = {}

    if t == "MOVE_TO":
        x, y = transform.to_platform(*command.target)
        params = {
            "target_x": _clamp(x, PLATFORM_X_LIMIT),
            "target_y": _clamp(y, PLATFORM_Y_LIMIT),
            "sprint": bool(command.sprint),
        }
    elif t == "FOLLOW_PLAYER":
        params = {
            "target_player_id": int(command.target_player_id),
            "target_team": command.target_team,
            "distance": float(command.distance),
        }
    elif t == "SHOOT":
        params = {"aim_location": command.aim_location, "power": float(command.power)}
    elif t == "PASS":
        # The wire calls this "type"; the model calls it pass_type so it does
        # not collide with the command type. This is the only renamed field.
        params = {"target_player_id": int(command.target_player_id), "type": command.pass_type}
    elif t == "GK_DISTRIBUTE":
        params = {"target_player_id": int(command.target_player_id), "method": command.method}
    elif t == "PRESS_BALL":
        params = {"intensity": float(command.intensity)}
    elif t == "MARK":
        params = {"target_player_id": int(command.target_player_id), "tightness": command.tightness}
    elif t == "SLIDE_TACKLE":
        params = {
            "target_player_id": int(command.target_player_id),
            "sprint": bool(command.sprint),
            "distance": float(command.distance),
        }
    elif t == "INTERCEPT":
        params = {"aggressive": bool(command.aggressive)}
    elif t == "SET_STANCE":
        params = {"stance": int(command.stance)}
    # CLEAR_OVERRIDE and RESET take no parameters.

    duration = command.duration
    wire_command: dict[str, object] = {"commandType": t, "parameters": params}
    if EMIT_PLAYER_ID:
        wire_command["playerId"] = int(player_index)
    # Integral durations go out as ints to match the documented sample exactly,
    # in case the platform's deserialiser is stricter than JSON.
    # A command's own duration wins; otherwise fall back to the squad-wide
    # Params.command_duration (0 = today's behaviour). See that field for why
    # this exists and why it cannot be tested without a live match.
    if not duration:
        duration = DURATION_DEFAULT
    wire_command["duration"] = (
        int(duration) if float(duration).is_integer() else float(duration))
    return [wire_command]


# Middle fifth of the goal mouth counts as central; outside it, pick a corner.
_CENTER_FRACTION = 0.1
# See aim_location_for: an assumption, isolated so reversing it is a one-line change.
_LEFT_IS_HIGH_Y = True


def aim_location_for(
    point: Sequence[float],
    *,
    goal_center: Sequence[float] = OPPONENT_GOAL_CENTER,
    goal_width: float = GOAL_WIDTH,
    high: bool = False,
) -> str:
    """Nearest legal `aim_location` for a point in the normalised frame.

    Provided because the old policy aimed SHOOT at a coordinate and the real
    platform takes one of five enum values, so something has to bridge them.

    Two things the workshop spec does not pin down, made explicit rather than
    buried:
      - T/B is a HEIGHT axis (top/bottom of the goal), and the observation is
        two-dimensional, so geometry cannot choose it. `high` does, defaulting
        to the low corner — the higher-percentage finish and the only half of
        the axis a 2-D policy can reason about at all.
      - L/R is assumed to be the shooter's left/right, i.e. increasing y is
        "left". If that turns out to be the keeper's view, flip `_LEFT_IS_HIGH_Y`
        and nothing else changes.
    """
    offset = point[1] - goal_center[1]
    deadzone = goal_width * _CENTER_FRACTION
    if abs(offset) <= deadzone:
        return "CENTER"
    side = "L" if (offset > 0) == _LEFT_IS_HIGH_Y else "R"
    return ("T" if high else "B") + side


# -------------------------------------------------------------- validation

def validate(command: AgentCommand | None, obs: Mapping) -> tuple[bool, str]:
    """Would the platform actually act on this? Returns (ok, reason).

    The reason string is the point. The AWS sample drops an invalid command in
    silence, so a model that hallucinates a pass to a player who is not on the
    pitch produces a team that is subtly worse with no trace of why. Every
    rejection here is a line someone can grep for.
    """
    if command is None:
        return False, "no command"

    t = command.type
    if t not in COMMAND_TYPES:
        return False, f"unknown commandType {t!r}"

    me = obs["you"]
    my_number = me["number"]
    teammates = {p["number"]: p for p in obs["teammates"]}
    opponents = {p["number"]: p for p in obs["opponents"]}

    if t in GK_ONLY and me["role"] != "GK":
        return False, f"{t} is goalkeeper-only, this player is {me['role']}"

    if t == "MOVE_TO":
        if command.target is None or not _finite(*command.target):
            return False, f"MOVE_TO target is missing or not finite: {command.target!r}"

    elif t == "SHOOT":
        if command.aim_location not in AIM_LOCATIONS:
            return False, f"SHOOT aim_location {command.aim_location!r} not in {AIM_LOCATIONS}"
        if not _unit(command.power):
            return False, f"SHOOT power {command.power!r} outside [0, 1]"

    elif t == "PASS":
        ok, why = _is_teammate(command.target_player_id, teammates, my_number, "PASS")
        if not ok:
            return False, why
        if command.pass_type not in PASS_TYPES:
            return False, f"PASS type {command.pass_type!r} not in {PASS_TYPES}"

    elif t == "GK_DISTRIBUTE":
        ok, why = _is_teammate(command.target_player_id, teammates, my_number, "GK_DISTRIBUTE")
        if not ok:
            return False, why
        if command.method not in DISTRIBUTION_METHODS:
            return False, f"GK_DISTRIBUTE method {command.method!r} not in {DISTRIBUTION_METHODS}"

    elif t == "MARK":
        # MARK carries no target_team on the wire, so it is the opposition by
        # definition; marking your own player is not expressible.
        if command.target_player_id not in opponents:
            return False, f"MARK target {command.target_player_id!r} is not an opponent"
        if command.tightness not in TIGHTNESS_VALUES:
            return False, f"MARK tightness {command.tightness!r} not in {TIGHTNESS_VALUES}"

    elif t == "FOLLOW_PLAYER":
        if command.target_team not in TEAM_SIDES:
            return False, f"FOLLOW_PLAYER target_team {command.target_team!r} not in {TEAM_SIDES}"
        # Unlike MARK, this command names a side, so following a teammate is
        # legal. Validate against whichever side it actually named.
        # `me` normally carries team_code, but validate is also called on
        # observations assembled by tests and by callers that build their own.
        # Treat a missing code as "not my side" rather than raising: the
        # opponent branch below is the stricter of the two.
        my_side = TEAM_SIDE.get(me.get("team_code"))
        if my_side is not None and command.target_team == my_side:
            ok, why = _is_teammate(command.target_player_id, teammates, my_number, "FOLLOW_PLAYER")
            if not ok:
                return False, why
        elif command.target_player_id not in opponents:
            return False, f"FOLLOW_PLAYER target {command.target_player_id!r} is not an opponent"
        if command.distance is None or not _finite(command.distance) or command.distance < 0:
            return False, f"FOLLOW_PLAYER distance {command.distance!r} is not a usable distance"

    elif t == "PRESS_BALL":
        if not _unit(command.intensity):
            return False, f"PRESS_BALL intensity {command.intensity!r} outside [0, 1]"

    elif t == "SET_STANCE":
        if command.stance not in (0, 1, 2):
            return False, f"SET_STANCE stance {command.stance!r} not 0, 1 or 2"

    # INTERCEPT, CLEAR_OVERRIDE and RESET have nothing left to get wrong.
    return True, ""


def _unit(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def _is_teammate(
    target: object, teammates: Mapping[int, dict], my_number: int, label: str
) -> tuple[bool, str]:
    if target == my_number:
        return False, f"{label} targets the player issuing it (#{my_number})"
    if target not in teammates:
        return False, f"{label} target {target!r} is not a teammate"
    return True, ""
