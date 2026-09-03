"""Hard rules applied to the FINAL command, whatever produced it.

Why this exists
---------------
Observed in real matches: the keeper wandered upfield, defenders neither
tackled nor got the ball away from their own goal, and forwards passed from
positions where they should have shot. The policy does not do these things —
the model does, and until now nothing overruled it. `_validate` only rejects
commands the platform would DROP; a keeper sprinting to the halfway line is
perfectly legal and perfectly wrong.

So this is the layer that says no. It runs after the model and after the
policy, so a rule holds no matter which of them produced the command.

Design rules for anything added here
------------------------------------
1. Only situations where there is one defensible answer. Anything arguable
   belongs in the prompt or the policy, not here — a guardrail that fires on a
   judgement call removes the judgement we are paying the model for.
2. Prefer CLAMPING to REPLACING. Pulling a keeper's target back onto its own
   third keeps the model's intent; swapping its command for another discards it.
3. Every override returns a reason. They are counted in /stats, so a rule that
   fires constantly is visible rather than silently rewriting the whole match.

Frame: everything here is the normalised observation frame — own goal at x=0,
opponent goal at x=pitch length, y increasing across the pitch.

NOTE ON CLEARANCES: this platform has no CLEAR command. The eleven types do not
include one, so "get it out of the box" has to be expressed as an AERIAL PASS
to the most advanced teammate, or GK_DISTRIBUTE by a keeper.
"""

from __future__ import annotations

import math

from command import AgentCommand
from wire import aim_location_for

#: Penalty area, as a fraction of pitch length from the goal line. Real
#: football is 16.5m of a 105m pitch; this keeps the same proportion.
BOX_DEPTH_FRACTION = 0.157

#: Half-width of the penalty area as a fraction of pitch width (40.3m of 68m).
BOX_HALF_WIDTH_FRACTION = 0.296

#: A keeper may not target beyond this fraction of the pitch. Generous enough
#: to sweep behind a high line, short of anything that abandons the goal.
GK_MAX_X_FRACTION = 0.33

#: Close enough to commit to a sliding challenge.
TACKLE_RANGE = 4.0

#: Close enough that pressing is worth the stamina.
PRESS_RANGE = 18.0

#: Commands that cannot be executed without the ball at your feet.
#:
#: The prompt says "Requires the ball" on each of these and the model ignores
#: it: measured, it answered PASS on 3 of 3 ticks where a TEAMMATE had
#: possession. Nothing caught it - `wire.validate` only rejects what the
#: platform would drop, and an impossible pass is well-formed. It just does
#: nothing, which is a wasted tick that reads as a healthy one.
BALL_REQUIRED = ("PASS", "SHOOT", "GK_DISTRIBUTE")


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _own_box(obs: dict) -> tuple[float, float, float]:
    """(max_x, min_y, max_y) of our own penalty area."""
    length = obs["pitch"]["length"]
    width = obs["pitch"]["width"]
    half = width * BOX_HALF_WIDTH_FRACTION
    return length * BOX_DEPTH_FRACTION, width / 2 - half, width / 2 + half


def _opp_box(obs: dict) -> tuple[float, float, float]:
    """(min_x, min_y, max_y) of the opponent's penalty area."""
    length = obs["pitch"]["length"]
    width = obs["pitch"]["width"]
    half = width * BOX_HALF_WIDTH_FRACTION
    return length * (1 - BOX_DEPTH_FRACTION), width / 2 - half, width / 2 + half


def in_own_box(obs: dict, point) -> bool:
    max_x, min_y, max_y = _own_box(obs)
    return point[0] <= max_x and min_y <= point[1] <= max_y


def in_opponent_box(obs: dict, point) -> bool:
    min_x, min_y, max_y = _opp_box(obs)
    return point[0] >= min_x and min_y <= point[1] <= max_y


def _most_advanced_mate(obs: dict):
    """Furthest-forward outfield teammate — the target for a clearance."""
    mates = [m for m in obs.get("teammates", ()) if m.get("role") != "GK"]
    return max(mates, key=lambda m: m["position"][0], default=None)


def _carrier(obs: dict):
    owner = obs.get("ball", {}).get("owner_id")
    if not owner:
        return None
    return next((o for o in obs.get("opponents", ()) if o["id"] == owner), None)


# ------------------------------------------------------------------- rules

def apply(cmd: AgentCommand, obs: dict,
          fallback: AgentCommand | None = None) -> tuple[AgentCommand, str | None]:
    """Return `(command, reason)`. `reason` is None when nothing was changed.

    `fallback` is the policy's command for this tick. It is used where the
    model asked for something impossible and there is no single obviously
    correct substitute — the policy has already worked out what this player
    should be doing given who has the ball, so deferring to it beats inventing
    a replacement here.
    """
    me = obs.get("you") or {}
    role = me.get("role")
    pos = me.get("position") or [0.0, 0.0]
    have_ball = obs.get("possession") == "you"
    they_have_it = obs.get("possession") == "opponent"

    # --- 0. You cannot kick a ball you do not have --------------------------
    # Checked before anything else: every rule below reasons about what to do
    # with the ball, and none of that applies if we have not got it.
    if cmd.type in BALL_REQUIRED and not have_ball:
        holder = obs.get("possession", "nobody")
        if fallback is not None and fallback.type not in BALL_REQUIRED:
            return fallback, f"{cmd.type} without the ball ({holder} has it)"
        return (
            AgentCommand(type="INTERCEPT", aggressive=True,
                         rationale="go and win it first"),
            f"{cmd.type} without the ball ({holder} has it)",
        )

    # --- 1. Keeper leash -----------------------------------------------------
    # The one that was actually losing matches. A keeper's MOVE_TO is legal
    # anywhere on the pitch, so nothing stopped the model walking it upfield.
    # Clamped, not rejected: the intent (come for the ball) is usually right,
    # only the distance is wrong.
    if role == "GK" and cmd.type == "MOVE_TO" and cmd.target is not None:
        limit = obs["pitch"]["length"] * GK_MAX_X_FRACTION
        if cmd.target[0] > limit:
            return (
                cmd.model_copy(update={"target": (limit, cmd.target[1])}),
                f"GK leash: target x {cmd.target[0]:.0f} -> {limit:.0f}",
            )

    # A keeper should also not be chasing the ball around the pitch.
    if role == "GK" and cmd.type in ("PRESS_BALL", "SLIDE_TACKLE", "MARK"):
        ball = obs.get("ball", {}).get("position") or [0.0, 0.0]
        if not in_own_box(obs, ball):
            goal = obs["pitch"]["your_goal"]["center"]
            return (
                AgentCommand(type="MOVE_TO", target=(goal[0] + 3.0, goal[1]),
                             rationale="hold the line"),
                f"GK must not {cmd.type} outside its own box",
            )

    # --- 2. Get it out of our own box ---------------------------------------
    # No CLEAR command exists, so a clearance is an aerial ball to the most
    # advanced teammate. Playing a short pass out of your own six-yard box is
    # how you concede; this is the one place "hoof it" is correct.
    if have_ball and in_own_box(obs, pos):
        if role == "GK":
            if cmd.type != "GK_DISTRIBUTE":
                mate = _most_advanced_mate(obs)
                if mate is not None:
                    return (
                        AgentCommand(type="GK_DISTRIBUTE",
                                     target_player_id=_index(mate["id"]),
                                     method="KICK",
                                     rationale="clear the area"),
                        "GK holding the ball in its own box",
                    )
        elif cmd.type not in ("PASS", "SHOOT"):
            mate = _most_advanced_mate(obs)
            if mate is not None:
                return (
                    AgentCommand(type="PASS", target_player_id=_index(mate["id"]),
                                 pass_type="AERIAL", rationale="clear the box"),
                    f"{cmd.type} while holding the ball in our own box",
                )

    # --- 3. Shoot when you are in their box ---------------------------------
    # A forward inside the penalty area with the ball has one job. Passing from
    # here is what "forwards don't shoot properly" looks like from the touchline.
    if have_ball and in_opponent_box(obs, pos) and cmd.type != "SHOOT":
        keeper = next((o for o in obs.get("opponents", ()) if o.get("role") == "GK"), None)
        aim_point = (obs["pitch"]["length"], _away_from(keeper, obs))
        return (
            AgentCommand(type="SHOOT",
                         aim_location=aim_location_for(aim_point),
                         power=0.85, rationale="in the box, shoot"),
            f"{cmd.type} with the ball inside their penalty area",
        )

    # --- 4. Actually challenge the carrier ----------------------------------
    # "Defenders don't tackle" was the report. If the man with the ball is
    # within a stride and we are not already engaging him, engage him.
    if they_have_it and role != "GK" and cmd.type not in (
            "SLIDE_TACKLE", "PRESS_BALL", "MARK", "INTERCEPT"):
        carrier = _carrier(obs)
        if carrier is not None:
            gap = _dist(pos, carrier["position"])
            if gap <= TACKLE_RANGE:
                return (
                    AgentCommand(type="SLIDE_TACKLE",
                                 target_player_id=_index(carrier["id"]),
                                 sprint=True, rationale="win it back"),
                    f"{cmd.type} with the carrier {gap:.0f}m away",
                )
            if gap <= PRESS_RANGE:
                return (
                    AgentCommand(type="PRESS_BALL", intensity=0.75,
                                 rationale="close him down"),
                    f"{cmd.type} with the carrier {gap:.0f}m away",
                )

    return cmd, None


def _away_from(keeper, obs: dict) -> float:
    """A y inside the goal mouth, on the side the keeper is not covering."""
    goal = obs["pitch"]["opponent_goal"]["center"]
    half = obs["pitch"]["opponent_goal"]["width"] / 2.0
    if keeper is None:
        return goal[1]
    # Aim to whichever post the keeper is further from.
    return goal[1] - half * 0.8 if keeper["position"][1] > goal[1] else goal[1] + half * 0.8


def _index(player_id) -> int:
    """Player index from a normalised id like "away_3"."""
    if isinstance(player_id, int):
        return player_id
    tail = str(player_id).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0
