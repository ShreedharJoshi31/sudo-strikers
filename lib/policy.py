"""Pure-Python policy — the safety net that always answers inside budget.

Why this exists
---------------
The platform discards any response that arrives after the decision budget and
substitutes IDLE (stand still). Over a 2-minute match at a 2-second interval
that is 60 decisions per player; every late one is a decision thrown away.

So the LLM is never on the critical path. This module produces a complete,
valid command in well under a millisecond, and `brain.py` only replaces it if
the model beats the deadline with something better.

Adapted from the Agentic Football Arena baseline AI (MIT). Thresholds are
collected in `Params` so `bench.py` can sweep them instead of us guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from command import AgentCommand
from wire import aim_location_for


@dataclass(frozen=True)
class Params:
    """Thresholds for the policy.

    SCALE NOTE — read before trusting any number here.

    Every value below was originally tuned against the Agentic Football Arena,
    an unofficial reconstruction played on a 40 x 25 pitch. The real Cup pitch
    is 110 x 70: 2.75x longer, 2.8x wider, 7.7x the area, with the same five
    players a side. The distances have therefore been rescaled geometrically
    (lengths x2.75, the width-axis term x2.8).

    A geometric rescale is a STARTING POINT, NOT A TUNING. Player and ball
    speeds are unpublished, so the ratio of "distance a player can cover in one
    tick" to "pitch size" is unknown and is what these gates really encode.
    Re-derive them from practice matches before believing any of them.

    Two consequences of the new ratio that the numbers alone do not show:
    each player now covers ~2.2x the area a real footballer does, so space is
    abundant and lanes are usually open; and pressing costs far more stamina
    over these distances than it did on the small pitch.

    Dimensionless values (fractions, ratios, weights, stamina) are NOT scaled.
    """

    # --- shooting: (max distance, minimum lane clearance) gates, nearest first
    shoot_near_dist: float = 30.25        # was 11.0
    shoot_near_lane: float = 1.92         # was 0.7
    shoot_mid_dist: float = 45.38         # was 16.5
    shoot_mid_lane: float = 5.50          # was 2.0
    shoot_far_dist: float = 63.25         # was 23.0
    shoot_far_lane: float = 9.90          # was 3.6
    #: Widest angle off the goal's centre line still worth shooting from, in
    #: degrees. The goal mouth is only 10 m wide on a 70 m pitch, so past this
    #: there is almost no target left: the ball crosses the face and is a
    #: turnover, not a shot. Participants measured 27 SHOOT commands producing
    #: ZERO recorded shots because they gated on distance alone - a striker
    #: parked wide reads as "in range" on any distance-only check, including a
    #: Euclidean one. 45 degrees is roughly the corner of the box.
    shoot_max_angle: float = 45.0
    tired_shoot_penalty: float = 5.50     # was 2.0; shrink near-range when stamina is low
    tired_stamina: float = 45.0           # stamina units, not a distance - unscaled

    # --- passing
    pass_max_dist: float = 66.0           # was 24.0
    pass_min_lane: float = 2.75           # was 1.0
    pass_ideal_dist: float = 27.5         # was 10.0; error grows per metre, so favour short
    # The four score terms are each (distance x weight), so a uniform rescale of
    # the distances leaves their relative balance untouched. The weights are
    # therefore unchanged - only the bare distance constant below moved.
    pass_gain_w: float = 1.0
    pass_lane_w: float = 1.6
    pass_dist_w: float = 0.28
    pass_goal_w: float = 0.20
    pass_goal_ref: float = 55.0           # was a hardcoded 20.0 inside the scorer
    #: Pressure above which we stop asking "is this a good pass?" and play the
    #: safest one regardless.
    #:
    #: MEASURED and left alone: on 393 replayed real on-ball ticks, sweeping
    #: this over 0.25 / 0.50 / 0.80 produced a BYTE-IDENTICAL command mix. The
    #: squeezed branch is not what drives our passing - `_best_pass` returning
    #: an option above `pass_min_success` is. If PASS share needs to come down,
    #: raise `pass_min_success` or bias `_best_pass`; changing this does nothing.
    pass_when_pressed: float = 0.25       # pressure units - unscaled
    pass_score_floor: float = 3.30        # was 1.2; scores scale with the distances
    pass_min_success: float = 0.45        # play it if the model says it likely arrives

    # --- carrying / clearing
    clear_pressure: float = 1.1           # pressure units - unscaled
    clear_own_third: float = 0.35         # fraction of pitch length - unscaled
    carry_step: float = 24.75             # was 9.0

    # --- defending
    tackle_range: float = 6.05            # was 2.2

    # --- support shape
    fwd_push: float = 22.0                # was 8.0
    mid_push: float = -5.50               # was -2.0
    def_drop: float = 24.75               # was 9.0
    spread_y: float = 8.40                # was 3.0; width axis, so x2.8

    # --- goalkeeper
    gk_depth_base: float = 2.75           # was 1.0
    gk_depth_gain: float = 0.10           # depth per unit distance, a ratio - unscaled
    gk_depth_max: float = 9.35            # was 3.4
    gk_claim_range: float = 8.25          # was 3.0
    gk_outlet_lane: float = 5.50          # was 2.0
    # gk_dive_speed / gk_dive_range are gone with GK_DIVE, which the real
    # platform does not implement. Keeper positioning is what saves goals.

    # --- command parameters the real platform takes and the arena did not.
    # PRESS_BALL is one dial covering what used to be two commands: above 0.3
    # the player attempts tackles, above 0.5 they sprint. So the old TACKLE is
    # simply a harder press, not a separate action.
    press_intensity: float = 0.62         # sprints and challenges, but rations stamina
    tackle_intensity: float = 0.88        # committed, for when the carrier is in range
    # Pressing costs far more over 110x70 than it did over 40x25, so these are
    # the first things to pull back if stamina curves look bad.
    shoot_power_near: float = 0.62        # placement beats power close in
    shoot_power_far: float = 1.0
    shoot_aim_spread: float = 4.4         # was 1.6 on the small pitch
    mark_tightness: str = "TIGHT"
    intercept_aggressive: bool = True
    #: Ticks to keep a command alive for. UNVALIDATED HYPOTHESIS - see below.
    #:
    #: We currently emit duration=0 on everything. AGENT_PROTOCOL.md section 4
    #: calls 0 "instant", but MEASURED in the 0-3 against Bonkers United: the
    #: forward sent 20 PASS commands on 20 distinct ticks, held the ball on all
    #: 20, had a median 1.4u pass to a receiver with 2.84u of lane clearance and
    #: no opponent within 1.7u - and the ball stayed with the passer 17 times.
    #: The platform echoed PASS back in `previousCommand` every time, so the
    #: command was received; the engine then ran AdvanceToGoalWithBall instead.
    #: Acceptance by type that match: MARK 100%, MOVE_TO 69%, PRESS_BALL 20%,
    #: PASS 0%. Positional commands stick; on-ball commands do not.
    #:
    #: The one participant who measured durations was sending 3-5 and found
    #: capping to 1 "worth more than any prompt change" - i.e. durations govern
    #: persistence, and 0 is below their entire tested range. So 1 is the
    #: smallest value that could let an on-ball command survive to execution.
    #:
    #: This CANNOT be validated offline: it changes what the ENGINE does, not
    #: what we decide. It needs one practice match, judged on PASS acceptance.
    #: 0 restores exactly today's behaviour.
    command_duration: int = 0
    #: Highest ball (metres above the turf) still worth an INTERCEPT. Every
    #: other distance check here is 2D, so without this gate a ball sailing
    #: overhead looks like a loose ball at your feet and the whole team chases
    #: something it cannot reach. A standing player wins the ball a little above
    #: head height; 2.5 m is that, and it is a Param so it can be swept.
    #: Measured ball height in real matches spans 0.139-0.986 platform units
    #: (x8 = 1.1-7.9 m), with 0.143 the resting radius. Gate high enough that
    #: only a genuinely airborne ball is skipped - suppressing a legitimate
    #: claim costs more than one wasted chase.
    intercept_max_height: float = 4.0
    #: If we are about to repeat the same holding run AND an opponent is inside
    #: this range, do something else instead. "Command diversity is the single
    #: biggest lever" (organisers), and a repeated MOVE_TO to a spot we already
    #: occupy is the least valuable tick available.
    #:
    #: SWEPT on 3,783 replayed real ticks, closed-loop (our own previous command
    #: fed back, as the platform does). Top-command share, with shot quality
    #: identical at every setting (7 good / 1 bad):
    #:      0.0 (off) 74.5%   8.0 71.7%   16.5 69.4%   25.0 62.4%   40.0 50.7%
    #: The one documented WINNING team in this competition ran MOVE_TO at 61%,
    #: so 25.0 lands on the only external benchmark available. 40.0 buys more
    #: diversity but starts pulling players off their station for its own sake.
    repeat_press_range: float = 25.0
    #: How close the previous target must be to count as "the same run".
    repeat_same_target: float = 3.0
    carry_sprint_stamina: float = 35.0    # carry at pace while there is gas left
    support_sprint_stamina: float = 25.0
    through_ball_space: float = 16.5      # free space behind a receiver to play into


DEFAULT = Params()


# --------------------------------------------------------------------- helpers

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_to_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + dx * t), py - (ay + dy * t))


def _lane(frm, to, opponents, ignore_ids=()) -> float:
    """Distance from the pass/shot line to the nearest opponent."""
    best = 99.0
    for o in opponents:
        if o["id"] in ignore_ids:
            continue
        best = min(best, _point_to_segment(o["position"], frm, to))
    return best


def _best_pass(obs: dict, under_pressure: bool):
    """Ask the interception model for a receiver and a pass type.

    Imported here rather than at module scope on purpose: `passing.py` takes
    its geometry helpers from this file, so a top-level import would be a
    cycle. Python caches the module, so after the first call this is a dict
    lookup and costs nothing measurable.

    Returns None if the model has nothing to offer or fails. Policy is the
    thing that must always answer, so it never lets an optional component
    take a decision down with it.
    """
    try:
        import passing
        return passing.best_pass(
            obs, policy="SAFEST" if under_pressure else "BEST_VALUE"
        )
    except Exception:
        return None


def _nearest_open_mate(obs: dict, mates: list, p: Params):
    """Last-ditch pass target if the interception model is unavailable.

    Deliberately crude - nearest teammate with a lane - because its only job is
    to keep the ball moving on a path that should never run.
    """
    pos = obs["you"]["position"]
    opts = [m for m in mates
            if _dist(pos, m["position"]) <= p.pass_max_dist
            and _lane(pos, m["position"], obs["opponents"]) >= p.pass_min_lane]
    if not opts:
        return None
    return min(opts, key=lambda m: (_dist(pos, m["position"]), m["id"]))


def _shot_power(d_goal: float, p: Params) -> float:
    """Interpolate SHOOT power from range.

    The platform takes power as 0..1 rather than a target point, and the goal
    is only 10 wide in a 70-wide pitch, so distance shooting is low percentage.
    Close in, trade power for placement; from range there is no point being coy.
    """
    span = max(1e-6, p.shoot_far_dist - p.shoot_near_dist)
    t = max(0.0, min(1.0, (d_goal - p.shoot_near_dist) / span))
    return round(p.shoot_power_near + (p.shoot_power_far - p.shoot_power_near) * t, 2)


def _jitter(obs: dict, spread: float) -> float:
    """Deterministic aim scatter in [-spread, +spread].

    Derived from the tick and player id rather than a stateful RNG, so a long
    lived server replays identically and one match cannot perturb the next.
    """
    key = f"{obs.get('tick', 0)}:{obs['you']['id']}"
    h = 2166136261
    for ch in key:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF * 2.0 - 1.0) * spread


def defensive_assignment(obs: dict) -> dict:
    """Who presses the ball, and who marks whom.

    Every player computes this from the same observation and ties break on id,
    so all four independently agree without any message passing. This is the
    only coordination in the system, and it is why five separate runtimes
    defend as a unit.

    Returns {"presser": player_id, "marks": {defender_id: threat_id}}.
    """
    me = obs["you"]
    ball = obs["ball"]["position"]
    goal = obs["pitch"]["your_goal"]["center"]
    carrier_id = obs["ball"]["owner_id"]

    # KEEP A STRIKER UP THE PITCH.
    #
    # Every non-GK player used to be eligible to press and mark, which reads as
    # "everyone defends" and is how a lone forward ends up tracking back into
    # its own box. Measured in a 1-4 defeat: the ball was in our half 81% of the
    # match and our most advanced player's median x was +0.41 - the halfway line
    # - while the opponent's most advanced sat at -0.93, inside our half. With
    # nobody ahead of the ball there is no outlet, so possession went back to
    # the keeper (38 GK_DISTRIBUTE in one match, 44% of its ticks), got hoofed,
    # and came straight back.
    #
    # AGENT_PROTOCOL.md section 6.4 says the same thing for the forward: press
    # when "they've got it in their own half - win it high", and otherwise
    # "hold your width and wait". So the forward joins the defensive pool only
    # when the ball is in the opponent's half.
    #
    # The single-presser invariant still holds: the forward is only excluded
    # while somebody else can do the job.
    outfield = [me] + [m for m in obs["teammates"] if m["role"] != "GK"]
    ball_in_our_half = ball[0] < obs["pitch"]["length"] / 2.0
    defenders = [d for d in outfield
                 if not (ball_in_our_half and d["role"] == "FORWARD")]
    if not defenders:
        defenders = outfield
    presser = min(defenders, key=lambda d: (_dist(d["position"], ball), d["id"]))

    remaining = [d for d in defenders if d["id"] != presser["id"]]
    threats = [o for o in obs["opponents"] if o["role"] != "GK" and o["id"] != carrier_id]
    threats.sort(key=lambda o: (_dist(o["position"], goal), o["id"]))

    marks: dict[str, str] = {}
    for t in threats:
        if not remaining:
            break
        picked = min(remaining, key=lambda d: (_dist(d["position"], t["position"]), d["id"]))
        remaining = [d for d in remaining if d["id"] != picked["id"]]
        marks[picked["id"]] = t["id"]

    return {"presser": presser["id"], "marks": marks}


def _cx(v: float, length: float) -> float:
    return max(0.6, min(length - 0.6, v))


def _cy(v: float, width: float) -> float:
    return max(0.6, min(width - 0.6, v))


# ----------------------------------------------------------------------- brain

class Policy:
    """One instance per team. `decide` is called per player, per tick."""

    def __init__(self, params: Params = DEFAULT, seed: int = 0) -> None:
        self.p = params

    def decide(self, player_id: str, obs: dict) -> AgentCommand:
        if obs["you"]["role"] == "GK":
            return self._goalkeeper(obs)
        possession = obs["possession"]
        if possession == "you":
            return self._on_ball(obs)
        if possession == "teammate":
            return self._support(obs)
        if possession == "opponent":
            return self._defend(obs)
        return self._loose(obs)

    # ------------------------------------------------------------- on the ball
    def _on_ball(self, obs: dict) -> AgentCommand:
        p = self.p
        me = obs["you"]
        pitch = obs["pitch"]
        goal = pitch["opponent_goal"]["center"]
        opps = obs["opponents"]
        mates = [m for m in obs["teammates"] if m["role"] != "GK"]
        gk_ids = tuple(o["id"] for o in opps if o["role"] == "GK")
        pos = me["position"]

        d_goal = _dist(pos, goal)
        lane = _lane(pos, goal, opps, ignore_ids=gk_ids)
        near = p.shoot_near_dist
        if me["stamina"] <= p.tired_stamina:
            near -= p.tired_shoot_penalty

        # How far off the goal's centre line we are standing. Distance alone is
        # not range: (50, 25) is 25 m from goal and 79 degrees off it, which is
        # a cross, not a chance.
        goal_angle = abs(math.degrees(math.atan2(
            abs(pos[1] - goal[1]), max(0.1, abs(goal[0] - pos[0])))))
        in_view = goal_angle <= p.shoot_max_angle

        if in_view and (
            (d_goal < near and lane > p.shoot_near_lane)
            or (d_goal < p.shoot_mid_dist and lane > p.shoot_mid_lane)
            or (d_goal < p.shoot_far_dist and lane > p.shoot_far_lane)
        ):
            aim_point = (goal[0], goal[1] + _jitter(obs, p.shoot_aim_spread))
            return AgentCommand(
                type="SHOOT",
                aim_location=aim_location_for(aim_point),
                power=_shot_power(d_goal, p),
                rationale=f"{d_goal:.0f}m, lane {lane:.1f}m",
            )

        # Receiver AND pass type both come from the interception model, which
        # compares time-to-lane against ball flight rather than measuring a
        # static gap. Under pressure ask for the safest ball; otherwise ask for
        # the one worth playing, which is not the same question.
        pressure = me.get("pressure", 0.0)
        squeezed = pressure > p.pass_when_pressed
        option = _best_pass(obs, squeezed)
        if option is not None and (option.p_success >= p.pass_min_success or squeezed):
            return AgentCommand(
                type="PASS",
                target_player_id=option.receiver_id,
                pass_type=option.type,
                rationale=(option.reason or f"pass to #{option.receiver_number}")[:70],
            )
        if option is None:
            fallback = _nearest_open_mate(obs, mates, p)
            if fallback is not None and squeezed:
                return AgentCommand(
                    type="PASS", target_player_id=fallback["id"], pass_type="GROUND",
                    rationale=f"pressed, simple ball to #{fallback['number']}",
                )

        # The platform has no CLEAR. The nearest real equivalent to hammering it
        # away is an AERIAL ball to whoever is furthest upfield: it travels over
        # a defender in the lane instead of through them, which is the whole
        # point of clearing. With nobody to aim at, carry it out instead.
        if pressure > p.clear_pressure and pos[0] < pitch["length"] * p.clear_own_third:
            outlet = max(mates, key=lambda m: m["position"][0], default=None)
            if outlet is not None:
                return AgentCommand(
                    type="PASS",
                    target_player_id=outlet["id"],
                    pass_type="AERIAL",
                    rationale="pressed deep, clear it long",
                )

        if pos[0] > pitch["length"] * 0.62:
            toward = (pos[0] + p.carry_step, goal[1] * 0.55 + pos[1] * 0.45)
        else:
            toward = (pos[0] + p.carry_step, pos[1] + self._free_side(obs) * 3.0)
        # No DRIBBLE on this platform: with possession, moving IS carrying.
        return AgentCommand(
            type="MOVE_TO",
            target=(_cx(toward[0], pitch["length"]), _cy(toward[1], pitch["width"])),
            sprint=me["stamina"] > p.carry_sprint_stamina,
            rationale="carry into space",
        )

    def _free_side(self, obs: dict) -> float:
        y = obs["you"]["position"][1]
        up = sum(1 for o in obs["opponents"] if o["position"][1] > y)
        return 1.0 if up < len(obs["opponents"]) - up else -1.0

    # ----------------------------------------------------------- off the ball
    def _support(self, obs: dict) -> AgentCommand:
        p = self.p
        me = obs["you"]
        pitch = obs["pitch"]
        ball = obs["ball"]["position"]
        home = me.get("home_position", me["position"])
        role = me["role"]

        if role == "FORWARD":
            x = min(pitch["length"] - 6.0, max(ball[0] + p.fwd_push, home[0] + 4.0))
        elif role == "MIDFIELDER":
            x = max(ball[0] + p.mid_push, home[0])
        else:
            x = max(home[0] - 1.0, ball[0] - p.def_drop)
        y = home[1] * 0.62 + ball[1] * 0.38
        for m in obs["teammates"]:
            if abs(m["position"][1] - y) < 2.5 and abs(m["position"][0] - x) < 4.0:
                y += p.spread_y if y < pitch["width"] / 2 else -p.spread_y
        target = (_cx(x, pitch["length"]), _cy(y, pitch["width"]))

        # DIVERSITY GATE. If this is the same holding run we already sent, the
        # run is still executing and re-sending it buys nothing. When an
        # opponent is close enough to be worth disturbing, press instead: it is
        # a different command, it is useful, and it is the lever the organisers
        # say decides matches. Only fires on a REPEAT, so a first-time run and
        # every on-ball decision are untouched.
        prev = obs.get("previous_command") or {}
        if prev.get("type") == "MOVE_TO" and prev.get("target") is not None:
            if _dist(prev["target"], target) < p.repeat_same_target:
                near = sorted(
                    (o for o in obs["opponents"]
                     if _dist(o["position"], me["position"]) < p.repeat_press_range),
                    key=lambda o: _dist(o["position"], me["position"]))
                if near:
                    # WHICH different command depends on who has the ball.
                    # Pressing indiscriminately is how a team empties its own
                    # half: participants measured a 45% PRESS_BALL share
                    # dragging every defender to the ball and losing 0-2. Only
                    # the designated presser presses - the same single-presser
                    # rule defensive_assignment() already enforces - and anyone
                    # else picks up the nearest opponent instead, which is
                    # useful, is not a repeat, and keeps the shape.
                    is_presser = (
                        obs["possession"] == "opponent"
                        and defensive_assignment(obs)["presser"] == me["id"]
                    )
                    if is_presser:
                        return AgentCommand(
                            type="PRESS_BALL",
                            intensity=p.press_intensity,
                            rationale="already holding that run - press instead",
                        )
                    return AgentCommand(
                        type="MARK",
                        target_player_id=near[0]["id"],
                        tightness="LOOSE",
                        rationale="already holding that run - pick a man up",
                    )

        return AgentCommand(
            type="MOVE_TO",
            target=target,
            sprint=me["stamina"] > p.support_sprint_stamina,
            rationale="offer an angle",
        )

    def _defend(self, obs: dict) -> AgentCommand:
        p = self.p
        me = obs["you"]
        pitch = obs["pitch"]
        carrier_id = obs["ball"]["owner_id"]

        plan = defensive_assignment(obs)
        if plan["presser"] == me["id"]:
            carrier = next((o for o in obs["opponents"] if o["id"] == carrier_id), None)
            # TACKLE does not exist; intensity above 0.3 already attempts one.
            if carrier is not None and _dist(me["position"], carrier["position"]) < p.tackle_range:
                return AgentCommand(
                    type="PRESS_BALL", intensity=p.tackle_intensity, rationale="in range"
                )
            return AgentCommand(
                type="PRESS_BALL", intensity=p.press_intensity, rationale="close the carrier"
            )

        mark = plan["marks"].get(me["id"])
        if mark is not None:
            return AgentCommand(
                type="MARK", target_player_id=mark, tightness=p.mark_tightness,
                rationale="pick up runner",
            )

        home = me.get("home_position", me["position"])
        return AgentCommand(
            type="MOVE_TO",
            target=(_cx(home[0] - 2.0, pitch["length"]), _cy(home[1], pitch["width"])),
            rationale="recover shape",
        )

    def _loose(self, obs: dict) -> AgentCommand:
        me = obs["you"]
        ball = obs["ball"]["position"]
        bv = obs["ball"]["velocity"]
        landing = (ball[0] + bv[0] * 0.6, ball[1] + bv[1] * 0.6)
        mine = _dist(me["position"], landing)
        others = [_dist(m["position"], landing) for m in obs["teammates"] if m["role"] != "GK"]
        airborne = obs["ball"].get("height", 0.0) > self.p.intercept_max_height
        # The same rule as defensive_assignment: a lone striker does not chase
        # loose balls into its own half. Excluding the forward from marking was
        # not enough on its own - measured in the 09:16 win, the forward still
        # spent 52% of its ticks defending, and INTERCEPT from HERE was the
        # single largest slice. A striker that sprints 6 units back for a 50-50
        # is a striker who is not there for the next counter.
        chases = not (me["role"] == "FORWARD"
                      and landing[0] < obs["pitch"]["length"] / 2.0)
        if chases and (not others or mine <= min(others) + 0.3) and not airborne:
            return AgentCommand(
                type="INTERCEPT", aggressive=self.p.intercept_aggressive,
                rationale="closest to the drop",
            )
        return self._support(obs)

    # -------------------------------------------------------------- goalkeeper
    def _goalkeeper(self, obs: dict) -> AgentCommand:
        p = self.p
        me = obs["you"]
        pitch = obs["pitch"]
        ball = obs["ball"]["position"]
        goal = pitch["your_goal"]["center"]
        pos = me["position"]

        # A keeper in possession restarts with GK_DISTRIBUTE, not PASS - the
        # platform drops a PASS from the keeper. THROW is the accurate option
        # and KICK the long one, so an open outlet gets thrown to and a covered
        # one gets kicked clear. That kick is also the only CLEAR we still have.
        if obs["possession"] == "you":
            mates = [m for m in obs["teammates"] if m["role"] != "GK"]
            if mates:
                safe = [m for m in mates
                        if _lane(pos, m["position"], obs["opponents"]) > p.gk_outlet_lane]
                if safe:
                    target = max(safe, key=lambda m: m["position"][0])
                    return AgentCommand(
                        type="GK_DISTRIBUTE", target_player_id=target["id"],
                        method="THROW", rationale="restart from the back",
                    )
                furthest = max(mates, key=lambda m: m["position"][0])
                return AgentCommand(
                    type="GK_DISTRIBUTE", target_player_id=furthest["id"],
                    method="KICK", rationale="no safe outlet, kick it long",
                )

        # There is no GK_DIVE on this platform. That costs nothing measurable:
        # sweeping the old dive range over 6-18 changed no result, because at a
        # ~2s decision interval a shot has already arrived. Positioning saves
        # goals; the angle-holding MOVE_TO below is the keeper's real work.
        if (_dist(pos, ball) < p.gk_claim_range
                and obs["ball"].get("height", 0.0) <= p.intercept_max_height
                and obs["possession"] != "opponent"):
            return AgentCommand(
                type="INTERCEPT", aggressive=p.intercept_aggressive, rationale="claim it"
            )

        dx, dy = ball[0] - goal[0], ball[1] - goal[1]
        n = math.hypot(dx, dy) or 1.0
        depth = p.gk_depth_base + min(p.gk_depth_max, _dist(ball, goal) * p.gk_depth_gain)
        return AgentCommand(
            type="MOVE_TO",
            target=(_cx(goal[0] + dx / n * depth, pitch["length"]),
                    _cy(goal[1] + dy / n * depth * 0.72, pitch["width"])),
            rationale="hold the angle",
        )
