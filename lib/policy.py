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


@dataclass(frozen=True)
class Params:
    # --- shooting: (max distance, minimum lane clearance) gates, nearest first
    shoot_near_dist: float = 11.0
    shoot_near_lane: float = 0.7
    shoot_mid_dist: float = 16.5
    shoot_mid_lane: float = 2.0
    shoot_far_dist: float = 23.0
    shoot_far_lane: float = 3.6
    tired_shoot_penalty: float = 2.0      # shrink near-range when stamina is low
    tired_stamina: float = 45.0

    # --- passing
    pass_max_dist: float = 24.0
    pass_min_lane: float = 1.0
    pass_ideal_dist: float = 10.0         # engine adds error per metre, so favour short
    pass_gain_w: float = 1.0
    pass_lane_w: float = 1.6
    pass_dist_w: float = 0.28
    pass_goal_w: float = 0.20
    pass_when_pressed: float = 0.25
    pass_score_floor: float = 1.2

    # --- carrying / clearing
    clear_pressure: float = 1.1
    clear_own_third: float = 0.35
    carry_step: float = 9.0

    # --- defending
    tackle_range: float = 2.2

    # --- support shape
    fwd_push: float = 8.0
    mid_push: float = -2.0   # swept: holding just behind the ball beats pushing ahead of it
    def_drop: float = 9.0
    spread_y: float = 3.0

    # --- goalkeeper
    gk_depth_base: float = 1.0    # swept: a keeper off its line cannot get back within a 2s tick
    gk_depth_gain: float = 0.10
    gk_depth_max: float = 3.4
    gk_dive_speed: float = 9.0
    gk_dive_range: float = 12.0   # commands persist 2s; a late dive is a conceded goal
    gk_claim_range: float = 3.0
    gk_outlet_lane: float = 2.0


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

    defenders = [me] + [m for m in obs["teammates"] if m["role"] != "GK"]
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

        if (
            (d_goal < near and lane > p.shoot_near_lane)
            or (d_goal < p.shoot_mid_dist and lane > p.shoot_mid_lane)
            or (d_goal < p.shoot_far_dist and lane > p.shoot_far_lane)
        ):
            return AgentCommand(
                type="SHOOT",
                target=(goal[0], goal[1] + _jitter(obs, 1.6)),
                rationale=f"{d_goal:.0f}m, lane {lane:.1f}m",
            )

        best, best_score = None, -1e9
        for m in mates:
            mp = m["position"]
            d = _dist(pos, mp)
            clear = _lane(pos, mp, opps)
            if d > p.pass_max_dist or clear < p.pass_min_lane:
                continue
            score = (
                (mp[0] - pos[0]) * p.pass_gain_w
                + clear * p.pass_lane_w
                - abs(d - p.pass_ideal_dist) * p.pass_dist_w
                + (20.0 - _dist(mp, goal)) * p.pass_goal_w
            )
            if score > best_score:
                best, best_score = m, score

        pressure = me.get("pressure", 0.0)
        if best is not None and (pressure > p.pass_when_pressed or best_score > p.pass_score_floor):
            return AgentCommand(
                type="PASS",
                target_player_id=best["id"],
                rationale=f"lane to #{best['number']}",
            )

        if pressure > p.clear_pressure and pos[0] < pitch["length"] * p.clear_own_third:
            return AgentCommand(type="CLEAR", rationale="pressed deep")

        if pos[0] > pitch["length"] * 0.62:
            toward = (pos[0] + p.carry_step, goal[1] * 0.55 + pos[1] * 0.45)
        else:
            toward = (pos[0] + p.carry_step, pos[1] + self._free_side(obs) * 3.0)
        return AgentCommand(
            type="DRIBBLE",
            target=(_cx(toward[0], pitch["length"]), _cy(toward[1], pitch["width"])),
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
        return AgentCommand(
            type="MOVE_TO",
            target=(_cx(x, pitch["length"]), _cy(y, pitch["width"])),
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
            if carrier is not None and _dist(me["position"], carrier["position"]) < p.tackle_range:
                return AgentCommand(type="TACKLE", target_player_id=carrier["id"], rationale="in range")
            return AgentCommand(type="PRESS_BALL", rationale="close the carrier")

        mark = plan["marks"].get(me["id"])
        if mark is not None:
            return AgentCommand(type="MARK", target_player_id=mark, rationale="pick up runner")

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
        if not others or mine <= min(others) + 0.3:
            return AgentCommand(type="INTERCEPT", rationale="closest to the drop")
        return self._support(obs)

    # -------------------------------------------------------------- goalkeeper
    def _goalkeeper(self, obs: dict) -> AgentCommand:
        p = self.p
        me = obs["you"]
        pitch = obs["pitch"]
        ball = obs["ball"]["position"]
        bv = obs["ball"]["velocity"]
        goal = pitch["your_goal"]["center"]
        pos = me["position"]

        if obs["possession"] == "you":
            mates = [m for m in obs["teammates"] if m["role"] != "GK"]
            safe = [m for m in mates
                    if _lane(pos, m["position"], obs["opponents"]) > p.gk_outlet_lane]
            if safe:
                target = max(safe, key=lambda m: m["position"][0])
                return AgentCommand(type="PASS", target_player_id=target["id"],
                                    rationale="restart from the back")
            return AgentCommand(type="CLEAR", rationale="no safe outlet")

        speed = math.hypot(bv[0], bv[1])
        if speed > p.gk_dive_speed and bv[0] < -2.0 and ball[0] < 16.0:
            t = max(0.05, (ball[0] - goal[0] - 0.4) / max(-bv[0], 0.1))
            aim_y = ball[1] + bv[1] * t
            if abs(aim_y - goal[1]) < pitch["your_goal"]["width"] and _dist(pos, ball) < p.gk_dive_range:
                return AgentCommand(type="GK_DIVE",
                                    target=(goal[0] + 0.6, _cy(aim_y, pitch["width"])),
                                    rationale="shot incoming")

        if _dist(pos, ball) < p.gk_claim_range and obs["possession"] != "opponent":
            return AgentCommand(type="INTERCEPT", rationale="claim it")

        dx, dy = ball[0] - goal[0], ball[1] - goal[1]
        n = math.hypot(dx, dy) or 1.0
        depth = p.gk_depth_base + min(p.gk_depth_max, _dist(ball, goal) * p.gk_depth_gain)
        return AgentCommand(
            type="MOVE_TO",
            target=(_cx(goal[0] + dx / n * depth, pitch["length"]),
                    _cy(goal[1] + dy / n * depth * 0.72, pitch["width"])),
            rationale="hold the angle",
        )
