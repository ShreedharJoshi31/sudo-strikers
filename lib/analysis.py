"""Derived features handed to the model alongside the raw observation.

These are the same four things the workshop exposes as remote Gateway tools —
pass options, shot evaluation, open space, defensive assignment — computed
locally as pure functions instead. Same numbers, no MCP round trip, and they
are already paid for because the policy computes them anyway.

The point is division of labour: geometry is something Python does exactly in
microseconds and a small model does badly under a one-second deadline. Give the
model the measurements and let it spend its budget on judgement.

Toggle with AFC_ANALYSIS=0 to measure whether it actually helps.
"""

from __future__ import annotations

from policy import (
    DEFAULT,
    Params,
    _dist,
    _lane,
    defensive_assignment,
)


def _round(v: float, n: int = 1) -> float:
    return round(float(v), n)


def pass_options(obs: dict, p: Params, limit: int = 3) -> list[dict]:
    """Teammates ranked the way the policy ranks them, with the inputs shown."""
    me = obs["you"]
    pos = me["position"]
    goal = obs["pitch"]["opponent_goal"]["center"]
    opps = obs["opponents"]

    scored = []
    for m in obs["teammates"]:
        if m["role"] == "GK":
            continue
        mp = m["position"]
        d = _dist(pos, mp)
        clear = _lane(pos, mp, opps)
        forward = mp[0] - pos[0]
        score = (
            forward * p.pass_gain_w
            + clear * p.pass_lane_w
            - abs(d - p.pass_ideal_dist) * p.pass_dist_w
            + (20.0 - _dist(mp, goal)) * p.pass_goal_w
        )
        scored.append({
            "id": m["id"],
            "number": m.get("number"),
            "role": m["role"],
            "distance": _round(d),
            "lane_clearance": _round(clear),
            "forward_gain": _round(forward),
            "blocked": clear < p.pass_min_lane or d > p.pass_max_dist,
            "score": _round(score, 2),
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


def shot(obs: dict, p: Params) -> dict:
    """Distance, lane and whether the policy's gates would fire."""
    me = obs["you"]
    pos = me["position"]
    goal = obs["pitch"]["opponent_goal"]["center"]
    opps = obs["opponents"]
    gk_ids = tuple(o["id"] for o in opps if o["role"] == "GK")

    d = _dist(pos, goal)
    lane = _lane(pos, goal, opps, ignore_ids=gk_ids)
    near = p.shoot_near_dist - (p.tired_shoot_penalty if me.get("stamina", 100) <= p.tired_stamina else 0.0)
    worth_it = (
        (d < near and lane > p.shoot_near_lane)
        or (d < p.shoot_mid_dist and lane > p.shoot_mid_lane)
        or (d < p.shoot_far_dist and lane > p.shoot_far_lane)
    )
    return {
        "distance_to_goal": _round(d),
        "lane_clearance": _round(lane),
        "worth_taking": worth_it,
    }


def open_space(obs: dict) -> dict:
    """Which flank is emptier, and how congested your own area is."""
    me = obs["you"]
    y = me["position"][1]
    width = obs["pitch"]["width"]
    above = sum(1 for o in obs["opponents"] if o["position"][1] > y)
    below = len(obs["opponents"]) - above
    near = sum(1 for o in obs["opponents"] if _dist(o["position"], me["position"]) < 6.0)
    return {
        "freer_flank": "high_y" if above < below else "low_y",
        "opponents_within_6m": near,
        "you_are_in_midfield_band": 0.3 * width < y < 0.7 * width,
    }


def marking(obs: dict) -> dict:
    """Your share of the defensive plan every teammate is computing too."""
    plan = defensive_assignment(obs)
    me_id = obs["you"]["id"]
    return {
        "presser": plan["presser"],
        "you_are_presser": plan["presser"] == me_id,
        "your_mark": plan["marks"].get(me_id),
        "all_marks": plan["marks"],
    }


def analyse(obs: dict, p: Params = DEFAULT, policy_suggests: str | None = None) -> dict:
    """Everything above, in one block, for the model's input."""
    out = {
        "shot": shot(obs, p),
        "pass_options": pass_options(obs, p),
        "space": open_space(obs),
        "pressure_on_you": _round(obs["you"].get("pressure", 0.0), 2),
    }
    if obs["possession"] == "opponent":
        out["defending"] = marking(obs)
    if policy_suggests:
        out["fallback_would_do"] = policy_suggests
    return out
