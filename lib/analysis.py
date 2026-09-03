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
    """Receivers ranked by whether the ball would actually arrive.

    Delegates to `passing.rank_passes`, which compares each opponent's time to
    reach the lane against the ball's time to cross it, rather than measuring a
    static gap. That is the difference between "someone is standing near this
    line" and "someone gets to this line first", and only the second one loses
    possession.

    The fields handed to the model are chosen to be actionable: a probability
    it can threshold, the seconds of margin, and WHO is the problem — so it can
    reason about picking a different ball rather than just seeing a low number.
    """
    try:
        import passing
    except Exception:
        return []

    squeezed = obs["you"].get("pressure", 0.0) > p.pass_when_pressed
    ranked = passing.rank_passes(
        obs, policy="SAFEST" if squeezed else "BEST_VALUE"
    )
    out = []
    for o in ranked[:limit]:
        out.append({
            "id": o.receiver_id,
            "number": o.receiver_number,
            "type": o.type,
            "p_success": _round(o.p_success, 2),
            "margin_s": _round(o.worst_margin, 2),
            "contested_by": o.worst_opponent_id,
            "forward_gain": _round(o.forward_gain),
            "distance": _round(o.distance),
            "blocked": o.p_success < p.pass_min_success,
        })
    return out


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


def valid_targets(obs: dict) -> dict:
    """The ids a target_player_id is actually allowed to name.

    Measured against Nova Micro: told only that FOLLOW_PLAYER "needs
    target_player_id", it answered `3` and once `825` — a bare index and a
    hallucination — and 5 of 8 decisions were rejected. The ids in play are
    `home_3` / `away_2` strings, and nothing in the prompt said so.

    Listing them costs a few dozen tokens and removes the guess entirely.
    """
    return {
        "teammates": [m["id"] for m in obs.get("teammates", ())],
        "opponents": [o["id"] for o in obs.get("opponents", ())],
    }


def analyse(obs: dict, p: Params = DEFAULT, policy_suggests: str | None = None) -> dict:
    """Everything above, in one block, for the model's input."""
    out = {
        "valid_targets": valid_targets(obs),
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
