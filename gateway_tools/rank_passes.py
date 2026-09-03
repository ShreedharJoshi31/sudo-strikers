"""Gateway tool: rank_passes — the FULL ranking under a chosen risk policy.

The prompt carries the top three under one fixed policy. This exists so the
model can ask the other question: "show me everything, ranked for safety" when
it is 1-0 up with a minute left, or "ranked for value" when it needs a goal.
Risk appetite is situational and the precomputed block cannot know it.
"""

from _common import body


def lambda_handler(event, context):
    import passing

    b = body(event)
    observation = b.get("observation") or {}
    if not observation:
        return {"error": "observation is required"}

    policy = b.get("policy", "BEST_VALUE")
    if policy not in ("SAFEST", "BEST_VALUE"):
        policy = "BEST_VALUE"

    try:
        ranked = passing.rank_passes(observation, policy=policy)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"could not rank: {type(exc).__name__}: {exc}"}

    limit = int(b.get("limit", 6))
    return {
        "policy": policy,
        "options": [
            {
                "receiver_id": o.receiver_id,
                "receiver_number": o.receiver_number,
                "pass_type": o.type,
                "p_success": round(o.p_success, 3),
                "margin_seconds": round(o.worst_margin, 2),
                "contested_by": o.worst_opponent_id,
                "forward_gain": round(o.forward_gain, 1),
                "distance": round(o.distance, 1),
            }
            for o in ranked[:limit]
        ],
    }
