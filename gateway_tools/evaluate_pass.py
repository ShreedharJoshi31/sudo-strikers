"""Gateway tool: evaluate_pass — score ONE specific pass the model is weighing.

Why this is worth a round trip
------------------------------
The prompt already carries the top three passes. This answers the question that
block cannot: "the option I am actually considering — a THROUGH ball to #9 —
does it arrive?" The model names the receiver and the type; it gets back a
probability, the margin in seconds, and which opponent is the problem.

Deployed as a Lambda behind AgentCore Gateway. Bundles lib/ so it scores passes
with the same interception model the agent uses locally — one implementation,
so the tool can never disagree with the prompt.
"""

from _common import body


def lambda_handler(event, context):
    import passing

    b = body(event)
    observation = b.get("observation") or {}
    receiver_id = b.get("receiver_id")
    pass_type = b.get("pass_type")

    if not observation or receiver_id is None:
        return {"error": "observation and receiver_id are required"}

    try:
        ranked = passing.rank_passes(observation, policy=b.get("policy", "BEST_VALUE"))
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"could not evaluate: {type(exc).__name__}: {exc}"}

    matches = [o for o in ranked if str(o.receiver_id) == str(receiver_id)]
    if pass_type:
        typed = [o for o in matches if o.type == pass_type]
        matches = typed or matches

    if not matches:
        # A receiver with no option at all is a real answer, not an error: it
        # means every pass type to them was rejected outright.
        return {
            "receiver_id": receiver_id,
            "available": False,
            "reason": "no viable pass to that player from here",
        }

    best = matches[0]
    return {
        "receiver_id": best.receiver_id,
        "available": True,
        "pass_type": best.type,
        "p_success": round(best.p_success, 3),
        "margin_seconds": round(best.worst_margin, 2),
        "contested_by": best.worst_opponent_id,
        "distance": round(best.distance, 1),
        "forward_gain": round(best.forward_gain, 1),
        "alternatives": [
            {"pass_type": o.type, "p_success": round(o.p_success, 3)}
            for o in matches[1:4]
        ],
    }
