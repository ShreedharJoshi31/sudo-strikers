"""Gateway tool: evaluate_position — what the pitch looks like FROM somewhere else.

Answers "should I move there first?" The precomputed block only ever describes
where the player is standing now. This re-runs the shot and space maths from a
hypothetical point, so the model can compare moving against acting.
"""

from _common import body


def lambda_handler(event, context):
    import analysis
    from policy import DEFAULT

    b = body(event)
    observation = b.get("observation") or {}
    x, y = b.get("x"), b.get("y")
    if not observation or x is None or y is None:
        return {"error": "observation, x and y are required"}

    pitch = observation.get("pitch", {})
    length = float(pitch.get("length", 110.0))
    width = float(pitch.get("width", 70.0))
    if not (0 <= float(x) <= length and 0 <= float(y) <= width):
        return {"error": f"({x}, {y}) is off a {length:.0f}x{width:.0f} pitch"}

    # Copy rather than mutate: the caller's observation must not move because a
    # hypothetical was evaluated.
    hypothetical = dict(observation)
    you = dict(observation.get("you", {}))
    here = you.get("position", [0.0, 0.0])
    you["position"] = [float(x), float(y)]
    hypothetical["you"] = you

    try:
        moved = analysis.analyse(hypothetical, DEFAULT)
        current = analysis.analyse(observation, DEFAULT)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"could not evaluate: {type(exc).__name__}: {exc}"}

    return {
        "from": [round(float(here[0]), 1), round(float(here[1]), 1)],
        "to": [round(float(x), 1), round(float(y), 1)],
        "shot_here": current["shot"],
        "shot_there": moved["shot"],
        "space_there": moved["space"],
        "better_for_shooting": (
            moved["shot"]["worth_taking"] and not current["shot"]["worth_taking"]
        ),
    }
