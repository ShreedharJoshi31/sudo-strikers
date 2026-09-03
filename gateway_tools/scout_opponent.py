"""Gateway tool: scout_opponent — what a specific opponent has done this match.

The prompt carries a short team-level scouting summary. This lets the model ask
about ONE player before committing to a duel with them: do they press, do they
tackle, are they tiring. That is a per-opponent question and there is no room to
push every opponent's profile into every tick.

Scouting is accumulated per match, so this tool is only useful once some ticks
have been observed. It says so rather than inventing a profile from nothing.
"""

from _common import body


def lambda_handler(event, context):
    b = body(event)
    profiles = b.get("profiles") or {}
    opponent_id = b.get("opponent_id")

    if opponent_id is None:
        return {"error": "opponent_id is required"}
    if not profiles:
        return {
            "opponent_id": opponent_id,
            "known": False,
            "reason": "no scouting data yet this match",
        }

    profile = profiles.get(str(opponent_id)) or profiles.get(opponent_id)
    if not profile:
        return {"opponent_id": opponent_id, "known": False,
                "reason": "that player has not been observed yet"}
    return {"opponent_id": opponent_id, "known": True, "profile": profile}
