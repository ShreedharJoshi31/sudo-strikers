"""AgentCommand — the only contract with the platform.

Kept dependency-free (pydantic only) so it can be copied into a deployed
runtime without dragging the rest of this package along.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CommandType = Literal[
    "MOVE_TO",
    "PASS",
    "SHOOT",
    "DRIBBLE",
    "PRESS_BALL",
    "MARK",
    "INTERCEPT",
    "TACKLE",
    "CLEAR",
    "IDLE",
    "GK_DIVE",
]

OUTFIELD: tuple[str, ...] = (
    "MOVE_TO", "PASS", "SHOOT", "DRIBBLE", "PRESS_BALL",
    "MARK", "INTERCEPT", "TACKLE", "CLEAR", "IDLE",
)

DOCS: dict[str, str] = {
    "MOVE_TO": "Run to `target` (x, y). Take space, cover a lane, or get back into shape.",
    "PASS": "Kick the ball to `target_player_id`. Requires possession.",
    "SHOOT": "Strike at the opponent goal. `target` aims at a point in the goal mouth. Requires possession.",
    "DRIBBLE": "Carry the ball toward `target`. Slower than running free, and tackleable.",
    "PRESS_BALL": "Chase the ball carrier at speed to force a mistake.",
    "MARK": "Shadow `target_player_id`, staying between them and your own goal.",
    "INTERCEPT": "Move to where the loose ball will be, to cut it out.",
    "TACKLE": "Attempt to dispossess `target_player_id`. A miss leaves you off balance.",
    "CLEAR": "Hammer the ball away from your own goal. Requires possession.",
    "IDLE": "Hold position and recover stamina.",
    "GK_DIVE": "Goalkeeper only. Dive toward `target` to extend your reach for a save.",
}


class AgentCommand(BaseModel):
    type: CommandType
    target_player_id: str | None = None
    target: tuple[float, float] | None = None
    rationale: str = Field(default="", description="One short line, for the replay log.")

    model_config = {"extra": "ignore"}


IDLE = AgentCommand(type="IDLE", rationale="fallback")
