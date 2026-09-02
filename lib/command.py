"""AgentCommand — the only contract with the AWS Agentic Football Cup platform.

Kept dependency-free (pydantic only) so it can be copied into a deployed
runtime without dragging the rest of this package along.

Why this file is shaped the way it is
-------------------------------------
The platform accepts exactly eleven command types and silently DROPS anything
else. No error, no rejection, no log line — a hallucinated ``DRIBBLE`` looks
from the outside exactly like a healthy tick, right up until you lose. So the
type list here is a closed ``Literal``, the per-command parameters are checked
at construction, and ``platform.validate`` re-checks against the live
observation before anything reaches the wire.

Two platform behaviours drive the rest of the design:

  - A decision that misses the 5-second budget does NOT idle the player. The
    player HOLDS THEIR LAST COMMAND. A bad command is therefore not a wasted
    tick, it is a wasted tick that keeps running. That is why the fallback at
    the bottom of this file is chosen for what it does when it repeats.
  - ``SET_STANCE``, ``CLEAR_OVERRIDE`` and ``RESET`` are sticky: they outlive
    the tick that issued them. They are listed in ``STICKY`` so callers can
    treat them as configuration rather than as a move.

Parameter names follow the wire format except where the wire name would
collide with something already on this model. The single rename is PASS's
``type`` -> ``pass_type``, because ``type`` is the command-type field.
``platform.to_wire`` puts the wire names back.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------- types

CommandType = Literal[
    "MOVE_TO",
    "FOLLOW_PLAYER",
    "SHOOT",
    "PASS",
    "GK_DISTRIBUTE",
    "PRESS_BALL",
    "MARK",
    "INTERCEPT",
    "SET_STANCE",
    "CLEAR_OVERRIDE",
    "RESET",
]

#: Every command the platform will accept. Anything else is dropped in silence.
COMMAND_TYPES: tuple[str, ...] = get_args(CommandType)

AimLocation = Literal["TL", "TR", "BL", "BR", "CENTER"]
PassType = Literal["GROUND", "AERIAL", "THROUGH"]
DistributionMethod = Literal["THROW", "KICK"]
Tightness = Literal["LOOSE", "TIGHT"]
TeamSide = Literal["HOME", "AWAY"]

#: Legal only for the goalkeeper (player index 0). The platform drops a
#: GK_DISTRIBUTE from an outfield player.
GK_ONLY: tuple[str, ...] = ("GK_DISTRIBUTE",)

#: These outlive the tick that issued them, so a caller must not treat them as
#: an ordinary per-tick move: issuing one every tick pins the team's shape, and
#: a stray RESET wipes overrides the rest of the squad is relying on.
STICKY: tuple[str, ...] = ("SET_STANCE", "CLEAR_OVERRIDE", "RESET")

#: What an outfield player may be told to do. `prompts.py` builds the command
#: menu from this, so a keeper sees eleven options and everyone else sees ten.
OUTFIELD: tuple[str, ...] = tuple(c for c in COMMAND_TYPES if c not in GK_ONLY)

# SET_STANCE takes an int, not a name; these are the three legal values.
STANCE_BALANCED = 0
STANCE_ATTACKING = 1
STANCE_DEFENSIVE = 2

# Filled in when a command omits them. Chosen so that a model that names the
# command and its target but forgets the trimming still gets a usable move
# through, instead of having the whole decision dropped over a missing float.
DEFAULT_POWER = 0.8              # firm but not maximum; keeps some placement
DEFAULT_INTENSITY = 0.7          # >0.5 sprints and >0.3 tackles, so this does both
DEFAULT_FOLLOW_DISTANCE = 3.0    # close enough to contest, far enough to react
DEFAULT_PASS_TYPE = "GROUND"     # the safest of the three
DEFAULT_DISTRIBUTION = "THROW"   # accurate; KICK is the long, low-percentage option
DEFAULT_TIGHTNESS = "TIGHT"

DOCS: dict[str, str] = {
    "MOVE_TO": "Run to a point. Take space, cover a passing lane, or get back into shape.",
    "FOLLOW_PLAYER": "Shadow one player at a fixed distance. Use to track a runner rather than hold a zone.",
    "SHOOT": "Strike at the opponent goal. Use when the lane to a corner is open.",
    "PASS": "Play the ball to a teammate. GROUND is safest; THROUGH plays the space behind them.",
    "GK_DISTRIBUTE": "Keeper only. Restart from the back once you hold it. THROW is accurate, KICK is long.",
    "PRESS_BALL": "Close down whoever has the ball. Above 0.5 you sprint, above 0.3 you attempt tackles.",
    "MARK": "Stay goal-side of one opponent. Use when a teammate is already pressing the ball.",
    "INTERCEPT": "Move onto the ball's path to cut it out. Use when it is loose or in flight.",
    "SET_STANCE": "Change the team's overall shape. Sticky: it lasts until something changes it.",
    "CLEAR_OVERRIDE": "Drop your standing order and go back to normal play. Sticky.",
    "RESET": "Clear every team override at once. Sticky: only to recover from a bad state.",
}


# ----------------------------------------------------------------- player ids

def player_index(value: object) -> int:
    """Parse any of the ids this codebase passes around into a wire index.

    The platform identifies a player by a bare integer 0..4 on the way out, but
    three other spellings reach this model on the way in: the platform's own
    ``"agentId_3"``, the team-qualified id ``platform.py`` puts in the
    normalised observation (``"home_3"``), and a plain numeric string from a
    model that decided to quote it.

    Deliberately strict — anything unrecognised raises rather than defaulting,
    because a silently wrong player index is a pass to the wrong person.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a player id
        raise ValueError(f"not a player id: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        tail = value.rsplit("_", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    raise ValueError(f"not a player id: {value!r}")


# --------------------------------------------------------------------- model

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Stance = Annotated[int, Field(ge=STANCE_BALANCED, le=STANCE_DEFENSIVE)]


class AgentCommand(BaseModel):
    """One decision for one player.

    Flat rather than a discriminated union of eleven models: the LLM fills this
    in through ``structured_output``, and one flat schema with a type field is
    markedly easier for a small model to emit correctly than a nested oneOf.
    The per-type parameter rules that a union would give for free are enforced
    by ``_check_parameters`` below instead.
    """

    type: CommandType

    # MOVE_TO. `target` is a point in the normalised, attack-relative frame
    # (see platform.Frame); to_wire converts it back to platform coordinates.
    target: tuple[float, float] | None = None
    sprint: bool = False

    # PASS / GK_DISTRIBUTE / MARK / FOLLOW_PLAYER
    target_player_id: int | None = None
    target_team: TeamSide | None = None   # FOLLOW_PLAYER only; the wire needs it
    distance: float | None = Field(default=None, ge=0.0)

    # SHOOT
    aim_location: AimLocation | None = None
    power: Unit | None = None

    # PASS
    pass_type: PassType | None = None     # goes onto the wire as "type"

    # GK_DISTRIBUTE
    method: DistributionMethod | None = None

    # PRESS_BALL
    intensity: Unit | None = None

    # MARK
    tightness: Tightness | None = None

    # INTERCEPT
    aggressive: bool = False

    # SET_STANCE
    stance: Stance | None = None

    duration: float = Field(default=0.0, ge=0.0)
    rationale: str = Field(default="", description="One short line, for the replay log.")

    model_config = {"extra": "ignore"}

    @field_validator("target_player_id", mode="before")
    @classmethod
    def _coerce_target(cls, v: object) -> object:
        """Accept "home_3" / "agentId_3" / "3" as well as 3.

        The wire wants an int, but the normalised observation uses team-qualified
        string ids, so both spellings turn up here. Normalising at the edge means
        exactly one place has to know they are the same thing.
        """
        return v if v is None else player_index(v)

    @model_validator(mode="after")
    def _check_parameters(self) -> AgentCommand:
        """Enforce, per command type, what the platform actually requires.

        Anything with no defensible default is required and raises; anything
        with an obvious one is filled in. The split matters because a raise here
        costs the whole decision (brain.py falls back to the policy), so it is
        reserved for things that genuinely cannot be guessed — which player, and
        which corner of the goal.
        """
        t = self.type

        if t == "MOVE_TO":
            if self.target is None:
                raise ValueError("MOVE_TO needs a target point")

        elif t == "FOLLOW_PLAYER":
            if self.target_player_id is None:
                raise ValueError("FOLLOW_PLAYER needs target_player_id")
            if self.target_team is None:
                # No default is possible: this model does not know which side
                # it is on, and guessing means shadowing a teammate.
                raise ValueError("FOLLOW_PLAYER needs target_team")
            if self.distance is None:
                self.distance = DEFAULT_FOLLOW_DISTANCE

        elif t == "SHOOT":
            if self.aim_location is None:
                raise ValueError("SHOOT needs an aim_location")
            if self.power is None:
                self.power = DEFAULT_POWER

        elif t == "PASS":
            if self.target_player_id is None:
                raise ValueError("PASS needs target_player_id")
            if self.pass_type is None:
                self.pass_type = DEFAULT_PASS_TYPE

        elif t == "GK_DISTRIBUTE":
            if self.target_player_id is None:
                raise ValueError("GK_DISTRIBUTE needs target_player_id")
            if self.method is None:
                self.method = DEFAULT_DISTRIBUTION

        elif t == "PRESS_BALL":
            if self.intensity is None:
                self.intensity = DEFAULT_INTENSITY

        elif t == "MARK":
            if self.target_player_id is None:
                raise ValueError("MARK needs target_player_id")
            if self.tightness is None:
                self.tightness = DEFAULT_TIGHTNESS

        elif t == "SET_STANCE":
            if self.stance is None:
                raise ValueError("SET_STANCE needs a stance (0, 1 or 2)")

        # INTERCEPT, CLEAR_OVERRIDE and RESET carry nothing that can be missing.
        return self

    @property
    def is_sticky(self) -> bool:
        """True when this command outlives the tick that issued it."""
        return self.type in STICKY

    @property
    def is_gk_only(self) -> bool:
        return self.type in GK_ONLY


# ------------------------------------------------------------- safe fallback

# The command to send when nothing better survived validation.
#
# It can no longer be IDLE: the platform has no IDLE command, and a decision
# that arrives late or invalid does not stand the player still — it REPEATS
# their previous command for another tick. So the fallback has to stay harmless
# on its second and third run, not just its first.
#
# INTERCEPT(aggressive=False) is that command:
#   - it takes no player id and no coordinates, making it the only command that
#     cannot itself be rejected for a bad parameter — which is what you want
#     from the branch that runs when everything else was rejected;
#   - it is legal for keeper and outfield alike, unlike GK_DISTRIBUTE, and it
#     does not commit a keeper to sprinting off his line the way PRESS_BALL
#     would;
#   - it is not sticky, so repeating it cannot poison later ticks the way a
#     stray SET_STANCE or RESET would;
#   - and on a five-a-side pitch, drifting onto the ball's path is rarely the
#     wrong idea, which is the whole definition of a safe default.
SAFE_DEFAULT = AgentCommand(type="INTERCEPT", aggressive=False, rationale="fallback")

# Deprecated alias. serve.py and runtime/main.py still do `from command import
# IDLE`; keeping the name lets them keep importing while they are updated.
# Remove it once they reference SAFE_DEFAULT — the name is now a lie about the
# type it holds.
IDLE = SAFE_DEFAULT
