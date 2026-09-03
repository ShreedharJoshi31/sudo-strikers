"""Squad — deadline-guarded LLM on top of a policy that never misses budget.

The ordering matters and is the whole point of this package:

    1. Compute the policy command first. Sub-millisecond, always valid.
    2. Ask the model, with a hard deadline well inside the platform budget.
    3. Use the model's answer only if it arrives in time and survives checks.

A late model answer costs nothing here, because a valid command already exists.
Without step 1 a late answer costs the whole tick.

WHAT A MISSED DECISION ACTUALLY COSTS
-------------------------------------
This was previously documented as "the platform substitutes IDLE" — that was
taken from a description of a different simulator and is wrong for the Cup.
The real platform makes the player HOLD THEIR LAST COMMAND. That is worse, not
better: a stale MOVE_TO keeps dragging a player toward a position that stopped
being useful seconds ago, and the tactical commands (SET_STANCE, CLEAR_OVERRIDE,
RESET) persist until explicitly cleared.

So step 1 matters more than it did, for a different reason. It is not insurance
against standing still; it is a guarantee that a *current* command always exists
to overwrite the stale one.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import analysis
import prompts
import wire
from command import SAFE_DEFAULT, AgentCommand
from memory import SquadMemory
from scouting import Scout
import gateway as gateway_mod
import guardrails
from policy import DEFAULT, Params, Policy


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


# The platform's decision timeout is 5 SECONDS, not the 1.0s this was built
# against — so the old 0.70 default was spending 14% of the available budget.
#
# We do not simply take 4.5s, for two reasons. Invocations arrive about every
# 2 seconds, so a decision that outruns the cadence starts overlapping the next
# one; and the workshop's own figures do not reconcile (a stated ~2s cadence
# against ~64 ticks in a 5-minute match, which implies ~4.7s). Until the true
# cadence is measured from tick/gameTime deltas in a practice match, sit inside
# the stated cadence and hold the rest as headroom.
#
# Raise this only once measurement justifies it. The policy absorbs every miss,
# so the failure mode of being too generous is silent staleness, not an error.
DEFAULT_DEADLINE = float(os.environ.get("AFC_LLM_DEADLINE", "1.80"))
DEFAULT_MODEL = os.environ.get("AFC_MODEL_ID", "amazon.nova-micro-v1:0")
DEFAULT_REGION = os.environ.get("AFC_REGION") or os.environ.get("AWS_DEFAULT_REGION")

#: Commands the model must never be allowed to play.
#:
#: Both are valid, both are accepted by the platform, and both give away the
#: thing we are here to do:
#:   CLEAR_OVERRIDE - hands this player back to the engine's own built-in AI,
#:                    so the agent stops playing for the rest of that hold.
#:   RESET          - wipes every override across the WHOLE TEAM, so one
#:                    player's bad tick benches the other four as well.
#:
#: Observed in production: a small model picks these when the situation looks
#: quiet, because "hand it back" reads as a safe, humble answer. It is the most
#: expensive answer on the list. Rejecting them costs nothing - `decide` has
#: already computed a valid policy command before the model was ever asked.
SURRENDER_COMMANDS = frozenset({"CLEAR_OVERRIDE", "RESET"})


#: The four roles a player can hold. Index 0 is always the keeper; the rest
#: come from the formation (see wire.DEFAULT_ROLES).
ROLES: tuple[str, ...] = ("GK", "DEFENDER", "MIDFIELDER", "FORWARD")


def _model_for_role(role: str) -> str:
    """Which model this position runs on.

    Positions do not carry equal load. The midfielder chooses between shooting,
    a through ball and holding shape on nearly every touch; the keeper spends
    most of the match holding an angle. Spending the same tokens and the same
    latency on both is a waste at one end and a handicap at the other.

    AFC_MODEL_<ROLE> wins, then AFC_MODEL_ID, then the built-in default. Roles
    are resolved independently, so setting one leaves the others alone.

        AFC_MODEL_MIDFIELDER=amazon.nova-pro-v1:0 \
        AFC_MODEL_ID=amazon.nova-micro-v1:0

    Raise a position only once /stats shows it has budget headroom to spare:
    llm_late climbing is the model being too slow for its deadline, and a late
    answer means the player keeps running the LAST command.
    """
    return os.environ.get(f"AFC_MODEL_{role.upper()}", "").strip() or DEFAULT_MODEL


#: Resolved once at import so a deployment's model map is visible in one place.
DEFAULT_MODELS: dict[str, str] = {r: _model_for_role(r) for r in ROLES}

# Both default on but are switchable, because neither has been measured against
# a real model yet. Turn one off and compare /stats and bench results.
USE_ANALYSIS = _flag("AFC_ANALYSIS")
USE_MEMORY = _flag("AFC_MEMORY")
USE_SCOUTING = _flag("AFC_SCOUTING")
#: Send COMPACT rosters instead of the full player records.
#:
#: Measured: the two rosters are 61% of the model's input (2,398 of 3,930
#: chars), and most of each record is dead weight to a model - `home_position`
#: is the opponent's formation slot, `team_code` is constant per list, `number`
#: duplicates `id`, and wire.py itself flags `speed_hint` as not trustworthy.
#: Worse, `analysis` has ALREADY reduced those rosters to the conclusions that
#: matter (pass options with p_success, shot verdict, marking assignment), so
#: the raw rows invite a small model to redo geometry it is bad at. Participants
#: report exactly that failure - one model answered with the literal text
#: "clamp(-5.2, -12, 12)" instead of a number.
#:
#: MEASURED, and it does NOT pay: on us.amazon.nova-2-lite the 30% token cut
#: moved p50 by 8 ms (977 -> 969, inside noise) and LOST a command type
#: (4 distinct -> 3). Latency here is model inference time, not prompt size.
#: So this defaults OFF. Set AFC_LEAN_INPUT=1 to re-test it on another model.
USE_LEAN_INPUT = _flag("AFC_LEAN_INPUT", "0")

#: Fields worth keeping on a compact roster row. `pressure` and `stamina` are
#: judgement inputs the analysis block does not repeat per player.
LEAN_PLAYER_FIELDS = ("id", "role", "position", "velocity", "stamina", "pressure")


def _lean(players: list[dict]) -> list[dict]:
    """A roster row trimmed to what a model can actually act on."""
    return [
        {k: p[k] for k in LEAN_PLAYER_FIELDS if k in p}
        for p in players
    ]

#: Send the full per-player dicts for the other nine players to the model.
#: OFF by default. Measured, they were 63% of the per-tick body and cut total
#: input from ~2360 to ~1656 tokens when removed - and `analysis` already
#: digests exactly those players into the answers the model is being asked for:
#: pass options with interception margins, shot quality, the defensive
#: assignment, and `valid_targets`, the ids it may legally name. Sending the
#: roster on top asks a small model to redo geometry that Python did exactly in
#: 0.3ms, in the one place where it is expensive.
#: Set AFC_RAW_ROSTER=1 to put it back and A/B it.
SEND_RAW_ROSTER = _flag("AFC_RAW_ROSTER", "0")


@dataclass
class PlayerSpec:
    player_id: str
    role: str
    number: int = 0
    prompt: str = ""


@dataclass
class Stats:
    llm_used: int = 0
    llm_late: int = 0
    llm_error: int = 0
    policy_used: int = 0
    latencies: list[float] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)

    def note_rejection(self, reason: str) -> None:
        """Count why a model answer was thrown away, so it can be read back."""
        key = (reason or "unknown").strip()[:80]
        self.rejections[key] = self.rejections.get(key, 0) + 1

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
        total = self.llm_used + self.policy_used
        return {
            "decisions": total,
            "llm_used": self.llm_used,
            "llm_late": self.llm_late,
            "llm_error": self.llm_error,
            "policy_used": self.policy_used,
            "llm_share": round(self.llm_used / total, 3) if total else 0.0,
            "p95_ms": round(p95 * 1000, 1),
            "rejections": dict(
                sorted(self.rejections.items(), key=lambda kv: -kv[1])[:8]
            ),
            "models": dict(sorted(self.models.items())),
        }


class Squad:
    def __init__(
        self,
        players: list[PlayerSpec] | None = None,
        *,
        tactics: str = "",
        role_prompts: dict[str, str] | None = None,
        params: Params = DEFAULT,
        model_id: str | None = None,
        models: dict[str, str] | None = None,
        region: str | None = None,
        deadline: float = DEFAULT_DEADLINE,
        use_llm: bool = False,
        seed: int = 0,
    ) -> None:
        self.policy = Policy(params, seed=seed)
        self.params = params
        self.memory = SquadMemory()
        # One Scout per process. Five runtimes each build their own from the
        # same tick stream and, because it is deterministic, arrive at the same
        # model without exchanging a single message - the same trick that lets
        # defensive_assignment() divide up marking with no orchestrator.
        self.scout = Scout()
        self.specs = {p.player_id: p for p in (players or [])}
        self.tactics = tactics
        self.role_prompts = role_prompts or {}
        self.deadline = deadline
        self.use_llm = use_llm
        self.model_id = model_id or DEFAULT_MODEL
        # Per-role map layered over the squad default, so a caller can override
        # one position without restating the other three.
        self.models = dict(DEFAULT_MODELS)
        if model_id:
            self.models = {r: model_id for r in ROLES}
        for role, mid in (models or {}).items():
            if mid:
                self.models[role.upper()] = mid
        self.region = region or DEFAULT_REGION
        self.stats = Stats()
        # Built even when no Gateway is configured; it is inert until loaded.
        self.gateway = gateway_mod.GatewayTools()
        self._agents: dict[str, object] = {}
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=5) if use_llm else None
        )

    # ------------------------------------------------------------------ LLM
    def _ensure_agent(self, player_id: str, obs: dict):
        """Build this player's Agent on first sight.

        Role and number come from the observation, so the server needs no
        second copy of the roster to drift out of sync with the platform.
        """
        if player_id in self._agents:
            return self._agents[player_id]

        from strands import Agent
        from strands.models.bedrock import BedrockModel

        me = obs["you"]
        role = me["role"]
        spec = self.specs.get(player_id)
        role_prompt = spec.prompt if spec else self.role_prompts.get(role, "")

        model_id = self.models.get(role) or self.model_id
        kwargs = {"model_id": model_id}
        if self.region:
            kwargs["region_name"] = self.region
        # Recorded so /stats can show which model actually served each position,
        # rather than which one the config intended.
        self.stats.models[player_id] = f"{role}:{model_id}"
        # Fetched once, on the first player built; [] when no Gateway is
        # configured or it cannot be reached, which is the pre-Gateway agent.
        tools = self.gateway.load()
        agent = Agent(
            model=BedrockModel(**kwargs),
            tools=tools,
            system_prompt=prompts.build(
                role=role,
                number=me.get("number", 0),
                tactics=self.tactics,
                role_prompt=role_prompt,
                length=obs["pitch"]["length"],
                width=obs["pitch"]["width"],
                goal_width=obs["pitch"]["your_goal"]["width"],
                budget=self.deadline,
            ),
            callback_handler=None,
        )
        self._agents[player_id] = agent
        return agent

    def _build_input(self, player_id: str, obs: dict, safe: AgentCommand) -> str:
        """Observation plus the precomputed features and this player's recent ticks.

        Both extras are additive to the raw observation, never a replacement:
        if the model ignores them it still has everything it had before.
        """
        payload = dict(obs)
        if not SEND_RAW_ROSTER:
            # The model keeps `you`, `ball`, `pitch` and `analysis`. What it
            # loses is nine raw player dicts it would have had to do geometry on
            # anyway. `analysis.valid_targets` still carries every id it may
            # name, so nothing it can legally emit becomes unreachable - it just
            # stops paying tokens to rediscover what was already computed.
            payload.pop("opponents", None)
            payload.pop("teammates", None)
        if USE_ANALYSIS:
            payload["analysis"] = analysis.analyse(
                obs, self.params, policy_suggests=safe.type
            )
        if USE_MEMORY:
            recent = self.memory.for_player(player_id).summary()
            if recent:
                payload["your_recent_ticks"] = recent
        if USE_SCOUTING:
            # What this opponent has actually done so far, in a few hundred
            # bytes of plain statements. Deliberately not raw numbers: the model
            # is better at acting on "their #3 presses hard and is tiring" than
            # on a float, and it costs fewer tokens to say.
            scouted = self.scout.summary()
            if scouted:
                payload["scouting"] = scouted
        return json.dumps(payload, separators=(",", ":"))

    def _think(self, player_id: str, obs: dict, model_input: str) -> AgentCommand:
        agent = self._agents[player_id]  # created by decide() before submit
        agent.messages = []  # stateless: context growth is latency growth
        with gateway_mod.session(self.gateway):
            return agent.structured_output(AgentCommand, model_input)

    # --------------------------------------------------------------- decide
    def decide(self, player_id: str, obs: dict) -> AgentCommand:
        t0 = time.monotonic()
        safe = self.policy.decide(player_id, obs)

        def finish(cmd: AgentCommand, source: str) -> AgentCommand:
            # Hard football rules, applied to model and policy alike. A rule
            # that only guarded the model would leave the policy free to do
            # the same thing, and the policy is what plays most ticks.
            try:
                cmd, why = guardrails.apply(cmd, obs, fallback=safe)
                if why:
                    self.stats.note_rejection(f"guardrail: {why}")
            except Exception as exc:  # noqa: BLE001 - never lose a tick to a rule
                self.stats.note_rejection(f"guardrail-error: {type(exc).__name__}")
            if source == "llm":
                self.stats.llm_used += 1
            else:
                self.stats.policy_used += 1
            self.stats.latencies.append(time.monotonic() - t0)
            if USE_MEMORY:
                self.memory.for_player(player_id).record(
                    obs.get("tick", 0), cmd.type, cmd.rationale, source
                )
            return cmd

        if not self.use_llm or self._pool is None:
            return finish(safe, "policy")

        try:
            self._ensure_agent(player_id, obs)
            model_input = self._build_input(player_id, obs, safe)
        except Exception:
            self.stats.llm_error += 1
            return finish(safe, "policy")

        future = self._pool.submit(self._think, player_id, obs, model_input)
        try:
            remaining = self.deadline - (time.monotonic() - t0)
            cmd = future.result(timeout=max(0.01, remaining))
        except TimeoutError:
            future.cancel()
            self.stats.llm_late += 1
            return finish(safe, "policy")
        except Exception:
            self.stats.llm_error += 1
            return finish(safe, "policy")

        # _validate is the safety net, so it must never be the thing that
        # fails. It reads fields off the observation, and an observation built
        # by something other than wire.to_observation can be missing one; a
        # raise here would escape decide() entirely instead of falling back.
        try:
            checked = self._validate(cmd, obs)
        except Exception as exc:  # noqa: BLE001
            self.stats.note_rejection(f"validate: {type(exc).__name__}: {exc}"[:80])
            checked = None
        if checked is None:
            self.stats.llm_error += 1
            return finish(safe, "policy")

        return finish(checked, "llm")

    # ------------------------------------------------------------ platform
    def handle(self, payload: dict, my_index: int | None = None) -> list[dict]:
        """One platform invocation, end to end: payload in, wire commands out.

        Lives here rather than in each entrypoint so the HTTP server and the
        AgentCore runtime cannot drift apart on the part that has to be exactly
        right - which team a player belongs to, and which way they are kicking.

        `my_index` pins which player this process controls. A per-position
        runtime passes its own index so a missing or wrong `myPlayers` cannot
        make the keeper play as a forward; the local server leaves it None and
        takes whatever the payload says, because it serves all five.

        Never raises. A thrown exception here would be a missed decision, and a
        missed decision does not idle the player, it leaves the LAST command
        running. Returning the safe default is strictly better than that: it is
        current, it is valid, and repeating it is harmless.
        """
        try:
            state = payload.get("gameState")
            if not isinstance(state, dict):
                raise ValueError("payload has no gameState")
            mine = payload.get("myPlayers") or []
            sent = int(mine[0]) if mine else None
            if my_index is None:
                my_index = sent if sent is not None else 0
            elif sent is not None and sent != my_index:
                # Worth surfacing: it means the platform's routing and this
                # deployment disagree about who this runtime is. The pin wins,
                # because a runtime prompted as a keeper should not start
                # issuing forward commands on someone else's behalf.
                self.stats.note_rejection(
                    f"routing: payload says player {sent}, this runtime is {my_index}")

            obs = wire.to_observation(payload, my_index=my_index)
            if USE_SCOUTING:
                # Before deciding: this tick's evidence should inform this
                # tick's choice. observe() is idempotent on the tick number, so
                # a replayed or duplicated invocation cannot skew the model.
                self.scout.observe(obs)
                self._attach_scouting(obs)
            cmd = self.decide(obs["you"]["id"], obs)

            ok, reason = wire.validate(cmd, obs)
            if not ok:
                # The policy produced something the platform would drop. That is
                # a bug in us, not in the model, so make it loud in the stats.
                self.stats.note_rejection(f"policy: {reason}")
                cmd = SAFE_DEFAULT
            return wire.to_wire(cmd, wire.frame_for(payload), my_index)
        except Exception as exc:  # noqa: BLE001 - a raise here costs the tick
            self.stats.note_rejection(f"handle: {type(exc).__name__}: {exc}"[:80])
            try:
                idx = my_index
                if idx is None:
                    idx = int((payload.get("myPlayers") or [0])[0])
                return wire.to_wire(SAFE_DEFAULT, wire.frame_for(payload), idx)
            except Exception:
                return [{"commandType": SAFE_DEFAULT.type, "playerId": 0,
                         "parameters": {"aggressive": False}, "duration": 0.0}]

    def _attach_scouting(self, obs: dict) -> None:
        """Replace the passing model's guessed constants with measured ones.

        `passing.py` has to assume a top speed and a reaction delay for every
        opponent, because the platform publishes neither. `scouting.py` measures
        both, per opponent, from this match's own ticks. This method is the join
        between them, and it is deliberately a dict key rather than an import:
        neither module knows the other exists, so either can be swapped or
        switched off without touching the other.

        Scout's top speed is already discounted for fatigue, so it substitutes
        for the guess rather than being scaled again on top of it.
        """
        for o in obs.get("opponents", ()):
            oid = o.get("id")
            if not oid:
                continue
            o["scout"] = {
                "top_speed": self.scout.effective_top_speed(oid),
                "reaction": self.scout.reaction_delay(oid),
            }

    def _validate(self, cmd: AgentCommand, obs: dict) -> AgentCommand | None:
        """Reject anything the platform would silently drop.

        Delegates to `wire.validate`, which checks against the live observation
        rather than against the command in isolation - a PASS to a real player
        who happens to be an opponent is well-formed and still wrong.

        The rejection REASON is kept, not just the verdict. The platform drops
        an invalid command without a word, so a model that quietly hallucinates
        looks exactly like a model that is playing well until the score says
        otherwise. `stats.rejections` is how that becomes visible.
        """
        if cmd is None:
            self.stats.note_rejection("no command returned")
            return None
        if cmd.type in SURRENDER_COMMANDS:
            # Well-formed, accepted by the platform, and self-harming. See
            # SURRENDER_COMMANDS. Treat exactly like a hallucination: keep the
            # policy command that was already computed, and make it visible.
            self.stats.note_rejection(f"surrender command {cmd.type}")
            return None
        ok, reason = wire.validate(cmd, obs)
        if not ok:
            self.stats.note_rejection(reason)
            return None
        return cmd

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
