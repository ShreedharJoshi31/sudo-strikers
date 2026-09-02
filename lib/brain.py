"""Squad — deadline-guarded LLM on top of a policy that never misses budget.

The ordering matters and is the whole point of this package:

    1. Compute the policy command first. Sub-millisecond, always valid.
    2. Ask the model, with a hard deadline well inside the platform budget.
    3. Use the model's answer only if it arrives in time and survives checks.

A late model answer costs nothing here, because a valid command already exists.
Without step 1 a late answer costs the whole tick.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import analysis
import prompts
from command import AgentCommand
from memory import SquadMemory
from policy import DEFAULT, Params, Policy


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


# Leaves room for network + serialisation inside the platform's 1.0s budget.
DEFAULT_DEADLINE = float(os.environ.get("AFC_LLM_DEADLINE", "0.70"))
DEFAULT_MODEL = os.environ.get("AFC_MODEL_ID", "amazon.nova-micro-v1:0")

# Both default on but are switchable, because neither has been measured against
# a real model yet. Turn one off and compare /stats and bench results.
USE_ANALYSIS = _flag("AFC_ANALYSIS")
USE_MEMORY = _flag("AFC_MEMORY")


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
        region: str | None = None,
        deadline: float = DEFAULT_DEADLINE,
        use_llm: bool = False,
        seed: int = 0,
    ) -> None:
        self.policy = Policy(params, seed=seed)
        self.params = params
        self.memory = SquadMemory()
        self.specs = {p.player_id: p for p in (players or [])}
        self.tactics = tactics
        self.role_prompts = role_prompts or {}
        self.deadline = deadline
        self.use_llm = use_llm
        self.model_id = model_id or DEFAULT_MODEL
        self.region = region
        self.stats = Stats()
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

        kwargs = {"model_id": self.model_id}
        if self.region:
            kwargs["region_name"] = self.region
        agent = Agent(
            model=BedrockModel(**kwargs),
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
        if USE_ANALYSIS:
            payload["analysis"] = analysis.analyse(
                obs, self.params, policy_suggests=safe.type
            )
        if USE_MEMORY:
            recent = self.memory.for_player(player_id).summary()
            if recent:
                payload["your_recent_ticks"] = recent
        return json.dumps(payload, separators=(",", ":"))

    def _think(self, player_id: str, obs: dict, model_input: str) -> AgentCommand:
        agent = self._agents[player_id]  # created by decide() before submit
        agent.messages = []  # stateless: context growth is latency growth
        return agent.structured_output(AgentCommand, model_input)

    # --------------------------------------------------------------- decide
    def decide(self, player_id: str, obs: dict) -> AgentCommand:
        t0 = time.monotonic()
        safe = self.policy.decide(player_id, obs)

        def finish(cmd: AgentCommand, source: str) -> AgentCommand:
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

        checked = self._validate(cmd, obs)
        if checked is None:
            self.stats.llm_error += 1
            return finish(safe, "policy")

        return finish(checked, "llm")

    def _validate(self, cmd: AgentCommand, obs: dict) -> AgentCommand | None:
        """Reject what the engine would silently turn into IDLE."""
        if cmd is None:
            return None
        if cmd.type == "GK_DIVE" and obs["you"]["role"] != "GK":
            return None
        if cmd.type in ("PASS", "MARK", "TACKLE"):
            known = {m["id"] for m in obs["teammates"]} | {o["id"] for o in obs["opponents"]}
            if cmd.target_player_id not in known:
                return None
        if cmd.type in ("MOVE_TO", "DRIBBLE", "GK_DIVE") and cmd.target is None:
            return None
        return cmd

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
