"""Local tests. No AWS, no LLM, no server.

Checks the policy answers every game situation with a valid, in-budget command.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import brain as brain_mod
from brain import Squad
from command import COMMAND_TYPES, GK_ONLY, AgentCommand
from policy import DEFAULT, Policy

PITCH = {
    "length": 110.0,
    "width": 70.0,
    "you_attack_toward": "+x",
    "opponent_goal": {"center": [110.0, 35.0], "width": 10.0},
    "your_goal": {"center": [0.0, 35.0], "width": 10.0},
}


def _player(pid, role, x, y, number=1, has_ball=False):
    return {
        "id": pid, "number": number, "name": pid, "role": role,
        "position": [x, y], "velocity": [0.0, 0.0], "stamina": 90,
        "has_ball": has_ball, "distance_to_ball": 5.0,
        "home_position": [x, y], "last_command": None,
    }


def obs(possession, *, me_role="FORWARD", me_xy=(82.5, 35.0), ball_xy=(82.5, 35.0),
        ball_vel=(0.0, 0.0), owner="home_3"):
    me = _player("home_3", me_role, *me_xy, number=9, has_ball=(possession == "you"))
    me["pressure"] = 0.3
    me["distance_to_opponent_goal"] = PITCH["length"] - me_xy[0]
    return {
        "match_id": "t", "tick": 1,
        "clock": {"elapsed": 10.0, "remaining": 110.0},
        "score": {"you": 0, "opponent": 0},
        "pitch": PITCH,
        "possession": possession,
        "ball": {"position": list(ball_xy), "velocity": list(ball_vel),
                 "owner_id": owner, "loose": possession == "loose"},
        "you": me,
        "teammates": [
            _player("home_0", "GK", 7.2, 35.0, 1),
            _player("home_1", "DEFENDER", 28.9, 22.4, 4),
            _player("home_2", "DEFENDER", 28.9, 47.6, 5),
            _player("home_4", "FORWARD", 68.8, 47.6, 11),
        ],
        "opponents": [
            _player("away_0", "GK", 102.9, 35.0, 1),
            _player("away_1", "DEFENDER", 81.1, 22.4, 4),
            _player("away_2", "DEFENDER", 81.1, 47.6, 5),
            _player("away_3", "FORWARD", 41.3, 22.4, 9),
            _player("away_4", "FORWARD", 41.3, 47.6, 11),
        ],
        "coach_message": "", "recent_events": [],
    }


VALID = set(COMMAND_TYPES)


def test_all_situations():
    print("=== every situation returns a valid command ===")
    pol = Policy(DEFAULT)
    cases = []
    for possession in ("you", "teammate", "opponent", "loose"):
        for role in ("GK", "DEFENDER", "MIDFIELDER", "FORWARD"):
            cases.append((possession, role))
    for possession, role in cases:
        o = obs(possession, me_role=role)
        cmd = pol.decide("home_3", o)
        assert cmd.type in VALID, f"invalid type {cmd.type}"
        if cmd.type in GK_ONLY:
            assert role == "GK", f"{cmd.type} from an outfield player"
        if cmd.type == "MOVE_TO":
            assert cmd.target is not None, f"{cmd.type} without target"
            x, y = cmd.target
            assert 0 <= x <= PITCH["length"] and 0 <= y <= PITCH["width"], "target off pitch"
        if cmd.type in ("PASS", "MARK", "GK_DISTRIBUTE", "FOLLOW_PLAYER"):
            assert cmd.target_player_id is not None, f"{cmd.type} without target_player_id"
        print(f"  [ok] {possession:9s} {role:11s} -> {cmd.type:11s} {cmd.rationale}")
    print(f"  {len(cases)} situations, all valid\n")


def test_gk_holds_the_angle():
    """The platform has no GK_DIVE, so positioning is all the keeper has.

    This replaces the old dive test. Losing the dive costs nothing measurable:
    sweeping its range over 6-18 on the old engine changed no result, because a
    shot arrives long before the next decision point.
    """
    print("=== GK holds the angle rather than diving ===")
    o = obs("opponent", me_role="GK", me_xy=(7.2, 35.0),
            ball_xy=(33.0, 35.0), ball_vel=(-18.0, 0.5), owner="away_3")
    cmd = Policy(DEFAULT).decide("home_0", o)
    assert cmd.type == "MOVE_TO", f"expected MOVE_TO, got {cmd.type}"
    x, _ = cmd.target
    assert 0 <= x < 20.0, f"keeper wandered off its line to x={x:.1f}"
    print(f"  [ok] {cmd.type} -> {cmd.target} ({cmd.rationale})\n")


def test_gk_distributes_not_passes():
    """A PASS from the keeper is dropped by the platform; GK_DISTRIBUTE is not."""
    print("=== GK restarts with GK_DISTRIBUTE ===")
    o = obs("you", me_role="GK", me_xy=(7.2, 35.0), ball_xy=(7.2, 35.0), owner="home_0")
    cmd = Policy(DEFAULT).decide("home_0", o)
    assert cmd.type == "GK_DISTRIBUTE", f"expected GK_DISTRIBUTE, got {cmd.type}"
    assert cmd.method in ("THROW", "KICK")
    print(f"  [ok] {cmd.type} {cmd.method} -> #{cmd.target_player_id} ({cmd.rationale})\n")


def test_shoots_when_clear():
    print("=== shoots from a clear position ===")
    o = obs("you", me_xy=(90.0, 35.0))
    o["opponents"] = [_player("away_0", "GK", 37.4, 12.5, 1)]
    cmd = Policy(DEFAULT).decide("home_3", o)
    assert cmd.type == "SHOOT", f"expected SHOOT, got {cmd.type}"
    # The platform takes one of five enum corners, not a coordinate.
    assert cmd.aim_location in ("TL", "TR", "BL", "BR", "CENTER")
    assert 0.0 <= cmd.power <= 1.0
    print(f"  [ok] {cmd.type} aim={cmd.aim_location} power={cmd.power} ({cmd.rationale})\n")


def test_budget():
    print("=== policy latency (platform timeout is 5000ms) ===")
    pol = Policy(DEFAULT)
    o = obs("opponent")
    pol.decide("home_3", o)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        pol.decide("home_3", o)
    per = (time.perf_counter() - t0) / n
    print(f"  {per * 1000:.4f} ms per decision  ({1.0 / per:,.0f}/s)")
    assert per < 0.005, f"policy too slow: {per * 1000:.2f}ms"
    print("  [ok] roughly five orders of magnitude inside budget\n")


def test_squad_without_llm():
    print("=== squad falls through to policy with no LLM ===")
    sq = Squad(use_llm=False)
    cmd = sq.decide("home_3", obs("you"))
    assert cmd.type in VALID
    s = sq.stats.summary()
    assert s["policy_used"] == 1 and s["llm_used"] == 0
    print(f"  [ok] {cmd.type}, stats {s}\n")




# ---------------------------------------------------------------------------
# Analysis, memory, and the LLM path — exercised with a fake model, no AWS.
# ---------------------------------------------------------------------------

import analysis as A             # noqa: E402
from memory import PlayerMemory  # noqa: E402
from policy import DEFAULT as P  # noqa: E402


class FakeAgent:
    """Stands in for a strands Agent. Records what it was sent."""

    def __init__(self, reply=None, delay=0.0, raises=False):
        self.messages = []
        self.seen = []
        self._reply = reply or AgentCommand(
            type="SHOOT", aim_location="TR", power=0.8, rationale="fake"
        )
        self._delay = delay
        self._raises = raises

    def structured_output(self, cls, text):
        self.seen.append(text)
        if self._delay:
            time.sleep(self._delay)
        if self._raises:
            raise RuntimeError("model exploded")
        return self._reply


def _squad_with(agent, **kw):
    sq = Squad(use_llm=True, **kw)
    sq._agents["home_3"] = agent      # pre-seeded, so _ensure_agent never imports strands
    return sq


def test_analysis():
    print("=== analysis block ===")
    a = A.analyse(obs("you"), P, policy_suggests="SHOOT")
    assert "shot" in a and "pass_options" in a and "space" in a
    assert isinstance(a["shot"]["worth_taking"], bool)
    assert a["fallback_would_do"] == "SHOOT"
    print(f"  shot      : {a['shot']}")
    print(f"  best pass : {a['pass_options'][0]}")
    print(f"  space     : {a['space']}")
    d = A.analyse(obs("opponent"), P)
    assert "defending" in d, "defensive plan missing when they have the ball"
    print(f"  defending : {d['defending']}")
    assert "defending" not in a, "defensive plan leaked into an on-ball tick"
    print()


def test_analysis_is_fast():
    print("=== analysis latency ===")
    o = obs("opponent")
    A.analyse(o, P)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        A.analyse(o, P)
    per = (time.perf_counter() - t0) / n
    print(f"  {per * 1000:.4f} ms per call")
    assert per < 0.005, f"analysis too slow: {per * 1000:.2f}ms"
    print("  [ok] still far inside budget\n")


def test_models_per_role():
    """Each position resolves its own model, independently of the others.

    The point of per-role models is that positions do not carry equal load: a
    midfielder choosing between a shot, a through ball and holding shape earns
    a bigger model than a keeper holding an angle. This checks the resolution
    order - AFC_MODEL_<ROLE>, then AFC_MODEL_ID, then the built-in default -
    without needing AWS or an actual model.
    """
    print("=== per-role model selection ===")
    import importlib, os

    keep = {k: os.environ.get(k) for k in
            ("AFC_MODEL_ID", "AFC_MODEL_MIDFIELDER", "AFC_MODEL_GK")}
    try:
        os.environ["AFC_MODEL_ID"] = "base-model"
        os.environ["AFC_MODEL_MIDFIELDER"] = "big-model"
        os.environ.pop("AFC_MODEL_GK", None)
        b = importlib.reload(brain_mod)

        assert b.DEFAULT_MODELS["MIDFIELDER"] == "big-model", b.DEFAULT_MODELS
        assert b.DEFAULT_MODELS["GK"] == "base-model", "GK should fall back"
        assert b.DEFAULT_MODELS["FORWARD"] == "base-model"
        print(f"  [ok] env map {b.DEFAULT_MODELS}")

        sq = b.Squad(use_llm=False, models={"gk": "keeper-model"})
        assert sq.models["GK"] == "keeper-model", "explicit override ignored"
        assert sq.models["MIDFIELDER"] == "big-model", "override leaked across roles"
        print(f"  [ok] explicit override is per-role, others untouched")

        wide = b.Squad(use_llm=False, model_id="one-model")
        assert set(wide.models.values()) == {"one-model"}, wide.models
        print(f"  [ok] squad-wide model_id still overrides every role\n")
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(brain_mod)
        # Reloading rebuilds every class in the module, so the name bound by
        # `from brain import Squad` at import time now points at a dead class.
        # Rebind it, or a later test patching brain.Squad patches something no
        # live object is an instance of — which fails as a confusing assertion
        # about the feature under test rather than about module identity.
        globals()["Squad"] = brain_mod.Squad


def test_memory():
    print("=== memory ===")
    m = PlayerMemory(depth=4)
    assert m.summary() is None
    for t in range(4):
        m.record(t, "MOVE_TO", "drifting", "policy")
    assert m.repeated("MOVE_TO") == 4
    assert m.summary()["repeating"] is True, "should flag a stuck player"
    m.record(4, "SHOOT", "chance", "llm")
    assert m.summary()["repeating"] is False
    print(f"  [ok] {m.summary()}\n")


def test_llm_used_when_fast():
    print("=== LLM answers in time ===")
    fake = FakeAgent()
    sq = _squad_with(fake)
    cmd = sq.decide("home_3", obs("you"))
    assert cmd.rationale == "fake", "should have used the model's command"
    assert sq.stats.summary()["llm_used"] == 1
    sent = json.loads(fake.seen[0])
    assert "analysis" in sent, "analysis block never reached the model"
    print(f"  [ok] used LLM; extra keys sent: {sorted(k for k in sent if k in ('analysis', 'your_recent_ticks'))}\n")


def test_llm_late_falls_back():
    print("=== LLM misses the deadline ===")
    sq = _squad_with(FakeAgent(delay=0.4), deadline=0.05)
    cmd = sq.decide("home_3", obs("you"))
    assert cmd.rationale != "fake", "a late answer must not be used"
    s = sq.stats.summary()
    assert s["llm_late"] == 1 and s["policy_used"] == 1
    print(f"  [ok] fell back to policy ({cmd.type}), stats {s}\n")


def test_llm_error_falls_back():
    print("=== LLM raises ===")
    sq = _squad_with(FakeAgent(raises=True))
    cmd = sq.decide("home_3", obs("you"))
    assert cmd.type in VALID
    assert sq.stats.summary()["llm_error"] == 1
    print(f"  [ok] fell back to policy ({cmd.type})\n")


def test_llm_invalid_rejected():
    print("=== LLM returns something the engine would drop ===")
    # Well-formed but not a real teammate: exactly what a hallucinating
    # model emits, and what the platform drops without a word.
    bad = AgentCommand(type="PASS", target_player_id=9, rationale="fake")
    sq = _squad_with(FakeAgent(reply=bad))
    cmd = sq.decide("home_3", obs("you"))
    assert cmd.rationale != "fake", "PASS to a nonexistent player must be rejected"
    assert sq.stats.summary()["llm_error"] == 1
    print(f"  [ok] rejected, played {cmd.type} instead\n")


def test_memory_reaches_model():
    print("=== memory reaches the model on later ticks ===")
    fake = FakeAgent()
    sq = _squad_with(fake)
    for _ in range(3):
        sq.decide("home_3", obs("you"))
    sent = json.loads(fake.seen[-1])
    assert "your_recent_ticks" in sent, "memory never reached the model"
    print(f"  [ok] {sent['your_recent_ticks']}\n")




# ---------------------------------------------------------------------------
# Gateway: tool wiring and — mostly — that every failure degrades safely.
# ---------------------------------------------------------------------------

import gateway as GW                      # noqa: E402


def test_gateway_off_by_default():
    print("=== no AFC_GATEWAY_URL: everything inert ===")
    g = GW.GatewayTools(url="")
    assert g.load() == [], "unconfigured gateway must yield no tools"
    st = g.status()
    assert st["configured"] is False and st["connected"] is False
    with GW.session(g):                    # must not raise
        pass
    print(f"  [ok] {st}\n")


def test_gateway_unreachable_degrades():
    print("=== configured but unreachable: no tools, no exception ===")
    g = GW.GatewayTools(url="https://127.0.0.1:9/mcp")   # port 9 discards
    tools = g.load()
    assert tools == [], "a dead gateway must not produce tools"
    assert g.status()["configured"] is True
    assert g.error is not None, "the failure should be recorded, not swallowed silently"
    with GW.session(g):                    # must still be usable
        pass
    print(f"  [ok] error recorded: {g.error[:60]}\n")


def test_gateway_loads_once():
    print("=== tools fetched once, not per tick ===")
    calls = {"n": 0}

    class OneShot(GW.GatewayTools):
        def _transport(self):
            calls["n"] += 1
            raise RuntimeError("boom")

    g = OneShot(url="https://example.invalid/mcp")
    for _ in range(5):
        g.load()
    assert calls["n"] <= 1, f"connected {calls['n']} times; must be at most once"
    print(f"  [ok] {calls['n']} connection attempt across 5 loads\n")


def test_squad_runs_without_gateway():
    print("=== squad still plays with no gateway ===")
    sq = Squad(use_llm=False)
    cmd = sq.decide("H4", obs("you"))
    assert cmd.type in VALID
    assert sq.gateway.status()["connected"] is False
    print(f"  [ok] {cmd.type} with gateway inert\n")


def test_tools_reach_the_agent():
    print("=== fetched tools are attached to the Agent ===")
    sentinel = [object(), object()]

    class Loaded(GW.GatewayTools):
        def load(self):
            self._tools = sentinel
            return sentinel

    captured = {}

    class FakeStrandsAgent:
        def __init__(self, **kw):
            captured.update(kw)
            self.messages = []

        def structured_output(self, cls, text):
            return AgentCommand(type="SHOOT", parameters={"aim_location": "TL", "power": 0.8})

    sq = Squad(use_llm=True)
    sq.gateway = Loaded(url="https://example.test/mcp")
    # Patch via type(sq), not brain.Squad: another test reloads the module, and
    # after a reload those are different class objects.
    cls = type(sq)
    real = cls._ensure_agent

    def patched(self, player_id, o):
        if player_id in self._agents:
            return self._agents[player_id]
        agent = FakeStrandsAgent(tools=self.gateway.load())
        self._agents[player_id] = agent
        return agent

    cls._ensure_agent = patched
    try:
        sq.decide("H4", obs("you"))
        assert captured.get("tools") is sentinel, "gateway tools never reached the Agent"
    finally:
        cls._ensure_agent = real
    print(f"  [ok] {len(sentinel)} tools passed through to the Agent\n")


def test_tool_handlers():
    print("=== lambda handlers return sane payloads ===")
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_tools"))
    import evaluate_position, scout_opponent

    o = obs("you")
    r = evaluate_position.lambda_handler(
        {"observation": o, "x": o["pitch"]["length"] - 5, "y": o["pitch"]["width"] / 2}, None)
    assert "error" not in r, r
    assert "shot_here" in r and "shot_there" in r
    print(f"  [ok] evaluate_position from={r['from']} to={r['to']} "
          f"better={r['better_for_shooting']}")

    off = evaluate_position.lambda_handler({"observation": o, "x": 9999, "y": 0}, None)
    assert "error" in off, "an off-pitch point must be rejected, not evaluated"
    print(f"  [ok] off-pitch rejected: {off['error']}")

    s = scout_opponent.lambda_handler({"opponent_id": "A3"}, None)
    assert s["known"] is False, "no profiles must report unknown, not invent one"
    s2 = scout_opponent.lambda_handler(
        {"opponent_id": "A3", "profiles": {"A3": "presses hard, tiring"}}, None)
    assert s2["known"] is True and "presses" in s2["profile"]
    print(f"  [ok] scout_opponent unknown-then-known\n")




# ---------------------------------------------------------------------------
# The five per-position runtimes: each must control its OWN player.
# ---------------------------------------------------------------------------

AGENT_INDEX = {"ai-gk": 0, "ai-def1": 1, "ai-def2": 2, "ai-mid": 3, "ai-fwd": 4}
AGENT_ROLE = {"ai-gk": "GK", "ai-def1": "DEFENDER", "ai-def2": "DEFENDER",
              "ai-mid": "MIDFIELDER", "ai-fwd": "FORWARD"}


def _platform_payload(my_index: int, team: str = "home") -> dict:
    """A payload shaped like the real thing, including the id collision.

    Both teams number their players agentId_0..agentId_4 and the possession
    field carries no team, which is the trap wire.py exists to handle.
    """
    players = []
    for code, base_x in (("home", -30.0), ("away", 30.0)):
        for i in range(5):
            players.append({
                "agentId": f"agentId_{i}", "teamCode": code,
                "position": {"x": base_x + i * 4.0, "y": -12.0 + i * 6.0},
                "velocity": {"x": 0.0, "y": 0.0}, "speed": 0.0,
                "stamina": 85.0, "isSprinting": False,
            })
    return {
        "gameState": {
            "gameTime": 42.0, "playMode": 0,
            "score": {"home": 1, "away": 1},
            "ball": {"position": {"x": -6.0, "y": 2.0},
                     "velocity": {"x": 1.0, "y": 0.0},
                     "possessionAgentId": "agentId_3"},
            "players": players,
        },
        "teamId": 0 if team == "home" else 1,
        "myPlayers": [my_index],
    }


def _load_agent(name: str):
    import importlib.util
    os.environ["AFC_USE_LLM"] = "0"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "agents", name, "src", "main.py")
    spec = importlib.util.spec_from_file_location(f"agent_{name}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_agents_pin_their_player():
    print("=== five runtimes, each pinned to its own player ===")
    for name, idx in AGENT_INDEX.items():
        m = _load_agent(name)
        assert m.MY_PLAYER_INDEX == idx, f"{name} controls {m.MY_PLAYER_INDEX}, expected {idx}"
        assert m.ROLE == AGENT_ROLE[name], f"{name} is {m.ROLE}, expected {AGENT_ROLE[name]}"
        assert m.ROLE_PROMPT.strip(), f"{name} has an empty role brief"
        cmds = m.squad.handle(_platform_payload(idx), my_index=m.MY_PLAYER_INDEX)
        assert isinstance(cmds, list) and cmds, f"{name} returned nothing"
        for c in cmds:
            assert c["playerId"] == idx, f"{name} emitted for player {c['playerId']}"
            assert c["commandType"] in COMMAND_TYPES, f"{name} bad type {c['commandType']}"
            assert "parameters" in c, f"{name} missing parameters: {c}"
        print(f"  [ok] {name:8s} player={idx} {m.ROLE:11s} -> {[c['commandType'] for c in cmds]}")
    print()


def test_agent_pin_beats_bad_routing():
    print("=== a wrong myPlayers must not change who a runtime plays as ===")
    for name, idx in AGENT_INDEX.items():
        m = _load_agent(name)
        cmds = m.squad.handle(_platform_payload((idx + 2) % 5), my_index=m.MY_PLAYER_INDEX)
        assert all(c["playerId"] == idx for c in cmds), \
            f"{name} followed the payload instead of its pin"
    print(f"  [ok] all five ignored a mis-routed myPlayers\n")


def test_agents_share_one_lib():
    print("=== agent files stay thin; logic lives in lib/ ===")
    import pathlib
    for name in AGENT_INDEX:
        p = pathlib.Path("agents") / name / "src" / "main.py"
        lines = [l for l in p.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        assert len(lines) < 70, f"{name}/src/main.py has {len(lines)} lines; logic is leaking out of lib/"
    print(f"  [ok] all five under 70 significant lines\n")


if __name__ == "__main__":
    test_all_situations()
    test_gk_holds_the_angle()
    test_gk_distributes_not_passes()
    test_shoots_when_clear()
    test_budget()
    test_squad_without_llm()
    test_analysis()
    test_analysis_is_fast()
    test_models_per_role()
    test_memory()
    test_llm_used_when_fast()
    test_llm_late_falls_back()
    test_llm_error_falls_back()
    test_llm_invalid_rejected()
    test_memory_reaches_model()
    test_gateway_off_by_default()
    test_gateway_unreachable_degrades()
    test_gateway_loads_once()
    test_squad_runs_without_gateway()
    test_tools_reach_the_agent()
    test_tool_handlers()
    test_agents_pin_their_player()
    test_agent_pin_beats_bad_routing()
    test_agents_share_one_lib()
    print("All local tests passed.")
