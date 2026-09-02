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
    print("All local tests passed.")
