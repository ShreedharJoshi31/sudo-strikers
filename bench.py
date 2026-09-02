"""Benchmark this team against the arena's baseline AI.

Runs in-process against the engine — no HTTP, no subprocess — because matches
driven over HTTP are not reproducible: whether a response lands inside the
decision window depends on real network timing, so the same seed gives
different scores. In-process runs are deterministic, which is what makes
parameter tuning meaningful.

    python bench.py                        # 20 matches vs crimson-rovers
    python bench.py --matches 40 --opponent teams/azure-arrows.yaml
    python bench.py --sweep gk_dive_range=9,12,15

Uses the arena's virtualenv, so run it as:
    ../agentic-football-arena/.venv/bin/python bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = Path(os.environ.get("AFC_ARENA", HERE.parent / "agentic-football-arena"))
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(ARENA))

from afc.agents.factory import build_client                     # noqa: E402
from afc.agents.local import LocalClient                        # noqa: E402
from afc.engine.config import EngineConfig                      # noqa: E402
from afc.engine.match import Match, TeamSlot, players_meta      # noqa: E402
from afc.engine.teams import TeamConfig                         # noqa: E402

from brain import Squad                                          # noqa: E402
from policy import DEFAULT, Params                               # noqa: E402


def run_one(my_cfg, opp_cfg, params: Params, seed: int, as_home: bool) -> dict:
    engine = EngineConfig()
    my_side = "home" if as_home else "away"
    opp_side = "away" if as_home else "home"

    squad = Squad(params=params, use_llm=False)
    my_meta = players_meta(my_cfg, my_side, engine)
    my_client = LocalClient({m["id"]: squad.decide for m in my_meta}, use_thread=False)

    opp_meta = players_meta(opp_cfg, opp_side, engine)
    opp_client = build_client(opp_cfg, opp_meta, engine, seed=seed)

    slots = {
        my_side: TeamSlot(my_cfg, my_side, my_client),
        opp_side: TeamSlot(opp_cfg, opp_side, opp_client),
    }
    match = Match(slots["home"], slots["away"], engine, seed=seed, log_path=None)
    result = asyncio.run(match.run())
    mine = result.stats["teams"][my_side]
    return {
        "mine": result.score_home if as_home else result.score_away,
        "theirs": result.score_away if as_home else result.score_home,
        "timeouts": mine["timeouts"],
        "shots": mine["shots"],
        "passes": mine["passes"],
        "pass_pct": mine["pass_accuracy_pct"],
    }


def evaluate(my_cfg, opponents: list, params: Params, matches: int, verbose: bool = True) -> dict:
    """Play `matches` against EVERY opponent and aggregate.

    Tuning against a single opponent overfits to that formation — measured:
    a GK change worth 2.12 ppg vs one team was worth 1.20 ppg vs another.
    Sweeping across the field is the guard against that.
    """
    w = d = l = gf = ga = timeouts = 0
    per_opponent = {}
    for opp_cfg in opponents:
        ow = od = ol = 0
        for i in range(matches):
            r = run_one(my_cfg, opp_cfg, params, seed=i + 1, as_home=(i % 2 == 0))
            gf += r["mine"]
            ga += r["theirs"]
            timeouts += r["timeouts"]
            ow += r["mine"] > r["theirs"]
            od += r["mine"] == r["theirs"]
            ol += r["mine"] < r["theirs"]
        w, d, l = w + ow, d + od, l + ol
        per_opponent[opp_cfg.name] = (ow * 3 + od) / matches
        if verbose:
            print(f"  {opp_cfg.name:<20} {ow}W {od}D {ol}L   {(ow * 3 + od) / matches:.2f} ppg")
    total = matches * len(opponents)
    pts = w * 3 + d
    return {"W": w, "D": d, "L": l, "gf": gf, "ga": ga, "pts": pts,
            "ppg": pts / total, "timeouts": timeouts, "per_opponent": per_opponent}


def report(label: str, s: dict, matches: int) -> None:
    print(f"\n  {label}")
    print(f"    record    {s['W']}W {s['D']}D {s['L']}L   ({s['pts']}/{matches * 3} pts, {s['pts'] / (matches * 3) * 100:.0f}%)")
    print(f"    goals     {s['gf']} for, {s['ga']} against  (diff {s['gf'] - s['ga']:+d})")
    print(f"    timeouts  {s['timeouts']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=10,
                    help="matches per opponent")
    ap.add_argument("--team", default=str(HERE / "team.yaml"))
    ap.add_argument("--opponent", default="all",
                    help="'all' for every team in the arena, or a path")
    ap.add_argument("--sweep", help="param=v1,v2,v3 — compare values of one Params field")
    args = ap.parse_args()

    my_cfg = TeamConfig.load(args.team)
    if args.opponent == "all":
        opp_paths = sorted((ARENA / "teams").glob("*.yaml"))
    else:
        p = Path(args.opponent)
        opp_paths = [p if p.is_absolute() else ARENA / p]
    opponents = [TeamConfig.load(p) for p in opp_paths]
    opp_label = "all opponents" if args.opponent == "all" else opp_paths[0].name

    if args.sweep:
        field, raw = args.sweep.split("=", 1)
        if field not in {f.name for f in dataclasses.fields(Params)}:
            print(f"unknown param: {field}", file=sys.stderr)
            return 2
        print(f"sweeping {field} over {raw}, {args.matches} matches vs each of "
              f"{len(opponents)} opponent(s)\n")
        rows = []
        for value in [float(v) for v in raw.split(",")]:
            params = dataclasses.replace(DEFAULT, **{field: value})
            s = evaluate(my_cfg, opponents, params, args.matches, verbose=False)
            rows.append((value, s))
            spread = " ".join(f"{v:.2f}" for v in s["per_opponent"].values())
            print(f"  {field}={value:<8g} {s['W']}W {s['D']}D {s['L']}L  "
                  f"{s['ppg']:.2f} ppg  diff {s['gf'] - s['ga']:+d}   per-opponent: {spread}")
        best = max(rows, key=lambda r: (r[1]["ppg"], r[1]["gf"] - r[1]["ga"]))
        print(f"\n  best: {field}={best[0]:g} at {best[1]['ppg']:.2f} ppg")
        print("  Check the per-opponent spread — a value that wins on average but "
              "collapses against one formation is a bad bet.")
        return 0

    total = args.matches * len(opponents)
    print(f"{args.matches} matches vs each of {len(opponents)} opponent(s) "
          f"= {total} total ({opp_label})\n")
    s = evaluate(my_cfg, opponents, DEFAULT, args.matches)
    report("current params", s, total)
    if s["timeouts"]:
        print("\n  Timeouts are lost decisions. Fix these before touching tactics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
