"""Local HTTP server speaking the platform contract.

    POST /invocations
    body: {"player_id": "H3", "observation": {...}}
    resp: {"type": "PASS", "target_player_id": "H4", "target": null, "rationale": "..."}

Point a team file at it with:

    agent:
      transport: http
      url: http://127.0.0.1:8081/invocations

Run:
    python serve.py                 # policy only, no AWS
    AFC_USE_LLM=1 python serve.py   # policy + Bedrock, needs credentials
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import brain as brain_mod
from brain import Squad

USE_LLM = os.environ.get("AFC_USE_LLM", "").lower() in ("1", "true", "yes")

TACTICS = os.environ.get("AFC_TACTICS", """
Win the ball back inside three seconds of losing it.
Move it forward early; short and clean beats long and hopeful.
Shoot when the lane is open rather than working for a better angle.
""")

ROLE_PROMPTS = {
    "GK": "Hold the angle between ball and goal. Claim loose balls in the area. Restart fast to the most advanced open teammate.",
    "DEFENDER": "Hold the back line. Do not both step up at once - read your partner.",
    "MIDFIELDER": "Link the lines. Offer an angle when a teammate has it, screen the middle when they do not.",
    "FORWARD": "Attack the space behind the last defender. Shoot early when the lane opens.",
}

app = FastAPI(title="afc-contender")
squad = Squad(tactics=TACTICS, role_prompts=ROLE_PROMPTS, use_llm=USE_LLM)


@app.post("/invocations")
async def invocations(request: Request):
    """The platform contract: {"gameState": ..., "teamId": N, "myPlayers": [i]}
    in, a JSON ARRAY of commands out."""
    payload = await request.json()
    return JSONResponse(squad.handle(payload))


@app.get("/stats")
async def stats():
    return squad.stats.summary()


@app.get("/ping")
async def ping():
    return {"ok": True, "llm": USE_LLM,
            "models": brain_mod.DEFAULT_MODELS,
            "deadline_s": squad.deadline,
            "analysis": brain_mod.USE_ANALYSIS, "memory": brain_mod.USE_MEMORY,
            "scouting": brain_mod.USE_SCOUTING}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8081")),
        log_level="warning",
    )
