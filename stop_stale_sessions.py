#!/usr/bin/env python3
"""Kill AgentCore runtime sessions so a redeploy actually reaches the platform.

WHY THIS EXISTS
---------------
`invoke_agent_runtime` pins a `runtimeSessionId` to the container - and
therefore the CODE VERSION - that was live when that session was first created.
Deploying a new version does NOT move an existing session onto it. The session
keeps being served by the old container until it is stopped or expires.

The competition platform reuses one long-lived session per slot
(`scrimmage-drill-<uuid>-posN`). So after a redeploy the platform keeps running
the OLD build while every fresh probe you make by hand runs the NEW one. The
symptom is a fitness check that fails on a bug you have already fixed and
verified, which is exactly as confusing as it sounds - it cost several deploy
cycles before it was spotted.

So: after EVERY deploy, stop the platform's sessions. `deploy-all.sh` calls this
automatically; run it by hand if you deploy some other way.

    python3 stop_stale_sessions.py                # scrape logs, stop what it finds
    python3 stop_stale_sessions.py --minutes 180  # look further back
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import boto3

#: runtime name -> the log group that records its invocations.
RUNTIMES = {
    "afc_gk_agent": "afc_gk_agent-BotAHM8AjC",
    "afc_def1_agent": "afc_def1_agent-QSIPlT4s5Q",
    "afc_def2_agent": "afc_def2_agent-W1r6kvCEPM",
    "afc_mid_agent": "afc_mid_agent-KrZiehBL1B",
    "afc_fwd_agent": "afc_fwd_agent-HiycB5BQU1",
}
ACCOUNT = "030253333865"
REGION = "us-east-1"

#: Sessions we created ourselves while testing. Stopping them is harmless but
#: noisy, and they are not what breaks a fitness run.
MINE = re.compile(r"^(verify|vfy|gkfmt|gkalt|afcwarm|repro|chk|fresh|test)")
SESSION = re.compile(r'"sessionId":\s*"([^"]+)"')


def sessions_from_logs(logs, group: str, minutes: int) -> set[str]:
    """Every session id this runtime has served recently."""
    start = int((time.time() - minutes * 60) * 1000)
    found: set[str] = set()
    token = None
    while True:
        kw = {"logGroupName": group, "startTime": start, "limit": 10000}
        if token:
            kw["nextToken"] = token
        page = logs.filter_log_events(**kw)
        for event in page.get("events", []):
            for sid in SESSION.findall(event["message"]):
                if not MINE.match(sid):
                    found.add(sid)
        token = page.get("nextToken")
        if not token:
            return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--profile", default="aws-football")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    logs = session.client("logs")
    core = session.client("bedrock-agentcore")

    stopped = 0
    for name, runtime_id in RUNTIMES.items():
        arn = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/{runtime_id}"
        group = f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT"
        try:
            ids = sessions_from_logs(logs, group, args.minutes)
        except Exception as exc:                      # log group may not exist yet
            print(f"  {name}: cannot read logs ({type(exc).__name__})")
            continue
        if not ids:
            print(f"  {name}: no platform sessions in the last {args.minutes}m")
            continue
        for sid in sorted(ids):
            try:
                core.stop_runtime_session(runtimeSessionId=sid, agentRuntimeArn=arn)
                print(f"  {name}: stopped {sid}")
                stopped += 1
            except core.exceptions.ResourceNotFoundException:
                pass                                  # already gone; that is the goal
            except Exception as exc:
                print(f"  {name}: {sid} -> {type(exc).__name__}")
    print(f"\n{stopped} session(s) stopped. The platform's next call starts a "
          f"fresh session on the version you just deployed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
