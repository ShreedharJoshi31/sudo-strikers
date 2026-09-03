#!/bin/bash
# Deploy just the ai-gk agent (player 0).
#
# Safe to run at the same time as the other four - staging is per agent and
# this only ever touches its own. Run all five at once with:
#
#   ./deploy-gk.sh & ./deploy-def1.sh & ./deploy-def2.sh & \
#   ./deploy-mid.sh & ./deploy-fwd.sh & wait
#
# Doing that skips the stale-session sweep, which is global and must not run
# five times over. Run it once yourself afterwards, or use ./deploy-all.sh
# which fans out and then sweeps once.
exec "$(cd "$(dirname "$0")" && pwd)/deploy-one.sh" ai-gk
