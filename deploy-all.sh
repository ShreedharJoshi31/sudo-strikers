#!/bin/bash
set -e

# ============================================================================
# Deploy all five agents IN PARALLEL, then sweep stale sessions once.
#
#   AWS_DEFAULT_REGION=us-east-1 ./deploy-all.sh
#   AWS_DEFAULT_REGION=us-east-1 ./deploy-all.sh ai-gk ai-mid   # a subset
#
# Each agent is deployed by deploy-one.sh, which is safe to run concurrently.
# Per-player wrappers (deploy-gk.sh, deploy-mid.sh, ...) exist if you would
# rather drive them yourself.
#
# The stale-session sweep happens ONCE, here, after every deploy has landed.
# It is global and racing it against in-flight deploys would defeat the point.
#
# Prerequisites:
#   pip install bedrock-agentcore-starter-toolkit
#   aws configure   (or export AWS_PROFILE)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION

ALL_AGENTS=("ai-gk" "ai-def1" "ai-def2" "ai-mid" "ai-fwd")
if [ "$#" -gt 0 ]; then AGENTS=("$@"); else AGENTS=("${ALL_AGENTS[@]}"); fi

echo "Checking prerequisites..."
for tool in agentcore aws rsync; do
  command -v "$tool" >/dev/null || {
    echo "ERROR: '$tool' not found."
    [ "$tool" = "agentcore" ] && echo "  Install: pip install bedrock-agentcore-starter-toolkit"
    exit 1; }
done

# Resolved once and exported, so five parallel workers do not each call sts.
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  echo "ERROR: no valid AWS credentials."; exit 1; }
export AWS_ACCOUNT_ID
echo "  account $AWS_ACCOUNT_ID / region $AWS_DEFAULT_REGION"
echo "  deploying ${#AGENTS[@]} agent(s) in parallel"
echo ""

# Two parallel indexed arrays, NOT an associative array: macOS ships bash 3.2,
# where `declare -A` does not exist and fails at run time.
PIDS=()
for agent in "${AGENTS[@]}"; do
  "$SCRIPT_DIR/deploy-one.sh" "$agent" &
  PIDS+=($!)
done

# Waited on individually rather than a bare `wait`, so a failure is attributed
# to the agent that caused it instead of just failing the batch.
DEPLOYED=(); FAILED=()
for i in "${!AGENTS[@]}"; do
  if wait "${PIDS[$i]}"; then DEPLOYED+=("${AGENTS[$i]}"); else FAILED+=("${AGENTS[$i]}"); fi
done

echo ""
echo "=========================================="
echo "  deployed: ${DEPLOYED[*]:-none}"
echo "  failed:   ${FAILED[*]:-none}"
echo "=========================================="

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "Not sweeping sessions: some agents failed, so the platform would be"
  echo "left running a mix of builds. Fix and re-run before the sweep."
  exit 1
fi

# A deployed version does NOT reach an existing session: AgentCore pins a
# runtimeSessionId to the container that was live when the session started, and
# the platform reuses one long-lived session per slot. Without this the
# competition keeps running the PREVIOUS build while every hand-made probe runs
# the new one - a fixed bug that still fails the fitness check.
echo ""
echo "Stopping stale platform sessions so this deploy actually takes effect..."
# Must be an interpreter whose botocore knows the `bedrock-agentcore` service.
# The system python3 usually does NOT: it raises UnknownServiceError, which is
# how this step silently did nothing the first time it ran.
AFC_PY="$SCRIPT_DIR/.venv/bin/python"
[ -x "$AFC_PY" ] || AFC_PY="python3"
if [ ! -f "$SCRIPT_DIR/stop_stale_sessions.py" ]; then
  echo "  WARNING: stop_stale_sessions.py is MISSING from the repo."
  echo "           The platform may keep running the PREVIOUS build."
elif ! "$AFC_PY" "$SCRIPT_DIR/stop_stale_sessions.py" --minutes 120; then
  echo "  WARNING: could not stop sessions - the platform may still be running"
  echo "           the previous build. Run stop_stale_sessions.py by hand."
fi

echo ""
echo "Register these ARNs with the platform, one per player index:"
for a in "${DEPLOYED[@]}"; do
  case "$a" in
    ai-gk)   idx=0; rt=afc_gk_agent   ;;
    ai-def1) idx=1; rt=afc_def1_agent ;;
    ai-def2) idx=2; rt=afc_def2_agent ;;
    ai-mid)  idx=3; rt=afc_mid_agent  ;;
    ai-fwd)  idx=4; rt=afc_fwd_agent  ;;
  esac
  echo "  player $idx  arn:aws:bedrock-agentcore:$AWS_DEFAULT_REGION:$AWS_ACCOUNT_ID:runtime/$rt"
done
