#!/bin/bash
set -e

# ============================================================================
# Deploy the four tool Lambdas, then create the Gateway and register them.
#
#   AWS_DEFAULT_REGION=us-east-1 ./deploy_gateway.sh
#
# Prints the export line for AFC_GATEWAY_URL at the end. Nothing in the agent
# uses the Gateway until that variable is set, so this is safe to run and
# ignore.
#
# Each Lambda bundles lib/ so the tools score passes with the SAME model the
# agent uses locally. One implementation: a tool can never disagree with the
# prompt it is meant to refine.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/_gwbuild"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION
LAMBDA_PREFIX="${AFC_LAMBDA_PREFIX:-afc-tool}"

for tool in aws python3 zip; do
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found."; exit 1; }
done

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) || {
  echo "ERROR: no valid AWS credentials."; exit 1; }
export AWS_ACCOUNT_ID
echo "  account $AWS_ACCOUNT_ID / region $AWS_DEFAULT_REGION"

LAMBDA_ROLE_ARN="${LAMBDA_ROLE_ARN:-}"
if [ -z "$LAMBDA_ROLE_ARN" ]; then
  echo "ERROR: set LAMBDA_ROLE_ARN to a role Lambda can assume."
  echo "  It needs only AWSLambdaBasicExecutionRole; the tools make no AWS calls."
  exit 1
fi

trap 'rm -rf "$BUILD"' EXIT
rm -rf "$BUILD"; mkdir -p "$BUILD"

deploy_one() {          # $1 = module basename, $2 = lambda name suffix
  local mod="$1" name="$LAMBDA_PREFIX-$2" stage="$BUILD/$2"
  mkdir -p "$stage"
  cp "$SCRIPT_DIR/gateway_tools/$mod.py" "$stage/"
  cp "$SCRIPT_DIR/gateway_tools/_common.py" "$stage/"
  rsync -a --exclude='__pycache__' "$SCRIPT_DIR/lib/" "$stage/lib/"
  (cd "$stage" && zip -qr ../"$2".zip .)

  if aws lambda get-function --function-name "$name" >/dev/null 2>&1; then
    echo "  updating $name"
    aws lambda update-function-code --function-name "$name" \
      --zip-file "fileb://$BUILD/$2.zip" --output text >/dev/null
  else
    echo "  creating $name"
    aws lambda create-function --function-name "$name" \
      --runtime python3.12 --role "$LAMBDA_ROLE_ARN" \
      --handler "$mod.lambda_handler" --timeout 10 --memory-size 512 \
      --zip-file "fileb://$BUILD/$2.zip" --output text >/dev/null
  fi
  # Gateway invokes the function through its own service role.
  aws lambda add-permission --function-name "$name" \
    --statement-id agentcore-gateway --action lambda:InvokeFunction \
    --principal bedrock-agentcore.amazonaws.com >/dev/null 2>&1 || true
}

deploy_one evaluate_pass      evaluate-pass
deploy_one rank_passes        rank-passes
deploy_one evaluate_position  evaluate-position
deploy_one scout_opponent     scout-opponent

echo ""
echo "  registering gateway targets ..."
URL=$(python3 "$SCRIPT_DIR/manage_gateway.py")

echo ""
echo "Gateway ready. Enable it with:"
echo ""
echo "  export AFC_GATEWAY_URL=$URL"
echo ""
echo "Then check the agent picked the tools up:  curl -s localhost:8081/ping"
