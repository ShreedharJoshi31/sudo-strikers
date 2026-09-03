"""Create the AgentCore Gateway and register the four Lambda tools on it.

    AWS_ACCOUNT_ID=... GATEWAY_ROLE_ARN=... python manage_gateway.py

Idempotent: an existing gateway with the same name is reused, and a target that
is already registered is left alone. Safe to re-run after adding a tool.

Prints the gateway URL on the last line so a deploy script can capture it:

    export AFC_GATEWAY_URL=$(python manage_gateway.py | tail -1)

The `inlinePayload` schemas below are what the model actually sees. They are the
tool documentation, so the descriptions say WHEN to call each one — a tool the
model cannot tell apart from the precomputed block will simply never be used.
"""

from __future__ import annotations

import os
import sys
import time

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
GATEWAY_NAME = os.environ.get("AFC_GATEWAY_NAME", "afc-contender-tools")
LAMBDA_PREFIX = os.environ.get("AFC_LAMBDA_PREFIX", "afc-tool")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")
ROLE_ARN = os.environ.get("GATEWAY_ROLE_ARN", "")

_OBSERVATION = {
    "type": "object",
    "description": "The current observation exactly as received this tick.",
}

TOOLS = [
    {
        "name": "evaluate_pass",
        "lambda": "evaluate-pass",
        "schema": {
            "name": "evaluate_pass",
            "description": (
                "Score ONE specific pass you are considering. Use when weighing a "
                "particular receiver or pass type that is not already the top option "
                "in `analysis.pass_options` — for example a THROUGH ball rather than "
                "the GROUND pass suggested. Returns arrival probability, the margin "
                "in seconds, and which opponent contests it."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation": _OBSERVATION,
                    "receiver_id": {"type": "string", "description": "Teammate to pass to."},
                    "pass_type": {"type": "string", "enum": ["GROUND", "AERIAL", "THROUGH"]},
                    "policy": {"type": "string", "enum": ["SAFEST", "BEST_VALUE"]},
                },
                "required": ["observation", "receiver_id"],
            },
        },
    },
    {
        "name": "rank_passes",
        "lambda": "rank-passes",
        "schema": {
            "name": "rank_passes",
            "description": (
                "Rank every available pass under a chosen risk appetite. Use when the "
                "situation calls for a different appetite than the default — SAFEST "
                "when protecting a lead, BEST_VALUE when chasing a goal. The block in "
                "your prompt only ever shows the top three under one fixed policy."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation": _OBSERVATION,
                    "policy": {"type": "string", "enum": ["SAFEST", "BEST_VALUE"]},
                    "limit": {"type": "integer", "description": "Max options (default 6)."},
                },
                "required": ["observation"],
            },
        },
    },
    {
        "name": "evaluate_position",
        "lambda": "evaluate-position",
        "schema": {
            "name": "evaluate_position",
            "description": (
                "What the pitch would look like from a different point. Use to decide "
                "whether MOVING somewhere first beats acting from where you stand — it "
                "re-runs the shot and space maths from the hypothetical position and "
                "compares it against your current one."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation": _OBSERVATION,
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["observation", "x", "y"],
            },
        },
    },
    {
        "name": "scout_opponent",
        "lambda": "scout-opponent",
        "schema": {
            "name": "scout_opponent",
            "description": (
                "What ONE opponent has actually done so far this match — how hard they "
                "press, whether they tackle, whether they are tiring. Use before "
                "committing to a duel with a specific player. Pass the `scouting` "
                "profiles from your prompt through as `profiles`."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "opponent_id": {"type": "string"},
                    "profiles": {"type": "object", "description": "The `scouting` block."},
                },
                "required": ["opponent_id"],
            },
        },
    },
]


def _client():
    return boto3.client("bedrock-agentcore-control", region_name=REGION)


def find_existing(client) -> str | None:
    try:
        for gw in client.list_gateways().get("items", []):
            if gw.get("name") == GATEWAY_NAME:
                return gw.get("gatewayId")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  could not list gateways: {exc}", file=sys.stderr)
    return None


def create(client) -> str:
    print(f"  creating gateway {GATEWAY_NAME}", file=sys.stderr)
    resp = client.create_gateway(
        name=GATEWAY_NAME,
        roleArn=ROLE_ARN,
        protocolType="MCP",
        # NONE keeps the workshop path simple. Put a token in front of it before
        # anything that is not a practice match.
        authorizerType="NONE",
    )
    return resp["gatewayId"]


def wait_ready(client, gateway_id: str, max_wait: int = 150) -> None:
    for _ in range(max_wait // 5):
        status = client.get_gateway(gatewayIdentifier=gateway_id).get("status")
        if status == "READY":
            return
        if status in ("FAILED", "DELETING"):
            raise RuntimeError(f"gateway entered {status}")
        time.sleep(5)
    raise TimeoutError(f"gateway not READY within {max_wait}s")


def register(client, gateway_id: str) -> None:
    existing = set()
    try:
        for t in client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", []):
            existing.add(t.get("name"))
    except Exception:                                          # noqa: BLE001
        pass

    for tool in TOOLS:
        if tool["name"] in existing:
            print(f"  target {tool['name']} already registered", file=sys.stderr)
            continue
        arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-{tool['lambda']}"
        print(f"  registering {tool['name']} -> {arn}", file=sys.stderr)
        client.create_gateway_target(
            gatewayIdentifier=gateway_id,
            name=tool["name"],
            targetConfiguration={
                "mcp": {
                    "lambda": {
                        "lambdaArn": arn,
                        "toolSchema": {"inlinePayload": [tool["schema"]]},
                    }
                }
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"}
            ],
        )


def main() -> int:
    if not ACCOUNT_ID or not ROLE_ARN:
        print("AWS_ACCOUNT_ID and GATEWAY_ROLE_ARN are required", file=sys.stderr)
        return 2

    client = _client()
    gateway_id = find_existing(client)
    if gateway_id:
        print(f"  reusing gateway {gateway_id}", file=sys.stderr)
    else:
        gateway_id = create(client)
        wait_ready(client, gateway_id)

    register(client, gateway_id)
    url = f"https://{gateway_id}.gateway.bedrock-agentcore.{REGION}.amazonaws.com/mcp"
    print(f"\n  export AFC_GATEWAY_URL={url}\n", file=sys.stderr)
    print(url)                                     # stdout: capturable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
