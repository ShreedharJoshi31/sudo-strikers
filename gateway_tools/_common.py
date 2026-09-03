"""Shared plumbing for the Gateway Lambda tools.

Each tool is its own Lambda (the pattern the workshop uses and the one known to
work with AgentCore Gateway targets), but they all need the same two things:
lib/ on the path, and a tolerant reader for the event body.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))


def body(event):
    """Gateway may deliver arguments inline or JSON-encoded under `body`."""
    if isinstance(event, dict) and isinstance(event.get("body"), str):
        return json.loads(event["body"])
    if isinstance(event, dict) and isinstance(event.get("body"), dict):
        return event["body"]
    return event if isinstance(event, dict) else {}
