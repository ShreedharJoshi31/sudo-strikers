"""AgentCore Gateway: real MCP tools the model can choose to call.

How this differs from `analysis.py`
-----------------------------------
`analysis.py` PUSHES a fixed block of numbers into every prompt. It is free and
always there, but it can only answer questions we thought of in advance.

These are PULL. The model decides when to ask, and asks about the specific
thing it is considering — "if I play a THROUGH ball to #9, does it arrive?",
"what does the pitch look like if I move here first?" That is the class of
question a precomputed block cannot answer, because it does not know what the
model is weighing.

Both exist on purpose. The precomputed block covers the common case at zero
cost; the tools cover the "what if" that actually needs a round trip.

Latency
-------
The platform budget is 5s and our internal deadline is 1.8s, so one or two tool
round trips fit where they would not have under the 1s budget this project was
originally written against. They are still not free, and a tool call that
overruns the deadline is handled the same way as a slow model: the policy
command computed before the model was ever asked is sent instead. A Gateway
outage costs latency, never a tick.

Enable by setting AFC_GATEWAY_URL. Unset, everything here is inert and the
agent is built exactly as it was before.
"""

from __future__ import annotations

import os
import threading

GATEWAY_URL = os.environ.get("AFC_GATEWAY_URL", "").strip()
GATEWAY_TOKEN = os.environ.get("AFC_GATEWAY_TOKEN", "").strip()
#: Fail fast: a Gateway that has not answered by now cannot help this tick.
CONNECT_TIMEOUT = float(os.environ.get("AFC_GATEWAY_TIMEOUT", "1.0"))


def enabled() -> bool:
    return bool(GATEWAY_URL)


class GatewayTools:
    """Holds the MCP client and the tool list fetched from it.

    Tools are fetched ONCE, on first use, and reused. Re-listing per tick would
    spend a round trip on something that does not change during a match.

    Every failure path here returns "no tools" rather than raising. A Gateway
    that is missing, misconfigured or down must degrade the squad to the
    behaviour it had before Gateway existed — never take it offline.
    """

    def __init__(self, url: str = "", token: str = "") -> None:
        self.url = url or GATEWAY_URL
        self.token = token or GATEWAY_TOKEN
        self._client = None
        self._tools: list = []
        self._tried = False
        self._error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _transport(self):
        from mcp.client.streamable_http import streamablehttp_client

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return streamablehttp_client(self.url, headers=headers)

    def load(self) -> list:
        """Connect and list tools. Safe to call repeatedly; only acts once."""
        if self._tried:
            return self._tools
        with self._lock:
            if self._tried:
                return self._tools
            self._tried = True
            if not self.url:
                return self._tools
            try:
                from strands.tools.mcp.mcp_client import MCPClient

                client = MCPClient(self._transport)
                # list_tools_sync must run inside the client context or the
                # connection is not yet open when the call is made.
                with client:
                    tools = client.list_tools_sync()
                self._client = client
                self._tools = list(tools)
            except Exception as exc:                     # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
                self._tools = []
        return self._tools

    # ----------------------------------------------------------------- access
    @property
    def client(self):
        return self._client

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def error(self) -> str | None:
        return self._error

    def status(self) -> dict:
        return {
            "configured": bool(self.url),
            "connected": bool(self._tools),
            "tool_count": len(self._tools),
            "tool_names": [getattr(t, "tool_name", getattr(t, "name", "?"))
                           for t in self._tools],
            "error": self._error,
        }


class _NullContext:
    """Used when there is no Gateway, so callers need no branch."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _ExclusiveSession:
    """One thread inside the MCP client at a time.

    Decisions run on a thread pool, and a call that misses its deadline is
    ABANDONED, not stopped — its thread is still inside the session when the
    next tick starts. MCPClient is not re-entrant, so the second thread used to
    fail instantly, which showed up as every decision after the first timeout
    erroring out. Measured: 4 of 5 ticks lost that way.

    So entry is serialised. If the previous call still holds the session, this
    one proceeds WITHOUT tools rather than failing: a decision with fewer
    options beats no decision at all.
    """

    def __init__(self, client, wait: float) -> None:
        self._client = client
        self._wait = wait
        self._held = False

    def __enter__(self):
        self._held = _SESSION_LOCK.acquire(timeout=self._wait)
        if not self._held:
            return None                       # busy: run tool-less this tick
        try:
            self._client.__enter__()
        except Exception:                     # noqa: BLE001
            _SESSION_LOCK.release()
            self._held = False
            return None
        return self._client

    def __exit__(self, *exc):
        if not self._held:
            return False
        try:
            self._client.__exit__(*exc)
        except Exception:                     # noqa: BLE001
            pass                              # never mask the real outcome
        finally:
            _SESSION_LOCK.release()
            self._held = False
        return False


#: Module-level: one Gateway connection is shared by every player in a process.
_SESSION_LOCK = threading.Lock()

#: How long to wait for the session before giving up on tools for this tick.
#: Short on purpose — the point is to not spend the decision budget queueing.
SESSION_WAIT = float(os.environ.get("AFC_GATEWAY_SESSION_WAIT", "0.25"))


def session(tools: GatewayTools | None):
    """Context manager to wrap an agent call in.

    The MCP connection has to be open while the agent runs, or a tool the model
    chooses to call has nothing underneath it. With no Gateway this is a no-op,
    so `with gateway.session(...)` is safe on every path.
    """
    if tools is None or tools.client is None:
        return _NullContext()
    return _ExclusiveSession(tools.client, SESSION_WAIT)
