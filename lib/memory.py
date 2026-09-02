"""Per-player short-term memory, held in process.

Deliberately not AgentCore Memory. That service puts a network retrieval inside
the one-second decision budget, and a late answer is a decision thrown away.
This is a bounded deque of the last few ticks, costs nothing, and covers the
thing memory is actually for here: not repeating yourself and noticing that
what you tried last tick did not work.

What the platform already gives you every tick — your own and your teammates'
`last_command`, `recent_events`, score, clock, stamina — is NOT duplicated here.
Only what the observation cannot tell you is stored.

Toggle with AFC_MEMORY=0 to measure whether it helps.
"""

from __future__ import annotations

from collections import deque


class PlayerMemory:
    """Last `depth` decisions for one player."""

    def __init__(self, depth: int = 4) -> None:
        self._entries: deque[dict] = deque(maxlen=depth)

    def record(self, tick: int, command_type: str, rationale: str, source: str) -> None:
        self._entries.append({
            "tick": tick,
            "did": command_type,
            "why": rationale[:60],
            "from": source,          # "llm" or "policy"
        })

    def recent(self) -> list[dict]:
        return list(self._entries)

    def repeated(self, command_type: str) -> int:
        """How many of the stored ticks issued this same command.

        A player that has issued MOVE_TO four ticks running is usually stuck,
        which is exactly the pattern a single-tick observation cannot show.
        """
        return sum(1 for e in self._entries if e["did"] == command_type)

    def summary(self) -> dict | None:
        if not self._entries:
            return None
        last = self._entries[-1]
        return {
            "your_last_few": [f"{e['did']} ({e['why']})" for e in self._entries],
            "repeating": self.repeated(last["did"]) >= 3,
        }


class SquadMemory:
    """One PlayerMemory per player id, created on demand."""

    def __init__(self, depth: int = 4) -> None:
        self._depth = depth
        self._players: dict[str, PlayerMemory] = {}

    def for_player(self, player_id: str) -> PlayerMemory:
        if player_id not in self._players:
            self._players[player_id] = PlayerMemory(self._depth)
        return self._players[player_id]

    def reset(self) -> None:
        self._players.clear()
