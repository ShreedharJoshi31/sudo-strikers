"""Opponent model, learned in-process over the ticks of one match.

Why this exists
---------------
Two things the platform never tells you decide most passes: how fast an
opponent actually is, and how long they take to react. Both are engine
constants nobody publishes. But every tick hands us all five opponents with
their positions and velocities, so we can measure them ourselves and stop
guessing after about a minute of football.

Deliberately NOT AgentCore Memory, for the same reason as `memory.py`: that
service puts a network retrieval inside the decision budget, and a decision
that arrives late is a decision thrown away. This is a few hundred bytes of
running aggregates per opponent, updated in tens of microseconds.

The determinism contract
------------------------
Five agent runtimes play for us, one per player, each invoked independently
and each given its own isolated memory context — there is no shared store and
no message passing. What keeps them in agreement is that they all receive the
*same full observation*, so they can each build the *same* model.

That only works if the model is a pure function of the tick sequence. So:

  - no wall clock and no RNG anywhere in this module;
  - every aggregation over a collection is either order-free (sums, counts) or
    explicitly sorted, and every `max`/`min` breaks ties on id, exactly as
    `policy.defensive_assignment` does;
  - updates are keyed on the integer `tick` and are idempotent — a replayed
    tick changes nothing, and a tick older than the newest one seen is dropped.

Two Scouts fed the same tick sequence produce byte-identical `snapshot()`s,
including across separate processes under different PYTHONHASHSEEDs — which is
the case that matters, since the five runtimes are five processes. Agents that
miss different ticks will diverge slightly, which is fine: these are smoothed
estimates, not shared secrets.

Bounded memory
--------------
Nothing here grows with match length. Per opponent: a dozen floats, a 3-slot
top-speed window, a 6-slot reaction deque, and two counters hard-capped at a
fixed number of keys. Per Scout: running means, an 8-slot deque of tick
intervals, and at most one live pass event, because there is only ever one
ball. A 3000-tick run holds the same structure as a 100-tick one.

Never raises. Missing `last_action`, missing `stamina`, an opponent that
appears or disappears mid-match, an empty opponent list, a malformed
observation, a tick that goes backwards — all absorbed, never propagated.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from policy import _dist

# ------------------------------------------------------------------- tunables


@dataclass(frozen=True)
class ScoutParams:
    """Every window length, decay rate and prior in one frozen place.

    The priors matter more than they look. They are what the passing model
    consumes before any data exists, and the sign of their error is not
    symmetric: under-estimating an opponent gets our pass intercepted, while
    over-estimating one only costs us a pass we could have made. So the
    defaults lean toward "assume they are fast and sharp".
    """

    # --- tick bookkeeping ------------------------------------------------
    tick_seconds_prior: float = 2.0   # platform interval; only a fallback
    tick_seconds_window: int = 8      # deque of measured game_time deltas
    # A long-lived process plays more than one match. A new match restarts the
    # count near zero, which a straggler from mid-match never does — so the
    # reset is keyed on "the tick is near the start" and not on "the tick went
    # backwards", which would throw the model away on a late redelivery.
    restart_tick: int = 2
    restart_min_ticks: int = 8

    # --- top speed: the self-calibrated V_TOP ----------------------------
    # 7.5 m/s is a real match sprint (7-8 m/s; elite humans peak near 10) and
    # is rounded up rather than down on purpose — see the class docstring.
    speed_prior: float = 7.5
    speed_prior_strength: float = 6.0   # prior worth ~6 observations
    speed_keep: int = 3                 # report the Kth fastest sample seen,
    speed_max_credible: float = 15.0    # which absorbs up to K-1 spikes
    speed_min_credible: float = 0.5
    speed_confident_samples: int = 12

    # --- reaction latency: the self-calibrated tau_react ------------------
    # Quantised to the tick interval, so a 2 s tick cannot resolve better than
    # ~1 s. 0.6 s says "assume they re-plan almost immediately", the safe side.
    react_prior: float = 0.6
    react_prior_strength: float = 3.0
    react_samples: int = 6              # bounded deque, median is reported
    react_window_ticks: int = 4         # give up on an event after this many
    react_align_cos: float = 0.7        # a 45 degree cone counts as "turning toward"
    react_min_speed: float = 1.0        # standing still is not a reaction
    react_lane_horizon: float = 12.0    # metres down the ball's line
    react_interest_radius: float = 25.0  # opponents further out are not involved
    react_min: float = 0.2
    react_max: float = 4.0
    ball_moving_speed: float = 4.0      # below this the ball is not "released"

    # --- stamina ----------------------------------------------------------
    stamina_min_span_s: float = 30.0    # need this long before a rate is real
    tired_stamina: float = 45.0         # matches policy.Params.tired_stamina
    # A drained player still runs at ~72% of fresh pace in most engines. Held
    # high deliberately: assuming they stay quick is the cheap mistake.
    speed_stamina_floor: float = 0.72

    # --- shape and press ---------------------------------------------------
    press_radius: float = 6.0           # opponents this close to the ball press
    gk_bias_m: float = 0.8              # keeper offset worth shooting away from
    gk_off_line_m: float = 2.5          # beyond this they are a sweeper-keeper

    # --- hard caps that keep memory bounded --------------------------------
    action_keys_max: int = 16
    mark_keys_max: int = 8

    # --- summary budget (this lands in a latency-critical prompt) ----------
    summary_max_notes: int = 5
    summary_note_chars: int = 92


# Not named DEFAULT: `policy.DEFAULT` already owns that name and both get
# imported into the same modules.
DEFAULT_SCOUT = ScoutParams()


# -------------------------------------------------------------------- helpers


def _f(value, default: float = 0.0) -> float:
    """float() that cannot raise, and rejects NaN/inf."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _xy(value) -> tuple[float, float]:
    """A 2-vector out of anything, or the origin."""
    try:
        return (_f(value[0]), _f(value[1]))
    except (TypeError, IndexError, KeyError):
        return (0.0, 0.0)


def _median(values: list[float]) -> float:
    """Median of a small list. Robust and, unlike a mean, spike-proof."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _shrink(observed: float, samples: int, prior: float, strength: float) -> float:
    """Blend a measurement into a prior by how much evidence backs it.

    One observation barely moves the answer; twenty own it. This is what lets
    `top_speed` and `reaction_delay` always return a usable number instead of
    None, and degrade smoothly rather than snapping when data arrives.
    """
    if samples <= 0:
        return prior
    w = samples / (samples + strength)
    return prior * (1.0 - w) + observed * w


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# `last_action` -> behaviour label. Spans both the current command vocabulary
# and the older one, because this reads whatever string the *engine* reports
# for an opponent, which is not necessarily the set we are allowed to send.
# Anything unmapped falls through to the raw string in `profile()` and simply
# contributes no phrase to `summary()`, so a new engine command shows up
# rather than being silently mislabelled.
_BEHAVIOUR = {
    "PRESS_BALL": "presser",
    "INTERCEPT": "presser",
    "TACKLE": "presser",
    "MARK": "marker",
    "FOLLOW_PLAYER": "marker",
    "MOVE_TO": "holder",
    "IDLE": "holder",
    "PASS": "carrier",
    "SHOOT": "carrier",
    "DRIBBLE": "carrier",
    "CLEAR": "carrier",
    "GK_DISTRIBUTE": "keeper",
    "GK_DIVE": "keeper",
    # team-shape orders say nothing about how this player behaves
    "SET_STANCE": "shape",
    "CLEAR_OVERRIDE": "shape",
    "RESET": "shape",
}


# ------------------------------------------------------------ per opponent


class _OpponentModel:
    """Running aggregates for one opponent. No per-tick history is kept."""

    __slots__ = (
        "id", "number", "role", "ticks",
        "_p", "_top", "_speed_n", "_prev_vel",
        "_react", "_stamina", "_stamina_t0", "_stamina_s0", "_stamina_span",
        "_def_n", "_def_x", "_def_y", "_att_n", "_att_x", "_att_y",
        "_actions", "_marks", "_mark_n",
        "_gk_n", "_gk_offset", "_gk_depth",
    )

    def __init__(self, opponent_id: str, params: ScoutParams) -> None:
        self._p = params
        self.id = opponent_id
        self.number: int | None = None
        self.role: str = ""
        self.ticks = 0

        # top speed: the K fastest samples ever seen, descending
        self._top: list[float] = []
        self._speed_n = 0
        # last tick's velocity, so "turning toward the lane" can be told apart
        # from "was already running that way"
        self._prev_vel: tuple[float, float] = (0.0, 0.0)

        self._react: deque[float] = deque(maxlen=params.react_samples)

        self._stamina: float | None = None
        self._stamina_t0: float | None = None
        self._stamina_s0: float | None = None
        self._stamina_span = 0.0

        # mean position, split by who has the ball
        self._def_n = 0
        self._def_x = 0.0
        self._def_y = 0.0
        self._att_n = 0
        self._att_x = 0.0
        self._att_y = 0.0

        self._actions: dict[str, int] = {}
        self._marks: dict[str, int] = {}
        self._mark_n = 0

        # keeper only
        self._gk_n = 0
        self._gk_offset = 0.0
        self._gk_depth = 0.0

    # -- updates ---------------------------------------------------------

    def see_speed(self, speed: float) -> None:
        p = self._p
        if not (p.speed_min_credible <= speed <= p.speed_max_credible):
            return  # a teleport or a parked player tells us nothing about V_TOP
        self._speed_n += 1
        if len(self._top) < p.speed_keep:
            self._top.append(speed)
            self._top.sort(reverse=True)
        elif speed > self._top[-1]:
            self._top[-1] = speed
            self._top.sort(reverse=True)

    def see_stamina(self, stamina: float, clock: float) -> None:
        self._stamina = stamina
        if self._stamina_t0 is None:
            self._stamina_t0 = clock
            self._stamina_s0 = stamina
        else:
            self._stamina_span = clock - self._stamina_t0

    def see_position(self, pos: tuple[float, float], they_have_ball: bool) -> None:
        if they_have_ball:
            self._att_n += 1
            self._att_x += pos[0]
            self._att_y += pos[1]
        else:
            self._def_n += 1
            self._def_x += pos[0]
            self._def_y += pos[1]

    def see_action(self, action: str) -> None:
        d = self._actions
        if action in d:
            d[action] += 1
        elif len(d) < self._p.action_keys_max:
            d[action] = 1

    def see_mark(self, our_player_id: str) -> None:
        d = self._marks
        self._mark_n += 1
        if our_player_id in d:
            d[our_player_id] += 1
        elif len(d) < self._p.mark_keys_max:
            d[our_player_id] = 1

    def see_keeper(self, offset: float, depth: float) -> None:
        self._gk_n += 1
        self._gk_offset += offset
        self._gk_depth += depth

    def record_reaction(self, seconds: float) -> None:
        p = self._p
        self._react.append(max(p.react_min, min(p.react_max, seconds)))

    # -- reads -----------------------------------------------------------

    def top_speed(self) -> float:
        """Calibrated V_TOP for this player, on fresh legs."""
        p = self._p
        if not self._top:
            return p.speed_prior
        # Early on use the running max. With a handful of glimpses the Kth
        # fastest is just "the slowest of three", which understates them — and
        # understating an opponent is the error that gets our pass cut out.
        # Once there is enough data for a spike to be recognisable as a spike,
        # switch to the Kth fastest, which one bad frame cannot inflate.
        observed = self._top[0] if self._speed_n < p.speed_confident_samples else self._top[-1]
        return _shrink(observed, self._speed_n, p.speed_prior, p.speed_prior_strength)

    def effective_top_speed(self) -> float:
        """V_TOP scaled by how tired they are right now.

        This is the payoff of tracking stamina: a tiring opponent covers less
        ground per second, so a lane that was cut out in the first minute is
        open in the fifth.
        """
        p = self._p
        base = self.top_speed()
        if self._stamina is None:
            return base
        frac = max(0.0, min(1.0, self._stamina / 100.0))
        return base * (p.speed_stamina_floor + (1.0 - p.speed_stamina_floor) * frac)

    def reaction_delay(self) -> float:
        p = self._p
        if not self._react:
            return p.react_prior
        return _shrink(
            _median(list(self._react)), len(self._react),
            p.react_prior, p.react_prior_strength,
        )

    def confidence(self) -> dict:
        p = self._p
        return {
            "speed": round(min(1.0, self._speed_n / p.speed_confident_samples), 2),
            "reaction": round(min(1.0, len(self._react) / p.react_samples), 2),
            "speed_samples": self._speed_n,
            "reaction_samples": len(self._react),
            "ticks_seen": self.ticks,
        }

    def stamina_drop_per_min(self) -> float:
        """Points of stamina lost per minute, or 0.0 until it is measurable."""
        if (
            self._stamina is None
            or self._stamina_s0 is None
            or self._stamina_span < self._p.stamina_min_span_s
        ):
            return 0.0
        return (self._stamina_s0 - self._stamina) / (self._stamina_span / 60.0)

    def projected_stamina(self, minutes_ahead: float) -> float:
        base = 100.0 if self._stamina is None else self._stamina
        return max(0.0, min(100.0, base - self.stamina_drop_per_min() * minutes_ahead))

    def zone(self, they_have_ball: bool) -> tuple[float, float] | None:
        if they_have_ball:
            if self._att_n == 0:
                return None
            return (self._att_x / self._att_n, self._att_y / self._att_n)
        if self._def_n == 0:
            return None
        return (self._def_x / self._def_n, self._def_y / self._def_n)

    def behaviour(self) -> str:
        if not self._actions:
            return "unknown"
        # tie broken on the action name so the label never depends on the
        # order actions happened to be first seen in
        top = max(self._actions.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return _BEHAVIOUR.get(top, top.lower())

    def marks(self) -> tuple[str | None, float]:
        """Which of our players they shadow, and how much of the time."""
        if not self._marks or self._mark_n == 0:
            return (None, 0.0)
        pid, n = max(self._marks.items(), key=lambda kv: (kv[1], kv[0]))
        return (pid, n / self._mark_n)

    def keeper(self) -> dict | None:
        if self._gk_n == 0:
            return None
        return {
            "off_line_m": self._gk_depth / self._gk_n,
            "side_bias_m": self._gk_offset / self._gk_n,
        }

    def snapshot(self) -> dict:
        """Raw state, for equality checks. Unrounded on purpose."""
        return {
            "id": self.id,
            "number": self.number,
            "role": self.role,
            "ticks": self.ticks,
            "top": list(self._top),
            "speed_n": self._speed_n,
            "prev_vel": list(self._prev_vel),
            "react": list(self._react),
            "stamina": self._stamina,
            "stamina_t0": self._stamina_t0,
            "stamina_s0": self._stamina_s0,
            "stamina_span": self._stamina_span,
            "def": [self._def_n, self._def_x, self._def_y],
            "att": [self._att_n, self._att_x, self._att_y],
            "actions": dict(sorted(self._actions.items())),
            "marks": dict(sorted(self._marks.items())),
            "mark_n": self._mark_n,
            "gk": [self._gk_n, self._gk_offset, self._gk_depth],
        }


# ------------------------------------------------------------------- scout


class Scout:
    """The opponent model. One per process; `observe` once per tick.

    Call `observe(obs)` with the same observation the policy gets, then read
    `top_speed`, `reaction_delay`, `profile`, `team` and `summary`.
    """

    def __init__(self, params: ScoutParams = DEFAULT_SCOUT) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        p = self.p
        self._opponents: dict[str, _OpponentModel] = {}
        self._last_tick: int | None = None
        self._ticks = 0
        self._malformed = 0

        # pitch geometry, learned from the first observation; the standard
        # 110x70 is only the value used before any tick has arrived
        self._length = 110.0
        self._width = 70.0

        self._clock = 0.0
        self._prev_clock: float | None = None
        self._prev_tick: int | None = None
        self._dt: deque[float] = deque(maxlen=p.tick_seconds_window)
        self._tick_seconds = p.tick_seconds_prior

        # ball state carried between ticks, for release detection
        self._prev_owner: str | None = None
        self._prev_ball_speed = 0.0
        # at most one live pass event, because there is only one ball
        self._event: dict | None = None

        # team aggregates
        self._press_n = 0
        self._press_sum = 0.0
        self._line_n = 0
        self._line_sum = 0.0
        self._compact_n = 0
        self._compact_sum = 0.0

        # how often each of our players is the nearest to somebody
        self._marked: dict[str, int] = {}

    # ------------------------------------------------------------ observe

    def observe(self, obs: dict) -> None:
        """Fold one tick into the model. Idempotent, keyed on `tick`.

        A repeat of a tick already folded in is a no-op, and a tick older than
        the newest one seen is dropped rather than applied out of order — both
        are needed because the platform may redeliver, and because an agent
        that times out simply never sees some ticks.
        """
        try:
            self._observe(obs)
        except Exception:      # noqa: BLE001 - a bad tick must never kill a decision
            self._malformed += 1

    def _observe(self, obs: dict) -> None:
        p = self.p
        if not isinstance(obs, dict):
            self._malformed += 1
            return

        tick = obs.get("tick")
        if not isinstance(tick, int) or isinstance(tick, bool):
            try:
                tick = int(tick)
            except (TypeError, ValueError):
                self._malformed += 1
                return

        if self._last_tick is not None:
            if tick <= p.restart_tick and self._ticks > p.restart_min_ticks:
                self.reset()          # a new match in a reused process
            elif tick <= self._last_tick:
                return  # replay or straggler: the model must not move

        # Claimed before any mutation, so a tick that blows up half way is
        # never re-applied by a retry.
        self._last_tick = tick
        self._ticks += 1

        self._advance_clock(tick, obs.get("game_time"))

        pitch = obs.get("pitch") or {}
        length = _f(pitch.get("length"), 110.0) or 110.0
        width = _f(pitch.get("width"), 70.0) or 70.0
        their_goal = self._goal(pitch, "opponent_goal", (length, width / 2.0))
        self._length = length
        self._width = width

        possession = obs.get("possession")
        they_have_ball = possession == "opponent"

        ball = obs.get("ball") or {}
        ball_pos = _xy(ball.get("position"))
        ball_vel = _xy(ball.get("velocity"))
        ball_speed = math.hypot(ball_vel[0], ball_vel[1])
        owner = ball.get("owner_id")
        if not isinstance(owner, str):
            owner = None

        opponents = obs.get("opponents")
        if not isinstance(opponents, list):
            opponents = []

        ours = self._our_players(obs)

        # 1. per-opponent signals, and the reaction check against the event
        #    that was already live before this tick
        outfield: list[tuple[float, float]] = []
        for o in opponents:
            if not isinstance(o, dict):
                continue
            oid = o.get("id")
            if not isinstance(oid, str) or not oid:
                continue
            m = self._opponents.get(oid)
            if m is None:
                m = _OpponentModel(oid, p)          # arrived mid-match: fine
                self._opponents[oid] = m
            m.ticks += 1

            num = o.get("number")
            if isinstance(num, int) and not isinstance(num, bool):
                m.number = num
            role = o.get("role")
            if isinstance(role, str) and role:
                m.role = role

            pos = _xy(o.get("position"))
            vel = _xy(o.get("velocity"))
            m.see_position(pos, they_have_ball)
            m.see_speed(math.hypot(vel[0], vel[1]))

            stamina = o.get("stamina")
            if stamina is not None:
                s = _f(stamina, -1.0)
                if 0.0 <= s <= 100.0:
                    m.see_stamina(s, self._clock)

            action = o.get("last_action")
            if isinstance(action, str) and action:
                m.see_action(action)

            if m.role == "GK":
                off, depth = self._keeper_geometry(pos, ball_pos, their_goal)
                m.see_keeper(off, depth)
            else:
                outfield.append(pos)
                # marking only means something while they are defending
                if possession in ("you", "teammate") and ours:
                    near = min(ours, key=lambda q: (_dist(q[1], pos), q[0]))
                    m.see_mark(near[0])
                    self._marked[near[0]] = self._marked.get(near[0], 0) + 1

            self._check_reaction(tick, oid, pos, vel)
            m._prev_vel = vel      # after the check: it compares against it

        # 2. team shape
        self._update_team(outfield, ball_pos, possession)

        # 3. a new release replaces the live event (one ball, one event)
        self._detect_release(tick, owner, ball_pos, ball_vel, ball_speed, opponents)

        self._prev_owner = owner
        self._prev_ball_speed = ball_speed

    # ---------------------------------------------------------- internals

    @staticmethod
    def _goal(pitch: dict, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
        spec = pitch.get(key)
        if isinstance(spec, dict) and spec.get("center") is not None:
            return _xy(spec["center"])
        return fallback

    @staticmethod
    def _our_players(obs: dict) -> list[tuple[str, tuple[float, float]]]:
        """(id, position) for us and our teammates, in a stable order."""
        out: list[tuple[str, tuple[float, float]]] = []
        me = obs.get("you")
        if isinstance(me, dict) and isinstance(me.get("id"), str):
            out.append((me["id"], _xy(me.get("position"))))
        mates = obs.get("teammates")
        if isinstance(mates, list):
            for t in mates:
                if isinstance(t, dict) and isinstance(t.get("id"), str):
                    out.append((t["id"], _xy(t.get("position"))))
        return out

    def _advance_clock(self, tick: int, game_time) -> None:
        """Keep a match clock, and learn the real tick interval from it.

        The interval is the unit reaction latency is measured in, and it is not
        documented anywhere, so it is measured too — median of the last few
        per-tick deltas, falling back to the prior until there are any.
        """
        t = _f(game_time, -1.0)
        if t < 0.0:
            t = tick * self._tick_seconds
        if self._prev_clock is not None and self._prev_tick is not None:
            dtick = tick - self._prev_tick
            dt = t - self._prev_clock
            if dtick > 0 and dt > 0.0:
                self._dt.append(dt / dtick)
                self._tick_seconds = _median(list(self._dt))
        self._clock = t
        self._prev_clock = t
        self._prev_tick = tick

    def _detect_release(
        self, tick: int, owner: str | None,
        ball_pos: tuple[float, float], ball_vel: tuple[float, float],
        ball_speed: float, opponents: list,
    ) -> None:
        """Note the moment the ball starts travelling down a lane.

        Two signatures: it left somebody's feet, or it went from slow to fast.
        Either way we project the lane a fixed distance ahead and remember who
        was close enough to it to plausibly be reacting to it.
        """
        p = self.p
        if ball_speed < p.ball_moving_speed:
            return
        left_feet = self._prev_owner is not None and owner != self._prev_owner
        sped_up = self._prev_ball_speed < p.ball_moving_speed
        if not (left_feet or sped_up):
            return

        n = math.hypot(ball_vel[0], ball_vel[1]) or 1.0
        aim = (
            ball_pos[0] + ball_vel[0] / n * p.react_lane_horizon,
            ball_pos[1] + ball_vel[1] / n * p.react_lane_horizon,
        )
        pending: list[str] = []
        for o in opponents:
            if not isinstance(o, dict):
                continue
            oid = o.get("id")
            if not isinstance(oid, str) or oid == self._prev_owner:
                continue  # whoever just kicked it is not reacting to it
            if _dist(_xy(o.get("position")), aim) <= p.react_interest_radius:
                pending.append(oid)
        pending.sort()
        self._event = {"tick": tick, "aim": aim, "pending": pending}

    def _check_reaction(
        self, tick: int, oid: str,
        pos: tuple[float, float], vel: tuple[float, float],
    ) -> None:
        """Has this opponent *turned* toward the live lane yet?

        Turned, not merely pointed: a player already running that way before
        the ball moved is anticipating, not reacting, and counting them would
        both understate their latency and let a wandering opponent score a
        reaction by coincidence. So the previous tick's heading must have been
        outside the cone and this one inside it.

        Resolution is one tick, so the honest point estimate is the midpoint of
        the interval the turn first showed up in: (ticks - 0.5) * interval.
        An opponent that never turns records nothing rather than a slow time —
        not reacting usually means not involved.
        """
        ev = self._event
        if ev is None:
            return
        delta = tick - ev["tick"]
        if delta < 1 or delta > self.p.react_window_ticks:
            return
        pending = ev["pending"]
        if oid not in pending:
            return

        speed = math.hypot(vel[0], vel[1])
        if speed < self.p.react_min_speed:
            return
        aim = ev["aim"]
        dx, dy = aim[0] - pos[0], aim[1] - pos[1]
        dn = math.hypot(dx, dy)
        if dn < 1e-6:
            return
        if (vel[0] * dx + vel[1] * dy) / (speed * dn) < self.p.react_align_cos:
            return

        pending.remove(oid)
        m = self._opponents.get(oid)
        if m is None:
            return
        pv = m._prev_vel
        pn = math.hypot(pv[0], pv[1])
        if pn > 1e-6 and (pv[0] * dx + pv[1] * dy) / (pn * dn) >= self.p.react_align_cos:
            return   # already heading there: this event says nothing about them
        m.record_reaction((delta - 0.5) * self._tick_seconds)

    def _update_team(
        self, outfield: list[tuple[float, float]],
        ball_pos: tuple[float, float], possession,
    ) -> None:
        if not outfield:
            return
        # compactness: mean distance from their own centroid. Meaningless for a
        # single player, so it is only sampled once there are two.
        n = len(outfield)
        if n >= 2:
            cx = sum(q[0] for q in outfield) / n
            cy = sum(q[1] for q in outfield) / n
            self._compact_n += 1
            self._compact_sum += sum(math.hypot(q[0] - cx, q[1] - cy) for q in outfield) / n

        if possession not in ("you", "teammate"):
            return  # shape only reads as "defensive" when they are defending
        # their goal is at +x, so their last outfielder is the highest x — the
        # offside line, and the amount of grass behind it
        self._line_n += 1
        self._line_sum += max(q[0] for q in outfield)
        self._press_n += 1
        near = sum(1 for q in outfield if _dist(q, ball_pos) <= self.p.press_radius)
        self._press_sum += near / len(outfield)

    @staticmethod
    def _keeper_geometry(
        gk: tuple[float, float], ball: tuple[float, float], goal: tuple[float, float],
    ) -> tuple[float, float]:
        """Signed offset from the ball-goal line, and distance off the line.

        A keeper holding the angle sits on the ball-goal segment; the offset is
        how far they cheat off it, positive toward the high-y side when the
        shot runs roughly along +x. That sign is exactly what tells us which
        corner to aim at.
        """
        dx, dy = goal[0] - ball[0], goal[1] - ball[1]
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return (0.0, _dist(gk, goal))
        ux, uy = dx / n, dy / n
        gx, gy = gk[0] - ball[0], gk[1] - ball[1]
        offset = -uy * gx + ux * gy        # left-normal component
        return (offset, _dist(gk, goal))

    def _model(self, opponent_id: str) -> _OpponentModel | None:
        return self._opponents.get(opponent_id) if isinstance(opponent_id, str) else None

    # --------------------------------------------------------------- reads

    def top_speed(self, opponent_id: str) -> float:
        """Calibrated V_TOP in m/s. Always a number, prior-backed when new."""
        m = self._model(opponent_id)
        return self.p.speed_prior if m is None else m.top_speed()

    def effective_top_speed(self, opponent_id: str) -> float:
        """V_TOP discounted for how tired they are right now."""
        m = self._model(opponent_id)
        return self.p.speed_prior if m is None else m.effective_top_speed()

    def reaction_delay(self, opponent_id: str) -> float:
        """Calibrated tau_react in seconds. Always a number."""
        m = self._model(opponent_id)
        return self.p.react_prior if m is None else m.reaction_delay()

    def confidence(self, opponent_id: str) -> dict:
        """Sample counts and 0-1 confidence behind the two calibrated numbers."""
        m = self._model(opponent_id)
        if m is None:
            return {"speed": 0.0, "reaction": 0.0,
                    "speed_samples": 0, "reaction_samples": 0, "ticks_seen": 0}
        return m.confidence()

    def known_opponents(self) -> list[str]:
        return sorted(self._opponents)

    def profile(self, opponent_id: str) -> dict:
        """The compact per-opponent picture. Prior-only for an unknown id."""
        m = self._model(opponent_id)
        if m is None:
            return {
                "id": opponent_id, "known": False,
                "top_speed": round(self.p.speed_prior, 2),
                "effective_top_speed": round(self.p.speed_prior, 2),
                "reaction_delay": round(self.p.react_prior, 2),
                "confidence": self.confidence(opponent_id),
            }
        mark_id, mark_share = m.marks()
        defend = m.zone(False)
        attack = m.zone(True)
        out = {
            "id": m.id, "known": True, "number": m.number, "role": m.role,
            "top_speed": round(m.top_speed(), 2),
            "effective_top_speed": round(m.effective_top_speed(), 2),
            "reaction_delay": round(m.reaction_delay(), 2),
            "confidence": m.confidence(),
            "behaviour": m.behaviour(),
            "actions": dict(sorted(m._actions.items(), key=lambda kv: (-kv[1], kv[0]))[:3]),
            "zone_defending": None if defend is None else (round(defend[0], 1), round(defend[1], 1)),
            "zone_attacking": None if attack is None else (round(attack[0], 1), round(attack[1], 1)),
            "marks": mark_id,
            "mark_share": round(mark_share, 2),
            "stamina": None if m._stamina is None else round(m._stamina, 1),
            "stamina_drop_per_min": round(m.stamina_drop_per_min(), 2),
            "tiring": m._stamina is not None and m._stamina <= self.p.tired_stamina,
        }
        gk = m.keeper()
        if gk is not None:
            out["keeper"] = {"off_line_m": round(gk["off_line_m"], 2),
                             "side_bias_m": round(gk["side_bias_m"], 2)}
        return out

    def team(self) -> dict:
        """Press intensity, defensive line and compactness for the whole side."""
        return {
            "ticks": self._ticks,
            "press_intensity": round(self._press_sum / self._press_n, 3) if self._press_n else 0.0,
            "line_x": round(self._line_sum / self._line_n, 1) if self._line_n else None,
            "compactness_m": round(self._compact_sum / self._compact_n, 1) if self._compact_n else None,
            "tick_seconds": round(self._tick_seconds, 2),
            "malformed_ticks": self._malformed,
        }

    def free_teammate(self) -> str | None:
        """Whichever of ours the fewest opponents shadow — likeliest to be free."""
        if not self._marked:
            return None
        return min(self._marked.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # -------------------------------------------------------------- prompt

    def summary(self) -> dict:
        """A handful of short sentences for the model's prompt. Hard-capped.

        Deliberately not numbers. This goes into a latency-critical path where
        every token is paid for twice, and a small model acts on "their #3
        presses hard and is tiring" far better than on a float it has to
        interpret. The geometry stays in `profile()` for the policy to consume.
        """
        p = self.p
        out: dict = {"ticks_watched": self._ticks}
        if self._ticks == 0 or not self._opponents:
            return out

        ranked = sorted(
            self._opponents.values(),
            key=lambda m: (999 if m.number is None else m.number, m.id),
        )
        notes: list[str] = []
        keeper_note: str | None = None
        for m in ranked:
            if m.role == "GK":
                if keeper_note is None:
                    keeper_note = self._keeper_note(m)
                continue
            if len(notes) < p.summary_max_notes:
                notes.append(_clip(self._opponent_note(m), p.summary_note_chars))
        if notes:
            out["opponents"] = notes
        shape = self._shape_notes()
        if shape:
            out["shape"] = shape
        if keeper_note:
            out["their_keeper"] = _clip(keeper_note, p.summary_note_chars)
        free = self.free_teammate()
        if free:
            out["least_marked"] = free
        return out

    def _opponent_note(self, m: _OpponentModel) -> str:
        who = f"#{m.number}" if m.number is not None else m.id
        bits: list[str] = []

        behaviour = m.behaviour()
        if behaviour == "presser":
            bits.append("presses hard")
        elif behaviour == "marker":
            bits.append("man-marks")
        elif behaviour == "carrier":
            bits.append("wants the ball")
        elif behaviour == "holder":
            bits.append("holds position")

        zone = m.zone(False)
        if zone is not None:
            # their goal is at +x, so a low mean x means they never come back
            frac = zone[0] / max(1.0, self._length)
            if frac < 0.35:
                bits.append("stays high, does not track back")
            elif frac > 0.72:
                bits.append("defends deep")

        if m._stamina is not None and m._stamina <= self.p.tired_stamina:
            bits.append(f"tiring ({m._stamina:.0f}%)")
        elif m.stamina_drop_per_min() > 6.0:
            bits.append("fading fast")

        mark_id, share = m.marks()
        if mark_id and share >= 0.5:
            bits.append(f"usually on {mark_id}")

        if not bits:
            bits.append("no read yet")
        return f"{who} " + ", ".join(bits[:3])

    def _keeper_note(self, m: _OpponentModel) -> str:
        gk = m.keeper()
        if gk is None:
            return ""
        bits = []
        if gk["off_line_m"] > self.p.gk_off_line_m:
            bits.append(f"comes {gk['off_line_m']:.0f}m off his line")
        else:
            bits.append("stays on his line")
        bias = gk["side_bias_m"]
        if bias > self.p.gk_bias_m:
            bits.append("cheats to the high-y side, aim low-y")
        elif bias < -self.p.gk_bias_m:
            bits.append("cheats to the low-y side, aim high-y")
        return ", ".join(bits)   # the "their_keeper" key already says who

    def _shape_notes(self) -> list[str]:
        t = self.team()
        notes: list[str] = []
        if t["line_x"] is not None:
            behind = self._length - t["line_x"]
            if behind > self._length * 0.28:
                notes.append(f"high line, {behind:.0f}m of space behind them")
            else:
                notes.append(f"deep block, only {behind:.0f}m behind them")
        if t["compactness_m"] is not None:
            notes.append(
                f"compact ({t['compactness_m']:.0f}m)" if t["compactness_m"] < 12.0
                else f"stretched ({t['compactness_m']:.0f}m)"
            )
        if self._press_n:
            pi = t["press_intensity"]
            notes.append("they swarm the ball" if pi > 0.4
                         else "they press lightly" if pi > 0.15
                         else "they sit off the ball")
        return notes

    # ------------------------------------------------------------ testing

    def snapshot(self) -> dict:
        """Complete model state, key-sorted, for byte-comparing two Scouts.

        The determinism guarantee this design rests on is exactly the claim
        that two Scouts fed the same ticks return equal snapshots.
        """
        ev = self._event
        return {
            "ticks": self._ticks,
            "last_tick": self._last_tick,
            "malformed": self._malformed,
            "clock": self._clock,
            "prev_clock": self._prev_clock,
            "prev_tick": self._prev_tick,
            "dt": list(self._dt),
            "tick_seconds": self._tick_seconds,
            "prev_owner": self._prev_owner,
            "prev_ball_speed": self._prev_ball_speed,
            "event": None if ev is None else {
                "tick": ev["tick"], "aim": list(ev["aim"]),
                "pending": sorted(ev["pending"]),
            },
            "press": [self._press_n, self._press_sum],
            "line": [self._line_n, self._line_sum],
            "compact": [self._compact_n, self._compact_sum],
            "marked": dict(sorted(self._marked.items())),
            "length": self._length,
            "opponents": {k: self._opponents[k].snapshot() for k in sorted(self._opponents)},
        }
