"""Pass selection by time-to-intercept, with stamina in the loop.

Why this exists
---------------
`policy._on_ball` scores a pass from static geometry: the perpendicular
distance from the lane to the *nearest* opponent. That has three failures this
module removes.

1.  It has no clock. A defender 8 m off the lane is scored the same whether
    they are fresh and already sprinting into it, or spent and facing the wrong
    way. On a 110 x 70 pitch shared by five players each covers ~2.2x the
    ground a real footballer does, so "can they get there before the ball" is
    the whole question, and a perpendicular distance barely asks it.
2.  It takes a MAX over opponents (nearest wins), so two defenders each
    half-covering a lane score exactly as well as one. They do not: the lane
    closes like (1 - p1)(1 - p2). This module multiplies.
3.  It cannot tell the three pass types apart, so it cannot notice that the
    ball can be lifted *over* the defender who is closing the lane.

The model, per candidate receiver:

    lead the receiver (they are moving)
      -> ball arrival time along the lane, per pass type, with rolling friction
      -> per-opponent time to each point of the lane: reaction, turn cost, and
         a top speed AND acceleration both scaled by stamina
      -> margin(s) = spare time the ball has at arc-length s
      -> a smooth interception probability from the worst margin
      -> PRODUCT across opponents
      -> a contest discount for the swarm around the receiver on arrival
      -> value, so a safe backwards pass loses to a slightly riskier forward one

UNITS AND CALIBRATION
---------------------
The Cup pitch is 110 x 70 and a real pitch is 105 x 68, so one pitch unit is
taken to be about one metre and the times below are seconds. Every physics
constant here is a PLACEHOLDER: the workshop publishes no engine constants, so
each default is a defensible *football* number rather than a measured *engine*
number, and each is a field on `PassParams` so it can be swept by name once
real match data exists. Nothing is hardcoded inline.

SIGN CONVENTION
---------------
`margin` is spare time *for the ball*: `t_opp(s) - t_ball(s)`. Positive means
the ball is through before the defender arrives; negative means the defender is
standing there waiting for it. The worst (minimum) margin over the sampled lane
then feeds `p_intercept = sigmoid(-margin / k)` with the intended sense, and
the smoothness means a near-miss degrades gracefully instead of flipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from policy import _dist, _lane, _point_to_segment

#: The three flight profiles the platform exposes. `PASS` takes no power or
#: speed parameter — the engine sets speed from the type — so this is a closed
#: set of three, not a continuous parameter to optimise over.
PASS_TYPES: tuple[str, ...] = ("GROUND", "THROUGH", "AERIAL")

_EPS = 1e-9
_BIG = 1e6

#: `worst_margin` when an opponent has no interception window at all — every
#: sampled point of an AERIAL flight was over their head.
NO_WINDOW = math.inf


@dataclass(frozen=True)
class PassParams:
    """Every constant the model uses. All defaults are PLACEHOLDERS.

    None of these came from the engine — the workshop publishes no ball speed,
    no friction, no player top speed. They are football-plausible values in
    metres and seconds, chosen so the *ratios* that actually drive the decision
    (ball time vs defender time) land in a sane range on a 110 x 70 pitch.
    Calibrate them against practice matches before believing any single one.
    """

    # --- ball flight, per pass type -----------------------------------------
    # Rolling friction makes time-to-arc-length super-linear, which is the
    # point: a long pass gives a defender disproportionately more time.
    #   s(t) = v0.t - a.t^2/2   =>   t(s) = (v0 - sqrt(v0^2 - 2.a.s)) / a
    ground_speed: float = 16.0        # PLACEHOLDER m/s off the foot: firm, driven, keeps possession
    ground_decel: float = 1.60        # PLACEHOLDER m/s^2 rolling on turf; range = v0^2/2a = 80 m
    through_speed: float = 18.0       # PLACEHOLDER: struck harder, it has to beat a defensive line
    through_decel: float = 1.40       # PLACEHOLDER: hit flatter, so it holds its pace a little better
    aerial_speed: float = 20.0        # PLACEHOLDER: a lofted switch leaves the boot fastest
    aerial_decel: float = 0.35        # PLACEHOLDER: air drag only, no turf — hence the long range
    stall_speed: float = 1.50         # PLACEHOLDER: crawl speed past max range, so an under-hit
                                      # pass degrades smoothly instead of becoming unreachable

    # --- passer stamina: power and accuracy ---------------------------------
    power_floor: float = 0.78         # PLACEHOLDER: an exhausted passer still strikes it at 78%
    power_exp: float = 0.80           # PLACEHOLDER: shaping of stamina -> power; <1 fades late
    aim_sigma: float = 0.030          # PLACEHOLDER rad (~1.7 deg) angular error when fresh
    aim_sigma_tired: float = 0.075    # PLACEHOLDER rad (~4.3 deg) angular error when spent
    catch_radius: float = 2.60        # PLACEHOLDER m: how far off the receiver can still collect it
    ground_sigma_mult: float = 1.00   # PLACEHOLDER: the easiest pass to place
    through_sigma_mult: float = 1.15  # PLACEHOLDER: weighting a pass into space is harder
    aerial_sigma_mult: float = 1.35   # PLACEHOLDER: a lofted ball is the hardest to place

    # --- through-ball geometry ----------------------------------------------
    through_lead: float = 6.50        # PLACEHOLDER m played ahead of the receiver's lead position
    through_goal_blend: float = 0.50  # PLACEHOLDER: 0 = purely along their run, 1 = purely at goal
    through_min_speed: float = 0.80   # PLACEHOLDER m/s below which "their run" has no direction
    pitch_margin: float = 1.50        # PLACEHOLDER m: keep the aim point inbounds

    # --- aerial height ------------------------------------------------------
    # A defender mid-lane cannot touch a ball that is over their head, which is
    # the entire reason AERIAL exists. Height is a parabola over the flight:
    #   h(u) = 4.H.u.(1-u),  u = s / L
    aerial_apex_ratio: float = 0.16   # PLACEHOLDER: apex as a fraction of pass length
    aerial_apex_min: float = 1.20     # PLACEHOLDER m: even a short chip leaves the floor
    aerial_apex_max: float = 6.00     # PLACEHOLDER m: the engine surely caps the trajectory
    reach_height: float = 2.30        # PLACEHOLDER m: standing head ~1.9, a jumping header ~2.5

    # --- opponent movement --------------------------------------------------
    # THIS BLOCK IS THE DOMINANT TERM. Stamina scales top speed *and*
    # acceleration, and over the short distances a lane interception actually
    # needs, acceleration matters more than top speed.
    v_top: float = 7.50               # PLACEHOLDER m/s sprint at full stamina
    accel: float = 4.00               # PLACEHOLDER m/s^2 from standing
    stamina_floor: float = 0.45       # PLACEHOLDER: a spent player still moves at 45% of pace
    stamina_exp: float = 0.60         # PLACEHOLDER: shaping of stamina -> pace
    react_base: float = 0.25          # PLACEHOLDER s to read the pass and change direction
    react_committed: float = 0.04     # PLACEHOLDER s when already going for the ball
    react_fatigue: float = 0.60       # PLACEHOLDER: extra reaction lag when fully spent
    react_max: float = 0.80           # PLACEHOLDER s cap: nobody is asleep for a whole second
    committed_actions: tuple[str, ...] = ("INTERCEPT", "PRESS_BALL", "TACKLE")
    turn_cost: float = 0.35           # PLACEHOLDER s to stop and turn when running fully away
    turn_probe_dt: float = 0.10       # s used to measure closing rate on the lane; not physics
    intercept_reach: float = 0.90     # PLACEHOLDER m: a leg stuck out is an interception too

    # --- lane sampling and the margin sigmoid -------------------------------
    samples: int = 11                 # PLACEHOLDER: 11 points resolve a lane to ~10% of its length
    sample_from: float = 0.06         # PLACEHOLDER: start just off the passer's boot
    margin_k: float = 0.25            # PLACEHOLDER s: sigmoid width; +-0.25 s spans most of 0..1
    meet_k: float = 0.35              # PLACEHOLDER s: wider — the receiver adjusts their own run

    # --- contest on arrival -------------------------------------------------
    contest_d0: float = 1.60          # PLACEHOLDER m: a 50/50 when a defender is this close on arrival
    contest_k: float = 1.10           # PLACEHOLDER m: softness of that 50/50
    ground_control: float = 1.00      # PLACEHOLDER: a rolling ball is the easiest first touch
    through_control: float = 0.94     # PLACEHOLDER: running onto it, so slightly less certain
    aerial_control: float = 0.86      # PLACEHOLDER: bringing a dropping ball down costs control

    # --- value: safety is not the only thing that matters -------------------
    w_keep: float = 0.55              # PLACEHOLDER: worth of simply retaining the ball
    w_prog: float = 0.85              # PLACEHOLDER: worth of progress toward x = 110
    w_shot: float = 0.70              # PLACEHOLDER: worth of the receiver's onward shot
    w_dist: float = 0.10              # PLACEHOLDER: mild preference for the economical pass
    # Progress saturates at gain_ref, so it must sit ABOVE the usual pass
    # range or the term stops discriminating: at 25 m (23% of this pitch) a
    # 25 m advance and a 35 m advance score identically, which is most of the
    # forward passes actually on offer here. A third of the pitch keeps it live.
    gain_ref: float = 35.0            # PLACEHOLDER m of progress that scores a full point
    dist_ref: float = 45.0            # PLACEHOLDER m of pass length that scores a full penalty
    shot_near: float = 6.0            # PLACEHOLDER m: inside this, shot quality is already maximal
    shot_scale: float = 12.0          # PLACEHOLDER m: e-folding distance of shot quality
    shot_lane_ref: float = 3.0        # PLACEHOLDER m of clearance for a fully open shooting lane
    value_floor: float = 0.05         # keeps score monotone in p_success for backwards passes
    # Under SAFEST, value must only break a near-tie. `value` spans roughly
    # 0.05..2.1 with the weights above, so this bounds the swing it can cause to
    # about 6%: any option with a clearer safety edge than that wins outright,
    # and two near-equally safe options are settled by which one goes forward.
    safest_value_w: float = 0.03      # PLACEHOLDER, but see the bound above

    # --- candidate selection ------------------------------------------------
    include_gk: bool = False          # mirrors policy: the keeper is not an outlet by default

    # --- measured opponent estimates ----------------------------------------
    # `v_top` and `react_base` above are the two constants this model most
    # wants and least knows, and `scouting.Scout` measures both per opponent
    # over the ticks of a match. Rather than import it (and couple two modules
    # that are useful separately), an opponent dict may simply carry them:
    #
    #     o["scout"] = {"top_speed": <effective_top_speed()>,   # m/s, stamina-adjusted
    #                   "reaction":  <reaction_delay()>}        # seconds
    #
    # Both keys are optional and each falls back to the placeholder on its own.
    # Set this False to A/B whether the measured numbers actually help.
    use_scout: bool = True
    scout_key: str = "scout"


DEFAULT = PassParams()
DEFAULT_PASS = DEFAULT  # explicit alias, so importers need not shadow policy.DEFAULT


@dataclass(frozen=True)
class PassOption:
    """One scored (receiver, pass type) pair. Ready to log or hand to a model."""

    receiver_id: str                            # passed through from the observation, uncoerced
    receiver_number: int | None
    type: str                                   # GROUND | THROUGH | AERIAL
    p_success: float
    p_lane: float                               # product over opponents of (1 - p_intercept)
    p_control: float                            # survives the swarm around the receiver
    p_delivery: float                           # lands inside the receiver's control radius
    p_meet: float                               # receiver actually reaches the aim point in time
    worst_margin: float                         # seconds, min over opponents; < 0 = beaten to it
    worst_opponent_id: str | None
    margins: tuple[tuple[str, float], ...]      # per-opponent worst margin, seconds
    forward_gain: float                         # metres of progress toward x = 110
    distance: float                             # length of the pass actually played
    flight_time: float                          # seconds from boot to aim point
    target: tuple[float, float]                 # where the ball is aimed (led / through point)
    score: float
    reason: str


# --------------------------------------------------------------------- helpers

def _num(v, fallback: float) -> float:
    """Any of None / missing / a string / NaN becomes the fallback."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    return fallback if f != f else f


def _xy(v, fallback: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    try:
        return float(v[0]), float(v[1])
    except (TypeError, IndexError, ValueError, KeyError):
        return fallback


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _sigmoid(x: float) -> float:
    """Logistic, saturating instead of overflowing on an extreme margin."""
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _pace(stamina, p: PassParams) -> float:
    """Stamina -> fraction of full pace, in [stamina_floor, 1].

    Applied to BOTH top speed and acceleration. Scaling acceleration is what
    makes stamina bite: a lane interception is a 3-5 m burst, and over that
    distance nobody reaches top speed at all, so a model that only lowered the
    tired player's ceiling would show almost no difference between fresh and
    spent — which is exactly the wrong answer.
    """
    frac = _clamp(_num(stamina, 100.0) / 100.0, 0.0, 1.0)
    return p.stamina_floor + (1.0 - p.stamina_floor) * (frac ** p.stamina_exp)


def _t_flight(s: float, v0: float, decel: float, stall: float) -> float:
    """Time for the ball to travel arc-length `s` under constant deceleration.

    Beyond max range the ball has stopped, so the remainder is charged at a
    crawl speed: continuous at the boundary, monotone, and it makes an
    over-long pass lose on time rather than raise or silently look fine.
    """
    if s <= 0.0:
        return 0.0
    v0 = max(v0, _EPS)
    if decel <= _EPS:
        return s / v0
    s_max = v0 * v0 / (2.0 * decel)
    if s < s_max:
        disc = v0 * v0 - 2.0 * decel * s
        return (v0 - math.sqrt(disc if disc > 0.0 else 0.0)) / decel
    return v0 / decel + (s - s_max) / max(stall, _EPS)


def _t_run(d: float, u: float, v_max: float, accel: float) -> float:
    """Time to cover `d`, starting at speed `u` and accelerating to `v_max`.

    The initial speed matters and is not a refinement: a receiver sprinting
    onto a through ball does NOT start from a standing start, and charging them
    a full acceleration ramp they have already paid roughly doubles their time
    to the ball. With u = 0 this reduces exactly to sqrt(2d/a) capped at v_max,
    which is what the opponent loop uses inline.
    """
    if d <= 0.0:
        return 0.0
    if v_max <= _EPS:
        return _BIG
    u = _clamp(u, 0.0, v_max)
    if accel <= _EPS:                              # already at pace, no ramp
        return d / v_max
    d_ramp = (v_max * v_max - u * u) / (2.0 * accel)
    if d <= d_ramp:
        return (math.sqrt(u * u + 2.0 * accel * d) - u) / accel
    return (v_max - u) / accel + (d - d_ramp) / v_max


def _unit(dx: float, dy: float, fallback: tuple[float, float] = (1.0, 0.0)) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < _EPS:
        return fallback
    return dx / n, dy / n


def _prepare_opponent(o: dict, p: PassParams) -> tuple:
    """Everything about one opponent that does not depend on the lane.

    Flattened to a tuple because it is read in the innermost loop, where
    attribute lookups are the difference between 300 us and 900 us.

    Returns (id, x', y', v_max, accel, ramp_d, ramp_t, tau, vx, vy).
    `(x', y')` is the effective start `O + v.tau`: they keep moving while they
    read the pass, which for a sprinter is most of a metre.
    """
    pace = _pace(o.get("stamina"), p)

    # Measured estimates beat placeholders when the caller has them (see
    # `use_scout`). `top_speed` from Scout is ALREADY stamina-adjusted, so it
    # replaces `v_top * pace` outright rather than being scaled again —
    # double-counting stamina there would be a silent, systematic error.
    scout = o.get(p.scout_key) if p.use_scout else None
    scout = scout if isinstance(scout, dict) else {}
    measured_v = _num(scout.get("top_speed"), 0.0)
    v_max = max(measured_v if measured_v > 0.0 else p.v_top * pace, _EPS)

    action = o.get("last_action")
    committed = isinstance(action, str) and action.strip().upper() in p.committed_actions
    measured_r = _num(scout.get("reaction"), -1.0)
    base_react = measured_r if measured_r >= 0.0 else p.react_base
    tau = p.react_committed if committed else base_react
    tau *= 1.0 + p.react_fatigue * (1.0 - pace)   # fatigue slows the read, not just the legs
    tau = _clamp(tau, 0.0, p.react_max)

    ox, oy = _xy(o.get("position"))
    vx, vy = _xy(o.get("velocity"))

    if bool(o.get("is_sprinting")):               # already at pace: no acceleration ramp
        accel, ramp_d, ramp_t = 0.0, 0.0, 0.0
    else:
        accel = max(p.accel * pace, _EPS)
        ramp_d = v_max * v_max / (2.0 * accel)
        ramp_t = v_max / accel

    return (o.get("id"), ox + vx * tau, oy + vy * tau,
            v_max, accel, ramp_d, ramp_t, tau, vx, vy)


def _turn_penalty(opp: tuple, frm: tuple[float, float], to: tuple[float, float],
                  p: PassParams) -> float:
    """Extra seconds for an opponent whose momentum carries them off the lane.

    Measured as the rate at which their lane distance is *growing*, by probing
    `_point_to_segment` a short step ahead. Using the shared helper twice beats
    re-deriving the projection here, and it is computed once per (opponent,
    lane) rather than per sample, so the inner loop stays at one hypot.
    """
    _, ox, oy, v_max, _, _, _, _, vx, vy = opp
    if abs(vx) < _EPS and abs(vy) < _EPS:
        return 0.0
    dt = max(p.turn_probe_dt, _EPS)
    d_now = _point_to_segment((ox, oy), frm, to)
    d_next = _point_to_segment((ox + vx * dt, oy + vy * dt), frm, to)
    closing = (d_now - d_next) / dt                # +ve = closing on the lane
    if closing >= 0.0:
        return 0.0
    return p.turn_cost * _clamp(-closing / v_max, 0.0, 1.0)


def _shot_quality(point: tuple[float, float], goal: tuple[float, float],
                  opp_view: list[dict], p: PassParams) -> float:
    """How good a shot the receiver would have from where the ball arrives.

    Distance decays exponentially and the lane to goal gates it linearly, so a
    receiver in a lovely spot behind a wall of bodies is not credited with a
    chance they do not have.
    """
    d_goal = _dist(point, goal)
    q_dist = math.exp(-max(0.0, d_goal - p.shot_near) / max(p.shot_scale, _EPS))
    clear = _lane(point, goal, opp_view) if opp_view else 99.0
    return q_dist * _clamp(clear / max(p.shot_lane_ref, _EPS), 0.0, 1.0)


def _flight_profile(ptype: str, p: PassParams, power: float) -> tuple[float, float, float, float]:
    """(v0, decel, sigma multiplier, control multiplier) for one pass type."""
    if ptype == "AERIAL":
        return p.aerial_speed * power, p.aerial_decel, p.aerial_sigma_mult, p.aerial_control
    if ptype == "THROUGH":
        return p.through_speed * power, p.through_decel, p.through_sigma_mult, p.through_control
    return p.ground_speed * power, p.ground_decel, p.ground_sigma_mult, p.ground_control


# ------------------------------------------------------------------- the model

def _evaluate(ptype: str, ctx: dict, mate: dict, opps: list[tuple],
              opp_view: list[dict], p: PassParams, safest: bool) -> PassOption | None:
    """Score one (receiver, pass type). None only for a degenerate lane."""
    P = ctx["passer"]
    goal = ctx["goal"]
    length = ctx["length"]
    width = ctx["width"]

    px, py = P
    R = _xy(mate.get("position"))
    rvx, rvy = _xy(mate.get("velocity"))

    v0, decel, sigma_mult, control_mult = _flight_profile(ptype, p, ctx["power"])

    # --- 1. lead the receiver. One iteration: receiver speed << ball speed, so
    # the fixed point has already converged to well inside the catch radius.
    t0 = _t_flight(_dist(P, R), v0, decel, p.stall_speed)
    lead = (_clamp(R[0] + rvx * t0, p.pitch_margin, length - p.pitch_margin),
            _clamp(R[1] + rvy * t0, p.pitch_margin, width - p.pitch_margin))

    # A THROUGH ball is aimed at the space ahead of them, not at them.
    if ptype == "THROUGH":
        gx, gy = _unit(goal[0] - lead[0], goal[1] - lead[1])
        if math.hypot(rvx, rvy) < p.through_min_speed:
            rx, ry = gx, gy                        # standing still: "their run" is toward goal
        else:
            rx, ry = _unit(rvx, rvy)
        b = _clamp(p.through_goal_blend, 0.0, 1.0)
        ux, uy = _unit(rx * (1.0 - b) + gx * b, ry * (1.0 - b) + gy * b)
        aim = (_clamp(lead[0] + ux * p.through_lead, p.pitch_margin, length - p.pitch_margin),
               _clamp(lead[1] + uy * p.through_lead, p.pitch_margin, width - p.pitch_margin))
    else:
        aim = lead

    L = _dist(P, aim)
    if L < _EPS:
        return None                                # receiver standing on the passer

    t_total = _t_flight(L, v0, decel, p.stall_speed)

    # --- 2. sample the lane: ball arrival time, and for AERIAL, whether the
    # ball is even reachable there.
    n = max(2, int(p.samples))
    dirx, diry = (aim[0] - px) / L, (aim[1] - py) / L
    u0 = _clamp(p.sample_from, 0.0, 0.99)
    step = (1.0 - u0) / (n - 1)
    apex = _clamp(p.aerial_apex_ratio * L, p.aerial_apex_min, p.aerial_apex_max) \
        if ptype == "AERIAL" else 0.0

    pts: list[tuple[float, float, float]] = []     # reachable samples: (x, y, t_ball)
    for k in range(n):
        u = u0 + step * k
        if apex > 0.0 and 4.0 * apex * u * (1.0 - u) > p.reach_height:
            continue                               # over their heads: no interception window
        s = L * u
        pts.append((px + dirx * s, py + diry * s, _t_flight(s, v0, decel, p.stall_speed)))

    # --- 3/4/5. worst margin per opponent -> p_intercept -> PRODUCT.
    # A product, not a max: two defenders each half-covering a lane really do
    # close it more than one does, which is the bug this replaces.
    inv_k = 1.0 / max(p.margin_k, _EPS)
    reach = p.intercept_reach
    hypot = math.hypot
    sqrt = math.sqrt
    p_lane = 1.0
    margins: list[tuple[str, float]] = []
    worst_margin = NO_WINDOW
    worst_id = None

    for opp in opps:
        oid, ox, oy, v_max, accel, ramp_d, ramp_t, _tau, _vx, _vy = opp
        base_t = _tau + _turn_penalty(opp, P, aim, p)
        m_o = NO_WINDOW
        for qx, qy, t_ball in pts:
            d = hypot(qx - ox, qy - oy) - reach
            if d <= 0.0:                           # already within reach of the lane
                t_opp = base_t
            elif accel <= _EPS:                    # is_sprinting: no ramp
                t_opp = base_t + d / v_max
            elif d <= ramp_d:
                t_opp = base_t + sqrt(2.0 * d / accel)
            else:
                t_opp = base_t + ramp_t + (d - ramp_d) / v_max
            m = t_opp - t_ball                     # spare time for the ball; < 0 = beaten to it
            if m < m_o:
                m_o = m
        if m_o < NO_WINDOW:                        # inf = never in reach of this flight
            p_lane *= 1.0 - _sigmoid(-m_o * inv_k)
        margins.append((oid, m_o))
        if m_o < worst_margin:
            worst_margin, worst_id = m_o, oid

    # --- 6. contest on arrival: a clean lane into a swarm still loses the ball.
    # Predicted positions, not current ones — the defender who closes on the
    # receiver during the flight is the one that decides this.
    d_contest = _BIG
    for _oid, ox, oy, _vm, _a, _rd, _rt, tau, vx, vy in opps:
        dt = t_total - tau                         # (ox, oy) is already advanced by tau
        d = hypot(ox + vx * dt - aim[0], oy + vy * dt - aim[1])
        if d < d_contest:
            d_contest = d
    p_control = _sigmoid((d_contest - p.contest_d0) / max(p.contest_k, _EPS)) * control_mult

    # --- delivery: stamina widens the aim cone and the cone widens with
    # distance, so a tired long pass misses by metres.
    miss = L * ctx["sigma"] * sigma_mult
    p_delivery = math.exp(-0.5 * (miss / max(p.catch_radius, _EPS)) ** 2)

    # --- does the receiver reach the aim point in time? Only a THROUGH ball
    # asks anything of them, but the same machinery answers for all three.
    # They arrive at `lead` still running, so credit the speed they already
    # carry along the last stretch — this is the whole point of a ball into
    # space, and charging them a standing start instead all but deletes it.
    r_pace = _pace(mate.get("stamina"), p)
    r_vmax = max(p.v_top * r_pace, _EPS)
    d_meet = _dist(lead, aim) - reach
    if d_meet > 0.0:
        mx, my = _unit(aim[0] - lead[0], aim[1] - lead[1])
        u_meet = max(0.0, rvx * mx + rvy * my)     # speed already going that way
    else:
        u_meet = 0.0
    t_meet = _t_run(d_meet, u_meet, r_vmax, max(p.accel * r_pace, _EPS))
    p_meet = _sigmoid((t_total - t_meet) / max(p.meet_k, _EPS))

    p_success = _clamp(p_lane * p_control * p_delivery * p_meet, 0.0, 1.0)

    # --- 7. value. A safe backwards pass is worthless, so safety alone must not
    # rank. The floor keeps the score monotone in p_success even when the value
    # terms go negative on a heavy pass backwards.
    forward = aim[0] - px
    value = max(p.value_floor,
                p.w_keep
                + p.w_prog * _clamp(forward / max(p.gain_ref, _EPS), -1.0, 1.0)
                + p.w_shot * _shot_quality(aim, goal, opp_view, p)
                - p.w_dist * _clamp(L / max(p.dist_ref, _EPS), 0.0, 1.0))
    score = p_success * (1.0 + p.safest_value_w * value) if safest else p_success * value

    num = mate.get("number")
    margin_txt = "clear" if worst_margin == NO_WINDOW else f"{worst_margin:+.2f}s"
    who = "" if worst_id is None else f" vs {worst_id}"
    return PassOption(
        receiver_id=mate.get("id"),
        receiver_number=num if isinstance(num, int) else None,
        type=ptype,
        p_success=p_success,
        p_lane=p_lane,
        p_control=_clamp(p_control, 0.0, 1.0),
        p_delivery=p_delivery,
        p_meet=p_meet,
        worst_margin=worst_margin,
        worst_opponent_id=worst_id,
        margins=tuple(margins),
        forward_gain=forward,
        distance=L,
        flight_time=t_total,
        target=aim,
        score=score,
        reason=(f"{ptype} #{num} {L:.0f}m lane {p_lane:.2f} "
                f"margin {margin_txt}{who} gain {forward:+.0f}m"),
    )


# --------------------------------------------------------------------- the API

def rank_passes(obs: dict, params: PassParams = DEFAULT,
                policy: str = "BEST_VALUE") -> list[PassOption]:
    """Every candidate receiver, best pass type for each, best option first.

    `policy` is "BEST_VALUE" (default: score is success x value) or "SAFEST"
    (score is success, with value only as a tiebreak — use it under pressure).
    Anything else is read as BEST_VALUE rather than raising: this runs inside a
    decision budget and must never be the reason a tick is lost.

    Pure. No I/O, no globals, and nothing in `obs` is mutated.
    """
    p = params if isinstance(params, PassParams) else DEFAULT
    safest = isinstance(policy, str) and policy.strip().upper() == "SAFEST"

    me = obs.get("you") or {}
    pitch = obs.get("pitch") or {}
    length = _num(pitch.get("length"), 110.0)
    width = _num(pitch.get("width"), 70.0)
    goal = _xy((pitch.get("opponent_goal") or {}).get("center"), (length, width * 0.5))

    # Deliberately NOT pruned by radius. Reachability already excludes the
    # irrelevant ones, and a distant, fast, well-angled opponent genuinely
    # matters — a radius filter would silently drop exactly that case.
    raw_opps = [o for o in (obs.get("opponents") or []) if isinstance(o, dict)]
    opps = [_prepare_opponent(o, p) for o in raw_opps]
    # Minimal, total view for `policy._lane`, which indexes "id" and "position"
    # directly and would raise on a malformed opponent.
    opp_view = [{"id": o.get("id"), "position": _xy(o.get("position"))} for o in raw_opps]

    frac = _clamp(_num(me.get("stamina"), 100.0) / 100.0, 0.0, 1.0) ** p.power_exp
    ctx = {
        "passer": _xy(me.get("position")),
        "goal": goal,
        "length": length,
        "width": width,
        "power": p.power_floor + (1.0 - p.power_floor) * frac,
        "sigma": p.aim_sigma + (p.aim_sigma_tired - p.aim_sigma) * (1.0 - frac),
    }

    my_id = me.get("id")
    out: list[PassOption] = []
    for mate in (obs.get("teammates") or []):
        if not isinstance(mate, dict) or mate.get("id") == my_id:
            continue
        if not p.include_gk and mate.get("role") == "GK":
            continue
        best: PassOption | None = None
        for ptype in PASS_TYPES:
            opt = _evaluate(ptype, ctx, mate, opps, opp_view, p, safest)
            if opt is not None and (best is None or opt.score > best.score):
                best = opt
        if best is not None:
            out.append(best)

    out.sort(key=lambda o: (-o.score, str(o.receiver_id)))
    return out


def best_pass(obs: dict, params: PassParams = DEFAULT,
              policy: str = "BEST_VALUE") -> PassOption | None:
    """The single best option, or None when there is no candidate receiver."""
    ranked = rank_passes(obs, params, policy)
    return ranked[0] if ranked else None
