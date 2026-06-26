#!/usr/bin/env python
"""
pdr_controller_sim.py — offline simulation of the FFN-only pdr feedback controller
(Math Agent Brief #7). Validates the controller MATH for stability/robustness before
any GPU time. Per Math: the sim canNOT prove the closed-loop K(t) matches the recorded
open-loop K(t) (changing mult changes training, hence future K). So we treat K(t) as an
EXOGENOUS gain trajectory and stress it with scenarios + noise. Goal is not perfect
replay — it's "does not oscillate, does not freeze, does not overreact."

Plant:        pdr = K(t) * mult          (linear, verified in Brief #7 §1)
Controller:   FFN-only, feedforward inversion m_ff = r/K_ema  +  small PI trim,
              log-space, no D, m in [m_min, 1.0], asymmetric rate limit, anti-windup,
              engage after warmup. (Brief #7 §5/§8, Math's spec.)

Usage:
  python tools/pdr_controller_sim.py                 # nominal + robustness matrix
  python tools/pdr_controller_sim.py --ref dn2_merge --kscen rise --noise 0.10 --plot
"""
import argparse, json, math, re, bisect, os, random

ROOT = "B:/checkpoints/current"
TOK_PER_STEP = 131072  # empirical kv2 (B=8 * T=2048 * world=8); ga ramps later -> sim is
                       # CONSERVATIVE on cadence late (more control updates/token than real).
WARMUP_STEP = 1500
CADENCE = 100          # control update every 100 steps (diagnostic cadence)


# ----------------------------- empirical data loaders -----------------------------
def _parse_jsonl(path):
    dec = json.JSONDecoder()
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        i = 0
        while i < len(s):
            while i < len(s) and s[i] in " \t\r\n":
                i += 1
            if i >= len(s):
                break
            try:
                o, e = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                break
            yield o
            i = e


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def load_kv2_Kffn(root):
    """Empirical FFN plant gain K_ffn = ffn_pdr / mult  vs tokenM, from kv2 gen_log."""
    import datetime
    pdr = []
    for line in open(os.path.join(root, "kv2/gen_log.txt"), encoding="utf-8", errors="replace"):
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[body-pdr\] pdr=[\d.e-]+ "
                      r"\(attn=[\d.e-]+ ffn=([\d.e-]+)\) \| body_lr_mult=([\d.]+)", line)
        if m:
            pdr.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                        float(m.group(2)), float(m.group(3))))
    vals = []
    for line in open(os.path.join(root, "kv2/val_log.txt"), encoding="utf-8", errors="replace"):
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| st:\s*(\d+) \| tok:\s*(\d+)", line)
        if m:
            vals.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                         int(m.group(2)), int(m.group(3))))
    out = []
    for ts, ffn, mult in pdr:
        _, st, tok = min(vals, key=lambda v: abs((v[0] - ts).total_seconds()))
        if st < WARMUP_STEP:
            continue
        out.append((tok / 1e6, ffn / mult))  # (tokM, K_ffn)
    return sorted(out)


def load_dn2_ffn(root):
    """DN2 FFN median pdr vs tokenM (the merge-target curve)."""
    out = []
    for d in _parse_jsonl(os.path.join(root, "dreadnought_v2/diagnostics.jsonl")):
        L = d.get("layers")
        if not L:
            continue
        ff = [x["ffn"]["param_delta_ratio"] for x in L
              if x.get("ffn", {}).get("param_delta_ratio") is not None]
        if ff:
            out.append((d.get("total_tokens", 0) / 1e6, _median(ff)))
    return sorted(out)


def _interp(curve, x):
    xs = [c[0] for c in curve]
    if x <= xs[0]:
        return curve[0][1]
    if x >= xs[-1]:
        return curve[-1][1]
    i = bisect.bisect_left(xs, x)
    (x0, y0), (x1, y1) = curve[i - 1], curve[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# ----------------------------- reference (setpoint) curves -----------------------------
def make_reference(name, dn2, kv2_start_ffn):
    """r(tokM) FFN-pdr setpoint. All hold = kv2_start through warmup (engaged externally)."""
    if name == "dn2_merge":
        # SMOOTH alpha-blend (Math Q8 §7): r = (1-a)*r_kv2_early + a*r_DN2, with a(t) a
        # smoothstep from 0 at LR-cap (197M) to 1 at the merge (~575M). Encodes the belief
        # "kv2 early plasticity is good; DN2 late consolidation is good" — no piecewise cliff.
        t0, t1 = 197.0, 575.0

        def kv2_early(t):  # preserve early regime: near-hold w/ gentle decline
            return _interp([(197, kv2_start_ffn), (400, 3.10e-3), (600, 2.90e-3)], t)

        def alpha(t):
            u = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            return u * u * (3 - 2 * u)  # smoothstep (C1 continuous)

        def r(t):
            a = alpha(t)
            return (1 - a) * kv2_early(t) + a * _interp(dn2, t)
        return r
    if name == "math_glide":
        # Math's example glide (Brief #7 §6): gentle, ends ~2.65e-3 @ 787M.
        knots = [(197, kv2_start_ffn), (328, 3.05e-3), (590, 2.85e-3),
                 (787, 2.65e-3), (1050, 2.55e-3)]
        return lambda t: _interp(knots, t)
    if name == "aggressive":
        # Faster descent toward DN2's proven-healthy ~2.28e-3 by 800M.
        knots = [(197, kv2_start_ffn), (300, 3.00e-3), (450, 2.65e-3),
                 (600, 2.45e-3), (800, 2.28e-3), (1050, 2.10e-3)]
        return lambda t: _interp(knots, t)
    raise ValueError(name)


# ----------------------------- exogenous K(t) scenarios -----------------------------
def make_K(kv2K, mode):
    """K_ffn(tokM). Uses recorded kv2 K where available; extrapolates per scenario beyond."""
    last_t, last_K = kv2K[-1]
    # local linear slope of recorded K near the end (per M tok)
    t0, K0 = kv2K[max(0, len(kv2K) - 6)]
    slope = (last_K - K0) / max(1e-6, (last_t - t0))

    def K(t):
        if t <= last_t:
            return _interp(kv2K, t)
        dt = t - last_t
        if mode == "recorded" or mode == "rise":
            return last_K + slope * dt                      # keep drifting up (worst case)
        if mode == "stabilize":
            return last_K                                   # gain flattens
        if mode == "fall":
            return max(last_K * 0.5, last_K - slope * dt)   # update structure relaxes
        if mode == "jump":
            base = last_K + slope * 0.3 * dt
            return base * (1.20 if 500 <= t <= 520 else 1.0)  # +20% transient at 500M
        raise ValueError(mode)
    return K


# ----------------------------- the controller under test -----------------------------
class FFNController:
    def __init__(self, alpha=0.15, kp=0.10, ki=0.01, i_clamp=0.5,
                 m_min=0.30, m_max=1.0, rate_down=0.05, rate_up=0.02, delay=0):
        self.a, self.kp, self.ki, self.iclamp = alpha, kp, ki, i_clamp
        self.m_min, self.m_max = m_min, m_max
        self.rdown, self.rup = rate_down, rate_up
        self.delay = delay                      # actuator lag in control-samples
        self.logK = None
        self.pdr_ema = None
        self.I = 0.0
        self.m = 1.0
        self._pending = []                      # for delay modeling

    def update(self, step, pdr_obs, r):
        if step < WARMUP_STEP:
            return 1.0                          # warmup gate: m=1.0, loop frozen
        # measurement smoothing
        self.pdr_ema = pdr_obs if self.pdr_ema is None else \
            math.exp((1 - self.a) * math.log(self.pdr_ema) + self.a * math.log(pdr_obs))
        K_inst = pdr_obs / max(self.m, 1e-6)
        self.logK = math.log(K_inst) if self.logK is None else \
            (1 - self.a) * self.logK + self.a * math.log(K_inst)
        K_ema = math.exp(self.logK)
        # feedforward inversion + log-error PI trim
        e = math.log(r / self.pdr_ema)
        new_I = max(-self.iclamp, min(self.iclamp, self.I + self.ki * e))
        m_cmd = (r / K_ema) * math.exp(self.kp * e + new_I)
        # rate limit (asymmetric: cool fast, reheat slow)
        m_cmd = max(self.m * (1 - self.rdown), min(self.m * (1 + self.rup), m_cmd))
        # output clamp
        m_clamped = max(self.m_min, min(self.m_max, m_cmd))
        # anti-windup: freeze integral if pinned at a rail in the error's direction
        if (m_clamped >= self.m_max and e > 0) or (m_clamped <= self.m_min and e < 0):
            new_I = self.I
        self.I = new_I
        # actuator delay
        self._pending.append(m_clamped)
        applied = self._pending.pop(0) if len(self._pending) > self.delay else self.m
        self.m = applied
        return applied


# ----------------------------- simulation + metrics -----------------------------
def simulate(ref_fn, K_fn, noise, ctrl, horizon_tok=850, rng=None):
    rng = rng or random.Random(1234)
    traj = []
    step = WARMUP_STEP - 3 * CADENCE
    while True:
        tok = step * TOK_PER_STEP / 1e6
        if tok > horizon_tok:
            break
        r = ref_fn(tok)
        K = K_fn(tok)
        pdr_true = K * ctrl.m
        pdr_obs = pdr_true * (1 + noise * rng.gauss(0, 1)) if noise else pdr_true
        m_new = ctrl.update(step, pdr_obs, r)
        traj.append(dict(step=step, tok=tok, r=r, K=K, m=m_new,
                         pdr_true=K * m_new, pdr_obs=pdr_obs))
        step += CADENCE
    return traj


def metrics(traj, ctrl):
    eng = [p for p in traj if p["step"] >= WARMUP_STEP]
    settle = eng[2:]  # allow a few samples to settle
    if not settle:
        return {}
    track = math.sqrt(sum((math.log(p["pdr_true"] / p["r"])) ** 2 for p in settle) / len(settle))
    dms = [settle[i]["m"] - settle[i - 1]["m"] for i in range(1, len(settle))]
    reversals = sum(1 for i in range(1, len(dms)) if dms[i] * dms[i - 1] < 0)
    max_dm = max((abs(d) / max(settle[i]["m"], 1e-6) for i, d in enumerate(dms)), default=0)
    overshoot = max((p["pdr_true"] / p["r"] - 1 for p in settle), default=0)
    froze = any(p["m"] <= ctrl.m_min + 1e-6 for p in settle)
    # base-LR alarm: floor pinned AND pdr > 1.1 r for >=3 consecutive
    alarm = False
    run = 0
    for p in settle:
        if p["m"] <= ctrl.m_min + 1e-6 and p["pdr_true"] > 1.1 * p["r"]:
            run += 1
            alarm = alarm or run >= 3
        else:
            run = 0
    return dict(track_rms=track, reversals=reversals, max_dm_pct=max_dm * 100,
                overshoot_pct=overshoot * 100, froze=froze, alarm=alarm,
                m_final=settle[-1]["m"])


def ascii_plot(traj, height=16):
    eng = [p for p in traj if p["step"] >= WARMUP_STEP - CADENCE]
    series = {"r": [p["r"] * 1e3 for p in eng], "pdr": [p["pdr_true"] * 1e3 for p in eng]}
    allv = [v for s in series.values() for v in s]
    lo, hi = min(allv), max(allv)
    hi = hi + 0.05 if hi == lo else hi
    marks = {"r": ".", "pdr": "#"}
    grid = [[" "] * len(eng) for _ in range(height)]
    for k, vals in series.items():
        for x, v in enumerate(vals):
            y = int((v - lo) / (hi - lo) * (height - 1))
            grid[height - 1 - y][x] = marks[k]
    print(f"  pdr(#) vs setpoint(.)   [{lo:.2f} .. {hi:.2f}] e-3")
    for row in grid:
        print("   |" + "".join(row))
    print("   +" + "-" * len(eng))
    # mult on a second mini-axis
    ms = [p["m"] for p in eng]
    print(f"  mult: {ms[0]:.3f} -> {ms[-1]:.3f}  (min {min(ms):.3f})")
    toks = [p["tok"] for p in eng]
    print(f"  tokens: {toks[0]:.0f}M -> {toks[-1]:.0f}M")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--ref", default="dn2_merge", choices=["dn2_merge", "math_glide", "aggressive"])
    ap.add_argument("--kscen", default="rise", choices=["recorded", "rise", "stabilize", "fall", "jump"])
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--ff-only", action="store_true", help="kp=ki=0 (pure feedforward)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--horizon", type=float, default=850)
    args = ap.parse_args()

    kv2K = load_kv2_Kffn(args.root)
    dn2 = load_dn2_ffn(args.root)
    kv2_start = kv2K[0][1]  # K_ffn at engage ~= observed ffn pdr at mult=1
    print(f"Loaded kv2 K_ffn: {len(kv2K)} pts, {kv2K[0][0]:.0f}M..{kv2K[-1][0]:.0f}M, "
          f"K {kv2K[0][1]*1e3:.3f}->{kv2K[-1][1]*1e3:.3f}e-3 (rising)")
    print(f"DN2 FFN curve: peak {max(dn2,key=lambda r:r[1])[1]*1e3:.3f}e-3, "
          f"-> {_interp(dn2,800)*1e3:.3f}e-3 @800M\n")

    def mk_ctrl():
        if args.ff_only:
            return FFNController(kp=0.0, ki=0.0)
        return FFNController()

    # ---- nominal detailed run ----
    ref = make_reference(args.ref, dn2, kv2_start)
    K = make_K(kv2K, args.kscen)
    ctrl = mk_ctrl()
    traj = simulate(ref, K, args.noise, ctrl, args.horizon)
    mt = metrics(traj, ctrl)
    print(f"=== NOMINAL: ref={args.ref}  K={args.kscen}  noise={args.noise:.0%}  "
          f"{'FF-only' if args.ff_only else 'FF+PI'} ===")
    print(f"  track_rms(log)={mt['track_rms']:.3f}  reversals={mt['reversals']}  "
          f"max|Δm|={mt['max_dm_pct']:.1f}%  overshoot={mt['overshoot_pct']:+.1f}%  "
          f"m_final={mt['m_final']:.3f}  froze={mt['froze']}  ALARM={mt['alarm']}")
    if args.plot:
        print()
        ascii_plot(traj)

    # ---- robustness matrix ----
    print("\n=== ROBUSTNESS MATRIX (track_rms / reversals / overshoot% / m_final / flags) ===")
    refs = ["dn2_merge", "math_glide", "aggressive"]
    scens = ["recorded", "rise", "stabilize", "fall", "jump"]
    print(f"  {'ref':>11} | " + " | ".join(f"{s:>22}" for s in scens))
    for rn in refs:
        rf = make_reference(rn, dn2, kv2_start)
        cells = []
        for sc in scens:
            c = mk_ctrl()
            t = simulate(rf, make_K(kv2K, sc), args.noise, c, args.horizon)
            m = metrics(t, c)
            flag = "FREEZE!" if m["alarm"] else ("frz" if m["froze"] else "ok")
            cells.append(f"{m['track_rms']:.2f}/{m['reversals']:>2}/{m['overshoot_pct']:+4.0f}/"
                         f"{m['m_final']:.2f}/{flag:>6}")
        print(f"  {rn:>11} | " + " | ".join(cells))
    print("\n  track_rms = RMS log-tracking error (lower=better; <0.05 ~tight)")
    print("  reversals = mult direction flips (oscillation proxy); overshoot = max pdr above setpoint")
    print("  flag: ok / frz(touched floor) / FREEZE!(floor+pdr>1.1r sustained = base-LR-too-high alarm)")


if __name__ == "__main__":
    main()
