#!/usr/bin/env python3
"""
stopmotion_filter.py
--------------------
Fit or apply a step-hold model to time-series angles. Also compute angular velocity and acceleration.

Usage:
  python stopmotion_filter.py --in_csv angles.csv --out_csv choppy.csv --fit_auto 1 --K 8 --sigma_t 0.02 --sigma_deg 0.5

If --fit_auto 1, K and sigma values are estimated from data. Otherwise pass them explicitly.
"""
import argparse, json
import numpy as np, pandas as pd

def unwrap_deg(x):
    rad = np.deg2rad(x)
    unwrapped = np.unwrap(rad)
    return np.rad2deg(unwrapped)

def moving_average(x, win=5):
    x = np.asarray(x, float)
    if win <= 1: return x.copy()
    pad = win//2
    xpad = np.pad(x, (pad,pad), mode='reflect')
    kern = np.ones(win)/win
    return np.convolve(xpad, kern, mode='valid')

def fit_params(t, theta, min_K=3, max_K=24, smooth_win=5):
    t = np.asarray(t, float); theta = unwrap_deg(theta)
    t0, t1 = t[0], t[-1]; T = max(1e-9, t1-t0); u = (t-t0)/T
    base = moving_average(theta, smooth_win)
    r = theta - base
    Ks = np.arange(min_K, max_K+1, dtype=int)
    best = None
    for K in Ks:
        bins = np.linspace(0,1,K+1)
        idx = np.digitize(u, bins[1:-1], right=False)
        step_vals = np.array([r[idx==k].mean() if np.any(idx==k) else 0.0 for k in range(K)])
        rhat = step_vals[idx]
        rmse = float(np.sqrt(np.mean((r-rhat)**2)))
        if best is None or rmse < best[0]:
            best = (rmse, K, step_vals, idx)
    _, K, step_vals, idx = best
    # timing jitter estimate (within-bin std of u)
    u_stds = [np.std(u[idx==k], ddof=1) for k in range(K) if np.sum(idx==k)>1]
    sigma_u = float(np.median(u_stds)) if u_stds else 0.0
    sigma_t = sigma_u * T
    sigma_deg = float(np.std((r - step_vals[idx]), ddof=1)) if len(r)>1 else 0.0
    return dict(K=int(K), sigma_t=float(sigma_t), sigma_deg=float(sigma_deg), smooth_win=int(smooth_win))

def apply_filter(t, theta, K, sigma_t, sigma_deg, smooth_win=5, rng=None):
    t = np.asarray(t, float); theta = unwrap_deg(theta)
    t0, t1 = t[0], t[-1]; T = max(1e-9, t1-t0); u = (t-t0)/T
    base = moving_average(theta, smooth_win)
    resid = theta - base
    if rng is None: rng = np.random.default_rng()
    bins = np.linspace(0,1,K+1)
    # jitter time then bin
    u_j = np.clip(u + rng.normal(0.0, sigma_t/max(T,1e-9), size=u.shape), 0.0, 1.0-1e-9)
    idx = np.digitize(u_j, bins[1:-1], right=False)
    step_vals = np.array([np.median(resid[idx==k]) if np.any(idx==k) else 0.0 for k in range(K)])
    resid_hat = step_vals[idx]
    # Laplace noise with std ~ sigma_deg
    if sigma_deg>0:
        b = sigma_deg/np.sqrt(2.0); noise = rng.laplace(0.0, b, size=resid_hat.shape)
    else:
        noise = 0.0
    theta_chop = base + resid_hat + noise
    return theta_chop

def finite_diff(y, t):
    """
    Backward-difference velocity and acceleration on irregular time grids.

    vel[0] = 0
    vel[i] = (y[i] - y[i-1]) / (t[i] - t[i-1])

    acc[0] = 0
    acc[i] = (vel[i] - vel[i-1]) / (t[i] - t[i-1])
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    n = len(y)
    vel = np.zeros(n, dtype=float)
    acc = np.zeros(n, dtype=float)

    if n <= 1:
        return vel, acc

    dt = np.diff(t)
    dt[dt == 0] = 1e-9  # avoid divide-by-zero

    # velocity: backward difference
    vel[1:] = np.diff(y) / dt

    if n <= 2:
        return vel, acc

    # acceleration: backward difference on velocity
    acc[1:] = np.diff(vel) / dt

    return vel, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--fit_auto", type=int, default=1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sigma_t", type=float, default=0.02)
    ap.add_argument("--sigma_deg", type=float, default=0.5)
    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    needed = {'time_sec','side','upperLeg','lowerLeg','ankle'}
    if not needed.issubset(df.columns):
        raise SystemExit("Input CSV must have: time_sec, side, upperLeg, lowerLeg, ankle")

    rng = np.random.default_rng(args.seed)
    params_used = {}  # side -> column -> params

    out_rows = []
    for side, g in df.groupby('side', sort=False):
        t = g['time_sec'].values
        for col in ('upperLeg','lowerLeg','ankle'):
            y = g[col].values
            if args.fit_auto:
                p = fit_params(t, y, smooth_win=args.smooth_win)
                K, sigma_t, sigma_deg = p['K'], p['sigma_t'], p['sigma_deg']
            else:
                K, sigma_t, sigma_deg = args.K, args.sigma_t, args.sigma_deg
            if side not in params_used:
                params_used[side] = {}
            params_used[side][col] = dict(K=K, sigma_t=sigma_t, sigma_deg=sigma_deg, smooth_win=args.smooth_win)
            
            ych = apply_filter(t, y, K, sigma_t, sigma_deg, args.smooth_win, rng)
            vel, acc = finite_diff(ych, t)
            out_rows.append(pd.DataFrame({
                'side': side,
                'time_sec': t,
                f'{col}_chop': ych,
                f'{col}_vel': vel,
                f'{col}_acc': acc,
            }))

    out = df.copy()
    for block in out_rows:
        out = out.merge(block, on=['side','time_sec'], how='left')

    out.to_csv(args.out_csv, index=False)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

    # also drop a JSON with the fitted params next to the CSV
    import os, json
    params_json = os.path.splitext(args.out_csv)[0] + "_params.json"
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(params_used, f, indent=2)
    print("Params written to:", params_json)


if __name__ == "__main__":
    main()

