#!/usr/bin/env python3
"""
video_retime_hold.py
--------------------
Retimes a video to emulate stop-motion by holding frames in K bins per second with timing jitter.
Also supports simple film grain and subpixel camera jitter.

Usage:
  python video_retime_hold.py --in input.mp4 --out out.mp4 --K 8 --sigma_t 0.02 --out_fps 30 --grain 0.02 --jitter_px 0.5
"""
import argparse, cv2, numpy as np

def add_grain(img, strength=0.02, rng=None):
    if strength <= 0: return img
    if rng is None: rng = np.random.default_rng()
    noise = rng.normal(0.0, strength*255.0, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def subpixel_jitter(img, jitter_px=0.5, rng=None):
    if jitter_px <= 0: return img
    if rng is None: rng = np.random.default_rng()
    h, w = img.shape[:2]
    dx = rng.normal(0.0, jitter_px); dy = rng.normal(0.0, jitter_px)
    M = np.array([[1,0,dx],[0,1,dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sigma_t", type=float, default=0.02, help="Timing jitter std (seconds)")
    ap.add_argument("--out_fps", type=float, default=None)
    ap.add_argument("--grain", type=float, default=0.0)
    ap.add_argument("--jitter_px", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.inp}")
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.out_fps or in_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    if not frames: raise SystemExit("No frames read.")

    dur = len(frames)/in_fps
    rng = np.random.default_rng(args.seed)
    n_out = int(round(dur * out_fps))
    out_frames = []
    for i in range(n_out):
        t = (i+0.5)/out_fps
        # quantize to K bins with jitter
        if args.K > 0:
            u = t/dur if dur>0 else 0.0
            bin_len = 1.0/args.K
            u_center = (int(u*args.K)+0.5)*bin_len
            u_held = np.clip(u_center + (args.sigma_t/dur if dur>0 else 0.0)*rng.normal(), 0.0, 1.0-1e-9)
        else:
            u_held = (t/dur) if dur>0 else 0.0
        src_t = u_held * dur
        idx = int(np.clip(round(src_t*in_fps), 0, len(frames)-1))
        img = frames[idx]
        if args.jitter_px>0: img = subpixel_jitter(img, args.jitter_px, rng)
        if args.grain>0: img = add_grain(img, args.grain, rng)
        out_frames.append(img)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(args.out, fourcc, out_fps, (w,h))
    for fr in out_frames: wr.write(fr)
    wr.release()
    print(f"Wrote retimed video -> {args.out} ({n_out} frames @ {out_fps} fps)")

if __name__ == "__main__":
    main()

