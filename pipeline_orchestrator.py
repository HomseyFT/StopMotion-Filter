#!/usr/bin/env python3
"""
pipeline_orchestrator.py
------------------------
End-to-end 2D stop-motion pipeline:

1) Extract limb angles from a 2D video (MediaPipe Pose).
2) Fit or use specified stop-motion timing/noise params.
3) Apply filter to angles; compute velocity and acceleration; write CSV.
4) Retime original video to stepped timing (optional).

Usage:
  python pipeline_orchestrator.py ^
    --in input.mp4 --out_dir out ^
    --fit_auto 1 --K 8 --sigma_t 0.02 --sigma_deg 0.5 ^
    --retime_video 1 --grain 0.02 --jitter_px 0.5

Outputs:
  out/angles.csv
  out/angles_choppy.csv
  out/video_choppy.mp4  (if retime_video=1)
"""
import argparse, os, subprocess, sys, json

def run(cmd):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fit_auto", type=int, default=1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sigma_t", type=float, default=0.02)
    ap.add_argument("--sigma_deg", type=float, default=0.5)
    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--retime_video", type=int, default=1)
    ap.add_argument("--grain", type=float, default=0.0)
    ap.add_argument("--jitter_px", type=float, default=0.0)
    ap.add_argument("--out_fps", type=float, default=None)
    ap.add_argument("--clip_id", default=None)
    ap.add_argument("--fps_override", type=float, default=None)
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    angles_csv = os.path.join(args.out_dir, "angles.csv")
    choppy_csv = os.path.join(args.out_dir, "angles_choppy.csv")
    out_mp4 = os.path.join(args.out_dir, "video_choppy.mp4")

    # 1) Pose -> angles
    cmd1 = [args.python, "pose_to_angles.py", "--in", args.inp, "--out_csv", angles_csv,
            "--clip_id", args.clip_id or "clip"]
    if args.fps_override:
        cmd1 += ["--fps_override", str(args.fps_override)]
    run(cmd1)

    # 2) Angles -> choppy angles
    cmd2 = [args.python, "stopmotion_filter.py", "--in_csv", angles_csv, "--out_csv", choppy_csv,
            "--smooth_win", str(args.smooth_win)]
    if args.fit_auto:
        cmd2 += ["--fit_auto", "1"]
    else:
        cmd2 += ["--fit_auto", "0", "--K", str(args.K), "--sigma_t", str(args.sigma_t), "--sigma_deg", str(args.sigma_deg)]
    run(cmd2)

    # 3) Video retime (optional)
    if args.retime_video:
        cmd3 = [args.python, "video_retime_hold.py", "--in", args.inp, "--out", out_mp4,
                "--grain", str(args.grain), "--jitter_px", str(args.jitter_px)]
        # For now use your chosen K and sigma_t. If fit_auto=1 you can also feed learned values here after parsing choppy CSV.
        cmd3 += ["--K", str(args.K), "--sigma_t", str(args.sigma_t)]
        if args.out_fps: cmd3 += ["--out_fps", str(args.out_fps)]
        run(cmd3)

    print("Pipeline complete.")
    print(json.dumps({
        "angles_csv": angles_csv,
        "choppy_csv": choppy_csv,
        "video_choppy": out_mp4 if args.retime_video else None
    }, indent=2))

if __name__ == "__main__":
    main()

