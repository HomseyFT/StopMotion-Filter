#!/usr/bin/env python3
"""
pose_to_angles.py
-----------------
Extract per-frame 2D limb angles from a video using MediaPipe Pose.

Outputs a CSV:
  clip_id,frame,time_sec,side,upperLeg,lowerLeg,ankle

- upperLeg: thigh vs vertical (deg)
- lowerLeg: knee flexion (hip–knee–ankle) (deg, 0=straight, 180=fully folded)
- ankle: dorsiflexion (knee–ankle–toe) (deg)

Usage:
  python pose_to_angles.py --in input.mp4 --out_csv out/angles.csv --clip_id MyClip --fps_override 30
"""
import argparse, os
import cv2, numpy as np, pandas as pd

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("MediaPipe is required. Install: pip install mediapipe") from e

mp_pose = mp.solutions.pose
POSE = mp_pose.PoseLandmark

def angle_deg(a, b, c):
    """Angle ABC in degrees between BA and BC (range [0, 180])."""
    v1 = a - b
    v2 = c - b
    n1 = v1 / (np.linalg.norm(v1) + 1e-9)
    n2 = v2 / (np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def dorsiflexion_angle(knee, ankle, toe):
    """Angle between shank (knee->ankle) and foot (ankle->toe)."""
    return angle_deg(knee, ankle, toe)

def get_xy(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=float)

def compute_angles(lms, w, h, side: str):
    if side == 'left':
        hip = POSE.LEFT_HIP.value; knee = POSE.LEFT_KNEE.value
        ankle = POSE.LEFT_ANKLE.value; toe = POSE.LEFT_FOOT_INDEX.value
    else:
        hip = POSE.RIGHT_HIP.value; knee = POSE.RIGHT_KNEE.value
        ankle = POSE.RIGHT_ANKLE.value; toe = POSE.RIGHT_FOOT_INDEX.value
    req = [hip, knee, ankle, toe]
    if any(lms[i].visibility < 0.5 for i in req):
        return None
    hip_xy = get_xy(lms, hip, w, h); knee_xy = get_xy(lms, knee, w, h)
    ankle_xy = get_xy(lms, ankle, w, h); toe_xy = get_xy(lms, toe, w, h)
    vertical_up = np.array([0.0, -1.0], dtype=float)
    thigh = knee_xy - hip_xy
    thigh_n = thigh / (np.linalg.norm(thigh) + 1e-9)
    upperLeg = float(np.degrees(np.arccos(np.clip(np.dot(thigh_n, vertical_up), -1.0, 1.0))))
    lowerLeg = angle_deg(hip_xy, knee_xy, ankle_xy)
    ankle_ang = dorsiflexion_angle(knee_xy, ankle_xy, toe_xy)
    return upperLeg, lowerLeg, ankle_ang

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input video path")
    ap.add_argument("--out_csv", required=True, help="CSV to write angles")
    ap.add_argument("--clip_id", default=None, help="Clip ID to store in CSV")
    ap.add_argument("--fps_override", type=float, default=None, help="Override FPS")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.inp}")
    fps = args.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_id = args.clip_id or os.path.splitext(os.path.basename(args.inp))[0]

    rows = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for side in ('left','right'):
                    ang = compute_angles(lms, w, h, side)
                    if ang is None: continue
                    upperLeg, lowerLeg, ankle = ang
                    rows.append(dict(clip_id=clip_id, frame=f, time_sec=f/fps, side=side,
                                     upperLeg=upperLeg, lowerLeg=lowerLeg, ankle=ankle))
            f += 1

    cap.release()
    if not rows:
        raise SystemExit("No angles extracted. Ensure subject is visible and mostly side view.")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote angles CSV -> {args.out_csv} (fps={fps:.3f})")

if __name__ == "__main__":
    main()

