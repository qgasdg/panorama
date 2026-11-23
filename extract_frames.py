#!/usr/bin/env python3
"""
extract_frames.py — frame extraction and capture check for the Computer Vision term project

Usage
  python extract_frames.py set1.mp4                  # Set 1: frames chosen for about 60% overlap
  python extract_frames.py set1.mp4 --every 15       # one frame every 15 frames (E7)
  python extract_frames.py set2.mp4 --set 2          # Set 2 (walking): rotation checks are skipped
  python extract_frames.py set1.mov --ffmpeg         # if an HEVC .mov file does not open

What it does
  1. Reads every frame and measures its sharpness (variance of the Laplacian) and its shift from
     the previous frame (phase correlation).
  2. For Set 1, converts the shift into an angle to estimate panning speed and total sweep, and
     warns when they fall outside Section 2.1 (under 10 deg/s, 60-80 deg in total).
  3. Chooses frames for the target overlap, taking the sharpest frame near each position, or one
     frame every N frames with --every.
  4. Saves the chosen frames, reduced to a working size (long side 1280 px by default), together
     with manifest.csv (every frame) and summary.json (the capture check). Submit summary.json.

Every run replaces the frames already in the output folder, so frames from an earlier run with
different settings are never mixed in.

Requirements: numpy, opencv-python. If your video does not open (some iPhone HEVC files), install
ffmpeg and add --ffmpeg, or convert the video to H.264 mp4 yourself.
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from your video and check the capture.")
    p.add_argument("video", help="input video (mp4/mov)")
    p.add_argument("-o", "--out", default=None, help="output folder (default: <video name>_frames)")
    p.add_argument("--set", type=int, choices=[1, 2], default=1,
                   help="1: rotation sweep (default); 2: walking sideways, where the rotation checks do not apply")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--every", type=int, default=None, help="keep one frame every N frames (E7)")
    sel.add_argument("--overlap", type=float, default=0.6, help="target overlap between kept frames (default 0.6)")
    p.add_argument("--long-side", type=int, default=1280, help="long side of the saved frames in pixels (default 1280)")
    p.add_argument("--hfov", type=float, default=70.0,
                   help="horizontal field of view in degrees, used to turn pixel shifts into angles (main lens: about 70)")
    p.add_argument("--blur-keep", type=float, default=0.5,
                   help="a frame is marked blurry when its sharpness is below this fraction of the median (default 0.5)")
    p.add_argument("--analysis-width", type=int, default=480, help="width used for the capture check (default 480)")
    p.add_argument("--fmt", choices=["jpg", "png"], default="jpg")
    p.add_argument("--ffmpeg", action="store_true", help="convert the video to H.264 with ffmpeg before reading it")
    return p.parse_args()


def convert_with_ffmpeg(src):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found. Install it, or convert the video to H.264 mp4 yourself.")
    dst = os.path.splitext(src)[0] + "_h264.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-an", dst], check=True)
    return dst


def downsample(img, long_side):
    """Gaussian blur, then resize: removes detail that the smaller grid cannot hold."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    sigma = max(0.3, 0.5 / s)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.resize(cv2.GaussianBlur(img, (k, k), sigma), (nw, nh), interpolation=cv2.INTER_LINEAR)


def analyze(cap, analysis_width, hfov_deg):
    """One pass over the video: sharpness of each frame and its shift from the previous one."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows, prev, win, i = [], None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i == 0:                                   # the decoder may apply the rotation tag: trust the frame
            H, W = frame.shape[:2]
        scale = analysis_width / W
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (analysis_width, max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
        blur = cv2.Laplacian(g, cv2.CV_64F).var()
        gf = np.float32(g)
        if win is None:
            win = cv2.createHanningWindow((gf.shape[1], gf.shape[0]), cv2.CV_32F)
        dx = dy = 0.0
        if prev is not None:
            (sx, sy), _ = cv2.phaseCorrelate(prev, gf, win)
            dx, dy = -sx / scale, -sy / scale        # in full-resolution pixels
        rows.append({"frame": i, "time_s": i / fps, "blur": float(blur), "dx_px": float(dx), "dy_px": float(dy)})
        prev = gf
        i += 1
    deg_per_px = hfov_deg / W
    cum = 0.0
    for r in rows:
        r["deg_per_frame"] = r["dx_px"] * deg_per_px
        cum += r["deg_per_frame"]
        r["cum_deg"] = cum
        r["deg_per_s"] = r["deg_per_frame"] * fps
    return rows, fps, W, H


def select_frames(rows, W, every, overlap, blur_keep):
    med = float(np.median([r["blur"] for r in rows]))
    for r in rows:
        r["sharp"] = r["blur"] >= blur_keep * med
    if every:
        chosen = [r["frame"] for r in rows[::every]]
    else:
        # one frame each time the accumulated shift reaches (1 - overlap) * width,
        # taking the sharpest frame within a few frames of that position
        step_px = (1.0 - overlap) * W
        cum_px = np.cumsum([abs(r["dx_px"]) for r in rows])
        chosen, target, last = [], 0.0, -1
        while True:
            idx = int(np.searchsorted(cum_px, target))
            if idx >= len(rows):
                break
            lo, hi = max(last + 1, idx - 6), min(len(rows) - 1, idx + 6)
            cand = [rows[k] for k in range(lo, hi + 1) if rows[k]["sharp"]] or [rows[k] for k in range(lo, hi + 1)]
            if cand:
                best = max(cand, key=lambda r: r["blur"])
                if best["frame"] != last:
                    chosen.append(best["frame"]); last = best["frame"]
            target += step_px
        # the end of the sweep is the end of the panorama: keep a frame near it
        if chosen and cum_px[-1] - cum_px[last] > 0.4 * step_px:
            lo = max(last + 1, len(rows) - 8)
            cand = [rows[k] for k in range(lo, len(rows)) if rows[k]["sharp"]] or rows[lo:]
            if cand:
                chosen.append(max(cand, key=lambda r: r["blur"])["frame"])
    for r in rows:
        r["selected"] = r["frame"] in set(chosen)
    return chosen, med


def main():
    a = parse_args()
    src = convert_with_ffmpeg(a.video) if a.ffmpeg else a.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Could not open the video. For an HEVC .mov file, add --ffmpeg or convert it to H.264 mp4.")
    out = a.out or os.path.splitext(os.path.basename(a.video))[0] + "_frames"
    os.makedirs(out, exist_ok=True)

    rows, fps, W, H = analyze(cap, a.analysis_width, a.hfov)
    cap.release()
    if len(rows) < 2:
        sys.exit("The video has fewer than two frames.")
    chosen, med_blur = select_frames(rows, W, a.every, a.overlap, a.blur_keep)

    # remove frames from any earlier run, so that a new --every or --overlap never mixes with old files
    old = glob.glob(os.path.join(out, "frame_*.jpg")) + glob.glob(os.path.join(out, "frame_*.png"))
    for f in old:
        os.remove(f)

    cap = cv2.VideoCapture(src)
    want, saved, i = set(chosen), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            img = downsample(frame, a.long_side)
            name = f"frame_{len(saved):03d}.{a.fmt}"
            cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95] if a.fmt == "jpg" else [])
            saved.append({"file": name, "src_frame": i, "width": img.shape[1], "height": img.shape[0]})
        i += 1
    cap.release()

    # ── capture check ──
    speeds = np.array([abs(r["deg_per_s"]) for r in rows[1:]])
    sweep = abs(rows[-1]["cum_deg"])
    vertical = np.array([abs(r["dy_px"]) for r in rows[1:]])
    n_sharp = sum(r["sharp"] for r in rows)
    warnings = []
    if H > W:
        warnings.append("The video is in portrait orientation. Section 2.1 asks for landscape.")
    if max(W, H) < 1900:
        warnings.append(f"The resolution is {W}x{H}. Section 2.1 asks for 1080p or higher.")
    if abs(fps - 30) > 3:
        warnings.append(f"The frame rate is {fps:.1f} fps. Section 2.1 asks for 30 fps, and the frame intervals in E7 assume it.")
    if a.set == 1:
        if np.percentile(speeds, 90) > 10:
            warnings.append(f"Panning is too fast: {np.percentile(speeds, 90):.1f} deg/s in the fastest 10% of the video "
                            f"(Section 2.1: under 10 deg/s). Pan more slowly.")
        if sweep > 90:
            warnings.append(f"The sweep is about {sweep:.0f} deg (Section 2.1: 60-80 deg). The end frames will be "
                            f"stretched heavily on a plane.")
        if sweep < 40:
            warnings.append(f"The sweep is only about {sweep:.0f} deg (Section 2.1: 60-80 deg). Sweep wider.")
        if np.median(vertical) > 0.02 * H:
            warnings.append(f"The camera moves up and down by {np.median(vertical):.1f} px per frame (median). "
                            f"Keep it level while rotating.")
    if n_sharp < 0.6 * len(rows):
        warnings.append(f"Many frames are blurry: {len(rows) - n_sharp} of {len(rows)} are below "
                        f"{a.blur_keep:.0%} of the median sharpness. Steady the phone or shoot in brighter light.")
    need = 3 if a.set == 1 else 2                     # Set 2 only needs one pair to show how it fails
    if len(chosen) < need:
        warnings.append(f"Fewer than {need} frames were kept. Raise --overlap or use a smaller --every"
                        + (", or sweep wider." if a.set == 1 else ", or walk farther."))

    summary = {
        "video": a.video, "set": a.set, "fps": round(fps, 2), "frames": len(rows), "resolution": [W, H],
        "hfov_deg_assumed": a.hfov,
        "sweep_deg_estimate": round(sweep, 1) if a.set == 1 else None,
        "pan_speed_deg_per_s": ({"median": round(float(np.median(speeds)), 2),
                                 "p90": round(float(np.percentile(speeds, 90)), 2),
                                 "max": round(float(speeds.max()), 2)} if a.set == 1 else None),
        "sharpness_median": round(med_blur, 1), "sharp_frames": n_sharp,
        "selection": {"mode": f"every {a.every}" if a.every else f"overlap {a.overlap}", "chosen_src_frames": chosen},
        "long_side": a.long_side, "saved": saved, "warnings": warnings,
    }
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "blur", "sharp", "dx_px", "dy_px",
                                           "deg_per_frame", "deg_per_s", "cum_deg", "selected"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Video: {W}x{H}, {fps:.1f} fps, {len(rows)} frames ({len(rows) / fps:.1f} s), Set {a.set}")
    if a.set == 1:
        print(f"Sweep: about {sweep:.0f} deg   Panning speed: median {np.median(speeds):.1f} deg/s, "
              f"fastest 10% {np.percentile(speeds, 90):.1f} deg/s")
    print(f"Sharpness: median {med_blur:.0f}, sharp frames {n_sharp}/{len(rows)}")
    if old:
        print(f"Removed {len(old)} frames left from an earlier run in {out}/")
    print(f"Kept {len(chosen)} frames ({summary['selection']['mode']}) -> {out}/frame_000.{a.fmt} ...  long side {a.long_side} px")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("  -", w)
        if a.set == 1 and any(k in " ".join(warnings) for k in ("Panning", "sweep", "up and down")):
            print("  (If this is Set 2, the video where you walk sideways, run again with --set 2.)")
    else:
        print("\nThe capture is within Section 2.1.")


if __name__ == "__main__":
    main()
