"""Seam metrics for a 24 fps, 28.5 s one take render (3 shots of 9 s, overlap 3.0 s).

Usage: python3 seam_metrics.py take_a.mp4 [take_b.mp4 ...]
Needs ffmpeg and numpy. Prints, per span, the mean of:
  sharp     Laplacian variance of the whole frame (higher = more detail)
  topsharp  the same for the top 300 rows (mostly static background)
  flick     median over pixels of |f[t] - 2 f[t-1] + f[t-2]| (0 on a still, steady picture)
  topflick  mean of the same for the top 300 rows
  diff      mean |f[t] - f[t-1]| (motion; a spike is a cut or a pop)
Also reports the peak local frame change per span (strongest 32 x 32 tile) and its zero-based frame.
Spans marked OV are where two context windows overlap at overlap 3.0 s.
Pass --spans "name:lo:hi,..." to use other spans (seconds).
"""
import subprocess
import sys

import numpy as np

W, H = 544, 960
TILE = 32
SPANS = [("solo 1-8.5", 1, 8.5), ("OV 9-12", 9, 12), ("solo 12.5-16", 12.5, 16),
         ("OV 16.5-21", 16.5, 21), ("solo 21.5-28", 21.5, 28)]


def measure(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev = prev2 = None
    rows = []
    while True:
        buf = proc.stdout.read(W * H)
        if len(buf) < W * H:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.float32)
        lap = f[1:-1, 1:-1] * 4 - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
        flick = topflick = diff = local_peak = 0.0
        if prev is not None:
            change = np.abs(f - prev)
            diff = float(change.mean())
            tiles = change.reshape(H // TILE, TILE, W // TILE, TILE)
            local_peak = float(tiles.mean(axis=(1, 3)).max())
        if prev2 is not None:
            second = np.abs(f - 2 * prev + prev2)
            flick, topflick = float(np.median(second)), float(second[:300].mean())
        rows.append((lap.var(), lap[:300].var(), flick, topflick, diff, local_peak))
        prev2, prev = prev, f
    return np.array(rows)


def main(argv):
    spans = SPANS
    if "--spans" in argv:
        i = argv.index("--spans")
        spans = [(n, float(a), float(b)) for n, a, b in (s.split(":") for s in argv[i + 1].split(","))]
        argv = argv[:i] + argv[i + 2:]
    for path in argv:
        a = measure(path)
        t = np.arange(len(a)) / 24
        print(path.rsplit("/", 1)[-1], len(a), "frames")
        for name, lo, hi in spans:
            frames = np.flatnonzero((t >= lo) & (t < hi))
            if not frames.size:
                print(f"  {name:14s} no samples")
                continue
            s = a[frames, :5].mean(0)
            peak_frame = frames[np.argmax(a[frames, 5])]
            print(f"  {name:14s} sharp={s[0]:6.1f} topsharp={s[1]:6.1f} flick={s[2]:5.2f} "
                  f"topflick={s[3]:5.2f} diff={s[4]:5.2f} "
                  f"local_peak={a[peak_frame, 5]:.2f} at frame {peak_frame} ({t[peak_frame]:.3f} s)")
        secs = len(a) // 24
        print("  per-second flick:", " ".join(f"{a[i * 24:(i + 1) * 24, 2].mean():.1f}" for i in range(secs)))
        print("  per-second diff :", " ".join(f"{a[i * 24:(i + 1) * 24, 4].mean():.1f}" for i in range(secs)))
        print(f"  largest frame-to-frame diff {a[:, 4].max():.2f} at {t[a[:, 4].argmax()]:.2f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
