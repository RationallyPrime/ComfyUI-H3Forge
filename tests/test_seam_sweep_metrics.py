"""A foreground jump can disappear in the sweep's median background-shimmer score."""
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_one_frame_foreground_jump_is_reported_in_its_span(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "experiments/seam-sweep-20261120/seam_metrics.py"
    spec = spec_from_file_location("seam_metrics", path)
    metrics = module_from_spec(spec)
    spec.loader.exec_module(metrics)

    before = np.zeros((metrics.H, metrics.W), dtype=np.uint8)
    before[400:700, 20:120] = 255
    after = np.zeros_like(before)
    after[400:700, 330:430] = 255
    frames = BytesIO(before.tobytes() * 4 + after.tobytes() * 4)
    monkeypatch.setattr(metrics.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(stdout=frames))

    metrics.main(["synthetic.mp4", "--spans", "still:0:0.16,join:0.16:0.33"])
    report = capsys.readouterr().out.splitlines()
    still = next(line for line in report if line.strip().startswith("still "))
    join = next(line for line in report if line.strip().startswith("join "))
    assert "local_peak=0.00" in still
    assert "flick= 0.00" in join and "topflick= 0.00" in join
    assert "local_peak=255.00 at frame 4 (0.167 s)" in join
