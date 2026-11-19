"""Video-only measurements at the diffusion wrapper's velocity boundary."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn.functional as F


def snapshot_steps(value: str) -> set[int]:
    if not value.strip():
        return set()
    try:
        steps = {int(item) for item in value.split(",")}
        if min(steps) >= 0:
            return steps
    except ValueError:
        pass
    raise ValueError("overlap_probe_steps must be comma-separated nonnegative step numbers")


def _rel_l2(a, b, dim=None):
    distance = torch.linalg.vector_norm(a - b, dim=dim)
    scale = torch.linalg.vector_norm((a + b) * 0.5, dim=dim)
    return torch.where(distance == 0, 0.0, distance / scale)


def _bands(v):
    # Pool each H x W plane independently. Exact target sizes retain odd edges.
    planes = v.reshape(-1, 1, *v.shape[-2:])
    half = F.avg_pool2d(planes, 2, ceil_mode=True)
    low = F.avg_pool2d(half, 2, ceil_mode=True)
    # Expand each parent back onto its own 2x2 footprint, cropping odd edges.
    # This gives an exactly zero detail band for a spatially constant field.
    mid = half - F.interpolate(low, scale_factor=2, mode="nearest")[..., :half.shape[-2], :half.shape[-1]]
    high = planes - F.interpolate(half, scale_factor=2, mode="nearest")[..., :planes.shape[-2], :planes.shape[-1]]
    return low, mid, high


@dataclass
class WindowOverlap:
    starts: tuple[int, int]
    lo: int
    hi: int
    predictions: list[torch.Tensor] = field(default_factory=list)

    def capture(self, start, velocity):
        if self.lo < self.hi:
            # A view would keep its entire model output alive until fusion.
            self.predictions.append(velocity[:, :, self.lo - start:self.hi - start].detach().clone())

    def record(self, directory, index, step, sigma, phase, x, fused, save):
        a, b = (v.float() for v in self.predictions)
        rows = slice(self.lo, self.hi)
        f = fused[:, :, rows].float()
        band_ratios = [_rel_l2(left, right) for left, right in zip(_bands(a), _bands(b))]
        totals = torch.stack([_rel_l2(a, b), *band_ratios, _rel_l2(a, f), _rel_l2(b, f)])
        ratios = torch.cat([totals, _rel_l2(a, b, dim=(0, 1, 3, 4))]).tolist()
        # Opposite nonzero predictions have zero mean. JSON null is undefined.
        ratios = [v if math.isfinite(v) else None for v in ratios]
        record = dict(step=step, sigma=sigma, phase=phase, overlap_index=index,
                      window_starts=self.starts, overlap_range=[self.lo, self.hi],
                      rel_l2=ratios[0], low=ratios[1], mid=ratios[2], high=ratios[3],
                      rel_l2_a_fused=ratios[4], rel_l2_b_fused=ratios[5], per_latent_rel_l2=ratios[6:])
        if save:
            from safetensors.torch import save_file

            noisy = x[:, :, rows].float()
            tensors = {name: (noisy - sigma * v).contiguous().cpu()
                       for name, v in (("x0_a", a), ("x0_b", b), ("x0_fused", f))}
            # The sampler can evaluate a step more than once. Preserve each snapshot.
            with NamedTemporaryFile(dir=directory, prefix=f"step-{step}-overlap-{index}-",
                                    suffix=".safetensors", delete=False) as file:
                path = Path(file.name)
            save_file(tensors, str(path))
            record["snapshot"] = path.name
        return record


def write_probe(directory, overlaps, step, sigma, phase, x, fused, steps):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "probe.jsonl").open("a", encoding="utf-8") as stream:
        for index, overlap in enumerate(overlaps):
            if overlap.lo < overlap.hi:
                record = overlap.record(directory, index, step, sigma, phase, x, fused, step in steps)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
