from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttentionPolicy:
    mode: str = "flex_sliding"
    temporal_window: float = 40.0
    spatial_radius: float = 8.0
    bridge_stride: int = 40
    first_dense_layers: int = 2
    first_dense_fraction: float = 0.15
    strict: bool = False
    feta_enabled: bool = False
    feta_strength: float = 2.0
    feta_max_gain: float = 1.15
    feta_first_layer: int = 6
    feta_last_layer: int = 42
    feta_max_spatial_samples: int = 128
    feta_max_heads: int = 8


@dataclass
class RuntimeState:
    policy: AttentionPolicy
    layout: Any = None
    block_index: int | None = None
    step_index: int | None = None
    total_steps: int | None = None
    mask_cache: OrderedDict = field(default_factory=OrderedDict)
    mask_cache_limit: int = 32
    mask_hits: int = 0
    mask_misses: int = 0
    mask_evictions: int = 0
    dense_calls: int = 0
    sparse_calls: int = 0
    feta_calls: int = 0
    declines: dict[str, int] = field(default_factory=dict)

    def note_decline(self, reason: str) -> None:
        self.declines[reason] = self.declines.get(reason, 0) + 1

    def begin_run(self) -> None:
        self.layout = None
        self.block_index = None
        self.step_index = None
        self.total_steps = None
        self.dense_calls = 0
        self.sparse_calls = 0
        self.feta_calls = 0
        self.mask_hits = 0
        self.mask_misses = 0
        self.mask_evictions = 0
        self.declines.clear()

    def stats(self) -> str:
        bits = [f"sparse={self.sparse_calls}", f"dense={self.dense_calls}", f"feta={self.feta_calls}"]
        if self.mask_hits or self.mask_misses:
            bits.append(f"masks=hit:{self.mask_hits},miss:{self.mask_misses},evict:{self.mask_evictions}")
        if self.declines:
            bits.append("declines=" + ",".join(f"{k}:{v}" for k, v in sorted(self.declines.items())))
        return " ".join(bits)


def resolve_step(transformer_options: dict) -> tuple[int | None, int | None]:
    """Best-effort sampler step resolution from ComfyUI transformer_options."""
    sample_sigmas = transformer_options.get("sample_sigmas")
    current = transformer_options.get("sigmas")
    if sample_sigmas is None:
        return None, None
    total = max(int(sample_sigmas.shape[0]) - 1, 0)
    if current is None or getattr(current, "numel", lambda: 0)() == 0:
        return None, total
    try:
        sigma = float(current.flatten()[0])
        vals = sample_sigmas.detach().float().flatten()
        idx = int((vals - sigma).abs().argmin())
        return min(idx, total), total
    except Exception:
        return None, total
