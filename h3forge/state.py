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

    def __post_init__(self):
        if self.feta_first_layer > self.feta_last_layer:
            raise ValueError(
                f"feta_first_layer ({self.feta_first_layer}) must not exceed "
                f"feta_last_layer ({self.feta_last_layer})"
            )


@dataclass
class RuntimeState:
    policy: AttentionPolicy
    default_policy: AttentionPolicy | None = None
    layout: Any = None
    block_index: int | None = None
    block_args: Any = None
    step_index: int | None = None
    total_steps: int | None = None
    current_sigma: float | None = None
    in_run: bool = False
    nag: Any = None
    nag_runtime: Any = None
    diffusion: Any = None
    blocks: Any = None
    mask_cache: OrderedDict = field(default_factory=OrderedDict)
    mask_cache_limit: int = 32
    mask_hits: int = 0
    mask_misses: int = 0
    mask_evictions: int = 0
    dense_calls: int = 0
    sparse_calls: int = 0
    feta_calls: int = 0
    nag_calls: int = 0
    declines: dict[str, int] = field(default_factory=dict)

    def note_decline(self, reason: str) -> None:
        self.declines[reason] = self.declines.get(reason, 0) + 1

    def begin_run(self) -> None:
        self.in_run = True
        self.layout = None
        self.block_index = None
        self.block_args = None
        self.step_index = None
        self.total_steps = None
        self.current_sigma = None
        # Config is re-resolved from the sampled clone's transformer_options on
        # every forward; resetting here keeps a stale config from surviving a
        # run whose forwards never resolve (e.g. an aborted sample).
        self.nag = None
        self.nag_runtime = None
        if self.default_policy is not None:
            self.policy = self.default_policy
        self.dense_calls = 0
        self.sparse_calls = 0
        self.feta_calls = 0
        self.nag_calls = 0
        self.mask_hits = 0
        self.mask_misses = 0
        self.mask_evictions = 0
        self.declines.clear()

    def stats(self) -> str:
        bits = [f"sparse={self.sparse_calls}", f"dense={self.dense_calls}", f"feta={self.feta_calls}"]
        if self.nag_calls:
            bits.append(f"nag={self.nag_calls}")
        if self.mask_hits or self.mask_misses:
            bits.append(f"masks=hit:{self.mask_hits},miss:{self.mask_misses},evict:{self.mask_evictions}")
        if self.declines:
            bits.append("declines=" + ",".join(f"{k}:{v}" for k, v in sorted(self.declines.items())))
        return " ".join(bits)


def resolve_step(transformer_options: dict, sigma: float | None = None) -> tuple[int | None, int | None]:
    """Best-effort sampler step resolution from ComfyUI transformer_options.

    Callers that already resolved the current sigma pass it in so the sigmas
    tensor is read (one GPU→CPU sync) once per forward, not once per resolver.
    """
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sample_sigmas is None:
        return None, None
    total = max(int(sample_sigmas.shape[0]) - 1, 0)
    if sigma is None:
        sigma = resolve_sigma(transformer_options)
    if sigma is None:
        return None, total
    try:
        vals = sample_sigmas.detach().float().flatten()
        idx = int((vals - sigma).abs().argmin())
        return min(idx, total), total
    except Exception:
        return None, total


def resolve_sigma(transformer_options: dict) -> float | None:
    """Best-effort current-sigma resolution, independent of the full schedule."""
    current = transformer_options.get("sigmas")
    if current is None or getattr(current, "numel", lambda: 0)() == 0:
        return None
    try:
        return float(current.flatten()[0])
    except Exception:
        return None
