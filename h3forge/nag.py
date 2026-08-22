from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .layout import target_segments

LOG = "[H3Forge]"

# H3 is a single-stream packed transformer: text, references, audio, and video
# share one self-attention per block, so faithful dual-branch NAG would cost
# close to a second full forward (the Flux case, not the Wan case). NAG-Lite
# instead adds a negative-text sidecar and guides only the text-conditioned
# contribution to the target audio/video rows. Two documented approximations:
# the sidecar text state is frozen at the refined embedding rather than
# re-evolved through earlier blocks, and standalone text attention does not
# share the packed softmax denominator (lite mode) — hence the name NAG-Lite,
# not "the official NAG implementation for H3".
_QKV_PATHS = ("attn.qkv_proj", "attn.qkv", "attn.to_qkv", "self_attn.qkv_proj", "attention.qkv_proj")
_NORM_PATHS = ("attn_norm", "norm1", "norm_attn", "attention_norm", "pre_norm")
NAG_MODES = ("lite", "faithful_selective")


@dataclass
class NAGConfig:
    negative_context: torch.Tensor
    mode: str = "lite"
    scale: float = 3.0
    tau: float = 2.5
    alpha: float = 0.15
    sigma_end: float = 0.70
    first_block: int = 8
    last_block: int = 28
    video_strength: float = 1.0
    audio_strength: float = 0.5
    strict: bool = False

    def __post_init__(self):
        if self.mode not in NAG_MODES:
            raise ValueError(f"NAG mode must be one of {NAG_MODES}, got {self.mode!r}")
        if self.first_block > self.last_block:
            raise ValueError(f"first_block ({self.first_block}) must not exceed last_block ({self.last_block})")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"nag_alpha must be within [0, 1], got {self.alpha}")
        if self.scale < 1.0:
            raise ValueError(f"nag_scale must be >= 1, got {self.scale}")
        if self.tau < 1.0:
            raise ValueError(f"nag_tau must be >= 1, got {self.tau}")
        if self.negative_context.ndim != 3 or self.negative_context.shape[0] != 1:
            raise ValueError(
                f"negative conditioning must be a [1, tokens, channels] context, "
                f"got {tuple(self.negative_context.shape)}"
            )


@dataclass
class NAGRuntime:
    refined: torch.Tensor | None = None
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def nag_combine(z_pos: torch.Tensor, z_neg: torch.Tensor, scale: float, tau: float, alpha: float) -> torch.Tensor:
    """Normalized Attention Guidance on attention features (arXiv NAG formulation).

    guided = pos*scale - neg*(scale-1), L1-renormalized against pos with cap tau,
    then alpha-blended back toward pos.
    """
    dtype = z_pos.dtype
    pos = z_pos.float()
    guided = pos * scale - z_neg.float() * (scale - 1.0)
    norm_pos = pos.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    ratio = guided.abs().sum(dim=-1, keepdim=True) / norm_pos
    guided = guided * (torch.clamp(ratio, max=tau) / ratio.clamp_min(1e-6))
    return (guided * alpha + pos * (1.0 - alpha)).to(dtype)


def _resolve_module(root, path: str):
    node = root
    for name in path.split("."):
        node = getattr(node, name, None)
        if node is None:
            return None
    return node


def discover_projection(block):
    """Locate a DiT block's fused qkv projection and (optionally) its pre-attention norm."""
    qkv = None
    for path in _QKV_PATHS:
        candidate = _resolve_module(block, path)
        if candidate is not None and callable(candidate):
            qkv = candidate
            break
    if qkv is None:
        return None, None
    for path in _NORM_PATHS:
        norm = _resolve_module(block, path)
        if norm is not None and callable(norm):
            return qkv, norm
    return qkv, None


def text_segment(layout) -> tuple[int, int]:
    text = [s for s in layout.segments if s[2] == "text"]
    if len(text) != 1:
        raise RuntimeError(f"expected one text segment in the packed layout, got {text}")
    return int(text[0][0]), int(text[0][1])


@torch.no_grad()
def negative_text_kv(state, cfg: NAGConfig, block_index: int, *, heads: int, head_dim: int,
                     device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the once-refined negative text through one block's qkv projection.

    The sidecar hidden state is frozen at the refined embedding: it is not
    re-evolved through earlier blocks and skips per-step modulated norms.
    """
    runtime = state.nag_runtime
    if runtime is None:
        runtime = state.nag_runtime = NAGRuntime()
    cached = runtime.kv_cache.get(block_index)
    if cached is not None:
        return cached

    if runtime.refined is None:
        negative = cfg.negative_context.to(device=device, dtype=dtype)
        preprocess = getattr(state.diffusion, "preprocess_text_embeds", None)
        runtime.refined = preprocess(negative) if preprocess is not None else negative

    blocks = state.blocks
    if blocks is None or not 0 <= block_index < len(blocks):
        raise RuntimeError(f"no DiT block module available for index {block_index}")
    qkv_module, norm_module = discover_projection(blocks[block_index])
    if qkv_module is None:
        raise RuntimeError(
            f"could not locate a fused qkv projection on DiT block {block_index}; "
            f"tried {', '.join(_QKV_PATHS)}"
        )

    hidden = runtime.refined
    if norm_module is not None:
        try:
            hidden = norm_module(hidden)
        except TypeError:
            # Modulated norms need per-step conditioning the sidecar does not carry.
            pass
    qkv = qkv_module(hidden)
    inner = qkv.shape[-1] // 3
    if qkv.shape[-1] != 3 * inner or inner != heads * head_dim:
        raise RuntimeError(
            f"qkv projection width {qkv.shape[-1]} does not factor into 3 x {heads} heads x {head_dim} dims"
        )
    tokens = int(qkv.shape[1])
    k = qkv[..., inner:2 * inner].reshape(1, tokens, heads, head_dim).transpose(1, 2).to(dtype)
    v = qkv[..., 2 * inner:].reshape(1, tokens, heads, head_dim).transpose(1, 2).to(dtype)
    runtime.kv_cache[block_index] = (k, v)
    return k, v


def apply_nag(state, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, *,
              skip_output_reshape: bool) -> torch.Tensor:
    """Inject the NAG text-guidance delta into target audio/video attention rows."""
    cfg = state.nag
    if cfg is None or state.layout is None or state.block_index is None:
        return out
    if not cfg.first_block <= state.block_index <= cfg.last_block:
        return out
    if cfg.alpha <= 0.0 or cfg.scale <= 1.0:
        return out
    if state.current_sigma is not None and state.current_sigma < cfg.sigma_end:
        return out
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != state.layout.seq_len:
        return out

    seg = target_segments(state.layout)
    t0, t1 = text_segment(state.layout)
    if t1 <= t0:
        return out
    heads, head_dim = int(q.shape[1]), int(q.shape[-1])
    k_neg, v_neg = negative_text_kv(state, cfg, state.block_index, heads=heads, head_dim=head_dim,
                                    device=q.device, dtype=q.dtype)
    if cfg.mode == "faithful_selective":
        # Genuinely shared softmax: rerun full-key attention for target rows with
        # the text partition swapped to the negative sidecar.
        k_pos_all, v_pos_all = k, v
        k_neg_all = torch.cat([k[:, :, :t0], k_neg, k[:, :, t1:]], dim=2)
        v_neg_all = torch.cat([v[:, :, :t0], v_neg, v[:, :, t1:]], dim=2)
    else:
        k_pos_all, v_pos_all = k[:, :, t0:t1], v[:, :, t0:t1]
        k_neg_all, v_neg_all = k_neg, v_neg

    out = out.clone()
    for start, stop, strength in ((seg.audio_start, seg.audio_stop, cfg.audio_strength),
                                  (seg.video_start, seg.video_stop, cfg.video_strength)):
        if strength == 0.0 or stop <= start:
            continue
        q_rows = q[:, :, start:stop, :]
        z_pos = F.scaled_dot_product_attention(q_rows, k_pos_all, v_pos_all)
        z_neg = F.scaled_dot_product_attention(q_rows, k_neg_all, v_neg_all)
        delta = nag_combine(z_pos, z_neg, cfg.scale, cfg.tau, cfg.alpha) - z_pos
        if strength != 1.0:
            delta = delta * strength
        if skip_output_reshape:
            out[:, :, start:stop, :].add_(delta.to(out.dtype))
        else:
            flat = delta.transpose(1, 2).reshape(1, stop - start, heads * head_dim)
            out[:, start:stop, :].add_(flat.to(out.dtype))
    state.nag_calls += 1
    return out
