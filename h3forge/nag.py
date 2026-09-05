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
    cache_key: tuple | None = None


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
    re-evolved through earlier blocks. Native timestep modulation, Q/K norm,
    and rotary positioning are applied before the keys are captured.
    """
    from comfy.ldm.minimax.model import rope_rotation_table

    runtime = state.nag_runtime
    if runtime is None:
        runtime = state.nag_runtime = NAGRuntime()
    key = (state.current_sigma, str(device), dtype)
    if key != runtime.cache_key:
        runtime.kv_cache.clear()
        runtime.cache_key = key
    if block_index in runtime.kv_cache:
        return runtime.kv_cache[block_index]
    if runtime.refined is None:
        runtime.refined = state.diffusion.preprocess_text_embeds(cfg.negative_context.to(device=device, dtype=dtype))
    block = state.blocks[block_index]
    args = state.block_args
    if args is None:
        raise RuntimeError("NAG requires the current native H3 block arguments")
    _, text_stop = text_segment(state.layout)
    text_row = next(row for a, _, row in args["mod_segments"]
                    if a < text_stop and isinstance(row, int) and row % 3 == 1)
    shift, scale, *_ = block.adaln_proj(args["t_emb"])
    hidden = block.norm1(runtime.refined[0])
    hidden = hidden * (1 + scale[text_row].to(hidden.dtype)) + shift[text_row].to(hidden.dtype)
    positions = torch.zeros(hidden.shape[0], 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(hidden.shape[0], dtype=torch.float64)
    rope = rope_rotation_table(state.diffusion.rope_freqs(positions, device), dtype)
    captured = []

    def capture(func, q, k, v, native_heads, *args, **kwargs):
        if native_heads != heads or k.shape[-1] != head_dim:
            raise RuntimeError("negative attention shape differs from the native positive stream")
        captured.append((k, v))
        # Native Attention owns Q/K norm, partial rotary layout, and quantized
        # projection semantics. Its normal output projection receives zeros;
        # there is no negative SxS attention or second DiT traversal here.
        return q.new_zeros((1, q.shape[2], native_heads * q.shape[-1]))

    block.attn(hidden, rope_freqs=rope, transformer_options={"optimized_attention_override": capture})
    if len(captured) != 1:
        raise RuntimeError("native H3 attention did not expose exactly one negative K/V pair")
    runtime.kv_cache[block_index] = captured[0]
    return captured[0]


def apply_nag(state, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, *,
              skip_output_reshape: bool, transformer_options: dict | None = None,
              attn_mask: torch.Tensor | None = None, packed_attention=None) -> torch.Tensor:
    """Inject the NAG text-guidance delta into target audio/video attention rows."""
    cfg = state.nag
    if cfg is None or state.layout is None or state.block_index is None:
        return out
    if not cfg.first_block <= state.block_index <= cfg.last_block:
        return out
    if cfg.alpha <= 0.0 or cfg.scale <= 1.0:
        return out
    if cfg.video_strength == 0.0 and cfg.audio_strength == 0.0:
        return out
    if state.current_sigma is not None and state.current_sigma < cfg.sigma_end:
        return out
    cond_or_uncond = (transformer_options or {}).get("cond_or_uncond")
    if cond_or_uncond is not None and any(flag != 0 for flag in cond_or_uncond):
        # NAG guides the positive branch only. Under a CFG guider the uncond
        # forward (batched with cond, or separate under low VRAM) must pass
        # through untouched, or the delta would distort the CFG difference.
        state.note_decline("nag-uncond-forward")
        return out
    if attn_mask is not None:
        # The real attention excluded masked keys; a maskless sidecar delta
        # would be computed over keys the model's output never saw.
        state.note_decline("nag-preexisting-mask")
        return out
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != state.layout.seq_len:
        state.note_decline("nag-unexpected-q-shape")
        return out

    seg = target_segments(state.layout)
    t0, t1 = text_segment(state.layout)
    if t1 <= t0:
        return out
    heads, head_dim = int(q.shape[1]), int(q.shape[-1])
    k_neg, v_neg = negative_text_kv(state, cfg, state.block_index, heads=heads, head_dim=head_dim,
                                    device=q.device, dtype=q.dtype)
    positive = out if skip_output_reshape else out.reshape(1, q.shape[2], heads, head_dim).transpose(1, 2)
    negative = None
    if cfg.mode == "faithful_selective":
        k_neg_all = torch.cat([k[:, :, :t0], k_neg, k[:, :, t1:]], dim=2)
        v_neg_all = torch.cat([v[:, :, :t0], v_neg, v[:, :, t1:]], dim=2)
        if packed_attention is None:
            negative = F.scaled_dot_product_attention(q, k_neg_all, v_neg_all)
        else:
            # Reuse the positive result and the same configured backend/mask for
            # the swapped text. Sparse execution must keep its visibility rule.
            negative = packed_attention(q, k_neg_all, v_neg_all, k_neg.shape[2])

    out = out.clone()
    for start, stop, strength in ((seg.audio_start, seg.audio_stop, cfg.audio_strength),
                                  (seg.video_start, seg.video_stop, cfg.video_strength)):
        if strength == 0.0 or stop <= start:
            continue
        q_rows = q[:, :, start:stop, :]
        if negative is None:
            z_pos = F.scaled_dot_product_attention(q_rows, k[:, :, t0:t1], v[:, :, t0:t1])
            z_neg = F.scaled_dot_product_attention(q_rows, k_neg, v_neg)
        else:
            z_pos, z_neg = positive[:, :, start:stop], negative[:, :, start:stop]
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
