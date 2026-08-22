import torch

from h3forge.context import blend_weights, window_starts
from h3forge.layout import expand_audio_range, padded_spatial_shape
from h3forge.prompt import (
    blend_segment_contexts,
    encode_pipe_prompt,
    make_segmented_extra_conds,
    pad_segment_contexts,
    segment_overlap_weights,
    split_pipe_prompt,
)


def test_windows_cover_full_range():
    starts = window_starts(61, 25, 5, 0)
    covered = set()
    for s in starts:
        covered.update(range(s, min(s + 25, 61)))
    assert covered == set(range(61))
    assert starts[0] == 0
    assert starts[-1] == 36


def test_phase_still_covers_full_range():
    starts = window_starts(61, 25, 5, 7)
    covered = set()
    for s in starts:
        covered.update(range(s, min(s + 25, 61)))
    assert covered == set(range(61))


def test_blend_positive():
    w = blend_weights(25, 5, device="cpu", dtype=torch.float32)
    assert torch.all(w > 0)
    assert w[0] < w[5]
    assert w[-1] < w[-6]


def test_odd_latent_spatial_shape_uses_post_pad_dimensions():
    assert padded_spatial_shape(95, 167, (1, 2, 2)) == (96, 168)


def test_audio_ranges_can_be_pinned_to_one_compiled_length():
    ranges = [(0, 141), (139, 282), (284, 426)]
    expanded = [expand_audio_range(r, total=426, target_length=143) for r in ranges]
    assert {a1 - a0 for a0, a1 in expanded} == {143}
    for old, new in zip(ranges, expanded):
        assert new[0] <= old[0] < old[1] <= new[1]


def test_pipe_prompt_split_and_escape():
    assert split_pipe_prompt(r"first | second with a \| literal | third") == [
        "first", "second with a | literal", "third",
    ]


def test_pipe_prompt_rejects_empty_segments():
    import pytest

    with pytest.raises(ValueError, match="empty segment"):
        split_pipe_prompt("first || third")


def test_segment_contexts_pad_to_one_compiled_shape():
    contexts = [torch.ones(1, 2, 3), torch.full((1, 4, 3), 2.0)]
    padded = pad_segment_contexts(contexts)
    assert [tuple(context.shape) for context in padded] == [(1, 4, 3), (1, 4, 3)]
    assert torch.equal(padded[0][:, :2], contexts[0])
    assert torch.count_nonzero(padded[0][:, 2:]) == 0


def test_segment_overlap_blends_boundary_windows():
    weights = segment_overlap_weights(20, 40, total=60, count=3)
    assert weights == [0.0, 1.0, 0.0]

    weights = segment_overlap_weights(15, 25, total=60, count=3)
    assert weights == [0.5, 0.5, 0.0]
    contexts = [torch.zeros(1, 2, 1), torch.full((1, 2, 1), 2.0), torch.full((1, 2, 1), 4.0)]
    assert torch.equal(blend_segment_contexts(contexts, weights), torch.ones(1, 2, 1))


def test_pipe_segments_are_encoded_independently_and_annotated():
    class Clip:
        @staticmethod
        def tokenize(text):
            return text

        @staticmethod
        def encode_from_tokens_scheduled(text):
            tokens = len(text.split())
            return [[torch.full((1, tokens, 2), float(tokens)), {"minimax_token_tags": torch.ones(tokens)}]]

    conditioning = encode_pipe_prompt(Clip(), "short text | a deliberately longer segment")
    context, metadata = conditioning[0]
    assert tuple(context.shape) == (1, 4, 2)
    assert metadata["h3forge_prompt_segment_count"] == 2
    assert [tuple(x.shape) for x in metadata["h3forge_prompt_segments"]] == [(1, 4, 2), (1, 4, 2)]


def test_segment_contexts_are_carried_through_minimax_payload():
    class Cond:
        def __init__(self, cond):
            self.cond = cond

        def _copy_with(self, cond):
            return Cond(cond)

    class BaseModel:
        @staticmethod
        def get_dtype_inference():
            return torch.float32

    class Diffusion:
        @staticmethod
        def preprocess_text_embeds(context):
            return context + 1

    def base_extra_conds(**kwargs):
        return {"minimax_payload": Cond({"seed": kwargs.get("seed", 0)})}

    wrapper = make_segmented_extra_conds(base_extra_conds, BaseModel(), Diffusion())
    raw = (torch.zeros(1, 3, 2), torch.ones(1, 3, 2))
    result = wrapper(device="cpu", seed=7, h3forge_prompt_segments=raw)
    payload = result["minimax_payload"].cond
    assert payload["seed"] == 7
    assert torch.equal(payload["h3forge_prompt_segments"][0], torch.ones_like(raw[0]))
    assert torch.equal(payload["h3forge_prompt_segments"][1], torch.full_like(raw[1], 2))
