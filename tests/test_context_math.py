import torch

from h3forge.context import blend_weights, window_starts
from h3forge.layout import expand_audio_range, padded_spatial_shape


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
