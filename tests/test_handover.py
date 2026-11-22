import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from h3forge.beat_refs import BEAT_REFS_KEY, parse_beat_refs
from h3forge.context import ContextPolicy, make_context_wrapper
from h3forge.handover import HANDOVER_KEY
from h3forge.prompt import encode_pipe_prompt, make_segmented_extra_conds
from test_audit_repairs import _context_inputs
from test_beat_refs import MODS, _executor, _take
from test_nodes import _import_nodes


class ByteTokenizer:
    """Groups leading space with words and newlines with punctuation. Unicode may span tokens."""
    def __init__(self):
        self.tokenizer = self
        self.pieces = []

    def tokenize_with_weights(self, text, **kwargs):
        pieces = re.findall(rb" ?[a-zA-Z]+|[.!?]\n+|.", text.encode(), flags=re.DOTALL)
        tokens = []
        for piece in pieces:
            if piece not in self.pieces:
                self.pieces.append(piece)
            tokens.append((self.pieces.index(piece), 1.0))
        return [tokens]

    def decode(self, ids, **kwargs):
        return b"".join(self.pieces[i] for i in ids).decode(errors="replace")


class Clip:
    def __init__(self):
        self.qwen = ByteTokenizer()
        self.tokenizer = SimpleNamespace(qwen3vl_32b=self.qwen)
        self.encoded = []

    def tokenize(self, text, minimax_ref_items=None):
        tokens = self.qwen.tokenize_with_weights(text)[0]
        if minimax_ref_items:
            tokens = [*self.qwen.tokenize_with_weights("<Picture 1>: ")[0], ({"image": True}, 1.0),
                      *self.qwen.tokenize_with_weights("<Audio 1>: ")[0], *tokens]
        return {"qwen3vl_32b": [tokens]}

    def encode_from_tokens_scheduled(self, tokens):
        ids, tags, words = [], [], []
        for token, weight in tokens["qwen3vl_32b"][0]:
            if isinstance(token, dict):
                ids.extend([1000] * 8)  # A single vision entry expands into eight Qwen rows.
                tags.extend([0] * 8)
            else:
                ids.append(token)
                words.append(token)
                tags.append(1)
        ids = torch.tensor(ids, dtype=torch.float32)
        raw = torch.stack((ids, ids.cumsum(dim=0)), dim=-1).unsqueeze(0)
        self.encoded.append(self.qwen.decode(words))
        return [[raw, {"minimax_token_tags": torch.tensor(tags)}]]


END = "The café door is open 🚪."
GLOBAL = "subject_definitions: The hall. " + END
BEATS = [
    "detailed_description: [Shot 1] " + END + " She turns. " + END
    + "\n\noverall_soundscape: Brass rings.\nnon_diegetic_music: Drums.",
    "detailed_description: [Shot 1] She crosses. The child waves!\n\noverall_soundscape: Footsteps.",
    "detailed_description: He walks on.",
]


@pytest.mark.parametrize("mode", ["reencode", "token_reuse"])
@pytest.mark.parametrize("references", [False, True])
def test_pipe_nodes_build_only_the_intended_sentence_variant(monkeypatch, mode, references):
    nodes = _import_nodes(monkeypatch)
    clip = Clip()
    native_calls = []
    shared = [{"kind": "audio", "ref_audio_t": 4, "audio_latent": torch.ones(1, 1, 2, 4)}]
    args = dict(prompt=" | ".join(BEATS), global_prompt=GLOBAL, text_handover=mode, segment_durations="2,3,2")
    if references:
        core = ModuleType("comfy_extras.nodes_minimax_h3")

        def execute(**kwargs):
            native_calls.append(kwargs)
            tokens = kwargs["clip"].tokenize(kwargs["prompt"], minimax_ref_items=["picture", "voice"])
            result = kwargs["clip"].encode_from_tokens_scheduled(tokens)
            result[0][1]["minimax_refs"] = shared
            return result, {"samples": "native"}

        core.MiniMaxH3ReferenceToVideo = SimpleNamespace(execute=execute)
        monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", core)
        node = nodes.H3ForgeReferencePipePrompt()
        cond, latent = node.encode(clip, width=128, height=128, length=124, vae=object(), audio_vae=object(),
                                   ref_image_1=object(), ref_audio_1=object(), **args)
        assert len(native_calls) == 1 and cond[0][1]["minimax_refs"] is shared and latent["samples"] == "native"
    else:
        node = nodes.H3ForgePipePrompt()
        cond, = node.encode(clip, **args)
    sockets = node.INPUT_TYPES()["optional"]
    assert sockets["text_handover"][1]["default"] == "off" and sockets["handover_steps"][1]["default"] == 16
    assert sockets["handover_sigmas"][0] == "SIGMAS"

    meta = cond[0][1]
    native, native_tags = meta["h3forge_prompt_segments"], meta["h3forge_prompt_segment_tags"]
    early = meta[HANDOVER_KEY]
    assert early["steps"] == 16 and early["cutoff"] is None and early["mode"] == mode
    assert meta["h3forge_prompt_segment_durations"] == (2, 3, 2) and len(early["segments"]) == 2
    prefix = "<Picture 1>: <Audio 1>: " if references else ""
    assert clip.encoded[:3] == [prefix + GLOBAL + "\n\n" + beat for beat in BEATS]
    if mode == "reencode":
        assert clip.encoded[3:] == [
            prefix + GLOBAL + "\n\n" + BEATS[1].replace("She crosses.", END + " She crosses."),
            prefix + GLOBAL + "\n\n" + BEATS[2].replace("He walks", "The child waves! He walks")]
        old_h = native[1].shape[1]
        assert early["segments"][0].shape[1] > old_h
        assert torch.equal(torch.where(early["tags"][0] == 0)[0], torch.where(native_tags[1] == 0)[0])
    else:
        assert len(clip.encoded) == 3
        for i, sentence in enumerate((END, "The child waves!")):
            merged = early["segments"][i]
            incoming = native[i + 1]
            assert torch.equal(merged[:, :incoming.shape[1]], incoming)
            copied = merged[:, incoming.shape[1]:]
            copied_ids = copied[0, :, 0].int().tolist()
            assert clip.qwen.decode(copied_ids).strip() == sentence
            assert torch.all(early["tags"][i][incoming.shape[1]:] == 1)
            # Punctuation shares a token with the following whitespace. Contextual values must
            # identify the final occurrence in the actual outgoing beat, including its visual prefix.
            audio_tail = clip.qwen.tokenize_with_weights(BEATS[i].split("\n\n", 1)[1])[0]
            stop = native[i].shape[1] - len(audio_tail)
            assert torch.equal(copied, native[i][:, stop - copied.shape[1]:stop])


@pytest.mark.parametrize("mode,steps,prompt", [
    ("off", 16, "first | second"), ("token_reuse", 0, "first | second"),
    ("reencode", 16, "only one beat")])
def test_disabled_and_single_beat_use_the_native_encoding(mode, steps, prompt):
    clip = Clip()
    actual = encode_pipe_prompt(clip, prompt, text_handover=mode, handover_steps=steps)
    native_calls = list(clip.encoded)
    expected = encode_pipe_prompt(clip, prompt)
    assert native_calls == prompt.split(" | ")
    assert actual[0][1].keys() == expected[0][1].keys() and HANDOVER_KEY not in actual[0][1]
    assert torch.equal(actual[0][0], expected[0][0])


class Cond:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return Cond(cond)


@pytest.mark.parametrize("mode", ["reencode", "token_reuse"])
@pytest.mark.parametrize("frozen", [False, True])
def test_selected_context_tags_refs_audio_and_native_tail(monkeypatch, capsys, mode, frozen):
    clip = Clip()
    sigmas = torch.linspace(1, 0, 31)
    prompt = ("integrated_multimodal_description: She rises. Her hand reaches the latch. | "
              "integrated_multimodal_description: She steps through the open arch and walks across the high bridge.")
    native = encode_pipe_prompt(clip, prompt)
    edited = encode_pipe_prompt(clip, prompt, text_handover=mode, handover_sigmas=sigmas if frozen else None)
    x, _, template = _take(monkeypatch)
    refines = []

    def refine(raw):
        refines.append(raw)
        return raw * 2 + 1

    def base_extra(**kwargs):
        return {"c_crossattn": Cond(refine(kwargs["cross_attn"])), "minimax_payload": Cond(template)}

    prepare = make_segmented_extra_conds(base_extra, SimpleNamespace(get_dtype_inference=lambda: torch.float32),
                                         SimpleNamespace(preprocess_text_embeds=refine))
    base = prepare(device="cpu", cross_attn=native[0][0], **native[0][1])["minimax_payload"].cond
    refines.clear()
    payload = prepare(device="cpu", cross_attn=edited[0][0], **edited[0][1])["minimax_payload"].cond
    assert len(refines) == 3  # two native variants, one additional refinement for the incoming window
    assert refines[-1] is edited[0][1][HANDOVER_KEY]["segments"][0]
    policy = ContextPolicy(window_frames=25, overlap_frames=8, stagger=False, handoff="lead")
    wrapper = make_context_wrapper(policy)
    spec = parse_beat_refs("dragons, lake | dragons, hall", MODS)

    def evaluate(prepared, idx, schedule):
        seen = []

        def run(local_x, timestep, text, opts, **kwargs):
            local = kwargs["minimax_payload"]
            seen.append((text, local["text_token_tags"], local["layout"].position_ids.clone(),
                         [float(z.flatten()[0]) for z in local["cond_video_latents"]], local_x[1].shape))
            marker = text.mean() + local["layout"].position_ids[-1, 0].float()
            return [torch.full_like(stream, marker) for stream in local_x]

        opts = {"sample_sigmas": schedule, "sigmas": sigmas[idx:idx + 1], BEAT_REFS_KEY: spec}
        result = wrapper(_executor(run), x, sigmas[idx:idx + 1] * 1000, native[0][0], opts, minimax_payload=prepared)
        assert len(seen) == 2 and seen[0][3] == [9, 1, 2] and seen[1][3] == [9, 1, 3]
        assert all(s[4][-1] < x[1].shape[-1] for s in seen)
        return result, seen

    steps = (0, 15, 0, 15, 16, 29) if frozen else tuple(range(30))
    for idx in steps:
        schedule = (sigmas[:17] if idx < 16 else sigmas[16:]) if frozen else sigmas
        result, seen = evaluate(payload, idx, schedule)
        expected, original = evaluate(base, idx, schedule)
        selected = payload[HANDOVER_KEY] if idx < 16 else None
        assert torch.equal(seen[0][0], base["h3forge_prompt_segments"][0])
        assert seen[1][0] is (selected["segments"][1] if selected else payload["h3forge_prompt_segments"][1])
        assert seen[1][1] is (selected["tags"][1] if selected else payload["h3forge_prompt_segment_tags"][1])
        if idx >= 16:
            assert all(torch.equal(a, b) for a, b in zip(result, expected))
            assert all(torch.equal(a[2], b[2]) for a, b in zip(seen, original))
        else:
            assert torch.equal(seen[0][2], original[0][2])
            incoming_rows, native_rows = seen[1][0].shape[1], original[1][0].shape[1]
            assert incoming_rows > base["layout"].signature[0]
            assert torch.equal(seen[1][2][incoming_rows:], original[1][2][native_rows:])
            assert float(seen[1][2][0, 0]) == base["layout"].signature[0] - incoming_rows
            assert not torch.equal(result[0], expected[0])
    assert "phase=native" in capsys.readouterr().out
    if not frozen:
        with pytest.raises(ValueError, match="connect handover_sigmas"):
            evaluate(payload, 0, sigmas[:17])
    assert len(refines) == 3


def test_audio_only_beat_and_invalid_schedule_are_rejected():
    clip = Clip()
    with pytest.raises(ValueError, match="visual description"):
        encode_pipe_prompt(clip, "overall_soundscape: bells. | she arrives.", text_handover="reencode")
    with pytest.raises(ValueError, match="descending"):
        encode_pipe_prompt(clip, "first | second", text_handover="reencode",
                           handover_sigmas=torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("mode", ["reencode", "token_reuse"])
@pytest.mark.parametrize("outgoing,ending", [
    ("He says once: <d>[English] Keep it steady. Set it down.</d> He closes his mouth.", "He closes his mouth."),
    ("The bell rings. She says <d>Wait. Come here!</d>", "She says <d>Wait. Come here!</d>"),
    ("门开了。她走出去。", "她走出去。"),
])
def test_final_visual_sentence_respects_native_dialogue_and_cjk(mode, outgoing, ending):
    clip = Clip()
    cond = encode_pipe_prompt(clip, outgoing + "\n\noverall_soundscape: Room tone. | The gate opens.",
                              text_handover=mode)
    if mode == "reencode":
        assert clip.encoded[-1] == ending + " The gate opens."
    else:
        meta = cond[0][1]
        suffix = meta[HANDOVER_KEY]["segments"][0][:, meta["h3forge_prompt_segments"][1].shape[1]:]
        assert clip.qwen.decode(suffix[0, :, 0].int().tolist()).strip() == ending


@pytest.mark.parametrize("prepacked", [True, False])
def test_repeated_windows_select_by_beat_on_the_native_packed_layout(prepacked):
    x, context, payload = _context_inputs(total=67, text_lengths=(3, 5, 4))
    if not prepacked:
        del payload["layout"]
    early = (context, torch.full((1, 8, 8), 12.0), torch.full((1, 6, 8), 23.0))
    payload[HANDOVER_KEY] = {"mode": "token_reuse", "steps": 16, "cutoff": 0.5,
                             "segments": early, "tags": tuple(torch.ones(c.shape[1]) for c in early)}
    wrapper = make_context_wrapper(ContextPolicy(25, 8, False))

    def run(phase):
        seen = []

        def evaluate(local_x, timestep, text, options, **kwargs):
            layout = kwargs["minimax_payload"]["layout"]
            seen.append((int(text[0, 0, 0]), layout.position_ids.clone(), text.shape[1]))
            return local_x

        wrapper(_executor(evaluate), x, phase * 1000, context,
                {"sigmas": phase, "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}, minimax_payload=payload)
        return seen

    edited, native = run(torch.tensor([1.0])), run(torch.tensor([0.5]))
    assert [s[0] for s in edited] == [1, 12, 12, 23]
    assert [s[0] for s in native] == [1, 2, 2, 3]
    for (_, edited_pos, e), (_, native_pos, n) in zip(edited, native):
        assert torch.equal(edited_pos[e:], native_pos[n:])
        assert float(edited_pos[0, 0]) == min(0, 5 - e)
