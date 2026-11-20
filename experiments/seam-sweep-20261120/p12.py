# Dragon p12 (20261120 19:12): p11 rule on the dragon take. Each beat's first sentence states the pose the previous beat ends on.
import json, os
name = "flightfield-dragon"; k = "facing the camera."
beats = [
 ("It stands on all four legs with its head held low and its wings folded. It lifts its head high on its long neck and turns it to the left to look out over the cliff toward the mountains. It turns its head back toward the camera and snorts, and a puff of dust lifts from the gravel in front of it. It stands still on all four legs with its wings folded, looking at the camera.",
  "An open mountain ledge with a light, steady wind. A huge slow breath, scales rasping as a long neck moves, and one deep snort that scatters gravel."),
 ("It stands still on all four legs with its wings folded against its back and its head toward the camera. Then it slowly unfolds both wings and spreads them wide until they reach across the whole width of the field, and holds them open. It beats them once, hard, raising a low cloud of dust along the ground, and folds them against its back again. It stands on all four legs with its wings folded as thin dust drifts around its feet.",
  "Light mountain wind. Leather wings unfold with a heavy creaking rustle. One huge whoomp of air, gravel skittering away, and the rustle of wings folding."),
 ("It stands on all four legs with its wings folded against its back, thin dust drifting around its feet. The dust settles. Then it lowers its body to the ground, folding its forelegs under its chest, and curls its long tail around its side. It rests its chin on the gravel, eyes on the camera, and blinks slowly once.",
  "Light mountain wind. A huge body settles onto gravel with a long grinding crunch. A tail drags heavily across stones. One deep, slow breath out.")]
a = json.load(open(f"take-{name}-p8.json")); shots = json.loads(a["35"]["inputs"]["shots_json"]); blocks = []
for s, (action, sound) in zip(shots, beats):
    block = s["text"][:s["text"].index(k) + len(k)]; blocks.append(block)
    s["text"] = block + " " + action + "\n\noverall_soundscape: " + sound + "\n\nnon_diegetic_music: N/A"
assert len(set(blocks)) == 1
a["35"]["inputs"]["shots_json"] = json.dumps(shots); json.dump(a, open(f"take-{name}-p12.json", "w"))
q = json.load(open("queue-pod4.json")); n = f"t-{name}-p12-lead"
if not any(e["name"] == n for e in q):
    fin = set(os.listdir("done")) | set(os.listdir("failed")); i = next(j for j, e in enumerate(q) if e["name"] not in fin)
    q.insert(i + 1, {"name": n, "api": f"take-{name}-p12.json", "opts": {"handoff": "lead", "lead": 0.5}, "seed": 42})
    json.dump(q, open("queue-pod4.json", "w"), indent=1); print("queued after", q[i]["name"])
print([len(s["text"]) for s in shots])
