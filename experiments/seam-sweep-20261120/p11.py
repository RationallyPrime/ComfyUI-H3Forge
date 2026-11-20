# p11 (20261120 18:35): standalone-beat rule applied to every beat. Each beat's first sentence states the
# opening pose and place, and that is the pose and place the previous beat ends on. Strict identical block.
# lead 0.5, seed 42. Usage (in /workspace/seam-gpu): python3 p11.py <queuefile> <athebyne-xaden|flightfield-violet-run>
import json, os, sys
qf, name = sys.argv[1], sys.argv[2]
key = "black trousers and black boots, "
T = {
 "athebyne-xaden": ("is the only person in the shot.", [
 ("He stands at the far end of the path, right in front of the gateway and small in the frame, with his back to the camera. He tips his head back and looks up at the storm clouds, then looks at the gateway. He turns around on the spot to face the camera and stands with his feet apart, the wind pulling at his hair.",
  "Dry wind hisses through desert brush and thunder rumbles a long way off. Boot soles turn with a gritty scrape on stone, and leather creaks."),
 ("He walks toward the camera along the paving stones at an even, unhurried pace, his eyes on the ground ahead of him. He stops a few paces from the camera and lifts his eyes to look straight ahead. He stands still with his feet apart and his arms at his sides.",
  "Slow, heavy boot steps on paving stones come closer and stop. Dry wind hisses through desert brush, and thunder rumbles a long way off."),
 ("He stands upright and still a few paces from the camera, large in the frame, with his feet apart and his arms at his sides, looking straight ahead. After a moment he crouches down, scoops up a handful of red dust from between the paving stones, and lets it run out of his fist so the wind carries it away to the right. He stands up, brushes his palm on his thigh, and looks back over his left shoulder at the fortress.",
  "Dry wind hisses through desert brush. Leather creaks as he crouches. Dry dust hisses from a hand into the wind. A palm brushes leather, and thunder rumbles a long way off.")]),
 "flightfield-violet-run": ("is the only person on the field.", [
 ("She stands a few paces from the camera, large in the frame, with her hands on her hips, rolls her neck from side to side and bounces twice on her toes. She turns around on the spot so her back is to the camera and jogs away straight down the field toward the fortress until she is far down the field and small in the frame.",
  "An open mountain ledge with a light, steady wind. Gravel crunches under two small hops, then under a turning boot, then in an even jogging rhythm that moves away."),
 ("She is far down the field, small in the frame, jogging away from the camera with her back to it. She slows, bends and touches the gravel with her right hand, then turns around to face the camera. She breaks into a hard sprint straight toward the camera, arms pumping, growing larger in the frame until she is close to the camera.",
  "Distant jogging steps on gravel slow, a hand pats the ground, and then fast hard footfalls build and come closer. Light mountain wind."),
 ("She is close to the camera and large in the frame, sprinting straight toward it with her arms pumping. She slows over three strides and stops a few paces from the camera. She bends forward with her hands on her knees, breathing hard, her shoulders rising and falling. She straightens up, puts her hands on her hips, tips her face up to the sun and breathes out.",
  "Fast footfalls on gravel slow and skid to a stop. Hard, ragged breathing, slowly steadying, and one long breath out. Light mountain wind.")])}
who, beats = T[name]
q = json.load(open(qf))
a = json.load(open(f"take-{name}.json")); shots = json.loads(a["35"]["inputs"]["shots_json"]); assert len(shots) == 3
blocks = []
for s, (action, sound) in zip(shots, beats):
    block = s["text"][:s["text"].index(key) + len(key)] + who; blocks.append(block)
    s["text"] = block + " " + action + "\n\noverall_soundscape: " + sound + "\n\nnon_diegetic_music: N/A"
assert len(set(blocks)) == 1
a["35"]["inputs"]["shots_json"] = json.dumps(shots); json.dump(a, open(f"take-{name}-p11.json", "w"))
n = f"t-{name}-p11-lead"
if not any(e["name"] == n for e in q):
    fin = set(os.listdir("done")) | set(os.listdir("failed"))
    i = next(k for k, e in enumerate(q) if e["name"] not in fin)  # the entry running now
    q.insert(i + 1, {"name": n, "api": f"take-{name}-p11.json", "opts": {"handoff": "lead", "lead": 0.5}, "seed": 42})
    json.dump(q, open(qf, "w"), indent=1); print("queued after", q[i]["name"])
print([len(s["text"]) for s in shots])
