# p21 (22:25): Hakon 22:20 "the fix isn't the sixteen step early handover it's a better prompt".
# Change vs p20 (first frame + lake RefMod + take b text): the held pose that ends beat 1 and opens beat 2 is now ONE identical
# passage that also states Tairn's SIZE and PLACE in frame terms (as the Flight field rule did with "small/large in the frame").
import json
POSE=("Tairn stands on all four legs on the rock just to the right of Sgaeyl, between the camera and the water, side-on to the camera and facing left toward her, with his wings folded. "
"He is enormous and close to the camera: his body fills the right half of the frame from the rock up past the horizon, his horned head is in the upper left of the frame, higher than the moon, and the tips of his horns almost touch the top of the frame. "
"Sgaeyl's raised head reaches only as high as his chest, and the moon is partly hidden behind his folded wing. A limp dead stag hangs from his jaws. He holds still, looking down at her.")
old1="He stands on all four legs beside her at the water's edge, side-on to the camera and towering over her, with the limp stag hanging from his jaws, and holds still, looking down at her."
old2=("Sgaeyl lies on the rock at the left of the ledge beside the water with her head raised, and Tairn stands on all four legs beside her at the water's edge, side-on to the camera and towering over her, his wings folded and a limp dead stag hanging from his jaws. He holds still, looking down at her.")
oldland="lands heavily at the water's edge to the right of Sgaeyl, the same distance from the camera as she is, raising a low cloud of dust."
newland="lands heavily on the rock just to the right of Sgaeyl, between the camera and the water, raising a low cloud of dust."
a=json.load(open("take-datenight-dragons-firstframe-w2x15-b.json")); s=json.loads(a["35"]["inputs"]["shots_json"])
assert old1 in s[0]["text"] and oldland in s[0]["text"] and old2 in s[1]["text"]
s[0]["text"]=s[0]["text"].replace(oldland,newland).replace(old1,POSE)
s[1]["text"]=s[1]["text"].replace(old2,"Sgaeyl lies on the rock at the left of the ledge beside the water with her head raised. "+POSE)
a["35"]["inputs"]["shots_json"]=json.dumps(s)
name="datenight-dragons-firstframe-w2x15-c"; json.dump(a,open(f"take-{name}.json","w"))
q=json.load(open("queue-sweep.json")); nm=f"t-{name}-lead"
if not any(e["name"]==nm for e in q):
    q.append({"name":nm,"api":f"take-{name}.json","opts":{"handoff":"lead","lead":0.5},"seed":42}); json.dump(q,open("queue-sweep.json","w"),indent=1); print("appended",nm)
print([len(x["text"]) for x in s]); print(s[1]["text"][s[1]["text"].index("Sgaeyl lies"):][:1500])
