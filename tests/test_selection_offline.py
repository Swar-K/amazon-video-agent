"""Offline test: free pre-scan + automatic footage selection + usage memory.   python tests/test_selection_offline.py"""
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from adagent import assets as A
from adagent import selection as SEL
from adagent.common import load_config
from adagent.project import load_project

cfg = load_config()
cfg["assets"].update(images_per_video=6, clips_per_video=1)
D = ROOT / "projects" / "TESTSEL01"
shutil.rmtree(D, ignore_errors=True)
(D / "assets").mkdir(parents=True)
(D / "facts.yaml").write_text("asin: TESTSEL01\nproduct_name: Select Test Bottle\ncategory: bottle\naudience: people\n"
                              "affiliate_url: https://amzn.to/3abc\nhands_on_notes: [a, b, c]\n")
rng = np.random.default_rng(7)
HUES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 20]


def good(i):
    img = np.zeros((960, 1280, 3), np.uint8)
    hsv = np.uint8([[[HUES[i % len(HUES)], 90 + 10 * (i % 5), 70 + 12 * (i % 7)]]])
    img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    for k in range(5 + i % 4):                                # sharp, different shapes/positions per image
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        x, y = int(rng.integers(50, 1000)), int(rng.integers(50, 700))
        (cv2.rectangle(img, (x, y), (x + int(rng.integers(60, 260)), y + int(rng.integers(60, 200))), c, -1) if k % 2 else
         cv2.circle(img, (x, y), int(rng.integers(40, 150)), c, -1))
    return img


goods = [good(i) for i in range(13)]
for i, g in enumerate(goods):
    cv2.imwrite(str(D / "assets" / f"good_{i:02d}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 92])
for k in range(3):                                            # near-duplicates of good_00 (tiny brightness change)
    cv2.imwrite(str(D / "assets" / f"dup_{k}.jpg"), cv2.convertScaleAbs(goods[0], alpha=1.0, beta=k + 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
for k in range(2):
    cv2.imwrite(str(D / "assets" / f"blurry_{k}.jpg"), cv2.GaussianBlur(goods[3 + k], (0, 0), 25))
cv2.imwrite(str(D / "assets" / "tiny.jpg"), cv2.resize(goods[5], (200, 150)))
cv2.imwrite(str(D / "assets" / "blank.jpg"), np.full((900, 1200, 3), 128, np.uint8))


def clip(path, blur):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for f in range(25 * 8):
        fr = np.full((360, 640, 3), (60, 90, 40), np.uint8)
        for k in range(12):
            cv2.circle(fr, (40 + 50 * k, 60 + 20 * (k % 5)), 20, (30 * k % 255, 200, 255 - 20 * k), -1)
        x = 20 + f * 2 if f > 60 else 20                      # nothing moves in the first 2.4 s, then a box moves
        cv2.rectangle(fr, (x, 200), (x + 90, 300), (255, 255, 255), -1)
        w.write(cv2.GaussianBlur(fr, (0, 0), 12) if blur else fr)
    w.release()


clip(D / "assets" / "clip_sharp.mp4", False)
clip(D / "assets" / "clip_blurry.mp4", True)
p = load_project("TESTSEL01")
files = sorted(x.name for x in (D / "assets").iterdir())

# ---- 1. pick: only good, varied, no junk; AI checks only for the pick
ai = []
A.describe = lambda cfg, frames, kind: (ai.append(1), {"description": "x", "shows_product": True, "has_human_face": False,
                                                       "has_price_or_rating_ui": False, "has_third_party_logo_or_watermark": False})[1]
ents = A.build_index_for(p, cfg, vision=True, lock="write")
names = [e["file"] for e in ents]
imgs = [n for n in names if n.endswith(".jpg")]
assert len(imgs) == 6 and [n for n in names if n.endswith(".mp4")] == ["clip_sharp.mp4"], names
assert not any(n.startswith(("blurry", "tiny", "blank")) for n in names), names
assert sum(1 for n in names if n in ("good_00.jpg",) or n.startswith("dup_")) <= 1, "one of each look-alike group"
assert len(ai) == len(names) == 7, f"AI checked only the pick, not all {len(files)} files: {len(ai)}"
all_ids = {n: f"A{i:02d}" for i, n in enumerate(files, 1)}
assert all(e["id"] == all_ids[e["file"]] for e in ents), "ids stay tied to the file, whatever is picked"
sel = json.loads((D / "selection.json").read_text())
assert sorted(sel["products"]["TESTSEL01"]) == sorted(names)
rep = (D / "out" / "footage_selection.md").read_text()
for why in ("too blurry", "too small", "almost blank", "near-duplicate of", "not needed for this video"):
    assert why in rep, why
print(f"  picked {len(imgs)} photos + 1 clip out of {len(files)} files; AI-checked only {len(ai)}; junk skipped with reasons: OK")

# ---- 2. the saved pick is used by 'render' even when new files arrive; --reselect / script pick again
cv2.imwrite(str(D / "assets" / "good_new.jpg"), good(12))
again = [e["file"] for e in A.build_index_for(p, cfg, vision=True, lock="read")]
assert sorted(again) == sorted(names), "saved pick is reused (script and render see the same files)"
assert json.loads((D / "selection.json").read_text()) == sel, "read mode never rewrites selection.json"
ai.clear()
A.build_index_for(p, cfg, vision=True, lock="read")
assert ai == [], "nothing is AI-checked twice"
print("  selection.json is reused by render, never rewritten by it: OK")

# ---- 3. usage memory: the next video of the same product uses different footage
shots = [SimpleNamespace(asset=e, clip_in=3.0 if e["kind"] == "clip" else 0.0) for e in ents for _ in range(2)]
proj_stub = SimpleNamespace(dir=D, asin="TESTSEL01", compare=None)
SEL.record_usage(proj_stub, shots, "honest_pros_cons", "Video one")
SEL.record_usage(proj_stub, shots, "honest_pros_cons", "Video one")      # re-rendering the same video does not count twice
u = json.loads((D / "asset_usage.json").read_text())
assert len(u["videos"]) == 1 and u["videos"][0]["files"]["clip_sharp.mp4"]["starts"] == [3.0]
nxt = [e["file"] for e in A.build_index_for(p, cfg, vision=True, lock="write", reselect=True) if e["file"].endswith(".jpg")]
assert len(set(nxt) & set(imgs)) <= 1, f"second video overlaps the first on {set(nxt) & set(imgs)}"
cfg_nopen = copy.deepcopy(cfg); cfg_nopen["assets"]["reuse_penalty"] = 0
print(f"  second video reuses {len(set(nxt) & set(imgs))} of 6 photos from the first: OK")

# ---- 4. clip moments: best first, moments an earlier video used last
ft = json.loads((D / ".cache" / "prescan.json").read_text())
clipf = next(v for v in ft.values() if v["kind"] == "clip" and v["quality"] > 0.2)
w = [x["start"] for x in sorted(clipf["windows"], key=lambda x: -x["score"])]
assert len(w) >= 1 and w[0] >= 2.0, f"the best moment is when something moves, not the still start: {w}"
order = SEL.ordered_windows(clipf, [w[0]])
assert order[-1] == w[0] or len(order) == 1, "a moment used before goes last"
e_clip = next(e for e in A.build_index_for(p, cfg, vision=True, lock="read") if e["kind"] == "clip")
assert e_clip["windows"] and all(isinstance(x, float) for x in e_clip["windows"])
print("  clip best moments (moving + sharp first, used ones last): OK")

# ---- 5. few files, pins, select_all, select: false, budgets
c2 = copy.deepcopy(cfg); c2["assets"].update(images_per_video=50, clips_per_video=5)
every = [e["file"] for e in A.build_index_for(p, c2, vision=True, lock="read", reselect=True)]
assert not any(n.startswith(("blurry", "tiny", "blank")) for n in every)
assert sum(1 for n in every if n.endswith(".jpg")) == 14 + 0 or sum(1 for n in every if n.endswith(".jpg")) >= 13, every
(D / "facts.yaml").write_text((D / "facts.yaml").read_text() + "assets_pin: [blurry_0.jpg]\n")
p2 = load_project("TESTSEL01")
assert "blurry_0.jpg" in [e["file"] for e in A.build_index_for(p2, cfg, vision=True, lock="read", reselect=True)], "pinned file is always used"
alln = [e["file"] for e in A.build_index_for(p, cfg, vision=False, lock="read", select_all=True)]
assert sorted(alln) == sorted(x.name for x in (D / "assets").iterdir() if x.suffix != ".json"), "select_all uses every file"
c3 = copy.deepcopy(cfg); c3["assets"]["select"] = False
assert len(A.build_index_for(p, c3, vision=False, lock="read")) == len(alln)
b = SEL.budgets(cfg, None)
assert b == (6, 1)
auto = load_config()
bi0, bc0 = SEL.budgets(auto, None)
assert 8 <= bi0 <= 36 and 3 <= bc0 <= 12, (bi0, bc0)
bi, bc = SEL.budgets(auto, "list", 5)
assert 5 <= bi <= 14 and 2 <= bc <= 5
print("  few files use all good ones; pins; select_all; select: false; auto budgets: OK")
shutil.rmtree(D, ignore_errors=True)
print("OK - selection: free pre-scan, varied/fresh pick, saved choice, usage memory, clip moments, AI checks only for the pick")
