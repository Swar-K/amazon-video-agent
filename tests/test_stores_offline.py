"""Offline test: AliExpress products alongside Amazon.   python tests/make_compare_fixture.py && python tests/test_stores_offline.py"""
import copy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml

from adagent import finish as F
from adagent import scriptgen as S
from adagent import timeline as T
from adagent import tts as V
from adagent.common import load_config
from adagent.project import apply_disclosure, check_text, load_project, store_of, stores_of, validate_facts

cfg = load_config()
P = ROOT / "projects"
ALI = "1005006123456789"
src = P / "TEST000002"
dst = P / ALI
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
f = yaml.safe_load((dst / "facts.yaml").read_text())
f.update(asin=int(ALI), store="aliexpress", affiliate_url="https://s.click.aliexpress.com/e/_DxYz123", product_name="Zeta Sports Bottle 24 oz",
         brand="Zeta", short_name="Zeta")
(dst / "facts.yaml").write_text(yaml.safe_dump(f, sort_keys=False))
shutil.rmtree(dst / ".cache", ignore_errors=True)

# ---- 1. a numeric AliExpress item id works as a folder name; store is read from facts / detected from the link
a = load_project(ALI)
assert store_of(a.facts) == "aliexpress" and stores_of(a) == {"aliexpress"}
errs, warns = validate_facts(a)
assert not errs, errs
assert any("OWN photos" in w for w in warns), warns
det = dict(a.facts); det.pop("store")
assert store_of(det) == "aliexpress", "detected from the s.click.aliexpress.com link"
for url in ("https://a.aliexpress.com/_Abc", "https://www.aliexpress.com/item/1005006123456789.html", "https://it.aliexpress.com/item/1.html"):
    assert store_of({"affiliate_url": url}) == "aliexpress", url
assert store_of({"affiliate_url": "https://amzn.to/3abc"}) == "amazon" and store_of({}) == "amazon"
bad = copy.deepcopy(a); bad.facts["store"] = "amazon"
e, _ = validate_facts(bad); assert any("store: aliexpress" in x for x in e), e
bad.facts["store"] = "temu"
e, _ = validate_facts(bad); assert any("store must be" in x for x in e), e
bad.facts["store"] = "aliexpress"; bad.facts["affiliate_url"] = "https://amzn.to/3abc"
e, _ = validate_facts(bad); assert any("must be an AliExpress link" in x for x in e), e
print("  numeric item ids, store detection, link checks: OK")

# ---- 2. policy: naming a store that is IN the video is fine; shipping / seller / order-count claims are not
assert check_text("The AliExpress one has a clever lid.", {"aliexpress"})[0] == []
assert check_text("Also sold on AliExpress.", set())[0], "still a retailer mention in an Amazon-only video"
for bad_line in ("It ships from overseas in two weeks.", "Delivery in 10 days.", "Free shipping on this one.", "More than 10,000 sold.", "The seller rating is high.",
                 "It has the Choice badge."):
    assert check_text(bad_line, {"aliexpress", "amazon"})[0], bad_line
assert check_text("The lid clicks shut and the body feels light.", {"aliexpress"})[0] == []
print("  policy: store named when it is in the video; shipping/seller/order claims blocked: OK")

# ---- 3. disclosure: Amazon sentence only when an Amazon product is in the video
c1 = copy.deepcopy(cfg); st = apply_disclosure(c1, a)
assert st == {"aliexpress"} and "Amazon" not in c1["disclosure"]["on_screen"] and "affiliate links" in c1["disclosure"]["on_screen"]
assert all("Amazon" not in x for x in c1["disclosure"]["spoken"])
assert c1["disclosure"]["description_line"] == c1["disclosure"]["on_screen"]
c0 = copy.deepcopy(cfg); apply_disclosure(c0, load_project("TEST000001"))
assert c0["disclosure"]["on_screen"].startswith("As an Amazon Associate") and c0["disclosure"]["description_line"].startswith("As an Amazon Associate")

MIX = P / "MIXCMP001"
shutil.rmtree(MIX, ignore_errors=True)
MIX.mkdir()
(MIX / "compare.yaml").write_text(f"title: Zeta vs TestCo bottles\nproducts: [TEST000001, '{ALI}']\ncategory: water bottle\naudience: commuters\n"
                                  "keywords: [water bottle comparison]\n")
m = load_project("MIXCMP001")
assert stores_of(m) == {"amazon", "aliexpress"} and m.compare["labels"] == ["TestCo", "Zeta"]
e, w = validate_facts(m); assert not e, e
c2 = copy.deepcopy(cfg); apply_disclosure(c2, m)
assert c2["disclosure"]["on_screen"].startswith("As an Amazon Associate") and "other stores" in c2["disclosure"]["on_screen"]
assert "Amazon Associate" in c2["disclosure"]["spoken"][0]
print("  disclosure: Amazon sentence only with an Amazon product; mixed videos get both: OK")

# ---- 4. description: store names and the right links
sc = {"title": "Zeta vs TestCo", "description_intro": "x", "tags": [], "angle": "comparison", "review": {}, "segments": []}
dummy = [{"id": "1", "section": "hook", "text": "Two bottles.", "product": "all", "onscreen": "", "asset_ids": [], "priority": 1, "source": "product"}]
sc["segments"] = dummy
out = ROOT / "projects" / "MIXCMP001" / "out"; out.mkdir(exist_ok=True)
segs = S.assemble(sc, m, c2)
arr = V.synthesize_all(m, c2, "mock", segs)
tl = T.build_timeline(segs, arr, c2, enforce=False)
F.write_description(m, c2, sc, tl, {}, out / "description.txt")
d = (out / "description.txt").read_text()
assert d.splitlines()[0].startswith("As an Amazon Associate") and "Products compared:" in d, d[:300]
assert "(Test Bottle 20 oz" in d or "TestCo" in d
assert "[Amazon]: https://amzn.to/" in d and "[AliExpress]: https://s.click.aliexpress.com/e/_DxYz123" in d, d
pin = (out / "pinned_comment.txt").read_text()
assert "Amazon Associate" in pin
single = {"title": "Zeta bottle", "description_intro": "x", "tags": [], "angle": "honest_pros_cons", "review": {}, "segments": dummy}
os1 = a.out_dir; os1.mkdir(parents=True, exist_ok=True)
segs1 = S.assemble(single, a, c1); tl1 = T.build_timeline(segs1, V.synthesize_all(a, c1, "mock", segs1), c1, enforce=False)
F.write_description(a, c1, single, tl1, {}, os1 / "description.txt")
d1 = (os1 / "description.txt").read_text()
assert "Amazon" not in d1.splitlines()[0] and "Product on AliExpress: https://s.click.aliexpress.com/e/_DxYz123" in d1, d1[:300]
assert "Amazon" not in (os1 / "pinned_comment.txt").read_text()
cmp_amz = load_project("CMPTEST001")
c3 = copy.deepcopy(cfg); apply_disclosure(c3, cmp_amz)
sca = copy.deepcopy(sc); sa = S.assemble(sca, cmp_amz, c3); ta = T.build_timeline(sa, V.synthesize_all(cmp_amz, c3, "mock", sa), c3, enforce=False)
F.write_description(cmp_amz, c3, sca, ta, {}, cmp_amz.out_dir / "description.txt") if cmp_amz.out_dir.mkdir(parents=True, exist_ok=True) is None else None
assert "Products compared on Amazon:" in (cmp_amz.out_dir / "description.txt").read_text(), "Amazon-only wording is unchanged"
print("  description / pinned comment: store tags, right links, Amazon-only unchanged: OK")

# ---- 5. the writer is told about stores (and only then)
fy = S.facts_yaml(m)
assert "store: Amazon" in fy and "store: AliExpress" in fy, fy[:400]
assert S.STORE_NOTE in S._guide(m) and S.STORE_NOTE not in S._guide(cmp_amz) and S.COMPARE_GUIDE in S._guide(m)
assert S.facts_yaml(a).startswith("STORE: AliExpress") and not S.facts_yaml(load_project("TEST000001")).startswith("STORE")
for sysmsg in (S.SYSTEM, S.COMPARE_SYSTEM, S.LIST_SYSTEM):
    assert "AliExpress Choice" in sysmsg and "order counts" in sysmsg
# AliExpress-only video, full disclosure mode: the spoken line is the general one
c4 = copy.deepcopy(c1); c4["disclosure"]["mode"] = "full"
sp = [x for x in S.assemble(single, a, c4) if x["id"] == "D"][0]["text"]
assert "affiliate links" in sp and "Amazon" not in sp, sp
print("  writer prompts and spoken disclosure are store-aware: OK")

for p in (dst, MIX):
    shutil.rmtree(p, ignore_errors=True)
print("OK - AliExpress alongside Amazon: single / compare / mixed, numeric ids, disclosure per store, policy, description, prompts")
