"""Offline test: the paid-voice safeguards (Inworld budget + ledger, OpenRouter opt-in, Edge -> Inworld backup voice).
No network, no API keys, nothing is billed: every HTTP call is replaced by a counter.
    python tests/test_paid_voice_guards_offline.py"""
import base64
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from adagent import tts as V
from adagent.common import load_config

tmp = Path(tempfile.mkdtemp(prefix="paidvoice_"))
calls = []


class R:
    def __init__(self, code=200, js=None, text=""):
        self.status_code, self._js, self.text = code, js, text or str(js)
        self.content = b"fake-audio"
        self.headers = {}

    def json(self):
        return self._js


def fake_post(url, **kw):
    calls.append(url)
    return R(200, {"audioContent": base64.b64encode(b"fake-audio").decode(), "usage": {"processedCharactersCount": 11}})


V.requests.post = fake_post
V._is_valid_audio = lambda p: True                       # the fake bytes are not real audio
V._to_clean_array = lambda raw, out: np.zeros(100, dtype=np.float32)
V.time.sleep = lambda s: None


def fresh_cfg(**inworld):
    cfg = load_config()
    cfg["voice"]["inworld"]["usage_ledger"] = str(tmp / "ledger.json")
    cfg["voice"]["inworld"].update(inworld)
    (tmp / "ledger.json").unlink(missing_ok=True)
    for k in ("INWORLD_API_KEY", "INWORLD_ALLOW_PAID_TTS", "OPENROUTER_API_KEY", "OPENROUTER_ALLOW_PAID_TTS"):
        os.environ.pop(k, None)
    calls.clear()
    return cfg


def ledger():
    return json.loads((tmp / "ledger.json").read_text())


def refused(fn, *a, **kw):
    """True if the call is refused (SystemExit or RuntimeError) and returns the message."""
    try:
        fn(*a, **kw)
    except (SystemExit, RuntimeError) as e:
        return str(e)
    raise AssertionError("expected a refusal")


iw = {"provider": "inworld", "voice_id": "Ashley"}
text = "Hello there, this is eleven."          # 28 chars

# ---- 1. Inworld: no key = refused, no HTTP
cfg = fresh_cfg()
assert "INWORLD_API_KEY" in refused(V._synth_inworld, cfg, iw, text, tmp / "a.mp3", "", "") and not calls
print("  Inworld without a key is refused before any request: OK")

# ---- 2. Inworld success writes the ledger with the provider's own character count
cfg = fresh_cfg(); os.environ["INWORLD_API_KEY"] = "k"
V._synth_inworld(cfg, iw, text, tmp / "a.mp3", "", "")
assert len(calls) == 1 and ledger()["inworld_characters_used"] == 11 and ledger()["inworld_voice_id"] == "Ashley", ledger()
V._synth_inworld(cfg, iw, text, tmp / "b.mp3", "", "")
assert ledger()["inworld_characters_used"] == 22, "the ledger accumulates"
print("  Inworld success: ledger counts characters and accumulates: OK")

# ---- 3. over the safety budget = refused with NO request (also in preflight)
cfg = fresh_cfg(free_character_budget=30); os.environ["INWORLD_API_KEY"] = "k"
(tmp / "ledger.json").write_text(json.dumps({"inworld_characters_used": 10}))
assert "safety budget" in refused(V._synth_inworld, cfg, iw, text, tmp / "c.mp3", "", "") and not calls
segs = [{"text": text}]
assert "budget" in refused(V.preflight, cfg, "inworld_female", segs)
print("  over budget: refused before any request, in the render and in the preflight: OK")

# ---- 4. paid use needs BOTH the config flag and the per-run switch
cfg = fresh_cfg(free_character_budget=30, allow_paid_fallback=True); os.environ["INWORLD_API_KEY"] = "k"
(tmp / "ledger.json").write_text(json.dumps({"inworld_characters_used": 10}))
refused(V._synth_inworld, cfg, iw, text, tmp / "d.mp3", "", "")                  # config yes, run switch no
os.environ["INWORLD_ALLOW_PAID_TTS"] = "true"
V._synth_inworld(cfg, iw, text, tmp / "d.mp3", "", "")                           # both yes
assert len(calls) == 1
cfg2 = fresh_cfg(free_character_budget=30, allow_paid_fallback=False); os.environ["INWORLD_API_KEY"] = "k"
os.environ["INWORLD_ALLOW_PAID_TTS"] = "true"
(tmp / "ledger.json").write_text(json.dumps({"inworld_characters_used": 10}))
refused(V._synth_inworld, cfg2, iw, text, tmp / "e.mp3", "", "")                 # run switch yes, config no
print("  paid Inworld only with config opt-in AND the per-run switch: OK")

# ---- 5. a segment over the size limit is refused
cfg = fresh_cfg(max_utf16_units=10); os.environ["INWORLD_API_KEY"] = "k"
assert "UTF-16" in refused(V._synth_inworld, cfg, iw, text, tmp / "f.mp3", "", "") and not calls
print("  oversize segment refused: OK")

# ---- 6. OpenRouter: every missing opt-in blocks it, in the render and in the preflight, with no request
orv = {"provider": "openrouter", "model_id": "mistralai/voxtral-mini-tts-2603", "voice": "en_paul_neutral"}
cfg = fresh_cfg()
assert "OPENROUTER_API_KEY" in refused(V._synth_openrouter, cfg, orv, text, tmp / "g.mp3", "", "")
os.environ["OPENROUTER_API_KEY"] = "k"
assert "disabled" in refused(V._synth_openrouter, cfg, orv, text, tmp / "g.mp3", "", "")           # allow_paid false
cfg["voice"]["openrouter"]["allow_paid"] = True
assert "per-run" in refused(V._synth_openrouter, cfg, orv, text, tmp / "g.mp3", "", "")            # no run switch
os.environ["OPENROUTER_ALLOW_PAID_TTS"] = "true"
assert "commercial" in refused(V._synth_openrouter, cfg, orv, text, tmp / "g.mp3", "", "").lower()  # not confirmed
assert not calls, calls
cfg["voice"]["openrouter"]["allow_paid"] = False
assert "disabled" in refused(V.preflight, cfg, "openrouter_voxtral", segs)
print("  OpenRouter: config flag, per-run switch and commercial-use confirmation are ALL required, nothing sent otherwise: OK")

# ---- 7. Edge fails -> the same-gender Inworld voice is used; OpenRouter is never an automatic backup
cfg = fresh_cfg(); os.environ["INWORLD_API_KEY"] = "k"
proj = SimpleNamespace(dir=tmp / "proj", cache_dir=tmp / "proj" / ".cache")
proj.cache_dir.mkdir(parents=True, exist_ok=True)


def edge_fails(*a, **k):
    raise RuntimeError("edge is down")


orig_edge = V.PROVIDERS["edge"]
V.PROVIDERS["edge"] = edge_fails
try:
    for key, want in (("edge_female", "Ashley"), ("edge_male", "Dennis")):
        calls.clear(); (tmp / "ledger.json").unlink(missing_ok=True)
        out = V.synthesize_all(proj, cfg, key, [{"text": "Backup voice check line one."}])
        assert len(out) == 1 and len(calls) == 1 and ledger()["inworld_voice_id"] == want, (key, ledger())
    rep = (proj.dir / "out" / "voice_report.txt").read_text()
    assert "BACKUP voice" in rep and "Lines read by a backup voice: 1 of 1" in rep, rep

    cfg_or = copy.deepcopy(cfg)
    cfg_or["voice"]["fallback"]["providers"] = ["openrouter"]
    cfg_or["voice"]["fallback"]["voice_keys"] = {"openrouter": {"female": "openrouter_voxtral", "male": "openrouter_voxtral"}}
    for k, v in (("allow_paid", True), ("commercial_use_confirmed", True)):
        cfg_or["voice"]["openrouter"][k] = v
    os.environ["OPENROUTER_API_KEY"] = "k"; os.environ["OPENROUTER_ALLOW_PAID_TTS"] = "true"
    calls.clear()
    msg = refused(V.synthesize_all, proj, cfg_or, "edge_female", [{"text": "A different line two."}])
    assert not calls, "OpenRouter must never be used as an automatic backup"
finally:
    V.PROVIDERS["edge"] = orig_edge
print("  Edge failure -> same-gender Inworld backup (+ voice_report.txt); OpenRouter never automatic: OK")

# ---- 8. a voice that is not Edge never silently falls back to another provider
cfg = fresh_cfg(); os.environ["INWORLD_API_KEY"] = "k"
orig_eleven = V.PROVIDERS["elevenlabs"]
V.PROVIDERS["elevenlabs"] = edge_fails
try:
    refused(V.synthesize_all, proj, cfg, "eleven_female", [{"text": "Third line, new text here."}])
    assert not calls
finally:
    V.PROVIDERS["elevenlabs"] = orig_eleven
print("  only Edge may fall back (a failing ElevenLabs voice stops the run instead of switching): OK")

print("OK - paid voices: Inworld budget/ledger/paid switch, OpenRouter opt-ins, backup-voice rules, voice report")
