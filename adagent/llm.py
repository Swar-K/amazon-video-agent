"""LLM wrapper. Provider is chosen in config.yaml (llm.provider): "groq" (default) or "anthropic".

Keys come from environment variables:  GROQ_API_KEY  /  ANTHROPIC_API_KEY
List the Groq models your key can use:   python -m adagent.llm
"""
from __future__ import annotations

import base64
import os
import re
import time
from functools import lru_cache

import cv2
import numpy as np
import requests

from .common import load_config, log

class DailyLimitError(RuntimeError):
    """Groq's per-day quota for a model is used up. Waiting a minute will not help, so callers fail over or stop at once."""


class RequestTooLargeError(RuntimeError):
    """A single request cannot fit the model's current per-minute token budget."""


_exhausted: set = set()   # models whose daily quota ran out during this run (skipped on later calls)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

PREF_TEXT = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]
PREF_VISION = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "meta-llama/llama-4-scout-17b-16e-instruct",
               "meta-llama/llama-4-maverick-17b-128e-instruct"]
_NOT_TEXT = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "vision", "scout", "maverick", "-vl", "compound", "allam")
_substitutes: dict = {}   # (requested model, is_vision) -> model that actually works on this account


@lru_cache(maxsize=1)
def _llm_cfg() -> dict:
    return load_config().get("llm", {})


def jpeg_b64(img_bgr: np.ndarray, max_side: int = 1280) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


# ------------------------------- Groq model discovery -----------------------
def groq_models(key: str) -> list[str]:
    r = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Could not list your Groq models (HTTP {r.status_code}). Check GROQ_API_KEY.")
    return sorted(m["id"] for m in r.json().get("data", []))


def _is_vision_id(i: str) -> bool:
    low = i.lower()
    m = re.search(r"qwen3\.(\d+)", low)       # Qwen 3.5 and later are natively multimodal
    return any(t in low for t in ("scout", "maverick", "vision", "-vl")) or bool(m and int(m.group(1)) >= 5)


def _auto_candidates(kind: str, exclude: set) -> list[str]:
    """Models in YOUR Groq account that can do this job and were not tried yet (each has its own daily quota)."""
    try:
        ids = groq_models(os.environ.get("GROQ_API_KEY", ""))
    except Exception:
        return []
    ok = [i for i in ids if i not in exclude and i not in _exhausted and
          (_is_vision_id(i) if kind == "vision" else not any(t in i.lower() for t in _NOT_TEXT))]
    return ok[:6]


def _pick_substitute(model: str, vision: bool, ids: list[str]) -> str | None:
    if model in ids:
        return model
    for m in (PREF_VISION if vision else PREF_TEXT):
        if m in ids:
            return m
    if vision:
        cands = [i for i in ids if _is_vision_id(i)]
    else:
        cands = [i for i in ids if not any(t in i.lower() for t in _NOT_TEXT)]
    return cands[0] if cands else None


def _model_missing(status: int, text: str) -> bool:
    t = text.lower()
    return status in (400, 403, 404) and any(k in t for k in ("model_not_found", "does not exist", "decommissioned", "no access", "not found"))


# ------------------------------- Groq ---------------------------------------
_REASON_KEYS = ("reasoning_effort", "reasoning_format", "include_reasoning")


def _reasoning_variants(model: str) -> list[dict]:
    """Options that keep a reasoning model's thinking short/hidden, most specific first, ending with 'none'."""
    m = model.lower()
    if "gpt-oss" in m:
        return [{"reasoning_effort": "low", "include_reasoning": False}, {"reasoning_effort": "low"}, {}]
    if any(t in m for t in ("qwen3", "qwq", "deepseek-r1")):
        return [{"reasoning_effort": "none", "reasoning_format": "hidden"}, {"reasoning_format": "hidden"},
                {"reasoning_effort": "none"}, {}]
    return [{}]


def _apply_reasoning(body: dict, level: int = 0) -> None:
    for k in _REASON_KEYS:
        body.pop(k, None)
    variants = _reasoning_variants(body["model"])
    body.update(variants[min(level, len(variants) - 1)])


def _ask_groq(model, system, user, images, max_tokens, temperature) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set (add it as a GitHub secret).")
    max_tokens = min(max_tokens, int(_llm_cfg().get("groq_max_tokens", 6500)))  # free-tier limits count requested tokens
    vision = bool(images)
    if vision:  # instructions go in the user turn: most reliable with image input
        content = [{"type": "text", "text": system + "\n\n" + user}]
        for im in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(im, int(_llm_cfg().get("vision_max_side", 640)))}})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    requested = model
    body = {"model": _substitutes.get((requested, vision), requested), "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "response_format": {"type": "json_object"}}
    level = 0
    _apply_reasoning(body, level)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last, swapped = "", False
    for attempt in range(8):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(3 + attempt * 4)
            continue
        if r.status_code == 200:
            choice = r.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                log("  WARNING: the model hit its token limit and the reply was cut off.")
            return choice["message"].get("content") or ""
        last = f"{r.status_code}: {r.text[:300]}"
        if _model_missing(r.status_code, r.text) and not swapped:
            swapped = True
            ids = groq_models(key)
            log(f"  Groq model '{body['model']}' is not available to your account. Your models: {', '.join(ids)}")
            sub = _pick_substitute(requested, vision, ids)
            if not sub or sub == body["model"]:
                raise RuntimeError(f"No usable {'vision' if vision else 'text'} model on your Groq account. "
                                   f"Set llm.model / llm.vision_model in config.yaml to one of: {', '.join(ids)}")
            log(f"  Using '{sub}' instead (set it in config.yaml to make this permanent).")
            _substitutes[(requested, vision)] = sub
            body["model"] = sub
            level = 0
            _apply_reasoning(body, level)
            continue
        if r.status_code == 400 and any(k in body for k in _REASON_KEYS) and "reason" in r.text.lower():
            level += 1  # this model rejected those reasoning options: try the next, simpler set
            _apply_reasoning(body, level)
            continue
        if r.status_code == 400 and "response_format" in body and "json" in r.text.lower():
            body.pop("response_format")  # model without JSON mode: our parser copes with plain text
            continue
        if r.status_code in (413, 429) and "request too large" in r.text.lower():
            # TPM accounting includes both the prompt and the requested completion budget.
            m = re.search(r"limit\s+(\d+),\s*requested\s+(\d+)", r.text, re.I)
            if m and ("tokens per minute" in r.text.lower() or "tpm" in r.text.lower()
                      or "output tokens" in r.text.lower() or "max_tokens" in r.text.lower()):
                limit, req = int(m.group(1)), int(m.group(2))
                prompt_estimate = max(0, req - int(body["max_tokens"]))
                reserve = max(256, int(limit * 0.04))
                available = limit - prompt_estimate - reserve
                if 256 <= available < body["max_tokens"]:
                    log(f"  Request exceeds {body['model']}'s TPM budget ({limit}); "
                        f"prompt estimate {prompt_estimate}, reducing max_tokens "
                        f"{body['max_tokens']} -> {available} and retrying.")
                    body["max_tokens"] = available
                    continue
                raise RequestTooLargeError(
                    f"Request cannot fit {body['model']}'s per-minute token budget ({limit} TPM): "
                    f"prompt estimate {prompt_estimate} tokens leaves only {max(0, available)} output tokens. "
                    "The prompt must be shortened or another model/provider used. " + last)
            raise RequestTooLargeError(f"Groq says this single request is too large for {body['model']}'s "
                                       f"per-minute token limit. " + last)
        if r.status_code == 429 and re.search(r"tokens per day|\(TPD\)|requests per day|\(RPD\)|per day", r.text, re.I):
            m = re.search(r"Limit (\d+), Used (\d+), Requested (\d+)", r.text)
            detail = f" ({int(m.group(2)):,} of {int(m.group(1)):,} used today)" if m else ""
            raise DailyLimitError(f"Groq's daily limit for the model '{body['model']}' is used up{detail}. "
                                  "Waiting a minute will not help: the quota refills gradually over about 24 hours.")
        if r.status_code == 429:
            wait = min(65.0, float(r.headers.get("retry-after", 8 + attempt * 6)))
            log(f"  Groq rate limit, waiting {wait:.0f}s ...")
            time.sleep(wait)
            continue
        if r.status_code == 413:
            raise RuntimeError("Groq says this request is too large for your plan's per-minute token limit. "
                               "Lower groq_max_tokens in config.yaml, or retry in a minute. " + last)
        if r.status_code >= 500:
            time.sleep(3 + attempt * 4)
            continue
        raise RuntimeError(f"Groq error {last}")
    raise RuntimeError(f"Groq kept failing: {last}")


# ------------------------------- Anthropic (optional) -----------------------
def _ask_anthropic(model, system, user, images, max_tokens, temperature) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it as a GitHub secret).")
    import anthropic
    client = anthropic.Anthropic()
    content: list[dict] = []
    for im in images or []:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(im)}})
    content.append({"type": "text", "text": user})
    last = None
    for attempt in range(4):
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                          system=system, messages=[{"role": "user", "content": content}])
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        except Exception as e:
            last = e
            time.sleep(2 + attempt * 4)
    raise RuntimeError(f"Anthropic API failed: {last}")



# -------------------------- Multi-provider vision --------------------------
def _vision_images(images: list[np.ndarray]) -> list[str]:
    return [jpeg_b64(im, int(_llm_cfg().get("vision_max_side", 640))) for im in images]


def _response_text(data: dict) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    except (KeyError, IndexError, TypeError):
        pass
    raise RuntimeError("Vision provider returned an empty or unrecognized response.")


def _ask_openai_compatible_vision(provider: str, model: str, system: str, user: str,
                                  images: list[np.ndarray], max_tokens: int, temperature: float) -> str:
    env_name, endpoint = {
        "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions"),
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions"),
    }[provider]
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not set; skipping {provider} vision.")
    content = [{"type": "text", "text": system + "\n\n" + user}]
    for encoded in _vision_images(images):
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["X-Title"] = "Amazon Video Agent"
    last = ""
    for attempt in range(2):
        try:
            r = requests.post(endpoint, headers=headers, json=body, timeout=180)
            if r.status_code == 200:
                return _response_text(r.json())
            last = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(2 + attempt * 3)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"{provider} vision request failed ({last}).")


def _ask_gemini_vision(model: str, system: str, user: str, images: list[np.ndarray],
                       max_tokens: int, temperature: float) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; skipping Gemini vision.")
    parts = [{"text": system + "\n\n" + user}]
    for encoded in _vision_images(images):
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": encoded}})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last = ""
    for attempt in range(2):
        try:
            r = requests.post(endpoint, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=180)
            if r.status_code == 200:
                data = r.json()
                try:
                    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError("Gemini returned no text content.")
            last = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(2 + attempt * 3)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Gemini vision request failed ({last}).")


def _ask_groq_vision_chain(model: str, system: str, user: str, images: list[np.ndarray],
                           max_tokens: int, temperature: float) -> str:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set; skipping Groq vision.")
    chain = [model] + [x for x in (_llm_cfg().get("vision_fallbacks") or []) if x != model]
    ok, out, last = _try_models(chain, system, user, images, max_tokens, temperature, daily_seen=False)
    if ok:
        return out
    if _llm_cfg().get("auto_fallbacks", True) and any(x in _exhausted for x in chain):
        extra = _auto_candidates("vision", set(chain))
        if extra:
            log(f"  Also trying other vision models from your Groq account: {', '.join(extra)}")
            ok, out, last2 = _try_models(extra, system, user, images, max_tokens, temperature, daily_seen=True)
            if ok:
                return out
            last = last or last2
    raise DailyLimitError(f"Groq vision models are unavailable or exhausted ({', '.join(chain)}). {last or ''}")


def _ask_vision_with_provider_failover(model: str, system: str, user: str, images: list[np.ndarray],
                                       max_tokens: int, temperature: float) -> str:
    cfg = _llm_cfg()
    providers = cfg.get("vision_providers") or ["groq", "gemini", "mistral", "openrouter"]
    models = cfg.get("vision_provider_models") or {}
    errors = []
    for provider in providers:
        provider = str(provider).lower()
        provider_model = models.get(provider)
        if not provider_model:
            errors.append(f"{provider}: no model configured")
            continue
        try:
            log(f"  Vision provider: {provider} ({provider_model})")
            if provider == "groq":
                return _ask_groq_vision_chain(model, system, user, images, max_tokens, temperature)
            if provider == "gemini":
                return _ask_gemini_vision(provider_model, system, user, images, max_tokens, temperature)
            if provider in ("mistral", "openrouter"):
                return _ask_openai_compatible_vision(provider, provider_model, system, user,
                                                     images, max_tokens, temperature)
            errors.append(f"{provider}: unsupported provider")
        except Exception as e:
            errors.append(f"{provider}: {str(e)[:220]}")
            log(f"  {provider} vision failed; falling back to the next provider ({str(e)[:160]}).")
    raise RuntimeError("All configured vision providers failed. " + " | ".join(errors))


def ask(model: str, system: str, user: str, images: list[np.ndarray] | None = None,
        max_tokens: int = 8000, temperature: float = 0.8) -> str:
    if images:
        if _llm_cfg().get("vision_providers"):
            return _ask_vision_with_provider_failover(model, system, user, images, max_tokens, temperature)
        if _llm_cfg().get("provider", "groq") != "groq":
            return _ask_anthropic(model, system, user, images, max_tokens, temperature)
    elif _llm_cfg().get("provider", "groq") != "groq":
        return _ask_anthropic(model, system, user, images, max_tokens, temperature)
    # Groq: when a model's DAILY quota is gone, move on to the next model listed under llm.vision_fallbacks /
    # llm.text_fallbacks in config.yaml (every model has its own daily quota).
    kind = "vision" if images else "text"
    chain = [model] + [x for x in (_llm_cfg().get(f"{kind}_fallbacks") or []) if x != model]
    ok, out, last = _try_models(chain, system, user, images, max_tokens, temperature, daily_seen=False)
    if ok:
        return out
    if _llm_cfg().get("auto_fallbacks", True) and any(x in _exhausted for x in chain):
        extra = _auto_candidates(kind, set(chain))      # every other suitable model your account offers
        if extra:
            log(f"  Also trying other {kind} models from your Groq account: {', '.join(extra)}")
            ok, out, last2 = _try_models(extra, system, user, images, max_tokens, temperature, daily_seen=True)
            if ok:
                return out
            chain = chain + extra
            last = last or last2
    if last is not None and not any(x in _exhausted for x in chain):
        raise last
    raise DailyLimitError(f"Every {kind} model has hit its daily limit today ({', '.join(chain)}). "
                          "Waiting a minute will not help: the quotas refill gradually over about 24 hours.")


def _try_models(chain, system, user, images, max_tokens, temperature, daily_seen: bool):
    """Try each model in order. (True, reply, None) on success; (False, None, last_error) when all are used up/unusable."""
    last = None
    for i, m in enumerate(chain):
        if m in _exhausted:
            continue
        try:
            return True, _ask_groq(m, system, user, images, max_tokens, temperature), None
        except DailyLimitError as e:
            _exhausted.add(m)
            daily_seen = True
            last = e
            nxt = next((x for x in chain[i + 1:] if x not in _exhausted), None)
            log(f"  Daily limit reached for '{m}'." + (f" Trying '{nxt}' instead ..." if nxt else ""))
        except RequestTooLargeError as e:
            log(f"  '{m}' cannot fit this request in its TPM budget ({str(e)[:160]}); trying the next model ...")
            last = last or e
        except RuntimeError as e:
            if not daily_seen and not any(x in _exhausted for x in chain[:i]):
                raise                           # the first-choice model failed for another reason: report it as before
            log(f"  Fallback model '{m}' could not be used ({str(e)[:140]}); trying the next one ...")
            last = last or e
    return False, None, last


if __name__ == "__main__":
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        raise SystemExit("Set GROQ_API_KEY first.")
    print("\n".join(groq_models(k)))
