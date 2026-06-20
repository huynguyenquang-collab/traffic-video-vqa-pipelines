"""Vietnamese-to-English translation helpers with disk cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator

INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")
TRANS_CACHE_INPUT = INPUT_BASE / "translation_cache.json"
TRANS_CACHE_PATH = Path("/kaggle/working/translation_cache.json")

_trans_cache: dict[str, str] = {}
_translator: GoogleTranslator | None = None


def load_translation_cache(
    cache_input: Path = TRANS_CACHE_INPUT,
    cache_path: Path = TRANS_CACHE_PATH,
) -> dict[str, str]:
    global _trans_cache
    if cache_input.exists():
        with cache_input.open("r", encoding="utf-8") as f:
            _trans_cache = json.load(f)
        print(f"Loaded translation cache from input ({len(_trans_cache)} entries)")
    elif cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            _trans_cache = json.load(f)
        print(f"Loaded translation cache from working ({len(_trans_cache)} entries)")
    else:
        _trans_cache = {}
        print("Starting fresh translation cache.")
    return _trans_cache


def get_translator() -> GoogleTranslator:
    global _translator
    if _translator is None:
        _translator = GoogleTranslator(source="vi", target="en")
    return _translator


def translate_vi_to_en(text: str) -> str:
    if not text or not text.strip():
        return text
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _trans_cache:
        return _trans_cache[key]
    try:
        result = get_translator().translate(text)
    except Exception:
        time.sleep(1)
        try:
            result = get_translator().translate(text)
        except Exception:
            return text
    _trans_cache[key] = result
    return result


def batch_translate(texts: list[str], cache_path: Path = TRANS_CACHE_PATH) -> list[str]:
    uncached = [
        text for text in texts if hashlib.md5(text.encode()).hexdigest() not in _trans_cache
    ]
    print(f"  {len(texts) - len(uncached)}/{len(texts)} already cached. Translating {len(uncached)} new...")
    results = []
    for i, text in enumerate(texts):
        results.append(translate_vi_to_en(text))
        if (i + 1) % 100 == 0:
            print(f"  Translated {i + 1}/{len(texts)}...")
    save_translation_cache(cache_path)
    return results


def save_translation_cache(cache_path: Path = TRANS_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(_trans_cache, f, ensure_ascii=False)
    print(f"Saved translation cache ({len(_trans_cache)} entries)")

