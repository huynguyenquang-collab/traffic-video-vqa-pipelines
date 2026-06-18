from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class TranslationCache:
    """Vietnamese-to-English translator with a durable JSON cache."""

    def __init__(self, cache_path: str | Path, *, source: str = "vi", target: str = "en") -> None:
        self.cache_path = Path(cache_path)
        self.source = source
        self.target = target
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)
        self._translator = None

    @property
    def translator(self):
        if self._translator is None:
            from deep_translator import GoogleTranslator

            self._translator = GoogleTranslator(source=self.source, target=self.target)
        return self._translator

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        try:
            result = self.translator.translate(text)
        except Exception:
            time.sleep(1)
            try:
                result = self.translator.translate(text)
            except Exception:
                result = text
        self.cache[key] = result
        return result

    def batch(self, texts: list[str], *, save_every: int = 100) -> list[str]:
        results = []
        for i, text in enumerate(texts, 1):
            results.append(self.translate(text))
            if i % save_every == 0:
                self.save()
        self.save()
        return results

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
