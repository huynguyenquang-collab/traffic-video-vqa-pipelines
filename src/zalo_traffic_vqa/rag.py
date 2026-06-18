from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .translation import TranslationCache


class TrafficSignRAG:
    """BM25 + SBERT + CLIP visual/text retrieval with reciprocal-rank fusion."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rag_cfg = cfg["rag"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.corpus: list[str] = []

    def build(self) -> "TrafficSignRAG":
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

        self.sbert_tok = AutoTokenizer.from_pretrained(self.cfg["models"]["sentence_encoder"])
        self.sbert_enc = AutoModel.from_pretrained(self.cfg["models"]["sentence_encoder"]).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(self.cfg["models"]["clip"])
        self.clip_model = CLIPModel.from_pretrained(self.cfg["models"]["clip"]).to(self.device).eval()

        corpus_cache = Path(self.rag_cfg["ref_corpus_cache"])
        try:
            ref_ds = load_dataset("parquet", data_files=self.rag_cfg["ref_parquet_glob"], split="train")
        except Exception:
            ref_ds = load_dataset(self.rag_cfg["ref_dataset"], split="train")

        if corpus_cache.exists():
            with corpus_cache.open("r", encoding="utf-8") as f:
                self.corpus = json.load(f)
        else:
            translator = TranslationCache(self.cfg["paths"]["translation_cache"])
            vi_rows = [
                f"{(row.get('category') or '').strip()} | {(row.get('meaning') or '').strip()} | {(row.get('description') or '').strip()}"
                for row in ref_ds
            ]
            self.corpus = translator.batch(vi_rows)
            corpus_cache.parent.mkdir(parents=True, exist_ok=True)
            with corpus_cache.open("w", encoding="utf-8") as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)

        self.bm25 = BM25Okapi([x.lower().split() for x in self.corpus])
        self.sbert_ref = self._load_or_encode_sbert()
        self.clip_img_embeds = self._load_or_encode_clip_images(ref_ds)
        return self

    def retrieve(
        self,
        pil_crops: list[Image.Image],
        question: str,
        choices: list[str],
        *,
        detected_classes: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        top_k = top_k or int(self.rag_cfg["top_k"])
        n = max(top_k * 4, 15)
        rankings: list[list[int]] = []
        queries = detected_classes or [question]
        for query in queries:
            if query:
                rankings.append(self._bm25_rank(query, n))
                rankings.append(self._sbert_rank(query, n))
                rankings.append(self._clip_text_rank(query, n))
        if pil_crops:
            rankings.append(self._clip_img_rank(pil_crops[:8], n))
        fused = self._rrf(rankings)[:top_k]
        return [self.corpus[i] for i in fused]

    def score_frame(self, frame_pil: Image.Image, question: str) -> float:
        with torch.no_grad():
            img = self.clip_proc(images=[frame_pil], return_tensors="pt").to(self.device)
            txt = self.clip_proc(text=[question], return_tensors="pt", padding=True, truncation=True).to(self.device)
            img_feat = F.normalize(self._clip_image_features(img.pixel_values), p=2, dim=-1)
            txt_feat = F.normalize(self._clip_text_features(**txt), p=2, dim=-1)
        return float((img_feat @ txt_feat.T).item())

    def _load_or_encode_sbert(self):
        cache = Path(self.rag_cfg["sbert_cache"])
        if cache.exists():
            return torch.load(cache, map_location="cpu")
        encoded = self._sbert_encode(self.corpus)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, cache)
        return encoded

    def _load_or_encode_clip_images(self, ref_ds):
        cache = Path(self.rag_cfg["clip_image_cache"])
        if cache.exists():
            return torch.load(cache, map_location="cpu")
        embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(ref_ds), 32), desc="CLIP ref images"):
                images = [img.convert("RGB") for img in ref_ds[i : i + 32]["image"]]
                inp = self.clip_proc(images=images, return_tensors="pt").to(self.device)
                embeds.append(F.normalize(self._clip_image_features(inp.pixel_values), p=2, dim=-1).cpu())
        out = torch.cat(embeds, dim=0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
        return out

    def _sbert_encode(self, texts: list[str], batch_size: int = 64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            enc = self.sbert_tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.sbert_enc(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(F.normalize(emb, p=2, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)

    def _bm25_rank(self, query: str, top_n: int) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    def _sbert_rank(self, query: str, top_n: int) -> list[int]:
        q = self._sbert_encode([query]).to(self.device)
        sims = (q @ self.sbert_ref.to(self.device).T).squeeze(0)
        return torch.topk(sims, min(top_n, sims.shape[0])).indices.tolist()

    def _clip_img_rank(self, pil_crops: list[Image.Image], top_n: int) -> list[int]:
        with torch.no_grad():
            inp = self.clip_proc(images=pil_crops, return_tensors="pt").to(self.device)
            feats = F.normalize(self._clip_image_features(inp.pixel_values), p=2, dim=-1).cpu()
        sims = feats @ self.clip_img_embeds.T
        return torch.topk(sims.max(dim=0).values, min(top_n, sims.shape[1])).indices.tolist()

    def _clip_text_rank(self, query: str, top_n: int) -> list[int]:
        with torch.no_grad():
            inp = self.clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = F.normalize(self._clip_text_features(**inp), p=2, dim=-1).cpu()
        sims = (feats @ self.clip_img_embeds.T).squeeze(0)
        return torch.topk(sims, min(top_n, sims.shape[0])).indices.tolist()

    def _rrf(self, rankings: list[list[int]]) -> list[int]:
        scores: dict[int, float] = {}
        k = int(self.rag_cfg["rrf_k"])
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def _clip_image_features(self, pixel_values):
        return self.clip_model.get_image_features(pixel_values=pixel_values)

    def _clip_text_features(self, **kwargs):
        return self.clip_model.get_text_features(**kwargs)


class ClipFrameScorer:
    """Lightweight CLIP scorer for no-RAG pipelines."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = CLIPProcessor.from_pretrained(cfg["models"]["clip"])
        self.model = CLIPModel.from_pretrained(cfg["models"]["clip"]).to(self.device).eval()

    def score_frame(self, frame_pil: Image.Image, question: str) -> float:
        with torch.no_grad():
            img = self.proc(images=[frame_pil], return_tensors="pt").to(self.device)
            txt = self.proc(text=[question], return_tensors="pt", padding=True, truncation=True).to(self.device)
            img_feat = F.normalize(self.model.get_image_features(pixel_values=img.pixel_values), p=2, dim=-1)
            txt_feat = F.normalize(self.model.get_text_features(**txt), p=2, dim=-1)
        return float((img_feat @ txt_feat.T).item())
