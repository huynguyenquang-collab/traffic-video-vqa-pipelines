"""English RAG retrieval utilities from the original notebook."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from .translation_utils import batch_translate, save_translation_cache

INPUT_BASE = Path("/kaggle/input/datasets/huyqn12/cropped-zalo")
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLIP_BASE = "openai/clip-vit-base-patch32"
RAG_TOP_K = 5
RRF_K = 60


class EnglishRagRetriever:
    def __init__(
        self,
        input_base: Path = INPUT_BASE,
        clip_model_path: str = CLIP_BASE,
        device: str | None = None,
    ) -> None:
        self.input_base = input_base
        self.clip_model_path = clip_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
        self._load_reference_data()
        self._build_indexes()

    def _load_models(self) -> None:
        self.sbert_tok = AutoTokenizer.from_pretrained(MULTILINGUAL_MODEL)
        self.sbert_enc = AutoModel.from_pretrained(MULTILINGUAL_MODEL).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(self.clip_model_path)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_path).to(self.device).eval()

    def _load_reference_data(self) -> None:
        try:
            self.ref_ds = load_dataset("parquet", data_files="/kaggle/input/ref_traffic_sign_rules/**/*.parquet", split="train")
        except Exception:
            self.ref_ds = load_dataset("ghbihuy/vietnam_traffic_sign", split="train")

        ref_cache = self.input_base / "ref_corpus_en.json"
        if ref_cache.exists():
            with ref_cache.open("r", encoding="utf-8") as f:
                self.corpus_strings = json.load(f)
            return

        corpus_vi = []
        for row in self.ref_ds:
            category = (row.get("category") or "").strip()
            meaning = (row.get("meaning") or "").strip()
            description = (row.get("description") or "").strip()
            corpus_vi.append(f"{category} | {meaning} | {description}")
        self.corpus_strings = batch_translate(corpus_vi)
        save_translation_cache()
        with Path("/kaggle/working/ref_corpus_en.json").open("w", encoding="utf-8") as f:
            json.dump(self.corpus_strings, f, ensure_ascii=False)

    def _build_indexes(self) -> None:
        self.bm25 = BM25Okapi([s.lower().split() for s in self.corpus_strings])
        sbert_input = self.input_base / "sbert_ref_embeds_en.pt"
        sbert_working = Path("/kaggle/working/sbert_ref_embeds_en.pt")
        if sbert_input.exists():
            self.sbert_ref = torch.load(sbert_input, map_location="cpu")
        elif sbert_working.exists():
            self.sbert_ref = torch.load(sbert_working, map_location="cpu")
        else:
            self.sbert_ref = self._sbert_encode(self.corpus_strings)
            torch.save(self.sbert_ref, sbert_working)

        clip_input = self.input_base / "clip_img_embeds_en.pt"
        clip_working = Path("/kaggle/working/clip_img_embeds_en.pt")
        if clip_input.exists():
            self.clip_img_embeds = torch.load(clip_input, map_location="cpu")
        elif clip_working.exists():
            self.clip_img_embeds = torch.load(clip_working, map_location="cpu")
        else:
            self.clip_img_embeds = self._encode_ref_images()
            torch.save(self.clip_img_embeds, clip_working)

    @staticmethod
    def _mean_pool(model_out, attention_mask):
        tok_emb = model_out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(tok_emb.size()).float()
        return (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

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
            emb = F.normalize(self._mean_pool(out, enc["attention_mask"]), p=2, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    def _clip_get_image_feats(self, pixel_values):
        return self.clip_model.get_image_features(pixel_values=pixel_values)

    def _clip_get_text_feats(self, **kwargs):
        return self.clip_model.get_text_features(**kwargs)

    def _encode_ref_images(self):
        from tqdm import tqdm

        embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.ref_ds), 32), desc="CLIP ref images"):
                imgs = [img.convert("RGB") for img in self.ref_ds[i : i + 32]["image"]]
                inp = self.clip_proc(images=imgs, return_tensors="pt").to(self.device)
                feats = F.normalize(self._clip_get_image_feats(inp.pixel_values), p=2, dim=-1)
                embeds.append(feats.cpu())
        return torch.cat(embeds, dim=0)

    def bm25_rank(self, query: str, top_n: int = 15):
        scores = self.bm25.get_scores(query.lower().split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    def sbert_rank(self, query: str, top_n: int = 15):
        q = self._sbert_encode([query]).to(self.device)
        sims = (q @ self.sbert_ref.to(self.device).T).squeeze(0)
        return torch.topk(sims, min(top_n, sims.shape[0])).indices.tolist()

    def clip_img_rank(self, pil_crops: list, top_n: int = 15):
        with torch.no_grad():
            inp = self.clip_proc(images=pil_crops, return_tensors="pt").to(self.device)
            feats = F.normalize(self._clip_get_image_feats(inp.pixel_values), p=2, dim=-1).cpu()
        sims = torch.matmul(feats, self.clip_img_embeds.T)
        return torch.topk(sims.max(dim=0).values, min(top_n, sims.shape[1])).indices.tolist()

    def clip_text_rank(self, query: str, top_n: int = 15):
        with torch.no_grad():
            inp = self.clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = F.normalize(self._clip_get_text_feats(**inp), p=2, dim=-1).cpu()
        sims = (feats @ self.clip_img_embeds.T).squeeze(0)
        return torch.topk(sims, min(top_n, sims.shape[0])).indices.tolist()

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = RRF_K):
        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def retrieve(self, pil_crops: list, question: str, choices: list[str], top_k: int = RAG_TOP_K, detected_classes: list[str] | None = None):
        n = max(top_k * 4, 15)
        rankings = []
        if detected_classes:
            for sign_class in detected_classes:
                if sign_class:
                    rankings.extend([
                        self.bm25_rank(sign_class, top_n=n),
                        self.sbert_rank(sign_class, top_n=n),
                        self.clip_text_rank(sign_class, top_n=n),
                    ])
        else:
            rankings.extend([
                self.bm25_rank(question, top_n=n),
                self.sbert_rank(question, top_n=n),
                self.clip_text_rank(question, top_n=n),
            ])
        if pil_crops:
            rankings.append(self.clip_img_rank(pil_crops[:8], top_n=n))
        return [self.corpus_strings[i] for i in self.rrf(rankings)[:top_k]] if rankings else []

    def score_frame_with_clip(self, frame_pil, question_text: str) -> float:
        with torch.no_grad():
            img_inp = self.clip_proc(images=[frame_pil], return_tensors="pt").to(self.device)
            txt_inp = self.clip_proc(text=[question_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            img_feat = F.normalize(self._clip_get_image_feats(img_inp.pixel_values), p=2, dim=-1)
            txt_feat = F.normalize(self._clip_get_text_feats(**txt_inp), p=2, dim=-1)
        return (img_feat @ txt_feat.T).item()

