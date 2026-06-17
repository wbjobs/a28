import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import pickle

from query_parser import QueryParser


class VectorSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = "./python/models_cache",
                 dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.segments = []
        self.segment_ids = []
        self.embeddings = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = self._load_encoder(model_name)

    def _load_encoder(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Loading sentence transformer model: {model_name}")
            model = SentenceTransformer(model_name)
            self.dimension = model.get_sentence_embedding_dimension()
            print(f"[INFO] Sentence transformer loaded. Embedding dimension: {self.dimension}")
            return model
        except Exception as e:
            print(f"[WARN] Failed to load sentence transformer: {e}. Using mock encoder.")
            return None

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder is not None:
            return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        rng = np.random.RandomState(42)
        return rng.randn(len(texts), self.dimension).astype(np.float32)

    def build_index(self, timeline: List[Dict], video_id: str) -> None:
        self.segments = timeline
        self.segment_ids = [f"{video_id}_{i}" for i in range(len(timeline))]

        texts = [QueryParser.build_text_representation(seg) for seg in timeline]
        self.embeddings = self._encode(texts).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.dimension)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / np.maximum(norms, 1e-10)
        self.index.add(normalized_embeddings)

        print(f"[INFO] FAISS index built with {len(timeline)} segments, dimension={self.dimension}")

    def search(self, query: str, top_k: int = 10,
               threshold: float = 0.3) -> List[Dict]:
        if self.index is None or len(self.segments) == 0:
            return []

        query_embedding = self._encode([query]).astype(np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        actual_k = min(top_k, len(self.segments))
        scores, indices = self.index.search(query_embedding, actual_k)

        parsed_query = QueryParser.parse(query)

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            score = float(scores[0][i])

            if idx < 0 or idx >= len(self.segments):
                continue

            segment = self.segments[idx]
            hybrid_score = self._compute_hybrid_score(segment, parsed_query, score)

            if hybrid_score >= threshold:
                results.append({
                    "segment_id": self.segment_ids[idx],
                    "segment_index": idx,
                    "timestamp": segment["timestamp"],
                    "duration": segment.get("duration", 1.0),
                    "scene": segment.get("scene", ""),
                    "objects": segment.get("objects", []),
                    "speech": segment.get("speech", ""),
                    "semantic_score": round(score, 4),
                    "hybrid_score": round(hybrid_score, 4),
                    "matched_filters": self._get_matched_filters(segment, parsed_query)
                })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

    def _compute_hybrid_score(self, segment: Dict, parsed_query: Dict,
                              semantic_score: float) -> float:
        keyword_bonus = 0.0

        objects_in_segment = set()
        for obj in segment.get('objects', []):
            obj_class = obj.get('class', '').lower()
            objects_in_segment.add(obj_class)

        for obj_filter in parsed_query.get('objects', []):
            obj_aliases = QueryParser.OBJECT_KEYWORDS.get(obj_filter, [obj_filter])
            if any(alias.lower() in objects_in_segment for alias in obj_aliases):
                keyword_bonus += 0.15

        scene = segment.get('scene', '').lower()
        for scene_filter in parsed_query.get('scenes', []):
            scene_aliases = QueryParser.SCENE_KEYWORDS.get(scene_filter, [scene_filter])
            if any(alias.lower() in scene for alias in scene_aliases):
                keyword_bonus += 0.1

        speech_text = segment.get('speech', '').lower()
        for keyword in parsed_query.get('keywords', []):
            if keyword.lower() in speech_text:
                keyword_bonus += 0.2

        for phrase in parsed_query.get('exact_phrases', []):
            if phrase.lower() in speech_text:
                keyword_bonus += 0.3

        return min(semantic_score + keyword_bonus, 1.0)

    def _get_matched_filters(self, segment: Dict, parsed_query: Dict) -> Dict:
        matched = {
            "objects": [],
            "scenes": [],
            "keywords": [],
            "exact_phrases": []
        }

        objects_in_segment = set()
        for obj in segment.get('objects', []):
            obj_class = obj.get('class', '').lower()
            objects_in_segment.add(obj_class)

        for obj_filter in parsed_query.get('objects', []):
            obj_aliases = QueryParser.OBJECT_KEYWORDS.get(obj_filter, [obj_filter])
            if any(alias.lower() in objects_in_segment for alias in obj_aliases):
                matched["objects"].append(obj_filter)

        scene = segment.get('scene', '').lower()
        for scene_filter in parsed_query.get('scenes', []):
            scene_aliases = QueryParser.SCENE_KEYWORDS.get(scene_filter, [scene_filter])
            if any(alias.lower() in scene for alias in scene_aliases):
                matched["scenes"].append(scene_filter)

        speech_text = segment.get('speech', '').lower()
        for keyword in parsed_query.get('keywords', []):
            if keyword.lower() in speech_text:
                matched["keywords"].append(keyword)

        for phrase in parsed_query.get('exact_phrases', []):
            if phrase.lower() in speech_text:
                matched["exact_phrases"].append(phrase)

        return matched

    def save(self, path: str) -> None:
        save_data = {
            "segments": self.segments,
            "segment_ids": self.segment_ids,
            "embeddings": self.embeddings,
            "dimension": self.dimension
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.segments = data["segments"]
        self.segment_ids = data["segment_ids"]
        self.embeddings = data["embeddings"]
        self.dimension = data["dimension"]

        self.index = faiss.IndexFlatIP(self.dimension)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / np.maximum(norms, 1e-10)
        self.index.add(normalized)
