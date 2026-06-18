import numpy as np
import faiss
import json
import logging
import os
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoFeature:
    video_id: str
    feature: np.ndarray
    metadata: Dict


class CLIPRetriever:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: str = "./python/models_cache",
                 feature_dim: int = 512):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        self.model = None
        self.preprocess = None
        self.index = None
        self.video_features: List[VideoFeature] = []
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            logger.info(f"[CLIP] Loading model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            if os.getenv('USE_GPU', 'false').lower() == 'true' and torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()
            self.feature_dim = self.model.config.projection_dim
            self._model_loaded = True
            logger.info(f"[CLIP] Model loaded, feature_dim={self.feature_dim}")
        except Exception as e:
            logger.warning(f"[CLIP] Failed to load CLIP model: {e}, using mock encoder")
            self._model_loaded = False

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._model_loaded:
            return self._encode_with_model(frame)
        return self._mock_encode(frame)

    def encode_frames_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.array([])

        if self._model_loaded:
            return self._encode_batch_with_model(frames)
        return np.stack([self._mock_encode(f) for f in frames])

    def _encode_with_model(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            image = Image.fromarray(frame)
        else:
            image = Image.fromarray(frame.astype(np.uint8))

        inputs = self.processor(images=image, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        feature = features.cpu().numpy().flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature.astype(np.float32)

    def _encode_batch_with_model(self, frames: List[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        images = []
        for frame in frames:
            if frame.shape[2] == 3 and frame.dtype == np.uint8:
                images.append(Image.fromarray(frame))
            else:
                images.append(Image.fromarray(frame.astype(np.uint8)))

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = features.cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-10)
        return features.astype(np.float32)

    def _mock_encode(self, frame: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(hash(frame.tobytes()) % (2**31))
        feature = rng.randn(self.feature_dim).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

    def encode_video(self, frames: List[np.ndarray],
                     sample_count: int = 8) -> np.ndarray:
        if not frames:
            return np.zeros(self.feature_dim, dtype=np.float32)

        if len(frames) > sample_count:
            indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        features = self.encode_frames_batch(sampled)
        if features.ndim == 1:
            return features

        avg_feature = np.mean(features, axis=0)
        norm = np.linalg.norm(avg_feature)
        if norm > 0:
            avg_feature = avg_feature / norm
        return avg_feature.astype(np.float32)

    def add_video(self, video_id: str, feature: np.ndarray,
                  metadata: Optional[Dict] = None):
        vf = VideoFeature(
            video_id=video_id,
            feature=feature,
            metadata=metadata or {}
        )
        self.video_features.append(vf)
        self._rebuild_index()

    def add_video_from_frames(self, video_id: str, frames: List[np.ndarray],
                              metadata: Optional[Dict] = None):
        feature = self.encode_video(frames)
        self.add_video(video_id, feature, metadata)

    def _rebuild_index(self):
        if not self.video_features:
            self.index = None
            return

        features = np.stack([vf.feature for vf in self.video_features])
        self.index = faiss.IndexFlatIP(self.feature_dim)
        self.index.add(features)

        logger.info(f"[CLIP] Index rebuilt with {len(self.video_features)} videos")

    def search_similar(self, query_feature: np.ndarray,
                       top_k: int = 5,
                       exclude_ids: Optional[List[str]] = None,
                       threshold: float = 0.3) -> List[Dict]:
        if self.index is None or len(self.video_features) == 0:
            return []

        query = query_feature.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        actual_k = min(top_k + len(exclude_ids or []), len(self.video_features))
        scores, indices = self.index.search(query, actual_k)

        results = []
        for i in range(actual_k):
            idx = int(indices[0][i])
            score = float(scores[0][i])

            if idx < 0 or idx >= len(self.video_features):
                continue

            vf = self.video_features[idx]

            if exclude_ids and vf.video_id in exclude_ids:
                continue

            if score < threshold:
                continue

            results.append({
                "video_id": vf.video_id,
                "similarity_score": round(score, 4),
                "metadata": vf.metadata
            })

        return results[:top_k]

    def search_by_frame(self, frame: np.ndarray,
                        top_k: int = 5,
                        exclude_ids: Optional[List[str]] = None,
                        threshold: float = 0.3) -> List[Dict]:
        feature = self.encode_frame(frame)
        return self.search_similar(feature, top_k, exclude_ids, threshold)

    def search_by_video(self, frames: List[np.ndarray],
                        top_k: int = 5,
                        exclude_ids: Optional[List[str]] = None,
                        threshold: float = 0.3) -> List[Dict]:
        feature = self.encode_video(frames)
        return self.search_similar(feature, top_k, exclude_ids, threshold)

    def save(self, path: str):
        data = {
            "video_features": [
                {"video_id": vf.video_id, "feature": vf.feature, "metadata": vf.metadata}
                for vf in self.video_features
            ],
            "feature_dim": self.feature_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[CLIP] Index saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.feature_dim = data["feature_dim"]
        self.video_features = []
        for item in data["video_features"]:
            self.video_features.append(VideoFeature(
                video_id=item["video_id"],
                feature=item["feature"],
                metadata=item["metadata"]
            ))
        self._rebuild_index()
        logger.info(f"[CLIP] Index loaded from {path}: {len(self.video_features)} videos")
