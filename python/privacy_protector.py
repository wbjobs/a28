import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    state: np.ndarray = None

    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, bbox: List[float], confidence: float):
        self.hit_streak += 1
        self.hits += 1
        self.time_since_update = 0
        self.bbox = bbox
        self.confidence = confidence


class ByteTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self._next_id = 1
        self._tracks: List[Track] = []
        self._lost_tracks: List[Track] = []
        self._max_lost = track_buffer

    def update(self, detections: List[Dict], privacy_classes: Optional[List[str]] = None) -> List[Track]:
        if privacy_classes is None:
            privacy_classes = ["person", "face", "plate"]

        high_dets = []
        low_dets = []

        for det in detections:
            cls = det.get("class", "")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [])

            is_privacy = any(pc in cls.lower() for pc in privacy_classes)

            if not is_privacy:
                continue

            if conf >= self.track_thresh:
                high_dets.append((cls, bbox, conf))
            elif conf >= self.track_thresh * 0.5:
                low_dets.append((cls, bbox, conf))

        for track in self._tracks:
            track.predict()

        matched, unmatched_tracks, unmatched_dets = self._match(
            self._tracks, high_dets
        )

        for t_idx, d_idx in matched:
            self._tracks[t_idx].update(high_dets[d_idx][1], high_dets[d_idx][2])

        second_matched, remaining_tracks, remaining_dets = self._match(
            [self._tracks[i] for i in unmatched_tracks],
            low_dets
        )

        for orig_idx, d_idx in second_matched:
            real_idx = unmatched_tracks[orig_idx]
            self._tracks[real_idx].update(low_dets[d_idx][1], low_dets[d_idx][2])

        for d_idx in remaining_dets:
            cls, bbox, conf = low_dets[d_idx]
            new_track = Track(
                track_id=self._next_id,
                class_name=cls,
                bbox=bbox,
                confidence=conf
            )
            self._next_id += 1
            self._tracks.append(new_track)

        for d_idx in unmatched_dets:
            cls, bbox, conf = high_dets[d_idx]
            new_track = Track(
                track_id=self._next_id,
                class_name=cls,
                bbox=bbox,
                confidence=conf
            )
            self._next_id += 1
            self._tracks.append(new_track)

        active_tracks = []
        lost_tracks = []

        for track in self._tracks:
            if track.time_since_update == 0:
                active_tracks.append(track)
            elif track.time_since_update <= self._max_lost:
                lost_tracks.append(track)

        self._tracks = active_tracks + lost_tracks

        return [t for t in active_tracks if t.hit_streak >= 1]

    def _match(self, tracks: List[Track], detections: List[Tuple]) -> Tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)))

        for t_idx, track in enumerate(tracks):
            for d_idx, (cls, bbox, conf) in enumerate(detections):
                iou = self._compute_iou(track.bbox, bbox)
                cost_matrix[t_idx, d_idx] = 1.0 - iou

        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        if cost_matrix.size == 0:
            return matched, unmatched_tracks, unmatched_dets

        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= (1.0 - self.match_thresh):
                matched.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_dets:
                    unmatched_dets.remove(c)

        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / max(union_area, 1e-10)


class PrivacyProtector:
    PRIVACY_CLASSES = {
        "person": "face",
        "face": "face",
        "license_plate": "plate",
        "plate": "plate"
    }

    FACE_EXPAND_RATIO = 0.3
    PLATE_EXPAND_RATIO = 0.15

    def __init__(self, detector, blur_strength: int = 25,
                 track_buffer: int = 30, pixelate_size: int = 10,
                 blur_mode: str = "pixelate"):
        self.detector = detector
        self.blur_strength = blur_strength
        self.pixelate_size = pixelate_size
        self.blur_mode = blur_mode
        self.tracker = ByteTracker(track_buffer=track_buffer)
        self._active_tracks: Dict[int, Track] = {}

    def process_frame(self, frame: np.ndarray,
                      detections: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[Dict]]:
        if detections is None:
            detections = self.detector.detect(frame)

        tracked = self.tracker.update(detections, list(self.PRIVACY_CLASSES.keys()))

        privacy_regions = []
        for track in tracked:
            self._active_tracks[track.track_id] = track

            region_type = self.PRIVACY_CLASSES.get(track.class_name, "unknown")
            expanded_bbox = self._expand_bbox(track.bbox, frame.shape, region_type)

            privacy_regions.append({
                "track_id": track.track_id,
                "type": region_type,
                "class": track.class_name,
                "bbox": expanded_bbox,
                "confidence": track.confidence,
                "age": track.age,
                "hits": track.hits
            })

        redacted_frame = self._apply_redaction(frame, privacy_regions)

        return redacted_frame, privacy_regions

    def _expand_bbox(self, bbox: List[float], frame_shape: Tuple,
                     region_type: str) -> List[float]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        if region_type == "face":
            expand = self.FACE_EXPAND_RATIO
            face_h = y2 - y1
            y1 = max(0, y1 - face_h * expand)
            y2 = min(h, y2 + face_h * expand * 0.5)
            face_w = x2 - x1
            x1 = max(0, x1 - face_w * expand * 0.3)
            x2 = min(w, x2 + face_w * expand * 0.3)
        elif region_type == "plate":
            expand = self.PLATE_EXPAND_RATIO
            plate_w = x2 - x1
            plate_h = y2 - y1
            x1 = max(0, x1 - plate_w * expand)
            x2 = min(w, x2 + plate_w * expand)
            y1 = max(0, y1 - plate_h * expand * 0.5)
            y2 = min(h, y2 + plate_h * expand * 0.5)

        return [int(x1), int(y1), int(x2), int(y2)]

    def _apply_redaction(self, frame: np.ndarray,
                         regions: List[Dict]) -> np.ndarray:
        result = frame.copy()

        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            if x1 >= x2 or y1 >= y2:
                continue

            roi = result[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if self.blur_mode == "pixelate":
                result[y1:y2, x1:x2] = self._pixelate(roi)
            elif self.blur_mode == "gaussian":
                result[y1:y2, x1:x2] = self._gaussian_blur(roi)
            elif self.blur_mode == "solid":
                result[y1:y2, x1:x2] = self._solid_fill(roi, region["type"])
            elif self.blur_mode == "mosaic":
                result[y1:y2, x1:x2] = self._mosaic(roi)

        return result

    def _pixelate(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < self.pixelate_size or w < self.pixelate_size:
            return roi

        small_h = max(1, h // self.pixelate_size)
        small_w = max(1, w // self.pixelate_size)

        small = np.resize(
            roi.reshape(small_h, self.pixelate_size, small_w, self.pixelate_size, 3).mean(axis=(1, 3)),
            (small_h, small_w, 3)
        ).astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(small)
        img = img.resize((w, h), Image.NEAREST)
        return np.array(img)

    def _gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        try:
            import cv2
            ksize = self.blur_strength * 2 + 1
            return cv2.GaussianBlur(roi, (ksize, ksize), 0)
        except ImportError:
            return self._pixelate(roi)

    def _solid_fill(self, roi: np.ndarray, region_type: str) -> np.ndarray:
        colors = {"face": (0, 0, 0), "plate": (0, 0, 0)}
        color = colors.get(region_type, (0, 0, 0))
        return np.full_like(roi, color)

    def _mosaic(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        block_size = max(4, min(h, w) // 8)
        result = roi.copy()

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = roi[y:min(y + block_size, h), x:min(x + block_size, w)]
                if block.size > 0:
                    mean_color = block.mean(axis=(0, 1)).astype(np.uint8)
                    result[y:min(y + block_size, h), x:min(x + block_size, w)] = mean_color

        return result

    def get_active_tracks(self) -> List[Dict]:
        return [
            {
                "track_id": t.track_id,
                "class": t.class_name,
                "bbox": t.bbox,
                "confidence": t.confidence,
                "age": t.age,
                "hits": t.hits,
                "hit_streak": t.hit_streak
            }
            for t in self._active_tracks.values()
        ]

    def reset(self):
        self.tracker = ByteTracker()
        self._active_tracks = {}
