import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Optional
from tqdm import tqdm
import time

from classifier import ImageClassifier
from detector import ObjectDetector
from speech_recognizer import SpeechRecognizer


class VideoProcessor:
    def __init__(self, classifier: ImageClassifier, detector: ObjectDetector,
                 speech_recognizer: SpeechRecognizer,
                 frame_sample_rate: int = 5):
        self.classifier = classifier
        self.detector = detector
        self.speech_recognizer = speech_recognizer
        self.frame_sample_rate = frame_sample_rate

    def process_video(self, video_path: str,
                      progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict:
        if progress_callback:
            progress_callback(0.0, "开始处理视频...")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if progress_callback:
            progress_callback(0.05, "提取语音内容...")
        speech_segments = self.speech_recognizer.transcribe(str(video_path))

        if progress_callback:
            progress_callback(0.2, "分析视频帧...")
        frame_timeline = self._process_frames(str(video_path), progress_callback)

        if progress_callback:
            progress_callback(0.85, "融合多模态信息...")
        timeline = self._merge_timeline(frame_timeline, speech_segments)

        if progress_callback:
            progress_callback(0.95, "构建时间线...")
        result = {
            "video_path": str(video_path),
            "total_duration": timeline[-1]["timestamp"] if timeline else 0,
            "segment_count": len(timeline),
            "timeline": timeline
        }

        if progress_callback:
            progress_callback(1.0, "处理完成！")

        return result

    def _process_frames(self, video_path: str,
                        progress_callback: Optional[Callable[[float, str], None]] = None) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0

        sample_interval = max(1, int(fps * self.frame_sample_rate))

        frame_results = []
        frame_idx = 0
        processed_count = 0

        total_samples = total_frames // sample_interval
        start_progress = 0.2
        end_progress = 0.85
        progress_range = end_progress - start_progress

        pbar = tqdm(total=total_samples, desc="Processing frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps

                scene, classifications = self.classifier.predict(frame)
                detections = self.detector.detect(frame)

                frame_results.append({
                    "timestamp": round(timestamp, 2),
                    "scene": scene,
                    "classifications": classifications,
                    "objects": detections
                })

                processed_count += 1
                pbar.update(1)

                if progress_callback and processed_count % 5 == 0:
                    progress = start_progress + (processed_count / max(1, total_samples)) * progress_range
                    progress_callback(min(progress, end_progress),
                                      f"已分析 {processed_count}/{total_samples} 帧")

            frame_idx += 1

        pbar.close()
        cap.release()

        return frame_results

    def _merge_timeline(self, frame_timeline: List[Dict],
                        speech_segments: List[Dict]) -> List[Dict]:
        merged = []

        all_times = set()
        for frame in frame_timeline:
            all_times.add(frame["timestamp"])
        for seg in speech_segments:
            all_times.add(seg["start"])
            all_times.add(seg["end"])

        sorted_times = sorted(all_times)

        for i, t in enumerate(sorted_times):
            next_t = sorted_times[i + 1] if i + 1 < len(sorted_times) else t + 1.0

            nearest_frame = self._find_nearest_frame(frame_timeline, t)
            overlapping_speech = self._find_speech_at_time(speech_segments, t)

            merged.append({
                "timestamp": round(t, 2),
                "duration": round(next_t - t, 2),
                "scene": nearest_frame["scene"] if nearest_frame else "unknown",
                "classifications": nearest_frame["classifications"] if nearest_frame else [],
                "objects": nearest_frame["objects"] if nearest_frame else [],
                "speech": overlapping_speech if overlapping_speech else ""
            })

        return merged

    def _find_nearest_frame(self, frame_timeline: List[Dict], timestamp: float) -> Optional[Dict]:
        if not frame_timeline:
            return None
        nearest = None
        min_diff = float('inf')
        for frame in frame_timeline:
            diff = abs(frame["timestamp"] - timestamp)
            if diff < min_diff:
                min_diff = diff
                nearest = frame
        return nearest

    def _find_speech_at_time(self, speech_segments: List[Dict], timestamp: float) -> str:
        texts = []
        for seg in speech_segments:
            if seg["start"] <= timestamp <= seg["end"]:
                texts.append(seg["text"])
        return " ".join(texts)
