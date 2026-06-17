import numpy as np
import threading
import queue
import time
import os
import logging
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from tqdm import tqdm

from classifier import ImageClassifier
from detector import ObjectDetector
from speech_recognizer import SpeechRecognizer
from gstreamer_decoder import GStreamerDecoder, GStreamerAudioExtractor
from llm_corrector import BatchLLMCorrector
from srt_generator import SRTGenerator

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    timestamp: float
    scene: str
    classifications: list
    objects: list
    frame_index: int = 0


@dataclass
class PipelineConfig:
    frame_sample_rate: float = 5.0
    vision_pool_size: int = 2
    classification_batch_size: int = 1
    detection_batch_size: int = 1
    enable_llm_correction: bool = True
    enable_srt_output: bool = True
    audio_sample_rate: int = 16000
    llm_batch_size: int = 4


class ParallelVideoProcessor:
    def __init__(self, classifier: ImageClassifier, detector: ObjectDetector,
                 speech_recognizer: SpeechRecognizer,
                 llm_corrector: Optional[BatchLLMCorrector] = None,
                 srt_generator: Optional[SRTGenerator] = None,
                 config: Optional[PipelineConfig] = None):
        self.classifier = classifier
        self.detector = detector
        self.speech_recognizer = speech_recognizer
        self.llm_corrector = llm_corrector
        self.srt_generator = srt_generator or SRTGenerator()
        self.config = config or PipelineConfig()

        self._vision_executor = ThreadPoolExecutor(
            max_workers=self.config.vision_pool_size,
            thread_name_prefix="vision"
        )
        self._audio_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="audio"
        )
        self._llm_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="llm"
        )

    def process_video(self, video_path: str,
                      progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if progress_callback:
            progress_callback(0.0, "初始化 GStreamer 流式解码器...")

        audio_temp_path = str(video_path.parent / f"{video_path.stem}_audio_temp.wav")

        speech_future = None
        vision_results = []
        speech_segments_raw = []
        speech_segments_corrected = []

        try:
            with GStreamerDecoder(str(video_path),
                                  sample_interval_sec=self.config.frame_sample_rate) as decoder:

                if progress_callback:
                    progress_callback(0.05, "启动并行流水线（视觉 + 语音三路并行）...")

                speech_future = self._audio_executor.submit(
                    self._run_speech_pipeline,
                    str(video_path),
                    audio_temp_path
                )

                vision_results = self._run_vision_pipeline(
                    decoder,
                    progress_callback
                )

            if speech_future:
                speech_segments_raw = speech_future.result(timeout=600)

            if progress_callback:
                progress_callback(0.80, "语音识别完成，进行 LLM 纠错...")

            if self.llm_corrector and self.config.enable_llm_correction and speech_segments_raw:
                context = self._build_llm_context(vision_results)
                llm_future = self._llm_executor.submit(
                    self.llm_corrector.correct_segments,
                    speech_segments_raw,
                    context
                )
                speech_segments_corrected = llm_future.result(timeout=300)
            else:
                speech_segments_corrected = speech_segments_raw

            if progress_callback:
                progress_callback(0.90, "融合多模态时间线...")

            timeline = self._merge_timeline(vision_results, speech_segments_corrected)

            srt_path = None
            if self.config.enable_srt_output and speech_segments_corrected:
                if progress_callback:
                    progress_callback(0.93, "生成 SRT 字幕文件...")

                video_id = video_path.stem
                srt_path = self.srt_generator.generate_from_segments(
                    speech_segments_corrected,
                    video_id
                )
                timeline = SRTGenerator.merge_subtitles_into_timeline(
                    timeline, speech_segments_corrected
                )

            if progress_callback:
                progress_callback(0.97, "构建最终结果...")

            result = {
                "video_path": str(video_path),
                "total_duration": timeline[-1]["timestamp"] + timeline[-1].get("duration", 0) if timeline else 0,
                "segment_count": len(timeline),
                "timeline": timeline,
                "speech_segments": speech_segments_corrected,
                "srt_path": srt_path,
                "pipeline_mode": "parallel_gstreamer",
                "llm_correction_applied": self.llm_corrector is not None and self.config.enable_llm_correction,
            }

            if progress_callback:
                progress_callback(1.0, "处理完成！")

            return result

        except Exception as e:
            logger.error(f"[ERROR] Video processing failed: {e}")
            raise
        finally:
            if os.path.exists(audio_temp_path):
                try:
                    os.remove(audio_temp_path)
                except OSError:
                    pass

    def _run_speech_pipeline(self, video_path: str,
                             audio_temp_path: str) -> List[Dict]:
        try:
            logger.info("[PIPELINE-AUDIO] Starting speech recognition pipeline...")

            extractor = GStreamerAudioExtractor(video_path, audio_temp_path)
            extract_ok = extractor.extract_audio()

            if extract_ok and os.path.exists(audio_temp_path):
                segments = self.speech_recognizer.transcribe(audio_temp_path)
            else:
                logger.warning("[PIPELINE-AUDIO] Audio extraction failed, trying direct video transcription")
                segments = self.speech_recognizer.transcribe(video_path)

            logger.info(f"[PIPELINE-AUDIO] Speech recognition complete: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"[PIPELINE-AUDIO] Speech pipeline failed: {e}")
            return []

    def _run_vision_pipeline(self, decoder: GStreamerDecoder,
                             progress_callback: Optional[Callable] = None) -> List[Dict]:
        logger.info("[PIPELINE-VISION] Starting parallel vision pipeline...")

        frame_queue: queue.Queue = queue.Queue(maxsize=200)
        result_list: List[FrameResult] = []
        result_lock = threading.Lock()
        futures: List[Future] = []

        total_estimate = int(decoder.duration / self.config.frame_sample_rate) if decoder.duration > 0 else 0
        processed_count = [0]
        start_progress = 0.05
        end_progress = 0.80
        progress_range = end_progress - start_progress

        def process_single_frame(packet):
            scene, classifications = self.classifier.predict(packet.frame)
            detections = self.detector.detect(packet.frame)

            return FrameResult(
                timestamp=packet.timestamp,
                scene=scene,
                classifications=classifications,
                objects=detections,
                frame_index=packet.frame_index
            )

        def on_frame_done(future: Future):
            try:
                result = future.result()
                with result_lock:
                    result_list.append(result)
                    processed_count[0] += 1

                    if progress_callback and processed_count[0] % 3 == 0:
                        progress = start_progress + (processed_count[0] / max(1, total_estimate)) * progress_range
                        progress_callback(
                            min(progress, end_progress - 0.01),
                            f"视觉分析: {processed_count[0]}/{total_estimate} 帧"
                        )
            except Exception as e:
                logger.warning(f"[VISION] Frame processing error: {e}")

        while True:
            packet = decoder.read_frame(timeout=15.0)
            if packet is None:
                break

            future = self._vision_executor.submit(process_single_frame, packet)
            future.add_done_callback(on_frame_done)
            futures.append(future)

        logger.info(f"[PIPELINE-VISION] All {len(futures)} frames submitted to thread pool, waiting...")

        if progress_callback:
            progress_callback(end_progress - 0.05, f"等待 {len(futures)} 帧推理完成...")

        for future in as_completed(futures, timeout=300):
            pass

        result_list.sort(key=lambda r: r.timestamp)

        return [
            {
                "timestamp": round(r.timestamp, 2),
                "scene": r.scene,
                "classifications": r.classifications,
                "objects": r.objects
            }
            for r in result_list
        ]

    def _build_llm_context(self, vision_results: List[Dict]) -> str:
        scene_counts = {}
        all_objects = []

        for vr in vision_results[:20]:
            scene = vr.get("scene", "")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            for obj in vr.get("objects", []):
                obj_class = obj.get("class", "")
                if obj_class not in all_objects:
                    all_objects.append(obj_class)

        top_scene = max(scene_counts, key=scene_counts.get) if scene_counts else "未知"
        objects_str = "、".join(all_objects[:10]) if all_objects else "无"

        return f"视频主要场景：{top_scene}；检测到的物体：{objects_str}"

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
            next_t = sorted_times[i + 1] if i + 1 < len(sorted_times) else t + self.config.frame_sample_rate

            nearest_frame = self._find_nearest_frame(frame_timeline, t)
            overlapping_speech = self._find_speech_at_time(speech_segments, t)

            entry = {
                "timestamp": round(t, 2),
                "duration": round(next_t - t, 2),
                "scene": nearest_frame["scene"] if nearest_frame else "unknown",
                "classifications": nearest_frame.get("classifications", []) if nearest_frame else [],
                "objects": nearest_frame.get("objects", []) if nearest_frame else [],
                "speech": overlapping_speech if overlapping_speech else ""
            }

            if nearest_frame and "speech_original" in (nearest_frame or {}):
                entry["speech_original"] = nearest_frame["speech_original"]

            merged.append(entry)

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
                texts.append(seg.get("text", ""))
        return " ".join(t for t in texts if t)

    def shutdown(self):
        self._vision_executor.shutdown(wait=False)
        self._audio_executor.shutdown(wait=False)
        self._llm_executor.shutdown(wait=False)


class VideoProcessor(ParallelVideoProcessor):
    def __init__(self, classifier, detector, speech_recognizer,
                 frame_sample_rate=5, **kwargs):
        config = PipelineConfig(frame_sample_rate=frame_sample_rate)

        llm_corrector = kwargs.get('llm_corrector')
        srt_generator = kwargs.get('srt_generator')

        super().__init__(
            classifier=classifier,
            detector=detector,
            speech_recognizer=speech_recognizer,
            llm_corrector=llm_corrector,
            srt_generator=srt_generator,
            config=config
        )
