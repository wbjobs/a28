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

from classifier import ImageClassifier
from detector import ObjectDetector
from speech_recognizer import SpeechRecognizer
from gstreamer_decoder import GStreamerDecoder, GStreamerAudioExtractor
from llm_corrector import BatchLLMCorrector
from srt_generator import SRTGenerator
from summary_generator import SummaryGenerator, SmartSummary
from clip_retriever import CLIPRetriever
from privacy_protector import PrivacyProtector
from rtmp_stream import RTMPStreamManager, LiveFrame

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    timestamp: float
    scene: str
    classifications: list
    objects: list
    frame_index: int = 0
    frame: Optional[np.ndarray] = None
    redacted_frame: Optional[np.ndarray] = None
    privacy_regions: list = field(default_factory=list)


@dataclass
class PipelineConfig:
    frame_sample_rate: float = 5.0
    vision_pool_size: int = 2
    classification_batch_size: int = 1
    detection_batch_size: int = 1
    enable_llm_correction: bool = True
    enable_srt_output: bool = True
    enable_summary: bool = True
    enable_privacy: bool = False
    enable_clip_index: bool = True
    audio_sample_rate: int = 16000
    llm_batch_size: int = 4
    privacy_blur_mode: str = "pixelate"
    privacy_blur_strength: int = 25
    rtmp_latency_ms: int = 1000
    rtmp_sample_interval: float = 2.0
    clip_sample_count: int = 8


class ParallelVideoProcessor:
    def __init__(self, classifier: ImageClassifier, detector: ObjectDetector,
                 speech_recognizer: SpeechRecognizer,
                 llm_corrector: Optional[BatchLLMCorrector] = None,
                 srt_generator: Optional[SRTGenerator] = None,
                 summary_generator: Optional[SummaryGenerator] = None,
                 clip_retriever: Optional[CLIPRetriever] = None,
                 privacy_protector: Optional[PrivacyProtector] = None,
                 rtmp_manager: Optional[RTMPStreamManager] = None,
                 config: Optional[PipelineConfig] = None):
        self.classifier = classifier
        self.detector = detector
        self.speech_recognizer = speech_recognizer
        self.llm_corrector = llm_corrector
        self.srt_generator = srt_generator or SRTGenerator()
        self.summary_generator = summary_generator
        self.clip_retriever = clip_retriever
        self.privacy_protector = privacy_protector
        self.rtmp_manager = rtmp_manager
        self.config = config or PipelineConfig()

        self._vision_executor = ThreadPoolExecutor(
            max_workers=self.config.vision_pool_size,
            thread_name_prefix="vision"
        )
        self._audio_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audio"
        )
        self._llm_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm"
        )

    def process_video(self, video_path: str,
                      progress_callback: Optional[Callable[[float, str], None]] = None,
                      privacy_mode: bool = False) -> Dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if progress_callback:
            progress_callback(0.0, "初始化 GStreamer 流式解码器...")

        audio_temp_path = str(video_path.parent / f"{video_path.stem}_audio_temp.wav")
        effective_privacy = privacy_mode or self.config.enable_privacy
        clip_frames = []

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

                vision_results, clip_frames = self._run_vision_pipeline(
                    decoder, progress_callback, collect_frames=self.config.enable_clip_index
                )

            if speech_future:
                speech_segments_raw = speech_future.result(timeout=600)

            if progress_callback:
                progress_callback(0.80, "语音识别完成，进行 LLM 纠错...")

            if self.llm_corrector and self.config.enable_llm_correction and speech_segments_raw:
                context = self._build_llm_context(vision_results)
                llm_future = self._llm_executor.submit(
                    self.llm_corrector.correct_segments,
                    speech_segments_raw, context
                )
                speech_segments_corrected = llm_future.result(timeout=300)
            else:
                speech_segments_corrected = speech_segments_raw

            if progress_callback:
                progress_callback(0.88, "融合多模态时间线...")

            timeline = self._merge_timeline(vision_results, speech_segments_corrected)

            srt_path = None
            if self.config.enable_srt_output and speech_segments_corrected:
                if progress_callback:
                    progress_callback(0.91, "生成 SRT 字幕文件...")

                video_id = video_path.stem
                srt_path = self.srt_generator.generate_from_segments(
                    speech_segments_corrected, video_id
                )
                timeline = SRTGenerator.merge_subtitles_into_timeline(
                    timeline, speech_segments_corrected
                )

            summary_data = None
            if self.summary_generator and self.config.enable_summary:
                if progress_callback:
                    progress_callback(0.94, "生成智能摘要...")

                summary = self.summary_generator.generate(timeline, speech_segments_corrected)
                summary_data = {
                    "summary": summary.summary,
                    "keywords": summary.keywords,
                    "scene_summary": summary.scene_summary,
                    "speech_summary": summary.speech_summary,
                    "objects_summary": summary.objects_summary
                }

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
                "summary": summary_data,
                "privacy_mode": effective_privacy,
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

    def process_live_frame(self, live_frame: LiveFrame, stream_id: str) -> Dict:
        scene, classifications = self.classifier.predict(live_frame.frame)
        detections = self.detector.detect(live_frame.frame)

        redacted_frame = None
        privacy_regions = []
        if self.privacy_protector and self.config.enable_privacy:
            redacted_frame, privacy_regions = self.privacy_protector.process_frame(
                live_frame.frame, detections
            )

        if self.clip_retriever and self.config.enable_clip_index:
            self.clip_retriever.encode_frame(live_frame.frame)

        return {
            "stream_id": stream_id,
            "timestamp": live_frame.timestamp,
            "frame_index": live_frame.frame_index,
            "scene": scene,
            "classifications": classifications,
            "objects": detections,
            "privacy_regions": privacy_regions,
            "has_redaction": redacted_frame is not None
        }

    def start_live_stream(self, rtmp_url: str, stream_id: Optional[str] = None) -> str:
        if not self.rtmp_manager:
            self.rtmp_manager = RTMPStreamManager(
                sample_interval_sec=self.config.rtmp_sample_interval,
                latency_ms=self.config.rtmp_latency_ms
            )

        def on_frame(live_frame: LiveFrame, sid: str):
            result = self.process_live_frame(live_frame, sid)

        sid = self.rtmp_manager.start_stream(rtmp_url, stream_id, on_frame)
        logger.info(f"[LIVE] Started stream {sid} from {rtmp_url}")
        return sid

    def stop_live_stream(self, stream_id: str):
        if self.rtmp_manager:
            self.rtmp_manager.stop_stream(stream_id)
            logger.info(f"[LIVE] Stopped stream {stream_id}")

    def index_video_for_clip(self, video_id: str, video_path: str) -> Dict:
        if not self.clip_retriever:
            return {"error": "CLIP retriever not available"}

        frames = []
        try:
            with GStreamerDecoder(video_path, sample_interval_sec=3.0) as decoder:
                while True:
                    packet = decoder.read_frame(timeout=15.0)
                    if packet is None:
                        break
                    frames.append(packet.frame)
        except Exception as e:
            logger.error(f"[CLIP] Failed to extract frames: {e}")
            return {"error": str(e)}

        if not frames:
            return {"error": "No frames extracted"}

        self.clip_retriever.add_video_from_frames(video_id, frames, {
            "video_path": video_path
        })

        clip_index_path = str(Path("./results") / "clip_index.pkl")
        self.clip_retriever.save(clip_index_path)

        return {
            "video_id": video_id,
            "frames_indexed": len(frames),
            "feature_dim": self.clip_retriever.feature_dim,
            "total_videos": len(self.clip_retriever.video_features)
        }

    def search_similar_videos(self, query_video_path: str,
                              top_k: int = 5,
                              exclude_id: Optional[str] = None,
                              threshold: float = 0.3) -> List[Dict]:
        if not self.clip_retriever:
            return []

        frames = []
        try:
            with GStreamerDecoder(query_video_path, sample_interval_sec=2.0) as decoder:
                while True:
                    packet = decoder.read_frame(timeout=15.0)
                    if packet is None:
                        break
                    frames.append(packet.frame)
        except Exception as e:
            logger.error(f"[CLIP] Query video extraction failed: {e}")
            return []

        if not frames:
            return []

        exclude_ids = [exclude_id] if exclude_id else None
        results = self.clip_retriever.search_by_video(
            frames, top_k=top_k, exclude_ids=exclude_ids, threshold=threshold
        )
        return results

    def search_similar_by_frame(self, frame: np.ndarray,
                                top_k: int = 5,
                                threshold: float = 0.3) -> List[Dict]:
        if not self.clip_retriever:
            return []
        return self.clip_retriever.search_by_frame(frame, top_k=top_k, threshold=threshold)

    def _run_speech_pipeline(self, video_path: str,
                             audio_temp_path: str) -> List[Dict]:
        try:
            logger.info("[PIPELINE-AUDIO] Starting speech recognition pipeline...")
            extractor = GStreamerAudioExtractor(video_path, audio_temp_path)
            extract_ok = extractor.extract_audio()

            if extract_ok and os.path.exists(audio_temp_path):
                segments = self.speech_recognizer.transcribe(audio_temp_path)
            else:
                segments = self.speech_recognizer.transcribe(video_path)

            logger.info(f"[PIPELINE-AUDIO] Complete: {len(segments)} segments")
            return segments
        except Exception as e:
            logger.error(f"[PIPELINE-AUDIO] Failed: {e}")
            return []

    def _run_vision_pipeline(self, decoder: GStreamerDecoder,
                             progress_callback: Optional[Callable] = None,
                             collect_frames: bool = False) -> Tuple[List[Dict], List[np.ndarray]]:
        logger.info("[PIPELINE-VISION] Starting parallel vision pipeline...")

        result_list: List[FrameResult] = []
        clip_frames: List[np.ndarray] = []
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

            redacted = None
            privacy_regions = []
            if self.privacy_protector and self.config.enable_privacy:
                redacted, privacy_regions = self.privacy_protector.process_frame(
                    packet.frame, detections
                )

            return FrameResult(
                timestamp=packet.timestamp,
                scene=scene,
                classifications=classifications,
                objects=detections,
                frame_index=packet.frame_index,
                frame=packet.frame if collect_frames else None,
                redacted_frame=redacted,
                privacy_regions=privacy_regions
            )

        def on_frame_done(future: Future):
            try:
                result = future.result()
                with result_lock:
                    result_list.append(result)
                    if collect_frames and result.frame is not None:
                        clip_frames.append(result.frame)
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

        for future in as_completed(futures, timeout=300):
            pass

        result_list.sort(key=lambda r: r.timestamp)

        vision_dicts = [
            {
                "timestamp": round(r.timestamp, 2),
                "scene": r.scene,
                "classifications": r.classifications,
                "objects": r.objects,
                "privacy_regions": r.privacy_regions if r.privacy_regions else []
            }
            for r in result_list
        ]

        return vision_dicts, clip_frames

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

            if nearest_frame and nearest_frame.get("privacy_regions"):
                entry["privacy_regions"] = nearest_frame["privacy_regions"]

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
        if self.rtmp_manager:
            self.rtmp_manager.shutdown_all()


class VideoProcessor(ParallelVideoProcessor):
    def __init__(self, classifier, detector, speech_recognizer,
                 frame_sample_rate=5, **kwargs):
        config = PipelineConfig(frame_sample_rate=frame_sample_rate)
        super().__init__(
            classifier=classifier, detector=detector,
            speech_recognizer=speech_recognizer,
            config=config, **kwargs
        )
