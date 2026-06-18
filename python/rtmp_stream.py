import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import numpy as np
import threading
import queue
import time
import json
import logging
import uuid
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass

Gst.init(None)
logger = logging.getLogger(__name__)


@dataclass
class LiveFrame:
    frame: np.ndarray
    timestamp: float
    frame_index: int
    pts_ns: int


class RTMPStreamManager:
    def __init__(self, frame_callback: Optional[Callable[[LiveFrame], None]] = None,
                 sample_interval_sec: float = 2.0,
                 latency_ms: int = 1000,
                 buffer_size: int = 30):
        self._streams: Dict[str, dict] = {}
        self._frame_callback = frame_callback
        self._sample_interval_sec = sample_interval_sec
        self._latency_ms = latency_ms
        self._buffer_size = buffer_size
        self._lock = threading.Lock()

    def start_stream(self, rtmp_url: str, stream_id: Optional[str] = None,
                     analysis_callback: Optional[Callable] = None) -> str:
        stream_id = stream_id or f"live_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if stream_id in self._streams:
                raise ValueError(f"Stream {stream_id} already exists")

            stream_info = {
                "rtmp_url": rtmp_url,
                "stream_id": stream_id,
                "status": "starting",
                "started_at": time.time(),
                "pipeline": None,
                "appsink": None,
                "frame_queue": queue.Queue(maxsize=self._buffer_size),
                "analysis_queue": queue.Queue(maxsize=100),
                "eos_event": threading.Event(),
                "error": None,
                "fps": 30.0,
                "width": 0,
                "height": 0,
                "frame_index": 0,
                "last_sample_time": -999.0,
                "glb_loop": None,
                "bus_thread": None,
                "analysis_thread": None,
                "analysis_callback": analysis_callback or self._frame_callback,
                "last_events": [],
                "stats": {
                    "frames_received": 0,
                    "frames_sampled": 0,
                    "frames_analyzed": 0,
                    "latency_ms": 0
                }
            }

            self._streams[stream_id] = stream_info

        try:
            self._build_and_start_pipeline(stream_id)
        except Exception as e:
            logger.error(f"[RTMP] Failed to start stream {stream_id}: {e}")
            with self._lock:
                stream_info["status"] = "failed"
                stream_info["error"] = str(e)
            raise

        return stream_id

    def _build_and_start_pipeline(self, stream_id: str):
        info = self._streams[stream_id]
        rtmp_url = info["rtmp_url"]

        pipeline_str = (
            f'rtmpsrc location="{rtmp_url}" '
            f'latency={self._latency_ms} ! '
            f'flvdemux name=demux ! '
            f'queue max-size-buffers=2 max-size-time=0 max-size-bytes=0 ! '
            f'decodebin ! '
            f'videoconvert ! '
            f'videoscale ! '
            f'video/x-raw,format=RGB,width=640 ! '
            f'appsink name=videosink emit-signals=true max-buffers=2 drop=true'
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            logger.warning(f"[RTMP] RTMP-specific pipeline failed: {e}, trying generic pipeline")
            pipeline_str = (
                f'uridecodebin uri="{rtmp_url}" ! '
                f'videoconvert ! '
                f'videoscale ! '
                f'video/x-raw,format=RGB,width=640 ! '
                f'appsink name=videosink emit-signals=true max-buffers=2 drop=true'
            )
            pipeline = Gst.parse_launch(pipeline_str)

        info["pipeline"] = pipeline
        appsink = pipeline.get_by_name("videosink")
        info["appsink"] = appsink

        appsink.connect("new-sample", self._on_new_sample, stream_id)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, stream_id)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start RTMP pipeline for {rtmp_url}")

        info["status"] = "running"

        info["glb_loop"] = GLib.MainLoop()
        info["bus_thread"] = threading.Thread(
            target=lambda: info["glb_loop"].run(),
            daemon=True
        )
        info["bus_thread"].start()

        if info["analysis_callback"]:
            info["analysis_thread"] = threading.Thread(
                target=self._analysis_worker,
                args=(stream_id,),
                daemon=True
            )
            info["analysis_thread"].start()

        logger.info(f"[RTMP] Stream {stream_id} started: {rtmp_url}")

    def _on_new_sample(self, appsink, stream_id: str) -> Gst.FlowReturn:
        info = self._streams.get(stream_id)
        if not info:
            return Gst.FlowReturn.OK

        sample = appsink.pull_sample()
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if not buffer or not caps:
            return Gst.FlowReturn.OK

        structure = caps.get_structure(0)
        success, width = structure.get_int("width")
        if not success:
            return Gst.FlowReturn.OK
        success, height = structure.get_int("height")
        if not success:
            return Gst.FlowReturn.OK

        info["width"] = width
        info["height"] = height
        if structure.has_field("framerate"):
            fps_num, fps_den = structure.get_fraction("framerate")
            if fps_den > 0:
                info["fps"] = fps_num / fps_den

        success, buffer_map = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            frame_data = np.frombuffer(buffer_map.data, dtype=np.uint8)
            expected_size = height * width * 3
            if frame_data.size >= expected_size:
                frame = frame_data[:expected_size].reshape((height, width, 3)).copy()
            else:
                return Gst.FlowReturn.OK
        finally:
            buffer.unmap(buffer_map)

        pts = buffer.pts
        pts_ns = pts if pts != Gst.CLOCK_TIME_NONE else 0
        timestamp = pts_ns / Gst.SECOND if pts_ns > 0 else info["frame_index"] / max(info["fps"], 1.0)

        info["stats"]["frames_received"] += 1
        info["frame_index"] += 1

        if timestamp - info["last_sample_time"] >= self._sample_interval_sec or info["frame_index"] <= 1:
            live_frame = LiveFrame(
                frame=frame,
                timestamp=round(timestamp, 3),
                frame_index=info["frame_index"],
                pts_ns=pts_ns
            )

            info["stats"]["frames_sampled"] += 1
            info["last_sample_time"] = timestamp

            try:
                info["frame_queue"].put_nowait(live_frame)
            except queue.Full:
                try:
                    info["frame_queue"].get_nowait()
                    info["frame_queue"].put_nowait(live_frame)
                except queue.Empty:
                    pass

            try:
                info["analysis_queue"].put_nowait(live_frame)
            except queue.Full:
                try:
                    info["analysis_queue"].get_nowait()
                    info["analysis_queue"].put_nowait(live_frame)
                except queue.Empty:
                    pass

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message, stream_id: str):
        info = self._streams.get(stream_id)
        if not info:
            return

        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info(f"[RTMP] Stream {stream_id} EOS")
            info["status"] = "ended"
            info["eos_event"].set()
            if info["glb_loop"]:
                try:
                    info["glb_loop"].quit()
                except Exception:
                    pass
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[RTMP] Stream {stream_id} error: {err.message}")
            info["status"] = "error"
            info["error"] = err.message
            info["eos_event"].set()
            if info["glb_loop"]:
                try:
                    info["glb_loop"].quit()
                except Exception:
                    pass
        elif t == Gst.MessageType.WARNING:
            warn, _ = message.parse_warning()
            logger.warning(f"[RTMP] Stream {stream_id} warning: {warn.message}")

    def _analysis_worker(self, stream_id: str):
        info = self._streams.get(stream_id)
        if not info or not info["analysis_callback"]:
            return

        callback = info["analysis_callback"]

        while not info["eos_event"].is_set():
            try:
                live_frame = info["analysis_queue"].get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                start_time = time.time()
                callback(live_frame, stream_id)
                elapsed_ms = (time.time() - start_time) * 1000
                info["stats"]["frames_analyzed"] += 1
                info["stats"]["latency_ms"] = elapsed_ms

                event = {
                    "timestamp": live_frame.timestamp,
                    "frame_index": live_frame.frame_index,
                    "processing_latency_ms": round(elapsed_ms, 1),
                    "stream_id": stream_id
                }
                info["last_events"].append(event)
                if len(info["last_events"]) > 50:
                    info["last_events"] = info["last_events"][-50:]

            except Exception as e:
                logger.warning(f"[RTMP] Analysis error for {stream_id}: {e}")

    def stop_stream(self, stream_id: str):
        with self._lock:
            info = self._streams.get(stream_id)
            if not info:
                return

            info["eos_event"].set()
            info["status"] = "stopped"

            if info["pipeline"]:
                info["pipeline"].send_event(Gst.Event.new_eos())
                time.sleep(0.2)
                info["pipeline"].set_state(Gst.State.NULL)
                info["pipeline"] = None

            if info["glb_loop"]:
                try:
                    info["glb_loop"].quit()
                except Exception:
                    pass

            logger.info(f"[RTMP] Stream {stream_id} stopped")

    def get_stream_info(self, stream_id: str) -> Optional[Dict]:
        info = self._streams.get(stream_id)
        if not info:
            return None

        return {
            "stream_id": stream_id,
            "rtmp_url": info["rtmp_url"],
            "status": info["status"],
            "started_at": info["started_at"],
            "uptime_sec": round(time.time() - info["started_at"], 1),
            "fps": info["fps"],
            "resolution": f"{info['width']}x{info['height']}" if info['width'] > 0 else "unknown",
            "error": info["error"],
            "stats": info["stats"],
            "last_events": info["last_events"][-5:]
        }

    def list_streams(self) -> List[Dict]:
        return [self.get_stream_info(sid) for sid in self._streams]

    def get_frame(self, stream_id: str, timeout: float = 3.0) -> Optional[LiveFrame]:
        info = self._streams.get(stream_id)
        if not info:
            return None
        try:
            return info["frame_queue"].get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown_all(self):
        for stream_id in list(self._streams.keys()):
            self.stop_stream(stream_id)
