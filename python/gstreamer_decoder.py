import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Generator, Callable
from dataclasses import dataclass


Gst.init(None)


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    frame_index: int
    duration: float


class GStreamerDecoder:
    def __init__(self, video_path: str, sample_interval_sec: float = 5.0):
        self.video_path = video_path
        self.sample_interval_sec = sample_interval_sec
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[GstApp.AppSink] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self._eos_event = threading.Event()
        self._error: Optional[str] = None
        self._fps: float = 30.0
        self._duration: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_index: int = 0
        self._last_sample_time: float = -999.0
        self._glb_loop: Optional[GLib.MainLoop] = None
        self._bus_thread: Optional[threading.Thread] = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def _build_pipeline(self) -> Gst.Pipeline:
        pipeline_str = (
            f'filesrc location="{self.video_path}" ! '
            f'decodebin name=demux ! '
            f'videoconvert ! '
            f'video/x-raw,format=RGB ! '
            f'appsink name=videosink emit-signals=true max-buffers=60 drop=true'
        )
        pipeline = Gst.parse_launch(pipeline_str)
        self._appsink = pipeline.get_by_name("videosink")

        demux = pipeline.get_by_name("demux")
        demux.connect("pad-added", self._on_pad_added)

        self._appsink.connect("new-sample", self._on_new_sample)

        return pipeline

    def _on_pad_added(self, element, pad):
        caps = pad.get_current_caps()
        if not caps:
            return
        structure = caps.get_structure(0)
        if structure.has_field("framerate"):
            fps_num, fps_den = structure.get_fraction("framerate")
            if fps_den > 0:
                self._fps = fps_num / fps_den
        if structure.has_field("width") and structure.has_field("height"):
            _, self._width = structure.get_int("width")
            _, self._height = structure.get_int("height")

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
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

        success, buffer_map = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            frame_data = np.frombuffer(buffer_map.data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(buffer_map)

        pts = buffer.pts
        if pts != Gst.CLOCK_TIME_NONE:
            timestamp = pts / Gst.SECOND
        else:
            timestamp = self._frame_index / max(self._fps, 1.0)

        duration_ns = buffer.duration
        if duration_ns != Gst.CLOCK_TIME_NONE:
            frame_dur = duration_ns / Gst.SECOND
        else:
            frame_dur = 1.0 / max(self._fps, 1.0)

        if timestamp - self._last_sample_time >= self.sample_interval_sec or self._frame_index == 0:
            packet = FramePacket(
                frame=frame,
                timestamp=round(timestamp, 3),
                frame_index=self._frame_index,
                duration=round(frame_dur, 3)
            )
            try:
                self._frame_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(packet)
                except queue.Empty:
                    pass
            self._last_sample_time = timestamp

        self._frame_index += 1
        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self._eos_event.set()
            if self._glb_loop:
                self._glb_loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._error = f"GStreamer error: {err.message} (debug: {debug})"
            print(f"[ERROR] {self._error}")
            self._eos_event.set()
            if self._glb_loop:
                self._glb_loop.quit()

    def _run_bus_loop(self):
        self._glb_loop = GLib.MainLoop()
        try:
            self._glb_loop.run()
        except Exception:
            pass

    def open(self) -> bool:
        try:
            self._pipeline = self._build_pipeline()
        except Exception as e:
            self._error = f"Failed to build pipeline: {e}"
            print(f"[ERROR] {self._error}")
            try:
                self._fallback_open()
                return True
            except Exception as fe:
                self._error = f"Fallback also failed: {fe}"
                return False

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        dur_query = self._pipeline.query_duration(Gst.Format.TIME)
        if dur_query:
            _, dur_ns = dur_query
            if dur_ns > 0:
                self._duration = dur_ns / Gst.SECOND

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._error = "Failed to start pipeline (state change to PLAYING failed)"
            print(f"[WARN] {self._error}, trying OpenCV fallback...")
            self._pipeline.set_state(Gst.State.NULL)
            try:
                self._fallback_open()
                return True
            except Exception as fe:
                self._error = f"Fallback also failed: {fe}"
                return False

        self._bus_thread = threading.Thread(target=self._run_bus_loop, daemon=True)
        self._bus_thread.start()

        return True

    def _fallback_open(self):
        import cv2
        print("[INFO] Using OpenCV fallback decoder (GStreamer unavailable)")
        self._use_fallback = True
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = total_frames / self._fps if self._fps > 0 else 0
        self._fallback_frame_idx = 0

    def _fallback_read(self) -> Optional[FramePacket]:
        import cv2
        sample_interval = max(1, int(self._fps * self.sample_interval_sec))

        while True:
            ret, frame = self._cap.read()
            if not ret:
                return None

            if self._fallback_frame_idx % sample_interval == 0:
                timestamp = self._fallback_frame_idx / self._fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                packet = FramePacket(
                    frame=frame_rgb,
                    timestamp=round(timestamp, 3),
                    frame_index=self._fallback_frame_idx,
                    duration=round(self.sample_interval_sec, 3)
                )
                self._fallback_frame_idx += 1
                return packet

            self._fallback_frame_idx += 1

    def read_frame(self, timeout: float = 10.0) -> Optional[FramePacket]:
        if getattr(self, '_use_fallback', False):
            return self._fallback_read()

        if self._eos_event.is_set() and self._frame_queue.empty():
            return None
        if self._error and self._frame_queue.empty():
            raise RuntimeError(self._error)

        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            if self._eos_event.is_set():
                return None
            return None

    def read_all_frames(self,
                        progress_callback: Optional[Callable[[int, int], None]] = None
                        ) -> list:
        frames = []
        total_estimate = int(self._duration / self.sample_interval_sec) if self._duration > 0 else 0
        count = 0

        while True:
            packet = self.read_frame(timeout=15.0)
            if packet is None:
                break
            frames.append(packet)
            count += 1
            if progress_callback and count % 5 == 0:
                progress_callback(count, total_estimate)

        return frames

    def close(self):
        if getattr(self, '_use_fallback', False):
            if hasattr(self, '_cap') and self._cap is not None:
                self._cap.release()
                self._cap = None
            return

        if self._pipeline:
            self._pipeline.send_event(Gst.Event.new_eos())
            time.sleep(0.3)
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        if self._glb_loop:
            try:
                self._glb_loop.quit()
            except Exception:
                pass

        if self._bus_thread and self._bus_thread.is_alive():
            self._bus_thread.join(timeout=3.0)

        drained = 0
        while not self._frame_queue.empty() and drained < 100:
            try:
                self._frame_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class GStreamerAudioExtractor:
    def __init__(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path

    def extract_audio(self) -> bool:
        try:
            pipeline_str = (
                f'filesrc location="{self.video_path}" ! '
                f'decodebin ! '
                f'audioconvert ! '
                f'audioresample ! '
                f'audio/x-raw,format=S16LE,rate=16000,channels=1 ! '
                f'wavenc ! '
                f'filesink location="{self.output_path}"'
            )
            pipeline = Gst.parse_launch(pipeline_str)
            pipeline.set_state(Gst.State.PLAYING)

            bus = pipeline.get_bus()
            bus.timed_pop_filtered(
                Gst.CLOCK_TIME_NONE,
                Gst.MessageType.EOS | Gst.MessageType.ERROR
            )

            pipeline.set_state(Gst.State.NULL)
            return True

        except Exception as e:
            print(f"[WARN] GStreamer audio extraction failed: {e}, using ffmpeg fallback")
            return self._ffmpeg_fallback()

    def _ffmpeg_fallback(self) -> bool:
        import subprocess
        try:
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', self.output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            return True
        except Exception as e:
            print(f"[WARN] ffmpeg audio extraction also failed: {e}")
            return False

    def extract_segment(self, start_sec: float, end_sec: float, output_path: str) -> bool:
        try:
            pipeline_str = (
                f'filesrc location="{self.video_path}" ! '
                f'decodebin ! '
                f'audioconvert ! '
                f'audioresample ! '
                f'audio/x-raw,format=S16LE,rate=16000,channels=1 ! '
                f'wavenc ! '
                f'filesink location="{output_path}"'
            )
            pipeline = Gst.parse_launch(pipeline_str)

            seek_flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
            pipeline.seek_simple(
                Gst.Format.TIME,
                seek_flags,
                int(start_sec * Gst.SECOND)
            )

            bus = pipeline.get_bus()

            def on_eos(message):
                pipeline.set_state(Gst.State.NULL)

            pipeline.set_state(Gst.State.PLAYING)

            start_time = time.time()
            while True:
                msg = bus.timed_pop(100 * Gst.MSECOND)
                if msg:
                    if msg.type == Gst.MessageType.EOS:
                        pipeline.set_state(Gst.State.NULL)
                        return True
                    elif msg.type == Gst.MessageType.ERROR:
                        pipeline.set_state(Gst.State.NULL)
                        return False

                elapsed = time.time() - start_time
                current_pos = start_sec + elapsed
                if current_pos >= end_sec:
                    pipeline.send_event(Gst.Event.new_eos())
                    time.sleep(0.5)
                    pipeline.set_state(Gst.State.NULL)
                    return True

                if elapsed > (end_sec - start_sec) + 5:
                    pipeline.set_state(Gst.State.NULL)
                    return False

        except Exception as e:
            print(f"[WARN] Audio segment extraction failed: {e}")
            return self._ffmpeg_segment_fallback(start_sec, end_sec, output_path)

    def _ffmpeg_segment_fallback(self, start_sec: float, end_sec: float, output_path: str) -> bool:
        import subprocess
        try:
            duration = end_sec - start_sec
            cmd = [
                'ffmpeg', '-ss', str(start_sec),
                '-i', self.video_path,
                '-t', str(duration),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            return True
        except Exception:
            return False
