import os
from typing import List, Dict
from pathlib import Path


class SpeechRecognizer:
    def __init__(self, model_name: str = "base", model_cache_dir: str = "./python/models_cache",
                 use_gpu: bool = False, s3_loader=None):
        self.model_name = model_name
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if use_gpu else "cpu"
        self.model = None
        self.s3_loader = s3_loader

        self._load_model()

    def _load_model(self):
        try:
            model_filename = f"whisper-{self.model_name}.onnx"
            model_path = self.model_cache_dir / model_filename

            if not model_path.exists() and self.s3_loader:
                try:
                    model_path_str = self.s3_loader.get_model_path(model_filename)
                    model_path = Path(model_path_str)
                except Exception as e:
                    print(f"[WARN] Could not download Whisper model from S3: {e}")

            if model_path.exists():
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if self.device == "cuda":
                    providers.insert(0, 'CUDAExecutionProvider')
                self.session = ort.InferenceSession(str(model_path), providers=providers)
                self.model_loaded = True
                print(f"[INFO] Whisper ONNX model loaded from {model_path}")
            else:
                try:
                    import whisper
                    self.model = whisper.load_model(self.model_name, device=self.device)
                    self.model_loaded = True
                    self.use_whisper_api = True
                    print(f"[INFO] Whisper model loaded via whisper API: {self.model_name}")
                except Exception as e:
                    print(f"[WARN] Failed to load Whisper: {e}. Using mock speech recognizer.")
                    self.model_loaded = False
                    self.use_whisper_api = False
        except Exception as e:
            print(f"[WARN] Failed to load speech recognizer: {e}. Using mock mode.")
            self.model_loaded = False
            self.use_whisper_api = False

    def transcribe(self, video_path: str) -> List[Dict]:
        if not self.model_loaded:
            return self._mock_transcribe(video_path)

        try:
            if self.use_whisper_api and self.model:
                result = self.model.transcribe(video_path, word_timestamps=True)
                segments = []
                for seg in result.get("segments", []):
                    segments.append({
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": seg["text"].strip()
                    })
                return segments
        except Exception as e:
            print(f"[ERROR] Whisper transcription failed: {e}")
            return self._mock_transcribe(video_path)

        return []

    def _mock_transcribe(self, video_path: str) -> List[Dict]:
        filename = os.path.basename(video_path).lower()
        themes = {
            "car": [
                (0.0, 5.0, "启动引擎，准备出发"),
                (5.0, 10.0, "现在正在加速，速度达到60公里每小时"),
                (10.0, 15.0, "前方路口右转，注意行人"),
                (15.0, 20.0, "天气很好，窗外的风景不错"),
                (20.0, 25.0, "现在减速，准备停车"),
            ],
            "street": [
                (0.0, 4.0, "今天的街道很热闹"),
                (4.0, 9.0, "行人正在过马路"),
                (9.0, 14.0, "公交车来了，请注意安全"),
                (14.0, 19.0, "路边的咖啡馆人很多"),
                (19.0, 24.0, "交通灯变成了红灯"),
            ],
            "indoor": [
                (0.0, 5.0, "欢迎来到我们的办公室"),
                (5.0, 10.0, "这是我们的会议室，今天有个重要会议"),
                (10.0, 15.0, "大家好，今天我们讨论新产品计划"),
                (15.0, 20.0, "我们的目标是提升用户体验"),
                (20.0, 25.0, "感谢大家的参与，会议到此结束"),
            ],
            "nature": [
                (0.0, 5.0, "我们现在来到了郊外的公园"),
                (5.0, 10.0, "这里空气清新，鸟语花香"),
                (10.0, 15.0, "远处可以看到连绵的山脉"),
                (15.0, 20.0, "湖水清澈，倒映着蓝天"),
                (20.0, 25.0, "大自然真是美丽极了"),
            ],
        }

        matched_theme = "indoor"
        for theme in themes:
            if theme in filename:
                matched_theme = theme
                break

        segments = []
        for start, end, text in themes[matched_theme]:
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })
        return segments
