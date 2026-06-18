import os
import sys
import json
import uuid
import threading
import logging
import base64
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s3_loader import S3ModelLoader
from classifier import ImageClassifier
from detector import ObjectDetector
from speech_recognizer import SpeechRecognizer
from video_processor import VideoProcessor, ParallelVideoProcessor, PipelineConfig
from vector_search import VectorSearchEngine
from llm_corrector import BatchLLMCorrector
from srt_generator import SRTGenerator
from summary_generator import SummaryGenerator
from clip_retriever import CLIPRetriever
from privacy_protector import PrivacyProtector
from rtmp_stream import RTMPStreamManager

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

processing_tasks = {}
search_engines = {}


def initialize_models():
    logger.info("[INFO] Initializing models...")
    s3_loader = S3ModelLoader()

    class_model_path = os.getenv('CLASSIFICATION_MODEL', 'classification_model.onnx')
    det_model_path = os.getenv('DETECTION_MODEL', 'yolov8n.onnx')
    whisper_model = os.getenv('WHISPER_MODEL', 'base')
    llm_model = os.getenv('LLM_MODEL', 'Qwen/Qwen2-7B-Instruct')
    clip_model = os.getenv('CLIP_MODEL', 'openai/clip-vit-base-patch32')
    use_vllm = os.getenv('USE_VLLM', 'false').lower() == 'true'
    vllm_url = os.getenv('VLLM_URL', 'http://localhost:8000')

    try:
        class_model = s3_loader.get_model_path(class_model_path)
    except FileNotFoundError:
        class_model = os.path.join(os.path.dirname(__file__), 'dummy.onnx')
        logger.warning("[WARN] Classification model not found, will use mock mode")

    try:
        det_model = s3_loader.get_model_path(det_model_path)
    except FileNotFoundError:
        det_model = os.path.join(os.path.dirname(__file__), 'dummy_detect.onnx')
        logger.warning("[WARN] Detection model not found, will use mock mode")

    use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'

    classifier = ImageClassifier(class_model, use_gpu=use_gpu)
    detector = ObjectDetector(det_model, use_gpu=use_gpu)
    speech_rec = SpeechRecognizer(
        model_name=whisper_model,
        model_cache_dir=str(s3_loader.cache_dir),
        use_gpu=use_gpu, s3_loader=s3_loader
    )

    llm_corrector = BatchLLMCorrector(
        model_name=llm_model, use_gpu=use_gpu,
        use_vllm=use_vllm, vllm_url=vllm_url
    )

    summary_generator = SummaryGenerator(
        model_name=llm_model, use_gpu=use_gpu,
        use_vllm=use_vllm, vllm_url=vllm_url
    )

    clip_retriever = CLIPRetriever(
        model_name=clip_model,
        cache_dir=str(s3_loader.cache_dir)
    )

    privacy_protector = PrivacyProtector(
        detector=detector,
        blur_mode=os.getenv('PRIVACY_BLUR_MODE', 'pixelate'),
        blur_strength=int(os.getenv('PRIVACY_BLUR_STRENGTH', '25'))
    )

    srt_generator = SRTGenerator(output_dir=str(RESULTS_DIR))

    config = PipelineConfig(
        frame_sample_rate=float(os.getenv('FRAME_SAMPLE_RATE', '5.0')),
        vision_pool_size=int(os.getenv('VISION_POOL_SIZE', '2')),
        enable_llm_correction=os.getenv('ENABLE_LLM_CORRECTION', 'true').lower() == 'true',
        enable_srt_output=True,
        enable_summary=os.getenv('ENABLE_SUMMARY', 'true').lower() == 'true',
        enable_privacy=os.getenv('ENABLE_PRIVACY', 'false').lower() == 'true',
        enable_clip_index=os.getenv('ENABLE_CLIP_INDEX', 'true').lower() == 'true',
        privacy_blur_mode=os.getenv('PRIVACY_BLUR_MODE', 'pixelate'),
        privacy_blur_strength=int(os.getenv('PRIVACY_BLUR_STRENGTH', '25')),
        rtmp_latency_ms=int(os.getenv('RTMP_LATENCY_MS', '1000')),
        rtmp_sample_interval=float(os.getenv('RTMP_SAMPLE_INTERVAL', '2.0')),
    )

    clip_index_path = RESULTS_DIR / "clip_index.pkl"
    if clip_index_path.exists():
        try:
            clip_retriever.load(str(clip_index_path))
        except Exception as e:
            logger.warning(f"[WARN] Failed to load CLIP index: {e}")

    processor = ParallelVideoProcessor(
        classifier=classifier, detector=detector,
        speech_recognizer=speech_rec,
        llm_corrector=llm_corrector,
        srt_generator=srt_generator,
        summary_generator=summary_generator,
        clip_retriever=clip_retriever,
        privacy_protector=privacy_protector,
        config=config
    )

    return processor


processor = initialize_models()


@app.route('/api/health', methods=['GET'])
def health_check():
    clip_available = processor.clip_retriever is not None and processor.clip_retriever._model_loaded
    summary_available = processor.summary_generator is not None and processor.summary_generator._model_loaded
    return jsonify({
        "status": "ok",
        "service": "video-understanding-python",
        "models": {
            "classifier": processor.classifier.model_loaded,
            "detector": processor.detector.model_loaded,
            "speech": processor.speech_recognizer.model_loaded
        },
        "features": {
            "llm_correction": processor.llm_corrector is not None and processor.llm_corrector._model_loaded,
            "summary": summary_available,
            "clip_search": clip_available,
            "privacy": processor.privacy_protector is not None,
            "rtmp_live": True
        },
        "pipeline": {
            "mode": "parallel_gstreamer",
            "vision_pool_size": processor.config.vision_pool_size,
            "frame_sample_rate": processor.config.frame_sample_rate,
            "privacy_enabled": processor.config.enable_privacy,
            "clip_indexed_videos": len(processor.clip_retriever.video_features) if processor.clip_retriever else 0
        }
    })


@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video_info(video_id):
    result_path = RESULTS_DIR / f"{video_id}.json"
    if result_path.exists():
        with open(result_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Video not found"}), 404


@app.route('/api/process', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get('video_path')
    video_id = data.get('video_id', str(uuid.uuid4()))
    privacy_mode = data.get('privacy_mode', False)

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = {
        "status": "processing", "progress": 0.0,
        "message": "任务已启动", "video_id": video_id, "result": None
    }

    def progress_callback(progress, message):
        if task_id in processing_tasks:
            processing_tasks[task_id]["progress"] = progress
            processing_tasks[task_id]["message"] = message

    def run_processing():
        try:
            result = processor.process_video(video_path, progress_callback, privacy_mode=privacy_mode)
            result["video_id"] = video_id

            if processor.clip_retriever and processor.config.enable_clip_index:
                try:
                    clip_index_path = str(RESULTS_DIR / "clip_index.pkl")
                    processor.clip_retriever.add_video(video_id, processor.clip_retriever.encode_video([]), {
                        "video_path": video_path
                    })
                except Exception as e:
                    logger.warning(f"[WARN] CLIP indexing skipped: {e}")

            result_path = RESULTS_DIR / f"{video_id}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            search_engine = VectorSearchEngine()
            search_engine.build_index(result["timeline"], video_id)
            search_engines[video_id] = search_engine
            index_path = RESULTS_DIR / f"{video_id}.index"
            search_engine.save(str(index_path))

            processing_tasks[task_id]["status"] = "completed"
            processing_tasks[task_id]["progress"] = 1.0
            processing_tasks[task_id]["message"] = "处理完成"
            processing_tasks[task_id]["result"] = result

        except Exception as e:
            import traceback
            traceback.print_exc()
            processing_tasks[task_id]["status"] = "failed"
            processing_tasks[task_id]["message"] = str(e)

    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "video_id": video_id, "status": "processing"})


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        response = {
            "task_id": task_id, "status": task["status"],
            "progress": task["progress"], "message": task["message"],
            "video_id": task["video_id"]
        }
        if task["result"]:
            r = task["result"]
            response["result"] = {
                "video_id": r.get("video_id"),
                "total_duration": r.get("total_duration"),
                "segment_count": r.get("segment_count"),
                "srt_path": r.get("srt_path"),
                "summary": r.get("summary"),
                "privacy_mode": r.get("privacy_mode"),
                "pipeline_mode": r.get("pipeline_mode")
            }
        return jsonify(response)
    return jsonify({"error": "Task not found"}), 404


@app.route('/api/search', methods=['POST'])
def search_video():
    data = request.json
    video_id = data.get('video_id')
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    threshold = data.get('threshold', 0.3)

    if not video_id or not query:
        return jsonify({"error": "video_id and query required"}), 400

    if video_id not in search_engines:
        index_path = RESULTS_DIR / f"{video_id}.index"
        result_path = RESULTS_DIR / f"{video_id}.json"
        if index_path.exists() and result_path.exists():
            search_engine = VectorSearchEngine()
            search_engine.load(str(index_path))
            with open(result_path, 'r', encoding='utf-8') as f:
                search_engine.segments = json.load(f).get("timeline", [])
            search_engines[video_id] = search_engine
        else:
            return jsonify({"error": "Video index not found"}), 404

    results = search_engines[video_id].search(query, top_k=top_k, threshold=threshold)
    return jsonify({"video_id": video_id, "query": query, "result_count": len(results), "results": results})


@app.route('/api/subtitles/<video_id>', methods=['GET'])
def get_subtitles(video_id):
    fmt = request.args.get('format', 'srt')
    if fmt == 'srt':
        srt_path = RESULTS_DIR / f"{video_id}.srt"
        if srt_path.exists():
            return send_file(str(srt_path), mimetype='text/plain', as_attachment=True, download_name=f"{video_id}.srt")
    elif fmt == 'vtt':
        vtt_path = RESULTS_DIR / f"{video_id}.vtt"
        if vtt_path.exists():
            return send_file(str(vtt_path), mimetype='text/vtt', as_attachment=True, download_name=f"{video_id}.vtt")
    elif fmt == 'json':
        json_path = RESULTS_DIR / f"{video_id}_subtitles.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
    return jsonify({"error": "Subtitles not found"}), 404


@app.route('/api/correct', methods=['POST'])
def correct_transcription():
    data = request.json
    video_id = data.get('video_id')
    segments = data.get('segments')
    context = data.get('context', '')

    if video_id:
        result_path = RESULTS_DIR / f"{video_id}.json"
        if not result_path.exists():
            return jsonify({"error": "Video result not found"}), 404
        with open(result_path, 'r', encoding='utf-8') as f:
            segments = json.load(f).get("speech_segments", [])
        if not segments:
            return jsonify({"error": "No speech segments found"}), 404
    elif not segments:
        return jsonify({"error": "video_id or segments required"}), 400

    if not processor.llm_corrector:
        return jsonify({"error": "LLM corrector not available"}), 503

    corrected = processor.llm_corrector.correct_segments(segments, context)
    return jsonify({"video_id": video_id, "corrected_segments": corrected})


@app.route('/api/summary/<video_id>', methods=['GET'])
def get_summary(video_id):
    result_path = RESULTS_DIR / f"{video_id}.json"
    if not result_path.exists():
        return jsonify({"error": "Video not found"}), 404

    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    summary = result.get("summary")
    if summary:
        return jsonify({"video_id": video_id, "summary": summary})

    if not processor.summary_generator:
        return jsonify({"error": "Summary generator not available"}), 503

    timeline = result.get("timeline", [])
    speech_segments = result.get("speech_segments", [])
    smart_summary = processor.summary_generator.generate(timeline, speech_segments)
    summary_data = {
        "summary": smart_summary.summary,
        "keywords": smart_summary.keywords,
        "scene_summary": smart_summary.scene_summary,
        "speech_summary": smart_summary.speech_summary,
        "objects_summary": smart_summary.objects_summary
    }

    result["summary"] = summary_data
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return jsonify({"video_id": video_id, "summary": summary_data})


@app.route('/api/summary/generate', methods=['POST'])
def generate_summary():
    data = request.json
    timeline = data.get('timeline', [])
    speech_segments = data.get('speech_segments')

    if not timeline:
        return jsonify({"error": "timeline required"}), 400

    if not processor.summary_generator:
        return jsonify({"error": "Summary generator not available"}), 503

    summary = processor.summary_generator.generate(timeline, speech_segments)
    return jsonify({
        "summary": summary.summary,
        "keywords": summary.keywords,
        "scene_summary": summary.scene_summary,
        "speech_summary": summary.speech_summary,
        "objects_summary": summary.objects_summary
    })


@app.route('/api/live/start', methods=['POST'])
def start_live_stream():
    data = request.json
    rtmp_url = data.get('rtmp_url')
    stream_id = data.get('stream_id')

    if not rtmp_url:
        return jsonify({"error": "rtmp_url required"}), 400

    try:
        sid = processor.start_live_stream(rtmp_url, stream_id)
        return jsonify({"stream_id": sid, "rtmp_url": rtmp_url, "status": "running"})
    except Exception as e:
        logger.error(f"[LIVE] Failed to start stream: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/live/stop/<stream_id>', methods=['POST'])
def stop_live_stream(stream_id):
    processor.stop_live_stream(stream_id)
    return jsonify({"stream_id": stream_id, "status": "stopped"})


@app.route('/api/live/status', methods=['GET'])
def list_live_streams():
    if not processor.rtmp_manager:
        return jsonify({"streams": []})
    return jsonify({"streams": processor.rtmp_manager.list_streams()})


@app.route('/api/live/status/<stream_id>', methods=['GET'])
def get_live_status(stream_id):
    if not processor.rtmp_manager:
        return jsonify({"error": "No stream manager"}), 404
    info = processor.rtmp_manager.get_stream_info(stream_id)
    if not info:
        return jsonify({"error": "Stream not found"}), 404
    return jsonify(info)


@app.route('/api/clip/index', methods=['POST'])
def index_video_clip():
    data = request.json
    video_id = data.get('video_id')
    video_path = data.get('video_path')

    if not video_id or not video_path:
        return jsonify({"error": "video_id and video_path required"}), 400

    if not processor.clip_retriever:
        return jsonify({"error": "CLIP retriever not available"}), 503

    try:
        result = processor.index_video_for_clip(video_id, video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clip/search', methods=['POST'])
def search_similar_videos():
    data = request.json
    query_video_path = data.get('video_path')
    top_k = data.get('top_k', 5)
    exclude_id = data.get('exclude_id')
    threshold = data.get('threshold', 0.3)

    if not query_video_path:
        return jsonify({"error": "video_path required"}), 400

    if not processor.clip_retriever:
        return jsonify({"error": "CLIP retriever not available"}), 503

    try:
        results = processor.search_similar_videos(
            query_video_path, top_k=top_k, exclude_id=exclude_id, threshold=threshold
        )
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clip/search_frame', methods=['POST'])
def search_by_frame():
    data = request.json
    frame_b64 = data.get('frame_base64')
    top_k = data.get('top_k', 5)
    threshold = data.get('threshold', 0.3)

    if not frame_b64:
        return jsonify({"error": "frame_base64 required"}), 400

    if not processor.clip_retriever:
        return jsonify({"error": "CLIP retriever not available"}), 503

    try:
        img_data = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        import cv2
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = processor.search_similar_by_frame(frame_rgb, top_k=top_k, threshold=threshold)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/privacy/detect', methods=['POST'])
def detect_privacy_regions():
    data = request.json
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({"error": "video_id required"}), 400

    result_path = RESULTS_DIR / f"{video_id}.json"
    if not result_path.exists():
        return jsonify({"error": "Video not found"}), 404

    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    privacy_regions = []
    for entry in result.get("timeline", []):
        regions = entry.get("privacy_regions", [])
        if regions:
            privacy_regions.append({
                "timestamp": entry["timestamp"],
                "regions": regions
            })

    return jsonify({"video_id": video_id, "privacy_regions": privacy_regions})


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == '__main__':
    port = int(os.getenv('PYTHON_PORT', 5000))
    logger.info(f"[INFO] Starting Python backend on port {port}")
    logger.info(f"[INFO] Pipeline: parallel_gstreamer | Vision pool: {processor.config.vision_pool_size}")
    logger.info(f"[INFO] Summary: {'enabled' if processor.config.enable_summary else 'disabled'}")
    logger.info(f"[INFO] Privacy: {'enabled' if processor.config.enable_privacy else 'disabled'}")
    logger.info(f"[INFO] CLIP: {'enabled' if processor.config.enable_clip_index else 'disabled'}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
