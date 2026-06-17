import os
import sys
import json
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s3_loader import S3ModelLoader
from classifier import ImageClassifier
from detector import ObjectDetector
from speech_recognizer import SpeechRecognizer
from video_processor import VideoProcessor
from vector_search import VectorSearchEngine


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
    print("[INFO] Initializing models...")
    s3_loader = S3ModelLoader()

    class_model_path = os.getenv('CLASSIFICATION_MODEL', 'classification_model.onnx')
    det_model_path = os.getenv('DETECTION_MODEL', 'yolov8n.onnx')
    whisper_model = os.getenv('WHISPER_MODEL', 'base')

    try:
        class_model = s3_loader.get_model_path(class_model_path)
    except FileNotFoundError:
        class_model = os.path.join(os.path.dirname(__file__), 'dummy.onnx')
        print(f"[WARN] Classification model not found, will use mock mode")

    try:
        det_model = s3_loader.get_model_path(det_model_path)
    except FileNotFoundError:
        det_model = os.path.join(os.path.dirname(__file__), 'dummy_detect.onnx')
        print(f"[WARN] Detection model not found, will use mock mode")

    use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'

    classifier = ImageClassifier(class_model, use_gpu=use_gpu)
    detector = ObjectDetector(det_model, use_gpu=use_gpu)
    speech_rec = SpeechRecognizer(
        model_name=whisper_model,
        model_cache_dir=str(s3_loader.cache_dir),
        use_gpu=use_gpu,
        s3_loader=s3_loader
    )

    processor = VideoProcessor(classifier, detector, speech_rec, frame_sample_rate=5)

    return processor


processor = initialize_models()


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "video-understanding-python",
        "models": {
            "classifier": processor.classifier.model_loaded,
            "detector": processor.detector.model_loaded,
            "speech": processor.speech_recognizer.model_loaded
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

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    task_id = str(uuid.uuid4())

    processing_tasks[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "任务已启动",
        "video_id": video_id,
        "result": None
    }

    def progress_callback(progress, message):
        if task_id in processing_tasks:
            processing_tasks[task_id]["progress"] = progress
            processing_tasks[task_id]["message"] = message

    def run_processing():
        try:
            result = processor.process_video(video_path, progress_callback)
            result["video_id"] = video_id

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

    return jsonify({
        "task_id": task_id,
        "video_id": video_id,
        "status": "processing"
    })


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        response = {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "video_id": task["video_id"]
        }
        if task["result"]:
            response["result"] = {
                "video_id": task["result"].get("video_id"),
                "total_duration": task["result"].get("total_duration"),
                "segment_count": task["result"].get("segment_count")
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

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400
    if not query:
        return jsonify({"error": "query is required"}), 400

    if video_id not in search_engines:
        index_path = RESULTS_DIR / f"{video_id}.index"
        result_path = RESULTS_DIR / f"{video_id}.json"

        if index_path.exists() and result_path.exists():
            search_engine = VectorSearchEngine()
            search_engine.load(str(index_path))

            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            search_engine.segments = result_data.get("timeline", [])

            search_engines[video_id] = search_engine
        else:
            return jsonify({"error": "Video index not found. Please process the video first."}), 404

    search_engine = search_engines[video_id]
    results = search_engine.search(query, top_k=top_k, threshold=threshold)

    return jsonify({
        "video_id": video_id,
        "query": query,
        "result_count": len(results),
        "results": results
    })


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == '__main__':
    port = int(os.getenv('PYTHON_PORT', 5000))
    print(f"[INFO] Starting Python backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
