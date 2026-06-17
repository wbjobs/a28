# 视频理解分析平台 (Video Understanding Platform)

一个多模态视频理解 Web 应用，基于 Flutter Web + Node.js + Python 混合架构。

## 功能特性

- 🎥 **视频上传与处理**: 支持 MP4/AVI/MOV/MKV/WebM 格式视频上传
- 🖼️ **图像分类**: 基于 ONNX Runtime 的预训练模型，识别每一帧场景类别
- 🔍 **目标检测**: YOLOv8 ONNX 模型检测帧中的物体和人物（80 COCO 类别）
- 🎙️ **语音识别**: Whisper 模型提取视频中的对话文字和时间戳
- 📊 **多模态时间线**: 融合三路信息，生成包含时间戳、场景、物体、对话的时间线
- 🎤 **语音查询**: 使用 Web Speech API 实现语音指令输入
- 🔎 **跨模态向量检索**: FAISS + SentenceTransformers，支持"找有汽车和'加速'这个词的片段"等复杂查询
- ☁️ **S3 模型动态加载**: 模型文件从 S3 自动下载缓存

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Flutter Web Frontend                       │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐   │
│  │ Video List │ │ Timeline   │ │ Voice Search / Results   │   │
│  └─────┬──────┘ └─────┬──────┘ └────────────┬─────────────┘   │
└────────┼───────────────┼─────────────────────┼─────────────────┘
         │               │                     │
         ▼               ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    Node.js Backend (3000)                     │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐   │
│  │ Upload     │ │ REST API   │ │ WebSocket (task progress)│   │
│  └─────┬──────┘ └─────┬──────┘ └────────────┬─────────────┘   │
└────────┼───────────────┼─────────────────────┼─────────────────┘
         │               │                     │
         └───────────────┴─────────────────────┘
                         │ HTTP
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Python Backend (5000)                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│  │ Classifier │ │ Detector   │ │ Whisper    │ │ FAISS Search│ │
│  │ (ONNX)     │ │ (YOLOv8)   │ │ (ASR)      │ │ (Vectors)  │ │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
│        │              │              │              │        │
│        └──────────────┴────────┬─────┴──────────────┘        │
│                               ▼                               │
│                    ┌────────────────────┐                     │
│                    │  S3 Model Loader   │                     │
│                    └────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Flutter 3.10+（仅构建前端时需要）
- （可选）S3 兼容存储用于模型托管

### 一键启动（Windows）

```bat
start.bat
```

### 一键启动（macOS / Linux）

```bash
chmod +x start.sh
./start.sh
```

### 手动启动

#### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `S3_BUCKET_NAME` | S3 存储桶名称 | `video-understanding-models` |
| `S3_REGION` | S3 区域 | `us-east-1` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | S3 凭证 | 可选，不配置则仅使用本地缓存 |
| `CLASSIFICATION_MODEL` | 分类模型文件名 | `classification_model.onnx` |
| `DETECTION_MODEL` | 检测模型文件名 | `yolov8n.onnx` |
| `WHISPER_MODEL` | Whisper 模型大小 | `base` |
| `NODE_PORT` | Node.js 服务端口 | `3000` |
| `PYTHON_PORT` | Python 服务端口 | `5000` |

#### 2. Python 后端

```bash
cd python
python -m venv venv
# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Python 服务将在 `http://localhost:5000` 启动。

> 💡 如未配置 S3 或模型文件不存在，系统会自动进入 mock 模式，使用模拟数据保证功能可演示。

#### 3. Node.js 后端

```bash
npm install
npm start
```

Node.js 服务将在 `http://localhost:3000` 启动。

#### 4. 构建 Flutter Web 前端（可选）

```bash
cd frontend
flutter pub get
flutter build web
```

构建产物会输出到 `frontend/build/web`，Node.js 服务会自动托管该目录。

如果不构建前端，可直接通过 API 接口使用后端服务。

## 使用方式

1. 打开 `http://localhost:3000`
2. 点击「上传视频」选择本地 MP4 文件
3. 等待系统自动完成处理（进度显示在顶部）
4. 查看左侧时间线，点击任意片段可跳转播放
5. 在右侧搜索框输入查询，或点击麦克风图标进行语音查询
   - 示例查询：
     - `找到有汽车的片段`
     - `查找有人和"加速"这个词的场景`
     - `户外自然场景中的对话`
     - `包含食物的室内画面`
6. 点击搜索结果可跳转并高亮对应视频片段

## API 接口

### Node.js API (`http://localhost:3000`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/upload` | 上传视频文件（multipart/form-data: `video`） |
| POST | `/api/process` | 启动视频分析处理 |
| GET | `/api/tasks/:task_id` | 查询处理任务状态 |
| GET | `/api/videos` | 列出所有视频 |
| GET | `/api/videos/:video_id` | 获取视频分析结果 |
| POST | `/api/search` | 跨模态搜索 |
| WS | `/ws` | WebSocket 实时任务进度推送 |

### Python API (`http://localhost:5000`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查及模型加载状态 |
| POST | `/api/process` | 启动视频处理 |
| GET | `/api/tasks/:task_id` | 查询任务进度 |
| GET | `/api/videos/:video_id` | 获取分析结果 |
| POST | `/api/search` | FAISS 向量检索 |

## 项目结构

```
a28/
├── server/                    # Node.js 后端
│   └── index.js              # Express API + WebSocket
├── python/                    # Python 后端
│   ├── app.py                # Flask 应用入口
│   ├── s3_loader.py          # S3 模型加载器
│   ├── classifier.py         # 图像分类模型 (ONNX)
│   ├── detector.py           # 目标检测模型 (YOLOv8 ONNX)
│   ├── speech_recognizer.py  # Whisper 语音识别
│   ├── video_processor.py    # 视频处理管道
│   ├── query_parser.py       # 查询解析
│   ├── vector_search.py      # FAISS 向量检索引擎
│   └── requirements.txt      # Python 依赖
├── frontend/                  # Flutter Web 前端
│   ├── lib/
│   │   ├── main.dart         # 应用入口
│   │   ├── models/models.dart
│   │   ├── services/
│   │   │   ├── api_service.dart
│   │   │   └── speech_recognition.dart  # Web Speech API 封装
│   │   ├── widgets/
│   │   │   ├── timeline_view.dart
│   │   │   ├── search_bar.dart
│   │   │   └── search_results.dart
│   │   └── pages/home_page.dart
│   ├── web/
│   │   └── index.html
│   └── pubspec.yaml
├── uploads/                   # 上传的视频文件（自动创建）
├── results/                   # 分析结果 JSON + FAISS 索引
├── package.json              # Node.js 依赖
├── start.bat / start.sh      # 启动脚本
└── .env.example              # 环境变量模板
```

## 模型部署说明

### 将模型上传至 S3

1. 准备 ONNX 格式的模型文件：
   - 图像分类：如 MobileNetV2、ResNet50 等（ONNX 格式）
   - 目标检测：YOLOv8n ONNX（可通过 `yolo export model=yolov8n.pt format=onnx` 导出）
   - Whisper：可选，也可直接使用 whisper Python 包在线下载

2. 上传至 S3 存储桶：

```bash
aws s3 cp classification_model.onnx s3://your-bucket/
aws s3 cp yolov8n.onnx s3://your-bucket/
```

3. 配置 `.env` 中的 S3 凭证

### 本地模型（无需 S3）

将 ONNX 模型文件放入 `python/models_cache/` 目录即可跳过 S3 下载。

## 技术栈

**前端**
- Flutter 3.x (Web)
- Web Speech API (语音识别)
- Video Player

**后端 (Node.js)**
- Express.js
- Multer (文件上传)
- WebSocket (进度推送)
- Axios (与 Python 通信)

**后端 (Python)**
- Flask
- ONNX Runtime
- OpenCV (视频帧读取)
- FAISS (向量检索)
- Sentence-Transformers (文本嵌入)
- Whisper (语音识别)
- Boto3 (S3 客户端)

## License

MIT
