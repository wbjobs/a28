require('dotenv').config();

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.NODE_PORT || 3000;
const PYTHON_URL = `http://localhost:${process.env.PYTHON_PORT || 5000}`;

app.use(cors());
app.use(express.json());

const UPLOAD_DIR = path.join(__dirname, '..', 'uploads');
const RESULTS_DIR = path.join(__dirname, '..', 'results');
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend', 'build', 'web');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${uuidv4()}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/') || file.originalname.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('只支持视频文件格式: mp4, avi, mov, mkv, webm'));
    }
  }
});

const activeTasks = new Map();

app.use('/uploads', express.static(UPLOAD_DIR));

if (fs.existsSync(FRONTEND_DIR)) {
  app.use(express.static(FRONTEND_DIR));
}

app.get('/api/health', async (req, res) => {
  let pythonStatus = 'unreachable';
  try {
    const pyRes = await axios.get(`${PYTHON_URL}/api/health`, { timeout: 3000 });
    pythonStatus = pyRes.data.status || 'ok';
  } catch (e) {
    pythonStatus = 'unreachable';
  }

  res.json({
    status: 'ok',
    service: 'video-understanding-node',
    python_service: pythonStatus,
    uploads_dir: UPLOAD_DIR
  });
});

app.post('/api/upload', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: '未接收到视频文件' });
    }

    const videoId = uuidv4();
    const fileInfo = {
      id: videoId,
      original_name: req.file.originalname,
      filename: req.file.filename,
      path: req.file.path,
      size: req.file.size,
      mimetype: req.file.mimetype,
      uploaded_at: new Date().toISOString(),
      status: 'uploaded'
    };

    const metadataFile = path.join(RESULTS_DIR, `${videoId}_meta.json`);
    fs.writeFileSync(metadataFile, JSON.stringify(fileInfo, null, 2));

    res.json({
      message: '视频上传成功',
      video: fileInfo
    });
  } catch (err) {
    console.error('[UPLOAD ERROR]', err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/process', async (req, res) => {
  try {
    const { video_id, video_path } = req.body;

    if (!video_id) {
      return res.status(400).json({ error: 'video_id 必填' });
    }

    let resolvedPath = video_path;
    if (!resolvedPath) {
      const metadataFile = path.join(RESULTS_DIR, `${video_id}_meta.json`);
      if (!fs.existsSync(metadataFile)) {
        return res.status(404).json({ error: '未找到该视频的上传记录' });
      }
      const meta = JSON.parse(fs.readFileSync(metadataFile, 'utf-8'));
      resolvedPath = meta.path;
    }

    if (!fs.existsSync(resolvedPath)) {
      return res.status(400).json({ error: '视频文件不存在' });
    }

    const pyRes = await axios.post(`${PYTHON_URL}/api/process`, {
      video_id,
      video_path: resolvedPath
    });

    const { task_id } = pyRes.data;
    activeTasks.set(task_id, {
      video_id,
      task_id,
      started_at: new Date().toISOString()
    });

    res.json(pyRes.data);
  } catch (err) {
    console.error('[PROCESS ERROR]', err.response ? err.response.data : err.message);
    res.status(500).json({
      error: err.response ? err.response.data.error : err.message
    });
  }
});

app.get('/api/tasks/:task_id', async (req, res) => {
  try {
    const pyRes = await axios.get(`${PYTHON_URL}/api/tasks/${req.params.task_id}`);

    if (pyRes.data.status === 'completed') {
      const taskInfo = activeTasks.get(req.params.task_id);
      if (taskInfo) {
        const videoId = taskInfo.video_id;
        const metadataFile = path.join(RESULTS_DIR, `${videoId}_meta.json`);
        if (fs.existsSync(metadataFile)) {
          const meta = JSON.parse(fs.readFileSync(metadataFile, 'utf-8'));
          pyRes.data.video_info = {
            original_name: meta.original_name,
            filename: meta.filename,
            url: `/uploads/${meta.filename}`
          };
        }
      }
    }

    res.json(pyRes.data);
  } catch (err) {
    console.error('[TASK STATUS ERROR]', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/videos/:video_id', async (req, res) => {
  try {
    const pyRes = await axios.get(`${PYTHON_URL}/api/videos/${req.params.video_id}`);

    const metadataFile = path.join(RESULTS_DIR, `${req.params.video_id}_meta.json`);
    if (fs.existsSync(metadataFile)) {
      const meta = JSON.parse(fs.readFileSync(metadataFile, 'utf-8'));
      pyRes.data.video_info = {
        original_name: meta.original_name,
        filename: meta.filename,
        url: `/uploads/${meta.filename}`
      };
    }

    res.json(pyRes.data);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      res.status(404).json({ error: '视频未找到' });
    } else {
      res.status(500).json({ error: err.message });
    }
  }
});

app.get('/api/subtitles/:video_id', async (req, res) => {
  try {
    const fmt = req.query.format || 'srt';
    const pyRes = await axios.get(
      `${PYTHON_URL}/api/subtitles/${req.params.video_id}?format=${fmt}`,
      { responseType: fmt === 'json' ? 'json' : 'stream' }
    );

    if (fmt === 'json') {
      res.json(pyRes.data);
    } else {
      const ext = fmt === 'vtt' ? 'vtt' : 'srt';
      res.setHeader('Content-Type', fmt === 'vtt' ? 'text/vtt' : 'text/plain');
      res.setHeader('Content-Disposition', `attachment; filename="${req.params.video_id}.${ext}"`);
      pyRes.data.pipe(res);
    }
  } catch (err) {
    if (err.response && err.response.status === 404) {
      res.status(404).json({ error: '字幕文件未找到' });
    } else {
      console.error('[SUBTITLES ERROR]', err.message);
      res.status(500).json({ error: err.message });
    }
  }
});

app.post('/api/correct', async (req, res) => {
  try {
    const pyRes = await axios.post(`${PYTHON_URL}/api/correct`, req.body);
    res.json(pyRes.data);
  } catch (err) {
    console.error('[CORRECT ERROR]', err.response ? err.response.data : err.message);
    res.status(500).json({
      error: err.response ? err.response.data.error : err.message
    });
  }
});

app.post('/api/search', async (req, res) => {
  try {
    const pyRes = await axios.post(`${PYTHON_URL}/api/search`, req.body);

    const videoId = req.body.video_id;
    const metadataFile = path.join(RESULTS_DIR, `${videoId}_meta.json`);
    if (fs.existsSync(metadataFile)) {
      const meta = JSON.parse(fs.readFileSync(metadataFile, 'utf-8'));
      pyRes.data.video_url = `/uploads/${meta.filename}`;
    }

    res.json(pyRes.data);
  } catch (err) {
    console.error('[SEARCH ERROR]', err.response ? err.response.data : err.message);
    res.status(500).json({
      error: err.response ? err.response.data.error : err.message
    });
  }
});

app.get('/api/videos', (req, res) => {
  try {
    const videos = [];
    const files = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('_meta.json'));

    for (const file of files) {
      const metaPath = path.join(RESULTS_DIR, file);
      const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
      const videoId = file.replace('_meta.json', '');
      const resultPath = path.join(RESULTS_DIR, `${videoId}.json`);

      videos.push({
        ...meta,
        id: videoId,
        url: `/uploads/${meta.filename}`,
        processed: fs.existsSync(resultPath)
      });
    }

    res.json({ videos: videos.sort((a, b) => new Date(b.uploaded_at) - new Date(a.uploaded_at)) });
  } catch (err) {
    console.error('[LIST VIDEOS ERROR]', err.message);
    res.status(500).json({ error: err.message });
  }
});

if (fs.existsSync(FRONTEND_DIR)) {
  app.get('*', (req, res) => {
    res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
  });
}

const server = app.listen(PORT, () => {
  console.log(`[INFO] Node.js server running on http://localhost:${PORT}`);
  console.log(`[INFO] Python backend expected at ${PYTHON_URL}`);
});

const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws) => {
  console.log('[WS] Client connected');

  let pollInterval;

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message.toString());

      if (data.type === 'poll_task' && data.task_id) {
        pollInterval = setInterval(async () => {
          try {
            const pyRes = await axios.get(`${PYTHON_URL}/api/tasks/${data.task_id}`);
            ws.send(JSON.stringify({
              type: 'task_update',
              task_id: data.task_id,
              data: pyRes.data
            }));

            if (['completed', 'failed'].includes(pyRes.data.status)) {
              clearInterval(pollInterval);
            }
          } catch (err) {
            ws.send(JSON.stringify({
              type: 'error',
              message: err.message
            }));
          }
        }, 1000);
      }

      if (data.type === 'stop_poll') {
        if (pollInterval) clearInterval(pollInterval);
      }
    } catch (e) {
      console.error('[WS] Invalid message:', e.message);
    }
  });

  ws.on('close', () => {
    if (pollInterval) clearInterval(pollInterval);
    console.log('[WS] Client disconnected');
  });
});

module.exports = app;
