import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/models.dart';

class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({String? baseUrl, http.Client? client})
      : baseUrl = baseUrl ?? 'http://localhost:3000',
        _client = client ?? http.Client();

  Future<List<VideoInfo>> listVideos() async {
    final response = await _client.get(Uri.parse('$baseUrl/api/videos'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return (data['videos'] as List)
          .map((v) => VideoInfo.fromJson(v))
          .toList();
    }
    throw Exception('加载视频列表失败: ${response.body}');
  }

  Future<VideoInfo> uploadVideo(List<int> bytes, String filename) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/upload'),
    );
    request.files.add(http.MultipartFile.fromBytes(
      'video',
      bytes,
      filename: filename,
    ));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return VideoInfo.fromJson(data['video']);
    }
    throw Exception('上传失败: ${response.body}');
  }

  Future<ProcessTask> startProcessing(String videoId, {String? videoPath}) async {
    final response = await _client.post(
      Uri.parse('$baseUrl/api/process'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'video_id': videoId, 'video_path': videoPath}),
    );
    if (response.statusCode == 200) {
      return ProcessTask.fromJson(jsonDecode(response.body));
    }
    throw Exception('启动处理失败: ${response.body}');
  }

  Future<TaskStatus> getTaskStatus(String taskId) async {
    final response = await _client.get(Uri.parse('$baseUrl/api/tasks/$taskId'));
    if (response.statusCode == 200) {
      return TaskStatus.fromJson(jsonDecode(response.body));
    }
    throw Exception('获取任务状态失败: ${response.body}');
  }

  Future<VideoResult> getVideoResult(String videoId) async {
    final response = await _client.get(Uri.parse('$baseUrl/api/videos/$videoId'));
    if (response.statusCode == 200) {
      return VideoResult.fromJson(jsonDecode(response.body));
    }
    throw Exception('获取视频结果失败: ${response.body}');
  }

  Future<SearchResult> searchVideo(String videoId, String query,
      {int topK = 10, double threshold = 0.3}) async {
    final response = await _client.post(
      Uri.parse('$baseUrl/api/search'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'video_id': videoId,
        'query': query,
        'top_k': topK,
        'threshold': threshold,
      }),
    );
    if (response.statusCode == 200) {
      return SearchResult.fromJson(jsonDecode(response.body));
    }
    throw Exception('搜索失败: ${response.body}');
  }
}
