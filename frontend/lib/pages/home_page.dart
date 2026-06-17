import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:universal_html/html.dart' as html;
import 'package:file_picker/file_picker.dart';

import '../services/api_service.dart';
import '../models/models.dart';
import '../widgets/timeline_view.dart';
import '../widgets/search_bar.dart';
import '../widgets/search_results.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  List<VideoInfo> _videos = [];
  VideoResult? _currentResult;
  SearchResult? _searchResult;
  Set<int> _highlightedIndices = {};
  String? _currentVideoUrl;
  bool _isUploading = false;
  bool _isProcessing = false;
  bool _isSearching = false;
  double _processProgress = 0.0;
  String _processMessage = '';
  String? _processingTaskId;

  @override
  void initState() {
    super.initState();
    _loadVideos();
  }

  ApiService get _api => context.read<ApiService>();

  Future<void> _loadVideos() async {
    try {
      final videos = await _api.listVideos();
      if (mounted) setState(() => _videos = videos);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('加载视频列表失败: $e')),
        );
      }
    }
  }

  Future<void> _pickAndUploadVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
      withData: true,
    );

    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    if (file.bytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('读取文件失败')),
      );
      return;
    }

    setState(() => _isUploading = true);

    try {
      final video = await _api.uploadVideo(file.bytes!, file.name);
      await _loadVideos();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('上传成功: ${video.originalName}')),
        );
        _startProcessing(video.id);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('上传失败: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  Future<void> _startProcessing(String videoId) async {
    setState(() {
      _isProcessing = true;
      _processProgress = 0.0;
      _processMessage = '准备处理...';
    });

    try {
      final task = await _api.startProcessing(videoId);
      _processingTaskId = task.taskId;
      _pollTaskStatus(task.taskId, videoId);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('启动处理失败: $e')),
        );
        setState(() => _isProcessing = false);
      }
    }
  }

  Future<void> _pollTaskStatus(String taskId, String videoId) async {
    int attempts = 0;
    const maxAttempts = 600;

    while (attempts < maxAttempts && mounted) {
      try {
        final status = await _api.getTaskStatus(taskId);
        if (mounted) {
          setState(() {
            _processProgress = status.progress;
            _processMessage = status.message;
          });
        }

        if (status.isCompleted) {
          if (mounted) {
            await _loadVideoResult(videoId);
            setState(() {
              _isProcessing = false;
              _processingTaskId = null;
            });
          }
          return;
        }

        if (status.isFailed) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('处理失败: ${status.message}')),
            );
            setState(() {
              _isProcessing = false;
              _processingTaskId = null;
            });
          }
          return;
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _processMessage = '状态检查中... ($e)';
          });
        }
      }

      attempts++;
      await Future.delayed(const Duration(seconds: 1));
    }

    if (mounted) {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _loadVideoResult(String videoId) async {
    try {
      final result = await _api.getVideoResult(videoId);
      if (mounted) {
        setState(() {
          _currentResult = result;
          _currentVideoUrl = result.videoInfo?.url ?? _api.baseUrl + result.videoPath;
          _searchResult = null;
          _highlightedIndices = {};
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('加载结果失败: $e')),
        );
      }
    }
  }

  Future<void> _performSearch(String query) async {
    if (query.trim().isEmpty || _currentResult == null) return;

    setState(() {
      _isSearching = true;
      _highlightedIndices = {};
    });

    try {
      final result = await _api.searchVideo(
        _currentResult!.videoId,
        query,
        topK: 15,
        threshold: 0.25,
      );
      if (mounted) {
        setState(() {
          _searchResult = result;
          _highlightedIndices = result.results
              .map((r) => r.segmentIndex)
              .toSet();
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('搜索失败: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isSearching = false);
    }
  }

  void _onSearchResultTap(int segmentIndex, double timestamp) {
    if (mounted) {
      setState(() {
        _highlightedIndices = {segmentIndex};
      });
    }

    final videoElements = html.document.querySelectorAll('video');
    if (videoElements.isNotEmpty) {
      final video = videoElements.first as html.VideoElement;
      try {
        video.currentTime = timestamp;
        video.play();
      } catch (_) {}
    }
  }

  void _onTimelineSeek(double timestamp) {
    final videoElements = html.document.querySelectorAll('video');
    if (videoElements.isNotEmpty) {
      final video = videoElements.first as html.VideoElement;
      try {
        video.currentTime = timestamp;
      } catch (_) {}
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          children: [
            Icon(Icons.smart_toy_outlined),
            SizedBox(width: 12),
            Text('视频理解分析平台'),
          ],
        ),
        actions: [
          if (_isProcessing)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Center(
                child: Row(
                  children: [
                    SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                        '${(_processProgress * 100).toStringAsFixed(0)}% - $_processMessage'),
                  ],
                ),
              ),
            ),
        ],
      ),
      body: Row(
        children: [
          _buildVideoListPanel(),
          Expanded(
            flex: 2,
            child: _buildMainContent(),
          ),
          Expanded(
            flex: 2,
            child: _buildRightPanel(),
          ),
        ],
      ),
    );
  }

  Widget _buildVideoListPanel() {
    return Container(
      width: 320,
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        border: Border(
          right: BorderSide(color: Theme.of(context).dividerColor),
        ),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _isUploading ? null : _pickAndUploadVideo,
                icon: _isUploading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child:
                            CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.upload_file),
                label: Text(_isUploading ? '上传中...' : '上传视频'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                '视频库 (${_videos.length})',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
            ),
          ),
          Expanded(
            child: _videos.isEmpty
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.video_library,
                              size: 48, color: Colors.grey[400]),
                          const SizedBox(height: 12),
                          Text(
                            '暂无视频',
                            style: TextStyle(color: Colors.grey[600]),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '点击上方按钮上传视频',
                            style: TextStyle(
                                color: Colors.grey[500], fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    itemCount: _videos.length,
                    itemBuilder: (context, index) {
                      final video = _videos[index];
                      final isSelected =
                          _currentResult?.videoId == video.id;
                      return Card(
                        margin: const EdgeInsets.symmetric(vertical: 4),
                        elevation: isSelected ? 2 : 0,
                        color: isSelected
                            ? Theme.of(context)
                                .colorScheme
                                .primary
                                .withOpacity(0.08)
                            : null,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                          side: BorderSide(
                            color: isSelected
                                ? Theme.of(context).colorScheme.primary
                                : Colors.transparent,
                            width: isSelected ? 2 : 0,
                          ),
                        ),
                        child: InkWell(
                          borderRadius: BorderRadius.circular(10),
                          onTap: () {
                            if (video.processed) {
                              _loadVideoResult(video.id);
                            } else {
                              _startProcessing(video.id);
                            }
                          },
                          child: Padding(
                            padding: const EdgeInsets.all(12.0),
                            child: Column(
                              crossAxisAlignment:
                                  CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(Icons.video_collection,
                                        size: 20,
                                        color: video.processed
                                            ? Colors.green
                                            : Colors.orange),
                                    const SizedBox(width: 8),
                                    Expanded(
                                      child: Text(
                                        video.originalName,
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                        style: const TextStyle(
                                            fontWeight: FontWeight.w600,
                                            fontSize: 13),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Row(
                                  children: [
                                    Text(
                                      video.sizeFormatted,
                                      style: TextStyle(
                                          fontSize: 11,
                                          color: Colors.grey[600]),
                                    ),
                                    const Spacer(),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 8, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: video.processed
                                            ? Colors.green.shade100
                                            : Colors.orange.shade100,
                                        borderRadius:
                                            BorderRadius.circular(10),
                                      ),
                                      child: Text(
                                        video.processed ? '已分析' : '待分析',
                                        style: TextStyle(
                                          fontSize: 10,
                                          fontWeight: FontWeight.w600,
                                          color: video.processed
                                              ? Colors.green.shade800
                                              : Colors.orange.shade800,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent() {
    return Container(
      color: Theme.of(context).scaffoldBackgroundColor,
      child: Column(
        children: [
          if (_currentVideoUrl != null)
            Container(
              width: double.infinity,
              color: Colors.black,
              height: 280,
              child: HtmlElementView(
                viewType: 'video-player-${_currentResult?.videoId}',
                onPlatformViewCreated: (_) {
                  final div = html.DivElement()
                    ..id = 'video-container-${_currentResult?.videoId}'
                    ..style.width = '100%'
                    ..style.height = '100%'
                    ..style.display = 'flex'
                    ..style.alignItems = 'center'
                    ..style.justifyContent = 'center';

                  final video = html.VideoElement()
                    ..src = _currentVideoUrl!
                    ..controls = true
                    ..style.maxWidth = '100%'
                    ..style.maxHeight = '100%';

                  div.append(video);

                  html.document
                      .getElementById(
                          'video-container-${_currentResult?.videoId}')
                      ?.replaceWith(div);
                },
              ),
            ),
          if (_currentVideoUrl == null && _currentResult == null)
            Expanded(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.ondemand_video,
                      size: 96,
                      color: Colors.grey[300],
                    ),
                    const SizedBox(height: 24),
                    Text(
                      '选择或上传一个视频开始分析',
                      style: TextStyle(
                          fontSize: 18, color: Colors.grey[600]),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '系统将自动识别场景、物体和语音对话',
                      style: TextStyle(
                          fontSize: 14, color: Colors.grey[500]),
                    ),
                  ],
                ),
              ),
            ),
          if (_currentResult != null)
            Expanded(
              child: TimelineView(
                segments: _currentResult!.timeline,
                totalDuration: _currentResult!.totalDuration,
                highlightedIndices: _highlightedIndices,
                onSeek: _onTimelineSeek,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildRightPanel() {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        border: Border(
          left: BorderSide(color: Theme.of(context).dividerColor),
        ),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: SearchBarWidget(
              onSearch: _performSearch,
              isLoading: _isSearching,
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: SearchResultsPanel(
              searchResult: _searchResult,
              isLoading: _isSearching,
              onResultTap: _onSearchResultTap,
            ),
          ),
        ],
      ),
    );
  }
}
