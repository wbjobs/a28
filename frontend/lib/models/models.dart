class VideoInfo {
  final String id;
  final String originalName;
  final String filename;
  final String? path;
  final int size;
  final String mimetype;
  final String uploadedAt;
  final String status;
  final String url;
  final bool processed;

  VideoInfo({
    required this.id,
    required this.originalName,
    required this.filename,
    this.path,
    required this.size,
    required this.mimetype,
    required this.uploadedAt,
    required this.status,
    required this.url,
    required this.processed,
  });

  factory VideoInfo.fromJson(Map<String, dynamic> json) {
    return VideoInfo(
      id: json['id'] ?? '',
      originalName: json['original_name'] ?? '',
      filename: json['filename'] ?? '',
      path: json['path'],
      size: json['size'] ?? 0,
      mimetype: json['mimetype'] ?? '',
      uploadedAt: json['uploaded_at'] ?? '',
      status: json['status'] ?? 'uploaded',
      url: json['url'] ?? '',
      processed: json['processed'] ?? false,
    );
  }

  String get sizeFormatted {
    if (size < 1024) return '$size B';
    if (size < 1024 * 1024) return '${(size / 1024).toStringAsFixed(1)} KB';
    if (size < 1024 * 1024 * 1024) {
      return '${(size / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
    return '${(size / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }
}

class ProcessTask {
  final String taskId;
  final String videoId;
  final String status;

  ProcessTask({
    required this.taskId,
    required this.videoId,
    required this.status,
  });

  factory ProcessTask.fromJson(Map<String, dynamic> json) {
    return ProcessTask(
      taskId: json['task_id'] ?? '',
      videoId: json['video_id'] ?? '',
      status: json['status'] ?? 'processing',
    );
  }
}

class TaskStatus {
  final String taskId;
  final String status;
  final double progress;
  final String message;
  final String videoId;
  final VideoInfo? videoInfo;
  final Map<String, dynamic>? result;

  TaskStatus({
    required this.taskId,
    required this.status,
    required this.progress,
    required this.message,
    required this.videoId,
    this.videoInfo,
    this.result,
  });

  factory TaskStatus.fromJson(Map<String, dynamic> json) {
    return TaskStatus(
      taskId: json['task_id'] ?? '',
      status: json['status'] ?? 'unknown',
      progress: (json['progress'] as num?)?.toDouble() ?? 0.0,
      message: json['message'] ?? '',
      videoId: json['video_id'] ?? '',
      videoInfo: json['video_info'] != null
          ? VideoInfo.fromJson(json['video_info'])
          : null,
      result: json['result'],
    );
  }

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get isProcessing => status == 'processing';
}

class DetectionObject {
  final String objClass;
  final List<double> bbox;
  final double confidence;

  DetectionObject({
    required this.objClass,
    required this.bbox,
    required this.confidence,
  });

  factory DetectionObject.fromJson(Map<String, dynamic> json) {
    return DetectionObject(
      objClass: json['class'] ?? '',
      bbox: (json['bbox'] as List?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class Classification {
  final String label;
  final double confidence;

  Classification({
    required this.label,
    required this.confidence,
  });

  factory Classification.fromJson(Map<String, dynamic> json) {
    return Classification(
      label: json['label'] ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class TimelineSegment {
  final double timestamp;
  final double duration;
  final String scene;
  final List<DetectionObject> objects;
  final List<Classification> classifications;
  final String speech;
  bool isHighlighted;

  TimelineSegment({
    required this.timestamp,
    required this.duration,
    required this.scene,
    required this.objects,
    required this.classifications,
    required this.speech,
    this.isHighlighted = false,
  });

  factory TimelineSegment.fromJson(Map<String, dynamic> json) {
    return TimelineSegment(
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
      duration: (json['duration'] as num?)?.toDouble() ?? 0.0,
      scene: json['scene'] ?? 'unknown',
      objects: (json['objects'] as List?)
              ?.map((o) => DetectionObject.fromJson(o))
              .toList() ??
          [],
      classifications: (json['classifications'] as List?)
              ?.map((c) => Classification.fromJson(c))
              .toList() ??
          [],
      speech: json['speech'] ?? '',
    );
  }

  String get timestampFormatted {
    final total = timestamp.toInt();
    final minutes = total ~/ 60;
    final seconds = total % 60;
    final ms = ((timestamp - total) * 100).toInt();
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}.${ms.toString().padLeft(2, '0')}';
  }

  String get objectNames => objects.map((o) => o.objClass).toSet().join(', ');
}

class VideoResult {
  final String videoId;
  final String videoPath;
  final double totalDuration;
  final int segmentCount;
  final List<TimelineSegment> timeline;
  final VideoInfo? videoInfo;

  VideoResult({
    required this.videoId,
    required this.videoPath,
    required this.totalDuration,
    required this.segmentCount,
    required this.timeline,
    this.videoInfo,
  });

  factory VideoResult.fromJson(Map<String, dynamic> json) {
    return VideoResult(
      videoId: json['video_id'] ?? '',
      videoPath: json['video_path'] ?? '',
      totalDuration: (json['total_duration'] as num?)?.toDouble() ?? 0.0,
      segmentCount: json['segment_count'] ?? 0,
      timeline: (json['timeline'] as List?)
              ?.map((s) => TimelineSegment.fromJson(s))
              .toList() ??
          [],
      videoInfo: json['video_info'] != null
          ? VideoInfo.fromJson(json['video_info'])
          : null,
    );
  }
}

class SearchMatch {
  final String segmentId;
  final int segmentIndex;
  final double timestamp;
  final double duration;
  final String scene;
  final List<DetectionObject> objects;
  final String speech;
  final double semanticScore;
  final double hybridScore;
  final Map<String, dynamic> matchedFilters;

  SearchMatch({
    required this.segmentId,
    required this.segmentIndex,
    required this.timestamp,
    required this.duration,
    required this.scene,
    required this.objects,
    required this.speech,
    required this.semanticScore,
    required this.hybridScore,
    required this.matchedFilters,
  });

  factory SearchMatch.fromJson(Map<String, dynamic> json) {
    return SearchMatch(
      segmentId: json['segment_id'] ?? '',
      segmentIndex: json['segment_index'] ?? 0,
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
      duration: (json['duration'] as num?)?.toDouble() ?? 0.0,
      scene: json['scene'] ?? '',
      objects: (json['objects'] as List?)
              ?.map((o) => DetectionObject.fromJson(o))
              .toList() ??
          [],
      speech: json['speech'] ?? '',
      semanticScore: (json['semantic_score'] as num?)?.toDouble() ?? 0.0,
      hybridScore: (json['hybrid_score'] as num?)?.toDouble() ?? 0.0,
      matchedFilters: json['matched_filters'] as Map<String, dynamic>? ?? {},
    );
  }

  String get timestampFormatted {
    final total = timestamp.toInt();
    final minutes = total ~/ 60;
    final seconds = total % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
}

class SearchResult {
  final String videoId;
  final String query;
  final int resultCount;
  final List<SearchMatch> results;
  final String? videoUrl;

  SearchResult({
    required this.videoId,
    required this.query,
    required this.resultCount,
    required this.results,
    this.videoUrl,
  });

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      videoId: json['video_id'] ?? '',
      query: json['query'] ?? '',
      resultCount: json['result_count'] ?? 0,
      results: (json['results'] as List?)
              ?.map((r) => SearchMatch.fromJson(r))
              .toList() ??
          [],
      videoUrl: json['video_url'],
    );
  }
}
