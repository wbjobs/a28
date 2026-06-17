import 'package:flutter/material.dart';
import '../models/models.dart';

class TimelineView extends StatelessWidget {
  final List<TimelineSegment> segments;
  final double totalDuration;
  final Set<int> highlightedIndices;
  final Function(double)? onSeek;

  const TimelineView({
    super.key,
    required this.segments,
    required this.totalDuration,
    this.highlightedIndices = const {},
    this.onSeek,
  });

  @override
  Widget build(BuildContext context) {
    if (segments.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32.0),
          child: Text(
            '暂无时间线数据',
            style: TextStyle(color: Colors.grey, fontSize: 16),
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '视频时间线',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                '共 ${segments.length} 个片段 · 总时长 ${_formatDuration(totalDuration)}',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.grey[600],
                    ),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemCount: segments.length,
            itemBuilder: (context, index) {
              final segment = segments[index];
              final isHighlighted = highlightedIndices.contains(index);
              return _SegmentCard(
                segment: segment,
                index: index,
                isHighlighted: isHighlighted,
                onSeek: onSeek,
              );
            },
          ),
        ),
      ],
    );
  }

  String _formatDuration(double seconds) {
    final total = seconds.toInt();
    final h = total ~/ 3600;
    final m = (total % 3600) ~/ 60;
    final s = total % 60;
    if (h > 0) {
      return '$h:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
    }
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }
}

class _SegmentCard extends StatelessWidget {
  final TimelineSegment segment;
  final int index;
  final bool isHighlighted;
  final Function(double)? onSeek;

  const _SegmentCard({
    required this.segment,
    required this.index,
    required this.isHighlighted,
    this.onSeek,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final sceneColor = _getSceneColor(segment.scene);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: isHighlighted ? Colors.yellow.shade50 : Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isHighlighted ? Colors.orange : Colors.grey.shade200,
          width: isHighlighted ? 2 : 1,
        ),
        boxShadow: isHighlighted
            ? [
                BoxShadow(
                  color: Colors.orange.withOpacity(0.2),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                )
              ]
            : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onSeek != null ? () => onSeek!(segment.timestamp) : null,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: sceneColor.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        segment.timestampFormatted,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: sceneColor,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: colorScheme.primary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        _sceneLabel(segment.scene),
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          color: colorScheme.primary,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    const Spacer(),
                    if (segment.duration > 0)
                      Text(
                        '+${segment.duration.toStringAsFixed(1)}s',
                        style: TextStyle(
                          color: Colors.grey[500],
                          fontSize: 12,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 10),
                if (segment.objects.isNotEmpty)
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: segment.objects
                        .map((obj) => Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 3),
                              decoration: BoxDecoration(
                                color: Colors.blue.shade50,
                                borderRadius: BorderRadius.circular(4),
                                border: Border.all(
                                    color: Colors.blue.shade200),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(Icons.inventory_2,
                                      size: 12, color: Colors.blue),
                                  const SizedBox(width: 4),
                                  Text(
                                    obj.objClass,
                                    style: const TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w500),
                                  ),
                                  const SizedBox(width: 4),
                                  Text(
                                    '${(obj.confidence * 100).toInt()}%',
                                    style: TextStyle(
                                        fontSize: 10,
                                        color: Colors.grey[600]),
                                  ),
                                ],
                              ),
                            ))
                        .toList(),
                  ),
                if (segment.objects.isNotEmpty)
                  const SizedBox(height: 10),
                if (segment.speech.isNotEmpty)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.green.shade50,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.green.shade200),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.record_voice_over,
                            size: 16, color: Colors.green),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            segment.speech,
                            style: const TextStyle(
                              fontSize: 13,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                if (segment.classifications.isNotEmpty &&
                    segment.objects.isEmpty)
                  Text(
                    '图像: ${segment.classifications.map((c) => c.label).take(3).join(', ')}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                      fontStyle: FontStyle.italic,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _getSceneColor(String scene) {
    switch (scene) {
      case 'outdoor_nature':
        return Colors.green;
      case 'city_urban':
      case 'road_traffic':
        return Colors.deepPurple;
      case 'indoor_room':
        return Colors.orange;
      case 'water_activity':
        return Colors.blue;
      case 'air_transport':
        return Colors.cyan;
      case 'sports':
        return Colors.red;
      case 'food_kitchen':
        return Colors.pink;
      default:
        return Colors.grey;
    }
  }

  String _sceneLabel(String scene) {
    const labels = {
      'outdoor_nature': '户外自然',
      'city_urban': '城市',
      'indoor_room': '室内',
      'road_traffic': '道路交通',
      'water_activity': '水上活动',
      'air_transport': '航空',
      'sports': '运动',
      'food_kitchen': '餐饮厨房',
      'general': '通用',
    };
    return labels[scene] ?? scene;
  }
}
