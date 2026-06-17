import 'package:flutter/material.dart';
import '../models/models.dart';

class SearchResultsPanel extends StatelessWidget {
  final SearchResult? searchResult;
  final Function(int, double)? onResultTap;
  final bool isLoading;

  const SearchResultsPanel({
    super.key,
    this.searchResult,
    this.onResultTap,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    if (isLoading) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 12),
              Text('正在进行向量检索...'),
            ],
          ),
        ),
      );
    }

    if (searchResult == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.search_off, size: 48, color: Colors.grey[400]),
              const SizedBox(height: 12),
              Text(
                '输入查询条件以搜索视频内容',
                style: TextStyle(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
      );
    }

    if (searchResult!.results.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.search_off, size: 48, color: Colors.orange),
              const SizedBox(height: 12),
              Text(
                '未找到匹配的片段',
                style: TextStyle(color: Colors.grey[700], fontSize: 16),
              ),
              const SizedBox(height: 4),
              Text(
                '查询: "${searchResult!.query}"',
                style: TextStyle(color: Colors.grey[500], fontSize: 12),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Row(
            children: [
              Icon(Icons.auto_awesome,
                  color: Theme.of(context).colorScheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '搜索结果',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.green.shade100,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${searchResult!.resultCount} 个匹配',
                  style: TextStyle(
                    color: Colors.green.shade800,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            '"${searchResult!.query}"',
            style: TextStyle(
              color: Colors.grey[600],
              fontStyle: FontStyle.italic,
            ),
          ),
        ),
        const SizedBox(height: 8),
        const Divider(height: 1),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemCount: searchResult!.results.length,
            itemBuilder: (context, index) {
              final result = searchResult!.results[index];
              return _ResultCard(
                result: result,
                rank: index + 1,
                onTap: onResultTap != null
                    ? () => onResultTap!(result.segmentIndex, result.timestamp)
                    : null,
              );
            },
          ),
        ),
      ],
    );
  }
}

class _ResultCard extends StatelessWidget {
  final SearchMatch result;
  final int rank;
  final VoidCallback? onTap;

  const _ResultCard({
    required this.result,
    required this.rank,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    CircleAvatar(
                      radius: 14,
                      backgroundColor: colorScheme.primary,
                      child: Text(
                        '$rank',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade100,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        result.timestampFormatted,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: colorScheme.primary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        _sceneLabel(result.scene),
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          color: colorScheme.primary,
                          fontSize: 12,
                        ),
                      ),
                    ),
                    const Spacer(),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.score, size: 14, color: Colors.green),
                            const SizedBox(width: 4),
                            Text(
                              '${(result.hybridScore * 100).toStringAsFixed(0)}%',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Colors.green,
                              ),
                            ),
                          ],
                        ),
                        Text(
                          '语义: ${(result.semanticScore * 100).toStringAsFixed(0)}%',
                          style: TextStyle(
                            fontSize: 10,
                            color: Colors.grey[500],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                if (result.objects.isNotEmpty)
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: [
                      ...result.objects.map((obj) => Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: Colors.blue.shade50,
                              borderRadius: BorderRadius.circular(4),
                              border:
                                  Border.all(color: Colors.blue.shade200),
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
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ],
                            ),
                          )),
                      ..._matchedFiltersChips(),
                    ],
                  ),
                if (result.objects.isNotEmpty)
                  const SizedBox(height: 10),
                if (result.speech.isNotEmpty)
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
                            result.speech,
                            style: const TextStyle(
                              fontSize: 13,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  List<Widget> _matchedFiltersChips() {
    final chips = <Widget>[];
    final filters = result.matchedFilters;

    if (filters['exact_phrases'] != null) {
      for (var phrase in filters['exact_phrases']) {
        chips.add(Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: Colors.purple.shade50,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: Colors.purple.shade200),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.quote, size: 12, color: Colors.purple),
              const SizedBox(width: 4),
              Text(
                '"$phrase"',
                style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: Colors.purple),
              ),
            ],
          ),
        ));
      }
    }

    if (filters['keywords'] != null) {
      for (var kw in filters['keywords']) {
        chips.add(Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: Colors.teal.shade50,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: Colors.teal.shade200),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.label, size: 12, color: Colors.teal),
              const SizedBox(width: 4),
              Text(
                kw,
                style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.teal),
              ),
            ],
          ),
        ));
      }
    }

    return chips;
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
    };
    return labels[scene] ?? scene;
  }
}
