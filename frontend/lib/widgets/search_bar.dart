import 'package:flutter/material.dart';
import '../services/speech_recognition.dart';

class SearchBarWidget extends StatefulWidget {
  final Function(String) onSearch;
  final bool isLoading;

  const SearchBarWidget({
    super.key,
    required this.onSearch,
    this.isLoading = false,
  });

  @override
  State<SearchBarWidget> createState() => _SearchBarWidgetState();
}

class _SearchBarWidgetState extends State<SearchBarWidget> {
  final TextEditingController _controller = TextEditingController();
  late SpeechRecognitionService _speechService;
  bool _isListening = false;
  String _interimText = '';

  final List<String> _exampleQueries = [
    '找到有汽车的片段',
    '查找有人和"加速"这个词的场景',
    '在户外自然场景中的对话',
    '包含食物的室内画面',
  ];

  @override
  void initState() {
    super.initState();
    _speechService = SpeechRecognitionService();
    _speechService.onStatusChanged.listen((listening) {
      if (mounted) setState(() => _isListening = listening);
    });
    _speechService.onTextChanged.listen((text) {
      if (mounted) {
        setState(() => _interimText = text);
      }
    });
    _speechService.onFinalText.listen((text) {
      if (mounted) {
        setState(() {
          _controller.text = text;
          _interimText = '';
        });
        widget.onSearch(text);
        _speechService.stop();
      }
    });
    _speechService.onError.listen((error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('语音识别: $error')),
        );
      }
    });
  }

  @override
  void dispose() {
    _speechService.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final displayText =
        _interimText.isNotEmpty ? _interimText : _controller.text;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.search,
                  color: Theme.of(context).colorScheme.primary),
              const SizedBox(width: 8),
              Text(
                '跨模态搜索',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const Spacer(),
              if (_speechService.isAvailable)
                Tooltip(
                  message: '语音输入',
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      borderRadius: BorderRadius.circular(24),
                      onTap: _toggleListening,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: _isListening
                              ? Colors.red.withOpacity(0.1)
                              : Colors.transparent,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: _isListening ? Colors.red : Colors.grey.shade300,
                            width: 2,
                          ),
                        ),
                        child: Icon(
                          _isListening ? Icons.mic : Icons.mic_none,
                          color: _isListening ? Colors.red : Colors.grey[600],
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            decoration: BoxDecoration(
              color: Theme.of(context).scaffoldBackgroundColor,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _interimText.isNotEmpty
                    ? Colors.orange
                    : Colors.grey.shade300,
                width: 2,
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    child: TextField(
                      controller: _controller,
                      style: TextStyle(
                        color: _interimText.isNotEmpty
                            ? Colors.orange[800]
                            : null,
                      ),
                      decoration: InputDecoration(
                        hintText:
                            '例如：找到有汽车和"加速"这个词的片段...',
                        hintStyle: TextStyle(color: Colors.grey[400]),
                        border: InputBorder.none,
                      ),
                      onSubmitted: widget.onSearch,
                    ),
                  ),
                ),
                if (widget.isLoading)
                  const Padding(
                    padding: EdgeInsets.all(12),
                    child: SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                else
                  IconButton(
                    icon: Icon(Icons.arrow_forward,
                        color: Theme.of(context).colorScheme.primary),
                    onPressed: () => widget.onSearch(_controller.text),
                  ),
              ],
            ),
          ),
          if (_interimText.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 8, left: 4),
              child: Row(
                children: [
                  Icon(Icons.mic, size: 14, color: Colors.red),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      _interimText,
                      style: TextStyle(
                        color: Colors.orange[800],
                        fontStyle: FontStyle.italic,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _exampleQueries
                .map((q) => ActionChip(
                      label: Text(q),
                      onPressed: () {
                        _controller.text = q;
                        widget.onSearch(q);
                      },
                      backgroundColor: Colors.blue.shade50,
                      side: BorderSide(color: Colors.blue.shade200),
                      labelStyle: TextStyle(
                          color: Colors.blue[700], fontSize: 12),
                    ))
                .toList(),
          ),
        ],
      ),
    );
  }

  void _toggleListening() {
    if (_isListening) {
      _speechService.stop();
    } else {
      _controller.text = '';
      _speechService.start();
    }
  }
}
