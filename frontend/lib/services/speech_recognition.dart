import 'dart:async';
import 'dart:js_interop';
import 'package:web/web.dart' as web;

@JS('window.SpeechRecognition')
external dynamic get _speechRecognitionCtor;

@JS('window.webkitSpeechRecognition')
external dynamic get _webkitSpeechRecognitionCtor;

class SpeechRecognitionService {
  dynamic _recognition;
  bool _isListening = false;
  final StreamController<String> _textController = StreamController<String>.broadcast();
  final StreamController<String> _finalTextController = StreamController<String>.broadcast();
  final StreamController<String> _errorController = StreamController<String>.broadcast();
  final StreamController<bool> _statusController = StreamController<bool>.broadcast();

  bool get isAvailable =>
      _speechRecognitionCtor != null || _webkitSpeechRecognitionCtor != null;

  bool get isListening => _isListening;

  Stream<String> get onTextChanged => _textController.stream;

  Stream<String> get onFinalText => _finalTextController.stream;

  Stream<String> get onError => _errorController.stream;

  Stream<bool> get onStatusChanged => _statusController.stream;

  SpeechRecognitionService() {
    _init();
  }

  void _init() {
    if (!isAvailable) return;

    final ctor = _speechRecognitionCtor ?? _webkitSpeechRecognitionCtor;
    _recognition = _createRecognition(ctor);

    _recognition.continuous = true;
    _recognition.interimResults = true;
    _recognition.lang = 'zh-CN';

    _recognition.onresult = (event) {
      String interimText = '';
      String finalText = '';

      for (var i = event.resultIndex; i < event.results.length; i++) {
        final transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalText += transcript;
        } else {
          interimText += transcript;
        }
      }

      if (interimText.isNotEmpty) {
        _textController.add(interimText);
      }
      if (finalText.isNotEmpty) {
        _finalTextController.add(finalText.trim());
      }
    }.toJS;

    _recognition.onerror = (event) {
      _errorController.add(event.error?.toString() ?? '未知语音识别错误');
      _setStatus(false);
    }.toJS;

    _recognition.onend = () {
      if (_isListening) {
        try {
          _recognition.start();
        } catch (_) {
          _setStatus(false);
        }
      } else {
        _setStatus(false);
      }
    }.toJS;

    _recognition.onstart = () {
      _setStatus(true);
    }.toJS;
  }

  dynamic _createRecognition(dynamic ctor) {
    return ctor.newInstance();
  }

  void _setStatus(bool listening) {
    _isListening = listening;
    _statusController.add(listening);
  }

  void start() {
    if (!isAvailable) {
      _errorController.add('当前浏览器不支持语音识别功能，请使用 Chrome 或 Edge 浏览器');
      return;
    }
    if (_isListening) return;
    try {
      _recognition.start();
    } catch (e) {
      _errorController.add('启动语音识别失败: $e');
    }
  }

  void stop() {
    if (!isAvailable || !_isListening) return;
    _isListening = false;
    try {
      _recognition.stop();
    } catch (_) {}
    _setStatus(false);
  }

  void dispose() {
    stop();
    _textController.close();
    _finalTextController.close();
    _errorController.close();
    _statusController.close();
  }
}

@JS('SpeechRecognitionEvent')
@staticInterop
class _SpeechRecognitionEvent {}

extension on _SpeechRecognitionEvent {
  external int get resultIndex;
  external _SpeechRecognitionResultList get results;
  external dynamic get error;
}

@JS('SpeechRecognitionResultList')
@staticInterop
class _SpeechRecognitionResultList {}

extension on _SpeechRecognitionResultList {
  external int get length;
  external _SpeechRecognitionResult operator [](int index);
}

@JS('SpeechRecognitionResult')
@staticInterop
class _SpeechRecognitionResult {}

extension on _SpeechRecognitionResult {
  external bool get isFinal;
  external _SpeechRecognitionAlternativeList get alternatives;

  _SpeechRecognitionAlternative call(int index) => alternatives[0];
}

@JS()
@staticInterop
class _SpeechRecognitionAlternativeList {}

extension on _SpeechRecognitionAlternativeList {
  external int get length;
  external _SpeechRecognitionAlternative operator [](int index);
}

@JS()
@staticInterop
class _SpeechRecognitionAlternative {
  external String get transcript;
  external double get confidence;
}
