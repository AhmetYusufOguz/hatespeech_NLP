import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'package:safe_chat_app/services/auth_service.dart';
import 'package:safe_chat_app/services/chat_service.dart';

class ChatPage extends StatefulWidget {
  final String recieverEmail;
  final String recieverId;
  const ChatPage({
    super.key,
    required this.recieverEmail,
    required this.recieverId,
  });

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController _messageController = TextEditingController();
  final ChatService _chatService = ChatService();
  final AuthService _authService = AuthService();
  StreamSubscription? _messageSubscription;

  Duration duration = const Duration();
  Duration position = const Duration();
  bool isPlaying = false;
  bool isLoading = false;
  bool isPause = false;

  List<Map<String, dynamic>> _messages = [];

  void sendMessage() async {
    if (_messageController.text.isNotEmpty) {
      final messageText = _messageController.text.trim();
      try {
        final result = await _chatService.sendMessage(
          widget.recieverId,
          messageText,
        );
        _messageController.clear();

        if (result['isHarmful'] == true && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '⚠️ Uyarı: Mesajınız "${result['label']}" olarak işaretlendi.\n${result['description']}',
                style: const TextStyle(fontSize: 14),
              ),
              backgroundColor: Colors.orange.shade800,
              duration: const Duration(seconds: 5),
            ),
          );
        }
      } catch (e) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to send message: $e')));
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _fetchMessages();
  }

  @override
  void dispose() {
    _messageSubscription?.cancel();
    _messageController.dispose();
    super.dispose();
  }

  void _fetchMessages() {
    final currentUserId = _authService.currentUser?.uid;
    if (currentUserId == null) return;

    _messageSubscription = _chatService
        .getMessages(currentUserId, widget.recieverId)
        .listen((snapshot) {
          setState(() {
            _messages = snapshot.docs.map((doc) {
              final data = doc.data() as Map<String, dynamic>?;
              if (data == null) {
                return {
                  'text': '',
                  'isSender': false,
                  'isHarmful': false,
                  'label': '',
                };
              }

              final text = data['message'] ?? '';
              final senderId = data['senderId'] ?? '';
              final isHarmful = data['isHarmful'] ?? false;
              final label = data['hateSpeechLabel'] ?? '';

              return {
                'text': text,
                'isSender': senderId == currentUserId,
                'isHarmful': isHarmful,
                'label': label,
              };
            }).toList();
          });
        });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlue,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white, size: 30.0),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: Text(
          widget.recieverEmail,
          style: const TextStyle(color: Colors.white),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: <Widget>[
                  for (var message in _messages)
                    BubbleNormal(
                      text: message['text'],
                      isSender: message['isSender'],
                      isHarmful: message['isHarmful'] ?? false,
                      label: message['label'] ?? '',
                      color: message['isSender']
                          ? const Color(0xFFE8E8EE)
                          : const Color(0xFF1B97F3),
                      tail: true,
                      textStyle: TextStyle(
                        fontSize: 16,
                        color: message['isSender']
                            ? Colors.black
                            : Colors.white,
                      ),
                    ),
                  const SizedBox(height: 10),
                ],
              ),
            ),
          ),
          MessageBar(
            controller: _messageController,
            onSend: (message) => sendMessage(),
            actions: [],
          ),
        ],
      ),
    );
  }
}

/// Simple chat bubble implementation used by this page.
class BubbleNormal extends StatefulWidget {
  final String text;
  final bool isSender;
  final Color color;
  final bool tail;
  final TextStyle? textStyle;
  final bool isHarmful;
  final String label;

  const BubbleNormal({
    super.key,
    required this.text,
    required this.isSender,
    required this.color,
    this.tail = false,
    this.textStyle,
    this.isHarmful = false,
    this.label = '',
  });

  @override
  State<BubbleNormal> createState() => _BubbleNormalState();
}

class _BubbleNormalState extends State<BubbleNormal> {
  bool _revealed = false;

  Future<void> _handleTapIfBlurred(BuildContext context) async {
    if (!(widget.isHarmful && !widget.isSender) || _revealed) return;

    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Hassas içerik'),
        content: Text(
          'Bu mesaj "${widget.label}" olarak işaretlendi. Görüntülemek istediğinize emin misiniz?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Vazgeç'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Göster'),
          ),
        ],
      ),
    );

    if (confirm == true && mounted) {
      setState(() {
        _revealed = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final alignment = widget.isSender
        ? Alignment.centerRight
        : Alignment.centerLeft;
    final radius = Radius.circular(12);
    final shouldBlur = widget.isHarmful && !widget.isSender && !_revealed;
    return Align(
      alignment: alignment,
      child: GestureDetector(
        onTap: () => _handleTapIfBlurred(context),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.75,
          ),
          decoration: BoxDecoration(
            color: widget.color,
            borderRadius: BorderRadius.only(
              topLeft: radius,
              topRight: radius,
              bottomLeft: widget.isSender ? radius : Radius.zero,
              bottomRight: widget.isSender ? Radius.zero : radius,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (shouldBlur) ...[
                Stack(
                  children: [
                    ImageFiltered(
                      imageFilter: ui.ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                      child: Text(
                        widget.text,
                        style:
                            widget.textStyle ??
                            const TextStyle(color: Colors.white),
                      ),
                    ),
                    Text(
                      widget.text,
                      style:
                          (widget.textStyle ??
                                  const TextStyle(color: Colors.white))
                              .copyWith(color: Colors.transparent),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.red.shade700,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '⚠️ Hassas İçerik: ${widget.label}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ] else ...[
                Text(
                  widget.text,
                  style:
                      widget.textStyle ?? const TextStyle(color: Colors.white),
                ),
                if (widget.isHarmful &&
                    widget.isSender &&
                    widget.label.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.orange.shade700,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '⚠️ ${widget.label}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}

/// Simple date chip used in the conversation.
class DateChip extends StatelessWidget {
  final DateTime date;
  const DateChip({super.key, required this.date});

  String _formatDate(DateTime d) {
    return '${d.month}/${d.day}/${d.year}';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Center(
        child: Chip(
          backgroundColor: Colors.grey.shade300,
          label: Text(
            _formatDate(date),
            style: const TextStyle(color: Colors.black54),
          ),
        ),
      ),
    );
  }
}

/// Minimal message bar with actions and a send button.
class MessageBar extends StatefulWidget {
  final ValueChanged<String>? onSend;
  final List<Widget>? actions;
  final TextEditingController controller;

  const MessageBar({
    super.key,
    this.onSend,
    this.actions,
    required this.controller,
  });

  @override
  State<MessageBar> createState() => _MessageBarState();
}

class _MessageBarState extends State<MessageBar> {
  void _handleSend() {
    final text = widget.controller.text.trim();
    if (text.isEmpty) return;
    widget.onSend?.call(text);
    widget.controller.clear();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.bottomCenter,
      child: SafeArea(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          color: Colors.white,
          child: Row(
            children: [
              if (widget.actions != null) ...widget.actions!,
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: widget.controller,
                          decoration: const InputDecoration(
                            hintText: 'Type a message',
                            border: InputBorder.none,
                          ),
                          onSubmitted: (_) => _handleSend(),
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.send, color: Colors.blue),
                        onPressed: _handleSend,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
