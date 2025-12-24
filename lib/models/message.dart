import 'package:cloud_firestore/cloud_firestore.dart';

class Message {
  final String senderId;
  final String senderEmail;
  final String receiverId;
  final String message;
  final Timestamp timestamp;
  final String? hateSpeechLabel;
  final int? hateSpeechPredictionId;
  final double? hateSpeechConfidence;
  final bool? isHarmful;
  final String? description;

  Message({
    required this.senderId,
    required this.senderEmail,
    required this.receiverId,
    required this.message,
    required this.timestamp,
    this.hateSpeechLabel,
    this.hateSpeechPredictionId,
    this.hateSpeechConfidence,
    this.isHarmful,
    this.description,
  });

  Map<String, dynamic> toMap() {
    return {
      'senderId': senderId,
      'senderEmail': senderEmail,
      'receiverId': receiverId,
      'message': message,
      'timestamp': timestamp,
      'hateSpeechLabel': hateSpeechLabel,
      'hateSpeechPredictionId': hateSpeechPredictionId,
      'hateSpeechConfidence': hateSpeechConfidence,
      'isHarmful': isHarmful,
      'description': description,
    };
  }
}
