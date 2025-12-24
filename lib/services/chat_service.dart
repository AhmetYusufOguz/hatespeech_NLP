import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:safe_chat_app/models/message.dart';
import 'package:safe_chat_app/services/hate_speech_service.dart';

class ChatService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final HateSpeechService _hateSpeechService = HateSpeechService();

  Stream<List<Map<String, dynamic>>> getUserStream() {
    return _firestore.collection("Users").snapshots().map((snapshot) {
      return snapshot.docs.map((doc) {
        final user = doc.data();
        return user;
      }).toList();
    });
  }

  Future<Map<String, dynamic>> sendMessage(
    String receiverId,
    String message,
  ) async {
    final String currentUserID = _auth.currentUser?.uid ?? '';
    final String currentUserEmail = _auth.currentUser?.email ?? '';
    final Timestamp timestamp = Timestamp.now();

    Map<String, dynamic>? analysis;
    try {
      analysis = await _hateSpeechService.analyzeText(message);
    } catch (_) {
      analysis = null;
    }

    Message newMessage = Message(
      senderId: currentUserID,
      senderEmail: currentUserEmail,
      receiverId: receiverId,
      message: message,
      timestamp: timestamp,
      hateSpeechLabel: analysis?['prediction'],
      hateSpeechPredictionId: analysis?['prediction_id'],
      hateSpeechConfidence: analysis?['confidence']?.toDouble(),
      isHarmful: analysis?['is_harmful'] ?? false,
      description: analysis?['description'],
    );

    List<String> ids = [currentUserID, receiverId];
    ids.sort();
    String chatRoomId = ids.join('_');

    await _firestore
        .collection('ChatRooms')
        .doc(chatRoomId)
        .collection('Messages')
        .add(newMessage.toMap());

    return {
      'isHarmful': newMessage.isHarmful ?? false,
      'label': newMessage.hateSpeechLabel ?? 'Hiçbiri',
      'description': newMessage.description ?? '',
    };
  }

  Stream<QuerySnapshot> getMessages(String userID, otherUserID) {
    List<String> id = [userID, otherUserID];
    id.sort();
    String chatRoomId = id.join('_');

    return _firestore
        .collection('ChatRooms')
        .doc(chatRoomId)
        .collection('Messages')
        .orderBy('timestamp', descending: false)
        .snapshots();
  }
}
