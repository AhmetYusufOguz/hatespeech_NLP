import 'dart:convert';
import 'package:http/http.dart' as http;

class HateSpeechService {
  static const List<String> hosts = [
    'http://10.0.2.2:8000', // Android emulator -> host machine
    'http://192.168.1.73:8000', // LAN fallback (real device or matching subnet)
  ];

  Future<Map<String, dynamic>?> analyzeText(String text) async {
    final requestBody = {'text': text};

    for (final host in hosts) {
      try {
        final uri = Uri.parse('$host/predict');
        final response = await http
            .post(
              uri,
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
              },
              body: jsonEncode(requestBody),
            )
            .timeout(const Duration(seconds: 6));

        if (response.statusCode == 200) {
          final data = jsonDecode(utf8.decode(response.bodyBytes));
          return {
            'prediction': data['prediction'] ?? 'Hiçbiri',
            'prediction_id': data['prediction_id'] ?? 0,
            'confidence': data['confidence'] ?? 0.0,
            'is_harmful': data['is_harmful'] ?? false,
            'description': data['description'] ?? '',
          };
        }
      } catch (_) {
        // try next host
        continue;
      }
    }

    return null;
  }
}
