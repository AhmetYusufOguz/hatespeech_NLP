import 'package:flutter/material.dart';
import 'package:safe_chat_app/services/auth_service.dart';
import 'package:safe_chat_app/services/chat_service.dart';
import 'package:safe_chat_app/views/login_page.dart';

import 'chat_page.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  final AuthService _authService = AuthService();
  final ChatService _chatService = ChatService();

  void logout() {
    final authService = AuthService();
    authService.signOut();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlue,
        title: const Text('Chats', style: TextStyle(color: Colors.white)),
        actions: [
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            color: Colors.white,
            iconSize: 30.0,
            onPressed: () {
              Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (context) => const LoginPage()),
                (route) => false,
              );
            },
          ),
        ],
      ),
      body: StreamBuilder<List<Map<String, dynamic>>>(
        stream: _chatService.getUserStream(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: \\${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(child: Text('No users found.'));
          }
          final users = snapshot.data!;
          final currentUser = _authService.currentUser;
          final filteredUsers = users
              .where((user) => user['email'] != currentUser?.email)
              .toList();
          return ListView.builder(
            itemCount: filteredUsers.length,
            itemBuilder: (context, index) {
              final user = filteredUsers[index];
              return _buildChatItem(
                email: user['email'] ?? 'Unknown',
                message: user['lastMessage'] ?? '',
                time: user['lastActive'] ?? '',
                isOnline: user['isOnline'] ?? false,
                recieverId: user['uid'] ?? '',
              );
            },
          );
        },
      ),
    );
  }

  Widget _buildChatItem({
    required String email,
    required String message,
    required String time,
    required bool isOnline,
    required String recieverId,
  }) {
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) =>
                ChatPage(recieverEmail: email, recieverId: recieverId),
          ),
        );
      },
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: Colors.grey,
          child: Icon(Icons.person, color: Colors.white),
        ),
        title: Text(email),
        subtitle: Text(message),
        trailing: Text(time),
      ),
    );
  }
}
