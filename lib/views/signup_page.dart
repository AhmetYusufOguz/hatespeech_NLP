import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:safe_chat_app/services/auth_service.dart';

class SignupPage extends StatefulWidget {
  const SignupPage({super.key});

  @override
  State<SignupPage> createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final ValueNotifier<bool> emailError = ValueNotifier(false);
  final ValueNotifier<bool> passwordError = ValueNotifier(false);

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    emailError.dispose();
    passwordError.dispose();
    super.dispose();
  }

  void register(BuildContext context) async {
    final authService = AuthService();
    final email = emailController.text.trim();
    final password = passwordController.text.trim();

    bool hasError = false;
    if (email.isEmpty) {
      emailError.value = true;
      hasError = true;
    } else {
      emailError.value = false;
    }
    if (password.isEmpty) {
      passwordError.value = true;
      hasError = true;
    } else {
      passwordError.value = false;
    }
    if (hasError) return;

    try {
      await authService.signUpWithEmailPassword(email, password);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Registration successful! Please log in.'),
        ),
      );
      Navigator.of(context).pop(); // Route to login page
    } on Exception catch (e) {
      String errorMessage = e.toString();
      if (errorMessage.contains('email-already-in-use')) {
        errorMessage = 'User already exists with this email.';
      }
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Registration Failed'),
          content: Text(errorMessage),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[300],
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: IntrinsicHeight(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(height: 18),
                      Row(
                        children: [
                          IconButton(
                            icon: const Icon(Icons.arrow_back),
                            onPressed: () => Navigator.of(context).pop(),
                          ),
                          const Spacer(),
                        ],
                      ),
                      const Text(
                        'SafeChat',
                        style: TextStyle(
                          color: Colors.lightBlue,
                          fontSize: 36,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      Center(
                        child: Container(
                          width: MediaQuery.of(context).size.width * 0.86,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 18,
                            vertical: 22,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const SizedBox(height: 6),
                              const Text(
                                'Create an Account?',
                                style: TextStyle(
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 18),
                              ValueListenableBuilder<bool>(
                                valueListenable: emailError,
                                builder: (context, hasError, child) {
                                  return Container(
                                    decoration: BoxDecoration(
                                      color: Colors.grey[200],
                                      borderRadius: BorderRadius.circular(12),
                                      border: hasError
                                          ? Border.all(
                                              color: Colors.red,
                                              width: 2,
                                            )
                                          : null,
                                    ),
                                    child: TextField(
                                      controller: emailController,
                                      inputFormatters: [
                                        FilteringTextInputFormatter.deny(
                                          RegExp(r'\s'),
                                        ),
                                      ],
                                      decoration: const InputDecoration(
                                        hintText: 'Email',
                                        filled: true,
                                        fillColor: Colors.transparent,
                                        contentPadding: EdgeInsets.symmetric(
                                          horizontal: 14,
                                          vertical: 12,
                                        ),
                                        border: InputBorder.none,
                                      ),
                                    ),
                                  );
                                },
                              ),
                              const SizedBox(height: 12),
                              ValueListenableBuilder<bool>(
                                valueListenable: passwordError,
                                builder: (context, hasError, child) {
                                  return Container(
                                    decoration: BoxDecoration(
                                      color: Colors.grey[200],
                                      borderRadius: BorderRadius.circular(12),
                                      border: hasError
                                          ? Border.all(
                                              color: Colors.red,
                                              width: 2,
                                            )
                                          : null,
                                    ),
                                    child: TextField(
                                      controller: passwordController,
                                      obscureText: true,
                                      inputFormatters: [
                                        FilteringTextInputFormatter.deny(
                                          RegExp(r'\s'),
                                        ),
                                      ],
                                      decoration: const InputDecoration(
                                        hintText: 'Password',
                                        filled: true,
                                        fillColor: Colors.transparent,
                                        contentPadding: EdgeInsets.symmetric(
                                          horizontal: 14,
                                          vertical: 12,
                                        ),
                                        border: InputBorder.none,
                                      ),
                                    ),
                                  );
                                },
                              ),
                              const SizedBox(height: 16),
                              SizedBox(
                                width: double.infinity,
                                height: 46,
                                child: ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.lightBlue[400],
                                    foregroundColor: Colors.white,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    elevation: 0,
                                  ),
                                  onPressed: () => register(context),
                                  child: const Text(
                                    'Register',
                                    style: TextStyle(fontSize: 16),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 6),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Expanded(
                        child:
                            Container(), // Pushes content up when keyboard appears
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
