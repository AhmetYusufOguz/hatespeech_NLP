import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:safe_chat_app/services/auth_service.dart';
import 'package:safe_chat_app/views/signup_page.dart';
import 'package:safe_chat_app/views/main_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final ValueNotifier<bool> emailError = ValueNotifier(false);
  final ValueNotifier<bool> passwordError = ValueNotifier(false);
  bool isLoading = false;

  Future<bool> login(BuildContext context) async {
    final AuthService authService = AuthService();
    try {
      await authService.signInWithEmailPassword(
        emailController.text.trim(),
        passwordController.text,
      );
      return true;
    } catch (e) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Login Failed'),
          content: Text(e.toString()),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        ),
      );
      return false;
    }
  }

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    emailError.dispose();
    passwordError.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[300],
      body: SafeArea(
        child: Stack(
          children: [
            SingleChildScrollView(
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  minHeight:
                      MediaQuery.of(context).size.height -
                      MediaQuery.of(context).padding.top,
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const SizedBox(height: 18),
                    Text(
                      'SafeChat',
                      style: TextStyle(
                        color: Colors.lightBlue[400],
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),

                    const SizedBox(height: 12),

                    // Center card
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
                              'Welcome to\nSafeChat login now!',
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 18),

                            // Email field
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
                                            width: 1.5,
                                          )
                                        : null,
                                  ),
                                  child: TextField(
                                    controller: emailController,
                                    inputFormatters: [
                                      FilteringTextInputFormatter.deny(
                                        RegExp(r'\\s'),
                                      ),
                                    ],
                                    decoration: InputDecoration(
                                      hintText: 'Email',
                                      filled: true,
                                      fillColor: Colors.transparent,
                                      contentPadding:
                                          const EdgeInsets.symmetric(
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

                            // Password field
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
                                            width: 1.5,
                                          )
                                        : null,
                                  ),
                                  child: TextField(
                                    controller: passwordController,
                                    obscureText: true,
                                    inputFormatters: [
                                      FilteringTextInputFormatter.deny(
                                        RegExp(r'\\s'),
                                      ),
                                    ],
                                    decoration: InputDecoration(
                                      hintText: 'Password',
                                      filled: true,
                                      fillColor: Colors.transparent,
                                      contentPadding:
                                          const EdgeInsets.symmetric(
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

                            // Login button
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
                                onPressed: isLoading
                                    ? null
                                    : () async {
                                        final email = emailController.text
                                            .trim();
                                        final password =
                                            passwordController.text;

                                        emailError.value = email.isEmpty;
                                        passwordError.value = password.isEmpty;

                                        if (emailError.value ||
                                            passwordError.value) {
                                          return;
                                        }

                                        setState(() {
                                          isLoading = true;
                                        });

                                        final success = await login(context);

                                        setState(() {
                                          isLoading = false;
                                        });

                                        if (success && mounted) {
                                          Navigator.pushReplacement(
                                            context,
                                            MaterialPageRoute(
                                              builder: (context) =>
                                                  const MainPage(),
                                            ),
                                          );
                                        }
                                      },
                                child: const Text(
                                  'Login',
                                  style: TextStyle(fontSize: 16),
                                ),
                              ),
                            ),

                            const SizedBox(height: 14),

                            // OR separator
                            Row(
                              children: const [
                                Expanded(
                                  child: Divider(
                                    color: Colors.black26,
                                    thickness: 1,
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.symmetric(
                                    horizontal: 8.0,
                                  ),
                                  child: Text(
                                    'OR',
                                    style: TextStyle(color: Colors.black45),
                                  ),
                                ),
                                Expanded(
                                  child: Divider(
                                    color: Colors.black26,
                                    thickness: 1,
                                  ),
                                ),
                              ],
                            ),

                            const SizedBox(height: 14),

                            // Sign up button
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
                                onPressed: () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => const SignupPage(),
                                    ),
                                  );
                                },
                                child: const Text(
                                  'Sign Up',
                                  style: TextStyle(fontSize: 16),
                                ),
                              ),
                            ),

                            const SizedBox(height: 6),
                          ],
                        ),
                      ),
                    ),

                    // keep spacing at bottom
                    const SizedBox(height: 8),
                  ],
                ),
              ),
            ),
            if (isLoading)
              Container(
                color: Colors.black.withOpacity(0.5),
                child: const Center(child: CircularProgressIndicator()),
              ),
          ],
        ),
      ),
    );
  }
}
