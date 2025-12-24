import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Future<UserCredential> signInWithEmailPassword(
    String email,
    String password,
  ) async {
    try {
      UserCredential userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      _firestore.collection('Users').doc(userCredential.user?.uid).set({
        'uid': userCredential.user?.uid,
        'email': userCredential.user?.email,
      });

      return userCredential;
    } on FirebaseAuthException catch (e) {
      throw Exception('Failed to sign in: ${e.code}');
    }
  }

  Future<UserCredential> signUpWithEmailPassword(
    String email,
    String password,
  ) async {
    try {
      UserCredential userCredential = await _auth
          .createUserWithEmailAndPassword(email: email, password: password);

      _firestore.collection('Users').doc(userCredential.user?.uid).set({
        'uid': userCredential.user?.uid,
        'email': userCredential.user?.email,
      });

      return userCredential;
    } on FirebaseAuthException catch (e) {
      throw Exception('Failed to sign up: ${e.code}');
    }
  }

  Future<void> signOut() async {
    await _auth.signOut();
  }

  User? get currentUser => _auth.currentUser;
}
