# Safe Chat App 🛡️💬

A real-time chat application powered by AI and NLP to detect and warn users about hate speech and harmful content. Built with Flutter and Firebase, this app analyzes messages before sending them and provides instant feedback about potentially offensive content.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Firebase Setup](#firebase-setup)
- [Backend Setup](#backend-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Integration](#api-integration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Safe Chat App is a modern messaging application that goes beyond simple text communication. It integrates an AI-powered hate speech detection system that analyzes every message in real-time before it's sent. The app helps create a safer online communication environment by:

- Detecting potentially harmful or offensive content
- Providing warnings with detailed descriptions about flagged content
- Categorizing harmful content with confidence scores
- Allowing users to reconsider their messages before sending

## ✨ Features

### Core Functionality
- **Real-time Messaging**: Send and receive messages instantly using Firebase Firestore
- **User Authentication**: Secure email/password authentication with Firebase Auth
- **Contact List**: View all registered users and start conversations
- **Message History**: Persistent chat history for all conversations

### AI-Powered Safety
- **Hate Speech Detection**: Real-time NLP analysis of message content
- **Multi-category Classification**: Identifies various types of harmful content
- **Confidence Scoring**: Each prediction comes with a confidence level
- **User Warnings**: Visual alerts when harmful content is detected
- **Detailed Descriptions**: Explanations of why content was flagged

### User Experience
- **Modern UI**: Clean and intuitive interface with chat bubbles
- **Real-time Updates**: Live message synchronization across devices
- **User Status**: See who's registered in the app
- **Responsive Design**: Works across different screen sizes

## 🛠️ Technology Stack

### Frontend
- **Flutter** (SDK ^3.9.2) - Cross-platform mobile development framework
- **Dart** - Programming language
- **Material Design** - UI/UX framework

### Backend & Services
- **Firebase Authentication** - User authentication and management
- **Cloud Firestore** - NoSQL cloud database for real-time data
- **Firebase Core** - Core Firebase functionality

### AI/ML Backend
- **Python FastAPI** - Backend API server (separate repository)
- **NLP Model** - Hate speech classification model
- **HTTP** - REST API communication

### Key Dependencies
```yaml
- firebase_core: ^4.2.0
- firebase_auth: ^6.1.1
- cloud_firestore: ^6.0.3
- chat_bubbles: ^1.7.0
- http: ^1.2.0
- cupertino_icons: ^1.0.8
```

## 🏗️ Architecture

The application follows a clean architecture pattern with separation of concerns:

```
lib/
├── main.dart                 # Application entry point
├── firebase_options.dart     # Firebase configuration
├── models/
│   └── message.dart         # Message data model
├── services/
│   ├── auth_service.dart    # Authentication logic
│   ├── chat_service.dart    # Chat functionality
│   └── hate_speech_service.dart # AI integration
└── views/
    ├── login_page.dart      # Authentication UI
    ├── signup_page.dart     # Registration UI
    ├── main_page.dart       # User list/home
    └── chat_page.dart       # Chat interface
```

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Flutter SDK** (3.9.2 or higher)
  - Download from: https://flutter.dev/docs/get-started/install
  - Verify installation: `flutter --version`

- **Dart SDK** (comes with Flutter)

- **Android Studio** or **Xcode** (for mobile development)
  - Android Studio for Android development
  - Xcode for iOS development (macOS only)

- **Git** - Version control
  - Download from: https://git-scm.com/downloads

- **Firebase Account**
  - Create one at: https://firebase.google.com/

- **Python 3.8+** (for the hate speech detection backend)

- **Code Editor** (recommended: VS Code or Android Studio)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/safe_chat_app.git
cd safe_chat_app
```

### 2. Install Flutter Dependencies

```bash
flutter pub get
```

### 3. Verify Flutter Installation

```bash
flutter doctor
```

Fix any issues reported by Flutter Doctor before proceeding.

### 4. Check Available Devices

```bash
flutter devices
```

## 🔥 Firebase Setup

### 1. Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add Project"
3. Enter project name (e.g., "safe-chat-app")
4. Follow the setup wizard

### 2. Enable Authentication

1. In Firebase Console, go to **Authentication**
2. Click "Get Started"
3. Enable **Email/Password** sign-in method
4. Save changes

### 3. Create Firestore Database

1. Go to **Firestore Database**
2. Click "Create Database"
3. Start in **test mode** (for development)
4. Choose a location close to your users
5. Create the following collection structure:
   ```
   Users/
   └── {userId}
       ├── uid: string
       └── email: string
   
   chat_rooms/
   └── {chatRoomId}
       └── messages/
           └── {messageId}
               ├── senderId: string
               ├── senderEmail: string
               ├── receiverId: string
               ├── message: string
               ├── timestamp: timestamp
               ├── hateSpeechLabel: string (optional)
               ├── hateSpeechPredictionId: number (optional)
               ├── hateSpeechConfidence: number (optional)
               ├── isHarmful: boolean
               └── description: string (optional)
   ```

### 4. Add Firebase to Your Flutter App

#### For Android:

1. In Firebase Console, click "Add App" → Android icon
2. Register your app with package name: `com.example.safe_chat_app`
3. Download `google-services.json`
4. Place it in: `android/app/google-services.json`

#### For iOS:

1. Click "Add App" → iOS icon
2. Register your app with bundle ID
3. Download `GoogleService-Info.plist`
4. Add to Xcode project in `ios/Runner/`

### 5. Install Firebase CLI and FlutterFire

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Install FlutterFire CLI
dart pub global activate flutterfire_cli

# Configure Firebase for Flutter
flutterfire configure
```

This will automatically update your `firebase_options.dart` file.

## 🤖 Backend Setup

The hate speech detection requires a separate Python backend server.

### 1. Set Up Python Backend

Create a new directory for the backend (outside this project):

```bash
mkdir hate-speech-api
cd hate-speech-api
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn transformers torch scikit-learn
```

### 4. Create API Server

Create `main.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    # Implement your NLP model here
    # This is a placeholder response
    return {
        "prediction": "Hiçbiri",
        "prediction_id": 0,
        "confidence": 0.95,
        "is_harmful": False,
        "description": "No harmful content detected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5. Start the Backend Server

```bash
python main.py
```

The server should now be running at `http://localhost:8000`

### 6. Configure App to Connect to Backend

Update the host addresses in `lib/services/hate_speech_service.dart` if needed:

- `http://10.0.2.2:8000` - For Android Emulator
- `http://localhost:8000` - For iOS Simulator
- `http://YOUR_LOCAL_IP:8000` - For physical devices (replace YOUR_LOCAL_IP)

## ▶️ Running the Application

### 1. Start Backend Server (in separate terminal)

```bash
cd hate-speech-api
python main.py
```

### 2. Run Flutter App

#### On Android Emulator/Device:

```bash
flutter run
```

#### On iOS Simulator (macOS only):

```bash
flutter run -d ios
```

#### On Specific Device:

```bash
# List devices
flutter devices

# Run on specific device
flutter run -d <device-id>
```

### 3. Build Release Version

#### Android APK:

```bash
flutter build apk --release
```

Output: `build/app/outputs/flutter-apk/app-release.apk`

#### iOS (macOS only):

```bash
flutter build ios --release
```

## 📁 Project Structure

```
safe_chat_app/
├── android/                  # Android native code
├── ios/                      # iOS native code
├── lib/
│   ├── main.dart            # App entry point
│   ├── firebase_options.dart # Firebase config
│   ├── models/
│   │   └── message.dart     # Message model class
│   ├── services/
│   │   ├── auth_service.dart          # Firebase Auth wrapper
│   │   ├── chat_service.dart          # Firestore chat operations
│   │   └── hate_speech_service.dart   # NLP API client
│   └── views/
│       ├── login_page.dart            # Login screen
│       ├── signup_page.dart           # Registration screen
│       ├── main_page.dart             # User list/home screen
│       └── chat_page.dart             # Chat interface
├── web/                     # Web support files
├── windows/                 # Windows desktop support
├── linux/                   # Linux desktop support
├── macos/                   # macOS desktop support
├── pubspec.yaml            # Flutter dependencies
├── firebase.json           # Firebase configuration
└── README.md              # This file
```

## 🔍 How It Works

### Message Flow

1. **User composes a message** in the chat interface
2. **Message is sent** to the backend via `ChatService`
3. **Hate speech analysis** is performed by calling the NLP API
4. **Analysis results** are returned with:
   - Prediction category
   - Confidence score
   - Harmful flag
   - Description
5. **Message is stored** in Firestore with analysis metadata
6. **User receives warning** if content is harmful
7. **Real-time updates** push message to recipient

### Authentication Flow

1. User enters email and password
2. Firebase Authentication validates credentials
3. User document is created/updated in Firestore
4. User is redirected to main page
5. JWT token is managed automatically by Firebase

### Data Storage

Messages are stored in a hierarchical structure:
```
chat_rooms/{userId1_userId2}/messages/{messageId}
```

This ensures:
- Efficient querying
- Real-time synchronization
- Proper access control
- Scalable architecture

## 🔌 API Integration

### Hate Speech Service Endpoints

The app expects a REST API with the following endpoint:

**POST** `/predict`

Request:
```json
{
  "text": "Message content to analyze"
}
```

Response:
```json
{
  "prediction": "Category name",
  "prediction_id": 0,
  "confidence": 0.95,
  "is_harmful": false,
  "description": "Detailed explanation"
}
```

### Host Configuration

The service supports multiple host configurations with automatic fallback:

1. Android Emulator: `http://10.0.2.2:8000`
2. LAN/Physical Device: `http://192.168.1.73:8000`

Update `lib/services/hate_speech_service.dart` with your backend URL.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Flutter team for the amazing framework
- Firebase for backend services
- Hugging Face for NLP models and tools
- The open-source community

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Note**: This app is for educational purposes. Ensure you comply with privacy laws and regulations when deploying in production.
