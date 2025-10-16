# Virtual Mouse

Virtual Mouse turns your webcam into an input device by tracking hand landmarks with MediaPipe and translating gestures into cursor movement, clicks, drags, and optional voice dictation. It is designed for hands-free interaction demos such as accessibility tooling, AR prototypes, or presentation control.

## Key Features

Smooth cursor control driven by index-finger motion within a configurable gesture frame.
Natural pinch gestures for left-click and sustained drag operations.
Optional speech-to-text mode (Google Speech Recognition via speech_recognition) triggered while the pinky stays raised.
Real-time FPS overlay, topmost preview window, and automatic webcam index fallback.

## Prerequisites

Python 3.9+
Packages: opencv-python, mediapipe, numpy, pynput, screeninfo, SpeechRecognition (and pyaudio for microphone input).
Webcam and microphone access.
Getting Started

## Install dependencies:
pip install opencv-python mediapipe numpy pynput screeninfo SpeechRecognition pyaudio
(Optional) Enable voice dictation by setting doVoiceType = True in virtualmouse.py.
Run the application from the project root:
python virtualmouse.py
Use the ESC key to exit; say “stop program” during voice mode to terminate from speech input.

## Gestures

Move pointer: index finger inside the on-screen frame.
Click: tap thumb to index base (MCP or PIP).
Drag: pinch thumb to index fingertip, release to drop.
Voice typing: with voice mode on, raise pinky to start listening, then speak text; say “enter” to submit a newline.
