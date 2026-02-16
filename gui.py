from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QScrollArea, QLabel, QFrame, 
                             QGraphicsOpacityEffect, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import sys
import os
import sounddevice
import speech_recognition as sr
from faster_whisper import WhisperModel

# ========================================================
# REAL WHISPER STT WORKER
# ========================================================
class STTWorker(QThread):
    text_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        # Use "tiny" for speed, "base" for better accuracy
        self.model_size = "tiny" 
        
    def run(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise for 0.5 seconds
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen until silence is detected or stop() is called
                audio = recognizer.listen(source, phrase_time_limit=10)
                
                if not self.is_running:
                    return

                # Save temporary audio to process with Whisper
                with open("temp_speech.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                # Load and Transcribe
                model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                segments, info = model.transcribe("temp_speech.wav", beam_size=5)
                
                text = " ".join([segment.text for segment in segments])
                self.text_ready.emit(text.strip())
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if os.path.exists("temp_speech.wav"):
                os.remove("temp_speech.wav")

    def stop(self):
        self.is_running = False

class ResponseWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, callback, text, file_path):
        super().__init__()
        self.callback = callback
        self.text = text
        self.file_path = file_path

    def run(self):
        response = self.callback(self.text, self.file_path)
        self.finished.emit(response)

class ChatBubble(QFrame):
    def __init__(self, text, is_user=True):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        self.bubble = QLabel(text)
        self.bubble.setWordWrap(True)
        self.bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.bubble.setMaximumWidth(350) 
        self.bubble.setContentsMargins(15, 12, 15, 12)
        
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        if is_user:
            self.bubble.setStyleSheet("""
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #76b900, stop:1 #5da200);
                color: white; border-radius: 15px; border-bottom-right-radius: 2px;
                font-family: 'Segoe UI'; font-size: 13px; font-weight: 500;
            """)
            self.layout.addStretch()
            self.layout.addWidget(self.bubble)
        else:
            self.bubble.setStyleSheet("""
                background-color: #2a2a2a; color: #ffffff; border: 1px solid #76b900;
                border-radius: 15px; border-bottom-left-radius: 2px;
                font-family: 'Segoe UI'; font-size: 13px;
            """)
            self.layout.addWidget(self.bubble)
            self.layout.addStretch()

class JarvisInference(QWidget):
    def __init__(self, on_message_callback):
        super().__init__()
        self.on_message_callback = on_message_callback
        self.current_file = None
        self.anims = []
        self.worker = None
        self.stt_worker = None
        self.is_listening = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Jarvis Interface")
        self.resize(500, 780)
        self.setAcceptDrops(True)
        self.setStyleSheet("QWidget { background-color: #050505; }")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 10, 20, 35)

        # --- TOP MENU ---
        self.top_bar = QHBoxLayout()
        self.settings_btn = self._create_icon_btn("⚙", "Erase history", self.clear_chat)
        self.volume_btn = self._create_icon_btn("🔊", "Voice response", lambda: print("Voice toggle"))
        self.top_bar.addWidget(self.settings_btn)
        self.top_bar.addWidget(self.volume_btn)
        self.top_bar.addStretch()
        self.main_layout.addLayout(self.top_bar)

        # --- CHAT BOX ---
        self.chat_box = QFrame()
        self.chat_box.setStyleSheet("background-color: #121212; border: 1px solid #222; border-radius: 25px;")
        self.chat_box_layout = QVBoxLayout(self.chat_box)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")
        self._style_scrollbar()
        
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(15)
        
        self.scroll.setWidget(self.container)
        self.chat_box_layout.addWidget(self.scroll)
        self.main_layout.addWidget(self.chat_box)

        # --- INPUT AREA ---
        self.input_container = QFrame()
        self.input_container.setFixedHeight(60)
        self.input_container.setStyleSheet("background-color: #121212; border-radius: 30px; border: 1px solid #76b900;")
        self.input_layout = QHBoxLayout(self.input_container)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet("border: none; background: transparent; color: white; font-size: 14px; padding-left: 10px;")
        self.input_field.returnPressed.connect(self.handle_send)
        
        # Mic Button
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(40, 40)
        self.mic_btn.setStyleSheet("QPushButton { background-color: transparent; color: #76b900; font-size: 18px; border-radius: 20px; }")
        self.mic_btn.clicked.connect(self.toggle_mic)

        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setStyleSheet("QPushButton { background-color: #76b900; color: #050505; border-radius: 20px; font-weight: bold; }")
        self.send_btn.clicked.connect(self.handle_send)
        
        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.mic_btn)
        self.input_layout.addWidget(self.send_btn)
        self.main_layout.addWidget(self.input_container)

    def _create_icon_btn(self, icon, tip, func):
        btn = QPushButton(icon)
        btn.setFixedSize(35, 35)
        btn.setToolTip(tip)
        btn.setStyleSheet("QPushButton { background: transparent; color: #76b900; font-size: 18px; }")
        btn.clicked.connect(func)
        return btn

    def toggle_mic(self):
        if not self.is_listening:
            self.is_listening = True
            self.mic_btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; font-size: 18px; border-radius: 20px; }")
            self.input_field.setPlaceholderText("Listening...")
            self.stt_worker = STTWorker()
            self.stt_worker.text_ready.connect(self.fill_stt_text)
            self.stt_worker.error_occurred.connect(lambda e: print(f"STT Error: {e}"))
            self.stt_worker.start()
        else:
            self.stop_mic()

    def stop_mic(self):
        self.is_listening = False
        self.mic_btn.setStyleSheet("QPushButton { background-color: transparent; color: #76b900; font-size: 18px; border-radius: 20px; }")
        self.input_field.setPlaceholderText("Type a message...")
        if self.stt_worker:
            self.stt_worker.stop()

    def fill_stt_text(self, text):
        if text:
            self.input_field.setText(text)
        self.stop_mic()

    def _style_scrollbar(self):
        self.scroll.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical { border: none; background: #050505; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #76b900; min-height: 30px; border-radius: 4px; }
            QScrollBar::add-line, QScrollBar::sub-line { height: 0px; }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.chat_box.setStyleSheet("background-color: #1a1a1a; border: 2px dashed #76b900; border-radius: 25px;")
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.chat_box.setStyleSheet("background-color: #121212; border: 1px solid #222; border-radius: 25px;")
        urls = event.mimeData().urls()
        if urls:
            self.current_file = urls[0].toLocalFile()
            self.input_field.setPlaceholderText(f"File attached: {os.path.basename(self.current_file)}")

    def handle_send(self):
        if self.is_listening:
            self.stop_mic()

        text = self.input_field.text().strip()
        if text:
            display_text = text
            file_to_send = self.current_file
            if file_to_send:
                display_text += f"\n[File: {os.path.basename(file_to_send)}]"
            
            self.add_message(display_text, True)
            self.input_field.clear()
            self.input_field.setPlaceholderText("Thinking...")
            self.current_file = None
            
            self.worker = ResponseWorker(self.on_message_callback, text, file_to_send)
            self.worker.finished.connect(self.display_response)
            self.worker.start()

    def display_response(self, response_text):
        self.add_message(response_text, False)
        self.input_field.setPlaceholderText("Type a message...")

    def add_message(self, text, is_user):
        bubble = ChatBubble(text, is_user)
        self.chat_layout.addWidget(bubble)
        self.container.adjustSize()
        
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))
        
        fade = QPropertyAnimation(bubble.opacity_effect, b"opacity")
        fade.setDuration(400)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.start()
        self.anims.append(fade)

    def clear_chat(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()