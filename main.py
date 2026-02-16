from multiprocessing import shared_memory
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from ultralytics import YOLOE
import numpy as np
import cv2
import sys
import os
import re

# Local Imports
import scripts.commands as robot
from gui import JarvisInference
from nim import Nvidia

# =========================
# CONFIG & INITIALIZATION
# =========================
nv = Nvidia(config_path="config.yaml")
model = YOLOE("models/yoloe.pt")
VISION_WINDOW = "Jarvis Vision"
META_FILE = "scripts/Apdata.txt"

memory = []
shm = None
frame = None

# Connect Robot ZMQ
robot.connect_zmq("tcp://localhost:5555")

if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        name, H, W, C, dtype_str = [f.readline().strip() for _ in range(5)]
    shm = shared_memory.SharedMemory(name=name)
    frame = np.ndarray((int(H), int(W), int(C)), dtype=np.dtype(dtype_str), buffer=shm.buf)
    print(f"[OK] Shared memory connected: {name}")

if not os.path.exists("vision"): os.makedirs("vision")

# =========================
# CLEAN PARSERS
# =========================
def CommandExtractor(text):
    commands = re.findall(r'\*(.*?)\*', text)
    clean_text = re.sub(r'\*.*?\*', '', text)
    clean_text = ' '.join(clean_text.split())
    return commands, clean_text

def CommandHandler(cmd):
    parts = cmd.split()
    if not parts: return
    name = parts[0].lower()

    if name in ["forward", "backward", "left", "right"]:
        try:
            duration = float(parts[1]) if len(parts) > 1 else None
        except:
            duration = None
            
        speeds = {"forward": (5, 5), "backward": (-5, -5), "left": (2, -2), "right": (-2, 2)}
        robot.execute_motion(*speeds[name], duration)

    elif name == "speed":
        modifier = parts[1] if len(parts) > 1 else "+"
        factor = 2.0 if modifier == "+" else 0.5
        robot.execute_motion(robot.last_left * factor, robot.last_right * factor)

    elif name == "stop":
        with robot.target_lock: robot.target_id += 1 
        robot.current_annotated_frame = None
        robot.execute_motion(0, 0)

    elif name == "vision":
        robot.vision_enabled = True

    elif name == "target":
        target_class = ' '.join(parts[1:]) if len(parts) > 1 else "person"
        robot.execute_target(target_class, frame, model)

# =========================
# INFERENCE LOGIC
# =========================
def Process(prompt, file):
    global memory
    media = []
    if frame is not None:
        cv2.imwrite("vision/vision.jpg", frame)
        media.append("vision/vision.jpg")
    if file: media.append(file)

    response = nv.ask(prompt, memory=memory, media_files=media)
    commands, clean_response = CommandExtractor(response)
    
    memory.append({"role": "user", "content": prompt})
    memory.append({"role": "assistant", "content": clean_response})

    for cmd in commands:
        CommandHandler(cmd)

    return clean_response

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Bridge the GUI timer to the command vision updater
    vision_timer = QTimer()
    vision_timer.timeout.connect(lambda: robot.update_vision_system(frame, VISION_WINDOW))
    vision_timer.start(30)

    window = JarvisInference(on_message_callback=Process)
    window.show()

    exit_code = app.exec()
    if shm: shm.close()
    cv2.destroyAllWindows()
    sys.exit(exit_code)