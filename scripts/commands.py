import time
import struct
import threading
import cv2
import zmq
import numpy as np

# =========================
# ZMQ CONNECTION & LOCKING
# =========================
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
zmq_lock = threading.Lock()

def connect_zmq(address="tcp://localhost:5555"):
    print("Waiting for Isaac...")
    socket.connect(address)
    while True:
        try:
            socket.send(struct.pack("dd", 0.0, 0.0))
            socket.recv()
            break
        except Exception:
            time.sleep(0.5)
    print("Connected ✔")

def safe_zmq_send(left, right):
    with zmq_lock:
        try:
            socket.send(struct.pack("dd", float(left), float(right)))
            return socket.recv()
        except Exception as e:
            print(f"[ZMQ ERROR] {e}")
            return None

# =========================
# ROBOT STATE
# =========================
vision_enabled = False
vision_window_open = False  
last_left = 0.0
last_right = 0.0

motion_lock = threading.Lock()
motion_id = 0

target_lock = threading.Lock()
target_id = 0
target_thread = None
current_annotated_frame = None

# =========================
# CORE FUNCTIONS
# =========================

def execute_motion(left, right, duration=None):
    global motion_id, last_left, last_right
    last_left, last_right = left, right
    
    with motion_lock:
        motion_id += 1
        current_id = motion_id

    def worker():
        print(f"[MOTION] {left}, {right}")
        safe_zmq_send(left, right)
        if duration is not None:
            start = time.time()
            while time.time() - start < duration:
                if current_id != motion_id: return
                time.sleep(0.05)
            if current_id == motion_id:
                print("[MOTION] STOP")
                safe_zmq_send(0, 0)

    threading.Thread(target=worker, daemon=True).start()

def execute_target(prompt_class, frame_source, model_instance, memory_timeout=2.0):
    global target_id, target_thread, current_annotated_frame, vision_enabled
    vision_enabled = True
    
    with target_lock:
        target_id += 1
        current_id = target_id

    def worker():
        global current_annotated_frame
        last_seen = None
        last_seen_time = None
        
        clean_class = prompt_class.strip().replace('"', '').replace("'", "")
        model_instance.set_classes([clean_class])

        while current_id == target_id:
            if frame_source is None: continue
            
            local_frame = frame_source.copy()
            results = model_instance.predict(local_frame, conf=0.2, verbose=False)
            img_width = local_frame.shape[1]
            target_center = None
            largest_area = 0

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    target_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if target_center:
                last_seen, last_seen_time = target_center, time.time()
            elif last_seen and (time.time() - last_seen_time) > memory_timeout:
                last_seen = None

            # Control Logic
            if last_seen:
                offset = ((last_seen[0] / img_width) - 0.5) * 200
                deadzone = 10
                if abs(offset) <= deadzone:
                    safe_zmq_send(5, 5)
                else:
                    mapped = max(0, min(3, (abs(offset) - deadzone) / 90 * 3))
                    if offset > 0: safe_zmq_send(5 - mapped, 5 + mapped)
                    else: safe_zmq_send(5 + mapped, 5 - mapped)
            else:
                safe_zmq_send(0, 0)

            temp_annotated = results[0].plot()
            if last_seen:
                cv2.circle(temp_annotated, (int(last_seen[0]), int(last_seen[1])), 8, (0, 255, 255), -1)
            current_annotated_frame = temp_annotated

        current_annotated_frame = None

    threading.Thread(target=worker, daemon=True).start()

def update_vision_system(raw_frame, window_name):
    global vision_enabled, vision_window_open, current_annotated_frame, target_id
    
    if vision_enabled and raw_frame is not None:
        display_frame = current_annotated_frame if current_annotated_frame is not None else raw_frame
        cv2.imshow(window_name, display_frame)
        vision_window_open = True
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            vision_enabled = False
            with target_lock: target_id += 1
            current_annotated_frame = None
    elif vision_window_open:
        try: cv2.destroyWindow(window_name)
        except: pass
        vision_window_open = False