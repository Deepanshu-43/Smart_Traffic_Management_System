# main.py
# Flask + YOLOv3 streaming server with dynamic camera switching
# Place this file in: Smart_Traffic_Management_System/New folder/main.py

import os
import time
import threading
from pathlib import Path

from flask import Flask, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import imutils

app = Flask(__name__)

# ------------------ CONFIG ------------------
CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3
FRAME_WIDTH = 700            # resize width for processing/display
DEFAULT_CAMERA = 0           # use PC webcam index 0 by default
# --------------------------------------------

# Resolve YOLO files relative to this script (safe, works on Windows)
BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = (BASE_DIR / ".." / "yolo object" / "yolo-coco").resolve()

LABELS_PATH = YOLO_DIR / "coco.names"
CFG_PATH = YOLO_DIR / "yolov3.cfg"
WEIGHTS_PATH = YOLO_DIR / "yolov3.weights"

# Check YOLO files exist
for p in (LABELS_PATH, CFG_PATH, WEIGHTS_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Required YOLO file not found: {p}. Make sure 'yolo-coco' folder is in '../yolo object/'")

# Load labels & colors
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
if not LABELS:
    raise ValueError(f"No labels found in {LABELS_PATH}")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO network
print(f"[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
layer_names = net.getLayerNames()
try:
    out_layers = net.getUnconnectedOutLayers()
    # out_layers can be array of ints or array of arrays -> flatten safely
    out_layers_flat = [int(x) for item in out_layers for x in (item if hasattr(item, "__iter__") else [item])]
    OUTPUT_LAYERS = [layer_names[i - 1] for i in out_layers_flat]
except Exception:
    # fallback
    OUTPUT_LAYERS = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
print(f"[INFO] YOLO loaded.")

# -------- camera state (shared) ----------
cap = None                  # cv2.VideoCapture object
cap_lock = threading.Lock() # protect cap switching
current_camera_source = None
# -----------------------------------------

def open_camera(source):
    """
    Open a cv2.VideoCapture for the given source.
    `source` can be an int (camera index) or a URL string.
    Returns the VideoCapture object if opened successfully, else None.
    """
    # if numeric string, convert to int
    try:
        src = int(source)
    except Exception:
        src = source  # keep as string (URL)

    new_cap = cv2.VideoCapture(src)
    # small warmup delay
    time.sleep(0.3)
    if not new_cap.isOpened():
        print(f"[WARN] Failed to open camera: {source}")
        new_cap.release()
        return None
    return new_cap

def switch_camera(source):
    """
    Switch global camera to `source`. Returns (True, message) on success.
    """
    global cap, current_camera_source
    
    if source == current_camera_source:
        return True, f"Camera {source} already active."

    new_cap = open_camera(source)
    if new_cap is None:
        return False, f"Unable to open camera: {source}"

    with cap_lock:
        # release old capture
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except Exception as e:
            print(f"[WARN] Exception releasing old camera: {e}")
            
        cap = new_cap
        current_camera_source = source
        print(f"[INFO] Switched camera to: {source}")
    return True, f"Switched to {source}"

# Initialize default camera on startup
ok, msg = switch_camera(DEFAULT_CAMERA)
if not ok:
    # still continue; frontend can attempt switching
    print(f"[WARN] Failed to open default camera ({DEFAULT_CAMERA}): {msg}")
else:
    print(f"[INFO] Default camera opened: {DEFAULT_CAMERA}")

def yolo_process_frame(frame):
    """
    Run YOLO on a single frame and draw boxes/labels.
    Returns annotated frame.
    """
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(OUTPUT_LAYERS)

    boxes = []
    confidences = []
    classIDs = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            if scores.size == 0:
                continue
            classID = int(np.argmax(scores))
            confidence = float(scores[classID])

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                classIDs.append(classID)

    idxs = []
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)

    # Draw bounding boxes & labels
    if len(idxs) > 0:
        # idxs might be numpy array of shape (N,1) or list
        flat = idxs.flatten() if hasattr(idxs, "flatten") else idxs
        for i in flat:
            i = int(i)
            x, y, w, h = boxes[i]
            if classIDs[i] >= len(LABELS):
                print(f"[WARN] Invalid classID {classIDs[i]} detected.")
                continue
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def generate_frames():
    """
    Generator that yields MJPEG frames (bytes) for Flask Response.
    It reads from the global `cap` and processes frames through YOLO.
    """
    global cap
    while True:
        with cap_lock:
            local_cap = cap

        if local_cap is None or not local_cap.isOpened():
            # no camera available or it disconnected; show error frame
            print("[WARN] No camera available. Yielding error frame.")
            error_frame = np.zeros((400, FRAME_WIDTH, 3), dtype="uint8")
            cv2.putText(error_frame, "Camera Feed Unavailable", (int(FRAME_WIDTH/2) - 130, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode(".jpg", error_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1) # Wait 1 sec before retrying
            continue

        ret, frame = local_cap.read()
        if not ret or frame is None:
            # try again quickly
            print("[WARN] Failed to read frame from camera.")
            time.sleep(0.05)
            continue

        try:
            annotated = yolo_process_frame(frame)
            ret2, jpeg = cv2.imencode(".jpg", annotated)
            if not ret2:
                continue
            frame_bytes = jpeg.tobytes()
        except Exception as e:
            # if YOLO or encoding fails, skip frame
            print(f"[ERROR] processing frame: {e}")
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ----------------- Flask Routes -----------------

@app.route("/")
def index():
    """Serve the main.html file."""
    # This assumes main.html is in the same directory as main.py
    return send_from_directory(".", "main.html")

@app.route("/video_feed")
def video_feed():
    """
    MJPEG stream endpoint. <img src="/video_feed">
    """
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_camera", methods=["POST"])
def api_set_camera():
    """
    JSON body: { "camera": 0 }  OR  { "camera": "1" } OR { "camera": "rtsp://..." }
    Returns JSON {status: "success"/"error", message: ...}
    """
    data = request.get_json(force=True, silent=True)
    if not data or "camera" not in data:
        return jsonify({"status": "error", "message": "Missing 'camera' in JSON body"}), 400

    source = data["camera"]
    success, message = switch_camera(source)
    if success:
        return jsonify({"status": "success", "camera": source, "message": message})
    else:
        return jsonify({"status": "error", "camera": source, "message": message}), 400

@app.route("/current_camera")
def api_current_camera():
    """Return current camera source."""
    with cap_lock:
        src = current_camera_source
    return jsonify({"camera": src})

@app.route("/health")
def health():
    with cap_lock:
        status = "ok" if (cap is not None and cap.isOpened()) else "error"
        src = current_camera_source
    return jsonify({"status": status, "camera": src})

# -------------------------------------------------
if __name__ == "__main__":
    print("="*50)
    print(f"YOLO server running.")
    print(f"YOLO dir: {YOLO_DIR}")
    print("Access the web UI at: http://localhost:5000/")
    print("Video stream: http://localhost:5000/video_feed")
    print("="*50)
    # threaded=True is important for handling multiple requests
    app.run(host="0.0.0.0", port=5000, threaded=True)