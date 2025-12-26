import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "test.pt")
MOBILENET_MODEL_PATH = os.path.join(BASE_DIR, "model", "mobilenetv2_custom_float32.tflite")
CLASS_NAMES = ["fresh", "spoiled"]
INPUT_SIZE = 224

# Biến toàn cục lưu model (Ban đầu để trống)
models = {
    "yolo": None,
    "tflite": None,
    "input_details": None,
    "output_details": None
}

app = FastAPI()

# --- HÀM LOAD MODEL (Chỉ chạy khi cần thiết) ---
def load_models_if_needed():
    # Chỉ load nếu chưa có model
    if models["yolo"] is None:
        print("⏳ Starting to load YOLO...")
        models["yolo"] = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO Loaded")

    if models["tflite"] is None:
        print("⏳ Starting to load TFLite...")
        interpreter = tf.lite.Interpreter(model_path=MOBILENET_MODEL_PATH)
        interpreter.allocate_tensors()
        models["tflite"] = interpreter
        models["input_details"] = interpreter.get_input_details()
        models["output_details"] = interpreter.get_output_details()
        print("✅ TFLite Loaded")

# --- CÁC HÀM XỬ LÝ ẢNH ---
def preprocess_tflite(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32)
    img = img / 127.5 - 1.0 
    return np.expand_dims(img, axis=0)

def classify_tflite(crop):
    interpreter = models["tflite"]
    input_details = models["input_details"]
    output_details = models["output_details"]
    
    interpreter.set_tensor(input_details[0]['index'], preprocess_tflite(crop))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    prob = tf.nn.softmax(output).numpy()
    return CLASS_NAMES[np.argmax(prob)], np.max(prob)

@app.get("/")
def home():
    return {"message": "HeoScan API is running (Lazy Loading Mode)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. QUAN TRỌNG: Load model tại đây (nếu chưa có)
    # Lần request đầu tiên sẽ chậm, các lần sau sẽ nhanh
    try:
        load_models_if_needed()
    except Exception as e:
        print(f"Model loading failed: {e}")
        return {"error": "Server is overloading, please try again."}

    # 2. Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # 3. Detect bằng YOLO
    results = models["yolo"](frame, conf=0.4, verbose=False)
    
    pork_found_count = 0 

    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            pork_found_count += 1

            # Crop và check độ tươi
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            freshness, conf = classify_tflite(crop)
            
            # Vẽ khung
            color = (0, 255, 0) if freshness == "fresh" else (0, 0, 255)
            label = f"{freshness.upper()} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 4. XỬ LÝ KHI KHÔNG TÌM THẤY THỊT
    if pork_found_count == 0:
        h, w = frame.shape[:2]
        text = "KHONG TIM THAY THIT"
        font_scale = 1.0 if w < 500 else 1.5
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    # 5. Trả ảnh về
    res, im_jpg = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")
