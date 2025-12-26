import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "test.pt")
MOBILENET_MODEL_PATH = os.path.join(BASE_DIR, "model", "mobilenetv2_custom_float32.tflite")
CLASS_NAMES = ["fresh", "spoiled"]
INPUT_SIZE = 224

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading models...")
    models["yolo"] = YOLO(YOLO_MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MOBILENET_MODEL_PATH)
    interpreter.allocate_tensors()
    models["tflite"] = interpreter
    models["input_details"] = interpreter.get_input_details()
    models["output_details"] = interpreter.get_output_details()
    print("✅ Models loaded!")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

# Hàm phụ trợ (giữ nguyên)
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
    return {"message": "HeoScan API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # 2. Detect bằng YOLO
    results = models["yolo"](frame, conf=0.4, verbose=False)
    
    pork_found_count = 0 # Biến đếm số lượng thịt tìm thấy

    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- QUAN TRỌNG: Kiểm tra class name ---
            # Nếu model của bạn chỉ train 1 class 'pork' thì index 0 luôn là pork.
            # Nếu bạn muốn chắc chắn, hãy uncomment dòng dưới:
            # cls_name = models["yolo"].names[int(box.cls.item())]
            # if cls_name != "pork": continue 
            
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

    # 3. XỬ LÝ KHI KHÔNG TÌM THẤY THỊT
    if pork_found_count == 0:
        # Lấy kích thước ảnh để vẽ chữ vào giữa
        h, w = frame.shape[:2]
        text = "KHONG TIM THAY THIT"
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Vẽ chữ đỏ nền trắng cho dễ đọc
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    # 4. Trả ảnh về (Dù có thịt hay không cũng trả về ảnh để hiện lên ResultScreen)
    res, im_jpg = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")
