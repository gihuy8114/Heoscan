import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io

# --- CẤU HÌNH ĐƯỜNG DẪN (Tương đối theo thư mục dự án) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "test.pt")
MOBILENET_MODEL_PATH = os.path.join(BASE_DIR, "model", "mobilenetv2_custom_float32.tflite")

CLASS_NAMES = ["fresh", "spoiled"]
INPUT_SIZE = 224

# Biến toàn cục để lưu model
models = {}

# --- HÀM KHỞI TẠO MODEL (Load 1 lần khi bật server) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading models...")
    # Load YOLO
    models["yolo"] = YOLO(YOLO_MODEL_PATH)
    
    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=MOBILENET_MODEL_PATH)
    interpreter.allocate_tensors()
    models["tflite"] = interpreter
    models["input_details"] = interpreter.get_input_details()
    models["output_details"] = interpreter.get_output_details()
    
    print("✅ Models loaded successfully!")
    yield
    # Clean up nếu cần
    models.clear()

app = FastAPI(lifespan=lifespan)

# --- HÀM XỬ LÝ PHỤ TRỢ ---
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

    input_data = preprocess_tflite(crop)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    prob = tf.nn.softmax(output).numpy()
    return CLASS_NAMES[np.argmax(prob)], np.max(prob)

@app.get("/")
def home():
    return {"message": "Pork Quality Check API is Running"}

# --- API CHÍNH: NHẬN ẢNH VÀ TRẢ KẾT QUẢ ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Đọc file ảnh từ App gửi lên
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Cannot read image"}

    # 2. Chạy YOLO detect thịt
    results = models["yolo"](frame, conf=0.4, verbose=False)
    pork_detected = False

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = models["yolo"].names[int(box.cls.item())]

            # Chỉ xử lý nếu là 'pork' (hoặc class bạn muốn)
            # Lưu ý: check lại xem model YOLO của bạn trả về tên class chính xác là gì
            # Nếu model bạn train chỉ có 1 class thì bỏ qua check tên cũng được
            # if cls_name.lower() != "pork": continue 
            
            pork_detected = True

            # Crop và chạy Classifier
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            freshness, conf = classify_tflite(crop)

            # Vẽ màu: Xanh (Fresh) - Đỏ (Spoiled)
            color = (0, 255, 0) if freshness == "fresh" else (0, 0, 255)
            label = f"{freshness.upper()} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 3. Mã hóa ảnh lại thành JPEG để trả về
    res, im_png = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")