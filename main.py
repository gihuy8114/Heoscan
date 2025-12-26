import os
import io
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

# --- CẤU HÌNH ---
# Chỉ import những thư viện nhẹ ở đây
app = FastAPI()

# Biến toàn cục lưu model và thư viện
models = {
    "yolo": None,
    "tflite": None,
    "input_details": None,
    "output_details": None,
    "tf": None,           # Lưu thư viện tensorflow
    "cv2": None,          # Lưu thư viện opencv
    "np": None,           # Lưu numpy
    "CLASS_NAMES": ["fresh", "spoiled"],
    "INPUT_SIZE": 224
}

# --- HÀM LOAD THƯ VIỆN & MODEL (Chỉ chạy khi có người gọi API) ---
def load_heavy_stuff_if_needed():
    # 1. Load thư viện trước (Lazy Import)
    if models["cv2"] is None:
        print("⏳ Importing libraries (CV2, Numpy, TF, YOLO)...")
        import cv2
        import numpy as np
        import tensorflow as tf
        from ultralytics import YOLO
        
        models["cv2"] = cv2
        models["np"] = np
        models["tf"] = tf
        models["YOLO_class"] = YOLO # Lưu class YOLO lại
        print("✅ Libraries Imported!")

    # 2. Load Models
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_PATH = os.path.join(BASE_DIR, "model", "test.pt")
    TFLITE_PATH = os.path.join(BASE_DIR, "model", "mobilenetv2_custom_float32.tflite")

    if models["yolo"] is None:
        print("⏳ Loading YOLO Model...")
        models["yolo"] = models["YOLO_class"](YOLO_PATH)
        print("✅ YOLO Loaded")

    if models["tflite"] is None:
        print("⏳ Loading TFLite Model...")
        interpreter = models["tf"].lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        models["tflite"] = interpreter
        models["input_details"] = interpreter.get_input_details()
        models["output_details"] = interpreter.get_output_details()
        print("✅ TFLite Loaded")

@app.get("/")
def home():
    # API này siêu nhẹ, server sẽ phản hồi ngay lập tức để Render biết là nó đang sống
    return {"message": "HeoScan API is Live (Super Lazy Mode)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load tất cả mọi thứ ở đây
        load_heavy_stuff_if_needed()
    except Exception as e:
        print(f"Error loading models: {e}")
        return {"error": "Server starting up, please try again in 30s."}

    # Lấy các biến ra cho gọn
    cv2 = models["cv2"]
    np = models["np"]
    tf = models["tf"]
    
    # 1. Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # 2. Detect bằng YOLO
    results = models["yolo"](frame, conf=0.4, verbose=False)
    pork_found_count = 0 

    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pork_found_count += 1

            # Crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # --- Xử lý TFLite ---
            # Preprocess
            img_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_crop = cv2.resize(img_crop, (models["INPUT_SIZE"], models["INPUT_SIZE"]))
            img_crop = img_crop.astype(np.float32)
            img_crop = img_crop / 127.5 - 1.0 
            input_data = np.expand_dims(img_crop, axis=0)

            # Predict
            interpreter = models["tflite"]
            interpreter.set_tensor(models["input_details"][0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(models["output_details"][0]['index'])[0]
            prob = tf.nn.softmax(output).numpy()
            
            freshness = models["CLASS_NAMES"][np.argmax(prob)]
            conf = np.max(prob)
            
            # Vẽ
            color = (0, 255, 0) if freshness == "fresh" else (0, 0, 255)
            label = f"{freshness.upper()} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 3. Xử lý khi không thấy thịt
    if pork_found_count == 0:
        h, w = frame.shape[:2]
        text = "KHONG TIM THAY THIT"
        font_scale = 1.0 if w < 500 else 1.5
        cv2.putText(frame, text, (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 3)

    # 4. Trả ảnh về
    res, im_jpg = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")
