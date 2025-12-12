from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
import io
from PIL import Image

# --- AYARLAR ---
app = FastAPI()

# Güvenlik Ayarları (CORS) - Her yerden erişime izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli Yükle
model = YOLO("best_s.onnx", task='detect')

# Duygu Renkleri ve Çevirileri
TR_EMOTIONS = {
    'Anger': 'KIZGIN', 'Contempt': 'KUCUMSEME', 'Disgust': 'TIKSINTI',
    'Fear': 'KORKU', 'Happy': 'MUTLU', 'Neutral': 'NOTR',
    'Sad': 'UZGUN', 'Surprise': 'SASKIN'
}

SUGGESTIONS = {
    'Anger': "⚠️ Gerilim var. Mola verin.",
    'Contempt': "🟠 Dikkat daginik. Onemi vurgulayin.",
    'Disgust': "🤢 Konu anlasilmadi. Ornegi degistirin.",
    'Fear': "🟣 Kaygi yuksek. Guven verin.",
    'Happy': "🟢 Sinif pozitif. Zor soru sorun.",
    'Neutral': "⚪ Standart akis. Soru sorun.",
    'Sad': "🔵 Enerji dusuk. Ses tonunu yukseltin.",
    'Surprise': "❗ Saskinlik var. Tekrar edin."
}

@app.get("/")
def read_root():
    return {"status": "EduSense API Calisiyor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Görüntüyü oku
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Görüntü İyileştirme (Contrast)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Tahmin
    results = model(img_processed, conf=0.15)
    
    response_data = {
        "emotion": "Neutral",
        "emotion_tr": "NOTR",
        "suggestion": "Analiz bekleniyor...",
        "box": []
    }

    if results[0].boxes:
        # En yüksek skorlu kutuyu al
        best_box = max(results[0].boxes, key=lambda x: x.conf[0])
        cls = int(best_box.cls[0].cpu().numpy())
        class_name = model.names[cls]
        
        # Koordinatlar
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        
        response_data["emotion"] = class_name
        response_data["emotion_tr"] = TR_EMOTIONS.get(class_name, class_name)
        response_data["suggestion"] = SUGGESTIONS.get(class_name, "...")
        response_data["box"] = [int(x1), int(y1), int(x2), int(y2)]

    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)