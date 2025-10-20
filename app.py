import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import traceback

IMG_SIZE = (128, 128)
THRESHOLD = 0.5
MODEL_PATH = "best_model.keras"

app = FastAPI(title="Cats vs Dogs API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods={"*"},
    allow_headers=["*"]
)

try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Error loading the model from-{MODEL_PATH}: {e}")

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Image file cannot be read.")
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def postprocess(pred: float):
    label_id = int(pred >= THRESHOLD)
    label = "dog" if label_id == 1 else "cat"
    confidence = float(pred) if label == "dog" else float(1.0 - pred)
    return label, confidence

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="File not uploaded.")
    file_bytes = await file.read()
    try:
        batch = preprocess_image(file_bytes)
        pred = model.predict(batch, verbose=0)[0][0]
        label, confidence = postprocess(float(pred))
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_score": round(float(pred), 6),
            "threshold": THRESHOLD,
            "input_size": IMG_SIZE,
        }
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(">>> PREDICT ERROR: \n", tb)
        raise HTTPException(status_code=500, detail="Prediction time error.")
    