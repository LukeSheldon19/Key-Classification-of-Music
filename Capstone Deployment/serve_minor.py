from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from typing import List
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO








app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create necessary directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML responses
templates = Jinja2Templates(directory="templates")

def load_ml_model():
    try:
        # model = load_model("minor_model.keras")
        model = load_model("../Saved_Models/best.h5")
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load ML model: {str(e)}"
        )

class MINORInput(BaseModel):
    image: List[List[List[float]]] 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not render template: {str(e)}"
        )

@app.post("/predict")
async def predict(input: MINORInput, model = Depends(load_ml_model)):
    try:
        # Validate input shape: (224, 224, 3)
        if len(input.image) != 224 or any(len(row) != 224 or len(row[0]) != 3 for row in input.image):
            raise ValueError("Input must be a 224x224x3 array")
        
        # Convert input to numpy array and reshape for model
        image_array = np.array(input.image, dtype=np.float32).reshape(1, 224, 224, 3)
        
        prediction = model.predict(image_array)
        
        return JSONResponse({
            "predicted_class": int(np.argmax(prediction)),
            "confidence": float(np.max(prediction)),
            "all_predictions": prediction.tolist()[0]
        })
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )






def create_chromagram(file_path_or_bytes):
    x, sampling_rate = librosa.load(file_path_or_bytes, sr=None)
    s = librosa.stft(x)
    H, P = librosa.decompose.hpss(s)
    chroma = librosa.feature.chroma_stft(S=np.abs(H), sr=sampling_rate)
    return chroma

def chromagram_to_image(chroma, target_size=(224, 224)):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    librosa.display.specshow(chroma, y_axis=None, x_axis=None, ax=ax)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

@app.post("/predict-upload")
async def predict_upload(file: UploadFile = File(...), model=Depends(load_ml_model)):
    try:
        # Validate file type (audio)
        if not file.content_type.startswith('audio/'):
            raise ValueError("Uploaded file must be an audio file (.wav, .mp3, etc.)")
        
        contents = await file.read()
        
        # Create chromagram
        chroma = create_chromagram(BytesIO(contents))
        img = chromagram_to_image(chroma)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Prepare batch (shape: 1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        
        return JSONResponse({
            "predicted_class": int(np.argmax(prediction)),
            "confidence": float(np.max(prediction)),
            "all_predictions": prediction.tolist()[0]
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio processing failed: {str(e)}"
        )