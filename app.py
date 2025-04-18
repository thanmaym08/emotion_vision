# app.py
import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mount for index.html
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

# Emotion model class
class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
num_classes = 7
model = EmotionEfficientNet(num_classes)

MODEL_PATH = "efficientnet_b4_emotion.pth"
print("📦 Checking model file:", os.path.exists(MODEL_PATH))
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            label = class_labels[pred_idx] if pred_idx < len(class_labels) else "unknown"

        return {"prediction": pred_idx, "label": label}

    except Exception as e:
        return {"error": str(e)}
