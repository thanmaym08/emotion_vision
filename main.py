from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os

app = FastAPI()

# ✅ Add health check route to prevent 404 on "/"
@app.get("/")
async def root():
    return {"message": "EmotionVision API is running ✅"}

# ✅ Enable CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify exact origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define the model class
class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ✅ Load the model
num_classes = 7
model = EmotionEfficientNet(num_classes)
try:
    state_dict = torch.load("efficientnet_b4_emotion.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)

# ✅ Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Label mapping
class_labels = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]

# ✅ Prediction endpoint
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

        return {
            "prediction": pred_idx,
            "label": label
        }

    except Exception as e:
        print("🚨 Prediction error:", e)
        return {
            "error": str(e)
        }

# ✅ Dynamic port binding (Render uses PORT env variable)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
