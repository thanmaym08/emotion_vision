import os
import io
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Load model class and weights
class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

num_classes = 7
model = EmotionEfficientNet(num_classes)

try:
    state_dict = torch.load("efficientnet_b4_emotion.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
class_labels = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprised"
]

# Prediction function for Gradio
def predict_emotion(image):
    try:
        # Process image
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            label = class_labels[pred_idx] if pred_idx < len(class_labels) else "unknown"
        
        return label
    except Exception as e:
        return str(e)

# Create Gradio interface
demo = gr.Interface(fn=predict_emotion, inputs=gr.Image(type="pil"), outputs="text")

# Run FastAPI locally (for local testing)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Launch Gradio interface
demo.launch()
