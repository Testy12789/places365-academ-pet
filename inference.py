import torch
from torchvision import transforms, models
from PIL import Image

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 365

# ================= MODEL =================
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("models/best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================= TRANSFORMS =================
inference_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ================= INFERENCE FUNCTION =================
def infer(image_path):
    img = Image.open(image_path).convert("RGB")
    img = inference_tf(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return pred, probs[0][pred].item()

# ================= EXAMPLE =================
if __name__ == "__main__":
    img_path = "exemple.jpg"
    pred_class, confidence = infer(img_path)
    print(f"Predicted class: {pred_class}, Confidence: {confidence:.4f}")
