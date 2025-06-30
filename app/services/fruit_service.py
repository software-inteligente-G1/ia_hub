import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# Ruta al modelo y clases
MODEL_PATH = Path("app/models/fruit_model.pth")
CLASS_NAMES_PATH = Path("app/models/class_names.txt")

# Cargar nombres de clase reales
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Reconstruir la arquitectura base ResNet50
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))  # 51 clases reales

# Cargar pesos entrenados
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Transformación de imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImagenNet
        std=[0.229, 0.224, 0.225]
    )
])

def predict_resnet_image(image: Image.Image) -> dict:
    try:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = int(predicted.item())
            return {
                "class_index": class_idx,
                "class_name": CLASSES[class_idx]
            }
    except Exception as e:
        raise RuntimeError(f"Error en predicción ResNet: {e}")
