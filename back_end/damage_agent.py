"""
Damage Assessment Agent - inference.
Loads the trained damage classifier and predicts severity for a new image.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib

MODEL_DIR = Path(__file__).parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_model = None
_class_names = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _load_model():
    global _model, _class_names
    if _model is None:
        model_path = MODEL_DIR / "damage_model.pt"
        class_names_path = MODEL_DIR / "damage_class_names.joblib"
        if not model_path.exists():
            raise RuntimeError(
                "Damage model not found. Run `python train_damage_model.py` first."
            )
        _class_names = joblib.load(class_names_path)

        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(_class_names))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model, _class_names


def classify_damage(image_path: str) -> dict:
    """
    Classifies the severity of car damage in the given image.
    Returns the predicted class and confidence scores for all classes.
    """
    model, class_names = _load_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities))
    predicted_class = class_names[predicted_idx]

    # Clean up the folder-name prefix (e.g. "01-minor" -> "minor")
    clean_label = predicted_class.split("-", 1)[-1] if "-" in predicted_class else predicted_class

    confidence_scores = {
        name.split("-", 1)[-1] if "-" in name else name: round(float(prob), 3)
        for name, prob in zip(class_names, probabilities)
    }

    return {
        "severity": clean_label,
        "confidence": round(float(probabilities[predicted_idx]), 3),
        "all_scores": confidence_scores,
    }