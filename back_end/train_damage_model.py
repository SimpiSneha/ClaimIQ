"""
Trains the Damage Assessment Agent.
Run this once (and again if you want to retrain):
    python train_damage_model.py

Uses a pretrained ResNet18 as a frozen feature extractor, with a small
trainable classifier head on top (transfer learning) - this is fast to
train and doesn't require a GPU, while still being a real, evaluable
computer vision model rather than an LLM guessing at an image.

Expects data in:
    data/Car Damage Severity Dataset/training/{01-minor,02-moderate,03-severe}/
    data/Car Damage Severity Dataset/validation/{01-minor,02-moderate,03-severe}/
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import joblib

from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path(__file__).parent.parent / "data" / "Car Damage Severity Dataset"
TRAIN_DIR = DATA_DIR / "training"
VAL_DIR = DATA_DIR / "validation"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ImageNet normalization stats - required since we're using a model
# pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # simple augmentation - cars can be photographed from either side
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # class_to_idx gives us the mapping, e.g. {'01-minor': 0, '02-moderate': 1, '03-severe': 2}
    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int):
    # Pretrained ResNet18, frozen except for the final classifier layer.
    # This is standard transfer learning: reuse ImageNet's learned visual
    # features, only train a small new head for our 3 damage classes.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer with one matching our classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(DEVICE)


def train_and_evaluate():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Expected training/validation folders under {DATA_DIR}. "
            f"Check the dataset is placed correctly."
        )

    train_loader, val_loader, class_names = get_data_loaders()
    print(f"Classes found: {class_names}")
    print(f"Training on {len(train_loader.dataset)} images, validating on {len(val_loader.dataset)} images")
    print(f"Using device: {DEVICE}\n")

    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    # Only the new final layer has requires_grad=True, so only it gets trained
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - train loss: {train_loss:.4f}, train acc: {train_acc:.3f}")

    # Final evaluation on the validation set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))

    # Save the trained model and class mapping
    torch.save(model.state_dict(), MODEL_DIR / "damage_model.pt")
    joblib.dump(class_names, MODEL_DIR / "damage_class_names.joblib")
    print(f"\nModel saved to {MODEL_DIR / 'damage_model.pt'}")


if __name__ == "__main__":
    train_and_evaluate()