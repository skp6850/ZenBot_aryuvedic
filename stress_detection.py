import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Constants
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 30
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Dataset
class FER2013Dataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.images = data['pixels'].tolist()
        self.labels = data['emotion'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.array(self.images[idx].split(), dtype='float32').reshape(48, 48)
        image = torch.tensor(image).unsqueeze(0) / 255.0
        image = image.repeat(3, 1, 1)
        label = int(self.labels[idx])
        return image, label

# Load data
dataset = FER2013Dataset("Evaluation/fer2013.csv")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Model
class ResNet18Emotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18Emotion, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = ResNet18Emotion(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        preds = torch.argmax(model(images), dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("resnet18_confusion_matrix.png")
plt.show()

# Save
torch.save(model.state_dict(), "emotion_model_resnet18.pth")
print("✅ Saved model as emotion_model_resnet18.pth")

# Export ONNX
dummy = torch.randn(1, 3, 48, 48).to(DEVICE)
torch.onnx.export(model, dummy, "emotion_model_resnet18.onnx", input_names=["input"], output_names=["output"], opset_version=11)
print("✅ Exported to ONNX: emotion_model_resnet18.onnx")
