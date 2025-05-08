import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, datasets
from model import CharCNN
import torch.nn.functional as F
from torch import nn, optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10

### Custom MNIST Loader from .npz
class MNISTFromNPZ(Dataset):
    def __init__(self, npz_path, train=True, transform=None):
        data = np.load(npz_path)
        self.transform = transform
        if train:
            self.images = data['x_train']
            self.labels = data['y_train']
        else:
            self.images = data['x_test']
            self.labels = data['y_test']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = int(self.labels[idx])
        image = transforms.ToPILImage()(image.reshape(28, 28))
        if self.transform:
            image = self.transform(image)
        return image, label

### Combined Datasets
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Load datasets
mnist_train = MNISTFromNPZ("dataset/mnist.npz", train=True, transform=transform)
mnist_test = MNISTFromNPZ("dataset/mnist.npz", train=False, transform=transform)

symbols_train = datasets.ImageFolder(root="dataset/symbols", transform=transform)
symbols_classes = symbols_train.classes  # For decoding

# Map symbol labels to integers after MNIST classes
offset = 10
symbols_train.targets = [t + offset for t in symbols_train.targets]
num_classes = 10 + len(symbols_classes)

# Combine datasets
train_dataset = ConcatDataset([mnist_train, symbols_train])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = CharCNN(num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Save
torch.save(model.state_dict(), "char_cnn.pth")
print("Model saved.")
