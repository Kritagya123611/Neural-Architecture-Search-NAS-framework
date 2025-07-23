# train.py
# Ye file ChildNet ko train karegi given a genotype and return karegi accuracy as reward

import torch
import torch.nn as nn
import torch.optim as optim
from child import ChildNet
from searchSpace import decode
from dataset import loaders  

# Ye controller se aayega later
tokens = [0, 4, 6, 1]  

# Genotype decode karo
genotype = decode(tokens)

# Model banao dynamically
model = ChildNet(genotype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# MNIST data load karo
train_loader, val_loader = loaders(batch_size=64)

# Loss function & optimizer define karo
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for few epochs (2-3)
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop + Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Accuracy calculate karo (reward)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
