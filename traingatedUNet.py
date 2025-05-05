import torch
from torch.utils.data import DataLoader
from gatedUNet import LungSegmentationNet3D
from loss import BCEDiceLoss
from preparedata import data_loader
import os
import matplotlib.pyplot as plt
import numpy as np
trainloader,valloader = data_loader()
# Model, loss function, optimizer
model = LungSegmentationNet3D().cuda()
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load the dataset

validation_loss = []
training_loss = []
train_loader = trainloader
val_loader = valloader


print(train_loader)

# Training function
def train_model(model, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            print(images.shape)
            images = images.cuda()
            masks = masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')
        model_dir = 'D:\\project2\\archive\\checkpoints'
        model_path = os.path.join(model_dir, "model"+str(epoch+1)+".pth")
        torch.save(model.state_dict(), model_path)
        validation_loss.append(val_loss)
        training_loss.append(train_loss)
# Train the model
train_model(model, criterion, optimizer, num_epochs=100)
model_dir = 'D:\\project2\\archive\\checkpoints'
model_path = os.path.join(model_dir, "modelFinal"+".pth")
torch.save(model.state_dict(), model_path)
x = np.linspace(1, 100, 100)
training_loss=np.array(training_loss)
validation_loss =np.array(validation_loss)
# Assign variables to the y axis part of the curve
training_loss=np.where(training_loss>10.0,10,training_loss)
validation_loss=np.where(validation_loss>10.0,10,validation_loss)
# Plotting both the curves simultaneously
plt.plot(x, training_loss, color='r', label='training loss')
plt.plot(x, validation_loss, color='g', label='validation loss')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training and validation loss for lung segmentation")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
