import matplotlib.pyplot as plt
from preparedata import data_loader
from gatedUNet import LungSegmentationNet3D
import torch
test_loader = data_loader()
path = r"D:\project2\archive\checkpoints\Model1Fully\model.pth"
model = LungSegmentationNet3D().cuda()
model.load_state_dict(torch.load(path))
model.eval()
def visualize_sample(image, mask, prediction, slice_idx,c):
    print("mask",mask.shape)
    print("image",image.shape)
    print("pred",prediction.shape)
    image = image[0, 0, slice_idx,:,:].cpu().numpy()  # Extract a slice
    mask = mask[0, 0, slice_idx,:,:].cpu().numpy()
    prediction = prediction[0, 0, slice_idx,:,:].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image"+str(c))
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth"+str(c))
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Prediction"+str(c))
    plt.imshow(prediction, cmap="gray")

    plt.show()

# Visualize the first sample
c = 0
for images, masks in test_loader:
    images = images.cuda()
    masks = masks.cuda() 
    outputs = model(images)
    preds = (outputs > 0.5).float()
    for i in range(9,16):
        visualize_sample(images, masks, preds, i,c)
    c = c+1 