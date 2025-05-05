import torch
from torch.utils.data import DataLoader
from gatedUNet import LungSegmentationNet3D
from gatedUnetParameters import dice_score,pixel_accuracy,precision,recall,iou
import numpy as np
from preparedata import data_loader
from gatedUNet import LungSegmentationNet3D
import matplotlib as plt
import numpy as np
test_loader = data_loader()
path = "D:\\project2\\archive\\checkpoints\\model.pth"
model = LungSegmentationNet3D().cuda()
model.load_state_dict(torch.load(path))
model.eval()
def evaluate_model(model, test_loader):
    model.eval()
    dice_scores = []
    pixel_accuracies = []
    precisions = []
    recalls = []
    ious = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.cuda()
            masks = masks.cuda()

            # Get model predictions
            outputs = model(images)
            preds = (outputs > 0.5).float()  # Threshold to get binary predictions

            # Calculate metrics
            dice_scores.append(dice_score(preds, masks))
            pixel_accuracies.append(pixel_accuracy(preds, masks))
            precisions.append(precision(preds, masks))
            recalls.append(recall(preds, masks))
            ious.append(iou(preds, masks))

    # Average metrics over the test set
    results = {
        'Dice Score': np.mean(dice_scores),
        'Pixel Accuracy': np.mean(pixel_accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'IoU': np.mean(ious),
        
    }
    x = np.arange(0,len(dice_scores))
    plt.plot(x,dice_scores)
    plt.show()
    return results

# Evaluate
results = evaluate_model(model, test_loader)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
