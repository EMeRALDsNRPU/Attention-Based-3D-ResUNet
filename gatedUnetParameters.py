
def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def precision(pred, target):
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    return tp / (tp + fp + 1e-6)

def recall(pred, target):
    tp = (pred * target).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp / (tp + fn + 1e-6)

def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection
    return (intersection + smooth) / (union + smooth)
