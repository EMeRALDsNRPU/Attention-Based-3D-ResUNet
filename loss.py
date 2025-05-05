import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
        dice_score = (2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2).sum(dim=2) + 
                                                          target.sum(dim=2).sum(dim=2).sum(dim=2) + self.smooth)
        return 1 - dice_score.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss + dice_loss

