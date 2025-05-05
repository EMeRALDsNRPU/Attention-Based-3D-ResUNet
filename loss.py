import torch
import torch.nn as nn

class WeightedDiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.8, weight_bce=0.2, smooth=1e-3):
        """
        Weighted Dice + BCE Loss with Automatic Class Weight Calculation.

        Args:
            weight_dice: Weight for Dice Loss contribution.
            weight_bce: Weight for BCE Loss contribution.
            smooth: Smoothing factor to avoid division by zero.
        """
        super(WeightedDiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # No reduction to apply class weights manually

    def forward(self, preds, targets):
        """
        Computes Weighted Dice + BCE Loss for 3D images with automatic class weight handling.

        Args:
            preds: Predicted logits (batch, channels, depth, height, width)
            targets: Ground truth tensor (batch, channels, depth, height, width)

        Returns:
            Weighted Dice + BCE Loss value.
        """
        # Convert logits to probabilities

        # Flatten spatial dimensions (D, H, W) while keeping batch & class channels
        preds = preds.view(preds.shape[0], preds.shape[1], -1)  # (B, C, D*H*W)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)  # (B, C, D*H*W)

        # Compute class weights dynamically
        class_weights = self.compute_class_weights(targets).to(preds.device).float()  # Ensure float tensor

        # --- Weighted Dice Loss ---
        intersection = torch.sum(preds * targets, dim=2)  # Sum over spatial dims
        union = torch.sum(preds, dim=2) + torch.sum(targets, dim=2)  # Sum over spatial dims
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # Dice loss per batch & class

        # Apply per-class weights
        dice_loss *= class_weights.view(1, -1)  # Ensure correct shape
        dice_loss = dice_loss.mean()  # Average over batch & classes

        # --- Weighted BCE Loss ---
        bce_loss = self.bce(preds, targets)  # Compute BCE loss per voxel
        bce_loss *= class_weights.view(1, -1, 1)  # Apply class weights (keep spatial dims)
        bce_loss = bce_loss.mean()  # Average BCE loss

        # Combine Dice and BCE losses
        total_loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss
        return total_loss

    def compute_class_weights(self, targets):
        """
        Computes class weights dynamically based on target label distribution.

        Args:
            targets: Ground truth tensor (batch, channels, depth, height, width)

        Returns:
            class_weights: Tensor of shape (C,) containing weights for each class.
        """
        with torch.no_grad():
            num_classes = targets.shape[1]  # Number of channels (1 for binary segmentation)
            total_voxels = targets.numel()  # Total number of voxels
            class_weights = torch.zeros(num_classes, device=targets.device)

            for c in range(num_classes):
                class_voxels = targets[:, c].sum()  # Foreground voxel count for class c
                bg_voxels = total_voxels - class_voxels  # Background voxel count
                
                # Compute inverse frequency weight
                if class_voxels > 0:
                    class_weights[c] = min(bg_voxels / (class_voxels + self.smooth), 10.0)
                    # class_weights[c] = bg_voxels / (class_voxels + self.smooth)
                else:
                    class_weights[c] = 1.0  # Default to 1 if no foreground pixels

            return class_weights.float()  # Ensure floating point

# Example Usage:
# preds = torch.randn((2, 1, 128, 128, 128))  # Batch=2, Channels=1, 3D Volume
# targets = torch.randint(0, 2, (2, 1, 128, 128, 128)).float()  # Binary labels
# loss_fn = WeightedDiceBCELoss(weight_dice=0.7, weight_bce=0.3)
# loss = loss_fn(preds, targets)
# loss.backward()  # Ensure gradients flow correctly
# print(loss)
