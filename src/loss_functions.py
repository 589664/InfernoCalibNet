import torch
import torch.nn as nn


class DynamicWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        """
        Initialize the dynamic weighted BCE loss.

        Args:
        - pos_weight: (Optional) Global class weight for the positive class. If provided,
        this will be used for weighting positive samples.
        """
        super(DynamicWeightedBCELoss, self).__init__()
        self.pos_weight = (
            pos_weight  # This will allow the option to use global class weights
        )

    def forward(self, output, target, weights=None):
        """
        Forward pass to compute the weighted binary cross-entropy loss.

        Args:
        - output: Model output (logits).
        - target: Ground truth labels (binary).
        - weights: (Optional) Per-batch dynamic weights for positive and negative classes.

        Returns:
        - Loss: The computed weighted BCE loss.
        """
        output = output.clamp(min=1e-5, max=1 - 1e-5)  # Avoid log(0)
        target = target.float()

        if weights is not None:
            assert (
                len(weights) == 2
            )  # Ensure two weights are provided (positive, negative)
            beta_P, beta_N = weights
        else:
            # If no dynamic weights provided, use global pos_weight if available
            if self.pos_weight is not None:
                beta_P = self.pos_weight
                beta_N = 1.0  # Implicitly treat negative class weight as 1
            else:
                # If neither dynamic weights nor global pos_weight is provided, use equal weighting
                beta_P = beta_N = 1.0

        # Compute the weighted loss
        loss_pos = -beta_P * (target * torch.log(output))
        loss_neg = -beta_N * ((1 - target) * torch.log(1 - output))
        loss = loss_pos + loss_neg

        return loss.mean()  # Return mean loss over the batch
