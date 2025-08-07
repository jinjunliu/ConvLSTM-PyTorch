import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    """
    Calculates L1 loss of spatial gradients between predictions and targets.
    This helps the model learn clearer edges and details, mitigating blur issues.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Use L1 loss to calculate gradient differences as it's less sensitive to outliers
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, pred, target):
        """
        Forward pass calculation.
        Args:
            pred (torch.Tensor): Model prediction output, shape [B, C, H, W].
            target (torch.Tensor): Ground truth label data, shape [B, C, H, W].
        Returns:
            torch.Tensor: Calculated gradient loss value.
        """
        # Calculate gradients in y direction
        # pred[:, :, 1:, :] takes data from the 2nd row to the last row
        # pred[:, :, :-1, :] takes data from the 1st row to the second-to-last row
        # Their difference gives the gradient (difference between adjacent pixels) in the y direction
        gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Calculate gradients in x direction
        gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gx_target = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Calculate L1 loss of gradient differences
        loss_gy = self.loss(gy_pred, gy_target)
        loss_gx = self.loss(gx_pred, gx_target)
        
        # Sum the gradient losses in x and y directions
        return loss_gy + loss_gx


class HybridLoss(nn.Module):
    """
    Combines MSE loss and gradient loss as a composite loss function.
    L_total = L_MSE + lambda_grad * L_gradient
    """
    def __init__(self, lambda_grad=0.1):
        """
        Initialize the composite loss function.
        Args:
            lambda_grad (float): Weight coefficient for the gradient loss term. This is a hyperparameter that needs tuning.
        """
        super(HybridLoss, self).__init__()
        if lambda_grad < 0:
            raise ValueError("lambda_grad must be non-negative.")
            
        self.lambda_grad = lambda_grad
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.gradient_loss = GradientLoss()
        
        print(f"HybridLoss initialized with lambda_grad = {self.lambda_grad}")

    def forward(self, pred, target):
        """
        Forward pass calculation.
        Args:
            pred (torch.Tensor): Model prediction output, shape [B, C, H, W].
            target (torch.Tensor): Ground truth label data, shape [B, C, H, W].
        Returns:
            torch.Tensor: Calculated total loss value.
        """
        # Calculate MSE loss
        loss_mse = self.mse_loss(pred, target)
        
        # If lambda_grad is 0, skip gradient loss calculation to save computation
        if self.lambda_grad == 0:
            return loss_mse
            
        # Calculate gradient loss
        loss_grad = self.gradient_loss(pred, target)
        
        # Combine both losses
        total_loss = loss_mse + self.lambda_grad * loss_grad
        
        return total_loss

# --- Usage Example ---
# This code only executes when running this file directly, used for demonstration and testing
if __name__ == '__main__':
    # Create some dummy data
    # B=2, C=1, H=64, W=64
    batch_size = 2
    height, width = 64, 64
    
    # Assume a smooth prediction result (e.g., a Gaussian blurred image)
    prediction = torch.randn(batch_size, 1, height, width).sigmoid()

    # Create a real target with sharp edges
    target = torch.zeros(batch_size, 1, height, width)
    target[:, :, 16:48, 16:48] = 1.0  # Create a square in the middle

    print("Testing HybridLoss functionality:")
    print(f"Input tensor shape: {prediction.shape}")
    print("-" * 30)

    # 1. Test lambda_grad = 0 case (equivalent to pure MSE)
    print("Case 1: lambda_grad = 0 (Pure MSE Loss)")
    loss_fn_mse_only = HybridLoss(lambda_grad=0)
    loss_val_mse_only = loss_fn_mse_only(prediction, target)
    # Manually calculate pure MSE for verification
    manual_mse = nn.MSELoss()(prediction, target)
    print(f"  HybridLoss result: {loss_val_mse_only.item():.6f}")
    print(f"  Manual MSE result: {manual_mse.item():.6f}")
    print("-" * 30)

    # 2. Test lambda_grad > 0 case
    lambda_val = 0.5
    print(f"Case 2: lambda_grad = {lambda_val}")
    loss_fn_hybrid = HybridLoss(lambda_grad=lambda_val)
    loss_val_hybrid = loss_fn_hybrid(prediction, target)
    
    # Manual calculation for verification
    manual_mse = nn.MSELoss()(prediction, target)
    grad_loss_fn = GradientLoss()
    manual_grad = grad_loss_fn(prediction, target)
    manual_total = manual_mse + lambda_val * manual_grad
    
    print(f"  HybridLoss result: {loss_val_hybrid.item():.6f}")
    print(f"  Manual step-by-step result: {manual_total.item():.6f}")
    print(f"    - MSE part:     {manual_mse.item():.6f}")
    print(f"    - Gradient part: {manual_grad.item():.6f}")
    print("-" * 30)