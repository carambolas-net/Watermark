import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from config import config
        
class LossFunction:
    def __init__(self, device=None):
        self.device = device if device is not None else config.device
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2
        # Create Gaussian kernel for SSIM computation
        self.register_gaussian_kernel()

    def register_gaussian_kernel(self, kernel_size=11, sigma=1.5):
        """Create a Gaussian kernel for SSIM computation"""
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        gaussian = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        kernel = gaussian.unsqueeze(1) @ gaussian.unsqueeze(0)
        kernel = kernel / kernel.sum()
        
        # Expand to 3 channels
        self.gaussian_kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.gaussian_kernel = self.gaussian_kernel.to(self.device)
    
    def ssim_loss(self, img1, img2):
        """
        Calculate SSIM (Structural Similarity Index) loss between two images
        Input: img1, img2 of shape (B, C, H, W), values in [0, 1] or [-1, 1]
        Output: 1 - SSIM (loss, lower is better)
        """
        # Ensure images are on the same device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Ensure 4D tensor
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Get padding value
        padding = self.gaussian_kernel.shape[-1] // 2
        
        # Compute mean values using Gaussian kernel
        mu1 = F.conv2d(img1, self.gaussian_kernel, padding=padding, groups=3)
        mu2 = F.conv2d(img2, self.gaussian_kernel, padding=padding, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variance and covariance using Gaussian kernel
        sigma1_sq = F.conv2d(img1 * img1, self.gaussian_kernel, padding=padding, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.gaussian_kernel, padding=padding, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.gaussian_kernel, padding=padding, groups=3) - mu1_mu2
        
        # Compute SSIM
        numerator1 = 2 * mu1_mu2 + self.c1
        denominator1 = mu1_sq + mu2_sq + self.c1
        numerator2 = 2 * sigma12 + self.c2
        denominator2 = sigma1_sq + sigma2_sq + self.c2
        
        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        # Return SSIM loss (1 - SSIM)
        return 1 - ssim_map.mean()

    def img_loss(self, img1, img2):
        return self.ssim_loss(img1, img2)*0.4 + nn.MSELoss()(img1, img2)*0.5
    
    def msg_loss(self, msg1, msg2):
        return nn.MSELoss()(msg1, msg2)