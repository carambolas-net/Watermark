import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from config import config
        

import torch
import torch.nn as nn
import torchvision

class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg_loss = nn.Sequential(*layers)

    def forward(self, img):
        return self.vgg_loss(img)


class LossFunction:
    def __init__(self, device=None):
        self.device = device if device is not None else config.device
        self.vgg_loss = VGGLoss(3,1,False).to(self.device)
        self.mse_loss = nn.MSELoss().to(device)
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)

    def img_loss(self, img1, img2):
        vgg_value1 = self.vgg_loss(img1) 
        vgg_value2 = self.vgg_loss(img2)
        return self.mse_loss(vgg_value1, vgg_value2)
        # return self.mse_loss(img1, img2)
        
    def msg_loss(self, msg1, msg2):
        return nn.MSELoss()(msg1, msg2)

    def adv_loss(self, pred, target_label):
        """对抗损失：BCE with logits"""
        return self.bce_with_logits_loss(pred, target_label)