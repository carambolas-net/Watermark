from .conv_bn_relu import ConvBNReLU
from torch import nn
import torch

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config):
        super(Discriminator, self).__init__()

        layers = [ConvBNReLU(3, config.conv_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNReLU(config.conv_channels, config.conv_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.conv_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        return X