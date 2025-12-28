from .conv_bn_relu import ConvBNReLU
from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        
        layers = [ConvBNReLU(3, config.conv_channels)]

        for _ in range(config.decoder_block - 1):
            layers.append(ConvBNReLU(config.conv_channels, config.conv_channels))

        layers.append(ConvBNReLU(config.conv_channels, config.message_length))
        
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(config.message_length, config.message_length)
        
    def forward(self,image):
        x = self.layers(image)
        x.squeeze_(3).squeeze_(2)
        output = self.linear(x)
        #output = torch.sigmoid(output)
        return output
        