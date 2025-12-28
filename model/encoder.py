from .conv_bn_relu import ConvBNReLU
from torch import nn
import torch
from tools import print_min_max

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        
        layers = [ConvBNReLU(3, config.conv_channels)]
        
        for _ in range(config.encoder_block - 1):
            layers.append(ConvBNReLU(config.conv_channels, config.conv_channels))
        
        self.conv_layers = nn.Sequential(*layers)
        self.merge_layer = ConvBNReLU(config.conv_channels + 3 + config.message_length, config.conv_channels)
        self.output_layer = nn.Conv2d(config.conv_channels, 3, kernel_size=1)
        
    def forward(self,image,message):
        H=image.size(2)
        W=image.size(3)
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1, H, W)
        #print_min_max(image, "input image")
        conv_image=self.conv_layers(image)
        #print_min_max(conv_image, "conv_image")
        merge_conved_image_message = torch.cat([conv_image,image,expanded_message], dim=1)
        #print_min_max(merge_conved_image_message, "merge_conved_image_message")
        merged = self.merge_layer(merge_conved_image_message)
        #print_min_max(merged, "merged")
        output = self.output_layer(merged)
        #print_min_max(output, "output before clamp")
        
        #output = torch.clamp(output, 0.0, 1.0)
        
        return output
        