from .encoder import Encoder
from .decoder import Decoder
import torch.nn as nn
from torchvision import transforms
from .noisy import Noisy

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.noisy = Noisy(config)
    
    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noisy_image = self.noisy(encoded_image)
        decoded_message = self.decoder(noisy_image)
        return encoded_image,noisy_image, decoded_message