from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

import torch.nn.functional as F
import torch.nn as nn
import torch

encoder_name='resnet101'
# encoder_name='timm-regnety_064'
in_channels = 3
encoder_depth = 5
encoder_weights = 'imagenet'
decoder_channels = (256, 128, 64, 32, 16)
decoder_use_batchnorm = True
decoder_attention_type = 'scse'
classes = 4
activation = None

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head1 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
    
    def forward(self,x):
        z = self.encoder(x)
        yhat1 = self.segmentation_head(self.decoder(*z))
        yhat2 = self.segmentation_head1(self.decoder1(*z))
        
        return yhat1, yhat2