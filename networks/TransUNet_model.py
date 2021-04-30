import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50

from .net_defs import PatchEmbed, TransformerLayer
from .scat_coeff_encoder import scatEncoder

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        resnet = resnet50(pretrained=True)
        resnet_blocks = list(resnet.children())
        self.resnet_features1 = nn.Sequential(*resnet_blocks[0:3])
        self.resnet_features2 = nn.Sequential(*resnet_blocks[3:5])
        self.resnet_features3 = nn.Sequential(*resnet_blocks[5]) 
        self.resnet_features4 = nn.Sequential(*resnet_blocks[6])

    def forward(self, x):
        x1 = self.resnet_features1(x)
        x2 = self.resnet_features2(x1)
        x3 = self.resnet_features3(x2)
        x  = self.resnet_features4(x3)
        return x, x1, x2, x3

class TransformerBlock(nn.Module):
    def __init__(self, cnn_feat_size=14, patch_size=1, num_layers=12, in_channels=1024, emb_dim=768):
        super(TransformerBlock, self).__init__() 
        self.patch_embedding = PatchEmbed(in_channels, cnn_feat_size, patch_size, emb_dim)
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(TransformerLayer(emb_dim))
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embedding(x)
        for transformer_layers in self.transformer_layers:
            x = transformer_layers(x)
        x = self.norm(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class CascadedUpsampler(nn.Module):

    def __init__(self, in_channels=768, num_classes=2, use_scat_encoder=False):
        super(CascadedUpsampler, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()

        self.use_scat_encoder = use_scat_encoder
        if self.use_scat_encoder:
            self.scat_encoder = scatEncoder()
            in_channels = in_channels * 2

        self.conv1 = nn.Conv2d(in_channels, 512, 3, 1, 1)

        self.cup_conv1 = nn.Conv2d(1024, 256, 3, 1, 1)
        self.cup_conv2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.cup_conv3 = nn.Conv2d(192, 64, 3, 1, 1)
        self.cup_conv4 = nn.Conv2d(64, 16, 3, 1, 1)

        # Segmentation head
        self.segmentationHead = SegmentationHead(16, num_classes)

    def forward(self, x, x1, x2, x3, scat_mat):
        
        # Reshape input with shape (n_patch, D) to H/P x W/P x D
        batch_size, num_patches, D = x.size()
        h = w = int(np.sqrt(num_patches))
        x = x.view(batch_size, h, w, D)
        x = x.permute(0, 3, 1, 2)

        if self.use_scat_encoder:
            scat_encoded_out = self.scat_encoder(scat_mat)
            x = torch.cat((x, scat_encoded_out), 1)

        x = self.conv1(x)
        x = self.relu(x)
        # cascade upsample portion
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.cup_conv1(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.cup_conv2(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.cup_conv3(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = self.cup_conv4(x)
        x = self.relu(x)

        x = self.segmentationHead(x)

        return x


class TransUNet(nn.Module):

    def __init__(self, num_classes, use_scat_encoder=False):
        super(TransUNet, self).__init__()

        # Resnet50 feature extraction for CUP and transformer block
        self.cnn = CNN()
        self.transformer_net = TransformerBlock(cnn_feat_size=14, patch_size=1, num_layers=12, emb_dim=768)
        self.cup = CascadedUpsampler(768, num_classes, use_scat_encoder)

    
    def forward(self, x, scat_mat=None):
        if x.shape[1] == 1: 
            x = x.repeat(1,3,1,1)
        x, x1, x2, x3 = self.cnn(x)
        x = self.transformer_net(x)
        x = self.cup(x, x1, x2, x3, scat_mat)
        return x