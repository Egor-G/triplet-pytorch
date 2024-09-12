import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TripletNetwork(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=64, embedding_dim=256):
        '''
            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.backbone = models.__dict__[backbone](weights='DEFAULT', progress=True)

        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_features = list(self.backbone.modules())[-1].out_features

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        self.embedding_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, img):

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat = self.backbone(img)
        class_pred = self.cls_head(feat)
        embedding = self.embedding_head(feat)
        return class_pred, embedding

