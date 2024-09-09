import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TripletNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
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

    def forward(self, img):

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat = self.backbone(img)
        return feat

