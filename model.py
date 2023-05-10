import torch
import torch.nn as nn
from body_features import BodyEncoder
from context import Context
from facial_features import FacialFeatures

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        # Body encoder
        self.body_encoder = BodyEncoder()

        # Context path
        self.context = Context()
        
        #Facial Features
        self.facial_features = FacialFeatures()
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Pass input image through body encoder
        body_features = self.body_encoder(x)
        
        # Pass body features and input image through context path
        context_features = self.context(x)
        
        # Pass facial features and input image through facial path
        facial_features = self.facial_feature_encoder(x)
        # Upsample body features to match size of context features
        body_features = nn.functional.interpolate(body_features, size=context_features.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate body and context features along the channel dimension
        fused_features = torch.cat([body_features, context_features,facial_features], dim=1)
        
        # Pass fused features through fusion layer
        fused_features = self.fusion(fused_features)
        
        # Pass fused features through output layer
        output = self.output_layer(fused_features)
        
        return output
