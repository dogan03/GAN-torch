import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.c = 3 
        self.h = int((input_size // self.c) ** 0.5)
        self.w = self.h
        
        self.model = nn.Sequential(
            nn.Conv2d(self.c, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.ds_size = self.h // 16
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.ds_size * self.ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output