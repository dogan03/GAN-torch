import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.c = 3  
        self.h = int((output_dim // self.c) ** 0.5)
        self.w = self.h
        
        self.init_size = self.h // 8
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim, 128 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, self.c, 3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img