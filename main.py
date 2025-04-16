import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Uniform
from torchvision import datasets, transforms

from discriminator import Discriminator
from generator import Generator
from img_loader import ImageLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 100
batch_size = 32
img_size = 32
data_loader = ImageLoader(batch_size=batch_size, img_size=img_size)

shape = data_loader.getShape()
flattened_shape = shape[0] * shape[1] * shape[2]
C, H, W = shape

generator = Generator(noise_dim=noise_dim, output_dim=flattened_shape).to(device)
discriminator = Discriminator(input_size=flattened_shape).to(device)

criterion = nn.BCELoss()

optim_generator = torch.optim.Adam(generator.parameters(), lr=3e-4)
optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=3e-4)

show_freq = 50
for i in range(10000000):
    for _ in range(1): 
        noise = torch.randn(batch_size, noise_dim, device=device)
        images = data_loader.sample().to(device)  

        with torch.no_grad():
            fake = generator(noise) 
            
        out_d_fake = discriminator(fake)
        out_d_real = discriminator(images)
        optim_discriminator.zero_grad()
        real_labels = torch.ones_like(out_d_real, device=device)
        fake_labels = torch.zeros_like(out_d_fake, device=device)
        dis_error = criterion(out_d_real, real_labels) + criterion(out_d_fake, fake_labels)
        dis_error.backward()
        optim_discriminator.step()
        
    noise = torch.randn(batch_size, noise_dim, device=device)
    fake = generator(noise) 
    out_d_fake = discriminator(fake)
    real_labels = torch.ones_like(out_d_fake, device=device)
    gen_error = criterion(out_d_fake, real_labels)
    optim_generator.zero_grad()
    gen_error.backward()
    optim_generator.step()

    fake_img = fake[0].detach().cpu() 
    img_np = fake_img.permute(1, 2, 0).numpy()  
    if i % show_freq == 0:
        plt.imshow(img_np)
        plt.show(block=False)
        plt.pause(0.01)
    print("Generator error: ", gen_error.item(), "Discriminator error: ", dis_error.item())