import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


class ImageLoader:
    def __init__(self, path="data", batch_size=32, img_size=64, shuffle=True):
        transform = transforms.Compose([transforms.Resize(img_size + 1),
                                transforms.CenterCrop(img_size),
                                 transforms.ToTensor()])
        dataset = datasets.ImageFolder(path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    def sample(self, vis=False):
        images = next(iter(self.dataloader))[0]
        
        if vis:
            plt.imshow(images[0].permute(1,2,0).numpy())
            plt.show()
        
        return images
    
    def getShape(self):
        images = next(iter(self.dataloader))[0]
        return images[0].shape