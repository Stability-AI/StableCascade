import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt

def download_image(url):
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def show_images(images, **kwargs):
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]    
    plt.figure(figsize=(kwargs.get("width", 32), kwargs.get("height", 32)))
    plt.axis("off")
    plt.imshow(torch.cat([torch.cat([i for i in images.clamp(0, 1)], dim=-1)], dim=-2).permute(1, 2, 0).cpu(), cmap='Greys')
    plt.show()