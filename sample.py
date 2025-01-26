import torch
from diffusion import DiffusionModel, beta_schedule
from unet import UNet, TemporalEncoding
from dataloader import CCDataset
import argparse
import matplotlib.pyplot as plt
from utils import gaussian_noise
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels = 1
num_timesteps = 1000
d_model = 64
w = 4
h = 1
fig = plt.figure(figsize=(8, 8))

beta_schedule = beta_schedule(num_timesteps=num_timesteps).to(device)
temb_model = TemporalEncoding(timesteps=num_timesteps, d_model=d_model).to(device)
loaded_model = DiffusionModel(betas=beta_schedule, out_channels=channels, channel_scales=(1, 2, 4, 8), d_model=d_model, device=device).to(device)
loaded_model.load_state_dict(torch.load("./checkpoints/best.pt"))
loaded_model.eval()

with torch.no_grad():
    # Draw 4 by 4 group of images
    for i in range(1, w * h + 1):
        #print(f"Generating image {i}...")
        img = loaded_model.generate(shape=(1, channels, 64, 64), noise_fn=gaussian_noise, temb_model=temb_model)
        #Normalize image to [0, 255]
        img = img + 1. / 2.
        #for i in range(1, w * h + 1):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        fig.add_subplot(h, w, i)
        plt.imshow(img)

    plt.savefig("./samples/sample.png")
