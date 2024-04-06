import torch
from diffusion import DiffusionModel, beta_schedule
from torchvision import datasets, transforms
from unet import UNet, TemporalEncoding
from torch.utils.data import DataLoader
from dataloader import CCDataset
import argparse
import matplotlib.pyplot as plt
from utils import gaussian_noise
import cv2
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels = 1
num_timesteps = 1000
d_model = 32
w = 11
h = 1
fig = plt.figure(figsize=(8, 8))

#Prepare data
mean = [0.5] * channels
std = [0.5] * channels
transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
training_data = CCDataset("./images", transform=transforms)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

beta_schedule = beta_schedule(num_timesteps=num_timesteps).to(device)
temb_model = TemporalEncoding(timesteps=num_timesteps, d_model=d_model).to(device)
loaded_model = DiffusionModel(betas=beta_schedule, out_channels=channels, channel_scales=(1, 2, 4, 8), d_model=d_model, device=device).to(device)
loaded_model.load_state_dict(torch.load("./checkpoints/best_model.pt"))
loaded_model.eval()

#Interpolate between 2 random characters
images, labels = next(iter(train_dataloader))
x1 = images[0:1, :, :, :].to(device)
x2 = images[1:, :, :, :].to(device)

with torch.no_grad():

    # Reconstruction
    time = num_timesteps - 1
    t_batched = (time * torch.ones(1)).type(torch.LongTensor).to(device)
    x_t = loaded_model.forward_sample(x1, t=t_batched)
    x_recon = x_t.clone()
    for t in tqdm(range(time, -1, -1)):
        times = (t * torch.ones(1)).type(torch.LongTensor).to(device)
        x_recon = loaded_model.backward_sample(x_recon, times, temb_model)

    #original
    img = x1
    img = img + 1. / 2.
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    #reconstruction
    img = x_recon
    img = img + 1. / 2.
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fig.add_subplot(1, 2, 2)
    plt.imshow(img)
    plt.savefig("./samples/recon_1.png")

    #Interpolation
    # for i in range(0, w * h):
    #     print(f"Interpolating images with alpha={0.1 * i}...")
    #     #img = loaded_model.generate(shape=(1, channels, 64, 64), noise_fn=gaussian_noise, temb_model=temb_model)
    #     if i > 0 and i < w * h - 1:
    #         img = loaded_model.interpolate(shape=(1, channels, 64, 64), x1=x1, x2=x2, t=num_timesteps//2, alpha=0.1*i, noise_fn=gaussian_noise, temb_model=temb_model)
    #     elif i == 0:
    #         img = x1
    #     else:
    #         img = x2
    #     #Normalize image to [0, 255]
    #     img = img + 1. / 2.
    #     img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     fig.add_subplot(h, w, i+1)
    #     plt.imshow(img)
    #
    # plt.savefig("./samples/interpolation_1.png")
