import torch
from diffusion import DiffusionModel, beta_schedule
from unet import UNet, TemporalEncoding
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloader import CCDataset, read_data
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
from utils import gaussian_noise
import cv2

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # argparser = ArgumentParser()
    # argparser.add_argument("--iterations", default=2000, type=int)
    # argparser.add_argument("--batch-size", default=256, type=int)
    # argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    # argparser.add_argument("--load-trained", default=0, type=int, choices=(0, 1))
    # args = argparser.parse_args()

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    #nn_module = UNet(3, 128, (1, 2, 4, 8))


    #Useful constants
    num_timesteps = 1000
    batch_size = 64
    d_model = 64
    iterations = 2000
    epochs = 500

    generate = True
    channels = 1
    #Prepare data
    mean = [0.5] * channels
    std = [0.5] * channels
    transforms = transforms.Compose(
        [
            # transforms.RandomCrop((h,w)),
            transforms.ToPILImage(),
            #transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=90, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=180, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=270, translate=(0.0,0.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = CCDataset("./images", transform=transforms)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    #Calculate weights for sampler
    print("Calculating sampling weights...")
    weights = read_data("chars.txt")
    #sampler = WeightedRandomSampler(weights, len(training_data), replacement=True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)


    beta_schedule = beta_schedule(num_timesteps=num_timesteps, type="linear").to(device)
    model = DiffusionModel(betas=beta_schedule, out_channels=channels, channel_scales=(1, 2, 4, 8), d_model=d_model, device=device).to(device)
    temb_model = TemporalEncoding(timesteps=num_timesteps, d_model=d_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    num_batches = len(train_set) // batch_size + 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * num_batches)
    # Debug anomalies in computation graph
    torch.autograd.set_detect_anomaly(True)


    '''
    Training Loop:
    1. Sample initial training batch from data
    2. Sample batched timesteps from uniform distribution
    3. Sample initial noise \epsilon from Gaussian
    4. SGD on \nabla_\theta || \epsilon - \epsilon_\theta(\sqrt{\alpha_bar_{t}} + \sqrt{1 - \alpha_bar_{t}} \epsilon, t )||^2
    '''

    #Load image data
    min_loss = 10000
    best_model = None
    for i in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader, start=1):
            x = images.to(device)
            bs = x.shape[0]
            t_batched = torch.randint(0, num_timesteps, (bs,)).to(device) #Initial batch of timesteps


            optimizer.zero_grad()
            loss = model.loss(x, t_batched, temb_model)
            loss.backward()
            optimizer.step()
            scheduler.step()

            

            print(f"[Epoch {i+1}] \t Iteration {batch_idx+1}: loss = {loss.item()}")

        #Evaluation 
        model.eval()
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(test_dataloader, start=1):
            x = images.to(device)
            bs = x.shape[0]
            t_batched = torch.randint(0, num_timesteps, (bs,)).to(device) #Initial batch of timesteps

            val_loss = model.loss(x, t_batched, temb_model)
            running_loss += val_loss.item()

        avg_val_loss = running_loss / len(val_set)
        print("============Validation============")
        print(f"[Epoch {i+1}]: loss = {avg_val_loss}")
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            #save best model so far
            torch.save(model.state_dict(), "./checkpoints/best.pt")

    # Save model
    # torch.save(model.state_dict(), "./checkpoints/best_model.pt")
