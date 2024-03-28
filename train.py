import torch
from diffusion import DiffusionModel, beta_schedule
from unet import UNet, TemporalEncoding
from torch.utils.data import DataLoader
from dataloader import CCDataset
import torch.optim as optim
from torchvision import datasets, transforms

if __name__ == "__main__":

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
    d_model = 128

    #Prepare data
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transforms = transforms.Compose(
        [
            # transforms.RandomCrop((h,w)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees=90, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=180, translate=(0.0,0.0)),
            # transforms.RandomAffine(degrees=270, translate=(0.0,0.0)),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std),
        ]
    )
    training_data = CCDataset("./images", transform=transforms)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    beta_schedule = beta_schedule(num_timesteps=num_timesteps)
    model = DiffusionModel(betas=beta_schedule)
    temb_model = TemporalEncoding(timesteps=num_timesteps, d_model=d_model)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

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
    train_images, train_labels = next(iter(train_dataloader))
    #print(train_images)
    t_batched = torch.randint(0, num_timesteps, (batch_size,)) #Initial batch of timesteps


    optimizer.zero_grad()
    loss = model.loss(train_images, t_batched, temb_model)

    print(loss.item())
    loss.backward()
    optimizer.step()




    '''
    Sampling
    '''
