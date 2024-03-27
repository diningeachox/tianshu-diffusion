import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
TemporalEncoding (based on positional encoding from Transformers)
'''
class TemporalEncoding(nn.Module):
    def __init__(self, timesteps, d_model):
        """
        Inputs
            timesteps - a training batch of timesteps
            d_model - Hidden dimensionality of the input.
        """
        super().__init__()



        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', self.calc_embedding(timesteps, d_model), persistent=False)

    def forward(self, x):
        #x = x + self.pe[:, :x.size(1)]
        #return x
        return self.pe[:, :x.size(1)]

    @staticmethod
    def calc_embedding(timesteps, d_model):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(timesteps.shape[0], d_model)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        positions = timesteps.unsqueeze(1) #The positions are the timesteps (analogous to a text with max_len = t_max)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        return pe # Shape = [timesteps.shape[0], d_model]

'''
AttentionBlocks
Similar to attention blocks in the Transformer model
'''
class AttentionBlock(nn.Module):
    def __init__(self, d):
        # x has shape (B, C, H, W)
        #d = x.shape[1]
        #Query
        self.qw = nn.Parameter(torch.randn(d, d))
        self.qb = nn.Parameter(torch.randn(d))
        #Key
        self.kw = nn.Parameter(torch.randn(d, d))
        self.kb = nn.Parameter(torch.randn(d))
        #Value
        self.vw = nn.Parameter(torch.randn(d, d))
        self.vb = nn.Parameter(torch.randn(d))
        #Proj_out
        self.proj_w = nn.Parameter(torch.randn(d, d))
        self.proj_b = nn.Parameter(torch.randn(d))

    def forward(self, x):
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        x = torch.permute(x, (0, 2, 3, 1)) # permute to shape (B, H, W, C) for tensordot
        q = torch.tensordot(x, self.qw, dims=1)
        q = q + self.qb #Broadcasting sum
        k = torch.tensordot(x, self.kw, dims=1)
        k = k + self.kb
        v = torch.tensordot(x, self.vw, dims=1)
        v = v + self.vb
        #Compute attention scores

        #Attention weights
        w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (d ** (-0.5)) #q dot k / sqrt(d)
        w = w.view(B, H, W, H * W)
        w = nn.Softmax()(w)
        w = w.view(B, H, W, H, W)

        h = torch.einsum('bhwHW,bHWc->bhwc', w, v)

        #Projection
        h = torch.tensordot(h, self.proj_w, dims=1)
        h = h + self.proj_b

        return x + h #Skip connection

'''
Residual blocks
Conv3x3
ReLU
Conv3x3
ReLU

Model weights are indexed by a time parameter t
'''
class ResidualBlock(nn.Module):
    def __init__(self, d_model, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.conv_shortcut = nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

        #Timestep embedding
        self.time_embedding_proj = nn.Sequential(
            #nn.Linear(time_c, time_c),
            nn.SiLU(), #Swish activation function
            nn.Linear(d_model, out_channels),
        )

    def forward(self, x, temb):
        out = self.norm1(x)
        out = self.conv1(out)
        out = F.relu(out)

        #Add timestep embedding
        out += self.time_embedding_proj(emb)

        out = self.norm2(x)
        out = self.conv2(out)
        out = F.relu(out)

        #Skip connection
        x = self.conv_shortcut(x)

        #out = nn.MaxPool2D(kernel_size=2)
        return out + x



'''
Backbone model for the DDPM model (U-Net)
'''

class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, n_res_blocks, attention_res, channel_scales=(1, 2, 4, 8), d_model=32):
        super().__init__()
        self.temb = TemporalEncoding()
        self.channels = n_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(n_channels, n_channels * 4)
        self.fc1 = nn.Linear(n_channels * 4, n_channels * 4)

        all_dims = (d_model, *[d_model * s for s in dim_scales])

        #Downsample blocks
        self.downsample = nn.ModuleList()

        self.downsample.append(nn.Conv2D(n_channels * 4, n_channels, kernel_size=3, stride=1, bias=True))
        for idx, (in_c, out_c) in enumerate(zip(
            all_dims[:-1],
            all_dims[1:],
        )):
        #for i in range(len(channel_scales)):
            for block in range(n_res_blocks):
                in_ch = in_c if block == 0 else out_c
                self.downsample.append(ResidualBlock(d_model, in_ch, out_ch))

            if idx != len(channel_scales) - 1:
                self.downsample.append(nn.Conv2D(in_c, out_c, kernel_size=3, stride=2, bias=False))

        #Middle
        self.middle = nn.ModuleList([
            ResidualBlock(d_model, all_dims[-1], all_dims[-1]),
            AttentionBlock(d_model),
            ResidualBlock(d_model, all_dims[-1], all_dims[-1])
        ])


        #Upsample blocks
        self.upsample = nn.ModuleList()
        for idx, (in_c, out_c, skip_c) in enumerate(zip(
            all_dims[::-1][:-1],
            all_dims[::-1][1:],
            all_dims[:-1][::-1],
        )):
            for block in range(n_res_blocks + 1):
                in_ch = in_c + skip_c if block == 0 else out_c
                self.upsample.append(ResidualBlock(d_model, in_ch, out_ch))

            if idx != len(channel_scales) - 1:
                self.upsample.append(nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2))

        self.end = nn.ModuleList(
                nn.GroupNorm(num_groups=32, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2D(all_dims[-1], out_channels, kernel_size=3, stride=2, bias=False)
        )


    def forward(self, x, t):
        B = x.shape[0]
        skip_connections = []

        # Timestep Embedding
        temb = self.temb(t, self.channels)
        temb = self.fc0(temb)
        temb = self.fc1(temb)
        assert temb.shape == [B, self.channels * 4]

        # Downsampling
        for block in self.downsample:
            if isinstance(block, ResidualBlock):
                x = block(x, temb)
                skip_connections.append(x) #Store these layers for skip connections
            else:
                x = block(x)

        # Middle
        for block in self.middle:
            if isinstance(block, ResidualBlock):
                x = block(x, temb)
            else:
                x = block(x)

        # Upsampling
        for block in self.upsample:
            if isinstance(block, ResidualBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, temb)
            else:
                x = block(x)

        # End (readout)
        x = self.end(x)

if __name__ == "__main__":
    model = UNet(n_channels, out_channels, n_res_blocks, attention_res, d_model=32)
