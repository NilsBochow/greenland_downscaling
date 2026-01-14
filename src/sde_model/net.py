import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ScoreUNet(nn.Module):
    def __init__(self,
                 marginal_prob_std,
                 resolution=64,
                 in_channels= 1,#3,
                 out_channels=1,
                 channels=(64, 64, 128, 128),
                 down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D"),
                 up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                 )):
        super().__init__()

        #self.physics_layer = PhysicsLayer()
        
        # Main UNet with modified input channels
        self.model = UNet2DModel(
            sample_size=resolution,
            in_channels=in_channels + 1,  # Original + ins + surf + ts
            out_channels=out_channels,
            block_out_channels=channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, insolation_map, surface_map):
        # Process conditioning maps
        insolation_map = insolation_map.unsqueeze(1).float()
        surface_map = surface_map.unsqueeze(1).float()
        
        net_input = torch.cat((x, surface_map), 1)  

        # Forward pass through UNet
        x = self.model(net_input, t).sample
        x = x / self.marginal_prob_std(t)[:, None, None, None]
        return x

