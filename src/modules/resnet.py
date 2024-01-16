import torch
from einops import rearrange
from typing import Iterable

from .util import AlphaBlender

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/diffusionmodules/util.py#L279
    """
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
        pos_channels,
        mid_channels=None,
        dims=2,
        kernel_size=3,
        group_nums=8
    ):
        super().__init__()
        
        self.dims = dims
        
        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = torch.nn.Sequential(
            conv_nd(dims, in_channels, mid_channels, kernel_size, padding=padding, bias=False),
            torch.nn.GroupNorm(group_nums, mid_channels),
            torch.nn.GELU()
        )

        self.conv2 = torch.nn.Sequential(
            conv_nd(dims, mid_channels, out_channels, kernel_size, padding=padding, bias=False),
            torch.nn.GroupNorm(group_nums, out_channels),
            torch.nn.GELU()
        )

        self.res_conv = conv_nd(dims, in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()

        self.emb_layer = torch.nn.Sequential(
            torch.nn.Linear(pos_channels, pos_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(pos_channels, mid_channels)
        )

    def forward(self, x, timesteps):
        h = self.conv1(x)
        emb_out = self.emb_layer(timesteps)[..., None, None]

        if self.dims == 3:
            emb_out = rearrange(emb_out, "b t c ... -> b c t ...")

        h = h + emb_out
        h = self.conv2(h)
        return h + self.res_conv(x)

class VideoResBlock(ResBlock):
    def __init__(
        self,
        in_channels, 
        out_channels,
        pos_channels,
        video_kernel_size,
        mid_channels=None,
        group_nums=8,
        merge_factor=0.5
    ):
        super().__init__(
            in_channels, 
            out_channels, 
            pos_channels,
            mid_channels=mid_channels
        )

        self.time_stack = ResBlock(
            out_channels,
            out_channels,
            pos_channels,
            dims=3,
            kernel_size=video_kernel_size,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy="learned_with_images",
            rearrange_pattern="b t -> b 1 t 1 1",
        )

    def forward(self, x, timesteps, num_video_frames, image_only_indicator):
        x = super().forward(x, timesteps)
        x_spatial = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x_temporal = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x_temporal = self.time_stack(
            x_temporal, rearrange(timesteps, "(b t) ... -> b t ...", t=num_video_frames)
        )
        output = self.time_mixer(
            x_spatial=x_spatial, x_temporal=x_temporal, image_only_indicator=image_only_indicator
        )
        output = rearrange(output, "b c t h w -> (b t) c h w")
        return output
