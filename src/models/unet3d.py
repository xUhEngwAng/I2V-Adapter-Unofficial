import torch
from einops import rearrange, repeat

from ..modules.attention import VideoTransformer
from ..modules.resnet import VideoResBlock
from ..modules.util import positional_emb

class DownBlock3D(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        pos_channels,
        block_depth,
        use_attention,
        kernel_size=[3, 1, 1],
        num_frames=8,
        context_channels=None
    ):
        super().__init__()
        self.num_frames = num_frames
        layers = []

        for ind in range(block_depth):
            modules = []
            in_channels = in_channels if ind == 0 else out_channels
            modules.append(VideoResBlock(in_channels, out_channels, pos_channels, kernel_size))
            
            if use_attention:
                modules.append(VideoTransformer(out_channels, context_channels))
                
            layers.append(torch.nn.ModuleList(modules))

        self.layers = torch.nn.ModuleList(layers)
        self.down_sampler = torch.nn.MaxPool2d(2)

    def forward(self, x, context, skips, timesteps, image_only_indicator):
        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1
        
        for modules in self.layers:
            for module in modules:
                if isinstance(module, VideoResBlock):
                    x = module(x, timesteps, frame_num, image_only_indicator)
                elif isinstance(module, VideoTransformer):
                    x = module(x, context, frame_num, image_only_indicator)
                else:
                    raise NotImplementedError

            skips.append(x)

        return self.down_sampler(x)

class UpBlock3D(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        pos_channels, 
        block_depth,
        use_attention,
        kernel_size=[3, 1, 1],
        num_frames=8,
        context_channels=None
    ):
        super().__init__()
        self.num_frames = num_frames
        layers = []

        for ind in range(block_depth):
            modules = []
            output_channels = out_channels if ind == block_depth-1 else in_channels // 2
            modules.append(VideoResBlock(in_channels, output_channels, pos_channels, kernel_size))
            
            if use_attention:
                modules.append(VideoTransformer(output_channels, context_channels))
                
            layers.append(torch.nn.ModuleList(modules))

        self.layers = torch.nn.ModuleList(layers)
        self.up_sampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, context, skips, timesteps, image_only_indicator):
        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1
            
        x = self.up_sampler(x)

        for modules in self.layers:
            x = torch.cat([x, skips.pop()], dim=1)
            
            for module in modules:
                if isinstance(module, VideoResBlock):
                    x = module(x, timesteps, frame_num, image_only_indicator)
                elif isinstance(module, VideoTransformer):
                    x = module(x, context, frame_num, image_only_indicator)
                else:
                    raise NotImplementedError
        
        return x

class UNet3D(torch.nn.Module):
    def __init__(
        self, 
        block_depth,
        widths,
        attention_levels,
        input_channels, 
        output_channels,
        device,
        kernel_size=[3, 1, 1],
        num_frames=8,
        num_groups=8, 
        pos_channels=256,
        context_channels=None,
        max_period=10000,
    ):
        super().__init__()
        assert(len(widths) == len(attention_levels))
        
        self.device = device
        self.pos_channels = pos_channels
        self.max_period = max_period
        self.num_frames = num_frames
        self.inc = torch.nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)

        down_layers = []
        bottleneck_layers = []
        up_layers = []

        for ind in range(len(widths)-1):
            use_attention = attention_levels[ind]
            down_layers.append(DownBlock3D(
                widths[ind], 
                widths[ind+1], 
                pos_channels, 
                block_depth, 
                use_attention, 
                num_frames=num_frames,
                context_channels=context_channels
            ))

        self.down_layers = torch.nn.ModuleList(down_layers)

        for _ in range(block_depth):
            use_attention = attention_levels[ind]
            bottleneck_layers.append(VideoResBlock(widths[-1], widths[-1], pos_channels, kernel_size))
            if use_attention:
                bottleneck_layers.append(VideoTransformer(widths[-1], context_channels))

        self.bottleneck_layers = torch.nn.ModuleList(bottleneck_layers)

        for ind in reversed(range(1, len(widths))):
            use_attention = attention_levels[ind-1]
            up_layers.append(UpBlock3D(
                widths[ind]*2, 
                widths[ind-1], 
                pos_channels, 
                block_depth, 
                use_attention, 
                num_frames=num_frames,
                context_channels=context_channels
            ))

        self.up_layers = torch.nn.ModuleList(up_layers)

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups, widths[0]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(widths[0], output_channels, kernel_size=1)
        )

    def forward(self, x, t, image_only_indicator, context=None):
        timesteps = positional_emb(t, self.pos_channels, self.max_period)
        x = self.inc(x)
        skips = []

        for downblock in self.down_layers:
            x = downblock(x, context, skips, timesteps, image_only_indicator)

        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1

        for layer in self.bottleneck_layers:
            if isinstance(layer, VideoResBlock):
                x = layer(x, timesteps, frame_num, image_only_indicator)
            elif isinstance(layer, VideoTransformer):
                x = layer(x, context, frame_num, image_only_indicator)
            else:
                raise NotImplementedError

        for upblock in self.up_layers:
            x = upblock(x, context, skips, timesteps, image_only_indicator)

        return self.out(x)

if __name__ == '__main__':
    block_depth = 3
    model_channels = 128
    channel_mults = [1, 2, 4]
    widths = [model_channels * mult for mult in channel_mults]
    attention_levels = [1, 1, 1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    out_channels = 4

    model = UNet3D(block_depth, widths, attention_levels, in_channels, out_channels, device).to(device)
    from torchinfo import summary

    num_frames = 8
    batch_size = 16
    latent_size = 16
    n_noise_steps = 1000
    
    batch_images = torch.randn([batch_size*num_frames, in_channels, latent_size, latent_size]).to(device)
    t = torch.randint(low=1, high=n_noise_steps, size=(batch_size*num_frames, 1)).to(device)
    image_only_indicator=torch.Tensor([False]).to(device)
    summary(model, input_data=[batch_images, t, image_only_indicator])
