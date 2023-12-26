import torch
from einops import rearrange

from unet3d import BasicTransformerBlock, ResBlock, positional_emb

class DownBlock(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        pos_channels,
        block_depth,
        use_attention,
        context_channels=None
    ):
        super().__init__()
        layers = []

        for ind in range(block_depth):
            modules = []
            in_channels = in_channels if ind == 0 else out_channels
            modules.append(ResBlock(in_channels, out_channels, pos_channels))
            
            if use_attention:
                modules.append(BasicTransformerBlock(out_channels, context_channels))
                
            layers.append(torch.nn.ModuleList(modules))

        self.layers = torch.nn.ModuleList(layers)
        self.down_sampler = torch.nn.MaxPool2d(2)

    def forward(self, x, skips, timesteps, context=None):
        _, _, h, w = x.shape
        
        for modules in self.layers:
            for module in modules:
                if isinstance(module, ResBlock):
                    x = module(x, timesteps)
                elif isinstance(module, BasicTransformerBlock):
                    x = rearrange(x, "b c h w -> b (h w) c")
                    x = module(x, context=context)
                    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
                else:
                    raise NotImplementedError

            skips.append(x)
            
        return self.down_sampler(x)

class UpBlock(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        pos_channels, 
        block_depth,
        use_attention,
        context_channels=None
    ):
        super().__init__()
        layers = []

        for ind in range(block_depth):
            modules = []
            output_channels = out_channels if ind == block_depth-1 else in_channels // 2
            modules.append(ResBlock(in_channels, output_channels, pos_channels))
            
            if use_attention:
                modules.append(BasicTransformerBlock(output_channels, context_channels))
                
            layers.append(torch.nn.ModuleList(modules))

        self.layers = torch.nn.ModuleList(layers)
        self.up_sampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skips, timesteps, context=None):
        x = self.up_sampler(x)
        _, _, h, w = x.shape

        for modules in self.layers:
            x = torch.cat([x, skips.pop()], dim=1)
            
            for module in modules:
                if isinstance(module, ResBlock):
                    x = module(x, timesteps)
                elif isinstance(module, BasicTransformerBlock):
                    x = rearrange(x, "b c h w -> b (h w) c")
                    x = module(x, context=context)
                    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
                else:
                    raise NotImplementedError

        return x

class UNet(torch.nn.Module):
    def __init__(
        self,
        block_depth,
        widths,
        attention_levels,
        input_channels, 
        output_channels,
        device,
        pos_channels=512,
        context_channels=None,
        max_period=10000
    ):
        super().__init__()
        assert(len(widths) == len(attention_levels))

        self.device = device
        self.pos_channels = pos_channels
        self.max_period = max_period
        self.inc = torch.nn.Conv2d(input_channels, widths[0], kernel_size=3, padding=1)
        self.channels = input_channels
        self.out_dim = output_channels
        self.self_condition = False
        self.random_or_learned_sinusoidal_cond = False
        
        down_layers = []
        bottleneck_layers = []
        up_layers = []

        for ind in range(len(widths)-1):
            use_attention = attention_levels[ind]
            down_layers.append(DownBlock(
                widths[ind], 
                widths[ind+1], 
                pos_channels, 
                block_depth, 
                use_attention, 
                context_channels
            ))

        self.down_layers = torch.nn.ModuleList(down_layers)

        for _ in range(block_depth):
            use_attention = attention_levels[ind]
            bottleneck_layers.append(ResBlock(widths[-1], widths[-1], pos_channels))
            if use_attention:
                bottleneck_layers.append(BasicTransformerBlock(widths[-1], context_channels))

        self.bottleneck_layers = torch.nn.ModuleList(bottleneck_layers)

        for ind in reversed(range(1, len(widths))):
            use_attention = attention_levels[ind-1]
            up_layers.append(UpBlock(
                widths[ind]*2, 
                widths[ind-1], 
                pos_channels, 
                block_depth, 
                use_attention, 
                context_channels
            ))

        self.up_layers = torch.nn.ModuleList(up_layers)

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(8, widths[0]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(widths[0], output_channels, kernel_size=1)
        )

    def forward(self, x, t, context=None, x_self_cond=None):
        timesteps = positional_emb(t, self.pos_channels, self.max_period)
        x = self.inc(x)
        skips = []

        for downblock in self.down_layers:
            x = downblock(x, skips, timesteps, context=context)

        _, _, h, w = x.shape

        for bottleneck_block in self.bottleneck_layers:
            if isinstance(bottleneck_block, ResBlock):
                x = bottleneck_block(x, timesteps)
            elif isinstance(bottleneck_block, BasicTransformerBlock):
                x = rearrange(x, "b c h w -> b (h w) c")
                x = bottleneck_block(x, context=context)
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            else:
                raise NotImplementedError

        for upblock in self.up_layers:
            x = upblock(x, skips, timesteps, context=context)

        return self.out(x)

if __name__ == '__main__':
    block_depth = 3
    model_channels = 128
    channel_mults = [1, 2, 3]
    widths = [model_channels * mult for mult in channel_mults]
    attention_levels = [0, 1, 1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    out_channels = 4

    model = UNet(block_depth, widths, attention_levels, in_channels, out_channels, device).to(device)
    from torchinfo import summary
    
    batch_size = 16
    latent_size = 16
    n_noise_steps = 1000
    
    batch_images = torch.randn([batch_size, in_channels, latent_size, latent_size]).to(device)
    t = torch.randint(low=1, high=n_noise_steps, size=(batch_size, 1)).to(device)
    y = torch.randn([batch_size, 512]).to(device)
    summary(model, input_data=[batch_images, t, y])
    