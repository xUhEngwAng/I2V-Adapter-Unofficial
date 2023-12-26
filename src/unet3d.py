import torch

from einops import rearrange, repeat
from typing import Iterable

def positional_emb(t, channels, max_period=10000):
    freqs = 1 / (max_period ** (torch.arange(0, channels, 2, device=t.device) / channels))
    pos_emb_sin = torch.sin(t.repeat(1, channels // 2) * freqs)
    pos_emb_cos = torch.cos(t.repeat(1, channels // 2) * freqs)
    return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

class AlphaBlender(torch.nn.Module):
    '''
    Code borrowed from 
    https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/diffusionmodules/util.py#L312
    '''
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: torch.Tensor = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x

class SelfAttention(torch.nn.Module):
    def __init__(self, n_channels, num_heads=4):
        super().__init__()
        self.n_channels = n_channels
        
        self.mha = torch.nn.MultiheadAttention(n_channels, num_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm([n_channels])
        self.ff_layer = torch.nn.Sequential(
            torch.nn.LayerNorm([n_channels]),
            torch.nn.Linear(n_channels, n_channels),
            torch.nn.GELU(),
            torch.nn.Linear(n_channels, n_channels)
        )

    def forward(self, x):
        x_ln = self.ln(x)
        x_mha, _ = self.mha(x_ln, x_ln, x_ln)
        x_mha = x_mha + x_ln
        x_ff = self.ff_layer(x_mha) + x_mha
        return x_ff

class BasicAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, head_dim=64, num_heads=8, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * num_heads
        context_dim = query_dim if context_dim is None else context_dim
        self.num_heads = num_heads
    
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim), 
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        '''
        Perform self-attention if `context` is None.
        Code heavily borrowed from 
        https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/attention.py#L255
        '''
        h = self.num_heads
        
        q = self.to_q(x)
        context = x if context is None else context
        # print(f'context shape: {context.shape}')
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        return self.to_out(out)

class BasicTransformerBlock(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, head_dim=64, num_heads=8):
        super().__init__()
        # attn1 default to self-attention
        self.attn1 = BasicAttention(query_dim, query_dim, head_dim, num_heads)
        # attn2 default to cross-attention if `context_dim` is provided
        self.attn2 = BasicAttention(query_dim, context_dim, head_dim, num_heads)
        self.norm1 = torch.nn.LayerNorm(query_dim)
        self.norm2 = torch.nn.LayerNorm(query_dim)

    def forward(self, x, context=None):
        # print(f'x shape: {x.shape}')
        # print(f'context shape: {context.shape}')
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return x

class VideoTransformer(BasicTransformerBlock):
    def __init__(
        self, 
        n_channels,
        context_channels=None,
        merge_factor=0.5
    ):
        super().__init__(n_channels, context_channels)
        
        self.n_channels = n_channels

        self.video_attn = BasicTransformerBlock(n_channels, context_channels)

        time_embed_dim = self.n_channels * 4
        self.frame_pos_embed = torch.nn.Sequential(
            torch.nn.Linear(self.n_channels, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, self.n_channels)
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy="learned_with_images"
        )
        
    def forward(self, x, context, num_frames, image_only_indicator):
        x_in = x
        _, _, h, w = x_in.shape
        # print(f'h={h}, w={w}, x shape: {x.shape}')
        
        x = rearrange(x, "b c h w -> b (h w) c")
        x_spatial = super().forward(x, context=context)

        # positional embedding for each frame
        frames = torch.arange(1, 1+num_frames, device=x.device)
        frames = repeat(frames, "t -> b t", b=x.shape[0] // num_frames)
        frames = rearrange(frames, "b t -> (b t) 1")
        pos_emb = positional_emb(frames, self.n_channels)
        emb_out = self.frame_pos_embed(pos_emb)[:, None, :]

        if context is not None:
            context = repeat(context, "b ... -> (b n) ...", n=h*w // num_frames)

        x_temporal = x_spatial + emb_out
        x_temporal = rearrange(x_spatial, "(b t) s c -> (b s) t c", t=num_frames)
        x_temporal = self.video_attn(x_temporal, context=context)
        x_temporal = rearrange(x_temporal, "(b s) t c -> (b t) s c", s=h*w)

        output = self.time_mixer(
            x_spatial=x_spatial,
            x_temporal=x_temporal,
            image_only_indicator=image_only_indicator,
        )
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)
        return output + x_in

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
        for modules in self.layers:
            for module in modules:
                if isinstance(module, VideoResBlock):
                    x = module(x, timesteps, self.num_frames, image_only_indicator)
                elif isinstance(module, VideoTransformer):
                    x = module(x, context, self.num_frames, image_only_indicator)
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
        x = self.up_sampler(x)

        for modules in self.layers:
            x = torch.cat([x, skips.pop()], dim=1)
            
            for module in modules:
                if isinstance(module, VideoResBlock):
                    x = module(x, timesteps, self.num_frames, image_only_indicator)
                elif isinstance(module, VideoTransformer):
                    x = module(x, context, self.num_frames, image_only_indicator)
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
        self.inc = conv_nd(2, input_channels, 128, kernel_size=3, padding=1)

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

        for layer in self.bottleneck_layers:
            if isinstance(layer, VideoResBlock):
                x = layer(x, timesteps, self.num_frames, image_only_indicator)
            elif isinstance(layer, VideoTransformer):
                x = layer(x, context, self.num_frames, image_only_indicator)
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
