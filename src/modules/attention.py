import torch
from einops import rearrange, repeat

from .util import AlphaBlender, positional_emb

class SelfAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super().__init__()
        
        self.mha = torch.nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm([emb_dim])
        self.ff_layer = torch.nn.Sequential(
            torch.nn.LayerNorm([emb_dim]),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, emb_dim)
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

        spatial_context = repeat(context, "b ... -> (b n) ...", n=num_frames) if context is not None else None
        temporal_context = repeat(context, "b ... -> (b n) ...", n=h * w) if context is not None else None
        
        x = rearrange(x, "b c h w -> b (h w) c")
        x_spatial = super().forward(x, context=spatial_context)

        # positional embedding for each frame
        frames = torch.arange(1, 1+num_frames, device=x.device)
        frames = repeat(frames, "t -> b t", b=x.shape[0] // num_frames)
        frames = rearrange(frames, "b t -> (b t) 1")
        pos_emb = positional_emb(frames, self.n_channels)
        emb_out = self.frame_pos_embed(pos_emb)[:, None, :]

        x_temporal = x_spatial + emb_out
        x_temporal = rearrange(x_temporal, "(b t) s c -> (b s) t c", t=num_frames)
        x_temporal = self.video_attn(x_temporal, context=temporal_context)
        x_temporal = rearrange(x_temporal, "(b s) t c -> (b t) s c", s=h*w)

        output = self.time_mixer(
            x_spatial=x_spatial,
            x_temporal=x_temporal,
            image_only_indicator=image_only_indicator,
        )
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)
        return output + x_in
