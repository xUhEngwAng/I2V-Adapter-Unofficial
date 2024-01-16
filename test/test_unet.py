import sys
import torch
import unittest

sys.path.append('./')

from src.models.unet import DownBlock, UpBlock
from src.models.unet import UNet

class TestDownBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.feat_map_size = 16
        self.in_channels = 64
        self.out_channels = 128
        self.pos_channels = 256
        self.block_depth = 3
        
        self.downblock_wo_attn = DownBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=False
        )

        self.downblock_self_attn = DownBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True
        )

        self.context_dim = 128
        self.context_seq_length = 42

        self.downblock_cross_attn = DownBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            context_channels=self.context_dim
        )

    def positional_emb(self, t, max_period=10000):
        freqs = 1 / (max_period ** (torch.arange(0, self.pos_channels, 2) / self.pos_channels))
        pos_emb_sin = torch.sin(t.repeat(1, self.pos_channels // 2) * freqs)
        pos_emb_cos = torch.cos(t.repeat(1, self.pos_channels // 2) * freqs)
        return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    def testDownBlock_wo_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        skips = []
        output = self.downblock_wo_attn(batch_feat_map, skips, timesteps, context=None)
        self.assertEqual(len(skips), self.block_depth)
        for skip in skips:
            self.assertEqual(skip.shape, (self.batch_size, self.out_channels, self.feat_map_size, self.feat_map_size))

        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size//2, self.feat_map_size//2))

    def testDownBlock_self_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        skips = []
        output = self.downblock_self_attn(batch_feat_map, skips, timesteps, context=None)
        self.assertEqual(len(skips), self.block_depth)
        for skip in skips:
            self.assertEqual(skip.shape, (self.batch_size, self.out_channels, self.feat_map_size, self.feat_map_size))

        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size//2, self.feat_map_size//2))

    def testDownBlock_cross_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        skips = []
        output = self.downblock_cross_attn(batch_feat_map, skips, timesteps, context=context)
        self.assertEqual(len(skips), self.block_depth)
        for skip in skips:
            self.assertEqual(skip.shape, (self.batch_size, self.out_channels, self.feat_map_size, self.feat_map_size))

        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size//2, self.feat_map_size//2))

class TestUpBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.feat_map_size = 16
        self.in_channels = 256
        self.out_channels = 128
        self.pos_channels = 256
        self.block_depth = 3
        
        self.upblock_wo_attn = UpBlock(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=False
        )

        self.upblock_self_attn = UpBlock(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True
        )

        self.context_dim = 128
        self.context_seq_length = 42

        self.upblock_cross_attn = UpBlock(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            context_channels=self.context_dim
        )

        self.skips = []

        for _ in range(self.block_depth):
            self.skips.append(torch.randn([self.batch_size, self.in_channels, 2*self.feat_map_size, 2*self.feat_map_size]))

    def positional_emb(self, t, max_period=10000):
        freqs = 1 / (max_period ** (torch.arange(0, self.pos_channels, 2) / self.pos_channels))
        pos_emb_sin = torch.sin(t.repeat(1, self.pos_channels // 2) * freqs)
        pos_emb_cos = torch.cos(t.repeat(1, self.pos_channels // 2) * freqs)
        return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    def testUpBlock_wo_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        output = self.upblock_wo_attn(batch_feat_map, self.skips, timesteps, context=None)
        self.assertEqual(len(self.skips), 0)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size*2, self.feat_map_size*2))

    def testUpBlock_self_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        output = self.upblock_self_attn(batch_feat_map, self.skips, timesteps, context=None)
        self.assertEqual(len(self.skips), 0)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size*2, self.feat_map_size*2))

    def testUpBlock_cross_attn_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, self.feat_map_size, self.feat_map_size])
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))

        output = self.upblock_cross_attn(batch_feat_map, self.skips, timesteps, context=context)
        self.assertEqual(len(self.skips), 0)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.feat_map_size*2, self.feat_map_size*2))

class testUNet(unittest.TestCase):
    def setUp(self):
        model_channels = 128
        channel_mults = [1, 2, 3]

        block_depth = 3
        widths = [model_channels * mult for mult in channel_mults]
        attention_levels = [0, 1, 1]

        self.in_channels = 4
        self.out_channels = 4
        self.batch_size = 16
        self.input_size = 32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.context_dim = 128
        self.context_seq_length = 42

        self.unet_uncond = UNet(
            block_depth,
            widths, 
            attention_levels,
            self.in_channels,
            self.out_channels,
            self.device
        ).to(self.device)

        self.unet_cond = UNet(
            block_depth,
            widths, 
            attention_levels,
            self.in_channels,
            self.out_channels,
            self.device,
            context_channels = self.context_dim
        ).to(self.device)

    def testUNet_uncond_output_shape(self):
        input = torch.randn((self.batch_size, self.in_channels, self.input_size, self.input_size)).to(self.device)
        t = torch.randint(low=1, high=1000, size=(self.batch_size, 1)).to(self.device)
        output = self.unet_uncond(input, t, context=None)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.input_size, self.input_size))

    def testUNet_cond_output_shape(self):
        input = torch.randn((self.batch_size, self.in_channels, self.input_size, self.input_size)).to(self.device)
        t = torch.randint(low=1, high=1000, size=(self.batch_size, 1)).to(self.device)
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim]).to(self.device)
        output = self.unet_cond(input, t, context=context)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.input_size, self.input_size))

if __name__ == '__main__':
    unittest.main()
