import sys
import torch
import unittest

sys.path.append('./')

from src.models.unet3d import DownBlock3D, UpBlock3D
from src.models.unet3d import UNet3D

class TestDownBlock3D(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.feat_map_size = 16
        self.in_channels = 64
        self.out_channels = 128
        self.pos_channels = 256
        self.block_depth = 3
        self.num_frames = 8
        
        self.downblock3d_wo_attn = DownBlock3D(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=False,
            num_frames=self.num_frames
        )

        self.downblock3d_self_attn = DownBlock3D(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            num_frames=self.num_frames
        )

        self.context_dim = 128
        self.context_seq_length = 42

        self.downblock3d_cross_attn = DownBlock3D(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            num_frames=self.num_frames,
            context_channels=self.context_dim
        )

    def positional_emb(self, t, max_period=10000):
        freqs = 1 / (max_period ** (torch.arange(0, self.pos_channels, 2) / self.pos_channels))
        pos_emb_sin = torch.sin(t.repeat(1, self.pos_channels // 2) * freqs)
        pos_emb_cos = torch.cos(t.repeat(1, self.pos_channels // 2) * freqs)
        return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    def downblock3d_output_shape_test(self, downblock3d, context, image_only_indicator):
        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1

        batch_feat_map = torch.randn([self.batch_size*frame_num, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size*frame_num, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size*frame_num, self.pos_channels))

        skips = []
        output = downblock3d(
            batch_feat_map, 
            context, 
            skips,
            timesteps,
            image_only_indicator=image_only_indicator
        )
        self.assertEqual(len(skips), self.block_depth)
        
        for skip in skips:
            self.assertEqual(skip.shape, (self.batch_size*frame_num, self.out_channels, self.feat_map_size, self.feat_map_size))

        self.assertEqual(output.shape, (self.batch_size*frame_num, self.out_channels, self.feat_map_size//2, self.feat_map_size//2))

    def testDownBlock3D_wo_attn_output_shape(self):
        self.downblock3d_output_shape_test(self.downblock3d_wo_attn, context=None, image_only_indicator=torch.Tensor([True]))
        self.downblock3d_output_shape_test(self.downblock3d_wo_attn, context=None, image_only_indicator=torch.Tensor([False]))

    def testDownBlock3D_self_attn_output_shape(self):
        self.downblock3d_output_shape_test(self.downblock3d_self_attn, context=None, image_only_indicator=torch.Tensor([True]))
        self.downblock3d_output_shape_test(self.downblock3d_self_attn, context=None, image_only_indicator=torch.Tensor([False]))

    def testDownBlock3D_cross_attn_output_shape(self):
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        self.downblock3d_output_shape_test(self.downblock3d_cross_attn, context=context, image_only_indicator=torch.Tensor([True]))
        self.downblock3d_output_shape_test(self.downblock3d_cross_attn, context=context, image_only_indicator=torch.Tensor([False]))

class TestUpBlock3D(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.feat_map_size = 16
        self.in_channels = 256
        self.out_channels = 128
        self.pos_channels = 256
        self.block_depth = 3
        self.num_frames = 8
        
        self.upblock_wo_attn = UpBlock3D(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=False,
            num_frames=self.num_frames
        )

        self.upblock_self_attn = UpBlock3D(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            num_frames=self.num_frames
        )

        self.context_dim = 128
        self.context_seq_length = 42

        self.upblock_cross_attn = UpBlock3D(
            2*self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.block_depth,
            use_attention=True,
            num_frames=self.num_frames,
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

    def upblock3d_output_shape_test(self, upblock3d, context, image_only_indicator):
        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1

        skips = []

        for _ in range(self.block_depth):
            skips.append(torch.randn([self.batch_size*frame_num, self.in_channels, 2*self.feat_map_size, 2*self.feat_map_size]))

        batch_feat_map = torch.randn([self.batch_size*frame_num, self.in_channels, self.feat_map_size, self.feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size*frame_num, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size*frame_num, self.pos_channels))

        output = upblock3d(
            batch_feat_map, 
            context, 
            skips,
            timesteps,
            image_only_indicator=image_only_indicator
        )
        self.assertEqual(len(skips), 0)
        self.assertEqual(output.shape, (self.batch_size*frame_num, self.out_channels, self.feat_map_size*2, self.feat_map_size*2))

    def testUpBlock_wo_attn_output_shape(self):
        self.upblock3d_output_shape_test(self.upblock_wo_attn, context=None, image_only_indicator=torch.Tensor([True]))
        self.upblock3d_output_shape_test(self.upblock_wo_attn, context=None, image_only_indicator=torch.Tensor([False]))

    def testUpBlock_self_attn_output_shape(self):
        self.upblock3d_output_shape_test(self.upblock_self_attn, context=None, image_only_indicator=torch.Tensor([True]))
        self.upblock3d_output_shape_test(self.upblock_self_attn, context=None, image_only_indicator=torch.Tensor([False]))

    def testUpBlock_cross_attn_output_shape(self):
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        self.upblock3d_output_shape_test(self.upblock_cross_attn, context=context, image_only_indicator=torch.Tensor([True]))
        self.upblock3d_output_shape_test(self.upblock_cross_attn, context=context, image_only_indicator=torch.Tensor([False]))

class testUNet3D(unittest.TestCase):
    def setUp(self):
        model_channels = 128
        channel_mults = [1, 2, 3]

        block_depth = 2
        widths = [model_channels * mult for mult in channel_mults]
        attention_levels = [0, 1, 1]

        self.in_channels = 4
        self.out_channels = 4
        self.batch_size = 8
        self.input_size = 16
        self.num_frames = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.context_dim = 128
        self.context_seq_length = 42

        self.unet3d_uncond = UNet3D(
            block_depth,
            widths, 
            attention_levels,
            self.in_channels,
            self.out_channels,
            self.device,
            num_frames=self.num_frames
        ).to(self.device)

        self.unet3d_cond = UNet3D(
            block_depth,
            widths, 
            attention_levels,
            self.in_channels,
            self.out_channels,
            self.device,
            num_frames=self.num_frames,
            context_channels = self.context_dim
        ).to(self.device)

    def unet3d_output_shape_test(self, unet3d, context, image_only_indicator):
        frame_num = self.num_frames
        if image_only_indicator:
            frame_num = 1

        input = torch.randn((
            self.batch_size*frame_num, 
            self.in_channels, 
            self.input_size, 
            self.input_size
        )).to(self.device)
        
        t = torch.randint(low=1, high=1000, size=(self.batch_size*frame_num, 1)).to(self.device)
        output = unet3d(input, t, image_only_indicator, context=context)
        self.assertEqual(output.shape, (self.batch_size*frame_num, self.out_channels, self.input_size, self.input_size))

    def testUNet3D_uncond_output_shape(self):
        self.unet3d_output_shape_test(self.unet3d_uncond, context=None, image_only_indicator=torch.tensor([True], device=self.device))
        self.unet3d_output_shape_test(self.unet3d_uncond, context=None, image_only_indicator=torch.tensor([False], device=self.device))

    def testUNet3D_cond_output_shape(self):
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim]).to(self.device)
        self.unet3d_output_shape_test(self.unet3d_cond, context=context, image_only_indicator=torch.tensor([True], device=self.device))
        self.unet3d_output_shape_test(self.unet3d_cond, context=context, image_only_indicator=torch.tensor([False], device=self.device))

if __name__ == '__main__':
    unittest.main()
