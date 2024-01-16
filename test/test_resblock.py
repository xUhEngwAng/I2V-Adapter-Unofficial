import sys
import torch
import unittest

sys.path.append('./')

from src.modules.resnet import ResBlock
from src.modules.resnet import VideoResBlock

class TestResBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.in_channels = 64
        self.out_channels = 128
        self.pos_channels = 256
        
        self.resblock_2d = ResBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels
        )

        self.resblock_3d = ResBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            dims=3, 
            kernel_size=[3, 1, 1],
        )

    def positional_emb(self, t, max_period=10000):
        freqs = 1 / (max_period ** (torch.arange(0, self.pos_channels, 2) / self.pos_channels))
        if len(t.shape) == 2:
            pos_emb_sin = torch.sin(t.repeat(1, self.pos_channels // 2) * freqs)
            pos_emb_cos = torch.cos(t.repeat(1, self.pos_channels // 2) * freqs)
        elif len(t.shape) == 3:
            pos_emb_sin = torch.sin(t.repeat(1, 1, self.pos_channels // 2) * freqs)
            pos_emb_cos = torch.cos(t.repeat(1, 1, self.pos_channels // 2) * freqs)
        else:
            raise ValueError(f"unsupported shape of timesteps: {t.shape}")
        return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    def test_ResBlock2d_output_shape(self):
        feat_map_size = 16
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, feat_map_size, feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, self.pos_channels))
    
        output = self.resblock_2d(batch_feat_map, timesteps)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, feat_map_size, feat_map_size))

    def test_ResBlock3d_output_shape(self):
        feat_map_size = 16
        frame_num = 8
        batch_feat_map = torch.randn([self.batch_size, self.in_channels, frame_num, feat_map_size, feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size, frame_num, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size, frame_num, self.pos_channels))
    
        output = self.resblock_3d(batch_feat_map, timesteps)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, frame_num, feat_map_size, feat_map_size))

class TestVideoResBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.in_channels = 64
        self.out_channels = 128
        self.pos_channels = 256
        self.video_kernel_size = [3, 1, 1]
        
        self.video_resblock = VideoResBlock(
            self.in_channels,
            self.out_channels,
            self.pos_channels,
            self.video_kernel_size
        )

    def positional_emb(self, t, max_period=10000):
        freqs = 1 / (max_period ** (torch.arange(0, self.pos_channels, 2) / self.pos_channels))
        pos_emb_sin = torch.sin(t.repeat(1, self.pos_channels // 2) * freqs)
        pos_emb_cos = torch.cos(t.repeat(1, self.pos_channels // 2) * freqs)
        return torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)

    def test_VideoResBlock_output_shape(self):
        feat_map_size = 16
        frame_num = 8

        batch_feat_map = torch.randn([self.batch_size * frame_num, self.in_channels, feat_map_size, feat_map_size])
        timesteps = self.positional_emb(torch.randint(low=1, high=1000, size=(self.batch_size * frame_num, 1)))
        self.assertEqual(timesteps.shape, (self.batch_size * frame_num, self.pos_channels))

        output = self.video_resblock(batch_feat_map, timesteps, frame_num, image_only_indicator=torch.Tensor([True]))
        self.assertEqual(output.shape, (self.batch_size * frame_num, self.out_channels, feat_map_size, feat_map_size))

        output = self.video_resblock(batch_feat_map, timesteps, frame_num, image_only_indicator=torch.Tensor([False]))
        self.assertEqual(output.shape, (self.batch_size * frame_num, self.out_channels, feat_map_size, feat_map_size))
        
if __name__ == '__main__':
    unittest.main()