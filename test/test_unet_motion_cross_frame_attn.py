import sys
import torch
import unittest

from diffusers import (
    MotionAdapter,
    AnimateDiffPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    UNetMotionModel
)

sys.path.append('./')

from src.models.unet_motion_cross_frame_attn import CrossFrameAttnDownBlockMotion
from src.models.unet_motion_cross_frame_attn import UNetMotionCrossFrameAttnModel

class TestCrossFrameAttnDownBlockMotion(unittest.TestCase):
    def setUp(self) -> None:
        self.in_channels = 64
        self.out_channels = 128
        self.temb_channels = 512
        self.cross_attention_dim = 768
        self.num_attention_heads = 8
        self.num_layers = 2

        self.cross_frame_attn_down_block = CrossFrameAttnDownBlockMotion(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            temb_channels=self.temb_channels,
            cross_attention_dim=self.cross_attention_dim,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
        )

    def test_CrossFrameAttnDownBlockMotion_output_shape(self):
        batch_size = 16
        height = 16
        width = 16
        num_frames = 8
        seq_len = 77
        batch_frames = batch_size * num_frames

        hidden_states = torch.randn((batch_frames, self.in_channels, height, width))
        temb = torch.randn((batch_frames, self.temb_channels))
        encoder_hidden_states = torch.randn((batch_frames, seq_len, self.cross_attention_dim))

        hidden_states, output_states = self.cross_frame_attn_down_block(
            hidden_states=hidden_states,
            temb=temb,
            enable_cross_frame_attn=True,
            encoder_hidden_states=encoder_hidden_states,
            num_frames=num_frames
        )

        self.assertEqual(hidden_states.shape, (batch_frames, self.out_channels, height // 2, width // 2))
        self.assertEqual(len(output_states), self.num_layers+1)

        for ind, output_state in enumerate(output_states):
            if ind != self.num_layers:
                self.assertEqual(output_state.shape, (batch_frames, self.out_channels, height, width))
            else:
                self.assertEqual(output_state.shape, (batch_frames, self.out_channels, height // 2, width // 2))

    def test_CrossFrameAttnDownBlockMotion_no_cross_frame_output_shape(self):
        batch_size = 16
        height = 16
        width = 16
        num_frames = 8
        seq_len = 77
        batch_frames = batch_size * num_frames

        hidden_states = torch.randn((batch_frames, self.in_channels, height, width))
        temb = torch.randn((batch_frames, self.temb_channels))
        encoder_hidden_states = torch.randn((batch_frames, seq_len, self.cross_attention_dim))

        hidden_states, output_states = self.cross_frame_attn_down_block(
            hidden_states=hidden_states,
            temb=temb,
            enable_cross_frame_attn=False,
            encoder_hidden_states=encoder_hidden_states,
            num_frames=num_frames
        )

        self.assertEqual(hidden_states.shape, (batch_frames, self.out_channels, height // 2, width // 2))
        self.assertEqual(len(output_states), self.num_layers+1)

        for ind, output_state in enumerate(output_states):
            if ind != self.num_layers:
                self.assertEqual(output_state.shape, (batch_frames, self.out_channels, height, width))
            else:
                self.assertEqual(output_state.shape, (batch_frames, self.out_channels, height // 2, width // 2))

class testUNetMotionCrossFrameAttnModel(unittest.TestCase):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 4
        self.block_out_channels = (320, 640, 1280, 1280)
        self.cross_attention_dim = 768
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.unet_model = UNetMotionCrossFrameAttnModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            block_out_channels=self.block_out_channels,
            cross_attention_dim=self.cross_attention_dim
        ).to(self.device)

    def test_UNetMotionCrossFrameAttnModel_output_shape(self):
        batch_size = 16
        height = 16
        width = 16
        num_frames = 8
        seq_len = 42

        sample = torch.randn((batch_size, num_frames, self.in_channels, height, width), device=self.device)
        timestep = torch.randint(low=1, high=1000, size=(batch_size, ), device=self.device)
        encoder_hidden_states = torch.randn((batch_size, seq_len, self.cross_attention_dim), device=self.device)

        output = self.unet_model(
            sample=sample,
            timestep=timestep,
            enable_cross_frame_attn=True,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]

        self.assertEqual(output.shape, (batch_size, num_frames, self.out_channels, height, width))

    def test_UNetMotionCrossFrameAttnModel_no_cross_frame_output_shape(self):
        batch_size = 16
        height = 16
        width = 16
        num_frames = 8
        seq_len = 42

        sample = torch.randn((batch_size, num_frames, self.in_channels, height, width), device=self.device)
        timestep = torch.randint(low=1, high=1000, size=(batch_size, ), device=self.device)
        encoder_hidden_states = torch.randn((batch_size, seq_len, self.cross_attention_dim), device=self.device)

        output = self.unet_model(
            sample=sample,
            timestep=timestep,
            enable_cross_frame_attn=False,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]

        self.assertEqual(output.shape, (batch_size, num_frames, self.out_channels, height, width))

    def test_UNetMotionCrossFrameAttnModel_load_pretrained(self):
        adapter = MotionAdapter.from_pretrained('./animatediff-motion-adapter-v1-5-2')
        unet = UNet2DConditionModel.from_pretrained('./stable-diffusion-v1-5/unet')
        unet_motion_cross_frame_attn_model = UNetMotionCrossFrameAttnModel.from_unet2d(unet, adapter)
        unet_motion_cross_frame_attn_model.freeze_unet_params()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'{count_parameters(unet_motion_cross_frame_attn_model)} parameters can be trained.')

if __name__ == '__main__':
    unittest.main()

