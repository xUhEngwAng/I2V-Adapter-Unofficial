import sys
import torch
import unittest

sys.path.append('./')

from src.modules.i2v_adapter import I2VAdapterTransformerBlock
from src.modules.i2v_adapter import I2VAdapterTransformer2DModel

class TestI2VAdapterTransformer2DModel(unittest.TestCase):
    def setUp(self):
        self.num_attention_heads = 8
        self.in_channels = 512
        self.out_channels = 512
        self.num_layers = 1
        self.cross_attention_dim = 2 * self.out_channels
        self.norm_num_groups = 32
        
        self.i2v_adapter_transformer_2d_model = I2VAdapterTransformer2DModel(
            self.num_attention_heads,
            self.out_channels // self.num_attention_heads,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            cross_attention_dim=self.cross_attention_dim,
            norm_num_groups=self.norm_num_groups,
        )

    def test_I2VAdapterTransformer2DModel_output_shape(self):
        width = 16
        height = 16
        batch_size = 4
        num_frames = 8
        encoder_seq_len = 77
        batch_frames = batch_size * num_frames

        hidden_states = torch.randn((batch_frames, self.in_channels, width, height))
        encoder_hidden_states = torch.randn((batch_frames , encoder_seq_len, self.cross_attention_dim))
        
        output = self.i2v_adapter_transformer_2d_model(
            hidden_states,
            enable_cross_frame_attn=True,
            num_frames=num_frames,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False
        )[0]
        
        self.assertEqual(output.shape, (batch_frames, self.out_channels, width, height))

    def test_I2VAdapterTransformer2DModel_no_cross_frame_output_shape(self):
        width = 16
        height = 16
        batch_size = 32
        encoder_seq_len = 77

        hidden_states = torch.randn((batch_size, self.in_channels, width, height))
        encoder_hidden_states = torch.randn((batch_size , encoder_seq_len, self.cross_attention_dim))
        
        output = self.i2v_adapter_transformer_2d_model(
            hidden_states,
            enable_cross_frame_attn=False,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False
        )[0]
        
        self.assertEqual(output.shape, (batch_size, self.out_channels, width, height))

class TestI2VAdapterTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 64
        self.encoder_seq_len = 77
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.attention_head_dim = self.hidden_size // self.num_attention_heads
        self.dropout = 0.0
        self.cross_attention_dim = 2 * self.hidden_size
        self.activation_fn = 'gelu'
       
        self.i2v_adapter_transformer_block = I2VAdapterTransformerBlock(
            self.hidden_size,
            self.num_attention_heads,
            self.attention_head_dim,
            dropout=self.dropout,
            cross_attention_dim=self.cross_attention_dim,
            activation_fn=self.activation_fn
       )

    def test_I2VAdapter_output_shape(self):
        num_frames = 8
        batch_frames = num_frames * self.batch_size

        hidden_states = torch.randn((batch_frames, self.seq_len, self.hidden_size))
        encoder_hidden_states = torch.randn((batch_frames , self.encoder_seq_len, self.cross_attention_dim))

        output = self.i2v_adapter_transformer_block(
            hidden_states,
            attention_mask=None,
            enable_cross_frame_attn=True,
            num_frames=num_frames,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None
        )

        self.assertEqual(output.shape, (batch_frames, self.seq_len, self.hidden_size))

    def test_I2VAdapter_no_cross_frame_output_shape(self):
        hidden_states = torch.randn((self.batch_size, self.seq_len, self.hidden_size))
        encoder_hidden_states = torch.randn((self.batch_size , self.encoder_seq_len, self.cross_attention_dim))

        output = self.i2v_adapter_transformer_block(
            hidden_states,
            attention_mask=None,
            enable_cross_frame_attn=False,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None
        )

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))

if  __name__ == "__main__":
    unittest.main()
