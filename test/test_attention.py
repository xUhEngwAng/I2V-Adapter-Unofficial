import sys
import torch
import unittest

sys.path.append('./')

from src.modules.attention import SelfAttention
from src.modules.attention import BasicAttention
from src.modules.attention import BasicTransformerBlock
from src.modules.attention import VideoTransformer

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.seq_length = 64
        self.emb_dim = 64
        
        self.self_attn = SelfAttention(self.emb_dim)

    def testSelfAttention_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.seq_length, self.emb_dim])
        output = self.self_attn(batch_feat_map)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.emb_dim))

class TestBasicAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.seq_length = 64
        self.query_dim = 64
        self.context_dim = 128
        self.context_seq_length = 42

        self.self_attn = BasicAttention(self.query_dim, context_dim=None)
        self.cross_attn = BasicAttention(self.query_dim, context_dim=self.context_dim)

    def testSelfAttention_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.seq_length, self.query_dim])
        output = self.self_attn(batch_feat_map)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.query_dim))

    def testCrossAttention_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.seq_length, self.query_dim])
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        output = self.cross_attn(batch_feat_map, context)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.query_dim))

class TestBasicTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.seq_length = 64
        self.query_dim = 64
        self.context_dim = 128
        self.context_seq_length = 42

        self.self_attn = BasicTransformerBlock(self.query_dim, context_dim=None)
        self.cross_attn = BasicTransformerBlock(self.query_dim, context_dim=self.context_dim)

    def testSelfAttention_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.seq_length, self.query_dim])
        output = self.self_attn(batch_feat_map)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.query_dim))

    def testCrossAttention_output_shape(self):
        batch_feat_map = torch.randn([self.batch_size, self.seq_length, self.query_dim])
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        output = self.cross_attn(batch_feat_map, context)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.query_dim))

class TestVideoTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.feat_map_size = 16
        self.n_channels = 64
        self.frame_num = 8
        self.context_dim = 128
        self.context_seq_length = 42

        self.video_self_transformer = VideoTransformer(self.n_channels, context_channels=None)
        self.video_cross_transformer = VideoTransformer(self.n_channels, context_channels=self.context_dim)

    def testVideoSelfTransformer_output_shape(self):
        batch_feat_map = torch.randn([
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ])
        output = self.video_self_transformer(
            batch_feat_map, 
            context=None, 
            num_frames=self.frame_num, 
            image_only_indicator=torch.Tensor([True])
        )
        self.assertEqual(output.shape, (
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ))

        # image_only_indicator is False
        output = self.video_self_transformer(
            batch_feat_map, 
            context=None, 
            num_frames=self.frame_num, 
            image_only_indicator=torch.Tensor([False])
        )
        self.assertEqual(output.shape, (
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ))

    def testVideoCrossTransformer_output_shape(self):
        batch_feat_map = torch.randn([
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ])
        context = torch.randn([self.batch_size, self.context_seq_length, self.context_dim])
        output = self.video_cross_transformer(
            batch_feat_map, 
            context=context, 
            num_frames=self.frame_num, 
            image_only_indicator=torch.Tensor([True])
        )
        self.assertEqual(output.shape, (
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ))

        # image_only_indicator is False
        output = self.video_cross_transformer(
            batch_feat_map, 
            context=context,
            num_frames=self.frame_num, 
            image_only_indicator=torch.Tensor([False])
        )
        self.assertEqual(output.shape, (
            self.batch_size*self.frame_num, 
            self.n_channels, 
            self.feat_map_size, 
            self.feat_map_size
        ))

if __name__ == '__main__':
    unittest.main()