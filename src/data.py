import logging
import numpy as np
import torch
import random

from einops import rearrange
from pathlib import Path
from torch.utils.data import Dataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatentImageDataset(Dataset):
    def __init__(self, latent_path):
        self.path = latent_path
        std_latent = 2*(1/0.18215)

        image_latents = torch.Tensor(np.load(latent_path)[:10000])
        image_latents = image_latents.clamp(-std_latent, std_latent) / std_latent
        # image_latents = (image_latents + 1) / 2
        self.image_latents = image_latents
        # self.image_latents = rearrange(image_latents, 'b h w c -> b c h w')
        
        logger.info(f'{len(self.image_latents)} image samples loaded from {latent_path}.')

    def __len__(self):
        return len(self.image_latents)

    def __getitem__(self, ind):
        return self.image_latents[ind]

class LatentVideoDataset(Dataset):
    def __init__(self, root_dir, bucket_size, num_frames=8):
        super().__init__()

        self.bucket_size = bucket_size
        self.num_frames = num_frames

        self.paths = [str(p) for p in Path(root_dir).glob('**/*.pt')]
        self.video_latents = []

        filter_cnt = 0
        std_latent = 2*(1/0.18215)

        # filter videos whose frame count is less than `bucket_size x num_frames`
        for path in self.paths:
            video_latent = torch.load(path, map_location='cpu')
            frame_cnt = len(video_latent)
            if bucket_size * num_frames <= frame_cnt:
                video_latent = video_latent.clamp(-std_latent, std_latent) / std_latent
                self.video_latents.append(video_latent)
            else:
                filter_cnt += 1

        logger.info(f'{len(self.video_latents)} video samples loaded from {root_dir}, with {filter_cnt} filtered.')

    def sample_frames(self, frames):
        '''
        Sample frames to `num_frames` according to `bucket_size`.
        Specifically, a random frame is sampled in every bucket, whereas 
        the starting point is also random sampled.
        '''
        frame_cnt = len(frames)
        start = random.randint(0, frame_cnt-self.bucket_size*self.num_frames)
        indices = []

        for _ in range(self.num_frames):
            sampled = random.randint(start, start+self.bucket_size-1)
            indices.append(sampled)
            start = start + self.bucket_size
            
        return frames[indices]

    def __len__(self):
        return len(self.video_latents)

    def __getitem__(self, ind):
        video_latent = self.video_latents[ind]
        return self.sample_frames(video_latent)
