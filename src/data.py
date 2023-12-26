import logging
import numpy as np
import torch
import torchvision
import random

from einops import rearrange
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def obtain_dataloader(batch_size, dataset_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class LatentImageDataset(Dataset):
    def __init__(self, latent_path, caption_path=None):
        std_latent = 2*(1/0.18215)

        image_latents = torch.Tensor(np.load(latent_path))
        image_latents = image_latents.clamp(-std_latent, std_latent) / std_latent
        # image_latents = (image_latents + 1) / 2
        self.image_latents = image_latents
        self.image_latents = rearrange(image_latents, 'b h w c -> b c h w')
        
        logger.info(f'{len(self.image_latents)} image samples loaded from {latent_path}.')

        '''
        self.text_embs = None
        if caption_path is not None:
            self.text_embs = torch.Tensor(np.load(caption_path))
            assert(len(self.text_embs) == len(self.image_latents))
            logger.info(f'{len(self.text_embs)} text embeddings loaded from {caption_path}.')
        '''

        self.prompts = None
        
        if caption_path is not None:
            prompts = []
            
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    prompts.append(line)

            prompts = prompts
            assert len(prompts) == len(self.image_latents), '# image latents and corresponding prompts should match.'
            self.prompts = prompts
            
            logger.info(f'{len(self.prompts)} text prompts loaded from {caption_path}.')

    def __len__(self):
        return len(self.image_latents)

    def __getitem__(self, ind):
        ret = {'data': self.image_latents[ind]}
        if self.prompts is not None:
            ret.update({'context': self.prompts[ind]})
        return ret

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
        return {'data': self.sample_frames(video_latent)}

if __name__ == '__main__':
    latent_image_path = './data/landscape_and_portrait/16_16_latent_embeddings.npy'
    prompts_path = './data/landscape_and_portrait/captions.txt'
    latent_image_dataset = LatentImageDataset(latent_image_path, prompts_path)
    dataloader = DataLoader(latent_image_dataset, shuffle=True, batch_size=64)
    print(next(iter(dataloader)))
