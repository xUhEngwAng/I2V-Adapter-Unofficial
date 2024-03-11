import os, csv, random
import logging
import numpy as np
import torch
import torchvision

from decord import VideoReader
from einops import rearrange
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor

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
        self.image_latents = image_latents
        self.image_latents = rearrange(image_latents, 'b h w c -> b c h w')
        
        logger.info(f'{len(self.image_latents)} image samples loaded from {latent_path}.')

        self.prompts = None 
        if caption_path is not None:
            prompts = []
            
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    prompts.append(line)

            assert len(prompts) == len(self.image_latents), '# image latents and corresponding prompts should match.'
            self.prompts = prompts
            
            logger.info(f'{len(self.prompts)} text prompts loaded from {caption_path}.')

    def get_prompts(self):
        return self.prompts

    def __len__(self):
        return len(self.image_latents)

    def __getitem__(self, ind):
        ret = {'data': self.image_latents[ind]}
        if self.prompts is not None:
            ret.update({'context': self.prompts[ind]})
        return ret

# code adapter from 
# https://github.com/guoyww/AnimateDiff/blob/main/animatediff/data/dataset.py
class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, 
            video_folder,
            sample_size=256, 
            sample_stride=4, 
            sample_n_frames=16,
            is_image=False,
        ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        logger.info(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(sample_size[0], antialias=True),
            torchvision.transforms.CenterCrop(sample_size),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.clip_image_processor = CLIPImageProcessor()
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        if not self.is_image:
            clip_image = self.clip_image_processor(images=pixel_values[0], return_tensors="pt").pixel_values[0]
        else:
            clip_image = self.clip_image_processor(images=pixel_values, return_tensors="pt").pixel_values[0]
            
        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(
            clip_image=clip_image,
            pixel_values=pixel_values, 
            text=name
        )
        return sample

class LatentVideoDataset(Dataset):
    def __init__(
        self, 
        latent_path, 
        frames_per_video_path, 
        caption_path, 
        bucket_size, 
        num_frames=8
    ):
        super().__init__()

        self.bucket_size = bucket_size
        self.num_frames = num_frames

        std_latent = 2*(1/0.18215)
        video_latents = torch.Tensor(np.load(latent_path))
        video_latents = video_latents.clamp(-std_latent, std_latent) / std_latent
        frames_per_video = np.load(frames_per_video_path)
        frames_per_video_acc = np.hstack((0, frames_per_video.cumsum()))

        self.video_latents = [
            video_latents[frames_per_video_acc[ind]: frames_per_video_acc[ind+1]]
            for ind in range(len(frames_per_video)) 
            if bucket_size * num_frames <= frames_per_video[ind]
        ]
        
        logger.info(f'{len(self.video_latents)} video samples loaded from {latent_path}, with {len(frames_per_video) - len(self.video_latents)} filtered.')

        self.prompts = None 
        if caption_path is not None:
            prompts = []
            
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    prompts.append(line)

            # prompts = prompts[:1000]
            assert len(prompts) == len(frames_per_video), '# video latents and corresponding prompts should match.'
            self.prompts = [
                prompts[ind]
                for ind in range(len(frames_per_video)) 
                if bucket_size * num_frames <= frames_per_video[ind]
            ]
            
            logger.info(f'{len(self.prompts)} text prompts loaded from {caption_path}.')

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

    def get_prompts(self):
        return self.prompts

    def __len__(self):
        return len(self.video_latents)

    def __getitem__(self, ind):
        video_latent = self.video_latents[ind]
        ret = {'data': self.sample_frames(video_latent)}
        
        if self.prompts is not None:
            ret.update({'context': self.prompts[ind]})
            
        return ret

if __name__ == '__main__':
    '''
    latent_image_path = './data/landscape_and_portrait/16_16_latent_embeddings.npy'
    prompts_path = './data/landscape_and_portrait/captions.txt'
    latent_image_dataset = LatentImageDataset(latent_image_path, prompts_path)
    dataloader = DataLoader(latent_image_dataset, shuffle=True, batch_size=64)
    print(next(iter(dataloader)))
    '''

    root_dir = './data/WebVid-10M/data/videos'
    condition_path = './data/WebVid-10M/results_2M_val.csv'

    dataset = WebVid10M(
        csv_path=condition_path,
        video_folder=root_dir,
        sample_size=256,
        sample_stride=4, 
        sample_n_frames=16,
        is_image=False,
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)

    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
