import argparse
import cv2
import numpy as np
import os
import random
import torch
import torchvision

from diffusers import AutoencoderKL
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        image_size, 
        video_exts=['mp4'], 
        prompt_exts=['txt'],
        max_frames=None
    ):
        super().__init__()

        self.video_paths = sorted([str(p) for ext in video_exts for p in Path(root_dir).glob(f'**/*.{ext}')])[:10000]
        self.prompts_paths = sorted([str(p) for ext in prompt_exts for p in Path(root_dir).glob(f'**/*.{ext}')])[:10000]
        self.max_frames = max_frames

        assert len(self.video_paths) == len(self.prompts_paths), f'# videos and their corresponding prompts should match, but found {len(self.video_paths)} and {len(self.prompts_paths)} respectively.'
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def seek_all_frames(self, path):
        vidcap = cv2.VideoCapture(path)
        ret, frame = vidcap.read()
        frames = []
        
        while ret and len(frames) < self.max_frames:
            frames.append(frame)
            ret, frame = vidcap.read()
            
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, ind):
        video_path = self.video_paths[ind]
        tensors = tuple(map(self.transform, self.seek_all_frames(video_path)))
        
        prompts_path = self.prompts_paths[ind]
        with open(prompts_path, 'r') as f:
            prompt = f.readline().strip()
        return video_path, torch.stack(tensors, dim=0), prompt

def _encode(vae, video_clip):
    with torch.no_grad():
        diag_gaussian_dist = vae.encode(video_clip, return_dict=False)[0]
        encoded = diag_gaussian_dist.sample()
        
    return encoded

def encode_video(vae, video, num_frames):
    '''
    Encode a single video file through a pre-trained vae.
    The result is saved to local storage for future usage.
    As the memory consumption is high when performing encoding, 
    we split the video to multiple slices and encode them one
    at a time.
    '''
    res = []
    
    for start_frame in range(0, len(video), num_frames):
        encoded = _encode(vae, video[start_frame: start_frame+num_frames])
        res.append(encoded)

    res = torch.concat(res, dim=0).cpu().numpy()
    return res

def save_to_local_storage(target_dir, video_encoded, prompts, frames_per_video):
    video_target_path = os.path.join(target_dir, f'sampled_{len(frames_per_video)}.npy')
    np.save(video_target_path, video_encoded)

    frames_per_video_path = os.path.join(target_dir, 'frames_per_video.npy')
    np.save(frames_per_video_path, frames_per_video)

    prompts_path = os.path.join(target_dir, 'prompts.txt')
    with open(prompts_path, 'w') as f:
        f.writelines(prompts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=128, help='adjust this argument to fit in your own device')
    parser.add_argument('--video_src_dir', type=str, default='./data/WebVid-10M/')
    parser.add_argument('--video_target_dir', type=str, default='./data/WebVid-10M-latent/')
    parser.add_argument('--max_frames', type=int, default=128)

    args = parser.parse_args()

    if not os.path.exists(args.video_target_dir):
        os.mkdir(args.video_target_dir)

    video_dataset = VideoDataset(args.video_src_dir, args.image_size, max_frames=args.max_frames)
    print(f'{len(video_dataset)} videos loaded from {args.video_src_dir}.')
    
    vae = AutoencoderKL.from_pretrained('./sd-vae-ft-ema').eval()
    vae.to(device)
    
    encoded = []
    prompts = ''
    video_num_frames = []

    for path, video_tensors, prompt in tqdm(video_dataset):
        # TODO: distribute the computation to multi-GPU by video folders
        try:
            res = encode_video(vae, video_tensors.to(device), args.num_frames)
            video_num_frames.append(len(res))
            encoded.append(res)
            prompts += prompt + '\n'
        except Exception as e:
            print(f'Failed to encode video {path}, with the following exception: {e}')

    prompts = prompts[:-1]
    assert len(encoded) == len(video_num_frames)
    
    encoded =  np.concatenate(encoded)
    frames_per_video = np.array(video_num_frames)
    save_to_local_storage(args.video_target_dir, encoded, prompts, frames_per_video)
    
    print(f'Done encoding a total of {len(video_dataset)} videos and saving the latents to {args.video_target_dir}.')
