import argparse
import cv2
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
    def __init__(self, root_dir, image_size, exts=['avi']):
        super().__init__()

        self.paths = [str(p) for ext in exts for p in Path(root_dir).glob(f'**/*.{ext}')]
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @staticmethod
    def _seek_all_frames(path):
        vidcap = cv2.VideoCapture(path)
        ret, frame = vidcap.read()
        frames = []
        
        while ret:
            frames.append(frame)
            ret, frame = vidcap.read()
    
        return frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ind):
        path = self.paths[ind]
        tensors = tuple(map(self.transform, self._seek_all_frames(path)))
        return path, torch.stack(tensors, dim=0)

def _encode(vae, video_clip):
    with torch.no_grad():
        diag_gaussian_dist = vae.encode(video_clip, return_dict=False)[0]
        encoded = diag_gaussian_dist.sample()
        
    return encoded

def encode_and_save(vae, video, num_frames, save_dir, video_path):
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

    res = torch.concat(res, dim=0)
    
    sub_dir, file_name = video_path.split('/')[-2], video_path.split('/')[-1]
    video_name = file_name.split('.')[-2]
    save_dir = os.path.join(save_dir, sub_dir)
    
    if not os.path.exists(save_dir):
        print(f'Directory {save_dir} doesnot exists, making...')
        os.mkdir(save_dir)
        
    save_path = os.path.join(save_dir, f'{video_name}.pt')
    torch.save(res, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=128, help='adjust this argument to fit in your own device')
    parser.add_argument('--video_src_dir', type=str, default='./data/ucf101/')
    parser.add_argument('--video_target_dir', type=str, default='./data/ucf101-latent/')

    args = parser.parse_args()

    if not os.path.exists(args.video_target_dir):
        os.mkdir(args.video_target_dir)

    video_dataset = VideoDataset(args.video_src_dir, args.image_size)
    print(f'{len(video_dataset)} videos loaded from {args.video_src_dir}.')
    
    vae = AutoencoderKL.from_pretrained('./sd-vae-ft-ema').eval()
    vae.to(device)

    for path, video_tensors in tqdm(video_dataset):
        # TODO: distribute the computation to multi-GPU by video folders
        encode_and_save(vae, video_tensors.to(device), args.num_frames, args.video_target_dir, path)

    print(f'Done encoding a total of {len(video_dataset)} videos and saving the latents to {args.video_target_dir}.')
