import argparse
import numpy as np
import os
import torch
import torchvision
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def obtain_dataloader(batch_size, dataset_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'{len(dataset)} images loaded from {dataset_path}.')
    return dataloader

def encode_and_save(vae, batch_images, save_dir):
    res = []
    
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset_path', type=str, default='./data/color_symbol_7k/128/')
    parser.add_argument('--target_path', type=str, default='./data/color_symbol_7k_latent.npy')

    args = parser.parse_args()
    
    dataloader = obtain_dataloader(args.batch_size, args.dataset_path)
    vae = AutoencoderKL.from_pretrained('./sd-vae-ft-ema').eval()
    vae.to(device)

    encoded = []

    for ind, (batch_images, _) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            diag_gaussian_dist = vae.encode(batch_images.to(device), return_dict=False)[0]
            encoded.append(diag_gaussian_dist.sample().cpu().numpy())

    encoded = np.concatenate(encoded)
    print(encoded.shape)
    np.save(args.target_path, encoded)
    print(f'Done encoding a total of {len(encoded)} images and saving the latents to {args.target_path}.')
    