import argparse
import logging
import os
import time
import torch

from einops import rearrange
from data import LatentVideoDataset, LatentImageDataset
from random import randint
from torch.utils.data import DataLoader
from unet3d import UNet3D
from util import save_image_grid

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# preset constants
latent_size = 16
input_channels = 4
output_channels = 4
model_channels = 128
channels_mults = [1, 2, 4]
attention_levels = [1, 1, 1]
block_depth = 1
n_noise_steps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_noise_scheduler():
    beta_start = 1e-4
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, n_noise_steps).to(device)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, beta, alpha_hat

alpha, beta, alpha_hat = prepare_noise_scheduler()

def noise_images(img, t):
    noise_sampled = torch.randn_like(img).to(device)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1-alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * img + sqrt_one_minus_alpha_hat * noise_sampled, noise_sampled

def decode_latents(vae, video_latents, num_frames):
    # TODO: 
    video_latents = rearrange(video_latents, "(b t) c h w -> b t c h w")
    ret = []
    
    with torch.no_grad():
        for latent in video_latents:
            decoded = vae.decode(video_latents, return_dict=False)[0]
            ret.append(decoded)

    return torch.cat(ret, axis=0)

def sample(image_only, model, n, num_frames=1):
    if image_only:
        image_only_indicator=torch.Tensor([True]).to(device)
    else:
        image_only_indicator=torch.Tensor([False]).to(device)
    
    model.eval()
    start = time.time()
    sampled_latents = torch.randn((n*num_frames, input_channels, latent_size, latent_size), dtype=torch.float).to(device)

    with torch.no_grad():
        for timestep in reversed(range(1, n_noise_steps)):
            t = torch.ones(n*num_frames, dtype=torch.long) * timestep
            pred = model(sampled_latents, t[:, None].to(device), image_only_indicator)
            alpha_t = alpha[t][:, None, None, None]
            alpha_hat_t = alpha_hat[t][:, None, None, None]
            beta_t = beta[t][:, None, None, None]
    
            if timestep > 1:
                noise = torch.randn_like(sampled_latents)
            else:
                noise = torch.zeros_like(sampled_latents)
                
            sampled_latents = 1/torch.sqrt(alpha_t)*(sampled_latents-((1-alpha_t)/(torch.sqrt(1-alpha_hat_t)))*pred)+torch.sqrt(beta_t)*noise

    model.train()
    logger.info(f'Done sampling {n*num_frames} latents, time elapsed: {time.time() - start: .3f}s.')
    return sampled_latents

def train(model, dataloader, args):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(args.n_epoch):
        batch_cnt = 0
        total_loss = 0
        model.train()
        start = time.time()
        
        if args.image_only:
            image_only_indicator=torch.Tensor([True]).to(device)
        else:
            image_only_indicator=torch.Tensor([False]).to(device)
        
        for ind, batch_images in enumerate(dataloader):
            batch_cnt += 1

            # expand each video to corresponding frames
            if not args.image_only:
                batch_images = rearrange(batch_images, "b t c h w -> (b t) c h w")
            
            batch_images = batch_images.to(device)
            # randomly sample a timestep for each image in the batch
            t = torch.randint(low=1, high=n_noise_steps, size=(len(batch_images), )).to(device)
            # perturb each image by noise step t
            perturbed_images, noises_sampled = noise_images(batch_images, t)
            # noise predition through UNet
            pred = model(perturbed_images, t[:, None], image_only_indicator)
            # calculate loss
            batch_loss = loss_fn(noises_sampled, pred)
            total_loss += batch_loss
    
            # backward pass
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        logger.info(f'[{epoch+1}|{args.n_epoch}] Training finished, time elapsed: {time.time()-start: .3f}s, total loss: {total_loss/batch_cnt: .3f}.')
        if (epoch+1) % args.sample_epoch == 0:
            sampled_latents = sample(args.image_only, model, 8, args.n_frames)
            torch.save(sampled_latents, os.path.join(args.result_path, f'sampled_latents_epoch_{epoch+1}.pt'))
            # save_image_grid(sampled_images, os.path.join(args.result_path, f'samples_epoch_{epoch+1}.png'))

        if (epoch+1) % args.checkpoint_epoch == 0:
            model_save_path = os.path.join(args.checkpoint_path, f'epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Model saved to {model_save_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bucket_size', type=int, default=3, help='a random frame is sampled from each bucket')
    parser.add_argument('--checkpoint_epoch', type=int, default=50, help='save model ckpt every specified epoches')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--dataset_path', type=str, default='./data/ucf101-latent/')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--n_frames', type=int, default=8, help='# frames sampled from each video')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--sample_epoch', type=int, default=10)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--use_checkpoint', type=str, default=None, help='checkpoint file to be loaded.')

    args = parser.parse_args()
    
    if args.task_name is None:
        args.task_name = f'align-your-latents_{randint(100000, 999999)}'
        
    args.result_path = os.path.join(args.result_path, args.task_name)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    if args.image_only:
        latent_image_dataset = LatentImageDataset(args.dataset_path)
        dataloader = DataLoader(latent_image_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        latent_video_dataset = LatentVideoDataset(args.dataset_path, args.bucket_size, args.n_frames)
        dataloader = DataLoader(latent_video_dataset, batch_size=args.batch_size, shuffle=True)

    widths = [model_channels * mult for mult in channels_mults]
    model = UNet3D(
        block_depth,
        widths,
        attention_levels,
        input_channels, 
        output_channels, 
        device, 
        num_frames=args.n_frames
    ).to(device)
    
    if not args.use_checkpoint is None:
        try:
            ckpt = torch.load(args.use_checkpoint)
            model.load_state_dict(ckpt)
            logger.info(f'Model loaded from checkpoint {args.use_checkpoint}.')
        except:
            logger.warning(f'Failed to load model from checkpoint {args.use_checkpoint}. Train from scratch instead.')

    start = time.time()
    logger.info(f'Start training task {args.task_name} for {args.n_epoch} epoches, with batch size {args.batch_size}.')
    train(model, dataloader, args)
    logger.info(f'Finish training task {args.task_name} after {args.n_epoch} epoches, time eplased: {time.time() - start: .3f}s.')

    model_save_path = os.path.join(args.checkpoint_path, f'epoch_{args.n_epoch}.pt')
    torch.save(model.state_dict(), model_save_path)
    logger.info(f'Model saved to {model_save_path}.')
