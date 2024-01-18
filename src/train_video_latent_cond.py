import argparse
import logging
import os
import sys
import time
import torch

from accelerate import Accelerator
from einops import rearrange
from random import randint, random
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPTextModel

sys.path.append('./')

from src.models.unet3d import UNet3D
from src.data import LatentImageDataset, LatentVideoDataset
from src.util import save_image_grid

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
attention_levels = [0, 1, 1]
block_depth = 3
n_noise_steps = 1000

# Accelerator initialization
accelerator = Accelerator()
device = accelerator.device

def prepare_noise_scheduler():
    beta_start = 1e-4
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, n_noise_steps, device=device)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, beta, alpha_hat

alpha, beta, alpha_hat = prepare_noise_scheduler()

def noise_images(img, t):
    noise_sampled = torch.randn_like(img, device=device)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1-alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * img + sqrt_one_minus_alpha_hat * noise_sampled, noise_sampled

def sample(
    model, 
    tokenizer, 
    text_encoder,
    num_frames,
    image_only_indicator,
    cfg_scale, 
    prompts
):
    '''
    sample `model` for each prompt for `n` times.
    '''
    model.eval()
    start = time.time()

    # encode input prompt string into textual latent space
    text_inputs = tokenizer(text=prompts, padding=True, truncation=True, max_length=text_encoder.config.max_position_embeddings, return_tensors="pt")
    text_embeddings = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
    batch_size = len(text_embeddings)

    # unconditional text embeddings for classifier free guidance
    max_length = text_inputs.input_ids.shape[-1]
    uncond_input = tokenizer(text=[""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    if image_only_indicator:
        num_frames = 1
        
    sampled_latents = torch.randn(
        (
            batch_size*num_frames,
            input_channels,
            latent_size,
            latent_size
        ), 
        dtype=torch.float, 
        device=device
    )

    for timestep in reversed(range(1, n_noise_steps)):
        # expand the latents for classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([sampled_latents] * 2)
        
        t = torch.ones(batch_size*num_frames, dtype=torch.long, device=device) * timestep

        with torch.no_grad():
            noise_pred = model(latent_model_input, torch.cat([t, t])[:, None], image_only_indicator, text_embeddings)
        
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
        alpha_t = alpha[t][:, None, None, None]
        alpha_hat_t = alpha_hat[t][:, None, None, None]
        beta_t = beta[t][:, None, None, None]

        if timestep > 1:
            noise = torch.randn_like(sampled_latents)
        else:
            noise = torch.zeros_like(sampled_latents)
            
        sampled_latents = 1 / torch.sqrt(alpha_t) * (sampled_latents - ((1 - alpha_t) / (torch.sqrt(1 - alpha_hat_t))) * noise_pred) + torch.sqrt(beta_t) * noise

    model.train()
    logger.info(f'Done sampling {batch_size} latents, time elapsed: {time.time() - start: .3f}s.')
    return sampled_latents

def train(model, optimizer, dataloader, eval_prompts, args):
    loss_fn = torch.nn.MSELoss()
    text_encoder = CLIPTextModel.from_pretrained("./clip-vit-base-patch32").eval().to(device)
    tokenizer = AutoProcessor.from_pretrained("./clip-vit-base-patch32")

    if args.image_only:
        image_only_indicator=torch.Tensor([True]).to(device)
    else:
        image_only_indicator=torch.Tensor([False]).to(device)
    
    for epoch in range(args.n_epoch):
        batch_cnt = 0
        total_loss = 0
        model.train()
        start = time.time()
        
        for ind, batch in enumerate(dataloader):
            batch_cnt += 1

            # classifier-free guidance
            if random() < args.uncondition_prob:
                prompts = [''] * len(batch['data'])
            else:
                prompts = batch['context']
                
            with torch.no_grad():
                text_inputs = tokenizer(text=prompts, padding=True, truncation=True, max_length=text_encoder.config.max_position_embeddings, return_tensors="pt")
                text_embeddings = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
                
            batch_data = batch['data']
            if not args.image_only:
                batch_data = rearrange(batch_data, "b t c h w -> (b t) c h w")
                
            # randomly sample a timestep for each image in the batch
            t = torch.randint(low=1, high=n_noise_steps, size=(len(batch_data), ), device=device)
            # perturb each image by noise step t
            perturbed_images, noises_sampled = noise_images(batch_data, t)
            # noise predition through UNet
            pred = model(perturbed_images, t[:, None], image_only_indicator, context=text_embeddings)
            # calculate loss
            batch_loss = loss_fn(noises_sampled, pred)
            total_loss += batch_loss
    
            # backward pass
            optimizer.zero_grad()
            accelerator.backward(batch_loss)
            optimizer.step()

        logger.info(f'[{epoch+1}|{args.n_epoch}] Training finished, time elapsed: {time.time()-start: .3f}s, total loss: {total_loss/batch_cnt: .3f}.')
        
        if (epoch+1) % args.sample_epoch == 0:
            sampled_latents = sample(
                model, 
                tokenizer, 
                text_encoder,
                args.n_frames,
                image_only_indicator,
                args.cfg_ratio, 
                eval_prompts
            )
            torch.save(sampled_latents, os.path.join(args.result_path, f'sampled_latents_epoch_{epoch+1}.pt'))
            # save_image_grid(sampled_images, os.path.join(args.result_path, f'samples_epoch_{epoch+1}.png'))

        if (epoch+1) % args.checkpoint_epoch == 0:
            model_save_path = os.path.join(args.checkpoint_path, f'epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Model saved to {model_save_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--bucket_size', type=int, default=3, help='a random frame is sampled from each bucket')
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--cfg_ratio', type=float, default=3.5, help='hyperparameter for classifier-free guidance')
    parser.add_argument('--checkpoint_epoch', type=int, default=50, help='save model ckpt every specified epoches')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--condition_path', type=str, default='./data/WebVid-10M-latent/prompts.txt')
    parser.add_argument('--dataset_path', type=str, default='./data/WebVid-10M-latent/sampled_20000.npy')
    parser.add_argument('--frames_per_video_path', type=str, default='./data/WebVid-10M-latent/frames_per_video.npy')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--n_frames', type=int, default=8, help='# frames sampled from each video')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--sample_epoch', type=int, default=10)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--uncondition_prob', type=float, default=0.15, help='unconditional probability')
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

    if args.condition_path is None or args.dataset_path is None or args.frames_per_video_path is None:
        logger.error('condition path and dataset path and frames_per_video path must be specified, but got None. Abort.')
        exit(-1)

    if args.image_only:
        latent_dataset = LatentImageDataset(args.dataset_path, args.condition_path)
    else:
        latent_dataset = LatentVideoDataset(
            args.dataset_path, 
            args.frames_per_video_path, 
            args.condition_path, 
            args.bucket_size, 
            args.n_frames
        )
        
    dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)
    eval_prompts = latent_dataset.get_prompts()[:args.eval_batch_size]
        
    widths = [model_channels * mult for mult in channels_mults]
    model = UNet3D(
        block_depth, 
        widths, 
        attention_levels, 
        input_channels,
        output_channels,
        device,
        num_frames=args.n_frames,
        context_channels=512
    )
    
    if not args.use_checkpoint is None:
        try:
            ckpt = torch.load(args.use_checkpoint)
            model.load_state_dict(ckpt)
            logger.info(f'Model loaded from checkpoint {args.use_checkpoint}.')
        except:
            logger.warning(f'Failed to load model from checkpoint {args.use_checkpoint}. Train from scratch instead.')

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optim, dataloader = accelerator.prepare(
        model, optim, dataloader
    )

    start = time.time()
    logger.info(f'Start training task {args.task_name} for {args.n_epoch} epoches, with batch size {args.batch_size}.')
    train(model, optim, dataloader, eval_prompts, args)
    logger.info(f'Finish training task {args.task_name} after {args.n_epoch} epoches, time eplased: {time.time() - start: .3f}s.')

    model_save_path = os.path.join(args.checkpoint_path, f'epoch_{args.n_epoch}.pt')
    torch.save(model.state_dict(), model_save_path)
    logger.info(f'Model saved to {model_save_path}.')
