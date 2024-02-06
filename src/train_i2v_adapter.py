import argparse
import logging
import os
import sys
import time
import torch

from diffusers import AutoencoderKL, DDPMScheduler, MotionAdapter, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from einops import rearrange
from random import randint, random
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append('./')

from src.models.unet_motion_cross_frame_attn import UNetMotionCrossFrameAttnModel
from src.data import VideoDataset, WebVid10M

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Accelerator initialization
accelerator = Accelerator()
device = accelerator.device

def train(
    model, 
    text_encoder, 
    tokenizer, 
    vae,
    noise_scheduler,
    optimizer, 
    dataloader, 
    args
):
    loss_fn = torch.nn.MSELoss(reduction='none')

    for epoch in range(args.n_epoch):
        batch_cnt = 0
        total_loss = 0
        model.train()
        start = time.time()
        
        for ind, batch in enumerate(dataloader):
            batch_cnt += 1

            # classifier-free guidance
            if random() < args.uncondition_prob:
                prompts = [''] * len(batch['pixel_values'])
            else:
                prompts = batch['text']
            
            batch_size = len(batch['pixel_values'])
            batch_data = rearrange(batch['pixel_values'], 'b f c h w -> (b f) c h w')
                
            with torch.no_grad():
                text_inputs = tokenizer(text=prompts, padding=True, truncation=True, max_length=text_encoder.config.max_position_embeddings, return_tensors="pt")
                text_embeddings = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
                hidden_states = vae.encode(batch_data).latent_dist.sample() * vae.config.scaling_factor
                hidden_states = rearrange(hidden_states, '(b f) c h w -> b f c h w', b=batch_size)

            # randomly sample a timestep for each video in the batch
            timesteps = torch.randint(low=1, high=noise_scheduler.config.num_train_timesteps, size=(batch_size, ), device=device)
            
            # perturb each video by noise step t except for the first frame
            noises_sampled = torch.randn(hidden_states.shape, device=device)
            noises_sampled[:, 0] = 0
            hidden_states = noise_scheduler.add_noise(hidden_states, noises_sampled, timesteps)
            
            # noise predition through UNet
            pred = model(
                hidden_states, 
                timesteps, 
                enable_cross_frame_attn=True,
                encoder_hidden_states=text_embeddings,
                return_dict=False
             )[0]
            
            # calculate loss without first frame
            non_first_frame_mask = torch.ones(noises_sampled.shape, device=device)
            non_first_frame_mask[:, 0] = 0

            batch_loss = loss_fn(noises_sampled, pred)
            batch_loss = (batch_loss * non_first_frame_mask.float()).sum()
            batch_loss = batch_loss / non_first_frame_mask.sum()
            total_loss += batch_loss
    
            # backward pass
            optimizer.zero_grad()
            accelerator.backward(batch_loss)
            optimizer.step()

        logger.info(f'[{epoch+1}|{args.n_epoch}] Training finished, time elapsed: {time.time()-start: .3f}s, total loss: {total_loss/batch_cnt: .3f}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--bucket_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--cfg_ratio', type=float, default=3.5, help='hyperparameter for classifier-free guidance')
    parser.add_argument('--checkpoint_epoch', type=int, default=50, help='save model ckpt every specified epoches')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--condition_path', type=str, default='./data/WebVid-10M/results_2M_val.csv')
    parser.add_argument('--dataset_path', type=str, default='./data/WebVid-10M/data/videos')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--image_size', type=int, default=256, help='image isze for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--n_frames', type=int, default=16, help='# frames sampled from each video')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--sample_epoch', type=int, default=10)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--uncondition_prob', type=float, default=0.15, help='unconditional probability')
    parser.add_argument('--use_checkpoint', type=str, default=None, help='checkpoint file to be loaded.')

    args = parser.parse_args()
    
    if args.task_name is None:
        args.task_name = f'I2VAdapter_{randint(100000, 999999)}'
        
    args.result_path = os.path.join(args.result_path, args.task_name)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    # initialize unet model from pretrained animate-diff
    adapter = MotionAdapter.from_pretrained('./animatediff-motion-adapter-v1-5-2')
    unet2d = UNet2DConditionModel.from_pretrained('./stable-diffusion-v1-5/unet')
    unet = UNetMotionCrossFrameAttnModel.from_unet2d(unet2d, adapter)
    unet.freeze_animatediff_params()

    n_trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6
    logger.info(f'unet loaded from pretrained animatediff, with {n_trainable_params}M trainable parameters.')

    # prepare noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # prepare lr scheduler
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # prepare text encoder and vae
    text_encoder = CLIPTextModel.from_pretrained('./stable-diffusion-v1-5/text_encoder').eval()
    tokenizer = CLIPTokenizer.from_pretrained('./stable-diffusion-v1-5/tokenizer')
    vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5/vae').eval()

    # prepare video dataset
    video_dataset = WebVid10M(
        csv_path=args.condition_path,
        video_folder=args.dataset_path,
        sample_size=args.image_size,
        sample_stride=args.bucket_size, 
        sample_n_frames=args.n_frames,
        is_image=False,
    )
    dataloader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=16,)
    
    unet, text_encoder, vae, optim, dataloader = accelerator.prepare(
        unet, text_encoder, vae, optim, dataloader
    )

    # start training
    start = time.time()
    logger.info(f'Start training task {args.task_name} for {args.n_epoch} epoches, with batch size {args.batch_size}.')
    
    train(
        unet, 
        text_encoder, 
        tokenizer, 
        vae, 
        noise_scheduler,
        optim, 
        dataloader, 
        args
    )

    logger.info(f'Finish training task {args.task_name} after {args.n_epoch} epoches, time eplased: {time.time() - start: .3f}s.')

    # save final model checkpoint to local storage
    model_save_path = os.path.join(args.checkpoint_path, f'epoch_{args.n_epoch}.pt')
    torch.save(unet.state_dict(), model_save_path)
    logger.info(f'Model saved to {model_save_path}.')
    