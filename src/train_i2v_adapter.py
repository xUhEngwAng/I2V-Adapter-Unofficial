import argparse
import logging
import os
import pandas as pd
import random
import sys
import time
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)
logger = logging.getLogger(__name__)
sys.path.append('./')

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    MotionAdapter,
    UNet2DConditionModel
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import export_to_gif

from PIL import Image
from accelerate import Accelerator
from einops import rearrange
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from src.data import WebVid10M
from src.models.unet_motion_cross_frame_attn import UNetMotionCrossFrameAttnModel
from src.modules.i2v_adapter import I2VAdapterModule
from src.pipelines.pipeline_i2v_adapter import I2VAdapterPipeline

# Accelerator initialization
accelerator = Accelerator(gradient_accumulation_steps=16)
device = accelerator.device

def train_loop(
    args,
    dataloader,
    optimizer,
    i2v_adapter_pipeline,
    vae,
    text_encoder,
    tokenizer,
    model,
    noise_scheduler,
    image_encoder
):
    loss_fn = torch.nn.MSELoss(reduction='none')

    for epoch in range(args.n_epoch):
        batch_cnt = 0
        total_loss = 0
        model.train()
        start = time.time()

        for ind, batch in enumerate(dataloader):
            batch_cnt += 1
            batch_size = len(batch['pixel_values'])
            batch_data = rearrange(batch['pixel_values'], 'b f c h w -> (b f) c h w')

            # classifier-free guidance
            prompts = batch['text']
            rand_num = random.random()

            if rand_num < args.uncond_prob_t + args.uncond_prob_ti:
                prompts = [''] * batch_size

            with torch.no_grad():
                # prepare text clip embedding
                text_inputs = tokenizer(
                    text=prompts, padding=True, truncation=True, max_length=text_encoder.config.max_position_embeddings, return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]

                hidden_states = vae.encode(batch_data).latent_dist.sample() * vae.config.scaling_factor
                hidden_states = rearrange(hidden_states, '(b f) c h w -> b f c h w', b=batch_size)

                # prepare image clip embedding for IP-Adapter
                image_embeds = image_encoder(batch['clip_image'].to(device)).image_embeds
                if args.uncond_prob_t < rand_num and \
                   rand_num < args.uncond_prob_t + args.uncond_prob_i + args.uncond_prob_ti:
                    image_embeds = torch.zeros_like(image_embeds)
                    # hidden_states[:, 0] = torch.zeros_like(hidden_states[:, 0])

                added_cond_kwargs = {'image_embeds': image_embeds}

            # randomly sample a timestep for each video in the batch
            timesteps = torch.randint(low=1, high=noise_scheduler.config.num_train_timesteps, size=(batch_size, ), device=device)

            # perturb each video by noise step t except for the first frame
            first_frames_latents = hidden_states[:, 0]
            noises_sampled = torch.randn(hidden_states.shape, device=device)
            noises_sampled[:, 0] = 0
            hidden_states = noise_scheduler.add_noise(hidden_states, noises_sampled, timesteps)
            hidden_states[:, 0] = first_frames_latents

            with accelerator.accumulate(model):
                # noise predition through UNet
                pred = model(
                    hidden_states,
                    timesteps,
                    enable_cross_frame_attn=True,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]

                # calculate loss without the first frame
                first_frame_mask = torch.ones(noises_sampled.shape, device=device)
                first_frame_mask[:, 0] = 0

                batch_loss = loss_fn(noises_sampled, pred)
                batch_loss = (batch_loss * first_frame_mask.float()).sum()
                batch_loss = batch_loss / first_frame_mask.sum()
                total_loss += batch_loss

                # backward pass
                optimizer.zero_grad()
                accelerator.backward(batch_loss)
                optimizer.step()

        logger.info(f'[{epoch+1}|{args.n_epoch}] Training finished, time elapsed: {time.time()-start: .3f}s, total loss: {total_loss/batch_cnt: .3f}.')

        if (epoch+1) % args.sample_epoch == 0:
            i2v_adapter = accelerator.unwrap_model(model).obtain_i2v_adapter_modules().to(device)
            i2v_adapter_pipeline.load_i2v_adapter(i2v_adapter)
            eval_batch_size = len(args.eval_prompts)

            output = i2v_adapter_pipeline(
                prompt=args.eval_prompts[0],
                condition_image=args.condition_images[0],
                ip_adapter_image=args.condition_images[0],
                negative_prompt="bad quality, worse quality",
                num_frames=args.n_frames,
                guidance_scale=args.cfg_ratio,
                num_inference_steps=25,
            )
            sampled_videos = output.frames
            sample_save_dir = os.path.join(args.result_path, f'epoch_{epoch+1}')
            if not os.path.exists(sample_save_dir):
                os.mkdir(sample_save_dir)

            for ind, frames in enumerate(sampled_videos):
                sample_save_path = os.path.join(sample_save_dir, f'sample_{ind}.gif')
                export_to_gif(frames, sample_save_path)

        if (epoch+1) % args.checkpoint_epoch == 0:
            model_save_path = os.path.join(args.checkpoint_path, f'epoch_{epoch+1}')
            accelerator.unwrap_model(model).save_i2v_adapter_modules(model_save_path)
            logger.info(f'I2V-Adapter Module saved to {model_save_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--bucket_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_data_path', type=str, default='./data/WebVid-10M/eval.csv')
    parser.add_argument('--cfg_ratio', type=float, default=7.5, help='hyperparameter for classifier-free guidance')
    parser.add_argument('--checkpoint_epoch', type=int, default=10, help='save model ckpt every specified epoches')
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
    parser.add_argument('--uncond_prob_i', type=float, default=0.05, help='unconditional probability for image')
    parser.add_argument('--uncond_prob_t', type=float, default=0.05, help='unconditional probability for text prompt')
    parser.add_argument('--uncond_prob_ti', type=float, default=0.05, help='unconditional probability for both image and text')
    parser.add_argument('--use_checkpoint', type=str, default=None, help='checkpoint file to be loaded.')

    args = parser.parse_args()

    if args.task_name is None:
        args.task_name = f'I2VAdapter_{random.randint(100000, 999999)}'

    args.result_path = os.path.join(args.result_path, args.task_name)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    # load evaluation prompts and conditioning images from eval_data_path
    eval_data_dir = os.path.dirname(args.eval_data_path)
    eval_data_df = pd.read_csv(args.eval_data_path)
    condition_images = []

    for image_path in eval_data_df['image_path']:
        abs_image_path = os.path.join(eval_data_dir, image_path)
        condition_images.append(Image.open(abs_image_path))

    args.condition_images = condition_images
    args.eval_prompts = eval_data_df['name'].tolist()

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

    # load various adapters
    motion_adapter = MotionAdapter.from_pretrained('./animatediff-motion-adapter-v1-5-2')

    ip_adapter = torch.load('./IP-Adapter/models/ip-adapter_sd15.bin', map_location='cpu')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        './IP-Adapter/', subfolder=Path('models', "image_encoder").as_posix(),
    ).eval()
    feature_extractor = CLIPImageProcessor()

    i2v_adapter = None
    if args.use_checkpoint is not None:
        i2v_adapter = I2VAdapterModule.from_pretrained(args.use_checkpoint)
        logger.info(f'Successfully loaded I2VAdapterModule from {args.use_checkpoint}.')

    # load unet from pretrained animatediff model
    model_path = './SG161222_Realistic_Vision_V5.1_noVAE/'
    unet2d = UNet2DConditionModel.from_pretrained(os.path.join(model_path, 'unet'))
    unet = UNetMotionCrossFrameAttnModel.from_unet2d(unet2d, motion_adapter, i2v_adapter)
    unet._load_ip_adapter_weights(ip_adapter)
    unet.freeze_animatediff_params()

    n_trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6
    logger.info(f'unet loaded from pretrained animatediff, with {n_trainable_params}M trainable parameters.')

    # prepare noise scheduler and optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # prepare I2VAdapterPipeline for sampling
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_path, 'text_encoder')).eval()
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
    vae = AutoencoderKL.from_pretrained(os.path.join(model_path, 'vae')).eval()
    scheduler = DDIMScheduler.from_pretrained(
        model_path, subfolder='scheduler', clip_sample=False, timestep_spacing='linspace', steps_offset=1
    )
    i2v_adapter = unet.obtain_i2v_adapter_modules()
    pipe = I2VAdapterPipeline(
        vae,
        text_encoder,
        tokenizer,
        unet2d,
        motion_adapter,
        i2v_adapter,
        scheduler,
        feature_extractor,
        image_encoder,
    )
    pipe.load_ip_adapter('./IP-Adapter/', subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.to(device)

    # sample pipeline
    output = pipe(
        prompt=args.eval_prompts[0],
        condition_image=args.condition_images[0],
        ip_adapter_image=args.condition_images[0],
        negative_prompt="bad quality, worse quality",
        num_frames=args.n_frames,
        guidance_scale=args.cfg_ratio,
        num_inference_steps=25,
        generator=torch.Generator(device=device).manual_seed(42),
    )

    sampled_videos = output.frames
    sample_save_dir = os.path.join(args.result_path, 'epoch_0')
    if not os.path.exists(sample_save_dir):
        os.mkdir(sample_save_dir)

    for ind, frames in enumerate(sampled_videos):
        sample_save_path = os.path.join(sample_save_dir, f'sample_{ind}.gif')
        export_to_gif(frames, sample_save_path)

    # save I2VAdapterModule on first initialization
    model_save_path = os.path.join(args.checkpoint_path, 'epoch_0')
    accelerator.unwrap_model(unet).save_i2v_adapter_modules(model_save_path)
    logger.info(f'Model saved to {model_save_path}.')

    # prepare unet for training
    unet, optim, dataloader = accelerator.prepare(
        unet, optim, dataloader
    )

    # start training
    start = time.time()
    logger.info(f'Start training task {args.task_name} for {args.n_epoch} epoches, with batch size {args.batch_size}.')

    train_loop(
        args,
        dataloader,
        optim,
        pipe,
        vae,
        text_encoder,
        tokenizer,
        unet,
        noise_scheduler,
        image_encoder
    )

    logger.info(f'Finish training task {args.task_name} after {args.n_epoch} epoches, time eplased: {time.time() - start: .3f}s.')

    # save final model checkpoint to local storage
    model_save_path = os.path.join(args.checkpoint_path, f'epoch_{args.n_epoch}')
    accelerator.unwrap_model(unet).save_i2v_adapter_modules(model_save_path)
    logger.info(f'Model saved to {model_save_path}.')

