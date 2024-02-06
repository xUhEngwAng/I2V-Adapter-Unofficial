from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import numpy as np
import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.unet_motion_model import MotionAdapter
from diffusers.pipelines.animatediff import AnimateDiffPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor, 
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPVisionModelWithProjection
)

from src.models.unet_motion_cross_frame_attn import UNetMotionCrossFrameAttnModel
from src.modules.i2v_adapter import I2VAdapterModule

def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    batch_size, num_frames, channels, height, width = video.shape
    outputs = []

    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx]
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

@dataclass
class I2VAdapterlineOutput(BaseOutput):
    frames: Union[torch.Tensor, np.ndarray]

class I2VAdapterPipeline(AnimateDiffPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        motion_adapter: MotionAdapter,
        i2v_adapter: I2VAdapterModule,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder
        )

        unet = UNetMotionCrossFrameAttnModel.from_unet2d(unet, motion_adapter, i2v_adapter)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            i2v_adapter=i2v_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

    def check_inputs(
        self,
        prompt,
        condition_image,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if condition_image is not None and (
            not isinstance(condition_image, PIL.Image.Image)
            and not isinstance(condition_image, list)
        ):
            raise ValueError(
                "`image` has to be of type `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from 
    # diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        num_frames, 
        height, 
        width, 
        dtype, 
        device, 
        generator, 
        latents=None
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, num_frames, channels, height, width = latents.shape
        latents = latents.reshape(batch_size * num_frames, channels, height, width)
        image = self.vae.decode(latents).sample
        
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        condition_image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):
        has_condition_image = condition_image is not None

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, condition_image, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_videos_per_prompt, output_hidden_state
            )
            if do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 3.5 Encode conditioning image
        if has_condition_image:
            condition_image = self.image_processor.preprocess(condition_image, height=height, width=width).to(device)
            condition_image_latents = self.vae.encode(condition_image).latent_dist.sample() * self.vae.config.scaling_factor

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # replace the first frame with the conditioning image
                if has_condition_image:
                    latents[:, 0] = condition_image_latents

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    enable_cross_frame_attn=has_condition_image,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        print(f'latents.shape={latents.shape}, condition_image_latents.shape={condition_image_latents.shape}')

        if has_condition_image:
            latents[:, 0] = condition_image_latents
        
        if output_type == "latent":
            return I2VAdapterlineOutput(frames=latents)

        # Post-processing
        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return I2VAdapterlineOutput(frames=video)

if __name__ == '__main__':
    from diffusers.utils import export_to_gif

    model_path = './stable-diffusion-v1-5'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_adapter = MotionAdapter.from_pretrained('./animatediff-motion-adapter-v1-5-2')
    unet2d = UNet2DConditionModel.from_pretrained('./stable-diffusion-v1-5/unet')
    text_encoder = CLIPTextModel.from_pretrained('./stable-diffusion-v1-5/text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained('./stable-diffusion-v1-5/tokenizer')
    vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5/vae')
    scheduler = DDIMScheduler.from_pretrained(
        model_path, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
    )

    i2v_adapter_pipeline = I2VAdapterPipeline(
        vae,
        text_encoder,
        tokenizer,
        unet2d,
        motion_adapter,
        None,
        scheduler,
    ).to(device)

    # enable memory savings
    i2v_adapter_pipeline.enable_vae_slicing()
    i2v_adapter_pipeline.enable_model_cpu_offload()

    # load conditioning image
    condition_image = PIL.Image.open('./data/WebVid-10M/eval_condition_images/mulatto_woman.jpg')

    output = i2v_adapter_pipeline(
        prompt=(
            "mulatto woman slowing turning her head towards the camera, bestquality, cinematic"
        ),
        condition_image=condition_image,
        negative_prompt="bad quality, worse quality",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    frames = output.frames[0]
    export_to_gif(frames, "mulatto_woman.gif")
