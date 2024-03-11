import sys
import torch
import unittest
from diffusers import DDPMScheduler

sys.path.append('./')

class TestFirstFramePertubation(unittest.TestCase):
    def setUp(self):
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.batch_size = 4
        self.frame_cnt = 16
        self.in_channels = 4
        self.width = 32
        self.height = 32

    def test_first_frame_pertubation(self):
        timesteps = torch.randint(low=1, high=1000, size=(self.batch_size, ))
        hidden_states = torch.randn((
            self.batch_size, self.frame_cnt, self.in_channels, self.height, self.width
        ))
        first_frames = hidden_states[:, 0]
        # perturb each video by noise step t except for the first frame
        noises_sampled = torch.randn(hidden_states.shape)
        noises_sampled[:, 0] = 0

        noised_hidden_states = self.noise_scheduler.add_noise(hidden_states, noises_sampled, timesteps)
        noised_first_frames = noised_hidden_states[:, 0]

        sqrt_alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(first_frames.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        print((first_frames * sqrt_alpha_prod)[:, 0, 0])
        print(noised_first_frames[:, 0, 0])

        self.assertTrue(torch.all(torch.eq(noised_first_frames, first_frames * sqrt_alpha_prod)))

if  __name__ == "__main__":
    unittest.main()

