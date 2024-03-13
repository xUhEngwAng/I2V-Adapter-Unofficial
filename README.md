## Introduction

This repository contains unofficial implementation of the paper [I2V-Adapter: A General Image-to-Video Adapter for Video Diffusion Models](https://i2v-adapter.github.io/). Due to the lack of computing resources we have (4 A100 80G GPU in total), we only trained the model with a small set of [WebVid-10M]() dataset with no self-collected high-quality video clips as reported in the original paper. We don't attempt to employ any data filtering strategies either. If someone has a robust model trained on a large amount of high-quality data and is willing to share it, feel free to make a pull request.

### Video Samples Compared with the Official Project Page

<table class="center">
    <tr>
        <td>Input Image</td>
        <td>Official samples from official project page</td>
        <td>Our Samples</td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/classical oriental woman.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/classical oriental woman.gif"></td>
        <td><img src="./assets/our_results/classical oriental woman.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/great wall sunrise.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/great wall sunrise.gif"></td>
        <td><img src="./assets/our_results/great wall sunrise.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/old man warms up himself by the bonfire.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/old man warms up himself by the bonfire.gif"></td>
        <td><img src="./assets/our_results/old man warms up himself by the bonfire.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/water pouring off a girl.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/water pouring off a girl.gif"></td>
        <td><img src="./assets/our_results/water pouring off a girl.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/astronaunt swimming under the sea.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/astronaunt swimming under the sea.gif"></td>
        <td><img src="./assets/our_results/astronaunt swimming under the sea.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/man on a boat sunset.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/man on a boat sunset.gif"></td>
        <td><img src="./assets/our_results/man on a boat sunset.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/puppy in front of a snow mountain.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/puppy in front of a snow mountain.gif"></td>
        <td><img src="./assets/our_results/puppy in front of a snow mountain.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/a man astounded by the explosion of the city.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/a man astounded by the explosion of the city.gif"></td>
        <td><img src="./assets/our_results/a man astounded by the explosion of the city.gif"></td>
    </tr>
    <tr>
        <td><img src="./assets/input_images/two cats sleeping in a swaddle.jpeg"></td>
        <td><img src="./assets/I2VAdapter-samples/two cats sleeping in a swaddle.gif"></td>
        <td><img src="./assets/our_results/two cats sleeping in a swaddle.gif"></td>
    </tr>
</table>

## Release Plans

- [x] Release training script.
- [x] Release inference script.
- [ ] Release unofficial pretrained weights.

## Repository Setup

### Prepare Environment

You can setup the repository by running the following commands

```
git clone https://github.com/xUhEngwAng/I2V-Adapter-Unofficial.git
cd I2V-Adapter-Unofficial

conda create -n I2VAdapter
pip install -r requirements.txt
```

### Download Base T2I Model

```
git lfs install
git clone https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE
```

### Download Pretrained AnimateDiff Motion Adapter

```
git clone https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2
```

### Download Pretrained IP-Adapter

```
git clone https://huggingface.co/h94/IP-Adapter
```

## Training

Before training, you should first download the corresponding video dataset. Take the frequently-used [WebVid-10M](https://www.robots.ox.ac.uk/~vgg/research/frozen-in-time/) dataset for example. You can download the video files and the `.csv` annotations and place them under the `./data` directory. If you use custom dataset, the dataset class under `src/data.py` should also be modified.

To train I2V-Adapter modules, run

```
accelerate launch ./src/train_image_to_video.py --task_name your_task_name --num_train_epochs 10 --checkpoint_epoch 2
```

You can also finetune the motion modules by passing in `--update_motion_modules`

```
accelerate launch ./src/train_image_to_video.py --task_name your_task_name --num_train_epochs 10 --checkpoint_epoch 2 --update_motion_modules
```

As mentioned in the original AnimateDiff and PIA paper, you can also first finetune the base T2I model by using the individual frames in the video dataset. This can be accomplished by running 

```
accelerate launch ./src/train_text_to_image.py 
```

## Inference

The condition images and text prompts are given via `./data/WebVid-25K/I2VAdapter-eval.csv`, you can freely alter this file and provide your own condition images and prompts. Then run the following commands:

```
python src/pipeline/pipeline_i2v_adapter.py --task_name I2VAdapter-25K-finetune --checkpoint_epoch 25
```

## Acknowledgements
This codebase is based on [diffusers](https://github.com/huggingface/diffusers) library and [AnimateDiff](https://github.com/guoyww/AnimateDiff). The implementation of first frame similarity prior is inspied by [PIA](https://github.com/open-mmlab/PIA). We thank all the contributors of these repositories. Additionally, we would like to thank the authors of I2VAdapter for their open research and foundational work, which inspired this unofficial implementation.
