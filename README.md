<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="https://pepy.tech/project/diffusers">
        <img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg">
    </a>
    <a href="https://twitter.com/diffuserslib">
        <img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib">
    </a>
</p>

🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or training your own diffusion models, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on [usability over performance](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [simple over easy](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), and [customizability over abstractions](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

🤗 Diffusers offers three core components:

- State-of-the-art [diffusion pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) that can be run in inference with just a few lines of code.
- Interchangeable noise [schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) for different diffusion speeds and output quality.
- Pretrained [models](https://huggingface.co/docs/diffusers/api/models/overview) that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems.

## Installation

We recommend installing 🤗 Diffusers in a virtual environment from PyPI or Conda. For more details about installing [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/#installation), please refer to their official documentation.

### PyTorch

With `pip` (official package):

```bash
pip install --upgrade diffusers[torch]
```

With `conda` (maintained by the community):

```sh
conda install -c conda-forge diffusers
```

### Flax

With `pip` (official package):

```bash
pip install --upgrade diffusers[flax]
```

Install tools to compute FID score
```python
import torch
!pip install torch-fidelity
!pip install torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
_ = torch.manual_seed(123)
```

fid = FrechetInceptionDistance(feature=64)

## Quickstart

Use Teacher Student Distillation to train Consistency Model

```python
import sys, importlib
import torch
sys.path.append('/mydrive/satellite_images/diffusers/')
utils = importlib.import_module('utils')
from utils import parse_args, train_student

args = parse_args()
args.push_to_hub = False
args.gradient_checkpointing = False
args.tracker_project_name = "super_resolution_distillation"

# hyper parameter for distillation
args.loss_type = "l2"
args.num_ddim_timesteps = 40
timestep_scaling_factor = 20
train_student_model(args)
```

You can also dig into the models and schedulers toolbox to build your own diffusion system:

```python

%cd /mydrive/satellite_images/diffusers/

# import consistency pipeline
from pipeline_ldm_consistency_sr import LDMConsistencySRPipeline
pipeline = LDMConsistencySRPipeline()

#Need to set the path for consistency model trained using distillation.
#I used GoogleDrive to save the models
UNET_PATH = "/content/gdrive/MyDrive/satellite_images/diffusers/models/con_unet_model_080824.pt"
VQVAE_PATH = "/content/gdrive/MyDrive/satellite_images/diffusers/models/con_vqvae_model_080824.pt"

# Use CUDA/GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline.to(device)
pipeline.vqvae.load_state_dict(torch.load(VQVAE_PATH))
pipeline.unet.load_state_dict(torch.load(UNET_PATH))
pipeline.vqvae.to(device)
pipeline.unet.to(device)
```
Create Custom Dataloader for satellite images

```python
import torch, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms

# Set the path to satellite images (test set). I have stored my images on Google drive
IMAGE_PATH = "/content/gdrive/MyDrive/satellite_images/dataset/test/images/"

class ImageDataset(Dataset):
    def __init__(self, root_dir, num_images=10000):
        self.root_dir = root_dir
        #self.transform = transform
        self.images = []

        for count, image_path in enumerate(os.listdir(root_dir)):
            #image = Image.open(os.path.join(root_dir, image_path))
            self.images.append(image_path)
            if (count >= (num_images-1)):
               break

    # Image transforms for dataset
    def transform(self, image, resolution=512):

       # crop if size greater than 512 x 512
       def crop_normalize(image):
       # get crop coordinates and crop image
         (w,h) = image.size
         resolution = 512

         if (w < resolution or h < resolution):
           if (w<=h):
             image = TF.crop(image,0,0,w,w)
           else:
             image = TF.crop(image,0,0,h,h)

           image = TF.resize(image, (resolution, resolution))
         else:
           c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
           image = TF.crop(image, c_top, c_left, resolution, resolution)

         image = TF.to_tensor(image)
         image = TF.normalize(image, [0.5], [0.5])
         return image

       image = crop_normalize(image)
       lr_image = TF.resize(image, (resolution//4, resolution//4))
       return image, lr_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(os.path.join(self.root_dir, image_path))
        #image.filter(ImageFilter.GaussianBlur(radius=2))
        image, lr_image = self.transform(image)
        return ({"hr_images":image, "lr_images":lr_image})

# Create a dataloader
dataset = ImageDataset(root_dir=IMAGE_PATH, num_images=2000)
eval_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
```

Run Inference and compute FID, MSE

```python
# Inference with 4 steps
# DDPM would need 100 steps
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
# Create a resize transform
resize_transform = T.Resize((512, 512),interpolation=InterpolationMode.NEAREST)

fid_sr = FrechetInceptionDistance(feature=64)
fid_lr = FrechetInceptionDistance(feature=64)

num_inference_steps = 4
sr_mse_values = []
lr_mse_values = []

def update_FID(fid, real_image, gen_image):
  real_img = ((real_image/2.0 + 0.5)*255).to(torch.uint8)
  gen_img = ((gen_image/2.0 + 0.5)*255).to(torch.uint8)
  fid.update(real_img, real=True)
  fid.update(gen_img, real=False)
  return

def compute_MSE(org_image, decode_image):
    shape = org_image.size()

    # Calculate MSE loss
    loss = nn.MSELoss(reduction='none')
    loss_result = torch.sum(loss(org_image,decode_image))
    return(loss_result/(shape[0]*shape[1]*shape[2]*shape[3]))

for step, batch in enumerate(eval_dataloader):
    # 1. Load and process the image and text conditioning
    hr_images = batch["hr_images"]
    lr_images = batch["lr_images"]
    sr_images = []
    for image_index in range(lr_images.shape[0]):
       (sr_image, dummy) = pipeline(lr_images[image_index:image_index+1],
                        num_inference_steps=num_inference_steps,
                        return_intermediate_images = False)
       sr_images.append(sr_image[0])

    sr_images = torch.stack(sr_images).to('cpu')
    lr_images = resize_transform(lr_images)
    #hr_images = hr_images.permute(0, 2, 3, 1)
    lr_images = resize_transform(lr_images)
    update_FID(fid_sr, hr_images, sr_images)
    update_FID(fid_lr, hr_images, lr_images)

    sr_mse_values.append(compute_MSE(hr_images.permute(0, 2, 3, 1),
                                    sr_images.permute(0, 2, 3, 1)))
    lr_mse_values.append(compute_MSE(hr_images.permute(0, 2, 3, 1),
                                     lr_images.permute(0, 2, 3, 1)))
```

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to launch your diffusion journey today!

## How to navigate the documentation

| **Documentation**                                                   | **What can I learn?**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | A basic crash course for learning how to use the library's most important features like using models and schedulers to build your own diffusion system, and training your own diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading_overview)                                                             | Guides for how to load and configure all the components (pipelines, models, and schedulers) of the library, as well as how to use different schedulers.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/pipeline_overview)                                             | Guides for how to use pipelines for different inference tasks, batched generation, controlling generated outputs and randomness, and how to contribute a pipeline to the library.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/opt_overview)                                                        | Guides for how to optimize your diffusion model to run faster and consume less memory.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for how to train a diffusion model for different tasks with different training techniques.                                                                                               |
## Contribution

We ❤️  contributions from the open-source community!
If you want to contribute to this library, please check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
You can look out for [issues](https://github.com/huggingface/diffusers/issues) you'd like to tackle to contribute to the library.
- See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities to contribute
- See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute exciting new diffusion models / diffusion pipelines
- See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Also, say 👋 in our public Discord channel <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>. We discuss the hottest trends about diffusion models, help each other with contributions, personal projects or just hang out ☕.


## Popular Tasks & Pipelines

<table>
  <tr>
    <th>Task</th>
    <th>Pipeline</th>
    <th>🤗 Hub</th>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Unconditional Image Generation</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/ddpm"> DDPM </a></td>
    <td><a href="https://huggingface.co/google/ddpm-ema-church-256"> google/ddpm-ema-church-256 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img">Stable Diffusion Text-to-Image</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5"> runwayml/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/unclip">unCLIP</a></td>
      <td><a href="https://huggingface.co/kakaobrain/karlo-v1-alpha"> kakaobrain/karlo-v1-alpha </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if">DeepFloyd IF</a></td>
      <td><a href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"> DeepFloyd/IF-I-XL-v1.0 </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/kandinsky">Kandinsky</a></td>
      <td><a href="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"> kandinsky-community/kandinsky-2-2-decoder </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/controlnet">ControlNet</a></td>
      <td><a href="https://huggingface.co/lllyasviel/sd-controlnet-canny"> lllyasviel/sd-controlnet-canny </a></td>
  </tr>
  <tr>
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/pix2pix">InstructPix2Pix</a></td>
      <td><a href="https://huggingface.co/timbrooks/instruct-pix2pix"> timbrooks/instruct-pix2pix </a></td>
  </tr>
  <tr>
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img">Stable Diffusion Image-to-Image</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5"> runwayml/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-guided Image Inpainting</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint">Stable Diffusion Inpainting</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-inpainting"> runwayml/stable-diffusion-inpainting </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Image Variation</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation">Stable Diffusion Image Variation</a></td>
      <td><a href="https://huggingface.co/lambdalabs/sd-image-variations-diffusers"> lambdalabs/sd-image-variations-diffusers </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Super Resolution</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale">Stable Diffusion Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"> stabilityai/stable-diffusion-x4-upscaler </a></td>
  </tr>
  <tr>
    <td>Super Resolution</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale">Stable Diffusion Latent Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler"> stabilityai/sd-x2-latent-upscaler </a></td>
  </tr>
</table>

## Popular libraries using 🧨 Diffusers

- https://github.com/microsoft/TaskMatrix
- https://github.com/invoke-ai/InvokeAI
- https://github.com/apple/ml-stable-diffusion
- https://github.com/Sanster/lama-cleaner
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- https://github.com/ashawkey/stable-dreamfusion
- https://github.com/deep-floyd/IF
- https://github.com/bentoml/BentoML
- https://github.com/bmaltais/kohya_ss
- +11.000 other amazing GitHub repositories 💪

Thank you for using us ❤️.

## Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:

- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim)
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
