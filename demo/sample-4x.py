import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

print('start')

# load model and scheduler
pipe = StableDiffusionUpscalePipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.set_use_memory_efficient_attention_xformers(True)
print('load model')

# load image
low_image1 = Image.open('../results/demo/1.png').convert('RGB')
low_image2 = Image.open('../results/demo/2.png').convert('RGB')
low_image3 = Image.open('../results/demo/3.png').convert('RGB')
print('load image')

# gen
up_images = pipe(
  prompt=[
    'a photo of an astronaut riding a horse on mars',
    'a photo of an astronaut riding a horse on mars',
    'a photo of an astronaut riding a horse on mars',
  ],
  image=[low_image1, low_image2, low_image3],
  num_inference_steps=50,
  guidance_scale=5,
  noise_level=50).images
print('upscale done!')

# output
for i, img in up_images:
  output_path = f'../results/demo/{i}-4x.png'
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  img.save(output_path)
  print(f"Image saved to {output_path}")
