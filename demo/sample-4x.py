import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import requests
from io import BytesIO

print('start')

# load model and scheduler
pipe = StableDiffusionPipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print('load model')

# load image
input_path = "../results/demo/1.png"
low_res_img = Image.open(input_path).convert("RGB")
print('load image')

# gen
prompt = "a photo of an astronaut riding a horse on mars"
up_image = pipe(prompt=prompt, image=low_res_img).images[0]
print('upscale done!')

# output
output_path = "../results/demo/1-4x.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
up_image.save(output_path)
print(f"Image saved to {output_path}")
