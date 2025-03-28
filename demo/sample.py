import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("../pretrained_models/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

save_path = "../results/demo/1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
image.save(save_path)
print(f"Image saved to {save_path}")
