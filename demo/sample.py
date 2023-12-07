from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

pipe = StableDiffusionPipeline.from_pretrained("../pretrained_models/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]

save_path = "../results/demo/1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
image.save(save_path)
print(f"Image saved to {save_path}")
