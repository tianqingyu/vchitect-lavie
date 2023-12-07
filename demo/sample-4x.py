import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

print('start')

# load model and scheduler
pipe = StableDiffusionUpscalePipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print('load model')

# load image
input_path = "../results/demo/1.png"
low_image = Image.open(input_path).convert('RGB')
# low_image = low_image.resize((256, 256))
print('load image')

# gen
up_image = pipe(prompt="a photo of an astronaut riding a horse on mars",
                image=low_image,
                num_inference_steps=20,
                guidance_scale=5,
                noise_level=20).images[0]
print('upscale done!')

# output
output_path = "../results/demo/1-4x.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
up_image.save(output_path)
print(f"Image saved to {output_path}")
