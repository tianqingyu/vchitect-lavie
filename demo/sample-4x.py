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
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))
print('load image')

# gen
prompt = "a photo of an astronaut riding a horse on mars"
up_image = pipe(prompt=prompt).images[0]
print('upscale done!')

# output
output_path = "../results/demo/1-4x.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
up_image.save(output_path)
print(f"Image saved to {output_path}")
