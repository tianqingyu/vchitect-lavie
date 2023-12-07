import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from einops import rearrange
import torchvision

print('start')

# load model and scheduler
pipe = StableDiffusionPipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print('load model')

# load image
input_path = "../results/demo/1.png"
vframes, aframes, info = torchvision.io.read_video(filename=input_path, pts_unit='sec', output_format='TCHW') # RGB
vframes = vframes / 255.
vframes = (vframes - 0.5) * 2 # T C H W [-1, 1]
t, _, h, w = vframes.shape
vframes = vframes.unsqueeze(dim=0) # 1 T C H W
vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()  # 1 C T H W
print('Input_shape:', vframes.shape)
print('load image')

# gen
prompt = "a photo of an astronaut riding a horse on mars"
generator = torch.Generator(device='cuda').manual_seed(10)
up_image = pipe(
  prompt,
  image=vframes,
  generator=generator,
  num_inference_steps=50,
  guidance_scale=5,
  noise_level=50
).images[0]
print('upscale done!')

# output
output_path = "../results/demo/1-4x.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
up_image.save(output_path)
print(f"Image saved to {output_path}")
