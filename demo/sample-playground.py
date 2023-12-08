import os
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "../pretrained_models/playground-v2-1024px-aesthetic",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16"
)
pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image  = pipe(prompt=prompt, guidance_scale=3.0).images[0]

# save image
save_path = "../results/demo/pg-1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
image.save(save_path)
print(f"Image saved to {save_path}")
