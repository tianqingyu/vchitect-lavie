import os
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2-1024px-aesthetic",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16"
)
pipe.to("cuda")

prompt = "A group of friends exploring a winding path in an enchanted forest filled with the sweet scent of wildflowers and birdsong, with a magical surprise awaiting around the next bend as sunlight filters through the leaves."
image  = pipe(prompt=prompt, guidance_scale=3.0).images[0]

# save image
save_path = "../results/demo/pg-1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
image.save(save_path)
print(f"Image saved to {save_path}")
