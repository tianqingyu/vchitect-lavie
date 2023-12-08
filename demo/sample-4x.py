import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import concurrent.futures

def upscale_image(img, idx, pipe):
    print(f'Start processing image {idx}')
    up_image = pipe(
        prompt='a photo of an astronaut riding a horse on mars',
        image=img,
        num_inference_steps=50,
        guidance_scale=5,
        noise_level=50
    ).images[0]
    print(f'Upscale done! img = {idx}')

    # output
    output_path = f'../results/demo/{idx}-4x.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    up_image.save(output_path)
    print(f"Image saved to {output_path}")

def main():
    print('start')

    # load model and scheduler
    pipe = StableDiffusionUpscalePipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_use_memory_efficient_attention_xformers(True)
    print('load model')

    # load image
    low_images = [Image.open(f'../results/demo/{i}.png').convert('RGB') for i in range(1, 4)]
    print('load image')

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        futures = [executor.submit(upscale_image, img, i, pipe) for i, img in enumerate(low_images)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
