import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import concurrent.futures

def upscale_image(img, idx, gpu_id):
    device = f'cuda:{gpu_id}'
    print(f'Start processing image {idx} on {device}')
    
    # 将模型移动到指定的 GPU
    # 加载模型
    pipe = StableDiffusionUpscalePipeline.from_pretrained("../pretrained_models/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    # pipe.set_use_memory_efficient_attention_xformers(True)
    pipe.enable_xformers_memory_efficient_attention()
    print('load model')

    # 处理图像
    with torch.cuda.device(gpu_id):
        up_image = pipe(
            prompt='a photo of an astronaut riding a horse on mars',
            image=img,
            num_inference_steps=50,
            guidance_scale=5,
            noise_level=50
        ).images[0]

    print(f'Upscale done! img = {idx}')

    # 输出路径
    output_path = f'../results/demo/{idx}-4x.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    up_image.save(output_path)
    print(f"Image saved to {output_path}")

def main():
    print('start')

    # 加载图像
    low_images = [Image.open(f'../results/demo/{i}.png').convert('RGB') for i in range(1, 4)]
    print('load image')

    # 使用 ThreadPoolExecutor 进行并行处理
    num_gpus = torch.cuda.device_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(upscale_image, img, idx, idx % num_gpus) for idx, img in enumerate(low_images)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
