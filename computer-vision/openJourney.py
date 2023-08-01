from diffusers import StableDiffusionPipeline
import torch
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "A young lady singing with a man, mdjrny-v4 style"
image = pipe(prompt).images[0]
image.save("./retro_cars.png")