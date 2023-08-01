from diffusers import StableDiffusionPipeline
import torch
model_id = "wavymulder/Analog-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "A young lady singing with a man, analog style"
image = pipe(prompt).images[0]
image.save("./retro_cars.png")
