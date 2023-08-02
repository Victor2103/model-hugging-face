import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from fastapi import FastAPI
import io
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

models = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5",
          "prompthero/openjourney", "wavymulder/Analog-Diffusion"]
# device = "cuda"

description="""# <p style="text-align: center;"> Welcome on your first stable diffusion application ! </p>

We provide you 4 models from Hugging Face of stable diffusion. You can do inference with all this models. Each model has his own documentation and all of them use the stable diffusion library. 

Here are the 4 links to the documentation on Hugging Face :

* Openjourney is an open source Stable Diffusion fine tuned model on Midjourney images. [Link](https://huggingface.co/prompthero/openjourney)

* Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. [Link](https://huggingface.co/CompVis/stable-diffusion-v1-4)

* Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. It is just the updated version of the model above. [Link](https://huggingface.co/runwayml/stable-diffusion-v1-5)

* This is a dreambooth model trained on a diverse set of analog photographs. [Link](https://huggingface.co/wavymulder/Analog-Diffusion)

Here's how the application works. We give a description and the model returns an image corresponding to the description. 

![local img](file=text-to-image.png)"""

class request_body(BaseModel):
    message: str
    model_id: str = models


def predict(message: str, model_id: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    if model_id == "prompthero/openjourney":
        message += ", mdjrny-v4 style"
    if model_id == "wavymulder/Analog-Diffusion":
        message += ", analog style"
    image = pipe(message).images[0]
    return (image)


@app.post("/stable-diffusion")
async def create_upload_file(data: request_body):
    input_text = data.message
    model_id = data.model_id
    image = predict(input_text, model_id=model_id)
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    return Response(content=byte_im, media_type="image/png")


with gr.Blocks(title="Image Generation !",theme='nota-ai/theme') as demo:
    gr.Markdown(description)
    with gr.Column():
        options = gr.Dropdown(
            choices=models, label='Select Image Generation Model', show_label=True)
    with gr.Row():
        text_input = gr.Textbox(label="Prompt something !", show_label=True)
        output_image = gr.Image(
            label="Here is the image create with your prompt", show_label=True, type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[
                       text_input, options], outputs=output_image)
    gr.Markdown("## Examples")
    gr.Examples(examples=[['A man in the space', "CompVis/stable-diffusion-v1-4"],
                          ['A woman singing in a bar', "prompthero/openjourney"],
                          ['A man drinking a beer in a stadium',
                              "wavymulder/Analog-Diffusion"],
                          ["A cat fighting with a dog", "runwayml/stable-diffusion-v1-5"]],
                cache_examples=True,
                inputs=[text_input, options],
                outputs=[output_image],
                fn=predict)

app = gr.mount_gradio_app(app, demo, path='/')
