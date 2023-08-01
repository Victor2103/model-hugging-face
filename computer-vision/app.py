import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from fastapi import FastAPI
import io
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

models = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5","prompthero/openjourney","wavymulder/Analog-Diffusion"]
# device = "cuda"


class request_body(BaseModel):
    message: str
    model_id: str = models


def predict(message: str, model_id: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    if model_id == "prompthero/openjourney":
        message+=", mdjrny-v4 style"
    if model_id == "wavymulder/Analog-Diffusion":
        message+=", analog style"
    image = pipe(message).images[0]
    return (image)


@app.post("/stable-diffusion")
async def create_upload_file(data: request_body):
    input_text = data.message
    model_id=data.model_id
    image = predict(input_text,model_id=model_id)
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    return Response(content=byte_im, media_type="image/png")


with gr.Blocks(title="Image Generation !") as demo:
    gr.Markdown("Generate an image with a prompt !")
    with gr.Column():
        options = gr.Dropdown(
            choices=models, label='Select Image Generation Model', show_label=True)
    with gr.Row():
        text_input = gr.Textbox(label="Input Text")
        output_image = gr.Image(label="Output Image", type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[
                       text_input, options], outputs=output_image)
    gr.Markdown("## Examples")
    gr.Examples(examples=[['A man in the space', "CompVis/stable-diffusion-v1-4"],
                          ['A woman singing in a bar', "prompthero/openjourney"],
                          ['A man drinking a beer in a stadium', "wavymulder/Analog-Diffusion"],
                          ["A cat fighting with a dog", "runwayml/stable-diffusion-v1-5"]],
                cache_examples=True,
                inputs=[text_input, options],
                outputs=[output_image],
                fn=predict)

app = gr.mount_gradio_app(app, demo, path='/')
