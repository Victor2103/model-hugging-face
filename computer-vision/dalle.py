import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from fastapi import FastAPI

app = FastAPI()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


def predict(message,model):
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to(device) 
    image = pipe(message).images[0]
    return(image)


with gr.Blocks(title="Image Generation !") as demo:
    gr.Markdown("Generate an image with a prompt !")
    #with gr.Column():
    #    options = gr.Dropdown(
    #        choices=models, label='Select Object Detection Model', show_label=True)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        output_image = gr.Image(label="Output Image", type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[prompt, model_id], outputs= output_image)
    gr.Markdown("## Examples")
    """gr.Examples(examples=[["examples/example_1.jpg", 'hustvl/yolos-tiny'],
                          ["examples/example_2.jpg", 'hustvl/yolos-small'],
                          ["examples/example_3.jpg", 'facebook/detr-resnet-50']],
                cache_examples=True,
                inputs=[image, options],
                outputs=[output_text, output_image],
                fn=predict)"""

app = gr.mount_gradio_app(app, demo, path='/')