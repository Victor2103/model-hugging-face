from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


class request_body(BaseModel):
    message: str


def predict(x, max_length):
    output = pipe(text_inputs=x, max_length=max_length)
    return output[0]['generated_text']


@app.post("/gpt2")
def generate_text(data: request_body, max_length: int = 50):
    # Get the input text
    input_text = data.message
    output = pipe(text_inputs=input_text, max_length=max_length)
    return {"input_text": predict(input_text, max_length)}


with gr.Blocks(title="Text Generation") as demo:
    gr.Markdown("# Generate some text with this demo ! ")
    text_input = gr.Textbox(label="Input Text")
    length_slider = gr.Slider(minimum=10, maximum=150)
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[
                       text_input, length_slider], outputs=output)
    gr.Markdown("## Examples")
    gr.Examples(examples=[["My name is James and i like", 75],
                          ["I go every day at the ", 110]],
                cache_examples=True,
                inputs=[text_input, length_slider],
                outputs=output,
                fn=predict)

app = gr.mount_gradio_app(app, demo, path='/')
