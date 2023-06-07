from transformers import AutoTokenizer
import transformers
import torch
import gradio as gr
from fastapi import FastAPI


app = FastAPI()

model = "tiiuae/falcon-7b-instruct"


def predict(prompt):
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        framework="pt",
        device_map="auto"
    )
    sequences = pipeline(
        prompt,
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = ""
    for seq in sequences:
        result += f"Result: {seq['generated_text']}"
    return (result)


with gr.Blocks(title="Chat GPT") as demo:
    gr.Markdown("# Speak with a chatbot here !")
    text_input = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[
                       text_input], outputs=output)
    gr.Markdown("## Examples")
    gr.Examples(examples=[["Imagine the journey of a simple man living in London who go to work. He doesn't have a car and childrens."],
                          ["I want to write a story to a little child of 10 years. Help me do this"]],
                cache_examples=True,
                inputs=[text_input],
                outputs=output,
                fn=predict)

app = gr.mount_gradio_app(app, demo, path='/')
