import transformers
import gradio as gr

model = transformers.pipeline(task="text-generation", model="gpt2")


def predict(x):
    output = model(x)
    return output[0]['generated_text']


with gr.Blocks() as demo:
    gr.Markdown("Generate some text with this demo ! ")
    input = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=input, outputs=output)


demo.launch(server_name="0.0.0.0")
