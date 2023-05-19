import gradio as gr

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def predict(x):
    output = pipe(text_inputs=x, max_length=50)
    return output[0]['generated_text']


with gr.Blocks() as demo:
    gr.Markdown("# Generate some text with this demo ! ")
    input = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=input, outputs=output)
    gr.Markdown("## Examples")
    gr.Examples(examples=["My name is jean and i like", "I go every day at the "],
                cache_examples=True,
                inputs=input,
                outputs=output,
                fn=predict)


demo.launch(server_name="0.0.0.0")
