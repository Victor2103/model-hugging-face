from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = "tiiuae/falcon-7b-instruct"
rick_model = AutoModelForCausalLM.from_pretrained("ericzhou/DialoGPT-Medium-Rick_v2")

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline_falcon = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    framework="pt",
    device=torch.device('cuda:0')
)

rick_tokenizer = AutoTokenizer.from_pretrained("ericzhou/DialoGPT-Medium-Rick_v2")



class request_body(BaseModel):
    message: str


def predict_ricky(input, history=[]):
    # tokenize the new input sentence
    new_user_input_ids = rick_tokenizer.encode(input + rick_tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # generate a response 
    history = rick_model.generate(bot_input_ids, max_length=4000, pad_token_id=rick_tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = rick_tokenizer.decode(history[0]).split("<|endoftext|>")
    #print('decoded_response-->>'+str(response))
    response = [(response[i], response[i+1]) for i in range(0, len(response)-1, 2)]  # convert to tuples of list
    print('response-->>'+str(response))
    
    return response[0][1]


def predict(prompt):
    sequences = pipeline_falcon(
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


@app.post('/falcon')
def generate_text(data: request_body):
    input_text = data.message
    return ({"output_text": f"{predict(input_text)}"})


models=['falcon','rick and morty']

with gr.Blocks(title="Chat GPT") as demo:
    gr.Markdown("# Speak with a chatbot here !")
    chatbot = gr.Chatbot()
    options = gr.Dropdown(
            choices=models, label='Select Chatbot Model', show_label=True)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history,options):
        if options == 'rick and morty':
            bot_message = predict_ricky(input=message,history=[])
        else:
            bot_message = predict(prompt=message)
        chat_history.append((message,bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot,options], [msg, chatbot])
    gr.Markdown("## Examples")
    gr.Examples(examples=[["Tell me from 1998 to 2002 all the NBA championship teams and their starting five in the finals.",[("","")],'rick and morty'],
                          ["Tell me from 1998 to 2002 all the NBA championship teams and their starting five in the finals.",[("","")],'rick and morty']],        
                cache_examples=True,
                inputs=[msg,chatbot,options],
                outputs=[msg,chatbot],
                fn=respond)

app = gr.mount_gradio_app(app, demo, path='/')
