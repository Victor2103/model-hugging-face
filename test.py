from fastapi import FastAPI

app = FastAPI()

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)




@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/gpt2/{input_text}")
async def read_item(input_text: str):
    output = pipe(text_inputs=input_text, max_length=50)
    return {"input_text": output[0]['generated_text']}
