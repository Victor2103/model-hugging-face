from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# define the data format


class request_body(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "Welcome on the API for text generation"}


@app.post("/gpt2")
def generate_text(data: request_body, max_length: int = 50):
    # Get the input text
    input_text = data.message
    output = pipe(text_inputs=input_text, max_length=max_length)
    return {"input_text": output[0]['generated_text']}


# uvicorn test:app --reload
