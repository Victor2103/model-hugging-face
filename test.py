from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

text = "Replace me by any text you'd like."

pipe=pipeline("text-generation",model=model,tokenizer=tokenizer)
generated_text=pipe(text_inputs=text,max_length=50)
print(generated_text[0]["generated_text"])
