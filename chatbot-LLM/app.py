from transformers import AutoTokenizer
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

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
   "Oui Oui is a cartoon caracter. He has a taxi and can honk when he wants. He has a taxi race. Imagine the discussion with this caracter in the taxi. He is very special and don't like to discuss.",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    


