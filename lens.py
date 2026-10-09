import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

text = "The capital of the state containing Dallas is"

inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
dallas_pos = tokens.index("▁Dallas")

with torch.no_grad():
    outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)