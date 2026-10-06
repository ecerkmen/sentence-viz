from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

text = "The capital of the state containing Dallas is"

print(model)

inputs = tokenizer(text, return_tensors="pt")

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))