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
    austin_id = tokenizer.encode("▁Austin", add_special_tokens=False)

    # hidden_states[26] is already after the final norm:
    # lm_head + cap on it gives exactly outputs["logits"] (checked 15 Oct)
    '''
    #Possibility 1: Snapshot 26 was saved before norm
    h26 = outputs["hidden_states"][26][0][8]
    normed = model.model.norm(h26)
    scores_one = model.lm_head(normed)
    scores_one_tanh = 30 * torch.tanh(scores_one / 30)

    #Possibility 2: Snapshot 26 was saved after norm
    scores_two = model.lm_head(h26)
    scores_two_tanh = 30 * torch.tanh(scores_two / 30)

    print(scores_one_tanh[22605])
    print(scores_two_tanh[22605])
    print(outputs["logits"][0][8][22605])
    '''