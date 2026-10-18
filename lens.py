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
    austin_id = tokenizer.convert_tokens_to_ids("▁Austin")

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
    hidden = outputs["hidden_states"] #hidden is a tuple, len 27
    for i in range(26):
        is_list = hidden[i][0][8] #shape (2304)
        normed = model.model.norm(is_list) #shape (2304)
        scores = model.lm_head(normed) #shape (256000)
        scores_tanh = 30 * torch.tanh(scores / 30) #shape (256000)
        winner_id = torch.argmax(scores_tanh).item()
        winner = tokenizer.convert_ids_to_tokens(winner_id)
        #Calculating _Austin's rank
        austin_rank = ((scores > scores[austin_id]).sum()) + 1
        print(f"{i}, {winner}, _Austin rank: {austin_rank}")

    scores26 = model.lm_head(hidden[26][0][8]) #shape (256000)
    scores26_tanh = 30 * torch.tanh(scores26 / 30) #shape (256000)
    scores26_winner_id = torch.argmax(scores26_tanh).item()
    scores26_winner = tokenizer.convert_ids_to_tokens(scores26_winner_id)
    #Calculating Austin's rank at snapshot 26
    austin_rank_h26 = ((scores26 > scores26[austin_id]).sum()) + 1
    print(f"26 {scores26_winner}, _Austin rank: {austin_rank_h26}")

    #Compare embed_tokens' table to lm_head's:
    #The tables inside embed_tokens and lm_head are the same. By the time the copied rows reach snapshot 0, they're multiplied by 48
    #The tables themselves never change; only lists do
    '''
    print(torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)) #True
    dallas_mul = model.model.embed_tokens.weight[26865] * 48
    print(torch.equal(hidden[0][0][7], dallas_mul)) #True
    '''