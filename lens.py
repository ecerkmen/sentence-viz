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
    texas_id = tokenizer.convert_tokens_to_ids("▁Texas")

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
        is_list = hidden[i][0][8] #shape (2304) for _is' list
        dallas_list = hidden[i][0][dallas_pos] #shape(2304) for _Dallas' list
        is_normed = model.model.norm(is_list) #shape (2304) for _is' list
        dallas_normed = model.model.norm(dallas_list) #shape (2304) for _Dallas' list
        is_scores = model.lm_head(is_normed) #shape (256000) for _is
        dallas_scores = model.lm_head(dallas_normed) #shape (256000) for _Dallas
        is_scores_tanh = 30 * torch.tanh(is_scores / 30) #shape (256000) for _is
        dallas_scores_tanh = 30 * torch.tanh(dallas_scores / 30) #shape (256000) for _Dallas
        is_winner_id = torch.argmax(is_scores_tanh).item()
        dallas_winner_id = torch.argmax(dallas_scores_tanh).item()
        is_winner = tokenizer.convert_ids_to_tokens(is_winner_id)
        dallas_winner = tokenizer.convert_ids_to_tokens(dallas_winner_id)
        #Calculating _Austin's rank at position 8
        austin_rank = ((is_scores > is_scores[austin_id]).sum()) + 1
        #Calculating _Texas' rank at position 7
        texas_rank = ((dallas_scores > dallas_scores[texas_id]).sum()) + 1
        print(f"Position 8: {i}, {is_winner}, _Austin rank: {austin_rank}")
        print(f"Position 7: {i}, {dallas_winner}, _Texas rank: {texas_rank}")

    is_scores26 = model.lm_head(hidden[26][0][8]) #shape (256000) for _is
    dallas_scores26 = model.lm_head(hidden[26][0][dallas_pos]) #shape (256000) for _Dallas
    is_scores26_tanh = 30 * torch.tanh(is_scores26 / 30) #shape (256000) for _is
    dallas_scores26_tanh = 30 * torch.tanh(dallas_scores26 / 30) #shape (256000) for _Dallas
    is_scores26_winner_id = torch.argmax(is_scores26_tanh).item()
    dallas_scores26_winner_id = torch.argmax(dallas_scores26_tanh).item()
    is_scores26_winner = tokenizer.convert_ids_to_tokens(is_scores26_winner_id)
    dallas_scores26_winner = tokenizer.convert_ids_to_tokens(dallas_scores26_winner_id)
    #Calculating Austin's rank at snapshot 26, position 8
    austin_rank_h26 = ((is_scores26 > is_scores26[austin_id]).sum()) + 1
    #Calculating Texas' rank at snapshot 26, position 7
    texas_rank_h26 = ((dallas_scores26 > dallas_scores26[texas_id]).sum()) + 1
    print(f"Position 8: 26 {is_scores26_winner}, _Austin rank: {austin_rank_h26}")
    print(f"Position 7: 26 {dallas_scores26_winner}, _Texas rank: {texas_rank_h26}")

    #Compare embed_tokens' table to lm_head's:
    #The tables inside embed_tokens and lm_head are the same. By the time the copied rows reach snapshot 0, they're multiplied by 48
    #The tables themselves never change; only lists do
    '''
    print(torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)) #True
    dallas_mul = model.model.embed_tokens.weight[26865] * 48
    print(torch.equal(hidden[0][0][7], dallas_mul)) #True
    '''
    #Checking top 5 IDs through snapshots 0-25, position 7
    for i in range(26):
        dallas_listh025 = outputs["hidden_states"][i][0][dallas_pos] #shape (2304)
        dallas_listh025_normed = model.model.norm(dallas_listh025) #shape (2304)
        dallas_listh025_scores = model.lm_head(dallas_listh025_normed)
        dallash025_topk = torch.topk(dallas_listh025_scores, 5)
        dallas_top5_tokens = tokenizer.convert_ids_to_tokens(dallash025_topk.indices.tolist())
        print(f"Snapshot {i}, Tokens: {dallas_top5_tokens}, Scores: {dallash025_topk.values}")

    #Checking top 5 IDs at snapshot 26, position 7
    dallas_scores26_topk = torch.topk(dallas_scores26, 5)
    dallas26_top5_tokens = tokenizer.convert_ids_to_tokens(dallas_scores26_topk.indices.tolist())
    print(f"Snapshot 26, Tokens: {dallas26_top5_tokens}, Scores: {dallas_scores26_topk.values}")