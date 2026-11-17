# sentence-viz

**Following a sentence through Gemma-2-2B, layer by layer.**

Ask Gemma-2-2B to complete *"The capital of the state containing Dallas is"* and it answers *Austin*. Getting there takes two steps: Dallas → Texas, then Texas → Austin. This project uses the logit lens to see at which of the model's 26 layers each step becomes readable.

## Question

Geva et al. (2023) showed that early MLP sublayers enrich the subject's representation, so fewer of its attributes can be read at the embedding than at later layers. Does this carry over to Gemma-2-2B? For example: at the position of *Dallas*, how highly does *Texas* rank before any layer has run, and how does that change through the layers?

## Method

The model returns 27 snapshots of the hidden state at every position: one after the embedding layer, and one after each of the 26 layers. The script reads two positions: *Dallas* (position 7) and *is* (position 8, where the answer is produced).

For each snapshot it:

1. applies the model's final normalisation (snapshot 26 already has it),
2. multiplies by the unembedding matrix, giving a score for each of the 256,000 tokens in the vocabulary,
3. applies Gemma's logit soft-capping: 30 · tanh(x / 30).

It then reports the top token, the top 5, the rank of *Texas* at *Dallas*, the rank of *Austin* at *is*, and how far *Austin*'s score is from the top token's.

Checks recorded in the script's comments:

- Snapshot 26 is already normalised: unembedding plus soft-capping reproduces the model's own output logits exactly.
- The embedding layer and the unembedding layer share one table.
- Snapshot 0 is the token's embedding row multiplied by 48 (√2304, the hidden size).

## Setup

You need:

- Python 3.11
- Access to Gemma-2-2B on Hugging Face: accept the licence on the [model page](https://huggingface.co/google/gemma-2-2b), then log in:

```bash
hf auth login
```

Install the packages and run:

```bash
pip install -r requirements.txt
python lens.py
```

As it stands, the script prints the top 5 tokens at *is* for every snapshot. The other readouts are in the script as commented-out prints, kept from earlier checks.

## Status

- The logit lens runs on the Dallas prompt.
- Next: the attention and MLP sublayers, to see where each change happens.
- Planned: an interactive visualisation of the sentence moving through the model's layers.

## Credits

- **The idea** comes from Anthropic's video [*Claude codes a sentence traveling through a brain*](https://www.youtube.com/watch?v=PK_twqwWqp4). It showed a sentence moving through a human brain; this project shows one moving through a model.
- **The prompt** is adapted from Anthropic's circuit-tracing research (Lindsey et al., 2025), whose [open-source release](https://www.anthropic.com/research/open-source-circuit-tracing) studied it in Gemma-2-2B as *"Fact: The capital of the state containing Dallas is"*.
- **The logit lens** was introduced by nostalgebraist (2020).

This is an independent project, not affiliated with Anthropic or Google.

## References

- Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. *EMNLP 2023*.
- Lindsey, J., et al. (2025). [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). *Transformer Circuits Thread*.
- Anthropic (2025). [Open-sourcing circuit tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing).
- nostalgebraist (2020). interpreting GPT: the logit lens. *LessWrong*.

## Built by

Ece Erkmen and Claude (Anthropic)
