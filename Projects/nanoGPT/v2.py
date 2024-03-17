"""
This is v2 of bigram.py. 

We want to move beyond the straightforward approach where token embeddings directly produce
logits (raw preditions or scores for each vocabulary token). Instead, we want to introduce
an intermediate layer or "level of abstraction" between the token embeddings and the logits.

- Embedding Dimension (n_embed): The number of dimensions in the embedding space. This is the
    number of dimensions in the token embeddings and the position embeddings. This is also the
    number of dimensions in the intermediate layer between the token embeddings and the logits.
- Linear Layer: This is the intermediate layer between the token embeddings and the logits. It
    is a linear layer that takes the sum of the token embeddings and the position embeddings as
    input and produces logits as output. The number of input dimensions is n_embed and the number
    of output dimensions is vocab_size.
- Position Embeddings: We introduce position embeddings to the model. This is a lookup table that
    maps each position in the sequence to a vector of n_embed dimensions. This is added to the token
    embeddings before being passed to the linear layer. This is to give the model some notion of
    position in the sequence. This is important because the model needs to know where in the sequence
    it is to make predictions. This is especially important for long sequences where the token
    embeddings alone may not be enough to capture the position information.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------------------------------#
# ----------------------------------- Hyperparameters --------------------------------------#
# ------------------------------------------------------------------------------------------#
batch_size = 32                 # how many independent sequences will we process in parallel?
block_size = 8                  # what is the maximum context length for predictions?
max_iters = 3000                # how many training iterations
eval_interval = 300             # how often to evaluate the model
learning_rate = 1e-2            # learning rate
eval_iters = 200                # how many iterations to average the loss over
n_embed = 32                    # size of the embedding dimension (features + positional encoding)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Data Preprocessing -----------------------------------#
# ------------------------------------------------------------------------------------------#
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # here are all the unique characters that occur in this text
vocab_size = len(chars)         # size of the vocabulary

stoi = { ch:i for i,ch in enumerate(chars) }        # create a mapping from characters to integers
itos = { i:ch for i,ch in enumerate(chars) }        # create a mapping from integers to characters
encode = lambda s: [stoi[c] for c in s]             # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # decoder: take a list of integers, output a string


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Train and Test Splits --------------------------------#
# ------------------------------------------------------------------------------------------#
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Data Loading -----------------------------------------#
# ------------------------------------------------------------------------------------------#
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # ix is a tensor of random integers of size (batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Estimate Loss ----------------------------------------#
# ------------------------------------------------------------------------------------------#
@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Bigram Language Model --------------------------------#
# ------------------------------------------------------------------------------------------#
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # now, we introduce a position embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # we also introduce a linear layer to produce the logits from the 
        # sum of the token and position embeddings
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            logits, loss = self(idx)    # get the predictions
            logits = logits[:, -1, :]   # focus only on the last time step: becomes (B, C)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities: (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence: (B, T+1)
        return idx


# ------------------------------------------------------------------------------------------#
# ----------------------------------- Model Training ----------------------------------------#
# ------------------------------------------------------------------------------------------#

# create a model and move it to the GPU
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))