
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('data/tiny_data.txt', 'r') as f:
    data = f.read()

torch.manual_seed(12345)
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(vocab_size)
print(''.join(chars))

ctoi = {k:i for i, k in enumerate(chars)}
itoc = {k:v for v,k in ctoi.items()}



encode  = lambda x:[ctoi[c] for c in x]
decode = lambda x: ''.join([itoc[c] for c in x])




data_enc = torch.tensor(encode(data), dtype=torch.long)



#################### HYPER PARAMS ################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
## Split data into: train, val
n = int(0.9 * len(data_enc))   
train_data = data_enc[:n]
val_data = data_enc[n:]
# Note that this training examples would be un-rolled into more examples
block_size = 256
batch_size  = 64
eval_iterations = 100
eval_interval = 500
total_steps = 100

num_heads = 6
head_size = 64
n_layers = 6
n_embd = num_heads * head_size
dropout_p = 0.2
lr = 5e-4

class MultiHeadAttention(nn.Module):
    ''' multi head of self attention in parallel'''
    def __init__(self, number_heads,head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(number_heads)])
        self.projection = nn.Linear(n_embd,n_embd)#, bias=False)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)

        out = self.projection(out)
        out = self.dropout(out)

        return out

class AttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd,head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.head_size = head_size
        ## assign the tril mask as buffer so that we have it, but won't train it as a parameter
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout_p)



    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5

        ## NOTE: we need to take just up to T in the mask, as when we generate, we won't have a context on size context_size
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = torch.softmax(wei,-1)

        wei = self.dropout(wei)

        out = wei @ v
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout_p)

        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' 
    Transformer block, it has two main components:
    communication part: which is the multihead attention
    computation part: which is the FFW module
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sah = MultiHeadAttention(num_heads,head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x + self.sah(self.ln1(x))
        x = x+ self.ffwd(self.ln2(x))
        return x 

class AraPoet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size,n_embd)
        self.position_embedding = nn.Embedding(block_size,n_embd)
        ## we need to add the star to acess what's inside, as squential takes a calls inhireted from Module
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)]
           
            
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        # ## add multi heads
        # self.self_attention_heads = MultiHeadAttention(num_heads, head_size)
        # # self.self_attention_heads = AttentionHead( 32)
        # self.ffw = FeedFoward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B,T = idx.shape
        loss = None
        token_embd = self.embed(idx) # (B,T,n_emb)

        ## adding posotional embedding
        pos_embd = self.position_embedding(torch.arange(T, device=device)) # (B,T,n_embd)

        x = token_embd + pos_embd  ## broadcasting 

        # ## applying self attention
        # x = self.self_attention_heads(x)
        # x = self.ffw(x)

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        # print(logits.shape)
        
        if not targets == None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx
    def generate(self, context, max_no):


        context = context.to(device)
        for _ in range(max_no):
            size = context.shape[1]
            # print(size)
            cur = context[:, -block_size:]  # B, T
            logits, _ = self(cur) # B*T , C 
            logits = logits[:,-1,:]
            props = F.softmax(logits, dim=-1)# B*T , C
            next_char = torch.multinomial(props, 1)  # B,1
            context = torch.cat((context, next_char), dim =1) # B, T+1
        return context