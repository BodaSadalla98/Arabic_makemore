import torch
import torch.functional as F
import model



block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

m = model.AraPoet()
m.load_state_dict(torch.load('arapoet.pt'))
m.to(device)
m.eval()


with open('data/tiny_data.txt', 'r') as f:
    data = f.read()

ctoi = {k:i for i, k in enumerate(model.chars)}
itoc = {k:v for v,k in ctoi.items()}

torch.manual_seed(12345)
chars = sorted(list(set(data)))
encode  = lambda x:[ctoi[c] for c in x]
decode = lambda x: ''.join([itoc[c] for c in x])




context = torch.zeros((1, 1), dtype=torch.long, device=device)

with open('out.txt' ,'w') as f:

    for i in range(10):
        text = decode(m.generate(context, 500)[0].tolist())
        f.write(text)
        f.write('\n\n--------------------------------------------------------\n\n')