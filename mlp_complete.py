import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

# build the vocabulary of characters and to/from mappings
chars = sorted(list(set("".join(words))))  # making the assumption that all letters of the alphabet exist in `words`
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0  # the model has a dictionary length of 27
print(f"stoi: {stoi}")
itos = {i:s for s,i in stoi.items()}

block_size = 3  # context length (number of preceeding characters)
X, Y = [], []
for w in words[:3]:  # intentionally just using the first three words
    context = [0] * block_size  # [0, 0, 0]
    for ch in w + ".":
        ix = stoi[ch]
        print(f"ch: {ch}, index: {ix}")
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

num_examples = len(X)
dict_len = len(stoi)  # I'm guessing this is an appropriate term for number of characters
X = torch.tensor(X)  # torch.Size([num_examples, block_size]) 
Y = torch.tensor(Y)  # torch.Size([num_examples])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)  # torch.Size([dict_len, num_features]); each dict entry has a 2D embedding
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)  # input from previous layer = 100, output/num units = 27 (for 27 chars)
b2 = torch.randn(27, generator=g)  # 1 bias for each unit/output
parameters = [C, W1, b1, W2, b2]
num_params = sum(p.nelement() for p in parameters)

# get the PyTorch autograd engine to track operations
for p in parameters:
    p.requires_grad = True

for i in range(10):
    # C[X] becomes the input to the hidden layer
    emb = C[X]  # ([num_examples, block_size, num_features]), e.g. ([16, 3, 2])  # a tensor of 2D embeddings for each X index
    # use emb.view to collapse block_size, num_features into a single
    # block_size * num_features dimension:
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2  # ([num_examples, num_outputs/dict_len])
    # since cross entropy is a common operation, there's a built in function for it
    # the following can be accomplished with the F.cross_entropy method:
    # counts = logits.exp()  # fake counts, all positive
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Y)  # only get the minibatch Y rows
    # print(f"iteration: {i}, loss: {loss.item()}")

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

C_numpy = C.detach().numpy()
plt.imshow(C_numpy, cmap="viridis")
plt.colorbar(label="Tensor Value")
plt.show()
