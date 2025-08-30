import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# NOTE: this is following the Neural Probabilistic Language Model paper
# a diagram of the model is found on page 6 of that paper

words = open("./names.txt", "r").read().splitlines()
random.seed(42)

# build the vocabulary of characters and to/from mappings
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # messy, but passing from global var
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)  # 27 rows, 2 columns, each of the 27 chars will have a 2D embedding
W1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)  # input from previous layer = 100, output/num units = 27 (for 27 chars)
b2 = torch.randn(27, generator=g)  # 1 bias for each unit/output
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
num_params = sum(p.nelement() for p in parameters)
# print(f"num parameters: {num_params}")  # 3481

# forward pass
# ============

# figure out the an appropriate learning rate
# create some learning rates to test
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre  # spaced exponentially in the range (0, 1)

# keep track of the learning rates that are used and their losses
lri = []
lossi = []

for i in range(30000):
    # construct minibatch
    # this is going to select 32 random rows from C[X]
    ix = torch.randint(0, Xtr.shape[0], (32,))  # does this need a generator, or is the idea to get a different batch each time (probably!)
    # only grab the minibatch rows
    emb = C[Xtr][ix]  # ([32, 3, 2])  # a tensor of 2D embeddings for each X index
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # -1 causes pytorch to infer the size from the 6
    logits = h @ W2 + b2  # ([32, 27])
    loss = F.cross_entropy(logits, Ytr[ix])  # only get the minibatch Y rows
    if (i % 1000 == 0):
        print(f"loss for iteration {i}: {loss}")

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    # lr = lrs[i]  # sample the learning rates
    # after tracking learning rates, we know that 0.1 is reasonable
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # commented out after settling on a learning rate
    # lri.append(lre[i])  # plotting the learning rate exponent instead of lr to make it easier to view the graph?
    # lossi.append(loss.item())
    # print(f"minibatch loss: {loss.item()}")  # probaby don't print them all out

# plot learning rate/loss
# plt.plot(lri, lossi)
# plt.show()
# it turns out that 10^-1 (0.1) is a good learning rate
# loss for entire training set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # -1 causes pytorch to infer the size from the 6
logits = h @ W2 + b2  # ([32, 27])
loss = F.cross_entropy(logits, Ydev)  # only get the minibatch Y rows
print(f"dev set loss: {loss.item()}")

# visualize the embeddings (works with 2D embeddings)
# plt.figure(figsize=(8, 8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
# for i in range (C.shape[0]):
#     plt.text(C[i, 0].item(), C[i,1].item, itos[i], ha="center", va="center", color="white")
# plt.grid("minor")
# plt.show()

# sample from the model


g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1)) @ W1 + b1
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))

  
