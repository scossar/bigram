import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# NOTE: this is following the Neural Probabilistic Language Model paper
# a diagram of the model is found on page 6 of that paper

words = open("./names.txt", "r").read().splitlines()

# build the vocabulary of characters and to/from mappings
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3  # context length (number of preceeding characters)
X, Y = [], []
for w in words:  # NOTE: in the printed examples I was using only the first 5 words
    context = [0] * block_size  # [0, 0, 0]
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

# with current block and example size: ([32, 3]) 32 examples, context of 3:
X = torch.tensor(X)  
# Size([32]) 32 examples:
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)  # 27 rows, 2 columns, each of the 27 chars will have a 2D embedding
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)  # input from previous layer = 100, output/num units = 27 (for 27 chars)
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

for i in range(10000):
    # construct minibatch
    ix = torch.randint(0, X.shape[0], (32,))  # does this need a generator, or is the idea to get a different batch each time (probably!)
    # only grab the minibatch rows
    emb = C[X][ix]  # ([32, 3, 2])  # a tensor of 2D embeddings for each X index
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # -1 causes pytorch to infer the size from the 6
    logits = h @ W2 + b2  # ([32, 27])
# since cross entropy is a common operation, there's a built in function for it
# the following can be accomplished with the F.cross_entropy method:
# counts = logits.exp()  # fake counts, all positive
# prob = counts / counts.sum(1, keepdims=True)
# loss = -prob[torch.arange(32), Y].log().mean()
# NOTE: always prefer the built in cross_entropy function in production
# `logits.exp()` can go out of range. e.g. exp(100) = inf
# if you ever implement this from scratch, Pytorch calculates the max value from the logits and subtracts it:
# logits = torch.tensor([-5, 2, 0, 100]) - 100  # works fine because of the normalization in the prob step
    loss = F.cross_entropy(logits, Y[ix])  # only get the minibatch Y rows
# tensor(17.7697)

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
print(f"minibatch loss: {loss.item()}")

# plot learning rate/loss
# plt.plot(lri, lossi)
# plt.show()
# it turns out that 10^-1 (0.1) is a good learning rate
# loss for entire training set
emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # -1 causes pytorch to infer the size from the 6
logits = h @ W2 + b2  # ([32, 27])
loss = F.cross_entropy(logits, Y)  # only get the minibatch Y rows
print(f"training set loss: {loss.item()}")
