# following along with "makemore Part 3" video
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

vocab_size = len(itos)
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the MLP hidden layer

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
# avoid saturating neurons:
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g) * (5/3)/(n_embd*block_size)**0.5  # kaiming initialization
# the bias can be removed from the hidden layer due to batch normalization (it's handled by bnbias)
# b1 = torch.randn(n_hidden, generator=g) * 0.01  # avoids pushing preact too high
# avoid high initial loss
# the idea is that the loss for the first batch shouldn't be something like 27. assume each output has an equal chance of being
# correct, so expected initial loss is:
expected_initial_loss = -torch.tensor(1.0/vocab_size).log()
print(f"expected initial loss: {expected_initial_loss}")
# scaling here prevents the softmax from being confidently wrong after initialization:
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01  # trying to get close to uniform distribution, but can't use 0
b2 = torch.randn(vocab_size, generator=g) * 0  # trying to get closs to uniform distribution for logits
# bngain and bnbias get updated after backprop just like other params
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
# keep a running count of bnmean and bnstd
# not part of gradient based optimization
bnmean_running = torch.zeros((1, n_hidden))  # mean will start near 0 (due to initialization method)
bnstd_running = torch.ones((1, n_hidden))  # std will start near 1 (due to initialization method)

# b1 gets removed due to batch normalization
# parameters = [C, W1, b1, W2, b2, bngain, bnbias]
parameters = [C, W1, W2, b2, bngain, bnbias]
num_params = sum(p.nelement() for p in parameters)
print(f"num_params: {num_params}")
for p in parameters:
    p.requires_grad = True

max_steps = 20000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

    # forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
    hpreact = embcat @ W1  # + b1  # hidden layer pre-activation (b1 gets removed due to batch normalization)
    # normalize the batch to a unit Gaussian distribution:
    # subtract the mean and divide by the standard deviation
    # scale by bngain (note it's initialized to 1)
    # offset by bnbias (note it's initialized to 0)
    bnmeani = hpreact.mean(0, keepdim=True)  # mean for batch
    bnstdi = hpreact.std(0, keepdim=True)  # std for batch
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact)  # hidden layer
    logits = h @ W2 + b2  # output layer
    loss = F.cross_entropy(logits, Yb)  # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

#     break
# plots a histogram of hidden layer activations; allows for checking that activations aren't in the >0.99 range
# plt.figure(figsize=(20, 10))
# plt.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")

plt.plot(lossi)
plt.show()

# calculate bnmean and bnstd for entire training set to use in testing and inference:
# this approach makes the calculations after training.
# instead of using this approach, we're calculating bnmean_running and bnstd_running
# with torch.no_grad():
#     emb = C[Xtr]
#     embcat = emb.view(emb.shape[0], -1)
#     hpreact = embcat @ W1 + b1
#     bnmean = hpreact.mean(0, keepdim=True)
#     bnstd = hpreact.std(0, keepdim=True)

@torch.no_grad()  # this decorator disables gradient tracking (at a function level)
def split_loss(split):
    x,y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]  # N, block_size, n_emb
    embcat = emb.view(emb.shape[0], -1)  # concatenate into (N, block_size * n_embd)
    hpreact = embcat @ W1  # + b1 (b1 gets removed due to batch normalization)
    # use bnmean and bnstd
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    # hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
    h = torch.tanh(hpreact)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss("train")
split_loss("val")

# sample the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        embcat = emb.view(1, -1)
        hpreact = embcat @ W1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
