# continued from mlp_rev_three.py
# this is a test, this is only a test
# Video: Building makemore Part 3: Activations & Gradients, BatchNorm
# notes: andrej_karpathy_mlp_review.md
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("./names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos)
block_size = 4  # context length
n_embd = 12
n_hidden = 160


# build the dataset
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

g = torch.Generator().manual_seed(2147483647)

C = torch.randn(
    (vocab_size, n_embd), generator=g
)  # look-up table C with a (currently) 2 dimensional embedding space
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)  # kaiming initialization
)
W2 = (
    torch.randn((n_hidden, n_hidden), generator=g)
    * (5 / 3)
    / (n_hidden**0.5)  # kaiming initialization
)
W3 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b3 = torch.randn(vocab_size, generator=g) * 0

# batch norm gain and shift
bngain = torch.ones((1, n_hidden))  # initially no gain
bnbias = torch.zeros((1, n_hidden))  # initially no bias/shift
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
bngain2 = torch.ones((1, n_hidden))  # initially no gain
bnbias2 = torch.zeros((1, n_hidden))  # initially no bias/shift
bnmean_running2 = torch.zeros((1, n_hidden))
bnstd_running2 = torch.ones((1, n_hidden))

parameters = [C, W1, W2, W3, b3, bngain, bnbias, bngain2, bnbias2]

num_params = sum(p.nelement() for p in parameters)
print("num params:", num_params)

for p in parameters:
    p.requires_grad = True

max_steps = 600000
batch_size = 32
# track loss for each step
lossi = []
stepi = []
for i in range(max_steps):
    # embedding layer
    # ###############
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))  # batch_size = B
    emb = C[Xtr[ix]]  # the embeddings for the training examples
    # print(emb.shape)  # torch.Size([batch_size, block_size, n_embd]) (B,T,C)
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors (B, T*C)
    hpreact = (
        embcat @ W1
    )  # + b1  # (B, T*C) @ (T*C, n_hidden) note that b1 can now be removed
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    # normalize the hidden state
    # every neuron will be unit Gaussian (for the batch examples)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    h = torch.tanh(hpreact)  # (B, n_hidden)

    hpreact2 = h @ W2
    bnmeani2 = hpreact2.mean(0, keepdim=True)
    bnstdi2 = hpreact2.std(0, keepdim=True)
    hpreact2 = bngain2 * (hpreact2 - bnmeani2) / bnstdi2 + bnbias2
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
        bnmean_running2 = 0.999 * bnmean_running2 + 0.001 * bnmeani2
        bnstd_running2 = 0.999 * bnstd_running2 + 0.001 * bnstdi2
    h2 = torch.tanh(hpreact2)

    logits = h2 @ W3 + b3
    loss = F.cross_entropy(logits, Ytr[ix])  # index into Y with minibatch

    # backward pass
    # #############
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    if i > 300000:
        lr = 0.005
    for p in parameters:
        p.data += -lr * p.grad  # type: ignore  (ignore Pyright warning about grad having type "None")

    # track stats
    if i % 10000 == 0 or i == max_steps - 1:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")

    # break  # for development

    stepi.append(i)
    # track log10 of loss for better visualization; log squashes the hockey stick part of the loss
    lossi.append(loss.log10().item())


# loss on train, dev, test sets
@torch.no_grad()  # disables gradient tracking for efficiency
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)

    hpreact2 = h @ W2
    hpreact2 = bngain2 * (hpreact2 - bnmean_running2) / bnstd_running2 + bnbias2
    h2 = torch.tanh(hpreact2)

    logits = h2 @ W3 + b3
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss("train")
split_loss("val")

# sample the model
g = torch.Generator().manual_seed(
    2147483647 + 18
)  # note that + 10 is used in the video

for _ in range(60):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        embcat = emb.view(1, -1)
        hpreact = embcat @ W1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)

        hpreact2 = h @ W2
        hpreact2 = bngain2 * (hpreact2 - bnmean_running2) / bnstd_running2 + bnbias2
        h2 = torch.tanh(hpreact2)

        logits = h2 @ W3 + b3
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
