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
block_size = 4


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


# these implementations are largely following the torch.nn layer module conventions
# I've added a `g` argument for the generator that I don't think is in the torch modules
# Karpathy has added an `out` attribute that we're using for examining the layers
class Linear:
    def __init__(self, fan_in, fan_out, g, bias=False):  # changing bias to false
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers trained with a running momentum update
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)
# note that bias=True in Karpathy's video, but I think it can be removed with batch norm (is handled by beta)
layers = [
    Linear(n_embd * block_size, n_hidden, g),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, g),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, g),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, g),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, g),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size, g),
    BatchNorm1d(vocab_size),
]

with torch.no_grad():
    # make last layer less confident
    # layers[-1].weight *= 0.1
    layers[
        -1
    ].gamma *= 0.1  # because the last layer is now BatchNorm1d, the gamma needs to be scaled, not the weight
    # apply tanh gain to all other layers
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print("num params:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad()  # for debugging, remove later?
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad  # pyright: ignore type

    # track stats
    if i % 10000 == 0 or i == max_steps - 1:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

    # break  # for development


# loss on train, dev, test sets
@torch.no_grad()  # disables gradient tracking for efficiency
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        if layer.__class__.__name__ == "BatchNorm1d":
            layer.training = False
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(split, loss.item())


split_loss("train")
split_loss("val")

# sample the model
g = torch.Generator().manual_seed(
    2147483647 + 12
)  # note that + 10 is used in the video

for _ in range(50):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        x = emb.view(1, -1)
        for layer in layers:
            if layer.__class__.__name__ == "BatchNorm1d":
                layer.training = False
            x = layer(x)
        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))


# for _ in range(0):
#     out = []
#     context = [0] * block_size
#     while True:
#         emb = C[torch.tensor([context])]
#         embcat = emb.view(1, -1)
#         hpreact = embcat @ W1
#         hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
#         h = torch.tanh(hpreact)
#         logits = h @ W2 + b2
#         probs = F.softmax(logits, dim=1)
#         ix = torch.multinomial(probs, num_samples=1, generator=g).item()
#         context = context[1:] + [ix]
#         out.append(ix)
#         if ix == 0:
#             break
#     print("".join(itos[i] for i in out))

# visualize histograms
# activation distributions
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(layers[:-1]):  # exclude the output layer
#     if isinstance(layer, Tanh):
#         t = layer.out
#         print(
#             "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
#             % (
#                 i,
#                 layer.__class__.__name__,
#                 t.mean(),
#                 t.std(),
#                 (t.abs() > 0.97).float().mean() * 100,
#             )
#         )
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f"layer {i} ({layer.__class__.__name__})")
# plt.legend(legends)
# plt.title("activation distribution")
# plt.show()
#
# # gradient distributions
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(layers[:-1]):  # exclude the output layer
#     if isinstance(layer, Tanh):
#         t = layer.out.grad
#         print(
#             "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
#             % (
#                 i,
#                 layer.__class__.__name__,
#                 t.mean(),  # pyright: ignore type
#                 t.std(),  # pyright: ignore type
#                 (t.abs() > 0.97).float().mean() * 100,  # pyright: ignore type
#             )
#         )
#         hy, hx = torch.histogram(t, density=True)  # pyright: ignore type
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f"layer {i} ({layer.__class__.__name__})")
# plt.legend(legends)
# plt.title("gradient distribution")
# plt.show()
