# following along with "makemore Part 4" video
# note that Karpathy is using `n_hidden=64` in the video,
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

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


# utility function for comparing manual gradients with PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()  # exact
    app = torch.allclose(dt, t.grad)  # approx
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}"
    )


n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)  # (27, 10)
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / (n_embd * block_size) ** 0.5
)  # (30, 64)
b1 = torch.randn(n_hidden, generator=g) * 0.1  # (64,)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1  # (64, 27)
b2 = torch.randn(vocab_size, generator=g) * 0.1  # (27,)
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0  # (1, 64)
bnbias = torch.randn((1, n_hidden)) * 0.1  # (1, 64)
# bnmean_running = torch.zeros((1, n_hidden))  # mean will start near 0 (due to initialization method)
# bnstd_running = torch.ones((1, n_hidden))  # std will start near 1 (due to initialization method)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
num_params = sum(p.nelement() for p in parameters)
print(f"num_params: {num_params}")
for p in parameters:
    p.requires_grad = True

batch_size = 32
n = batch_size
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
# Xb (32, 3); Yb (32,)

# forward pass
emb = C[Xb]  # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)  # (32, 30)
hprebn = embcat @ W1 + b1  # (32, 30) @ (30, 64) + (64,) --> (32, 64)

# BatchNorm layer
# sum columns (find average preact for each unit)
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = (
    hprebn - bnmeani
)  # subtract average (uses broadcasting) bndiff.shape([32, 200]) (batch_size, n_hidden)
bndiff2 = bndiff**2
bnvar = (
    1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
)  # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5) ** -0.5  # 1/standard_deviation
bnraw = bndiff * bnvar_inv  # normalized values (mean=0, variance=1)
hpreact = bngain * bnraw + bnbias  # hpreact.shape torch.Size([32, 64])
# Non-linearity
h = torch.tanh(hpreact)  # h.shape torch.Size([32, 64])
# Linear layer 2
logits = h @ W2 + b2  # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes  # subtract max for numerical stability
counts = norm_logits.exp()  # (32, 27)
counts_sum = counts.sum(1, keepdim=True)  # (32, 1)
counts_sum_inv = counts_sum**-1  # (32, 1)
probs = counts * counts_sum_inv  # (32, 27) * (32, 1) -> (32, 27)
logprobs = probs.log()  # (32, 27)
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [
    logprobs,
    probs,
    counts,
    counts_sum,
    counts_sum_inv,  # afaik there is no cleaner way
    norm_logits,
    logit_maxes,
    logits,
    h,
    hpreact,
    bnraw,
    bnvar_inv,
    bnvar,
    bndiff2,
    bndiff,
    hprebn,
    bnmeani,
    embcat,
    emb,
]:
    t.retain_grad()
loss.backward()
print(f"loss: {loss}")

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n
dprobs = probs**-1 * dlogprobs  # note that probs**-1 == 1.0 / probs; either will work
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts_sum = -(counts_sum**-2) * dcounts_sum_inv
dcounts = torch.ones_like(counts) * dcounts_sum
dcounts += counts_sum_inv * dprobs

dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= (
    1  # this is essentially dZ = A - Y; note that Y = 0 for non-label indices
)
dlogits /= n
dh = dlogits @ W2.T

# plt.figure(figsize=(8, 8))
# plt.imshow(dlogits.detach(), cmap="gray")
# plt.show()

cmp("logprobs", dlogprobs, logprobs)
cmp("probs", dprobs, probs)
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)
cmp("counts_sum", dcounts_sum, counts_sum)
cmp("counts", dcounts, counts)
cmp("logits", dlogits, logits)
cmp("h", dh, h)
