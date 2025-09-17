# continuing from `mmrev_bigram_stat_two.py`; casting the bigram model into the neural network framework
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

numchars = len("".join(words))
print("numchars", numchars)
# create the training set of bigrams (x,y)
xs, ys = [], []  # the inputs and the labels

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}  # works because `chars` is a list
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)

# 27 neurons; note the requires_grad argument
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(1000):
    # forward pass
    # encode integers into vectors, that can then be fed into a neural net:
    xenc = F.one_hot(
        xs, num_classes=27
    ).float()  # torch.Size([num_examples, num_classes]); cast to float!

    # the model has a single linear layer, with no bias and no activation functions

    logits = xenc @ W  # log-counts
    # note: the next two lines are a "softmax" function
    counts = logits.exp()  # convert the log counts to positive values; equivalent to the N array in previous model
    probs = counts / counts.sum(1, keepdim=True)  # probabilities for next character
    # for the loss function, we need to get the probabilities assigned to the true labels
    loss = (
        -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    )  # regularization component (see notes)
    print(
        "loss in forward pass:", loss.item()
    )  # loss in the statistical model was approx 2.45

    # backward pass
    W.grad = None  # an efficient way of setting the gradients to zero
    loss.backward()  # fills in the gradients of all the intermediate values back to the W tensor
    # note: for such a simple model a high learning rate can be achieved
    W.data += -5 * W.grad  # type: ignore  (suppress Pyright warning about type "None")

# sampling from the neural net
# ############################

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = (
            xenc @ W
        )  # essentially just plucks out the row of W corresponding to ix
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[int(ix)])
        if ix == 0:
            break
    print("".join(out))

# # print stats
# nlls = torch.zeros(5)
# for i in range(5):
#     # i-th bigram:
#     x = xs[i].item()  # input character index
#     y = ys[i].item()  # label character index
#     print("------------")
#     print(f"bigram example {i + 1}: {itos[int(x)]}{itos[int(y)]} (indexes {x},{y})")
#     print("input to neural net:", x)
#     print("output probabilities from neural net:", probs[i])
#     print("label (actual next character):", y)
#     p = probs[i, int(y)]
#     print("probability assigned by neural net to the correct character:", p.item())
#     logp = torch.log(p)
#     print("log likelihood:", logp.item())
#     nll = -logp
#     print("negative log likelihood:", nll.item())
#     nlls[i] = nll
#
# print("===============")
# print("average negative log likelihood, i.e. loss =", nlls.mean().item())
