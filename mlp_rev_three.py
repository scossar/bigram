# continued from mlp_rev_two.py
# second pass through Karpathy's MLP video
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

block_size = 3  # context length
n_embd = 10
n_hidden = 200


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
    (27, n_embd), generator=g
)  # look-up table C with a (currently) 2 dimensional embedding space
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)  # num biases = num units/outputs
W2 = torch.randn((n_hidden, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

num_params = sum(p.nelement() for p in parameters)
print("num params:", num_params)

for p in parameters:
    p.requires_grad = True

# track loss for each step
lossi = []
stepi = []
for i in range(50000):
    # embedding layer
    # ###############
    ix = torch.randint(0, Xtr.shape[0], (32,))  # batch_size = B = 32

    emb = C[Xtr[ix]]  # the embeddings for the training examples
    # print(emb.shape)  # torch.Size([batch_size, block_size, n_embd]) (B,T,C)

    # hidden layer
    # ############

    # the number of inputs to the layer is T*C (block_size*n_embd)
    # the number of outputs is a hyperparameter; hardcoded to 100 for now

    # the following matrix multiplication can't be done `emb @ W1 + b` (B,T,C) @ (B*T, n_hidden)
    h = torch.tanh(
        emb.view(emb.shape[0], -1) @ W1 + b1
    )  # calling the view method to reshape emb; tanh squashes results between (-1, 1)
    # print(h.shape)  # torch.Size([B, 100])

    # output layer (is this technically the output layer?)
    # ############
    # see notes cross-entropy loss section
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])  # index into Y with minibatch
    stepi.append(i)
    # track log10 of loss for better visualization; log squashes the hockey stick part of the loss
    lossi.append(loss.log10().item())

    # backward pass
    # #############
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    if i < 100000:
        lr = 0.1
    else:
        lr = 0.01
    for p in parameters:
        p.data += -lr * p.grad  # type: ignore  (ignore Pyright warning about grad having type "None")

# plt.plot(stepi, lossi)
# plt.show()

# loss for dev set
emb = C[Xdev]
h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(
    "dev set loss:", loss.item()
)  # try to beat Karpathy's result of 2.17 by tuning hyperparameters

# # while the embedding dimension is 2, we can easily visualize the embedding vectors
# # visualize the embeddings (works with 2D embeddings)
# plt.figure(figsize=(8, 8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(
#         C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white"
#     )
# plt.grid(True, "minor")
# plt.show()

# sample the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(
            emb.view(1, -1) @ W1 + b1
        )  # emb.view(1, -1) because it's just a single example
        logits = h @ W2 + b2
        probs = F.softmax(
            logits, dim=1
        )  # exponentiates the logits and makes them sum to 1.0
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
