# continued from mlp_rev.py
# second pass through Karpathy's MLP video
# notes: andrej_karpathy_mlp_review.md
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# build the dataset
block_size = 3  # context length
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)  # the label is the current character
        context = context[1:] + [ix]  # creates a sliding context window

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape)  # torch.Size([228146, 3])

g = torch.Generator().manual_seed(2147483647)

C = torch.randn(
    (27, 2), generator=g
)  # look-up table C with a (currently) 2 dimensional embedding space
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)  # num biases = num units/outputs
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

num_params = sum(p.nelement() for p in parameters)
print("num params:", num_params)

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(
    -3, 0, 1000
)  # lre means learning rate exponent; we want 1000 steps between -3 and 0
lrs = 10**lre  # step linearly between the exponents of the learning rates (1e-3, 1)

lri = []  # track learning rate used for each iteration i
lossi = []  # track loss for each iteration i

for i in range(1000):
    # embedding layer
    # ###############
    ix = torch.randint(0, X.shape[0], (32,))  # batch_size = B = 32

    emb = C[X[ix]]  # the embeddings for the training examples
    # print(emb.shape)  # torch.Size([B, 3, 2]) (B,T,C) B = batch_size

    # hidden layer
    # ############

    # the number of inputs to the layer is T*C (block_size*n_embd)
    # the number of outputs is a hyperparameter; hardcoded to 100 for now

    # note: for the current implementation, emb has the shape (B, 3, 2)
    # the following matrix multiplication can't be done `emb @ W1 + b` (B, 3, 2) @ (6, 100)
    h = torch.tanh(
        emb.view(-1, 6) @ W1 + b1
    )  # calling the view method to reshape emb; tanh squashes results between (-1, 1)
    # print(h.shape)  # torch.Size([B, 100])

    # output layer (is this technically the output layer?)
    # ############
    # see notes cross-entropy loss section
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])  # index into Y with minibatch
    print(loss.item())

    # backward pass
    # #############
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad  # type: ignore  (ignore Pyright warning about grad having type "None")

    # track learning rate stats
    lri.append(
        lre[i]
    )  # track the exponent; the plot's easier to read than using lri.append(lr)
    lossi.append(loss.item())

# plot learning rate, loss
plt.plot(lri, lossi)
plt.show()

# loss for full training set
emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print("loss for all of X", loss.item())
