# following building makemore part 5: building a wavenet
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
block_size = 8

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

# ^ copied from mlp_activations_gradients_batchnorm.py
# check that things still work
# for x, y in zip(Xtr[:20], Ytr[:20]):
#     print("".join(itos[ix.item()] for ix in x), "-->", itos[y.item()])
# ........ --> y
# .......y --> u
# ......yu --> h
# .....yuh --> e
# ....yuhe --> n
# ...yuhen --> g
# ..yuheng --> .
# ........ --> d
# .......d --> i
# ......di --> o
# .....dio --> n
# ....dion --> d
# ...diond --> r
# ..diondr --> e


class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
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
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running "momentum update")
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)


    def __call__(self, x):
        if self.training:
            # forward pass
            # Note that the `dim` is used to set the dimentions that are _reduced over_
            # this is covered at ~43:00 (fixing batchnorm1d bug)
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True)  # batch mean
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    # only the backprop trained params are returned (not running_mean, running_var)
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))  # torch.shape([vocab_size, embedding_dim])
        print("Embedding weight shape", self.weight.shape)

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:

    def __init__(self,n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]


# new code ========================================================================================
torch.manual_seed(42);  # seed random for reproducability (not sure about the `;` character)

n_embd = 10  # dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the hidden layer of the MLP

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

# I _think_ calling model.layers[1] here is correct
with torch.no_grad():
    model.layers[-1].weight *= 0.1  # last layer less confident

parameters = model.parameters()
num_params = sum(p.nelement() for p in parameters)
# print("num parameters", num_params)
# num parameters 12097

for p in parameters:
    p.requires_grad = True

# scratch ========================================

ix = torch.randint(0, Xtr.shape[0], (4,))  # creates a batch of 4 examples
Xb, Yb = Xtr[ix], Ytr[ix]
logits = model(Xb)
# print("Xb.shape", Xb.shape)
# print("Xb", Xb)
# Xb.shape torch.Size([4, 8])
# Remember that 0 corresponds to the character ".", it's expected to get a lot of them
# Xb tensor([[ 0,  0,  0,  0,  0,  0, 12, 15],
#         [ 0,  0,  0,  0,  0,  0, 20,  1],
#         [ 0,  0,  0,  0,  0,  0,  0,  0],
#         [ 0,  0,  0,  0,  0,  0,  0, 11]])

for layer in model.layers:
    print(layer.__class__.__name__, ":", tuple((layer.out.shape)))

