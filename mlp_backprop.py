# following along with "makemore Part 4" video
# note that Karpathy is using `n_hidden=64` in the video,
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

ex_X, ex_Y = build_dataset(words)
print(ex_X[:3])
print(ex_Y[:3])
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
    print(f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}")

n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the MLP hidden layer

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
# avoid saturating neurons:
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g) * (5/3)/(n_embd*block_size)**0.5  # kaiming initialization
# note the scaling is different than in previous code, for demo purposes
b1 = torch.randn(n_hidden, generator=g) * 0.1  # using b1 just for "fun" :(
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1  # using 0.1 instead of 0.01 (for demonstration purposes)
b2 = torch.randn(vocab_size, generator=g) * 0.1
# bngain and bnbias get updated after backprop just like other params
# I'm following the initialization used in the video
# without a generator the results are going to vary from the video (?)
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1
# keep a running count of bnmean and bnstd
# not part of gradient based optimization
bnmean_running = torch.zeros((1, n_hidden))  # mean will start near 0 (due to initialization method)
bnstd_running = torch.ones((1, n_hidden))  # std will start near 1 (due to initialization method)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
num_params = sum(p.nelement() for p in parameters)
print(f"num_params: {num_params}")
for p in parameters:
    p.requires_grad = True

max_steps = 1000000
batch_size = 32
lossi = []

# copied from video's jupyter notebook
batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
print(ix)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
# Xb.shape ([32, 3])  ([batch_size, context])
emb = C[Xb] # embed the characters into vectors
# emb.shape ([32, 3, 10])  ([batch_size, block_size, n_embd])  # the embed vector for each index of Xb
# Xb[1] tensor([18, 14, 21])
# look at the characters associated with Xb[1] (using index 1 for no good reason)
# for i in Xb[1]:
#     print(itos[i.item()])  # returns "r", "n", "u"
# look at the vector associated with the second row, first column of C (the character "r"):
# print(C[Xb][1][0])
# [ 1.2815, -0.6318, -1.2464,  0.6830, -0.3946,  0.0144,  0.5722,  0.8673,
#          0.6315, -1.2230]

embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# embcat.shape ([32, 30])  ([batch_size, block_size*n_embd])
# each row of embcat corresponds to a training example
# simplified demo:
# In [29]: three_d
# Out[29]:
# tensor([[[ 1,  2,  3,  4],
#          [ 5,  6,  7,  8],
#          [ 9, 10, 11, 12]],
#
#         [[13, 14, 15, 16],
#          [17, 18, 19, 20],
#          [21, 22, 23, 24]]])
#
# In [30]: three_d.view(three_d.shape[0], -1)
# Out[30]:
# tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#         [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
# the `emb` vector in words: for each batch example for each character in the example's context grab the associated embedding vector
# or "for each batch example return a 2D array of (context_character, context_character_embedding)"
# each row of embcat is made up of [emb_vector_example_context_1, emb_vector_example_context_2, ...]

# Linear layer 1
# W1.shape torch.Size([30, 200])
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
# sum columns (find average preact for each unit)
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani  # subtract average (uses broadcasting) bndiff.shape([32, 200]) (batch_size, n_hidden)
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5  # 1/standard_deviation
bnraw = bndiff * bnvar_inv  # normalized values (mean=0, variance=1)
hpreact = bngain * bnraw + bnbias  # hpreact.shape torch.Size([32, 200])
# Non-linearity
h = torch.tanh(hpreact) # h.shape torch.Size([32, 200])
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
print("probs[0]", probs[0])
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
print(f"loss: {loss}")

# I guess I'm doing this:
# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one

# -----------------
# YOUR CODE HERE :)
# -----------------

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0/probs) * dlogprobs
# this confused the hell out of me:
# the general pattern for broadcasting is apply the chain rule normally, then
# sum over the dimensions that were broadcast
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

# a change to counts will affect the loss through how counts how much counts affects counts_sum
# and how much counts_sum affects the loss (`dcounts_sum`)

# note that counts affects the loss through counts_sum and probs
# NOTE: this is the long way of doing it, instead, use:
# dlogits = F.softmax(logits, 1)
# dlogits[range(n), Yb] -= 1
# dlogits /= n
#
# a bunch of intermediate calculations can be skipped
dcounts = counts_sum_inv * dprobs
dcounts += torch.ones_like(counts) * dcounts_sum  # essentially replicating dcounts_sum in a (batch_size, 27) tensor
# the derivative of the natural exponential function is the function itself
dnorm_logits = norm_logits.exp() * dcounts  # this could/should be simpflied to `counts * dcounts` (since `counts = norm_logits.exp()`)
# dlogit_maxes = (torch.ones_like(logits) * -1 * dnorm_logits).sum(1, keepdim=True)
# can be reduced to
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
# see makemore_backprop_help.md for a breakdown of what's going on here
# or just map out what's going on with the simplfied case of
# c11 c12 c13 = a11 a12 a13 - b1
# c21 c22 c23 = a21 a22 a23 - b2
# c31 c32 c33 = a31 a32 a33 - b3
# ^ the derivative of any element of c with respect to any element of a is 1, therefore
# dlogits[i, j] = 1 * dnorm_logits[i, j]  # each element of logits only affects the corresponding element of norm_logits
dlogits = dnorm_logits.clone()  # `.clone` is used for safety here
# the dlogits calculation makes sense. check out the notes for details. the logic is that the local
# derivative of logit_maxes with respect to logits will be 1 for the index of the max value of each
# row and 0 for all other elements in the row
dlogits += (F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])) * dlogit_maxes
# dlogits.shape torch.Size([32, 27])
# what can be learned about dlogits? What does the derivative of logits tell us?
# think of dlogits as a force that's pulling up on the probability of the correct index, and pulling down on the probabilities of
# the incorrect indices:
print("dlogits[0] * n", dlogits[0]*n)  # scaling by n here to remove averaging and make results easier to visualize
plt.figure(figsize=(8, 8))
plt.imshow(dlogits.detach(), cmap="gray")
plt.show()
# dlogits[0] * n tensor([ 0.0125, -0.7906,  0.0194,  0.0317,  0.0612,  0.0159,  0.0099,  0.0137,
#          0.0286,  0.0261,  0.0615,  0.0085,  0.0220,  0.0062,  0.0219,  0.0195,
#          0.0259,  0.0062,  0.0022,  0.0192,  0.1266,  0.0452,  0.0801,  0.0140,
#          0.0049,  0.0552,  0.0528], grad_fn=<MulBackward0>)
#
# note that in the condensed version of the dlogits calculation we use `dlogits[range(n), Yb] -= 1`, so for the correct prediction
# dlogits = probability - 1 (or prob - Y (the true label)). The derivative that corresponds with the true label will always be negative,
# all other derivatives will be positive. When applied during backprop, this will pull up on params that produce the correct output and push
# down on params that produce the incorrect predictions. Intuition: think of the neural network as a massive pulley system.

# W2.shape torch.Size([200, 27])
dh = dlogits @ W2.T  # ([32, 200])
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)  # don't use keepdim=True here the result should have torch.size([27]) not torch.size([1, 27])
# g\prime(z) = 1 - a^2
dhpreact = (1 - h**2) * dh # approximate: True
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)  # sum(0) because it's summing across examples shape ([1, 200])
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
dbndiff2 = 1.0/(n-1) * torch.ones_like(bndiff2) * dbnvar
# dbndiff = (2.0 * bndiff) * dbndiff2
# dbndiff += bnvar_inv * dbnraw
# not overthinking this, the mean is calculated, the subtracted to find the diff, but
# I think my dbndiff calculation has more error than it should. Karpathy uses:
dbndiff = bnvar_inv * dbnraw
dbndiff += (2*bndiff) * dbndiff2
# doesn't seem to make a difference though
dbnmeani = (-dbndiff).sum(0) 
dhprebn = dbndiff.clone()  # bndiff = hprebn - bnmeani
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)  # distribute the mean

dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)

for k in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[k,j]
    dC[ix] += demb[k,j]

# cmp('logprobs', dlogprobs, logprobs)
# cmp('probs', dprobs, probs)
# cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
# cmp('counts_sum', dcounts_sum, counts_sum)
# cmp('counts', dcounts, counts)
# cmp('norm_logits', dnorm_logits, norm_logits)
# cmp('logit_maxes', dlogit_maxes, logit_maxes)
# cmp('logits', dlogits, logits)
# cmp('h', dh, h)
# cmp('W2', dW2, W2)
# cmp('b2', db2, b2)
# cmp('hpreact', dhpreact, hpreact)
# cmp('bngain', dbngain, bngain)
# cmp('bnbias', dbnbias, bnbias)
# cmp('bnraw', dbnraw, bnraw)
# cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
# cmp('bnvar', dbnvar, bnvar)
# cmp('bndiff2', dbndiff2, bndiff2)
# cmp('bndiff', dbndiff, bndiff)
# cmp('bnmeani', dbnmeani, bnmeani)
# cmp('hprebn', dhprebn, hprebn)
# cmp('embcat', dembcat, embcat)
# cmp('W1', dW1, W1)
# cmp('b1', db1, b1)
# cmp('emb', demb, emb)
# cmp('C', dC, C)
