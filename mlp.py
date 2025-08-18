import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# NOTE: this is following the Neural Probabilistic Language Model paper
# a diagram of the model is found on page 6 of that paper

words = open("./names.txt", "r").read().splitlines()
# print(len(words))
# 32022  # lots of words, so I'll avoid printing them all

# build the vocabulary of characters and to/from mappings
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset ~9:50

block_size = 3  # context length (number of preceeding characters)
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size  # [0, 0, 0]
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print("".join(itos[i] for i in context), "--->", itos[ix])
        context = context[1:] + [ix]

# the character representation of the dataset:
# emma
# ... ---> e
# ..e ---> m
# .em ---> m
# emm ---> a
# mma ---> .
# olivia
# ... ---> o
# ..o ---> l
# .ol ---> i
# oli ---> v
# liv ---> i
# ivi ---> a
# via ---> .
# ava
# ... ---> a
# ..a ---> v
# .av ---> a
# ava ---> .
# isabella
# ... ---> i
# ..i ---> s
# .is ---> a
# isa ---> b
# sab ---> e
# abe ---> l
# bel ---> l
# ell ---> a
# lla ---> .
# sophia
# ... ---> s
# ..s ---> o
# .so ---> p
# sop ---> h
# oph ---> i
# phi ---> a
# hia ---> .

X = torch.tensor(X)
Y = torch.tensor(Y)

# print(X)
# print(Y)
# the actual (numeric) data:
# tensor([[ 0,  0,  0],
#         [ 0,  0,  5],
#         [ 0,  5, 13],
#         [ 5, 13, 13],
#         [13, 13,  1],
#         [ 0,  0,  0],
#         [ 0,  0, 15],
#         [ 0, 15, 12],
#         [15, 12,  9],
#         [12,  9, 22],
#         [ 9, 22,  9],
#         [22,  9,  1],
#         [ 0,  0,  0],
#         [ 0,  0,  1],
#         [ 0,  1, 22],
#         [ 1, 22,  1],
#         [ 0,  0,  0],
#         [ 0,  0,  9],
#         [ 0,  9, 19],
#         [ 9, 19,  1],
#         [19,  1,  2],
#         [ 1,  2,  5],
#         [ 2,  5, 12],
#         [ 5, 12, 12],
#         [12, 12,  1],
#         [ 0,  0,  0],
#         [ 0,  0, 19],
#         [ 0, 19, 15],
#         [19, 15, 16],
#         [15, 16,  8],
#         [16,  8,  9],
#         [ 8,  9,  1]])
# tensor([ 5, 13, 13,  1,  0, 15, 12,  9, 22,  9,  1,  0,  1, 22,  1,  0,  9, 19,
#          1,  2,  5, 12, 12,  1,  0, 19, 15, 16,  8,  9,  1,  0])

# print(X.shape, X.dtype, Y.shape, Y.dtype)
# torch.Size([32, 3]) torch.int64 torch.Size([32]) torch.int64

# build the embedding lookup table C
# NOTE: the rows of matrix C are referred to as "embeddings"
#
# we have 27 possible characters and we're going to embed them in a lower dimensional space
# in the paper (A Neural Probabalistic Language Model) they have 17000 words and embed them in spaces as small
# as 30 dimensions
# we'll start by "cramming" our 27 possible characters into a 2D space

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)  # 27 rows, 2 columns, each of the 27 chars will have a 2D embedding

# remember this is a char (actually int) lookup table. e.g. "." corresponds to C[0], "e" corresponds to C[5]
# so the embedding for "e"/5 can be found with C[5]

# print(C[5])
# tensor([-0.4713,  0.7868])  identical to the one-hot approach below

# an alternate approach for embedding characters:
# ===============================================
# the 5 in torch.tensor(5) is used in the example below to represent the char at index 5 ("e")
# e_one_hot = F.one_hot(torch.tensor(5), num_classes=27).float()
# print(e_one_hot @ C)
# tensor([-0.4713,  0.7868])  identical to the indexing approach above
# having the two approaches produce identical results tells us we can think of the
# embedding of the integer part of the model either as a lookup table or as the first layer
# of the neural net - that layer has no non-linearity, its weight matrix is C
# ^ the process is to encode integers into a one-hot matrix, then feed them into a neural net
#
# for this model, we're just going to index because it's faster

# Pytorch allows for indexing with lists or tensors
# print(C[torch.tensor([5, 6, 7])])
# tensor([[-0.4713,  0.7868],
#         [-0.3284, -0.4330],
#         [ 1.3729,  2.9334]])
# so this works:
# print(C[X].shape, C[X])
# torch.Size([32, 3, 2])  # for every one of the (32,3) X integers, we've retrieved the 2D embedding vector:
# tensor([[[ 1.5674, -0.2373],  # C[0]
        #  [ 1.5674, -0.2373],  # C[0]
        #  [ 1.5674, -0.2373]], # C[0]
        #
        # [[ 1.5674, -0.2373],  # C[0]
        #  [ 1.5674, -0.2373],  # C[0]
        #  [-0.4713,  0.7868]], # C[5]
        #
        # [[ 1.5674, -0.2373],  # C[0]
        #  [-0.4713,  0.7868],  # C[5]
        #  [ 2.4448, -0.6701]], # C[13]  # print(stoi["m"]) # prints 13
#  ...

# more confirmation that it works:
# X[13, 2] is the integer 1, so these are equivalent:
# print(C[X][13,2])
# tensor([-0.0274, -1.1008])
# print(C[1])
# tensor([-0.0274, -1.1008])

# to embed all of the integers of X:
# ==================================
emb = C[X]  # torch.Size([32, 3, 2])

# construct the hidden layer
# ==========================

# the number of inputs to the layer is going to be 3x2 because we have 3 2D embeddings
# I'm understanding that to be block_size x embedding dimension for now (note: block_size is correct interpretation)
# so the first dimension of W1 will be 6 (num inputs) and the second dimension will be up to us, e.g. 100

W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)

# note: we can't do `emb @ W1 + b1` because you can't multiply Size([32,3,2]) by Size([6,100])
# torch provides multiple ways of doing this
# torch.cat `https://docs.pytorch.org/docs/stable/generated/torch.cat.html`
embc1 = torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], 1)  # contatenate across dimension 1
# print(embc1.shape)
# torch.Size([32, 6])
# print(embc1)  # examine the printed C[X] data above to see how the dimensions were squashed
# tensor([[ 1.5674, -0.2373,  1.5674, -0.2373,  1.5674, -0.2373],
#         [ 1.5674, -0.2373,  1.5674, -0.2373, -0.4713,  0.7868],
#         [ 1.5674, -0.2373, -0.4713,  0.7868,  2.4448, -0.6701],
#...

# the problem with the above approach is that it doesn't generalize (can't handle changed block size)
# instead, torch.unbind: `https://docs.pytorch.org/docs/stable/generated/torch.unbind.html`
# torch.unbind removes a tensor dimension and "returns a tuple of all slices along a given dimension,
# already without it" (?) (The "already without it part is confusing me")

emb_dim_1 = torch.unbind(emb,1)
# print(emb_dim_1)
# returns 3 tuples, 1 for each "row" in the 1 dimension
# the tuples retured are the equivalent of [emb[:,0,:], emb[:,1,:], emb[:,2,:]]
# so:
embc1 = torch.cat(torch.unbind(emb, 1), 1)
# print(embc1.shape)
# torch.Size([32, 6])

# but THE CAT APPROACH CREATES NEW MEMORY -- VERY INEFFICIENT -- there's still a better way:
a = torch.arange(18)
# print(a)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17])
# print(a.view(2, 9))
# tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
#         [ 9, 10, 11, 12, 13, 14, 15, 16, 17]])
# print(a.view(3, 3, 2))
# tensor([[[ 0,  1],
#          [ 2,  3],
#          [ 4,  5]],
#
#         [[ 6,  7],
#          [ 8,  9],
#          [10, 11]],
#
#         [[12, 13],
#          [14, 15],
#          [16, 17]]])

# the `view` approach works because pytorch tensors have a `storage` property
# `storage` is how the tensor is represented in memory
# print(a.storage())
#  0
#  1
#  2
#  3
#  4
#  5
#  6
#  7
#  8
#  9
#  10
#  11
#  12
#  13
#  14
#  15
#  16
#  17
# [torch.storage.TypedStorage(dtype=torch.int64, device=cpu) of size 18]
# when you call a.view(), nothing is being changed in memory (think of approaces to flat arrays in C)

# h = emb.view(32, 6) @ W1 + b1  # hardcoded first dim size
# h = emb.view(emb.shape[0], 6) @ W1 + b1  # shape[0] will work for any size of emb
# note that I've added the tanh activation function here:
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)  # -1 causes pytorch to infer the size from the 6
# print(h)
# print(h.shape)
# tensor([[-1.6952e+00,  8.5502e+00,  1.6284e+00,  ...,  2.2642e+00,
#          -1.9505e-01,  1.8469e+00],
#         [ 2.8741e-01,  4.3343e+00,  1.0142e+00,  ...,  2.8221e+00,
#           3.9128e+00,  3.4733e+00],
#         [-3.1026e+00,  9.9601e+00, -1.3306e+00,  ..., -5.7069e-01,
#          -5.9107e+00, -6.9120e-03],
#         ...,
#         [-4.3248e+00,  7.4938e+00, -1.6386e+00,  ..., -5.1557e+00,
#          -3.3276e+00, -3.2464e+00],
#         [-1.4951e+00,  5.6195e+00,  2.5079e+00,  ..., -1.0607e+00,
#          -5.2543e-01,  3.4893e+00],
#         [-1.4982e+00,  8.5941e+00,  1.8897e+00,  ...,  2.4983e+00,
#           6.9596e+00,  2.6822e+00]])
# torch.Size([32, 100])

# remember:
# - emb.view(32,6) * W1.shape([6, 100]) + b1.shape([10])
# - 32 is the current number of examples
# - 6 is the context (3) times the embeddings dimension (2)
# - 6 represents the hidden layer inputs
# - 100 represents the num hidden layer units
# - Shape(32, 6) @ Shape(6, 100) returns Shape(32,100)

# softmax (output(I think)) layer
# ===============================

W2 = torch.randn((100, 27), generator=g)  # input from previous layer = 100, output/num units = 27 (for 27 chars)
b2 = torch.randn(27, generator=g)  # 1 bias for each unit/output

logits = h @ W2 + b2
# print(logits.shape)
# torch.Size([32, 27])

counts = logits.exp()  # fake counts, all positive
prob = counts / counts.sum(1, keepdims=True)
# print(prob.shape)
# torch.Size([32, 27])
# every row of prob sums to 1, e.g.first row:
# print(prob[0].sum())
# tensor(1.)

# the actual row that comes next (the label)
# print(Y)
# tensor([ 5, 13, 13,  1,  0, 15, 12,  9, 22,  9,  1,  0,  1, 22,  1,  0,  9, 19,
#          1,  2,  5, 12, 12,  1,  0, 19, 15, 16,  8,  9,  1,  0])
# index into the rows of prob; for each row, pluck out the probability assigned to the correct character:
# print(prob[torch.arange(32), Y])  # hardcoding 32 (num_examples) for now
# tensor([1.5213e-14, 1.2830e-12, 1.9647e-08, 3.1758e-10, 5.6763e-12, 1.0823e-10,
#         1.8821e-14, 1.1087e-08, 1.6134e-09, 2.1917e-03, 5.3863e-08, 3.1970e-04,
#         2.0283e-10, 3.5709e-11, 6.2336e-07, 5.1704e-07, 1.4206e-01, 9.5657e-09,
#         2.0670e-09, 2.5181e-02, 7.6846e-05, 2.8706e-12, 1.6961e-09, 5.6464e-15,
#         4.4656e-03, 2.6851e-09, 3.5864e-05, 2.3389e-04, 1.6890e-09, 9.5614e-01,
#         9.7404e-10, 2.1230e-12])
#
# ^ remember that the idea value for all of the probabilites is 1. (after training)

# negative log likelihood
# =======================
loss = -prob[torch.arange(32), Y].log().mean()
# print(loss)
# tensor(17.7697)
