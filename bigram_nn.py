import torch
import torch.nn.functional as F  # for one hot encoding

import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
# create a mapping of characters to ints
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0  # start/end char at position 0
itos = {i:s for s,i in stoi.items()}

# create the training set of bigrams
# the training set is made up of two lists: inputs and targets
xs, ys = [], []

for w in words[:1]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# convert the lists to tensors
xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs)  # input
print(ys)  # label

# one hot encoding
# xy and ys are currently integers:
# print(xs.dtype)
# torch.int64
# it doesn't make sense to plug integers into a neural network - they're going to be scaled by weights (floats), etc
# a common way of encoding integers for a neural network is to use one hot encodeing
# for example, consider:
# tensor([0, 5, 13, 13, 1])
# for the 13 that's found at index 2, one hot encoding will generate a vector of length set by `num_classes` param
# that's all zeros, except the 13th element gets set to 1

xenc = F.one_hot(xs, num_classes=27).float()  # note the casting to floats here
# before encoding: tensor([ 5, 13, 13,  1,  0])
# print(xenc)
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# print(xenc.shape)
# torch.Size([5, 27])
#
# visualize the data
# plt.imshow(xenc)
# plt.show()

# create a single neuron
W = torch.randn((27, 1))
# print(W)
# tensor([[ 0.0889],
#         [ 0.8179],
#         [-1.0868],
#         [ 0.5660],
#         [ 1.6392],
#         [-0.5717],
#         [-0.7537],
#         [ 1.0059],
#         [-1.0089],
#         [-1.1053],
#         [ 0.0771],
#         [-0.0874],
#         [ 0.2186],
#         [-0.7776],
#         [ 0.9686],
#         [ 2.3502],
#         [-2.5148],
#         [ 0.2148],
#         [-0.8584],
#         [-0.1234],
#         [ 1.4667],
#         [-0.3330],
#         [-0.3720],
#         [ 0.3765],
#         [ 1.1037],
#         [ 2.7204],
#         [-0.5050]])

# multiply xenc by weights (remember xs is just for a single word at this point)
# multiplying a (5, 27) vector by a (27, 1) vector
xencxw = xenc @ W
# print(xencxw)
# tensor([[ 0.1512],
#         [-1.6734],
#         [ 0.7368],
#         [ 0.7368],
#         [ 0.1907]])

# make W a (27, 27) tensor (create 27 neurons?) will confirm below
W = torch.randn((27, 27))
# xencxw = xenc @ W

# I'm not following what's going on here trying to work it out below
# print(f"xenc: shape({xenc.shape})\n{xenc}")
# xenc: shape(torch.Size([5, 27]))
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.]])
#
# ^ there are (currently) 5 training examples, each example has 27 features
# creating the weights with `torch.randn(27, 1)` is the equivalent of creating a layer with a single neuron?
# creating the weights with `torch.randn(27, 27)` is the equivalent of creating a layer with 27 neurons?
# it would follow that a (27, 4) weight tensor would be a layer of 4 neurons
#
# I can confirm, we're now considering the first layer of a neural net that has 27 output neurons
# we're creating a neural network that has a single layer with 27 units, no bias and no non-linear activation function

# here's an example of the current output
# print(f"xencxw: shape({xencxw.shape})\n{xencxw}")
# xencxw: shape(torch.Size([5, 27]))
# tensor([[ 0.9157, -1.1949, -0.6579, -1.6561, -0.5483,  0.6740,  0.3865,  0.2823,
#          -0.7082,  1.3589,  1.7582, -1.3670,  0.4947,  1.3773,  0.4708,  2.5496,
#           0.0878,  1.0902,  0.9118,  0.4733,  1.4545,  1.1569, -0.9024,  0.5288,
#          -0.0152,  0.3443, -1.2367],
#         [ 1.6370, -1.3359, -0.8960, -1.1821,  0.6415, -1.9526, -0.7332, -0.0771,
#          -1.1686,  0.7783,  0.3565,  0.2411, -0.1121,  2.4776, -0.7895, -1.7970,
#           0.6104,  1.1059, -0.8752,  1.1398,  1.3923, -0.7590, -0.5133,  0.8804,
#          -0.1991,  0.7950, -0.2837],
#         [ 0.3346,  0.3882,  1.3994, -0.0570,  1.2033,  0.3246,  0.4926,  0.1865,
#           0.4397,  1.3159, -1.0355, -0.7940, -1.2553,  0.0586, -1.7021,  0.2652,
#          -0.0050,  0.3918, -0.0936,  1.0515,  0.5090, -1.0251,  1.3118, -0.3469,
#           0.5384, -1.3608, -0.4594],
#         [ 0.3346,  0.3882,  1.3994, -0.0570,  1.2033,  0.3246,  0.4926,  0.1865,
#           0.4397,  1.3159, -1.0355, -0.7940, -1.2553,  0.0586, -1.7021,  0.2652,
#          -0.0050,  0.3918, -0.0936,  1.0515,  0.5090, -1.0251,  1.3118, -0.3469,
#           0.5384, -1.3608, -0.4594],
#         [ 1.1948, -0.3876, -0.2243, -0.6187,  0.4262,  0.2262,  0.2783, -0.1902,
#           0.0998, -0.6278,  0.1034,  1.3687,  0.0866, -1.3669,  0.3165,  0.4097,
#          -1.7592, -0.4446,  0.3717, -0.0384, -0.5613, -1.0983, -0.5922,  0.3302,
#           0.2290,  0.7210, -1.0948]])
#
# we want to convert these numbers to probabilities. a property of probabilities is that they
# are positive numbers that sum to 1
#
# t: 1:22:13

# we want `xenc @ W` to represent the probabilities of the next character
# to do this we'll consider the 27 numbers of each row to represent log counts (am I
# overcomplicating this by reading log counts as logarithmic counts?)
# to get the counts from the log counts, we're going to take the log counts and exponentiate them

logits = xenc @ W  # logits (log-counts)
counts = logits.exp()  # 0 to infinity
probs = counts / counts.sum(1, keepdim=True)
print(f"probs:\n{probs}")
summed_probs = probs.sum(1, keepdim=True)
print(f"summed_probs:\n{summed_probs}")
# tensor([[0.0191, 0.0148, 0.0675, 0.0728, 0.3263, 0.0359, 0.0032, 0.0449, 0.0033,
#          0.0083, 0.0393, 0.0105, 0.0162, 0.0560, 0.0136, 0.0083, 0.0083, 0.0107,
#          0.0111, 0.0098, 0.0158, 0.0100, 0.1162, 0.0245, 0.0079, 0.0107, 0.0350],
#         [0.0226, 0.0096, 0.0103, 0.0046, 0.0121, 0.0329, 0.1715, 0.0062, 0.1804,
#          0.0731, 0.0131, 0.0388, 0.0166, 0.0082, 0.0212, 0.0222, 0.0500, 0.0453,
#          0.0548, 0.0068, 0.0258, 0.0223, 0.0104, 0.0893, 0.0141, 0.0301, 0.0078],
#         [0.1139, 0.0235, 0.0111, 0.0284, 0.0284, 0.0055, 0.0461, 0.1858, 0.0221,
#          0.0349, 0.0036, 0.0129, 0.0384, 0.0028, 0.0625, 0.0168, 0.0579, 0.0321,
#          0.0283, 0.0043, 0.0105, 0.0438, 0.1536, 0.0044, 0.0122, 0.0099, 0.0061],
#         [0.1139, 0.0235, 0.0111, 0.0284, 0.0284, 0.0055, 0.0461, 0.1858, 0.0221,
#          0.0349, 0.0036, 0.0129, 0.0384, 0.0028, 0.0625, 0.0168, 0.0579, 0.0321,
#          0.0283, 0.0043, 0.0105, 0.0438, 0.1536, 0.0044, 0.0122, 0.0099, 0.0061],
#         [0.0341, 0.0160, 0.0490, 0.0511, 0.0199, 0.0231, 0.0110, 0.0407, 0.0968,
#          0.0211, 0.0175, 0.0106, 0.0911, 0.0137, 0.0031, 0.0136, 0.0047, 0.0516,
#          0.1862, 0.0201, 0.0119, 0.0257, 0.0745, 0.0183, 0.0237, 0.0641, 0.0067]])
# summed_probs:
# tensor([[1.0000],
#         [1.0000],
#         [1.0000],
#         [1.0000],
#         [1.0000]])

# SUMMARY --------------------------------------------------->>>

# I'm redefining things here, but `words` is defined at the top of the file

xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# convert the lists to tensors
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()  # number of examples
print("num examples: ", num)

g = torch.Generator().manual_seed(2147483647)
# requires_grads is needed to get pytorch to build a computational graph
W = torch.randn(27, 27, generator=g, requires_grad=True)

xenc = F.one_hot(xs, num_classes=27).float()  # note the casting to floats here

for k in range(200):
    # I think this can be defined outside the loop ^
    # xenc = F.one_hot(xs, num_classes=27).float()  # note the casting to floats here
    logits = xenc @ W

# this is the softmax function
# softmax `z_i = e^z_i / sum_row e^z` (messy way of writing it)
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)  # keepdims or keepdim? both seem to work

# look at ys
# print(ys)
# tensor([ 5, 13, 13,  1,  0])
# this is telling us that for the 0th row of xs, the label is pointing to the 5th element,
# the second row, the label is pointing to the 13th element, etc
# the probabilities that the neural network assigns to the correct next character:
# print(probs[torch.arange(5), ys])
# tensor([0.0123, 0.0181, 0.0267, 0.0737, 0.0150])

# more correctly, the negative log likelihood loss function
    loss = -probs[torch.arange(num), ys].log().mean()
    print(f"loss: {loss.item()}")

# backward pass
# set the gradients to 0
    W.grad = None  # pytorch way of resetting gradients
# the same as with micrograd, pytorch has build a complete computational graph
# backward() fills in all the gradients of all the intermediate steps, all the way back to W
    loss.backward()
# print(W.grad)

# update the params (a very high learning rate can be used for this model)
# a good loss, based on stats from counting would be ~2.45
    W.data += -50 * W.grad

# REGULARIZATION ------------------------------------->>
# starting at 1:52
# gradient based frameworks have an equivalent to smoothing
# if all weights are initialized to 0: `W = torch.zeros((27, 27))`
# then all logits will become 0: `xenc @ W = [tensor of zeros]`
# counts will then all be 1: `math.e**0 = 1`
# probabilities will then be uniform: `counts / counts.sum(1, keepdims=True) = [tensor of 1s]`
# therefore: trying to incentivize W to be near 0 produces a more uniform distribution (smoothing)
# Regularization augments the loss function to incentivize it to push weights towards 0
# e.g.: loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
# the loss function above is saying that if Ws are non-zero there's an additional loss (note that 0**2 = 0)

# analyze the results
# print("\n\n")
# nlls = torch.zeros(5)
# for i in range(5):
#     x = xs[i].item()  # input character index
#     y = ys[i].item()  # label character index
#     print("====================")
#     print(f"bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})")
#     print("input to the neural net:", x)
#     print("output probabilities from the neural net:", probs[i])
#     print("label (actual next character):", y)
#     p = probs[i, y]  # the label i.e. 5, is used to index into the probability distribution (probs)
#     print("probability asssigned by the next to the correct character:", p.item())
#     logp = torch.log(p)
#     print("log likelihood:", logp.item())
#     nll = -logp
#     print("negative log likelihood:", nll.item())
#     nlls[i] = nll
#
# print("===========")
# print("average negative log likelihood, i.e. loss =", nlls.mean().item())
