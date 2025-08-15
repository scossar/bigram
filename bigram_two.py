# starting at ~22:15 in the video
import torch
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

#################################################
# Bigram language model
#################################################

# Bigram model: predict the next character in a sequence from the character that immediately preceeds it
# The elements of the array P (calculated below) are the "parameters" of the bigram language model
# P (capital P) is the probabilities of each pair in the training data

# getting rid of start/end characters used in bigram_one
N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
# create a mapping to ints
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0  # start/end char at position 0
itos = {i:s for s,i in stoi.items()}
print(stoi)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# # visualize the data
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")

# plt.show()

# print(N[0,:])  # the counts for each character at the start of a word

# create a probability vector for first row
# note that N[0] and N[0, :] are equivalent
p = N[0].float()
p = p / p.sum()
print(f"probabilities:\n{p}")

# use torch.multinomial for sampling: give the method a probability and it will return an integer
# use a torch Generator so we're (non-sic) all getting the same values
g = torch.Generator().manual_seed(2147483647)
# e.g.:
# p = torch.rand(3, generator=g)
# p = p / p.sum()

# video: ~30:35, why aren't I getting the same value as in the video?
# I think he forgot to redefine p
# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(ix)

# sample some words, note the break statement
for i in range(20):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p / p.sum()
        # check to see if a uniform distribution does better/worse 
        # returns results like "zoglkurkicqzktyhwmvmzimjttainrlkfukzkktda", so
        # we can have some confidence that the model trained on bigrams is doing something
        # p = torch.ones(27) / 27.0  
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))
# ~33:14

# efficiency improvement: instead of repeatedly calculating the probabilities for each row
# prepare a probabilities matrix
P = N.float()
divisor = P.sum(1, keepdim=True)  # look at broadcasting rules if you get confused by this
print(divisor.shape)  # for the divisor, sum across rows, return a 27,1 vector 
P = P / divisor  # broadcasting (the 1 dimension is copied 27 times, then does elementwise division)
print(sum(P[0]))  # should be 1.0

# then (same as above, but more efficient)
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))

# analyzing the quality of the output

for w in words[:3]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        print(f"{ch1}{ch2}: {prob:.4f}")

# .e: 0.0478
# em: 0.0377
# mm: 0.0253
# ma: 0.3899
# a.: 0.1960
# .o: 0.0123
# ol: 0.0780
# li: 0.1777
# iv: 0.0152
# vi: 0.3541
# ia: 0.1381
# a.: 0.1960
# .a: 0.1377
# av: 0.0246
# va: 0.2495
# a.: 0.1960
#
# How can the above probabilities be summarized into a single number?
# "Maximum likelihood estimation"  
#
# the "likelihood" is the product of all the P probabilities
# a model is "good" if the product of the probabilities is high
# the product of the probabilities will be a small number, as all probs are between 0, 1
# so the log of the probabilities is used:


print("log probs:")
for w in words[:3]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

# .e: 0.0478 -3.0408  # smaller probs tend toward negative infinity
# em: 0.0377 -3.2793
# mm: 0.0253 -3.6772
# ma: 0.3899 -0.9418  # larger probs tend towards 0
# a.: 0.1960 -1.6299
# .o: 0.0123 -4.3982
# ol: 0.0780 -2.5508
# li: 0.1777 -1.7278
# iv: 0.0152 -4.1867
# vi: 0.3541 -1.0383
# ia: 0.1381 -1.9796
# a.: 0.1960 -1.6299
# .a: 0.1377 -1.9829
# av: 0.0246 -3.7045
# va: 0.2495 -1.3882
# a.: 0.1960 -1.6299
#
# Note: log(a*b*c) = log(a) + log(b) + log(c)
# so:

log_likelihood = 0.0
for w in words[:3]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        log_likelihood += logprob

print(log_likelihood)
# tensor(-38.7856)
#
# the issue here is that we want to create a loss function
# loss functions have the semantics (convention) that low is good, so we want the
# "negative log likelihood"
nll = -log_likelihood
print(f"{nll=}")  # note the {nll=}, it returns:
# nll=tensor(38.7856)

# people tend to use the average negative loss likelihood:
log_likelihood = 0.0
n = 0
for w in words[:3]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        log_likelihood += logprob
        n += 1  # for calculating the average

nll = -log_likelihood
print(f"negative log likelihood: {nll:.4f}")
print(f"average negative log likelihood: {nll/n}")

# negative log likelihood: 38.7856
# average negative log likelihood: 2.424102306365967  # the lowest this can get is 0, the lower the better
#
# now try for entire training set

log_likelihood = 0.0
n = 0
for w in words:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        log_likelihood += logprob
        n += 1  # for calculating the average

nll = -log_likelihood
print(f"negative log likelihood: {nll:.4f}")
print(f"average negative log likelihood: {nll/n}")
# negative log likelihood: 559891.7500
# average negative log likelihood: 2.454094171524048  # for entire training set
#
# you can analyze the probability for any word

log_likelihood = 0.0
n = 0
for w in ["simon"]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        log_likelihood += logprob
        n += 1  # for calculating the average

nll = -log_likelihood
print(f"negative log likelihood for simon: {nll:.4f}")
print(f"average negative log likelihood simon: {nll/n}")

#################################
# model smoothing
# the more smoothing you add, the more uniform the model, the less smoothing added, the more peaked the model
#################################
# consider:
log_likelihood = 0.0
n = 0
for w in ["andrejq"]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        print(f"{ch1}{ch2} logprob: {logprob:.4f}")
        log_likelihood += logprob
        n += 1  # for calculating the average

nll = -log_likelihood
print(f"negative log likelihood for andrejq: {nll:.4f}")
print(f"average negative log likelihood andrejq: {nll/n}")
# .a logprob: -1.9829
# an logprob: -1.8296
# nd logprob: -3.2594
# dr logprob: -2.5620
# re logprob: -2.0127
# ej logprob: -5.9171
# jq logprob: -inf  # jq doesn't occur in the training set
# q. logprob: -2.2736
# negative log likelihood for andrejq: inf
# average negative log likelihood andrejq: inf

P = (N+1).float()  # smooth the model by adding 1. you can smooth as much as you like. e.g.: P = (N+5).float()
divisor = P.sum(1, keepdim=True)  # look at broadcasting rules if you get confused by this
P = P / divisor  # broadcasting (the 1 dimension is copied 27 times, then does elementwise division)
log_likelihood = 0.0
n = 0
for w in ["andrejq"]:  # be careful about printing the whole list!
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)  # torch.log([[1.]]) = tensor([[0.]]); torch.log([[0.5]]) = tensor([[-0.6931]])
        print(f"{ch1}{ch2} logprob: {logprob:.4f}")
        log_likelihood += logprob
        n += 1  # for calculating the average

nll = -log_likelihood
print(f"negative log likelihood for andrejq: {nll:.4f}")
print(f"average negative log likelihood andrejq: {nll/n}")
# .a logprob: -1.9835
# an logprob: -1.8302
# nd logprob: -3.2594
# dr logprob: -2.5646
# re logprob: -2.0143
# ej logprob: -5.9004
# jq logprob: -7.9817
# q. logprob: -2.3331
# negative log likelihood for andrejq: 27.8672
# average negative log likelihood andrejq: 3.4834020137786865

