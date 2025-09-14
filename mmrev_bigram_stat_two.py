# continuing from `mmrev_bigram.py`
# the simplist bigram language model implementation is to just count
# how often each pair occurs in the dataset

import torch
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

print("number of names:", len(words))
shortest_name = min(len(w) for w in words)
print("shortest name:", shortest_name)
longest_name = max(len(w) for w in words)
print("longest name:", longest_name)

# We're going to store this information in a 2D array. the rows will be the first character
# of each bigram and the columns will be the second character. Each entry in the array will tell
# us how often the second character follows the first character in the dataset.

N = torch.zeros(
    (27, 27), dtype=torch.int32
)  # one row/column for each character + extra row/column for "."
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}  # works because `chars` is a list
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(2147483647)

# convert the numeric counts (N) to probabilities (P); probabilites are a series of numbers in the range (0, 1) that sum to 1
P = (N + 1).float()  # model smoothing; add 1 to every count
P /= P.sum(
    1, keepdim=True
)  # `/=` is an "in-place" operation; more efficient than `P = P / P.sum(1, keepdim=True)`
# confirm that it's worked:
# print(P[0].sum())  # tensor(1.)
# print(P[1].sum())  # tensor(1.)

# sampling for new names
for i in range(5):
    out = []
    ix = 0  # start from the start (".") character
    while True:
        p = P[
            int(ix)
        ]  # the probability distribution for `ix` (the row that represents the first character)
        # p = (
        #     torch.ones(27) / 27.0
        # )  # to convince yourself the model is doing something, check the results with an even distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[int(ix)])
        if ix == 0:
            break
    print("".join(out))

# the above code implements a trained model; the following code supply a way of analyzing the quality of the model
# ################################################################################################################

# the product of the probabilities is the likelihood;
# the log of the product of the probabilities is the log likelihood
log_likelihood = 0.0  # remember: log(a*b*c) = log(a) + log(b) + log(c)
n = 0  # count of iterations for averaging loss
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[
            ix1, ix2
        ]  # what probability is assigned by the trained model to the actual next character
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")  # don't print for full word list!

print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"average nll: {nll / n}")

# we can also test the probability of an individual word that may not exist in the dataset:

log_likelihood = 0.0  # remember: log(a*b*c) = log(a) + log(b) + log(c)
n = 0  # count of iterations for averaging loss
for w in ["annajq"]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[
            ix1, ix2
        ]  # what probability is assigned by the trained model to the actual next character
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(
            f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}"
        )  # don't print for full word list!

print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"average nll: {nll / n}")
