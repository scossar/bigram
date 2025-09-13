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

# visualize the array (simple):
# plt.imshow(N)
# plt.show()

# the counts array of the entire dataset:
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(
            j, i, str(N[i, j].item()), ha="center", va="top", color="gray"
        )  # cast item to string
plt.axis("off")
# plt.show()

first_row = N[
    0, :
]  # all columns for first row; the columns represent the counts for how many times the itos[column_index] itos[row_index]
# Note: N[0][0] would be the count of bigrams (".", "."), an empty string
# N[0][1] is a count of all names that begin with "a" (".a"); N[0][2] is a count of all names that begin with "b" (".b")
# N[1][0] is a count of all names that end with "a" ("a.")
# print(first_row)
# tensor([   0, 4410, 1306, 1542, 1690, 1531,  417,  669,  874,  591, 2422, 2963,
#         1572, 2538, 1146,  394,  515,   92, 1639, 2055, 1308,   78,  376,  307,
#          134,  535,  929], dtype=torch.int32)

# convert the counts to probabilities; e.g.:
# p = N[0].float()
# p = p / p.sum()  # probabilities

# sample from the distribution with torch.multinomial
# torch.multinomial samples from the multinomial probability distribution (p); give it probabilities and it
# returns integers
# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p = p / p.sum()
# print(p)
# tensor([0.6064, 0.3033, 0.0903])  # expect ~60% 0, ~30% 1, ~9% 2
# samptest = torch.multinomial(
#     p, num_samples=20, replacement=True, generator=g
# )  # replacement=True allows indices to appear more than once.
# print(samptest)
# tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])


g = torch.Generator().manual_seed(2147483647)
P = N.float()
P /= P.sum(
    1, keepdim=True
)  # `/=` is an "in-place" operation; more efficient than `P = P / P.sum(1, keepdim=True`
# confirm that it's worked:
# print(P[0].sum())  # tensor(1.)
# print(P[1].sum())  # tensor(1.)
for i in range(5):
    out = []
    ix = 0  # start from the start (".") character
    while True:
        p = P[int(ix)]
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
# .e: 0.0478 -3.0408
# em: 0.0377 -3.2793
# mm: 0.0253 -3.6772
# ma: 0.3899 -0.9418
# a.: 0.1960 -1.6299
# .o: 0.0123 -4.3982
# ol: 0.0780 -2.5508
# ...

# we can also test the probability of an individual word that may not exist in the dataset:

log_likelihood = 0.0  # remember: log(a*b*c) = log(a) + log(b) + log(c)
n = 0  # count of iterations for averaging loss
for w in ["an"]:
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
