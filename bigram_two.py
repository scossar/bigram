# starting at ~22:15 in the video
import torch
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

#################################################
# Bigram language model
#################################################

# Bigram model: predict the next character in a sequence from the character that immediately preceeds it

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
# foo = torch.rand(3, generator=g)
# foo = foo / foo.sum()
# foo_samps = torch.multinomial(foo, num_samples=20, replacement=True, generator=g)  # replacement=True means an int gets added back to available selections
# print(foo)
# print(foo_samps)

# video: ~30:35, why aren't I getting the same value as in the video?
# I think he forgot to redefine p
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(ix)

# sample a word, note the break statement
ix = 0
while True:
    p = N[ix].float()
    p = p / p.sum()
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    print(f"itos[ix]: {itos[ix]}")
    if ix == 0:
        break

