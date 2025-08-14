# upto ~22:15 in the video
import torch
import matplotlib.pyplot as plt

words = open("./names.txt", "r").read().splitlines()

# dataset info
print(f"len(words): {len(words)}")
print(f"shortest word: {min(len(w) for w in words)}")
print(f"longest word: {max(len(w) for w in words)}")

# NOTE: each word is a sequence of characters that's telling us about the likelihood
# of one character following another. Each word also contains a start and an end, so there's also info
# about what character is likely to come after the start and before the end.

#################################################
# Bigram language model
#################################################

# Bigram model: predict the next character in a sequence from the character that immediately preceeds it

# quick look at the biggrams
for w in words[:3]:
    # zip takes two iterators, pairs them up and creates an iterator
    # over the tuples of their consecutive entries
    # if any one of the lists (one of the iterators) is shorter than the other
    # zip will halt and return
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)

# now add info about start and end
# and also get bigram counts
b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # NOTE: be careful about printing the full list!

# b.items() returns a dict_items of key/value tuples
# sorted() sorts by kv[0] by default, the lambda (with the - sign) gets it to sort by
# the value in reverse
b_sorted = sorted(b.items(), key = lambda kv: -kv[1])
print(b_sorted[:3])

# store the data in a 2D array where the rows are the first character of the bigram
# and the columns are the second character
# the entries will tell us how often the second character follows the first character in the dataset

# torch example:
a = torch.zeros((3, 5))
print(f"a:\n{a}")
print(f"a.dtype: {a.dtype}")  # returns torch.float32

# because we're storing counts, we'll use, e.g:
# torch.zeros((3, 5), dtype=torch.int32)

# to represent all letters, plus the start and end characters
N = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
# create a mapping to ints
stoi = {s:i for i,s in enumerate(chars)}
stoi["<S>"] = 26
stoi["<E>"] = 27

for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# visualize the data
# simple, but not useful:
# plt.imshow(N)

# invert the dict
itos = {i:s for s,i in stoi.items()}
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")

# the plot makes it obvious we've ended up with an entire row of 0 values (the end
# token will never start a bigram) and an entire column of 0 values (the start character will never
# be the second character unless we allow for empty strings as words)

# plt.show()
