import torch
import torch.nn.functional as F

words = open("./names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0  # start/end char is represented by 0

xs, ys = [], []

for w in words[:1]:  # just selecting the first word for now
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        # print(ch1, ch2)
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)  # inputs
        ys.append(ix2)  # labels

# pairs for first word:
# . e
# e m
# m m
# m a
# a .

# print(xs)
# print(ys)
# [0, 5, 13, 13, 1]  # inputs
# [5, 13, 13, 1, 0]  # labels

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()  # number of examples
g = torch.Generator().manual_seed(2147483647)
# W corresponds to 27 neurons/outputs x 27 classes/features
W = torch.randn(27, 27, generator=g, requires_grad=True)
xenc = F.one_hot(xs, num_classes=27).float()
# print(xenc.shape)
# torch.Size([5, 27]) num_examples x classes/features per example
# print(xenc[0])
# tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0.])  # the tensor that represents the "." character
# print(xenc[1])
# tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0.])  # the tensor that represents the "e" character

for k in range(100):
    logits = xenc @ W
    # print(logits.shape)
    # torch.Size([5, 27])

    counts = logits.exp()  # e^logit (counts range is 0 to +inf)
    probs = counts / counts.sum(1, keepdims=True)  # normaized counts (so they can be used as probabilities)

    if (k == 0 or k == 99):
        print(f"probs for k = {k}:\n{probs}")

    # confirm that the loss function is finding the probability associated with each y label:
    # print(probs[torch.arange(num), ys])
    # print(probs[0,5].item(), probs[1,13].item(), probs[2,13].item(), probs[3,1].item(), probs[4,0].item())
    # tensor([0.0123, 0.0181, 0.0267, 0.0737, 0.0150], grad_fn=<IndexBackward0>)
    # 0.01228625513613224 0.018050700426101685 0.026691533625125885 0.07367686182260513 0.014977526850998402

    # calculate negative log likelihood
    loss = -probs[torch.arange(num), ys].log().mean()  # or add `+ 0.1*(W**2).mean()` for regularization
    print(f"loss: {loss.item()}")

    W.grad = None  # pytorch way of resetting gradients
    loss.backward()

    # update the params (a very high learning rate can be used for this model)
    W.data += -1 * W.grad
