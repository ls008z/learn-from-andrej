import torch
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(2147483647)

data_path = "/Users/leos/LearnCS/learn-from-andrej/makemore/names.txt"

words = open(data_path, "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

N = torch.zeros((len(chars) + 1, len(chars) + 1), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# plot the matrix as a heatmap
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(len(chars) + 1):
    for j in range(len(chars) + 1):
        chstr = itos[i] + itos[j]
        # really weird indexing here
        plt.text(j, i, chstr, ha="center", va="bottom", color="black")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.show()

P = (N + 1).float()
# some notes on broadcasting:
# 27, 27
# 27, 1
# the 1 is getting broadcasted to 27
P /= P.sum(dim=1, keepdim=True)
# website for more information on broadcasting:
# https://pytorch.org/docs/stable/notes/broadcasting.html

ix = 0
while True:
    p = P[ix].float()
    ix = torch.multinomial(p, num_samples=1, generator=g, replacement=True).item()
    ch = itos[ix]
    if ch == ".":
        break
    print(ch, end="")


log_likelihood = 0.0
n = 0
for w in ["jq"]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}->{ch2}: {logprob:.4f}")
nll = -log_likelihood / n
print(f"average negative log likelihood: {nll:.4f}")
