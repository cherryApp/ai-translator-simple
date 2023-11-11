import sys
import timeit
import torch
from transformers import pipeline

if torch.backends.mps.is_available():
    print('MPS')

a_cpu = torch.rand(1000, device='cpu')
b_cpu = torch.rand((1000, 1000), device='cpu')
a_mps = torch.rand(1000, device='mps')
b_mps = torch.rand((1000, 1000), device='mps')

print('cpu', timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
print('mps', timeit.timeit(lambda: a_mps @ b_mps, number=100_000))
# sys.exit(0)

# print(torch.__version__)
# sys.exit(0)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-hu", device=device)
print(pipe("I wish I hadn't seen such a horrible film."))

# expected output: Bárcsak ne láttam volna ilyen szörnyű filmet.
