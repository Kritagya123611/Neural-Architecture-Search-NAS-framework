# test_childnet.py

from child import ChildNet
from searchSpace import decode

tokens = [0, 4, 6, 1]  # 3 conv blocks, 1 dense

genotype = decode(tokens)
print("Genotype:", genotype)

model = ChildNet(genotype)
print(model)