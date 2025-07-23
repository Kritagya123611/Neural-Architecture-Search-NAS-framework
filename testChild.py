# test_childnet.py
from child import ChildNet
from searchSpace import decode
# Dummy genotype: 3 conv blocks + 1 dense layer
tokens = [0, 4, 12, 1]  # randomly chosen tokens
genotype = decode(tokens)

print("Genotype:", genotype)

model = ChildNet(genotype)
print(model)
