# iss file mein hum search space define karenge

from itertools import product

# 4 property hoga
CONV_KERNELS = [3, 5]               # kernel size: 3x3 ya 5x5
FILTER_SIZES = [32, 64]            # filters: 32 ya 64
ACTIVATIONS = ['relu', 'tanh']     # activation: relu ya tanh
POOLING = ['max', 'avg']           # pooling type: max ya average
DENSE_SIZES = [128, 256]             # dense layer size: 128 ya 256

# tokens ko actual search space mein convert karte hain
def decode(tokens):
    conv_blocks = [conv_block_options[t] for t in tokens[:-1]]     # har token se conv block banta hai
    dense_units = [DENSE_UNITS[tokens[-1]]]                        # last token se dense layer decide hoti hai

    return {
        'conv_blocks': conv_blocks,
        'dense_units': dense_units
    }