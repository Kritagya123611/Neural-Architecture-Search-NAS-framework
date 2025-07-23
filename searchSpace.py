# iss file mein hum search space define karenge

from itertools import product

# 4 properties jo hum search karenge
CONV_KERNELS = [3, 5]               # kernel size: 3x3 ya 5x5
FILTER_SIZES = [32, 64]            # filters: 32 ya 64
ACTIVATIONS = ['relu', 'tanh']     # activation: relu ya tanh
POOLING = ['max', 'avg']           # pooling type: max ya average
DENSE_SIZES = [128, 256]           # dense layer size: 128 ya 256
conv_block_options = [
    {'kernel_size': 3, 'filters': 32, 'activation': 'relu'},
    {'kernel_size': 3, 'filters': 32, 'activation': 'tanh'},
    {'kernel_size': 3, 'filters': 64, 'activation': 'relu'},
    {'kernel_size': 3, 'filters': 64, 'activation': 'tanh'},
    {'kernel_size': 5, 'filters': 32, 'activation': 'relu'},
    {'kernel_size': 5, 'filters': 32, 'activation': 'tanh'},
    {'kernel_size': 5, 'filters': 64, 'activation': 'relu'},
    {'kernel_size': 5, 'filters': 64, 'activation': 'tanh'},
]
DENSE_UNITS = [128, 256]

# tokens ko actual neural net components mein decode karte hain
def decode(tokens):
    # har token se conv block banate hain 
    conv_blocks = [conv_block_options[t] for t in tokens[:-1]]
    # last token dense layer unit decide karega
    dense_units = [DENSE_UNITS[tokens[-1]]]
    return {
        'conv_blocks': conv_blocks,
        'dense_units': dense_units
    }
