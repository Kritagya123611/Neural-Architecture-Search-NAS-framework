# child_model.py
# Is file mein hum ChildNet class define kar rahe hain
# Ye class dynamically layers banata hai based on controller's recipe

import torch
import torch.nn as nn

class ChildNet(nn.Module):
    def __init__(self, genotype, input_channels=1, num_classes=10):
        """
        genotype: dictionary jisme conv_blocks aur dense_units hain
        input_channels: MNIST ke liye 1 (grayscale image)
        num_classes: 10 digits (0-9)
        """
        super().__init__()

        layers = []  # PyTorch layers yahan store karenge
        in_ch = input_channels  # starting input channels = 1

        # ✅ Step 1: Convolution blocks add karte hain
        for (kernel, out_ch, activation, pooling) in genotype['conv_blocks']:
            # Conv Layer
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2))
            layers.append(nn.BatchNorm2d(out_ch))  # normalization for faster training

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

            # Pooling
            if pooling == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.append(nn.AvgPool2d(kernel_size=2))

            in_ch = out_ch  # next layer ke liye input channels update karo

        # ✅ Step 2: Flatten before Dense layers
        layers.append(nn.Flatten())

        # Flattened size calculate karna padega dynamically
        # MNIST images: 28x28 → har pooling se size half hota hai
        conv_blocks_count = len(genotype['conv_blocks'])
        final_size = 28 // (2 ** conv_blocks_count)  # pooling reduce karta hai
        flat_input_size = in_ch * final_size * final_size  # H x W x C

        # ✅ Step 3: Dense (fully connected) layer(s)
        for units in genotype['dense_units']:
            layers.append(nn.Linear(flat_input_size, units))
            layers.append(nn.ReLU())
            flat_input_size = units  # next dense layer input = current output

        # ✅ Step 4: Final Output layer (10 digits classification)
        layers.append(nn.Linear(flat_input_size, num_classes))

        # ✅ Final model as Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
