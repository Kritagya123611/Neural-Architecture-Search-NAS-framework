# 🧠 Neural Architecture Search (NAS) in PyTorch

This repository implements a simplified version of **Neural Architecture Search (NAS)** using PyTorch — inspired by the paper *“Neural Architecture Search with Reinforcement Learning”* by Zoph & Le (2016).

> 🎯 The goal: Build an AI that can generate and evaluate other neural networks — a core idea behind AutoML.

---

## 🚀 What This Project Does

This framework builds and trains neural networks **dynamically** from a list of tokens (genotypes), representing architecture choices like:

- 🧱 Convolution kernel size (3×3 or 5×5)
- 🧠 Number of filters (32, 64, etc.)
- ⚡ Activation function (ReLU or Tanh)
- 🌀 Pooling type (Max or Avg)
- 🏗️ Dense layer size (128, 256, 512)

Each set of tokens is converted into a custom PyTorch model (`ChildNet`) and trained on **MNIST** to evaluate its performance.

---

## 🎮 Core Components

| File | Description |
|------|-------------|
| `searchSpace.py` | Defines the modular search space and token decoding logic |
| `child.py`       | Dynamically builds a PyTorch model from a genotype |
| `dataset.py`     | Loads and splits the MNIST dataset into train/val sets |
| `train.py`       | Trains the generated model and returns validation accuracy |
| `testChild.py`   | Tests model construction from tokens |

---

## 📊 Example Result

A generated model like this:
```python
tokens = [0, 4, 6, 1]  # 3 conv blocks + 1 dense layer
Will be decoded to:

{
  'conv_blocks': [
    {'kernel_size': 3, 'filters': 32, 'activation': 'relu'},
    {'kernel_size': 5, 'filters': 32, 'activation': 'relu'},
    {'kernel_size': 5, 'filters': 64, 'activation': 'relu'}
  ],
  'dense_units': [256]
}
```
And after training for 3 epochs:

bash
Copy code
Validation Accuracy: 99.03%
🔥 Features
 Modular architecture encoding and decoding

 Dynamic model creation via tokenized genotype

 Validation-based reward signal (for controller in next step)

 99%+ accuracy on MNIST in <3 epochs

 Reinforcement Learning controller (coming soon)

📦 Stack Used
1) Python 3.11+

2) PyTorch

3) torchvision

4) MNIST dataset

[Coming Soon] REINFORCE policy gradient for controller
