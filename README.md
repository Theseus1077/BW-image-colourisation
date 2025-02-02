# BW Image Colourization

## Overview
This project focuses on colorizing black-and-white images using deep learning techniques. The implemented model leverages convolutional neural networks (CNNs) to learn and predict realistic colors for grayscale images.

## Features
- Automatic colorization of grayscale images
- Implementation using PyTorch
- Custom image testing support
- Dataset preprocessing and augmentation

## Dataset
This project uses the **CIFAR-10** dataset for training. CIFAR-10 is a well-known dataset consisting of 60,000 32x32 color images in 10 classes.

### Download CIFAR-10
To download the dataset manually, visit: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternatively, use the following Python code to download and preprocess CIFAR-10 using PyTorch:
```python
from torchvision import datasets, transforms

# Download and load CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
```

## Contributors
- [Theseus1077](https://github.com/Theseus1077)

