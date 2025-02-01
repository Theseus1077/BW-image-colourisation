# BW Image Colourization

## Overview
This project focuses on colorizing black-and-white images using deep learning techniques. The implemented model leverages convolutional neural networks (CNNs) to learn and predict realistic colors for grayscale images.

## Features
- Automatic colorization of grayscale images
- Implementation using PyTorch
- Pretrained models for better accuracy
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

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Theseus1077/BW-image-colourisation.git
   cd BW-image-colourisation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Test the model with custom images:
   ```bash
   python test.py --image_path path/to/your/image.jpg
   ```

## Usage
- Train the model using the provided dataset.
- Test the trained model on grayscale images.
- Modify model parameters in `config.py` for experimentation.

## Model Architecture
The project utilizes a **U-Net** based deep learning architecture, commonly used for image segmentation and restoration tasks. The model consists of:
- **Encoder**: Extracts feature representations from grayscale images.
- **Decoder**: Predicts color channels based on learned features.
- **Skip connections**: Preserve fine details during reconstruction.

## Challenges Faced
- **Hardware limitations**: Training deep networks required optimization to reduce resource consumption.
- **Dataset constraints**: Additional data augmentation was applied to enhance generalization.
- **Color accuracy**: Loss functions such as perceptual loss were introduced to improve results.

## Future Improvements
- Implementation of GANs for more realistic colorization.
- Integration of user-guided color inputs.
- Extending the model to support higher-resolution images.

## Contributors
- [Theseus1077](https://github.com/Theseus1077)

