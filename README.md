# Face Generator using DCGAN (PyTorch)

This project uses a Deep Convolutional GAN (DCGAN) to generate realistic human face images based on a custom image dataset. It uses PyTorch for model building and training.
## Features

- DCGAN-based generator and discriminator
- Fully convolutional architecture (CNN)
- Generate 64x64 RGB human face images
- Custom dataset support from ./images folder
- GPU acceleration if available
## Project Structure

.
├── images/             # Your dataset
├── try_image_gen.py    # Main training + generation script
├── generator.pth       # Saved Generator model
└── README.md           # This file
## Installation

Install the dependencies using pip:

```bash
pip install torch torchvision matplotlib tqdm
```

---

### 🛠️ 5. How to Use

**How to run or test the project?**

## Usage

1. Place your images in the `./images` folder inside subfolders.
2. Run the script:
```bash
python try_image_gen.py
```

---

### 🧠 6. Model Details (For ML Projects)

**Explain architecture and logic**, so others can learn from it.

## Model Architecture

- Generator: 5 ConvTranspose2d layers to upsample noise to image
- Discriminator: 5 Conv2d layers to classify real vs fake
- Loss: Binary Cross Entropy
- Optimizer: Adam

## Training Info

- Dataset: Custom images in ./images
- Image size: 64x64
- Epochs: 30
- Batch size: 64
- Latent dimension: 100
## Output

The model generates realistic human faces after training. Here's an example output:

![Generated Face](example_face.png)
## License

This project is licensed under the MIT License.
## Acknowledgements

- DCGAN paper: [Radford et al.](https://arxiv.org/abs/1511.06434)
- PyTorch framework

