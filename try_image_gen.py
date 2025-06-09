import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 64
batch_size = 64
latent_dim = 100
epochs = 100  # increased epochs
lr = 0.0002
beta1 = 0.5

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1,1]
])

# Dataset & loader
dataset = datasets.ImageFolder(root="./images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Found classes: {dataset.classes}")
print(f"Total images: {len(dataset)}")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator (no Sigmoid at end)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
            # No Sigmoid here!
        )

    def forward(self, x):
        return self.net(x).view(-1)  # Flatten output to [batch]

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
print("Starting training...")
for epoch in range(epochs):
    for imgs, _ in tqdm(dataloader):
        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        # Label smoothing for real images
        real_labels = torch.full((batch_size_curr,), 0.9, device=device)  # smooth real labels
        fake_labels = torch.zeros(batch_size_curr, device=device)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()

        # Real images
        outputs_real = discriminator(real_imgs)
        loss_real = criterion(outputs_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise)
        outputs_fake = discriminator(fake_imgs.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        # Total discriminator loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise)
        outputs = discriminator(fake_imgs)
        loss_G = criterion(outputs, real_labels)  # want generator to fool discriminator
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {loss_D.item():.4f} G Loss: {loss_G.item():.4f}")

    # Save sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim, 1, 1, device=device)
            sample_imgs = generator(sample_noise)
            utils.save_image((sample_imgs + 1) / 2, f"generated_epoch_{epoch+1}.png", nrow=4)
        print(f"Saved sample images at epoch {epoch+1}")

# Save generator model
torch.save(generator.state_dict(), "generator.pth")
print("Generator model saved as generator.pth")

# Generate and show one face
def generate_one_face(model_path="generator.pth"):
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_img = gen(noise).cpu().squeeze(0)
    img = (fake_img + 1) / 2  # scale to [0,1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.title("Generated Human Face")
    plt.imshow(img)
    plt.show()

generate_one_face()
