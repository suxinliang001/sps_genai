from torch import nn


ARCHITECTURE_ID = "assignment3-mnist-v1"
LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.network(images)


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name="GAN"):
    if model_name.upper() != "GAN":
        raise ValueError("This assignment module implements GAN only.")
    return GAN()


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
