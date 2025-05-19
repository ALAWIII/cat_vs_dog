import torch

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
DATA_DIR = "./PetImages"  # Updated path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
