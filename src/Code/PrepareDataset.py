
from medmnist import OrganAMNIST, INFO
import ssl
import matplotlib.pyplot as plt

# Bypass SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# Load the OrganAMNIST dataset information and download the train dataset
train_dataset = OrganAMNIST(split="train", download=True)

