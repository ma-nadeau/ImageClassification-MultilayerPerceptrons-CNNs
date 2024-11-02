from medmnist import OrganAMNIST, INFO
import ssl
import matplotlib.pyplot as plt

# Bypass SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# Load the OrganAMNIST dataset information and download the train dataset
info = INFO['organamnist']
train_dataset = OrganAMNIST(split="train", download=True)

# Display each image in the dataset
for data in train_dataset:
    image, label  = data 
    plt.imshow(image, cmap="gray") 

    if isinstance(label, (list, tuple)):
        label_value = label[0]  # Extract the first element if it's a list or tuple
    else:
        label_value = label  # Otherwise, use the label directly

    plt.title(label_value)
    plt.axis("off")
    plt.show()
    
