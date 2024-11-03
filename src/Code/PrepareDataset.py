import ssl

import numpy as np
from medmnist import INFO
from MedMnistCode import get_loader
from src.Code import MedMnistCode

# Bypass SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

data_flag = 'organamnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 32
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(MedMnistCode, info['python_class'])

# load the data
train_dataset = DataClass(split='train', download=download, size=28)
test_dataset = DataClass(split='test', download=download, size=28)

print(train_dataset)
# encapsulate data into dataloader form
def data_generator():
    train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_loader = get_loader(dataset=test_dataset, batch_size=BATCH_SIZE)
    return train_loader, test_loader

# for x, y in train_loader:
#     print(x.shape, y.shape)
#     break
# for x, y in test_loader:
#     print(x.shape, y.shape)
#     break


# for x, y in train_loader:
#     # Flatten each image in the batch to a 1D vector of 784 elements
#     x = np.array(x).reshape(x.shape[0], -1)  # Converts from (32, 28, 28) to (32, 784)
#     y = np.array(y).reshape(y.shape[0], -1)  # Converts from (32, 1) to (32,)
#
#     break
#
# for x, y in test_loader:
#     # Flatten each image in the batch to a 1D vector of 784 elements
#     x = np.array(x).reshape(x.shape[0], -1)  # Converts from (32, 28, 28) to (32, 784)
#     y = np.array(y).reshape(y.shape[0], -1)  # Converts from (32, 1) to (32,)
#
#     break

# train_dataset.montage(length=20)
#
# montage_image = train_dataset.montage(length=20)
# plt.imshow(montage_image)
# plt.axis('off')
# plt.show()
#
