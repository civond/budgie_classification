from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataframe, train=True, transform=None):
        self.transform = transform
        self.images = dataframe['path'] # Path
        self.labels = dataframe['label'] # Label
        self.train=train
        unique_targets = np.unique(self.labels)
        print(f"Unique Targets: {unique_targets}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("L"))
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)

        return image, label