import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode

'''
Dataset of all standard Chinese characters
'''
class CCDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = [f"char_{i}.png" for i in range(2000)] #20971 characters
        self.img_dir = img_dir
        self.transform = transform
        self.augmentations = 10

        #Load character images
        self.orig_images = []
        self.images = []
        for path in self.img_labels:
            img_path = os.path.join(self.img_dir, path)
            image = read_image(img_path, mode=ImageReadMode.GRAY)
            #Augment the image with small perturbations
            for i in range(self.augmentations):
                new_image = image
                self.images.append(new_image)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels[idx])
        #image = read_image(img_path)
        #image = read_image(img_path, mode=ImageReadMode.GRAY)
        actual_idx = idx // self.augmentations
        label = self.img_labels[actual_idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
