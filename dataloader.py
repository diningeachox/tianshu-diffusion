import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode

#Data taken from http://technology.chtsai.org/charfreq/characters.html
def read_data(path, n=1000):
     chars = []
     probs = []
     total_weight = 0
     with open(path, encoding='utf8') as file:
         while True:
             line = file.readline()
             if len(line) == 0:
                 break
             char = line[:1]
             freq = int(line[6:13].strip())
             strokes = int(line[15:17].strip())
             chars.append(char)
             probs.append(freq)
             total_weight += freq
     total = np.sum(np.array(probs)[:n])
     return np.array(probs[:n]) / total

'''
Dataset of all standard Chinese characters
'''
class CCDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = [f"char_{i}.png" for i in range(1000)] #13060 characters
        self.img_dir = img_dir
        self.transform = transform
        self.augmentations = 1

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

if __name__ == "__main__":
    weights = read_data("chars.txt")
    print(weights)
