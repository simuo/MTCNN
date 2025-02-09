import os
from PIL import Image
import torch
import numpy as np


class loaddata():
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, 'positive.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'part.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'negative.txt')).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].split()
        confidence = torch.Tensor([int(strs[1].strip())])
        iou = torch.Tensor([float(strs[2])])
        offset = torch.Tensor([float(strs[3].strip()), float(strs[4].strip()), float(strs[5].strip()), float(strs[6].strip())])
        img_path = os.path.join(self.path, strs[0])
        img = Image.open(img_path)
        img_data = torch.Tensor(np.array(img) / 255 - 0.5)
        img_data = img_data.permute(2, 0, 1)

        return img_data, confidence, iou, offset


# data = loaddata(r'F:\celeba3\12')
# # data = loaddata(r'F:\data\img')
# print(data[0])
