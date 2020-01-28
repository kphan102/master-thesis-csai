import os
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import datasets, transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomErasing(object):
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class GanReIdFolder(datasets.ImageFolder):

    def __init__(self, attr_frame, *args, **kwargs):
        # attr_frame is a csv file
        super(GanReIdFolder, self).__init__(*args, **kwargs)
        self.attr_frame = pd.read_csv(attr_frame)

    def __getitem__(self, index):
        sample, target = super(GanReIdFolder, self).__getitem__(index)
        attributes = self.attr_frame.iloc[target, 1:].values
        attributes = torch.LongTensor(attributes)
        return sample, target, attributes # sample, target(class of sample), attributes


class TripletReIdFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletReIdFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        cams = [os.path.basename(s[0]).split('c')[1][0] for s in self.samples]
        self.cams = np.asarray(cams)

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)

        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
       
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
           t = i%len(rand)
           tmp_index = pos_index[rand[t]]
           result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]

        pos_path = self._get_pos_sample(target, index)

        sample = self.loader(path)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])

        if self.transform is not None:
            sample = self.transform(sample)
            pos0 = self.transform(pos0)
            pos1 = self.transform(pos1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        c,h,w = pos0.shape
        pos = torch.cat((pos0.view(1,c,h,w), pos1.view(1,c,h,w)), 0)
        pos_target = target

        return sample, target, pos, pos_target
        