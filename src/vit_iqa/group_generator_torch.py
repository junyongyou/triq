import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import random


class GroupGenerator(Dataset):
    """
    Generator to supply group image data, individual dataset should go to individual group because they can have different resolutions
    """
    def __init__(self, image_file_groups, score_groups, batch_size, transform=None, shuffle=True):
        self.image_file_groups = image_file_groups
        self.score_groups = score_groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        if transform:
            # do image augmentation by left-right flip
            self.transform = transform
        self.read_data()

    def __len__(self):
        return sum(self.group_length)

    def shuffle_dataset(self):
        for i in range(len(self.image_file_groups)):
            random.shuffle(self.index_groups[i])
            print('{} group shuffled'.format(i))

    def read_data(self):
        images_scores = list(zip(self.image_file_groups, self.score_groups))
        self.image_file_groups, self.score_groups = zip(*images_scores)

        self.index_groups = []
        self.group_length = []
        for i in range(len(self.image_file_groups)):
            self.index_groups.append(np.arange(len(self.image_file_groups[i])))
            self.group_length.append(len(self.image_file_groups[i]) // self.batch_size)

    def __getitem__(self, item):
        lens = 0
        idx_0 = len(self.group_length) - 1
        for i, data_len in enumerate(self.group_length):
            lens += data_len
            if item < lens:
                idx_0 = i
                break
        item -= (lens - self.group_length[idx_0])

        images = []
        y_scores = []

        for idx_1 in self.index_groups[idx_0][item * self.batch_size: (item + 1) * self.batch_size]:
            image = Image.open(self.image_file_groups[idx_0][idx_1])
            if self.transform:
                image = self.transform(image)
            images.append(image)
            y_scores.append(torch.tensor(self.score_groups[idx_0][idx_1]))

        # return torch.squeeze(torch.stack(images), dim=0), torch.squeeze(torch.stack(y_scores), dim=0)
        return torch.stack(images), torch.stack(y_scores)


