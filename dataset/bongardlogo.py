import json
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUM_PER_CLASS = 7


class BongardLogoDataset(Dataset):
    def __init__(
        self,
        data_root,
        dset="train",
        img_size=512,
        use_augs=False,
        n_samples=-1,
    ):
        self.dset = dset

        split_file = os.path.join(data_root, "ShapeBongard_V2_split.json")
        split = json.load(open(split_file, "r"))

        self.tasks = sorted(split[dset])
        self.n_tasks = len(self.tasks)
        print("found %d tasks in dset %s" % (self.n_tasks, dset))

        task_paths = [
            os.path.join(data_root, task.split("_")[0], "images", task)
            for task in self.tasks
        ]

        self.task_paths = task_paths

        norm_params = {"mean": [0.5], "std": [0.5]}  # grey-scale to [-1, 1]
        normalize = transforms.Normalize(**norm_params)

        if use_augs:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.75, 1.2)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        self.n_samples = n_samples

    def __len__(self):
        if self.n_samples > 0:
            return self.n_samples
        return len(self.task_paths)

    def __getitem__(self, index):
        task_path = self.task_paths[index]

        neg_imgs = []
        for idx in range(NUM_PER_CLASS):
            file_path = "%s/0/%d.png" % (task_path, idx)
            img = Image.open(file_path).convert("L")
            neg_imgs.append(self.transform(img))
        pos_imgs = []
        for idx in range(NUM_PER_CLASS):
            file_path = "%s/1/%d.png" % (task_path, idx)
            img = Image.open(file_path).convert("L")
            pos_imgs.append(self.transform(img))

        neg_imgs = torch.stack(neg_imgs, dim=0)
        pos_imgs = torch.stack(pos_imgs, dim=0)

        if self.dset == "train":
            perm = np.random.permutation(NUM_PER_CLASS)
            pos_imgs = pos_imgs[perm]
            perm = np.random.permutation(NUM_PER_CLASS)
            neg_imgs = neg_imgs[perm]

        pos_support = pos_imgs[:-1]
        neg_support = neg_imgs[:-1]
        pos_query = pos_imgs[-1]
        neg_query = neg_imgs[-1]

        x_support = torch.cat([pos_support, neg_support], dim=0)
        randp = torch.rand(1)
        label = 0
        if (randp < 0.5):
            x_query = torch.stack([pos_query, neg_query])
        else:
            x_query = torch.stack([neg_query, pos_query])
            label = 1
        x = torch.cat([x_support, x_query], dim=0)
        x_size = x.shape
        x = x.view(x_size[0], x_size[2], x_size[3])
        
        return x, label

