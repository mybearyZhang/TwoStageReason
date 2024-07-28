import os
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms

NUM_PER_CLASS = 6

class BongardDataset(Dataset):
    def __init__(self, data_root, img_size=256, use_clip=False, n_samples=-1):
        self.task_paths = []
        for problem in os.listdir(data_root):
            if not problem.startswith("p"):
                continue
            self.task_paths.append(os.path.join(data_root, problem))
        ops = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
        if use_clip:
            norm_params = {
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
            }
            normalize = transforms.Normalize(**norm_params)
            ops.append(normalize)
        else:
            norm_params = {"mean": [0.5], "std": [0.5]}  # grey-scale to [-1, 1]
            normalize = transforms.Normalize(**norm_params)
            ops.append(normalize)
        self.transform = transforms.Compose(ops)
        self.use_clip = use_clip
        self.n_samples = n_samples

    def __len__(self):
        """
        In order to expand the size of the dataset and reduce variance, we
        consider every possible choice of positive and negative query for each problem.
        As a result, the dataset size is expanded by a factor of NUM_PER_CLASS**2.
        """
        if self.n_samples > 0:
            return self.n_samples
        return len(self.task_paths) * NUM_PER_CLASS * NUM_PER_CLASS

    def __getitem__(self, i):
        problem_index = i // (
            NUM_PER_CLASS * NUM_PER_CLASS
        )  # NUM_PER_CLASS**2 indices for each problem
        task_path = self.task_paths[problem_index]

        pos_imgs = []
        for j in range(NUM_PER_CLASS):
            file_path = "%s/%d.png" % (task_path, j)
            img = Image.open(file_path).convert("L")
            if self.use_clip:
                img = transforms.functional.to_grayscale(img, num_output_channels=3)
            pos_imgs.append(self.transform(img))

        neg_imgs = []
        for j in range(NUM_PER_CLASS, 2 * NUM_PER_CLASS):
            file_path = "%s/%d.png" % (task_path, j)
            img = Image.open(file_path).convert("L")
            if self.use_clip:
                img = transforms.functional.to_grayscale(img, num_output_channels=3)
            neg_imgs.append(self.transform(img))

        neg_imgs = torch.stack(neg_imgs, dim=0)
        pos_imgs = torch.stack(pos_imgs, dim=0)

        pos_neg_choice = i % (
            NUM_PER_CLASS * NUM_PER_CLASS
        )  # Defines the particular choice of positive and negative query
        pos_choice = pos_neg_choice % NUM_PER_CLASS
        neg_choice = pos_neg_choice // NUM_PER_CLASS
        neg_support = torch.cat([neg_imgs[:neg_choice], neg_imgs[neg_choice + 1 :]])
        pos_support = torch.cat([pos_imgs[:pos_choice], pos_imgs[pos_choice + 1 :]])
        neg_query = neg_imgs[neg_choice]
        pos_query = pos_imgs[pos_choice]

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