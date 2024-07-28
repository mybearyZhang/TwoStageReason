import json
import os

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class BongardHoiDataset(Dataset):
    def __init__(
        self,
        data_root,
        data_split="unseen_obj_unseen_act",
        dset="test",
        balance_dataset=True,  # Use the cleaned dataset described in the supplementary
        use_augs=False,
        img_size=224,
        backbone="clip",
        n_samples=-1,
    ):
        self.data_root = data_root
        self.dset = dset
        self.balance_dataset = balance_dataset

        assert dset in ["val", "test", "train"]

        if self.balance_dataset:
            cache_path = "cache/bongard_hoi_clean"
        else:
            cache_path = "cache/bongard_hoi_release"

        bongard_splits_datapath = os.path.join(data_root, cache_path)
        data_split = dset if dset == "train" else f"{dset}_{data_split}"
        data_file = os.path.join(
            bongard_splits_datapath, f"bongard_hoi_{data_split}.json"
        )
        self.task_list = []
        with open(data_file, "r") as fp:
            task_items = json.load(fp)
            for task in task_items:
                task_data = {}
                neg_samples = task[1]
                pos_samples = task[0]

                task_data["pos_samples"] = pos_samples
                task_data["neg_samples"] = neg_samples
                task_data["annotation"] = task[-1].replace("++", " ")

                self.task_list.append(task_data)

        # Transforms
        if backbone == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        elif backbone == "dino_vit_base":
            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]],
            )
        else:
            raise ValueError("Using unsupported backbone on HOI")

        if use_augs:
            ops = [
                transforms.Resize(img_size * 2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(),
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.1, 2.0),
                    ratio=(0.8, 1.2),
                    interpolation=BICUBIC,
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            ops = [
                transforms.Resize(img_size, interpolation=BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]

        self.data_transform = transforms.Compose(ops)
        self.n_samples = n_samples

    def __len__(self):
        if self.n_samples > 0:
            return self.n_samples
        return len(self.task_list)

    def load_image(self, sample):
        path = sample["im_path"]
        im_path = os.path.join(self.data_root, path.replace("./", ""))
        if not os.path.isfile(im_path):
            print("File doesn't exist: {}".format(im_path))
            if "/pic/image/val" in im_path:
                im_path = im_path.replace("val", "train")
            elif "/pic/image/train" in im_path:
                im_path = im_path.replace("train", "val")
        try:
            image = Image.open(im_path).convert("RGB")
        except:
            print("File error: ", im_path)
            image = Image.open(im_path).convert("RGB")

        image = self.data_transform(image)

        return image

    def __getitem__(self, idx):
        task = self.task_list[idx]

        pos_samples = task["pos_samples"]
        neg_samples = task["neg_samples"]

        pos_images = [self.load_image(f) for f in pos_samples]
        neg_images = [self.load_image(f) for f in neg_samples]
        pos_images_stacked = torch.stack(pos_images, dim=0)
        neg_images_stacked = torch.stack(neg_images, dim=0)

        pos_support = pos_images_stacked[:-1]
        neg_support = neg_images_stacked[:-1]
        pos_query = pos_images_stacked[-1]
        neg_query = neg_images_stacked[-1]

        x_support = torch.cat([pos_support, neg_support], dim=0)
        x_query = torch.stack([pos_query, neg_query])
        randp = torch.rand(1)
        label = 0
        if (randp < 0.5):
            x_query = torch.stack([pos_query, neg_query])
        else:
            x_query = torch.stack([neg_query, pos_query])
            label = 1
        x = torch.cat([x_support, x_query], dim=0)
        x_size = x.shape
        x = x.view(-1, x_size[2], x_size[3])

        return x, label