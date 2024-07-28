import os
import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset.raven import RAVENDataset
from dataset.cvrt import CVRDataset
from dataset.bongard import BongardDataset
from dataset.bongardhoi import BongardHoiDataset
from dataset.bongardlogo import BongardLogoDataset
from dataset.clevrer import PhysicsCLEVRDataset
from dataset.filteredcophy import *
from dataset.fckeypoint import *
from dataset.VQAv2 import *
from dataset.VQA import *

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class RavenDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=4, training=True,
                 use_cache=True, save_cache=True, split='*', subset=None, image_size=80, flip=True, permute=False, n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        self.dataset = RAVENDataset(self.data_dir, None,
                               use_cache=use_cache, save_cache=save_cache,
                               split=split, subset=subset,
                               image_size=image_size, transform=None, flip=flip, permute=permute, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CvrDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=4, split='train', image_size=128, n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        self.dataset = CVRDataset(self.data_dir, split=split, image_size=image_size, transform=None, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SvrtDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        
        transform_list = []
        # data augmentation
        transform_list.extend([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        transform = torchvision.transforms.Compose(transform_list)

        self.dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, split),
            transform)
        if n_samples > 0:
            training_images = n_samples
        else:
            training_images = 28000
        if training_images > 0:
            neg_samples = list(range(0, training_images // 2))
            pos_samples = [n + (len(self.dataset) // 2) for n in range(0, training_images // 2)]
            indexes = neg_samples + pos_samples
            self.dataset = torch.utils.data.Subset(self.dataset, indexes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BongardDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers = 4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        self.dataset = BongardDataset(self.data_dir, img_size=256, use_clip=False, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BongardHoiDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split=0.0, num_workers = 4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        self.dataset = BongardHoiDataset(self.data_dir, data_split="unseen_obj_unseen_act", dset='test', balance_dataset=False, use_augs=False, img_size=224, backbone="clip", n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BongardLogoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split= 0.0, num_workers = 4, split = 'train', image_size = 512, n_samples=-1, val_samples=-1):
        self.data_dir = data_dir
        self.dataset = BongardLogoDataset(self.data_dir, dset=split, img_size = 512, use_augs = False, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# class ClevrerDataLoader(BaseDataLoader):
#     def __init__(self, data_dir, batch_size, shuffle = False, validation_split= 0.0, num_workers = 4, split = '*', image_size = 512):
#         self.data_dir = data_dir
#         self.dataset = PhysicsCLEVRDataset(data_dir, phase = "train")
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class FCBallsDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split=0.0, num_workers=4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir 
        self.dataset = ballsCF_Keypoints(path = data_dir, mode = split, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class FCBlockTowerDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split=0.0, num_workers=4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir 
        self.dataset = blocktowerCF_Keypoints(path = data_dir, mode = split, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class FCCollisionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split=0.0, num_workers=4, split='train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir 
        self.dataset = collisionCF_Keypoints(path = data_dir, mode =split, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# class VQAv2DataLoader(BaseDataLoader):
#     def __init__(self, data_path, batch_size, shuffle = False, validation_split=0.0, num_workers=4, split='train', image_size=512, n_samples=-1):
#         self.data_dir = data_path
#         self.dataset = VQAv2Dataset(split=split, data_path=data_path, n_samples=n_samples)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class VQADataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = False, validation_split= 0.0, num_workers = 4, split = 'train', n_samples=-1, val_samples=-1):
        self.data_dir = data_dir 
        self.dataset = VqaDataset(input_dir= data_dir, mode =split, n_samples=n_samples)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)