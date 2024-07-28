import os
import random
import glob
import numpy as np
import skimage.transform
import skimage.io

import torch
from torch.utils.data import Dataset

import warnings


class ToTensor(object):
    def __call__(self, sample):
        to_tensor(sample)


def to_tensor(sample):
    return torch.tensor(sample, dtype=torch.float32)


class RAVENDataset(Dataset):
    def __init__(self, root, cache_root, split=None, image_size=80, transform=None,
                 use_cache=False, save_cache=False, in_memory=False, subset=None, flip=False, permute=False, n_samples=-1):
        self.root = root
        self.cache_root = cache_root if cache_root is not None else root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.flip = flip
        self.permute = permute
        self.n_samples = n_samples

        def _set_paths():
            if os.path.isdir(os.path.join(self.root, 'data')):
                self.data_dir = os.path.join(self.root, 'data')
            else:
                self.data_dir = self.root

            if self.use_cache:
                self.cached_dir = os.path.join(self.cache_root, 'cache', f'{self.split}_{self.image_size}')

        _set_paths()

        if subset is not None:
            subsets = [subset]
            assert os.path.isdir(os.path.join(self.data_dir, subset))
        else:
            subsets = os.listdir(self.data_dir)

        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.data_dir, i, f"*{split}.npz"))]
            file_names.sort()
            self.file_names += [os.path.join(i, f) for f in file_names]

        self.memory = None
        if in_memory:
            self.load_memory()

    def load_memory(self):
        self.memory = [None] * len(self.file_names)
        from tqdm import tqdm
        for idx in tqdm(range(len(self.file_names)), 'Loading Memory'):
            image, data, _ = self.get_data(idx)
            d = {'target': data["target"],
                 'meta_target': data["meta_target"],
                 'structure': data["structure"],
                 'meta_structure': data["meta_structure"],
                 'meta_matrix': data["meta_matrix"]
                 }
            self.memory[idx] = (image, d)
            del data

    def save_image(self, image, file):
        image = image.numpy()
        os.makedirs(os.path.dirname(file), exist_ok=True)
        image_file = os.path.splitext(file)[0] + '.png'
        skimage.io.imsave(image_file, image.reshape(self.image_size, self.image_size))

    def load_image(self, file):
        image_file = os.path.splitext(file)[0] + '.png'
        gen_image = skimage.io.imread(image_file).reshape(1, self.image_size, self.image_size)
        if self.transform:
            gen_image = self.transform(gen_image)
        gen_image = to_tensor(gen_image)
        return gen_image

    def load_cached_file(self, file):
        try:
            data = np.load(file)
            return data
        except:
            print(f'Error - Could not open existing file {file}')
            return None

    def save_cached_file(self, file, data):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.savez_compressed(file, **data)

    def __len__(self):
        if self.n_samples > 0:
            return self.n_samples
        return len(self.file_names)

    def get_data(self, idx):
        data_file = self.file_names[idx]
        if self.memory is not None and self.memory[idx] is not None:
            resize_image, data = self.memory[idx]
            return resize_image, data, data_file
        else:
            no_cache = True
            # Try to load a cached file for faster fetching
            if self.use_cache:
                cached_path = os.path.join(self.cached_dir, data_file)
                if os.path.isfile(cached_path):
                    data = self.load_cached_file(cached_path)
                    if data is not None:
                        resize_image = data['image'].astype(np.uint8)
                        return resize_image, data, data_file

                if no_cache and not self.save_cache:
                    warnings.warn(f'Error - Expected to load cached data "{data_file}" but cache was not found')

            # Load original file otherwise
            data_path = os.path.join(self.data_dir, data_file)
            try:
                data = np.load(data_path)
            except:
                print(f"Cannot load file {data_file}")
                raise

            image = data["image"].reshape(16, 160, 160)
            if self.image_size != 160:
                resize_image = []
                for idx in range(0, 16):
                    resize_image.append(
                        skimage.transform.resize(image[idx, :, :], (self.image_size, self.image_size),
                                                 order=1, preserve_range=True, anti_aliasing=True))
                resize_image = np.stack(resize_image, axis=0).astype(np.uint8)
            else:
                resize_image = image.astype(np.uint8)

            # Optional: save a cached file for further use
            if self.use_cache:
                if self.save_cache:
                    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                    d = {'image': resize_image,
                         'target': data["target"],
                         'meta_target': data["meta_target"],
                         'structure': data["structure"],
                         'meta_structure': data["meta_structure"],
                         'meta_matrix': data["meta_matrix"]
                         }
                    self.save_cached_file(cached_path, d)
                else:
                    raise ValueError(f'Error cache file {cached_path} not found')

        return resize_image, data, data_file

    def __getitem__(self, idx):
        resize_image, data, data_file = self.get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure = data["structure"]
        structure_encoded = data["meta_matrix"]
        del data

        if self.transform:
            resize_image = self.transform(resize_image)
        resize_image = to_tensor(resize_image)

        if self.flip:
            if random.random() > 0.5:
                resize_image[[0, 1, 2, 3, 4, 5, 6, 7]] = resize_image[[0, 3, 6, 1, 4, 7, 2, 5]]

        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
                target = new_target

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        return resize_image, target#, meta_target, structure_encoded, data_file
