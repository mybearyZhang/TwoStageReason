from torch.utils.data import Dataset
import os
import numpy as np
import torch


class Keypoints(Dataset):
    def __init__(self, mode="train", path="/Disk4/xuyue/reasoning/jiting/cophydataset/OUTPUT_PATH", n_samples=-1):
        super(Keypoints, self).__init__()
        self.index = None
        self.dataloc = path
        print(self.dataloc)
        assert os.path.isdir(self.dataloc)
        self.state_size = 0
        self.mode = mode
        self.n_samples = n_samples

    def __len__(self):
        if self.n_samples > 0:
            return self.n_samples
        return len(self.index)

    def __getitem__(self, item):
        ex = self.index[item]
        keypoints_ab = np.load(os.path.join(self.dataloc, str(ex), "keypoints_ab.npy"))
        keypoints_cd = np.load(os.path.join(self.dataloc, str(ex), "keypoints_cd.npy"))

        T, K, S = keypoints_ab.shape
        speed_ab = np.concatenate([torch.zeros(1, K, S), keypoints_ab[1:] - keypoints_ab[:-1]], axis=0)
        speed_cd = np.concatenate([torch.zeros(1, K, S), keypoints_cd[1:] - keypoints_cd[:-1]], axis=0)
        
        keypoints_ab = np.concatenate([keypoints_ab, speed_ab], axis=-1)
        keypoints_cd = np.concatenate([keypoints_cd, speed_cd], axis=-1)
        c = keypoints_cd[:, 0]
        keypoints_ab = torch.from_numpy(keypoints_ab)
        keypoints_cd = torch.from_numpy(keypoints_cd)
        c = torch.from_numpy(c)
        c = torch.unsqueeze(c,dim = 1)
        output = torch.cat([keypoints_ab,c],dim = 1)
        return output, keypoints_cd


class blocktowerCF_Keypoints(Keypoints):
    def __init__(self, **kwargs):
        super(blocktowerCF_Keypoints, self).__init__(**kwargs)

        with open(f"../DATA/cophydataset/blocktowerCF_4_{self.mode}.txt", "r") as file:       
            self.index = [int(k) for k in file.readlines()]
            #/home/xuyue/reasoning/jiting/FilteredCoPhy/Datasets/blocktowerCF_4_train.txt
            #/HDD/DATA/mingyu/reasoning/cophydataset/blocktowerCF_4_train.txt
        self.state_size = 14


class ballsCF_Keypoints(Keypoints):
    def __init__(self, **kwargs):
        super(ballsCF_Keypoints, self).__init__(**kwargs)

        with open(f"../DATA/cophydataset/ballsCF_4_{self.mode}.txt", "r") as file:
            self.index = [int(k) for k in file.readlines()]
        self.state_size = 14


class collisionCF_Keypoints(Keypoints):
    def __init__(self, **kwargs):
        super(collisionCF_Keypoints, self).__init__(**kwargs)

        with open(f"../DATA/cophydataset/collisionCF_{self.mode}.txt", "r") as file:
            self.index = [int(k) for k in file.readlines()]
        self.state_size = 14