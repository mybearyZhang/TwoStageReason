
import json
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from random import randint
import torchvision
import pybullet as pb
import cv2
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
class CophyBallDataset(Dataset):
    def __init__(self, mode="train", resolution=112, load_cd=True, sampling_mode="rand",
                 load_ab=False, load_state=False, path="/HDD/DATA/mingyu/reasoning/cophydataset/CoPhy_112/ballsCF"):
        super(CophyBallDataset, self).__init__()

        self.load_index(f"Datasets/ballsCF_4")
        self.video_length = 150
        assert sampling_mode in ['rand', 'fix', "full"]

        self.index = None
        self.dataloc = path
        print(self.dataloc)
        assert os.path.isdir(self.dataloc)

        self.mode = mode
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.video_length = 150

    def load_index(self, splits_filename):
        with open(f"{splits_filename}_{self.mode}.txt", "r") as file:
            self.index = [int(k) for k in file.readlines()]

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 1, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix

    def convert_to_2d(self, pose, view, projection, resolution):
        center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
        center_pose = view @ center_pose
        center_pose = projection @ center_pose
        center_pose = center_pose[:3] / center_pose[-1]
        center_pose = (center_pose + 1) / 2 * resolution
        center_pose[1] = resolution - center_pose[1]
        return center_pose[:2].astype(int).flatten()

    def get_rgb(self, filedir, sampling_mode, video_length):
        if sampling_mode == "full":
            rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.permute(0, 3, 1, 2)
            r = list(range(150))
        else:
            t = randint(0, int(0.15 * video_length)) if sampling_mode == "rand" else int(0.15 * video_length)

            r = [t, t + int(0.15 * video_length)]
            capture = cv2.VideoCapture(filedir)
            list_rgb = []
            for i in r:
                capture.set(1, i)
                ret, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                list_rgb.append(frame)
            rgb = np.stack(list_rgb, 0)
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
            rgb = torch.FloatTensor(rgb)
        return rgb, r

    def __getitem__(self, item):
        ex = self.index[item]
        out = {'ex': ex}
        if self.load_ab:
            ab = os.path.join(self.dataloc, str(ex), "ab", 'rgb.mp4')
            rgb_ab, r_ab = self.get_rgb(ab, self.sampling_mode, self.video_length)
            out['rgb_ab'] = rgb_ab

        if self.load_cd:
            cd = os.path.join(self.dataloc, str(ex), "cd", 'rgb.mp4')
            rgb_cd, r_cd = self.get_rgb(cd, self.sampling_mode, self.video_length)
            out['rgb_cd'] = rgb_cd

        if self.load_state:

            states = np.load(os.path.join(self.dataloc, str(ex), 'cd', 'states.npy'))

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(self.convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112))
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]

        return out
    

class CophyBlockTowerDataset(Dataset):
    def __init__(self, mode="train", resolution=112, load_cd=True, sampling_mode="rand",
                 load_ab=False, load_state=False, path="/HDD/DATA/mingyu/reasoning/cophydataset/CoPhy_112/blocktowerCF/4"):
        super(CophyBlockTowerDataset, self).__init__()
        self.load_index(f"Datasets/blocktowerCF_4")
        assert sampling_mode in ['rand', 'fix', "full"]

        self.index = None
        self.dataloc = path
        print(self.dataloc)
        assert os.path.isdir(self.dataloc)

        self.mode = mode
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.video_length = 0

    def load_index(self, splits_filename):
        with open(f"{splits_filename}_{self.mode}.txt", "r") as file:
            self.index = [int(k) for k in file.readlines()]

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix

    def convert_to_2d(self, pose, view, projection, resolution):
        center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
        center_pose = view @ center_pose
        center_pose = projection @ center_pose
        center_pose = center_pose[:3] / center_pose[-1]
        center_pose = (center_pose + 1) / 2 * resolution
        center_pose[1] = resolution - center_pose[1]
        return center_pose[:2].astype(int).flatten()

    def get_rgb(self, filedir, sampling_mode, video_length):
        if sampling_mode == "full":
            rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.permute(0, 3, 1, 2)
            r = list(range(150))
        else:
            t = randint(0, int(0.15 * video_length)) if sampling_mode == "rand" else int(0.15 * video_length)

            r = [t, t + int(0.15 * video_length)]
            capture = cv2.VideoCapture(filedir)
            list_rgb = []
            for i in r:
                capture.set(1, i)
                ret, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                list_rgb.append(frame)
            rgb = np.stack(list_rgb, 0)
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
            rgb = torch.FloatTensor(rgb)
        return rgb, r

    def __getitem__(self, item):
        ex = self.index[item]
        out = {'ex': ex}
        if self.load_ab:
            ab = os.path.join(self.dataloc, str(ex), "ab", 'rgb.mp4')
            rgb_ab, r_ab = self.get_rgb(ab, self.sampling_mode, self.video_length)
            out['rgb_ab'] = rgb_ab

        if self.load_cd:
            cd = os.path.join(self.dataloc, str(ex), "cd", 'rgb.mp4')
            rgb_cd, r_cd = self.get_rgb(cd, self.sampling_mode, self.video_length)
            out['rgb_cd'] = rgb_cd

        if self.load_state:

            states = np.load(os.path.join(self.dataloc, str(ex), 'cd', 'states.npy'))

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(self.convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112))
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]

        return out



class CophyCollisionDataset(Dataset):
    def __init__(self, mode="train", resolution=112, load_cd=True, sampling_mode="rand",
                 load_ab=False, load_state=False, path="/HDD/DATA/mingyu/reasoning/cophydataset/CoPhy_112/collisionCF"):
        super(CophyCollisionDataset, self).__init__()
        assert sampling_mode in ['rand', 'fix', "full"]
        self.load_index(f"Datasets/collisionCF")
        self.video_length = 75
        self.index = None
        self.dataloc = path
        print(self.dataloc)
        assert os.path.isdir(self.dataloc)

        self.mode = mode
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.video_length = 0

    def load_index(self, splits_filename):
        with open(f"{splits_filename}_{self.mode}.txt", "r") as file:
            self.index = [int(k) for k in file.readlines()]

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix

    def convert_to_2d(self, pose, view, projection, resolution):
        center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
        center_pose = view @ center_pose
        center_pose = projection @ center_pose
        center_pose = center_pose[:3] / center_pose[-1]
        center_pose = (center_pose + 1) / 2 * resolution
        center_pose[1] = resolution - center_pose[1]
        return center_pose[:2].astype(int).flatten()

    def get_rgb(self, filedir, sampling_mode, video_length):
        if sampling_mode == "full":
            rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.permute(0, 3, 1, 2)
            r = list(range(150))
        else:
            t = randint(0, int(0.15 * video_length)) if sampling_mode == "rand" else int(0.15 * video_length)

            r = [t, t + int(0.15 * video_length)]
            capture = cv2.VideoCapture(filedir)
            list_rgb = []
            for i in r:
                capture.set(1, i)
                ret, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                list_rgb.append(frame)
            rgb = np.stack(list_rgb, 0)
            rgb = 2 * (rgb / 255) - 1
            rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
            rgb = torch.FloatTensor(rgb)
        return rgb, r

    def __getitem__(self, item):
        ex = self.index[item]
        out = {'ex': ex}
        if self.load_ab:
            ab = os.path.join(self.dataloc, str(ex), "ab", 'rgb.mp4')
            rgb_ab, r_ab = self.get_rgb(ab, self.sampling_mode, self.video_length)
            out['rgb_ab'] = rgb_ab

        if self.load_cd:
            cd = os.path.join(self.dataloc, str(ex), "cd", 'rgb.mp4')
            rgb_cd, r_cd = self.get_rgb(cd, self.sampling_mode, self.video_length)
            out['rgb_cd'] = rgb_cd

        if self.load_state:

            states = np.load(os.path.join(self.dataloc, str(ex), 'cd', 'states.npy'))

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(self.convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112))
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]

        return out





