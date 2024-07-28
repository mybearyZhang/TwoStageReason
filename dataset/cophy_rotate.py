import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os

class RotateCophyDataset(Dataset):
    def __init__(self, data_dir, sector=4, rotate=False, noise_dir="test_44.JPEG"):
        self.data_dir = data_dir
        self.samples = os.listdir(data_dir)
        self.sector = sector
        self.rotate = rotate
        self.noise_dir = noise_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_path = os.path.join(self.data_dir, sample_name)
        image = cv2.imread(self.noise_dir)
        image_height, image_width, _ = image.shape
        
        # 预处理
        cap = cv2.VideoCapture(sample_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.noise_dir:
                frame[0:image_height, 0:image_width] = image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        
        # 将所有帧堆叠成张量作为特征
        features = torch.stack(frames)
        
        if not self.rotate:
            # 提取旋转角度
            _, _, angle_str = sample_name.split('_')
            angle_str, _ = angle_str.split('.')
            if self.sector == 4:
                label = int(int(angle_str) / 90 - 1)
            elif self.sector == 8:
                label = int(int(angle_str) / 45 - 1)
            elif self.sector == 18:
                label = int(int(angle_str) / 20 - 1)
            else:
                raise ValueError('sector must be 4 or 8 or 18')
        
        features = features.reshape(450, 224, 224)
        #print(features.size())
        return features, label

    def transform(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)