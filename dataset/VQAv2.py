import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import time
import clip

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

class VQAv2Dataset(Dataset):
    IMAGE_PATH = {
        "train": ("train2014", "v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json"),
        "val": ("val2014", "v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json"),
        "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json")}

    def __init__(self, split, data_path="",
                 image_transforms=trans, question_transforms=None, tokenize=None,
                 answer_selection=most_common_from_dict,
                 verbose=True, testing=False, n_samples=-1):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        self.n_samples = n_samples
        start_time = time.time()
        self.split = split
        self.testing = testing
        self.answer_selection = answer_selection
        assert split in ["train", "val", "testdev", "test"]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))

        if verbose:
            print(f"Start loading VQAv2 Dataset from {path}", flush=True)

        # Questions
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(
            lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")

        # Annotations
        if not testing:
            path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                data = json.load(f)
            df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
            df["image_id"] = df["image_id_x"]
            if not all(df["image_id_y"] == df["image_id_x"]):
                print("There is something wrong with image_id")
            del df["image_id_x"]
            del df["image_id_y"]
        self.df = df
        if n_samples == -1:
            self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading VQAv2 Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        question = self.df.iloc[index]["question"]
        question_id = self.df.iloc[index]["question_id"]
        split = self.split
        if not self.testing:
            main_answer = self.df.iloc[index]["multiple_choice_answer"]  # Already extracted main answer
            answers = self.df.iloc[index][
                "answers"]  # list of dicts: [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, ...]
            selected_answers = self.answer_selection(
                self.df.iloc[index]["answers"])  # Apply answer_selection() function to list of dict
            # print("main", main_answer)
            # print("answers", answers)
            # print("select", selected_answers)

        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, image_path))
        # print("image_path ---->", image_path)
        with open(image_path, "rb") as f:
            img = Image.open(f)
            if img.mode == 'RGB':
                transform = transforms.Compose([
                    trans,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                img = transform(img)
            elif img.mode == 'L':  # 单通道图像
                transform = transforms.Compose([
                        trans,
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
                img = transform(img)
                img = torch.cat([img,img,img],dim=0)
        # Load, transform and tokenize question
        if self.question_transforms:
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)
            question = torch.Tensor(question)  # Convert question to tensor
        tok = clip.tokenize(question, truncate=True) 
        question = tok.squeeze()
        data = (img, question)
        # print(img.size(), question.size())
        # print(type(main_answer))
        if self.testing:
            target = None
        else:
            target = main_answer
        tok = clip.tokenize(target, truncate=True) 
        target = tok.squeeze().float()
        return data, target
    
    
    def __len__(self):
        return self.n_samples