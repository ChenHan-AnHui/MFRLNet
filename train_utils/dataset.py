import os
import random
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ICCADDataset(Data.Dataset):
    def __init__(self, args, mode="train", case=None, ratio=1, firstepochs=60):
        base_dir = args.data_path
        self.ratio = ratio
        self.firstepochs = firstepochs

        self.list_dir = os.path.join(base_dir, mode, case) if case else os.path.join(base_dir, mode)
        self.names = None
        self.hs_names = np.array(["HS" + "/" + name for name in os.listdir(os.path.join(self.list_dir, "HS"))])
        self.nhs_names = np.array(["NHS" + "/" + name for name in os.listdir(os.path.join(self.list_dir, "NHS"))])
        self.dataset_update()

        self.mode = mode
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485], [.229]),  # Default mean and std
        ])

    def dataset_update(self, epoch=0):
        if self.ratio == 1 or epoch < self.firstepochs:
            sample_nhs_names = self.nhs_names
        else:
            sample_nhs_names = np.random.permutation(self.nhs_names)[:int(len(self.nhs_names) * self.ratio)]
        self.names = np.concatenate((self.hs_names, sample_nhs_names))

    def __getitem__(self, i):
        name = self.names[i]
        img_path = os.path.join(self.list_dir, name)
        label = name.split("/")[0]
        if label == "HS":
            label = np.array([255], dtype=np.float32)
        elif label == "NHS":
            label = np.array([0], dtype=np.float32)
        else:
            raise ValueError("Unkown label !")

        img = Image.open(img_path)

        if self.mode == "train":
            img, mask = self._sync_transform(img)
        elif self.mode == "test":
            img, mask = self._testval_sync_transform(img)
        else:
            raise ValueError("Unkown self.mode !")

        img, mask = self.transform(img), transforms.ToTensor()(mask)

        return img, mask, label / 255

    def __len__(self):
        return len(self.names)

    def __filename__(self):
        return self.names

    def _sync_transform(self, img):
        img = img.resize((self.base_size, self.base_size), Image.NEAREST)

        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        mask = img

        return img, mask

    def _testval_sync_transform(self, img):
        img = img.resize((self.base_size, self.base_size), Image.NEAREST)
        
        mask = img

        return img, mask
