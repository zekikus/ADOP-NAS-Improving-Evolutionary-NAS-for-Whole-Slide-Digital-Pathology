import os
import numpy as np
from skimage.transform import resize 
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import PngImagePlugin


class TCGA_Dataset(Dataset):
    def __init__(self, path, mode, is_nas_phase=False, is_vit=False):
        
        LARGE_ENOUGH_NUMBER = 100
        PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
        self.mode = mode
        self.is_vit = is_vit
        self.is_nas_phase = is_nas_phase
        self.data_path = path
        
        if self.is_nas_phase:
            self.data_path += "_NAS"
        # TCGA-YG-AA3O-01Z-00-DX1_Melanocytic_SKCM_patch_11.png
        self.class_mapping = {"UCS": 0, "CESC": 1, "OV": 2, "CHOL": 3, "LIHC": 4, "PAAD": 5, "SKCM": 6, "UVM": 7}
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        img = np.load(f"{self.data_path}/{img_file}")['arr_1']
        label = img_file.split("_")[2]

        if self.is_vit:
            img = resize(img, (3, 224, 224), anti_aliasing=True)

        if self.mode == "test":
            return img, self.class_mapping[label], img_file
        else:
            return img, self.class_mapping[label]

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)