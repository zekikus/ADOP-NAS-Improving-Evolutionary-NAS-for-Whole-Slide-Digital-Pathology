import os
import glob
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class EBHI_Dataset(Dataset):
    def __init__(self, split=None, nas_stage = False, percentage = None, transform = None):
        if split == 'train':
            with open('data/train_split.txt') as f:
                paths = [line.strip() for line in f]
        elif split == 'val':
            with open('data/val_split.txt') as f:
                paths = [line.strip() for line in f]
        elif split == 'test':
            with open('data/test_paths.txt') as f:
                paths = [line.strip() for line in f]


        """
        # For multi-class
        self.class_map= {
        "Adenocarcinoma": 4,
        "High-grade IN": 3,
        "Low-grade IN": 2,
        "Polyp": 1,
        "Normal": 0
        }
        """

        # For binary
        self.class_map= {
        "Adenocarcinoma": 1,
        "High-grade IN": 1,
        "Low-grade IN": 0,
        "Polyp": 0,
        "Normal": 0
        }

        # Set seed value
        random.seed(42)

        self.data=[]
        for path in paths:
            path = path.replace("\\", "/")
            selected_patches_paths = glob.glob(os.path.join(f"{path}_patches/*.jpg"))
            for selected_patch in selected_patches_paths:
                img_class = selected_patch.split('/')[1]            
                self.data.append([img_class, selected_patch])

        # It will work only for NAS stage
        # Select number of k samples from each class
        if nas_stage:
            selected_data = []
            nbr_sample_per_class = int((len(self.data) * percentage) // 5)
            self.data = np.array(self.data)
            for key in self.class_map:
                filtered_patches = self.data[self.data[:,0] == key].tolist()
                selected_data.extend(filtered_patches[:nbr_sample_per_class])
            
            del self.data
            self.data = selected_data

        self.transform = transform
                    
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx): 
        class_name, img_path = self.data[idx]
        image_name = os.path.basename(img_path).split('_patch')[0]
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path, image_name, class_id