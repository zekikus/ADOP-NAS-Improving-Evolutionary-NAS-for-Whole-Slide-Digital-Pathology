import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class SPIDERDataset(Dataset):
    """
    A PyTorch Dataset that loads stitched images on the fly from flat patch data.

    Assumes the following structure inside data_dir:
    - metadata.json
    - images/    (flat folder containing all patch images)

    The metadata.json file is expected to contain a list of records like:
    {
    "slide_id": "slide_identifier",
    "image_name": "central_patch_filename.png",
    "context_info": {
    "0_0": "patch_filename.png",
    "0_1": "patch_filename.png",
    ...
    },
    "class": "ClassName",
    "split": "train"  // or "test", etc.
    }

    For context sizes > 1, the dataset stitches patches together into one image (of size
    context_size * patch_size by context_size * patch_size). For context size 1, it simply loads
    the central patch.

    The dataset also builds a mapping from class names to numeric labels (and vice versa) which
    can be retrieved via get_class_to_label() and get_label_to_class().

    Args:
    data_dir (str): Directory containing metadata.json and an images folder.
    context_size (int): Context size to use: valid options are 1, 3, or 5.
    split (str): Which split to load ("train", "val", "test", etc.). 
                 If "val" is requested but doesn't exist in metadata, it will be created 
                 from 10% of the training data.
    transform: Optional transform to apply to the stitched PIL image (e.g., a torchvision transform).
    val_split_ratio (float): Ratio of training data to use for validation (default: 0.1).
    random_seed (int): Random seed for reproducible train/val splits (default: 42).
    """

    def __init__(self, data_dir, context_size=5, split="train", transform=None, 
                 val_split_ratio=0.1, random_seed=42, records = None, nas_stage = None, percentage = None):
        self.data_dir = data_dir
        self.context_size = context_size
        self.split = split
        self.transform = transform
        self.val_split_ratio = val_split_ratio
        self.random_seed = random_seed

        # Fixed parameter: each patch is 224 pixels.
        self.patch_size = 224

        # The original flattened grid is assumed to be 5x5.
        self.full_grid_size = 5
        self.center_index = self.full_grid_size // 2  # For a 5x5 grid, this is 2.

        # Load metadata.
        metadata_file = os.path.join(data_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            all_records = json.load(f)

        if split == "train" or split == "val":
            self.train_records, self.val_records = self._create_train_val_splits(all_records)
            if split == "train":
                self.records = self.train_records
            else:  # split == "val"
                self.records = self.val_records
        else:
            # Filter records by the split.
            self.records = [
                rec for rec in all_records if rec.get("split", split) == self.split
            ]

        # Build class mappings using all available data to ensure consistency
        all_classes = sorted(set(rec["class"] for rec in all_records))
        self.class_to_label = {cls: idx for idx, cls in enumerate(all_classes)}
        self.label_to_class = {idx: cls for cls, idx in self.class_to_label.items()}

         # Perform sampling if percentage is provided and in nas_stage
        if percentage is not None and nas_stage == True:

            # Group image paths by class
            class_to_paths = {k:[] for k, v in self.class_to_label.items()}
            for record in self.records:
                class_to_paths[record['class']].append(record)

            # Sample percentage from each class
            random.seed(42)
            sampled_paths = []
            nbr_samples = int(len(self.records) * percentage / len(class_to_paths.keys()))
            for class_id, paths in class_to_paths.items():
                n = max(1, int(nbr_samples))
                sampled_paths.extend(random.sample(paths, n) if len(paths) > 0 else [])

            self.records = sampled_paths

        

        # Define images directory.
        self.images_dir = os.path.join(data_dir, "images")

    def _create_train_val_splits(self, all_records):
        """Create consistent train/val splits from training data."""
        # Get all training records
        original_train_records = [rec for rec in all_records if rec.get("split", "train") == "train"]
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Split by class to maintain class distribution
        class_records = {}
        for record in original_train_records:
            class_name = record["class"]
            if class_name not in class_records:
                class_records[class_name] = []
            class_records[class_name].append(record)
        
        train_records = []
        val_records = []
        
        for class_name, records in class_records.items():
            # Shuffle records for this class
            shuffled_records = records.copy()
            random.shuffle(shuffled_records)
            
            # Calculate validation size (at least 1 sample per class if possible)
            val_size = max(1, int(len(shuffled_records) * self.val_split_ratio))
            val_size = min(val_size, len(shuffled_records))  # Don't exceed available samples
            
            # Split the records
            class_val_records = shuffled_records[:val_size]
            class_train_records = shuffled_records[val_size:]
            
            val_records.extend(class_val_records)
            train_records.extend(class_train_records)
        
        return train_records, val_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Compute bounds for the desired context.
        lower_bound = (self.full_grid_size - self.context_size) // 2
        upper_bound = lower_bound + self.context_size - 1

        # Create a new blank image for the stitched result.
        stitched_width = self.context_size * self.patch_size
        stitched_height = self.context_size * self.patch_size
        stitched_image = Image.new("RGB", (stitched_width, stitched_height))

        # Loop over grid coordinates to load and paste patches.
        for i in range(lower_bound, upper_bound + 1):
            for j in range(lower_bound, upper_bound + 1):
                # For the center patch, use record["image_name"]; otherwise, use context_info.
                if i == self.center_index and j == self.center_index:
                    patch_filename = record["image_name"]
                else:
                    key = f"{i}_{j}"
                    if key not in record["context_info"]:
                        # If a patch is missing, use a blank image.
                        patch = Image.new("RGB", (self.patch_size, self.patch_size))
                        paste_x = (j - lower_bound) * self.patch_size
                        paste_y = (i - lower_bound) * self.patch_size
                        stitched_image.paste(patch, (paste_x, paste_y))
                        continue
                    patch_filename = record["context_info"][key]

                patch_path = os.path.join(self.images_dir, patch_filename)
                try:
                    patch = Image.open(patch_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading patch {patch_path}: {e}")
                    patch = Image.new("RGB", (self.patch_size, self.patch_size))

                paste_x = (j - lower_bound) * self.patch_size
                paste_y = (i - lower_bound) * self.patch_size
                stitched_image.paste(patch, (paste_x, paste_y))

        # For context_size of 1, we simply use the center patch (without stitching)
        if self.context_size == 1:
            stitched_image = Image.open(
                os.path.join(self.images_dir, record["image_name"])
            ).convert("RGB")

        # Apply optional transform.
        if self.transform is not None:
            stitched_image = self.transform(stitched_image)

        # Get the numeric label.
        label = self.class_to_label[record["class"]]

        return stitched_image, label

    def get_class_to_label(self):
        return self.class_to_label

    def get_label_to_class(self):
        return self.label_to_class
    
    def get_split_info(self):
        """Return information about the current split."""
        class_counts = {}
        for record in self.records:
            class_name = record["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "split": self.split,
            "total_samples": len(self.records),
            "class_distribution": class_counts
        }

"""
if __name__ == "__main__":
    # Example usage:
    from torchvision import transforms

    # Define a simple transform that converts the PIL image to a tensor.
    transform = transforms.ToTensor()

    # Create dataset instances for all splits
    train_dataset = SPIDERDataset(
        data_dir="/home/fsmvu/Desktop/SPIDER-colorectal/", 
        context_size=5, 
        split="train", 
        transform=transform,
        nas_stage=False,
        percentage=0.1
    )
    
    val_dataset = SPIDERDataset(
        data_dir="/home/fsmvu/Desktop/SPIDER-colorectal/", 
        context_size=5, 
        split="val", 
        transform=transform,
        records=train_dataset.val_records,
        nas_stage=False,
        percentage=0.1
    )

    test_dataset = SPIDERDataset(
        data_dir="/home/fsmvu/Desktop/SPIDER-colorectal/",
        context_size=5,
        split="test",
        transform=transform
    )

    print("Train dataset info:", train_dataset.get_split_info())
    print("Validation dataset info:", val_dataset.get_split_info())
    print("Test dataset info:", test_dataset.get_split_info())
    
    print("\\nClass to label mapping:", train_dataset.get_class_to_label())

    # Verify no overlap between train and val
    train_images = set(rec["image_name"] for rec in train_dataset.records)
    val_images = set(rec["image_name"] for rec in val_dataset.records)
    overlap = train_images.intersection(val_images)
    print(f"\\nOverlap between train and val: {len(overlap)} samples")
    
    # Access one sample from each split
    train_img, train_label = train_dataset[0]
    val_img, val_label = val_dataset[0]
    test_img, test_label = test_dataset[0]
    
"""