import os
import glob
import io
import torch
import pickle
import random
from collections import defaultdict
from torchinfo import summary

def write_file(fname, data):
    with open(fname, "w") as f:
        f.write(data)

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cuda'))
        else:
            return super().find_class(module, name)

def create_train_val_splits(input_file, train_file, val_file, val_ratio=0.2, seed=42):
    """
    Create train and validation splits from a given text file while maintaining class balance.

    Args:
        input_file (str): Path to the input text file containing file paths and class labels.
        train_file (str): Path to save the training split.
        val_file (str): Path to save the validation split.
        val_ratio (float): Proportion of data to use for validation (default: 0.2).
        seed (int): Random seed for reproducibility (default: 42).
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Read the input file and group data by class
    class_data = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                class_label = line.split("\\")[1]
                class_data[class_label].append(line)

    # Create train and validation splits
    train_split = []
    val_split = []

    for class_label, files in class_data.items():
        random.shuffle(files)  # Shuffle files for randomness
        split_idx = int(len(files) * (1 - val_ratio))
        train_split.extend([f"{file}" for file in files[:split_idx]])
        val_split.extend([f"{file}" for file in files[split_idx:]])

    # Save the splits to files
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_split))
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_split))

    print(f"Train split saved to: {train_file}")
    print(f"Validation split saved to: {val_file}")

def getBestModelNumbers(data_flag):
    result = []
    for file in glob.glob(f"results/{data_flag}/*.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)
            result.append((data.fitness, data.solNo))

    return sorted(result, key=lambda x: x[0])[-3:]

def readPickleFile(file, data_flag, path="results"):
    with open(f"{path}/{data_flag}/model_{file}.pkl", "rb") as f:
        data = pickle.load(f)
    
    return data

def get_stats_flops(data_flag, modelNo):

    ch = 1
    if data_flag in ['pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist']:
        ch = 3

    print(data_flag)
   
    # Load Model
    model = None
    with open(f"results/{data_flag}/model_{modelNo}.pkl", "rb") as f:
        model = GPU_Unpickler(f).load()
    
    print("Model No:", model.solNo)
    result = summary(model, input_size=(1, ch, 28, 28))
    write_file(f"results/{data_flag}/{modelNo}_summary.txt", str(result))
    del model

"""
# Example usage
input_txt = "data/train_paths.txt"  # Replace with your input file
train_txt = "train_split.txt"
val_txt = "val_split.txt"
create_train_val_splits(input_txt, train_txt, val_txt, val_ratio=0.2)
"""
