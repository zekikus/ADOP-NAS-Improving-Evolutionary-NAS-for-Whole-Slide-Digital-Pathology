import io
import os
import torch
import pickle
import random
import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.spider_loader import SPIDERDataset
from utils.helpers import getBestModelNumbers
from torch.cuda.amp import autocast, GradScaler
from utils.save_best_model import BestModelCheckPoint



import warnings
warnings.filterwarnings("ignore")

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(modelNo, data_flag):
    path = f"results/{data_flag}"
    NUM_CLASSES = 13
    BATCH_SIZE = 64

    # Enable benchmark mode for faster training if input sizes don't change
    torch.backends.cudnn.benchmark = True

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset instances for all splits
    train_dataset = SPIDERDataset(
        data_dir="SPIDER-colorectal/SPIDER-colorectal", 
        context_size=1, 
        split="train", 
        transform=data_transform,
        nas_stage=False,
        percentage=1.0
    )
    
    val_dataset = SPIDERDataset(
        data_dir="SPIDER-colorectal/SPIDER-colorectal", 
        context_size=1, 
        split="val", 
        transform=data_transform,
        records=train_dataset.val_records,
        nas_stage=False,
        percentage=1.0
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    print(train_dataset.__len__())
    print(val_dataset.__len__())

    for seed in [0, 1234, 3074]:
        log = ""
        seed_torch(seed)

        checkpoint = BestModelCheckPoint(modelNo, path=f"{path}")

        # Loss Function
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

        # Load Model
        model = None
        with open(f"{path}/model_{modelNo}.pkl", "rb") as f:
            model = GPU_Unpickler(f).load()
        
        model = model.to(device)
        print("\nModel No:", model.solNo, "Seed:", seed)

        # Reduced learning rate to prevent NaN values
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        for epoch in range(200):
            train_loss = []
            train_acc = []

            # Train Phase
            model.train()
            for inputs, labels in tqdm.tqdm(train_loader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()

                # Clear gradients before forward pass
                optimizer.zero_grad(set_to_none=True)

                # Mixed precision forward pass
                with autocast():
                    output = model(inputs)
                    error = loss_fn(output, labels)
                
                # Check for NaN values
                if torch.isnan(error):
                    print(f"NaN loss detected at epoch {epoch}. Skipping batch.")
                    continue
                
                # Mixed precision backward pass
                scaler.scale(error).backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights with scaled gradients
                scaler.step(optimizer)
                scaler.update()

                # Record metrics
                train_loss.append(error.item())
                train_acc.append(metric_fn(output, labels).item())

            # Validation Phase
            val_loss = []
            val_acc = []
            model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm.tqdm(val_loader):

                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()
                    with autocast():
                        output = model(inputs)
                        error = loss_fn(output, labels)
                    val_acc.append(metric_fn(output, labels).item())
                    val_loss.append(error)

            avg_tr_loss = sum(train_loss) / len(train_loss)
            avg_tr_score = sum(train_acc) / len(train_acc)
            avg_val_loss = sum(val_loss) / len(val_loss)
            avg_val_score = sum(val_acc) / len(val_acc)

            # Update learning rate based on validation performance
            scheduler.step(avg_val_score)

            # Log results
            txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_acc_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_acc: {avg_val_score}"
            log += txt
            print(txt)

            # Save best model
            checkpoint.check(avg_val_score, model, seed)

            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("Learning rate too small, stopping training")
                break

        # Write Log
        with open(f"{path}/log_{modelNo}_seed_{seed}.txt", "w") as f:
            f.write(log)

if __name__ == '__main__':

    device = torch.device('cuda:0')
    data_flag = 'SPIDER'
    best_model_numbers = [i[1] for i in getBestModelNumbers(data_flag)]
    for idx, i in enumerate(best_model_numbers):
        print("Model No:", i)
        main(modelNo=i, data_flag=data_flag)
