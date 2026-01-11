import io
import os
import torch
import pickle
import random
import timm
import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import Accuracy
from torchvision.models import resnet18, resnet34, mobilenet_v3_small, mobilenet_v2, efficientnet_v2_m, swin_v2_b
from torch.cuda.amp import autocast, GradScaler
from utils.save_best_model import BestModelCheckPoint
from utils.bcnb_dataset import BCNB_Dataset


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


def main(Model, model_name, data_flag):
    path = f"results/{data_flag}"
    NUM_CLASSES = 2
    BATCH_SIZE = 64

    # Enable benchmark mode for faster training if input sizes don't change
    torch.backends.cudnn.benchmark = True

    # load the train and val data for NAS
    train_dataset = BCNB_Dataset('data/train_patches.txt', percentage=0.1, nas_stage=False)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    val_dataset = BCNB_Dataset('data/validation_patches.txt', percentage=0.1, nas_stage=False)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


    print(train_dataset.__len__())

    for seed in [0, 1408, 3074]:#[42, 1337, 999, 1, 314159, 12345, 3333]:
        log = ""
        seed_torch(seed)

        checkpoint = BestModelCheckPoint(model_name, path=f"{path}")

        # Loss Function
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

        if model_name == "swin_v2_b" or model_name == "maxvit_t":
            model = Model(weights=None, num_classes=NUM_CLASSES)
        else:
            # Load Model
            model = Model(pretrained=False, num_classes=NUM_CLASSES)
        
        model = model.to(device)
        print("\nModel Name:", model_name, "Seed:", seed)

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
            for inputs, labels, _ in tqdm.tqdm(train_loader):
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
                for inputs, labels, _ in tqdm.tqdm(val_loader):

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
        with open(f"{path}/log_{model_name}_seed_{seed}.txt", "w") as f:
            f.write(log)

if __name__ == '__main__':

    device = torch.device('cuda:0')
    data_flag = 'BCNB'
    #{"alexnet": alexnet, "resnet50": resnet50, "densenet121": densenet121, "vgg16_bn": vgg16_bn}.items()
    
    model_dict = {
        "resnet18": resnet18(pretrained=False, num_classes=2),
        "resnet34": resnet34(pretrained=False, num_classes=2),
        "mobilenet_v3_small": mobilenet_v3_small(pretrained=False, num_classes=2),
        "mobilenet_v2": mobilenet_v2(pretrained=False, num_classes=2),
        "efficientnet_v2_m": efficientnet_v2_m(),
        "swin_v2_b": swin_v2_b,
    }

    for model_name, model in model_dict.items():
        print("Model Name:", model_name)
        main(Model=model, model_name=model_name, data_flag=data_flag)
