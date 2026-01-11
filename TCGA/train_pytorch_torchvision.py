import io
import os
import timm
import torch
import pickle
import random
import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import Accuracy
from torch.cuda.amp import autocast, GradScaler
from utils.save_best_model import BestModelCheckPoint
from utils.tcga_loader import TCGA_Dataset
from torchvision.models import resnet18, resnet34, mobilenet_v3_small, mobilenet_v2, efficientnet_v2_m, swin_v2_b, resnet50, vgg16_bn, alexnet, densenet121, mnasnet0_5, mnasnet1_0, mnasnet0_75, mnasnet1_3

import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(modelNo, data_flag, Model):
    path = f"results/{data_flag}"
    BATCH_SIZE = 64

    # Enable benchmark mode for faster training if input sizes don't change
    torch.backends.cudnn.benchmark = True

    is_vit_model = modelNo in ["deit3_small_patch16_224", "vit_base_patch16_224", "beitv2_base_patch16_224"]
    print("Is ViT model:", is_vit_model)

    train_dataset = TCGA_Dataset("Datasets/Train", mode="training", is_nas_phase=False, is_vit=is_vit_model)
    val_dataset = TCGA_Dataset("Datasets/Val", mode="val", is_nas_phase=False, is_vit=is_vit_model)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(train_dataset.__len__())
    print(val_dataset.__len__())

    for seed in [0, 1234, 3074]:

        log = ""
        seed_torch(seed)

        checkpoint = BestModelCheckPoint(modelNo, path=f"{path}")

        # Loss Function
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

        if "mnasnet" in model_name:
            model = Model(weights=None, num_classes=NUM_CLASSES)
        elif model_name not in ["deit3_small_patch16_224", "vit_base_patch16_224", "beitv2_base_patch16_224"]:
            model = Model(weights=None, num_classes=NUM_CLASSES)
        else:
            model = Model

        model = model.to(device)
        print("\nModel No:", modelNo, "Seed:", seed)

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
    
    NUM_CLASSES = 8
    data_flag = 'TCGA_256'

    #check folder
    if not os.path.exists(f"results/{data_flag}"):
        os.makedirs(f"results/{data_flag}")
    
    model_dict = {
        #"resnet18": resnet18,
        #"resnet34": resnet34,
        #"resnet50": resnet50,
        #"vgg16_bn": vgg16_bn,
        #"alexnet": alexnet,
        #"densenet121": densenet121,
        #"mobilenet_v3_small": mobilenet_v3_small,
        #"mobilenet_v2": mobilenet_v2,
        #"efficientnet_v2_m": efficientnet_v2_m,
        "deit3_small_patch16_224": timm.create_model("deit3_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        "vit_base_patch16_224": timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        "beitv2_base_patch16_224": timm.create_model("beitv2_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        #"mnasnet0_5": mnasnet0_5,
        #"mnasnet0_75": mnasnet0_75,
        #"mnasnet1_0": mnasnet1_0,
        #"mnasnet1_3": mnasnet1_3,
    }

    device = torch.device('cuda:0')

    for model_name, model in model_dict.items():
        print("Model Name:", model_name)
        main(modelNo=model_name, data_flag=data_flag, Model=model)
