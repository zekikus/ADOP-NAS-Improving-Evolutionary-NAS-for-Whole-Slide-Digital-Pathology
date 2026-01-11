import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.spider_loader import SPIDERDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
from torchvision.models import (
    resnet18, resnet34, resnet50, vgg16_bn, alexnet, densenet121,
    mobilenet_v3_small, mobilenet_v2, efficientnet_v2_m,
    mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
)
import timm
import warnings
warnings.filterwarnings("ignore")


def calculate_metrics(y_true, y_pred, num_classes=13):
    """
    Calculate comprehensive metrics for multi-class classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes in the dataset
    
    Returns:
        Dictionary containing all metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics (average across all classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted-averaged metrics (weighted by support)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate Sensitivity, Specificity, PPV, NPV for each class
    sensitivity_per_class = []
    specificity_per_class = []
    ppv_per_class = []
    npv_per_class = []
    
    for i in range(num_classes):
        # True Positives, False Positives, True Negatives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity_per_class.append(sensitivity)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
        
        # PPV (Precision) = TP / (TP + FP)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        ppv_per_class.append(ppv)
        
        # NPV = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_per_class.append(npv)
    
    # Macro-averaged Sensitivity, Specificity, PPV, NPV
    sensitivity_macro = np.mean(sensitivity_per_class)
    specificity_macro = np.mean(specificity_per_class)
    ppv_macro = np.mean(ppv_per_class)
    npv_macro = np.mean(npv_per_class)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'sensitivity_macro': sensitivity_macro,
        'specificity_macro': specificity_macro,
        'ppv_macro': ppv_macro,
        'npv_macro': npv_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'sensitivity_per_class': sensitivity_per_class,
        'specificity_per_class': specificity_per_class,
        'ppv_per_class': ppv_per_class,
        'npv_per_class': npv_per_class,
        'confusion_matrix': cm
    }
    
    return metrics


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and collect predictions
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
    
    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)


def load_model(model_name, model_class, checkpoint_path, num_classes, device):
    """
    Load trained model from checkpoint
    
    Args:
        model_name: Name of the model
        model_class: Model class or instance
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Initialize model
    if "mnasnet" in model_name:
        model = model_class(weights=None, num_classes=num_classes)
    elif model_name not in ["deit3_small_patch16_224", "vit_base_patch16_224", "beitv2_base_patch16_224"]:
        model = model_class(weights=None, num_classes=num_classes)
    else:
        model = model_class
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    return model


def main():
    # Configuration
    NUM_CLASSES = 13
    data_flag = 'SPIDER'
    BATCH_SIZE = 64
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Model dictionary
    model_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "vgg16_bn": vgg16_bn,
        "alexnet": alexnet,
        "densenet121": densenet121,
        "mobilenet_v3_small": mobilenet_v3_small,
        "mobilenet_v2": mobilenet_v2,
        "efficientnet_v2_m": efficientnet_v2_m,
        "deit3_small_patch16_224": timm.create_model("deit3_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        "vit_base_patch16_224": timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        "beitv2_base_patch16_224": timm.create_model("beitv2_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
        "mnasnet0_5": mnasnet0_5,
        "mnasnet0_75": mnasnet0_75,
        "mnasnet1_0": mnasnet1_0,
        "mnasnet1_3": mnasnet1_3,
    }
    
    # Data preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create test dataset
    test_dataset = SPIDERDataset(
        data_dir="SPIDER-colorectal/SPIDER-colorectal",
        context_size=1,
        split="test",
        transform=data_transform,
        nas_stage=False,
        percentage=1.0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    
    # Results storage
    all_results = []
    
    # Evaluate each model with each seed
    seeds = [0, 1234, 3074]
    
    for model_name, model_class in model_dict.items():
        print(f"\n{'='*80}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'='*80}")
        
        for seed in seeds:
            checkpoint_path = f"results/{data_flag}/model_{model_name}_seed_{seed}.pt"
            
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            print(f"\nSeed: {seed}")
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load model
            model = load_model(model_name, model_class, checkpoint_path, NUM_CLASSES, device)
            
            # Evaluate model
            y_true, y_pred = evaluate_model(model, test_loader, device)
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred, NUM_CLASSES)
            
            # Print results
            print(f"\n--- Results for {model_name} (Seed: {seed}) ---")
            print(f"Accuracy:     {metrics['accuracy']:.4f}")
            print(f"Precision:    {metrics['precision_macro']:.4f} (macro), {metrics['precision_weighted']:.4f} (weighted)")
            print(f"Recall:       {metrics['recall_macro']:.4f} (macro), {metrics['recall_weighted']:.4f} (weighted)")
            print(f"F1-Score:     {metrics['f1_macro']:.4f} (macro), {metrics['f1_weighted']:.4f} (weighted)")
            print(f"Sensitivity:  {metrics['sensitivity_macro']:.4f} (macro)")
            print(f"Specificity:  {metrics['specificity_macro']:.4f} (macro)")
            print(f"PPV:          {metrics['ppv_macro']:.4f} (macro)")
            print(f"NPV:          {metrics['npv_macro']:.4f} (macro)")
            
            # Store results
            result_row = {
                'Model': model_name,
                'Seed': seed,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Macro)': metrics['recall_macro'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'F1-Score (Weighted)': metrics['f1_weighted'],
                'Sensitivity': metrics['sensitivity_macro'],
                'Specificity': metrics['specificity_macro'],
                'PPV': metrics['ppv_macro'],
                'NPV': metrics['npv_macro']
            }
            all_results.append(result_row)
            
            # Save per-class metrics
            per_class_results = []
            for class_idx in range(NUM_CLASSES):
                per_class_row = {
                    'Model': model_name,
                    'Seed': seed,
                    'Class': class_idx,
                    'Precision': metrics['precision_per_class'][class_idx],
                    'Recall': metrics['recall_per_class'][class_idx],
                    'F1-Score': metrics['f1_per_class'][class_idx],
                    'Sensitivity': metrics['sensitivity_per_class'][class_idx],
                    'Specificity': metrics['specificity_per_class'][class_idx],
                    'PPV': metrics['ppv_per_class'][class_idx],
                    'NPV': metrics['npv_per_class'][class_idx]
                }
                per_class_results.append(per_class_row)
            
            # Save per-class results to CSV
            per_class_df = pd.DataFrame(per_class_results)
            per_class_csv_path = f"results/{data_flag}/per_class_metrics_{model_name}_seed_{seed}.csv"
            per_class_df.to_csv(per_class_csv_path, index=False)
            print(f"Per-class metrics saved to: {per_class_csv_path}")
            
            # Save confusion matrix
            cm_path = f"results/{data_flag}/confusion_matrix_{model_name}_seed_{seed}.csv"
            cm_df = pd.DataFrame(metrics['confusion_matrix'])
            cm_df.to_csv(cm_path, index=True)
            print(f"Confusion matrix saved to: {cm_path}")
    
    # Save overall results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = f"results/{data_flag}/test_results_all_models.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"All results saved to: {results_csv_path}")
        print(f"{'='*80}")
        
        # Calculate and display average metrics per model
        print("\n--- Average Metrics Across Seeds ---")
        avg_results = results_df.groupby('Model').mean(numeric_only=True)
        print(avg_results.to_string())
        
        # Save average results
        avg_csv_path = f"results/{data_flag}/test_results_average.csv"
        avg_results.to_csv(avg_csv_path)
        print(f"\nAverage results saved to: {avg_csv_path}")


if __name__ == '__main__':
    main()