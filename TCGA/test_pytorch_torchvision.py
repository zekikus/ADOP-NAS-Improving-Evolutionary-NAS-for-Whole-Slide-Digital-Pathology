import io
import torch
import timm
import tqdm
import pickle
import numpy as np
import torch.utils.data as data
from utils.tcga_loader import TCGA_Dataset
from torchvision.models import resnet18, resnet34, mobilenet_v3_small, mobilenet_v2, efficientnet_v2_m, swin_v2_b, resnet50, vgg16_bn, alexnet, densenet121, mnasnet0_5, mnasnet1_0, mnasnet0_75, mnasnet1_3


class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)
        
def main(data_flag = None, modelNo = None, seed = None, txt="", Model = None):

    NUM_CLASSES = 8

    is_vit_model = modelNo in ["deit3_small_patch16_224", "vit_base_patch16_224", "beitv2_base_patch16_224"]
    print("Is ViT model:", is_vit_model)

    test_dataset = TCGA_Dataset("Datasets/Test", mode="test", is_nas_phase=False, is_vit=is_vit_model)

    # encapsulate data into dataloader form
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = None
    if "mnasnet" in model_name:
        model = Model(weights=None, num_classes=NUM_CLASSES)
    elif model_name not in ["deit3_small_patch16_224", "vit_base_patch16_224", "beitv2_base_patch16_224"]:
        model = Model(pretrained=False, num_classes=NUM_CLASSES)
    else:
        model = Model

    # Load pre-trained weights
    print("Model No:", modelNo, "Seed:", seed)
    print("Load Model...")
    model.load_state_dict(torch.load(f"results/{data_flag}/model_{modelNo}_seed_{seed}.pt", map_location=device))
    model.to(device)

    true_labels = dict()
    predictions = dict()

    model.eval()
    with torch.no_grad():
        for inputs, labels, img_file in tqdm.tqdm(test_loader):
            outputs = model(inputs.to(device))
            outputs = outputs.softmax(dim=-1)
            preds = outputs.argmax(dim=-1).cpu().numpy()

            img_file_name = img_file[0].split("_")[0]
            
            predictions.setdefault(img_file_name, [])
            predictions[img_file_name].append(preds[0])
            true_labels[img_file_name] = labels.item()

    # Majority Voting
    final_predictions = dict()
    for img, pred_list in predictions.items():
        final_predictions[img] = np.bincount(pred_list).argmax()
    
    # Calculate accuracy class wise and overall
    class_map = {
        0: "UCS",
        1: "CESC",
        2: "OV",
        3: "CHOL",
        4: "LIHC",
        5: "PAAD",
        6: "SKCM",
        7: "UVM"
    }
    class_correct = {i: 0 for i in range(8)}
    class_total = {i: 0 for i in range(8)}
    total_correct = 0
    total_samples = 0
    for img, pred in final_predictions.items():
        true_label = true_labels[img]
        class_total[true_label] += 1
        total_samples += 1
        if true_label == pred:
            class_correct[true_label] += 1
            total_correct += 1
    overall_accuracy = total_correct / total_samples
    txt += f"Overall Accuracy: {overall_accuracy:.4f}\n"
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    for class_id, total in class_total.items():
        if total > 0:
            accuracy = class_correct[class_id] / total
            txt += f"Class {class_map[class_id]} Accuracy: {accuracy:.4f} ({class_correct[class_id]}/{total})\n"
            print(f"Class {class_map[class_id]} Accuracy: {accuracy:.4f} ({class_correct[class_id]}/{total})")
        else:
            print(f"Class {class_map[class_id]} has no samples.")

    # Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    y_true = [true_labels[img] for img in final_predictions.keys()]
    y_pred = [pred for pred in final_predictions.values()]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_map.values(), yticklabels=class_map.values(), cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Model {modelNo} Seed {seed}')
    plt.savefig(f'confusion_matrix_model_{modelNo}_seed_{seed}.png')
    plt.close()

    # class-wise precision, recall, F1-score, npv, ppv, specificity
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    specificity = []
    npv = []
    for i in range(len(class_map)):
        tn = sum(cm[j][k] for j in range(len(class_map)) for k in range(len(class_map)) if j != i and k != i)
        fp = sum(cm[j][i] for j in range(len(class_map)) if j != i)
        fn = sum(cm[i][k] for k in range(len(class_map)) if k != i)
        tp = cm[i][i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
        npv_value = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv.append(npv_value)
    
    for i, class_name in class_map.items():
        txt += f"Class {class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}, Specificity: {specificity[i]:.4f}, NPV: {npv[i]:.4f}\n"
        print(f"Class {class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}, Specificity: {specificity[i]:.4f}, NPV: {npv[i]:.4f}")   

    txt += "\n"
    return txt
    

if __name__ == "__main__":

    NUM_CLASSES = 8
    device = torch.device('cuda:0')

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
    
    for model_name, model in model_dict.items():
        result_txt = ""
        for seed in [0, 1234, 3074]:
            result_txt += f"Model {model_name} Seed {seed}\n"
            result_txt += main(data_flag="TCGA_256", modelNo=model_name, seed=seed, Model=model)

        with open(f"results/TCGA_256/test_results_model_{model_name}.txt", "w") as f:
            f.write(result_txt)