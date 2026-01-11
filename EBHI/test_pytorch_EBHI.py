import io
import torch
import time
import tqdm
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from utils.ebhi_dataset import EBHI_Dataset
from torch.utils.data import DataLoader, Subset

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

def main(data_flag = None, modelNo = None, seed = None):
    test_dataset = EBHI_Dataset(split='test', nas_stage=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    indexes = None
    with open('img_path_index_dict.pkl', 'rb') as file:     
        # A new file will be created 
        indexes = pickle.load(file)

    # Load Model
    model = None
    with open(f"results/{data_flag}/model_{modelNo}.pkl", "rb") as f:
        model = GPU_Unpickler(f).load()

    # Load pre-trained weights
    print("Model No:", model.solNo, "Seed:", seed)
    print("Load Model...")
    model.load_state_dict(torch.load(f"results/{data_flag}/model_{modelNo}_seed_{seed}.pt", map_location=device))
    model.to(device)

    model.eval()

    start_time = time.time()
    img_prediction = {}
    for img, idx_list in indexes.items():
        trainset_1 = Subset(test_dataset, idx_list)
        test_dataloader = DataLoader(trainset_1, batch_size=1, shuffle=False)

        predictions = list()
        for imgs, img_paths, image_names, class_ids in tqdm.tqdm(test_dataloader):
            outputs = model(imgs.to(device))
            outputs = outputs.softmax(dim=-1)
            predictions.append(outputs.argmax().tolist())

        img_prediction[img]=Counter(predictions).most_common(1)[0][0]

    elapsed_time = time.time() - start_time
    print("Elapsed Time:", (elapsed_time))
    # For binary
    class_map= {
        "Adenocarcinoma": 1,
        "High-grade IN": 1,
        "Low-grade IN": 0,
        "Polyp": 0,
        "Normal": 0
    }  

    keys = list(img_prediction.keys())
    keys[0].split('_')
    
    true_values= 0
    false_values= 0
    conf_matrix = np.zeros((2, 2))
    for img, pred in img_prediction.items():
        class_name = img.split('/')[1]
        class_id = class_map[class_name]
        conf_matrix[class_id, pred] += 1
        if class_id == pred:
            true_values+=1
        else:
            false_values+=1

    cm = conf_matrix.astype(int)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Not a normal'], 
                yticklabels=['Normal', 'Not a normal'],
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    # Add detailed statistics
    tp, fn, fp, tn = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)

    stats_text = f"""
    Accuracy: {true_values/(true_values+false_values):.3f}    
    Precision: {precision:.3f}
    Recall: {recall:.3f}
    F1-Score: {f1_score:.3f}
    Sensitivity: {sensitivity:.3f}
    Specificity: {specificity:.3f}
    PPV: {ppv:.3f}
    NPV: {npv:.3f}
    
    True Negatives: {tn}
    False Positives: {fp}
    False Negatives: {fn}
    True Positives: {tp}
    Elapsed Time: {elapsed_time:.2f} seconds
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'Model_{modelNo}_seed_{seed}.png', dpi=300, bbox_inches='tight')

    return stats_text

if __name__ == '__main__':
    device = torch.device('cuda:1')
    for model_name in [72, 79, 105]:
        stats_text = ""
        for seed in [0, 1234, 3074]:
            text = main(data_flag='EBHI_42', modelNo=model_name, seed=seed)
            stats_text += text + "\n"

        with open(f"model_{model_name}.txt", "w") as f:
            f.write(stats_text)
