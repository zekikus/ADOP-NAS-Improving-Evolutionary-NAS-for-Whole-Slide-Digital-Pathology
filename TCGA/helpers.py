import glob
import shutil

# Create dictionary of classes and their corresponding file names
dict_ = {}
for file in glob.glob("Datasets/Patches_npz/*.npz"):
    class_name = file.replace("Datasets/Patches_npz/", "").split("_")[1:3]
    file_name = file.replace("Datasets/Patches_npz/", "").split("_")[0]

    dict_.setdefault("_".join(class_name), [])
    if file_name not in dict_["_".join(class_name)]:
        dict_["_".join(class_name)].append(file_name)

# Split into train and validation sets (80% train, 20% val)
train_dict = {}
val_dict = {}
for key, value in dict_.items():
    print(f"{key}: {len(value)}")
    train_dict[key] = value[: int(len(value) * 0.8)]
    val_dict[key] = value[int(len(value) * 0.8) :]


# Create lists of npz files for train and val sets
train_npz = []
val_npz = []

for key, value in train_dict.items():
    print(f"{key} train: {len(value)}")
    for file_name in value:
        train_npz.extend(glob.glob(f"Datasets/Patches_npz/{file_name}_{key}_*.npz"))

for key, value in val_dict.items():
    print(f"{key} val: {len(value)}")
    for file_name in value:
        val_npz.extend(glob.glob(f"Datasets/Patches_npz/{file_name}_{key}_*.npz"))

# Create directories for train and val sets
for file in train_npz:
    shutil.copy(file, "Datasets/Train/")

for file in val_npz:
    shutil.copy(file, "Datasets/Val/")

print(f"Train set size: {len(train_npz)}")
print(f"Val set size: {len(val_npz)}")