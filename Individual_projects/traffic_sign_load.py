import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data
data_path = "/Users/Monod/.cache/kagglehub/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1"

# config
NUM_CLASSES = 43

# Update these paths to match your Kaggle environment
TRAIN_DIR = os.path.join(data_path, "Train")
TEST_DIR = os.path.join(data_path, "Test")
TEST_CSV = os.path.join(data_path, "Test.csv")  # We'll use this for test labels

CLASS_NAMES = {
    0: "Speed limit (20)",
    1: "Speed limit (30)",
    2: "Speed limit (50)",
    3: "Speed limit (60)",
    4: "Speed limit (70)",
    5: "Speed limit (80)",
    6: "End speed limit(80)",
    7: "Speed limit (100)",
    8: "Speed limit (120)",
    9: "No passing",
    10: "No passing >3.5t",
    11: "Right-of-way",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No >3.5t vehicles",
    17: "No entry",
    18: "General caution",
    19: "Danger curve left",
    20: "Danger curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Ice/snow",
    31: "Wild animals",
    32: "End speed+passing",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight/right",
    37: "Go straight/left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout",
    41: "End no passing",
    42: "End no passing >3.5t",
}

print("=" * 55)
print("TASK 8: TRAFFIC SIGN RECOGNITION (with tqdm)")
print("=" * 55)
print(f"Classes    : {NUM_CLASSES}")

print("Loading data from folder structure...")

train_paths = []
train_labels = []

# Loop through each class folder (0 to 42)
for class_id in tqdm(range(NUM_CLASSES), desc="Scanning class folders"):
    class_folder = os.path.join(TRAIN_DIR, str(class_id))
    if os.path.exists(class_folder):
        # Get all PNG files in the folder
        for img_file in os.listdir(class_folder):
            if img_file.endswith(".png"):
                img_path = os.path.join(class_folder, img_file)
                train_paths.append(img_path)
                train_labels.append(class_id)

print(f"\nTotal training images found: {len(train_paths):,}")

if len(train_paths) == 0:
    print("ERROR: No training images found! Please check the path:")
    print(f"TRAIN_DIR = {TRAIN_DIR}")
    print("\nContents of Train directory:")
    if os.path.exists(TRAIN_DIR):
        print(os.listdir(TRAIN_DIR)[:10])  # Show first 10 items
    else:
        print(f"Directory does not exist: {TRAIN_DIR}")
    raise Exception("No training images found")

# Check class distribution
class_counts = Counter(train_labels)
most_common = class_counts.most_common(1)[0]
least_common = min(class_counts.items(), key=lambda x: x[1])
print(f"Most common class     : {CLASS_NAMES[most_common[0]]} ({most_common[1]} imgs)")
print(
    f"Least common class    : {CLASS_NAMES[least_common[0]]} ({least_common[1]} imgs)"
)

sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("EDA - GTSRB Traffic Signs", fontsize=15, fontweight="bold")

# Class distribution
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, NUM_CLASSES))
axes[0].barh(
    range(NUM_CLASSES),
    [v for _, v in sorted_counts],
    color=colors,
    edgecolor="white",
    height=0.8,
)
axes[0].set_yticks(range(NUM_CLASSES))
axes[0].set_yticklabels([CLASS_NAMES[c][:22] for c, _ in sorted_counts], fontsize=6)
axes[0].set_title("Class Distribution (Training Set)")
axes[0].set_xlabel("Number of Images")
axes[0].grid(axis="x", alpha=0.3)

# Sample images (5 random classes x 3 images)
sample_classes = [0, 1, 14, 17, 38]  # speed, stop, no entry, keep right
img_grid = []
for cls in sample_classes:
    cls_indices = [i for i, l in enumerate(train_labels) if l == cls][:3]
    row_imgs = []
    for idx in cls_indices:
        img = Image.open(train_paths[idx])
        if img is not None:
            img = img.convert("RGB")  # ensures RGB
            img = img.resize((48, 48))  # resize
            img = np.array(img)  # convert to array (like OpenCV)
            row_imgs.append(img)
    if row_imgs:  # Only add if we found images
        # Pad if less than 3 images
        while len(row_imgs) < 3:
            row_imgs.append(np.zeros((48, 48, 3), dtype=np.uint8))
        img_grid.append(row_imgs)

if img_grid:
    grid_img = np.vstack([np.hstack(row) for row in img_grid])
    axes[1].imshow(grid_img)
    axes[1].set_title(
        "Sample Signs: Speed20 | Speed30 | Stop | No Entry | Keep Right\n(3 examples each)"
    )
    axes[1].axis("off")

plt.tight_layout()
plt.savefig("eda_traffic.png", dpi=150, bbox_inches="tight")
print("\n[Saved] eda_traffic.png")
