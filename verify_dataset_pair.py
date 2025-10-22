import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ================================
# 🧩 Setup Paths
# ================================
dataset_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"
real_json_path = os.path.join(dataset_path, "real_images.json")
pairs_json_path = os.path.join(dataset_path, "train_pairs.json")

# ================================
# 🔍 Load JSON Data
# ================================
with open(real_json_path, "r") as f:
    real_data = json.load(f)["images"]

with open(pairs_json_path, "r") as f:
    pairs_data = json.load(f)["pairs"]

real_map = {img["id"]: img for img in real_data}
print(f"✅ Loaded {len(real_data)} real images and {len(pairs_data)} warped pairs.")

# ================================
# 🎯 Select Pair to Visualize
# ================================
# You can manually choose the index, or set to "random"
pair_index = 2  # ← change this (e.g., 0, 1, 2, etc.)
# pair_index = random.randint(0, len(pairs_data) - 1)  # ← uncomment to pick random pair

pair = pairs_data[pair_index]
src_id = pair["source_id"]
src_info = real_map[src_id]

print(f"\n🔍 Inspecting pair {pair_index + 1}/{len(pairs_data)}")
print("Pair ID:", pair["id"])
print("Source ID:", src_id)

# ================================
# 🖼️ Paths + Data
# ================================
imgA_path = os.path.join(dataset_path, pair["image1"])
imgB_path = os.path.join(dataset_path, pair["image2"])

kpsA = np.array(src_info["keypoints"], np.float32)
kpsB = np.array(pair["keypoints2"], np.float32)
H = np.array(pair["H"], np.float32)

print("\n📁 Image Paths:")
print("Original Image:", imgA_path)
print("Warped Image:", imgB_path)
print("\n📍 Keypoint Counts: A =", len(kpsA), "| B =", len(kpsB))
print("\n📏 Homography Matrix:\n", H)

# ================================
# 🧭 Visualization Helpers
# ================================
def show_keypoints(image_path, keypoints, title):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Missing image: {image_path}")
        return np.zeros((100, 100, 3), np.uint8)
    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ================================
# 📸 Show Images Side-by-Side
# ================================
imgA = show_keypoints(imgA_path, kpsA, "Original")
imgB = show_keypoints(imgB_path, kpsB, "Warped")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imgA)
plt.title("Original Image + Keypoints")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imgB)
plt.title("Warped Image + Keypoints")
plt.axis("off")

plt.suptitle(f"Pair {pair_index + 1}: {pair['id']}", fontsize=14)
plt.tight_layout()
plt.show()
