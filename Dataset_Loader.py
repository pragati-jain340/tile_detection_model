import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


# ===============================
# TilePairDataset Class
# ===============================
class TilePairDataset(Dataset):
    def __init__(self, dataset_dir, use_rgb=True, include_heatmaps=False):
        """
        Args:
            dataset_dir (str): Directory containing 'real_images.json' and 'train_pairs.json'
            use_rgb (bool): Convert grayscale to 3-channel (useful for pretrained CNNs)
            include_heatmaps (bool): Stack keypoint heatmaps with image channels
        """
        self.dataset_dir = dataset_dir
        self.use_rgb = use_rgb
        self.include_heatmaps = include_heatmaps

        # --- Load JSON files ---
        self.real_json = os.path.join(dataset_dir, "real_images.json")
        self.pairs_json = os.path.join(dataset_dir, "train_pairs.json")

        if not os.path.exists(self.real_json):
            raise FileNotFoundError(f"❌ Missing real_images.json in {dataset_dir}")
        if not os.path.exists(self.pairs_json):
            raise FileNotFoundError(f"❌ Missing train_pairs.json in {dataset_dir}")

        with open(self.real_json, "r") as f:
            self.real_data = json.load(f)["images"]
        with open(self.pairs_json, "r") as f:
            self.pairs_data = json.load(f)["pairs"]

        # --- Map source_id → base image info ---
        self.real_map = {img["id"]: img for img in self.real_data}

        print(f"✅ Loaded {len(self.real_data)} original images and {len(self.pairs_data)} warped pairs.")

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        pair = self.pairs_data[idx]
        src_id = pair["source_id"]

        # --- Match source image for keypoints ---
        if src_id not in self.real_map:
            raise KeyError(f"❌ Source ID {src_id} not found in real_images.json")

        src_info = self.real_map[src_id]
        kpsA = np.array(src_info["keypoints"], np.float32)
        kpsB = np.array(pair["keypoints2"], np.float32)
        H = np.array(pair["H"], np.float32)

        # --- Load image paths directly from pair JSON ---
        imgA_path = os.path.join(self.dataset_dir, pair["image1"])
        imgB_path = os.path.join(self.dataset_dir, pair["image2"])

        imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)

        if imgA is None or imgB is None:
            raise FileNotFoundError(f"❌ Image not found: {imgA_path} or {imgB_path}")

        imgA = imgA.astype(np.float32) / 255.0
        imgB = imgB.astype(np.float32) / 255.0

        # --- Create binary heatmaps ---
        heatA = np.zeros_like(imgA)
        heatB = np.zeros_like(imgB)
        for (x, y) in kpsA:
            if 0 <= int(x) < heatA.shape[1] and 0 <= int(y) < heatA.shape[0]:
                heatA[int(y), int(x)] = 1.0
        for (x, y) in kpsB:
            if 0 <= int(x) < heatB.shape[1] and 0 <= int(y) < heatB.shape[0]:
                heatB[int(y), int(x)] = 1.0

        # --- Convert to 3-channel if needed ---
        if self.use_rgb:
            imgA = np.repeat(imgA[None, ...], 3, axis=0)  # (3, H, W)
            imgB = np.repeat(imgB[None, ...], 3, axis=0)
        else:
            imgA = imgA[None, ...]  # (1, H, W)
            imgB = imgB[None, ...]

        # --- Stack heatmaps if required ---
        if self.include_heatmaps:
            imgA = np.concatenate([imgA, heatA[None, ...]], axis=0)
            imgB = np.concatenate([imgB, heatB[None, ...]], axis=0)

        # --- Convert to torch tensors ---
        imgA = torch.tensor(imgA, dtype=torch.float32)
        imgB = torch.tensor(imgB, dtype=torch.float32)
        heatA = torch.tensor(heatA[None, ...], dtype=torch.float32)
        heatB = torch.tensor(heatB[None, ...], dtype=torch.float32)
        H = torch.tensor(H, dtype=torch.float32)

        return imgA, imgB, heatA, heatB, H


# ===============================
# 🔍 Test Loader
# ===============================
if __name__ == "__main__":
    dataset_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"

    dataset = TilePairDataset(
        dataset_dir=dataset_path,
        use_rgb=True,          # Convert grayscale → RGB-like
        include_heatmaps=True  # Stack heatmap as extra channel
    )

    print(f"Total pairs: {len(dataset)}")

    imgA, imgB, heatA, heatB, H = dataset[0]
    print("✅ Dataset sample loaded successfully!")
    print("ImageA shape:", imgA.shape)
    print("ImageB shape:", imgB.shape)
    print("Heatmap A shape:", heatA.shape)
    print("Heatmap B shape:", heatB.shape)
    print("Homography:\n", H)

    # Compare the loaded H with the one in JSON
    pair_json_path = os.path.join(dataset_path, "train_pairs.json")
    with open(pair_json_path, "r") as f:
        pairs_data = json.load(f)["pairs"]
    
    # Take the first pair for comparison
    original_H = np.array(pairs_data[0]["H"], dtype=np.float32)
    
    print("\n🔍 Homography Comparison:")
    print("From JSON file:\n", original_H)
    print("From Dataset Loader:\n", H.numpy())
    
    # Check if both are almost identical
    if np.allclose(original_H, H.numpy(), atol=1e-6):
        print("✅ Homography matrix matches perfectly!")
    else:
        print("⚠️ WARNING: Homography matrix differs!")

    # print(f"\n🔍 Verifying image pair links:")
    # for i, pair in enumerate(dataset.pairs_data):
    #     src_id = pair["source_id"]
    #     if src_id not in dataset.real_map:
    #         print(f"❌ Missing link for pair {pair['id']} (source_id={src_id})")
    #     else:
    #         src_file = dataset.real_map[src_id]["filename"]
    #         warp_file = pair["image2"]
    #         print(f"✅ Pair {i+1}: {src_file} ↔ {warp_file}")



