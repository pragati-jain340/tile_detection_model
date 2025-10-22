import os, glob, json, torch, cv2, numpy as np
from torch.utils.data import Dataset

# ===============================
# TilePairDataset Class
# ===============================
class TilePairDataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir  # ✅ Save for later use
        self.json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"Found {len(self.json_files)} JSON files in {json_dir}")

    def __getitem__(self, idx):
        # Read JSON
        with open(self.json_files[idx], 'r') as f:
            data = json.load(f)

        # Load grayscale images
        imgA = cv2.imread(os.path.join(self.json_dir, data['imageA']),
                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        imgB = cv2.imread(os.path.join(self.json_dir, data['imageB']),
                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Load keypoints and homography
        kpsA = np.array(data['keypointsA'], np.float32)
        kpsB = np.array(data['keypointsB'], np.float32)
        H = np.array(data['H'], np.float32)

        # Create binary heatmaps for detector supervision
        heatA = np.zeros_like(imgA)
        heatB = np.zeros_like(imgB)
        for (x, y) in kpsA:
            if 0 <= int(y) < heatA.shape[0] and 0 <= int(x) < heatA.shape[1]:
                heatA[int(y), int(x)] = 1.0
        for (x, y) in kpsB:
            if 0 <= int(y) < heatB.shape[0] and 0 <= int(x) < heatB.shape[1]:
                heatB[int(y), int(x)] = 1.0

        # Convert everything to tensors
        return (torch.tensor(imgA[None,...]),
                torch.tensor(imgB[None,...]),
                torch.tensor(heatA[None,...]),
                torch.tensor(heatB[None,...]),
                torch.tensor(H))

    def __len__(self):
        return len(self.json_files)


# ===============================
# Test Loader
# ===============================
if __name__ == "__main__":
    dataset = TilePairDataset("tile_detection_model\dataset_train")
    print(f"Total samples: {len(dataset)}")

    # Try reading the first one
    imgA, imgB, heatA, heatB, H = dataset[0]
    print("✅ JSON loaded successfully!")
    print("ImageA shape:", imgA.shape)
    print("ImageB shape:", imgB.shape)
    print("Heatmap shape:", heatA.shape)
    print("Homography matrix:\n", H)
