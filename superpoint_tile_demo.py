import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_loader import get_superpoint_model   # ✅ Shared model loader
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Load Fine-Tuned SuperPoint Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = r"C:\pragati\codes\lightglue\tile_detection_model\checkpoints\superpoint_tiles_finetuned.pth"
model = get_superpoint_model(weights_path)  # ✅ Uses singleton loader
model.eval()
print(f"✅ SuperPoint model loaded on {device}")


# ===============================
# 2️⃣ Helper: Non-Maximum Suppression (NMS)
# ===============================
def nms(det_map, conf_thresh=0.3, nms_dist=4):
    """
    Apply simple NMS on a detection map to get stable keypoints.
    """
    keypoints = []
    mask = det_map > conf_thresh
    ys, xs = np.where(mask)

    for (x, y) in zip(xs, ys):
        x0, y0 = max(0, x - nms_dist), max(0, y - nms_dist)
        x1, y1 = min(det_map.shape[1], x + nms_dist + 1), min(det_map.shape[0], y + nms_dist + 1)
        patch = det_map[y0:y1, x0:x1]
        if det_map[y, x] == np.max(patch):
            keypoints.append(cv2.KeyPoint(float(x), float(y), _size=3))

    return keypoints


# ===============================
# 3️⃣ Inference Function
# ===============================
def detect_keypoints(image_path, conf_thresh=0.3, nms_dist=4):
    """
    Run SuperPoint inference on a single image.
    Returns keypoints, descriptors, and detector map.
    """
    # --- Load grayscale image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")

    h, w = img.shape
    img_t = torch.tensor(img / 255.0, dtype=torch.float32)[None, None].to(device)

    # --- Forward pass ---
    with torch.no_grad():
        det_map, desc_map = model(img_t)
        # Upsample detection map to full image resolution
        det_map = F.interpolate(det_map, size=(h, w), mode="bilinear", align_corners=False)
        det_map = det_map[0, 0].cpu().numpy()

    # --- Keypoint detection ---
    keypoints = nms(det_map, conf_thresh=conf_thresh, nms_dist=nms_dist)

    # --- Descriptor extraction ---
    desc_map = desc_map[0].cpu().numpy()
    desc_map = desc_map / np.linalg.norm(desc_map, axis=0, keepdims=True)

    descriptors = []
    for kp in keypoints:
        x, y = int(kp.pt[0] / 8), int(kp.pt[1] / 8)  # 8x downsampling factor
        x = np.clip(x, 0, desc_map.shape[2] - 1)
        y = np.clip(y, 0, desc_map.shape[1] - 1)
        descriptors.append(desc_map[:, y, x])

    descriptors = np.array(descriptors)
    return keypoints, descriptors, img, det_map


# ===============================
# 4️⃣ Run Inference on a Test Image
# ===============================
test_image = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_test\Img46 (1).jpg"

print(f"\n🔍 Running inference on: {test_image}")
kps, descs, img, det_map = detect_keypoints(test_image, conf_thresh=0.5, nms_dist=4)

print(f"✅ Detected {len(kps)} keypoints")
print(f"🧩 Descriptor shape: {descs.shape}")


# ===============================
# 5️⃣ Visualization
# ===============================
# Draw keypoints
vis = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Detector map heatmap
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(vis[..., ::-1])
plt.title("Detected Keypoints (SuperPoint)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(det_map, cmap='hot')
plt.title("Detector Heatmap")
plt.axis('off')
plt.show()

# Save keypoint visualization
save_path = os.path.join(os.path.dirname(test_image), "tile_keypoints_detected.png")
cv2.imwrite(save_path, vis)
print(f"💾 Saved visualization → {save_path}")
